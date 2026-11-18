"""Guess-ahead (speculative decoding) controller: pick how many tokens to guess, 0 = off.

The rule: choose the draft length k that gives the most tokens per second, where
    tokens per check = (1 - a^(k+1)) / (1 - a)          a = chance each guess is right
    seconds per check = k * t_draft + t_check + new_bytes(k) / ssd_speed
new_bytes(k) is what the check must read from the SSD for k guesses. For a dense model
it barely grows with k (every check reads the whole model anyway). For a MoE model it
grows with every extra guess, because each guess can pull in experts nobody else used.
That growth is what makes guessing backfire on MoE models that mostly fit in RAM.
"""


def tokens_per_check(a, k):
    return k + 1.0 if a >= 1.0 else (1 - a ** (k + 1)) / (1 - a)


def choose_k(a, t_draft, t_check, new_bytes, ssd_gb_s, k_max=8, new_ram_bytes=None, ram_gb_s=None):
    """Return (k, tokens_per_second). k = 0 means: don't guess, run a normal step.

    new_ram_bytes(k): for MoE models, the extra bytes a check reads from RAM because the
    k guesses use experts the next word alone wouldn't (experts already in memory still
    cost a read). Leave it None for dense models, where a check reads the same weights.
    This is only the starting guess; KTuner below measures and has the final say."""
    def seconds(k):
        s = k * t_draft + t_check + new_bytes(k) / (ssd_gb_s * 1e9)
        if new_ram_bytes is not None:
            s += new_ram_bytes(k) / (ram_gb_s * 1e9)
        return s

    best_k, best_rate = 0, 1.0 / seconds(0)
    for k in range(1, k_max + 1):
        rate = tokens_per_check(a, k) / seconds(k)
        if rate > best_rate:
            best_k, best_rate = k, rate
    return best_k, best_rate


class AcceptanceTracker:
    """Running estimate of a, the chance that each guessed token is right.

    A check stops at the first wrong guess, so the guesses after it are never tested.
    Count only tested guesses: a = accepted / (accepted + rejections), where a check has
    one rejection if it stopped early and none if every guess was accepted. Dividing by
    all guesses instead would under-count a (0.8 reads as about 0.59 at k = 4).
    Both counts are exponential moving averages."""

    def __init__(self, start=0.6, weight=0.1):
        self.accepted, self.tested, self.w = start, 1.0, weight

    @property
    def a(self):
        return self.accepted / self.tested if self.tested else 0.0

    def update(self, guessed, accepted):
        if guessed:
            rejected = 1 if accepted < guessed else 0
            self.accepted = (1 - self.w) * self.accepted + self.w * accepted
            self.tested = (1 - self.w) * self.tested + self.w * (accepted + rejected)
        return self.a


class KTuner:
    """Measure, don't only predict (the Cascade method, arXiv 2506.20675).

    choose_k() gives the starting k. After that, real speed decides: every `period`
    tokens, briefly run k - 1, k and k + 1, measure real tokens per second for each,
    and keep the fastest. k = 0 (no guessing) is tried too, except while guessing is
    clearly winning (at least `skip_zero_gain` times the last measured k = 0 speed);
    then it is re-checked only every `zero_every` tests, so probing costs little.
    While guessing keeps losing (k = 0 wins), test half as often each time, up to
    `max_period`; as soon as guessing wins, test at the normal rate.

    Use: k = tuner.next_k(); run one step with k guesses; tuner.record(tokens, seconds).
    """

    def __init__(self, k_start, k_max=8, trial_tokens=64, period=512, max_period=8192,
                 skip_zero_gain=1.2, zero_every=8):
        self.k, self.k_max = k_start, k_max
        self.trial_tokens = trial_tokens
        self.period = self.base_period = period
        self.max_period = max_period
        self.skip_zero_gain, self.zero_every = skip_zero_gain, zero_every
        self.since = 0          # tokens written since the last test
        self.tests = 0          # tests finished so far
        self.queue = []         # candidates still to try in this test
        self.results = {}       # k -> measured tokens per second in this test
        self.last = {}          # results of the previous test
        self.k0_rate = None     # last measured speed without guessing
        self.trial = None       # [k, tokens, seconds] for the candidate being tried

    def candidates(self):
        c = {max(self.k - 1, 0), self.k, min(self.k + 1, self.k_max)}
        best = max(self.last.values()) if self.last else None
        if (self.k0_rate is None or best is None or best < self.skip_zero_gain * self.k0_rate
                or self.tests % self.zero_every == 0):
            c.add(0)
        return sorted(c)

    def next_k(self):
        """The k to use for the next step."""
        if self.trial is None and self.since >= self.period:
            self.queue, self.results = self.candidates(), {}
            self.trial = [self.queue.pop(0), 0, 0.0]
        return self.trial[0] if self.trial else self.k

    def record(self, tokens, seconds):
        """Report how many tokens the last step produced and how long it took."""
        if self.trial is None:
            self.since += tokens
            return
        self.trial[1] += tokens
        self.trial[2] += seconds
        if self.trial[1] >= self.trial_tokens:
            k, t, s = self.trial
            self.results[k] = t / s
            if self.queue:
                self.trial = [self.queue.pop(0), 0, 0.0]
            else:
                self._finish()

    def _finish(self):
        best = max(self.results, key=self.results.get)
        if 0 in self.results:
            self.k0_rate = self.results[0]
        self.last, self.tests = dict(self.results), self.tests + 1
        self.k, self.trial, self.since = best, None, 0
        if best == 0:   # guessing lost: test less often
            self.period = min(self.period * 2, self.max_period)
        else:
            self.period = self.base_period
