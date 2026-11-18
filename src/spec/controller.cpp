// Guess-ahead controller: a line-for-line port of reference/controller.py (spec Appendix B).
// tests/cpp/test_controller.cpp checks it against that code's own example decisions.

#include "spec/kr_controller.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

extern "C" double kr_tokens_per_check(double a, int k) {
    return a >= 1.0 ? k + 1.0 : (1.0 - std::pow(a, k + 1)) / (1.0 - a);
}

extern "C" int kr_choose_k_ex(double a, double t_draft, double t_check,
                              kr_bytes_fn new_bytes, void * new_bytes_user, double ssd_gbps,
                              int k_max,
                              kr_bytes_fn new_ram_bytes, void * new_ram_bytes_user, double ram_gbps,
                              double * rate) {
    auto seconds = [&](int k) {
        double s = k * t_draft + t_check + (new_bytes ? new_bytes(new_bytes_user, k) : 0.0) / (ssd_gbps * 1e9);
        if (new_ram_bytes != nullptr) {
            s += new_ram_bytes(new_ram_bytes_user, k) / (ram_gbps * 1e9);
        }
        return s;
    };

    int best_k = 0;
    double best_rate = 1.0 / seconds(0);
    for (int k = 1; k <= k_max; ++k) {
        const double r = kr_tokens_per_check(a, k) / seconds(k);
        if (r > best_rate) {
            best_k = k;
            best_rate = r;
        }
    }
    if (rate != nullptr) {
        *rate = best_rate;
    }
    return best_k;
}

extern "C" int kr_choose_k(const kr_ctrl * c, double t_draft, double t_check, kr_bytes_fn new_bytes, double ssd_gbps) {
    return kr_choose_k_ex(kr_acceptance_rate(&c->acceptance), t_draft, t_check, new_bytes, c->new_bytes_user, ssd_gbps,
                          c->k_max, nullptr, nullptr, 0.0, nullptr);
}

extern "C" void kr_acceptance_init(kr_acceptance * t, double start, double weight) {
    t->accepted = start;
    t->tested = 1.0;
    t->w = weight;
}

extern "C" double kr_acceptance_rate(const kr_acceptance * t) {
    return t->tested != 0.0 ? t->accepted / t->tested : 0.0;
}

extern "C" double kr_acceptance_update(kr_acceptance * t, int guessed, int accepted) {
    if (guessed) {
        const int rejected = accepted < guessed ? 1 : 0;
        t->accepted = (1 - t->w) * t->accepted + t->w * accepted;
        t->tested = (1 - t->w) * t->tested + t->w * (accepted + rejected);
    }
    return kr_acceptance_rate(t);
}

struct kr_ktuner {
    int k;
    int k_max;
    int trial_tokens;
    int period;
    int base_period;
    int max_period;
    double skip_zero_gain;
    int zero_every;

    long long since = 0;                          // tokens written since the last test
    int tests = 0;                                // tests finished so far
    std::vector<int> queue;                       // candidates still to try in this test
    std::vector<std::pair<int, double>> results;  // k -> measured tokens per second, in insertion order
    std::vector<std::pair<int, double>> last;     // results of the previous test
    bool has_k0_rate = false;
    double k0_rate = 0.0;                         // last measured speed without guessing

    bool has_trial = false;                       // [k, tokens, seconds] for the candidate being tried
    int trial_k = 0;
    long long trial_tok = 0;
    double trial_sec = 0.0;

    std::vector<int> candidates() const {
        std::set<int> c = { std::max(k - 1, 0), k, std::min(k + 1, k_max) };
        bool has_best = !last.empty();
        double best = 0.0;
        for (const auto & r : last) {
            best = (&r == &last.front()) ? r.second : std::max(best, r.second);
        }
        if (!has_k0_rate || !has_best || best < skip_zero_gain * k0_rate || tests % zero_every == 0) {
            c.insert(0);
        }
        return std::vector<int>(c.begin(), c.end()); // sorted
    }

    void start_trial(int kk) {
        has_trial = true;
        trial_k = kk;
        trial_tok = 0;
        trial_sec = 0.0;
    }

    int next_k() {
        if (!has_trial && since >= period) {
            queue = candidates();
            results.clear();
            const int first = queue.front();
            queue.erase(queue.begin());
            start_trial(first);
        }
        return has_trial ? trial_k : k;
    }

    void record(int tokens, double seconds) {
        if (!has_trial) {
            since += tokens;
            return;
        }
        trial_tok += tokens;
        trial_sec += seconds;
        if (trial_tok >= trial_tokens) {
            results.emplace_back(trial_k, trial_tok / trial_sec);
            if (!queue.empty()) {
                const int nk = queue.front();
                queue.erase(queue.begin());
                start_trial(nk);
            } else {
                finish();
            }
        }
    }

    void finish() {
        // max() keeps the first of equal values, as Python's max over a dict does
        int best = results.front().first;
        double best_rate = results.front().second;
        for (const auto & r : results) {
            if (r.second > best_rate) {
                best = r.first;
                best_rate = r.second;
            }
        }
        for (const auto & r : results) {
            if (r.first == 0) {
                has_k0_rate = true;
                k0_rate = r.second;
            }
        }
        last = results;
        tests += 1;
        k = best;
        has_trial = false;
        since = 0;
        if (best == 0) { // guessing lost: test less often
            period = std::min(period * 2, max_period);
        } else {
            period = base_period;
        }
    }
};

extern "C" kr_ktuner * kr_ktuner_create(int k_start, int k_max, int trial_tokens, int period, int max_period,
                                        double skip_zero_gain, int zero_every) {
    kr_ktuner * t = new kr_ktuner();
    t->k = k_start;
    t->k_max = k_max;
    t->trial_tokens = trial_tokens;
    t->period = t->base_period = period;
    t->max_period = max_period;
    t->skip_zero_gain = skip_zero_gain;
    t->zero_every = zero_every;
    return t;
}

extern "C" kr_ktuner * kr_ktuner_create_default(int k_start) {
    return kr_ktuner_create(k_start, 8, 64, 512, 8192, 1.2, 8);
}

extern "C" void kr_ktuner_destroy(kr_ktuner * t) { delete t; }
extern "C" int  kr_ktuner_next_k(kr_ktuner * t) { return t->next_k(); }
extern "C" void kr_ktuner_record(kr_ktuner * t, int tokens, double seconds) { t->record(tokens, seconds); }
extern "C" int  kr_ktuner_current_k(const kr_ktuner * t) { return t->k; }

extern "C" void kr_ktuner_test_now(kr_ktuner * t) {
    t->since = std::max<long long>(t->since, t->period);
}

extern "C" int kr_ktuner_testing(const kr_ktuner * t) {
    return t->has_trial ? 1 : 0;
}
