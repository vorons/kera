"""Several chats at once (kera serve --multi-user, docs/DECISIONS.md D-023, llama.cpp hook 7).

  python tests/integration/test_multi_user.py --kera build/bin/kera --model tiny/qwen3_dense.gguf
  python tests/integration/test_multi_user.py --kera build/bin/kera --model tiny/nemotron_h_moe.gguf
  python tests/integration/test_multi_user.py --kera build/bin/kera --model tiny/glm_moe_dsa_mtp.gguf --guess --no-sharing

Two servers with three chats each, one with --multi-user and one without, get the same requests at the same moments:
  1. a chat writing a long answer, and a second chat with the same start (a system prompt) and its own question, sent
     once the first is writing: with --multi-user the second shares the first's start instead of reading it (the
     server counts it as cached), and without it reads it all;
  2. a long prompt, and a short one sent while the long one is being read: with --multi-user the short one is
     answered first;
  3. three chats with the same start sent at the same moment (agents with one system prompt): with --multi-user one
     reads the start and the others wait for it, then share it;
  4. with one chat at a time (-np 1), two requests waiting while a third is answered, the later one with a lower
     "priority": with --multi-user it is answered first, without it they go in turn.
A hybrid model (attention plus a recurrent state) shares a start up to the other chat's nearest checkpoint, a few
tokens short of it. A model that can't share one (sliding windows, or guessing with a draft model or MTP head: --guess
--no-sharing) must read it all on both servers, and still get the order. Guessing is off unless --guess. Every answer (greedy) must be the same on both servers, but for section 6's rule: how many chats share a step changes
how the sums round, in stock llama.cpp too, so where two answers part, the words must have been a near tie (the top two
within 0.1 percentage points) and each server's word one of those two.
"""
import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

# the tiny models hold 512 tokens per chat
SYSTEM, QUESTION, ANSWER = 300, 10, 190
NEAR = 32  # a hybrid model shares up to a checkpoint, made this close to a prompt's end (4 + the batch)
LONG, SHORT = 480, 40


def tokens(start, n):
    return [100 + (start + 7 * i) % 900 for i in range(n)]


GUESS = False


def server(kera, model, port, extra, log, slots=3):
    env = dict(os.environ, KERA_HOME=tempfile.mkdtemp(prefix="kera-multi-"))
    lf = open(log, "w")
    # one thread and small batches, so a prompt takes many steps
    p = subprocess.Popen([kera, "serve", "-m", model, "--port", str(port), "-c", str(512 * slots), "-np", str(slots), "-kvu",
                          "-b", "16", "-ub", "16", "-t", "1", "-tb", "1", "--no-warmup", "--no-kv-store",
                          *([] if GUESS else ["--no-auto-guess"]), *extra], stdout=lf, stderr=subprocess.STDOUT, env=env)
    for _ in range(600):
        if p.poll() is not None:
            sys.exit(f"FAIL: the server stopped (exit code {p.returncode}); see {log}:\n" + pathlib.Path(log).read_text()[-3000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return p
        except Exception:
            time.sleep(0.25)
    sys.exit("FAIL: the server didn't start")


def complete(port, prompt, n_predict, on_chunk=None, priority=None):
    """A greedy completion, streamed; on_chunk sees each piece as it comes. Its tokens, timings and when it ended"""
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "return_tokens": True, "cache_prompt": True,
            "stream": True, "return_progress": True, "n_probs": 2}
    if priority is not None:
        body["priority"] = priority
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    toks, probs, timings = [], [], {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            c = json.loads(line[6:])
            if on_chunk:
                on_chunk(c)
            if "prompt_progress" not in c:  # a progress report carries a placeholder token
                toks += c.get("tokens", [])
                probs += c.get("completion_probabilities", [])
            if c.get("stop"):
                timings = c.get("timings", {})
                break
    return {"tokens": toks, "probs": probs, "timings": timings, "done_at": time.monotonic()}


def same_but_near_ties(a, b):
    """'' if the two answers are the same, or part only where a's top two words were a near tie; else why not"""
    n = min(len(a["tokens"]), len(b["tokens"]))
    i = next((k for k in range(n) if a["tokens"][k] != b["tokens"][k]), None)
    if i is None:
        return "" if len(a["tokens"]) == len(b["tokens"]) else "one answer is longer"
    top = a["probs"][i]["top_logprobs"] if i < len(a["probs"]) else []
    if len(top) < 2:
        return f"they part at word {i} with no probabilities to judge it"
    p1, p2 = math.exp(top[0]["logprob"]), math.exp(top[1]["logprob"])
    if abs(p1 - p2) <= 1e-3 and b["tokens"][i] in (top[0]["id"], top[1]["id"]):
        print(f"  (they part at word {i} of {n}, a near tie: {p1:.5f} against {p2:.5f})")
        return ""
    return f"they part at word {i}, which wasn't a near tie ({p1:.5f} against {p2:.5f})"


def while_running(port, first, ready, second):
    """Sends `first`; once ready(chunk) is true for one of its pieces, sends `second`. Both results"""
    went = threading.Event()
    out = {}

    def run_first():
        out["first"] = complete(port, *first, on_chunk=lambda c: went.set() if ready(c) else None)
        went.set()

    t = threading.Thread(target=run_first)
    t.start()
    went.wait(600)
    out["second"] = complete(port, *second)
    t.join()
    return out["first"], out["second"]


def all_at_once(port, requests):
    """Sends every request at the same moment; their results, in order"""
    out = [None] * len(requests)
    go = threading.Barrier(len(requests))

    def one(i):
        go.wait()
        out[i] = complete(port, *requests[i])

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(requests))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def priority_order(kera, model, port, extra, log):
    """One chat at a time: while a long answer is written, a request with priority 5 and then one with -1 wait for the
    slot; which is answered first. Tried again (up to 3 times) if the long answer ended before both were waiting"""
    p = server(kera, model, port, extra, log, slots=1)
    try:
        for attempt in range(3):
            writing, res = threading.Event(), {}

            def busy():
                writes = lambda c: writing.set() if c.get("tokens") and "prompt_progress" not in c else None  # noqa: E731
                res["busy"] = complete(port, tokens(40000 + attempt, 40), 470, on_chunk=writes)["done_at"]
                writing.set()

            def ask(name, prompt, priority):
                res[name + "_sent"] = time.monotonic()
                res[name] = complete(port, prompt, 8, priority=priority)["done_at"]

            first = threading.Thread(target=busy)
            first.start()
            writing.wait(600)
            five = threading.Thread(target=ask, args=("five", tokens(41000 + attempt, 40), 5))
            five.start()
            time.sleep(0.02)
            minus_one = threading.Thread(target=ask, args=("minus_one", tokens(42000 + attempt, 40), -1))
            minus_one.start()
            for t in (first, five, minus_one):
                t.join()
            if res["busy"] > res["minus_one_sent"] + 0.05:  # both were waiting for the slot
                return "minus_one" if res["minus_one"] < res["five"] else "five"
        return "unclear: the long answer kept ending before both requests were waiting"
    finally:
        p.terminate()
        p.wait(timeout=60)


def run(kera, model, port, extra, log):
    p = server(kera, model, port, extra, log)
    try:
        system = tokens(0, SYSTEM)
        # 1. a second chat with the same start, sent once the first has read its prompt and is writing
        shared = while_running(port, (system + tokens(2000, QUESTION), ANSWER), lambda c: len(c.get("tokens", [])) > 0,
                               (system + tokens(3000, QUESTION), 16))
        # 2. a short prompt, sent while a long one is being read
        def reading(c):
            pp = c.get("prompt_progress")
            return pp is not None and 0 < pp["processed"] < pp["total"] - pp["cache"] - 64
        order = while_running(port, (tokens(5000, LONG), 8), reading, (tokens(9000, SHORT), 8))
        # 3. three chats with the same start at the same moment
        start = tokens(20000, SYSTEM)
        together = all_at_once(port, [(start + tokens(30000 + 1000 * k, QUESTION), 8) for k in range(3)])
        return shared, order, together
    finally:
        p.terminate()
        p.wait(timeout=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8570)
    ap.add_argument("--no-sharing", action="store_true", help="the model's memory can't share a start")
    ap.add_argument("--guess", action="store_true", help="with Kera's guessing ahead (an MTP head is a draft model)")
    args = ap.parse_args()
    global GUESS
    GUESS = args.guess
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-multi-test-"))

    (s_first, s_second), (o_long, o_short), t_off = run(args.kera, args.model, args.port, [], work / "off.log")
    (m_first, m_second), (m_long, m_short), t_on = run(args.kera, args.model, args.port + 1, ["--multi-user"],
                                                       work / "on.log")

    fails = []
    pairs = [("the writing chat", s_first, m_first), ("the chat with the same start", s_second, m_second),
             ("the long prompt", o_long, m_long), ("the short prompt", o_short, m_short)]
    pairs += [(f"chat {k + 1} of three at once", a, b) for k, (a, b) in enumerate(zip(t_off, t_on))]
    for name, a, b in pairs:
        why = same_but_near_ties(a, b) if a["tokens"] else "no answer"
        if why:
            fails.append(f"{name}'s answer differs with --multi-user: {why}")
    cached_off, cached_on = s_second["timings"].get("cache_n", -1), m_second["timings"].get("cache_n", -1)
    print(f"the chat with the same start: {cached_off} tokens cached without --multi-user, {cached_on} with")
    shared_said = "sharing" in (work / "on.log").read_text()
    if cached_off >= SYSTEM:
        fails.append("without --multi-user the second chat already had the start (the chats didn't overlap)")
    if args.no_sharing:
        if cached_on >= SYSTEM or shared_said:
            fails.append("the server shared a start this model's memory can't share")
    else:
        if cached_on < SYSTEM - NEAR:
            fails.append(f"with --multi-user the second chat shared {cached_on} tokens, not the {SYSTEM}-token start")
        if not shared_said:
            fails.append("the server with --multi-user didn't say it shared a start")
    shared_off = sum(r["timings"].get("cache_n", 0) for r in t_off)
    shared_on = sum(r["timings"].get("cache_n", 0) for r in t_on)
    print(f"three chats with the same start at once: {shared_off} tokens shared without --multi-user, {shared_on} with")
    if args.no_sharing:
        if shared_on >= SYSTEM:
            fails.append("three chats at once shared a start this model's memory can't share")
    elif shared_on < 2 * (SYSTEM - NEAR):
        fails.append(f"of three chats with the same start at once, two should have shared it ({shared_on} tokens shared)")
    pr_off = priority_order(args.kera, args.model, args.port + 2, [], work / "priority-off.log")
    pr_on = priority_order(args.kera, args.model, args.port + 3, ["--multi-user"], work / "priority-on.log")
    shown = {"minus_one": "-1", "five": "5"}
    print(f"of two waiting requests, answered first: priority {shown.get(pr_off, pr_off)} without --multi-user, "
          f"priority {shown.get(pr_on, pr_on)} with")
    if pr_on.startswith("unclear") or pr_off.startswith("unclear"):
        print("  (priorities not checked: this machine wrote the long answer before both requests were waiting)")
    else:
        if pr_on != "minus_one":
            fails.append(f"with --multi-user the request with the lower priority wasn't answered first ({pr_on})")
        if pr_off != "five":
            fails.append(f"without --multi-user the waiting requests didn't go in turn ({pr_off})")
    first_off = "short" if o_short["done_at"] < o_long["done_at"] else "long"
    first_on = "short" if m_short["done_at"] < m_long["done_at"] else "long"
    print(f"answered first: the {first_off} prompt without --multi-user, the {first_on} with")
    if first_on != "short":
        fails.append("with --multi-user the short prompt wasn't answered first")
    if fails:
        print("FAIL: " + "; ".join(fails) + f" (logs in {work})")
        sys.exit(1)
    print("PASS: " + ("short prompts first (no sharing for this model's memory)" if args.no_sharing else
                      "shared starts and short prompts first") + ", with the same answers but for near ties")


if __name__ == "__main__":
    main()
