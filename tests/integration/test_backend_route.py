"""Each message on the back-end that finishes it sooner (kera serve --backend auto, docs/DECISIONS.md D-024).

  python tests/integration/test_backend_route.py --kera build/bin/kera --model tiny/qwen3_dense.gguf

The machine here has one back-end, so the model runs twice on the CPU (KERA_AUTO_BACKENDS=cpu,cpu), and each engine is
given speeds shaped like the Minisforum's instead of measuring them (KERA_AUTO_SPEEDS): engine 0 writes faster, engine
1 reads long prompts faster. Both ways kera serve runs them are tested: both loaded (memory for two copies), and one at
a time (KERA_AUTO_KEEP=both or one, whatever memory this machine has free). Through the one port chat apps use:
  1. a short chat message goes to engine 0 and a long prompt to engine 1 (one at a time: loaded in place of engine 0
     first), its follow-up to engine 1 too (it holds the prompt); then short messages go to engine 0 again (one at a
     time: loaded back once that pays, which with a tiny model's loading time of a fraction of a second is at
     once); streamed and not, OpenAI's,
     Anthropic's and llama.cpp's APIs;
  2. every answer (greedy) is the same as a plain kera serve's on the CPU;
  3. a request that isn't a prompt (/v1/models), and one the engine refuses (400), come back as the engine sent them;
  4. stopping kera serve stops its engines, and so does killing it outright.
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# engine 0: reads at 0.01 s a token plus 2e-4 s for each token held, writes at 0.01 s; engine 1: reads at 0.01 s
# flat, writes at 0.1 s. A short message is sooner on 0 (0.5 s against 1.1 s), a prompt of 400 tokens on 1 (4 s
# against 20 s), by more than the time a tiny model takes to load.
SPEEDS = "0.01,2e-4,0.01,0;0.01,0,0.1,0"
LONG = 400


def start(kera, model, port, extra, log, env=None):
    env = dict(os.environ, KERA_HOME=tempfile.mkdtemp(prefix="kera-route-"), **(env or {}))
    lf = open(log, "w")
    return subprocess.Popen([kera, "serve", "-m", model, "--port", str(port), "-c", "1024", "-np", "2", "-t", "2",
                             "--no-warmup", "--no-kv-store", "--no-auto-guess", *extra],
                            stdout=lf, stderr=subprocess.STDOUT, env=env)


def wait_ready(p, port, log, engines=0):
    """Until /health answers, and with engines > 0, until that many engines say they are ready"""
    for _ in range(1200):
        if p.poll() is not None:
            sys.exit(f"FAIL: kera serve stopped (exit code {p.returncode}):\n" + pathlib.Path(log).read_text()[-3000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            if pathlib.Path(log).read_text().count("engine is ready") >= engines:
                return
        except Exception:
            pass
        time.sleep(0.25)
    sys.exit("FAIL: kera serve didn't start:\n" + pathlib.Path(log).read_text()[-3000:])


def post(port, path, body):
    """status, the engine that answered, the body (the streamed pieces joined, for a stream)"""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.headers.get("X-Kera-Engine"), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("X-Kera-Engine"), e.read().decode()


def answer(path, text):
    """The answer's words, from any of the three APIs, streamed or not"""
    if text.lstrip().startswith("{"):
        j = json.loads(text)
        if path == "/completion":
            return j["content"]
        if path == "/v1/messages":
            return "".join(c.get("text", "") for c in j["content"])
        return j["choices"][0]["message"]["content"]
    out = ""
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        c = json.loads(line[6:])
        if path == "/completion":
            out += c.get("content", "")
        elif c.get("choices"):
            out += c["choices"][0].get("delta", {}).get("content") or ""
    return out


def tokens(start, n):
    return [100 + (start + 7 * i) % 900 for i in range(n)]


def children(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def running(pid):
    try:
        return "zombie" not in pathlib.Path(f"/proc/{pid}/status").read_text().lower()
    except OSError:
        return False


def gone(pids, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not any(running(p) for p in pids):
            return True
        time.sleep(0.25)
    return False


def check(failures, ok, what, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what + (f": {detail}" if detail else ""))


def run(a, rp, pp, one_at_a_time, work, failures):
    mode = "one at a time" if one_at_a_time else "both loaded"
    print(f"\n== {mode}")
    env = {"KERA_AUTO_KEEP": "one" if one_at_a_time else "both"}
    log = work / f"router-{'one' if one_at_a_time else 'both'}.log"
    router = start(a.kera, a.model, rp, ["--backend", "auto"], log, env)
    wait_ready(router, rp, log, engines=1 if one_at_a_time else 2)
    said = "one at a time" if one_at_a_time else "runs on both"
    if said not in log.read_text():
        failures.append(f"{mode}: kera serve didn't say '{said}'")

    chat = [{"role": "user", "content": "Tell me about the sea."}]
    long_prompt = tokens(1, LONG)
    long_answer = {}
    cases = [
        # (what, path, body, the engine it should go to)
        ("a short chat message", "/v1/chat/completions", {"messages": chat, "max_tokens": 8, "temperature": 0}, "0"),
        ("the same, streamed", "/v1/chat/completions",
         {"messages": chat, "max_tokens": 8, "temperature": 0, "stream": True}, "0"),
        ("Anthropic's API", "/v1/messages", {"messages": chat, "max_tokens": 8, "temperature": 0}, "0"),
        ("a long prompt", "/completion", {"prompt": long_prompt, "n_predict": 8, "temperature": 0, "stream": True,
                                          "return_tokens": True}, "1"),
        ("its follow-up", "/completion", None, "1"),
    ]
    for what, path, body, want in cases:
        if body is None:  # the follow-up: the long prompt, its answer, and a few more words
            body = {"prompt": long_prompt + long_answer["tokens"] + tokens(9, 10), "n_predict": 8, "temperature": 0}
        s, engine, text = post(rp, path, body)
        s2, _, text2 = post(pp, path, body)
        if s != 200 or s2 != 200:
            failures.append(f"{mode}, {what}: status {s} (plain kera serve: {s2}): {text[:200]}")
            continue
        if path == "/completion" and body.get("stream"):
            long_answer["tokens"] = [t for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"
                                     for t in json.loads(line[6:]).get("tokens", [])]
        got, ref = answer(path, text), answer(path, text2)
        check(failures, engine == want and got == ref and got != "",
              f"{what}: engine {engine} (expected {want}), answer {'same' if got == ref else 'DIFFERENT'} ({got[:30]!r})",
              f"{mode}: answer {got!r} against {ref!r}")
    if one_at_a_time:
        check(failures, "loading CPU in place of CPU for a message" in log.read_text(),
              "the long prompt loaded engine 1 in place of engine 0 first")

    # short messages: on engine 0, at once with both loaded; one at a time, once they would have saved its loading
    n, engine, same = 0, None, True
    while n < 30 and engine != "0":
        body = {"prompt": tokens(5 + n, 20), "n_predict": 8, "temperature": 0}
        _, engine, text = post(rp, "/completion", body)
        _, _, text2 = post(pp, "/completion", body)
        same = same and answer("/completion", text) == answer("/completion", text2)
        n += 1
    check(failures, engine == "0" and same and (n == 1 if not one_at_a_time else n <= 20),
          f"short messages go to engine 0 again after {n} (answers {'same' if same else 'DIFFERENT'})")
    if one_at_a_time:
        check(failures, log.read_text().count("loading CPU in place of CPU") == 2, "engine 0 was loaded back for them")

    # a chat app that goes away mid-answer: the engine stops writing it too
    req = urllib.request.Request(f"http://127.0.0.1:{rp}/completion", data=json.dumps(
        {"prompt": tokens(3, 20), "n_predict": 400, "temperature": 0, "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    resp.readline()
    resp.close()
    for _ in range(40):
        if "cancel" in log.read_text().lower():
            break
        time.sleep(0.25)
    check(failures, "cancel" in log.read_text().lower(), "a request the chat app gave up is cancelled in the engine")

    with urllib.request.urlopen(f"http://127.0.0.1:{rp}/v1/models", timeout=10) as r:
        models = json.loads(r.read())
    check(failures, bool(models.get("data") or models.get("models")), "/v1/models through kera serve")
    s, _, text = post(rp, "/v1/chat/completions", {"messages": "not a list"})
    check(failures, s == 400 and "error" in text, f"a request the engine refuses comes back as it sent it ({s})")

    # stopping kera serve stops its engines; killing it outright does too
    engines = children(router.pid)
    router.send_signal(signal.SIGINT)
    router.wait(timeout=60)
    check(failures, len(engines) >= 1 and gone(engines, 30), f"stopping kera serve stops its engines ({len(engines)})")
    router = start(a.kera, a.model, rp, ["--backend", "auto"], work / "router-kill.log", env)
    wait_ready(router, rp, work / "router-kill.log", engines=1 if one_at_a_time else 2)
    engines = children(router.pid)
    router.kill()
    router.wait()
    check(failures, len(engines) >= 1 and gone(engines, 30), "killing kera serve outright stops its engines")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8620)
    a = ap.parse_args()
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-route-logs-"))
    os.environ["KERA_AUTO_BACKENDS"] = "cpu,cpu"
    os.environ["KERA_AUTO_SPEEDS"] = SPEEDS
    rp, pp = a.port, a.port + 1
    plain = start(a.kera, a.model, pp, ["--backend", "cpu"], work / "plain.log")
    wait_ready(plain, pp, work / "plain.log")
    failures, logs = [], []
    for one in (False, True):
        logs.append(run(a, rp, pp, one, work, failures))
    plain.send_signal(signal.SIGINT)
    plain.wait(timeout=60)
    if failures:
        print(f"\nFAIL ({work}):\n  " + "\n  ".join(failures))
        for log in logs:
            print(f"--- {log.name}\n" + "\n".join(l for l in log.read_text().splitlines() if l.startswith("Kera:"))[-4000:])
        sys.exit(1)
    print("\nPASS: each message went to the engine that finishes it sooner, with the same answers")


if __name__ == "__main__":
    main()
