"""Kera at its shipped defaults answers as stock llama.cpp does (rule one, D-033; docs/NEGATIVE_RESULTS.md N-018).

  python tests/integration/test_stock_defaults.py --kera build/bin/kera --llama-server build/bin/llama-server \
      --model DIR/SmolVLM-256M-Instruct-Q8_0.gguf

`kera serve` with no options (guessing ahead, saved chats and the rest of what it adds) against llama.cpp's own server
with no options, on the same real quantized chat model (copied alone into a folder, so neither attaches an image
reader), greedy. For a question through the chat API (the model's own chat template) and a list the model goes on
copying through the completion API (so Kera's guesses are taken), the words must be the same and their
log-probabilities equal to rounding. A word may differ only at a near tie
(section 6: the top two within 0.1 percentage points), where checking several guessed words at once can round it the
other way, as another batch size would (N-003); the rest of the reply is then not compared.
"""
import argparse
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

WORDS = ("amber basil cedar dune ember fjord garnet harbor iris juniper kelp lantern meadow nectar orchid pebble quartz "
         "ridge saffron thistle umber velvet willow yarrow zephyr anchor beacon cobalt delta estuary falcon glacier "
         "heron inlet jasper kestrel lagoon mistral").split()
LIST = " ".join(WORDS)
PROMPTS = {
    # through the chat API, with the model's own chat template
    "chat": "In two or three sentences, why do leaves change colour in autumn?",
    # a list the model goes on copying, through the completion API: long enough for llama.cpp's copying guesser (a
    # 24-word match with 48 more after it), so Kera's guesses are taken
    "copied": "Word list:\n" + LIST + "\nWord list:\n" + LIST + "\nWord list:\n" + " ".join(WORDS[:3]),
}


def start(argv, port, log, env):
    p = subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    for _ in range(600):
        if p.poll() is not None:
            sys.exit(f"FAIL: {argv[0]} stopped:\n{pathlib.Path(log).read_text(errors='replace')[-3000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return p
        except Exception:
            time.sleep(0.25)
    p.kill()
    sys.exit(f"FAIL: {argv[0]} didn't start")


def ask(port, name, text):
    """(each word with its log-probability and the top two, the reply, the server's timings)"""
    if name == "chat":
        path, body = "/v1/chat/completions", {"messages": [{"role": "user", "content": text}], "max_tokens": 96,
                                              "temperature": 0, "top_k": 1, "logprobs": True, "top_logprobs": 2}
    else:
        path, body = "/completion", {"prompt": text, "n_predict": 96, "temperature": 0, "top_k": 1, "n_probs": 2}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    if name == "chat":
        return j["choices"][0]["logprobs"]["content"], j["choices"][0]["message"]["content"], j.get("timings", {})
    return j["completion_probabilities"], j["content"], j.get("timings", {})


def compare(stock, kera, tol=0.02):
    """(ok, how far they matched, the largest log-probability difference, why not)"""
    worst = 0.0
    for i, (s, k) in enumerate(zip(stock, kera)):
        if s["token"] != k["token"]:
            top = sorted((t["logprob"] for t in s.get("top_logprobs", [])), reverse=True)
            gap = math.exp(top[0]) - math.exp(top[1]) if len(top) >= 2 else 1.0
            if gap <= 1e-3:
                return True, i, worst, f"a near tie at word {i} ({gap:.5f} apart): not compared further"
            return False, i, worst, f"word {i}: stock {s['token']!r}, Kera {k['token']!r} (top two {gap:.4f} apart)"
        worst = max(worst, abs(s["logprob"] - k["logprob"]))
    if len(stock) != len(kera):
        return False, min(len(stock), len(kera)), worst, f"lengths {len(stock)} and {len(kera)}"
    return worst <= tol, len(stock), worst, "" if worst <= tol else f"log-probabilities differ by {worst:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--llama-server", required=True, help="llama.cpp's own server from the same build")
    ap.add_argument("--model", required=True, help="a small real quantized chat model")
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="log-probability difference allowed (the CPU's; a graphics chip checking guessed words in a "
                         "batch rounds them differently: 0.05 on the GB10)")
    a = ap.parse_args()
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-stock-"))
    model = work / "model" / pathlib.Path(a.model).name
    model.parent.mkdir()
    try:
        os.link(a.model, model)
    except OSError:
        shutil.copy2(a.model, model)
    env = dict(os.environ, KERA_HOME=str(work / "home"))
    stock = start([a.llama_server, "-m", str(model), "--port", str(a.port)], a.port, work / "stock.log", env)
    kera = start([a.kera, "serve", "-m", str(model), "--port", str(a.port + 1)], a.port + 1, work / "kera.log", env)
    failures = []
    try:
        for name, text in PROMPTS.items():
            s, s_text, _ = ask(a.port, name, text)
            k, k_text, timings = ask(a.port + 1, name, text)
            ok, n, worst, why = compare(s, k, a.tol)
            took = f"{timings.get('draft_n_accepted', 0)} of {timings.get('draft_n', 0)} guesses taken"
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {n} words the same, log-probabilities within {worst:.2e}; "
                  f"{took}{'; ' + why if why else ''}")
            if not ok:
                failures.append(f"{name}: {why}\n    stock: {s_text!r}\n    Kera:  {k_text!r}")
            if name == "copied" and timings.get("draft_n_accepted", 0) == 0:
                failures.append("copied: Kera took no guesses, so guessing wasn't checked "
                                f"(kera serve's log: {work / 'kera.log'})")
    finally:
        for p in (stock, kera):
            p.terminate()
            p.wait(timeout=60)
    if failures:
        print(f"\nFAIL ({work}):\n  " + "\n  ".join(failures))
        sys.exit(1)
    shutil.rmtree(work, ignore_errors=True)
    print("\nPASS: at its defaults, Kera's answers are stock llama.cpp's, guessed words included")


if __name__ == "__main__":
    main()
