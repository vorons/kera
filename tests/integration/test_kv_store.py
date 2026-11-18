"""Saved chat states survive a restart (spec 4b item 5, M2): chat with `kera serve`, stop it, start it
again, and send the same chat plus one message. The prompt must come back from the SSD instead of
being read again, and the answer must be the same as without the store. Then the wrong-identity test
(section 6 item 5, Gate 5b (f)): restarted with a different KV precision, a changed model file, or another
rope setting, given on the command line or through llama.cpp's LLAMA_ARG_* variables, the server must not reuse
anything that was saved (rule one, D-033: a chat only continues from memory this run would compute).

  python tests/integration/test_kv_store.py --kera build/bin/kera --model tiny/nemotron_h_moe.gguf
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

WORDS = "the of and to in is that for it as with was on be by at this are from or have an not they which".split()


def text(i, n):
    return " ".join(WORDS[(i * 7 + k) % len(WORDS)] for k in range(n)) + "."


def start(kera, model, port, home, extra=(), env_extra=None):
    env = dict(os.environ, KERA_HOME=home, **(env_extra or {}))
    log = open(os.path.join(home, f"server-{time.time():.0f}.log"), "w")
    p = subprocess.Popen([kera, "serve", "-m", model, "--port", str(port), "-c", "8192", "-np", "1", "--no-warmup", *extra],
                         stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return p, log
        except Exception:
            time.sleep(0.25)
    p.kill()
    sys.exit("server didn't start")


def stop(p):
    p.send_signal(signal.SIGINT)
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()


def chat(port, messages):
    body = {"messages": messages, "max_tokens": 24, "temperature": 0, "cache_prompt": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8417)
    ap.add_argument("--words", type=int, default=150, help="words per message (smaller for models with a short context)")
    args = ap.parse_args()
    home = tempfile.mkdtemp(prefix="kera-kv-")

    w = args.words
    messages = [{"role": "system", "content": text(0, w + w // 3)}]
    p, log = start(args.kera, args.model, args.port, home)
    try:
        for turn in range(3):
            messages.append({"role": "user", "content": text(turn + 1, w)})
            r = chat(args.port, messages)
            messages.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or ""})
        messages.append({"role": "user", "content": text(9, max(8, w // 4))})
        expected = chat(args.port, messages)    # the answer with the chat still in the slot
        messages.pop()
    finally:
        stop(p)                                 # saved states are written before it exits

    saved = [f for root, _, files in os.walk(os.path.join(home, "kvstore")) for f in files if f.endswith(".kvp")]
    print(f"saved chat states on disk: {len(saved)}")
    assert saved, "nothing was saved to the SSD"

    p, log = start(args.kera, args.model, args.port, home)
    try:
        time.sleep(2)                           # the model hash is cached, so the store opens at once
        messages.append({"role": "user", "content": text(9, max(8, w // 4))})
        r = chat(args.port, messages)
    finally:
        stop(p)
        log.close()
    t = r.get("timings", {})
    n_prompt = r["usage"]["prompt_tokens"]
    print(f"after the restart: {t.get('cache_n')} of {n_prompt} prompt tokens reused, {t.get('prompt_n')} read")
    assert t.get("cache_n", 0) > n_prompt * 0.8, "the chat was read again instead of loaded from the SSD"
    a = r["choices"][0]["message"].get("content")
    b = expected["choices"][0]["message"].get("content")
    assert a == b, f"the answer changed after loading from the SSD:\n{a!r}\n{b!r}"
    print("the chat came back from the SSD and the answer is unchanged")

    # the same chat under another identity must be read from scratch, never loaded
    changed = os.path.join(home, "changed.gguf")
    shutil.copyfile(args.model, changed)
    with open(changed, "ab") as f:
        f.write(b"\0" * 32)                    # a different file (and hash); llama.cpp ignores the tail
    for what, model, extra, env in [
            ("another KV precision (32-bit)", args.model, ["-ctk", "f32", "-ctv", "f32"], None),
            ("a changed model file", changed, [], None),
            ("another rope base on the command line", args.model, ["--rope-freq-base", "500"], None),
            # (another value than the line above: the same setting given either way is the same identity)
            ("another rope base from LLAMA_ARG_ROPE_FREQ_BASE", args.model, [], {"LLAMA_ARG_ROPE_FREQ_BASE": "400"})]:
        p, log = start(args.kera, model, args.port, home, extra, env)
        try:
            time.sleep(2)
            r = chat(args.port, messages)
        finally:
            stop(p)
            log.close()
        reused = r.get("timings", {}).get("cache_n", 0)
        print(f"with {what}: {reused} prompt tokens reused")
        assert reused == 0, f"saved state was reused with {what}"
    print("PASS: the chat came back from the SSD with the same answer, and never under another identity")


if __name__ == "__main__":
    main()
