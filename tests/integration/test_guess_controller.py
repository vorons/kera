"""Guessing ahead must never change greedy output (spec section 6, Gate 6 (a)): Kera's guesser switch inside
llama.cpp's server (spec 11.1, Appendix B), with the guessers Kera picks by itself, with two guessers at once,
with llama.cpp's fixed guess length, and with the model's experts streamed from the SSD (M6).

  python tests/integration/test_guess_controller.py --kera build/bin/kera --model tiny/deepseek_v3_mla.gguf
  python tests/integration/test_guess_controller.py --kera build/bin/kera --model tiny/glm_moe_dsa_mtp.gguf --mtp

With --mtp the model has its own MTP head: Kera must use it by itself, and the test checks it guessed.

Every run keeps the model's memory of the text in 32 bits, without flash attention (--exact-maths, on by
default): then checking several words at once rounds exactly like writing them one by one, and the words must
be identical. With llama.cpp's defaults (16-bit memory, flash attention) the two round differently, and a
choice between two nearly equal words can flip (docs/NEGATIVE_RESULTS.md N-003).
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

PROMPT = "one two three four five six seven eight nine ten one two three four five six seven"
EXACT_MATHS = ["-ctk", "f32", "-ctv", "f32", "-fa", "off"]


def run(kera, model, port, extra, n_predict, log):
    """The words written (token ids), whether the run stopped at the end of the context, and words per second"""
    env = dict(os.environ, KERA_HOME=tempfile.mkdtemp(prefix="kera-guess-"))
    with open(log, "w") as lf:
        p = subprocess.Popen([kera, "serve", "-m", model, "--port", str(port), "-c", "4096", "-np", "1",
                              "--no-warmup", "--no-kv-store", *extra], stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(600):
                if p.poll() is not None:
                    sys.exit(f"FAIL: the server stopped (exit code {p.returncode}); see {log}:\n"
                             + pathlib.Path(log).read_text()[-3000:])
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except Exception:
                    time.sleep(0.25)
            body = {"prompt": PROMPT, "n_predict": n_predict, "temperature": 0, "return_tokens": True}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
            return r["tokens"], bool(r.get("truncated")), r.get("timings", {}).get("predicted_per_second", 0.0)
        finally:
            p.terminate()
            p.wait(timeout=60)


def same_words(base, base_full, out, out_full):
    """Identical, or one is the other cut short where the context ran out (checking guesses needs room in it)"""
    if out == base:
        return True
    n = min(len(base), len(out))
    short_is_full = out_full if len(out) < len(base) else base_full
    return out[:n] == base[:n] and short_is_full and abs(len(out) - len(base)) <= 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", action="store_true", help="the model has an MTP head; check Kera uses it")
    ap.add_argument("--n-predict", type=int, default=1500)
    ap.add_argument("--port", type=int, default=8430)
    ap.add_argument("--llama-defaults", action="store_true",
                    help="run with llama.cpp's own 16-bit memory and flash attention instead of exact maths")
    args = ap.parse_args()
    common = [] if args.llama_defaults else EXACT_MATHS
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-guess-test-"))
    # a draft model: the same weights under another name (so streaming the target doesn't claim it too)
    draft = work / "draft.gguf"
    shutil.copyfile(args.model, draft)
    stream = ["--stream", "kera", "--stream-budget-gb", "0.002"]  # a cache smaller than one layer's experts

    cases = [
        ("Kera's own choice of guessers", []),
        ("copying and a draft model, switched by Kera", ["--spec-type", "ngram-simple,draft-simple", "-md", str(draft)]),
        ("llama.cpp's fixed guess length", ["--spec-type", "ngram-simple", "--no-guess-controller"]),
        ("streamed experts, Kera's own choice of guessers", stream),
        ("streamed experts and a draft model", stream + ["-md", str(draft)]),
    ]
    base, base_full, s0 = run(args.kera, args.model, args.port, ["--no-auto-guess"] + common, args.n_predict,
                              work / "base.log")
    print(f"no guessing: {len(base)} words, {s0:.0f} words/s")
    failed = []
    for i, (name, extra) in enumerate(cases):
        log = work / f"case{i}.log"
        out, out_full, rate = run(args.kera, args.model, args.port + 1 + i, extra + common, args.n_predict, log)
        text = log.read_text(errors="replace")
        same = same_words(base, base_full, out, out_full)
        if same:
            print(f"ok   {name}: {len(out)} words, identical, {rate:.0f} words/s")
        else:
            d = next((k for k in range(min(len(base), len(out))) if base[k] != out[k]), min(len(base), len(out)))
            print(f"FAIL {name}: the words differ from word {d} on (of {len(base)} and {len(out)}); see {log}")
            failed.append(name)
        if i in (0, 3) and args.mtp:
            if "MTP head" not in text.split("guessing so far in this chat:")[-1] or "MTP head" not in text:
                print(f"FAIL {name}: Kera didn't guess with the model's MTP head; see {log}")
                failed.append(name + " (MTP)")
        if i in (3, 4) and "streaming the model's experts" not in text:
            print(f"FAIL {name}: the experts weren't streamed; see {log}")
            failed.append(name + " (streaming)")
        if i == 4:
            # a draft model with the same weights is always right, so every step checks several words: the
            # union growth (spec Phase 6 item 4) must be logged
            growth = [l for l in text.splitlines() if "checking guesses used more experts per layer" in l]
            if growth:
                print("     union growth: " + growth[-1].split("than writing one word: ")[-1])
            else:
                print(f"FAIL {name}: no union growth in the log; see {log}")
                failed.append(name + " (union growth)")
    if failed:
        print(f"FAIL: {', '.join(failed)}")
        return 1
    print("PASS: the same words with and without guessing, in every setup")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
