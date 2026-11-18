"""Models that make images (D-027): `kera serve -m MODEL` serves an image model with the image engine (OpenAI's
/v1/images/generations), and `kera image` makes a picture in a terminal.

  python tests/integration/test_image.py --kera build/bin/kera --models DIR

DIR holds sd_turbo-f16-q8_0.gguf (Green-Sky's SD-Turbo, 2 GB; downloaded if missing). Checked: kera serve recognises
it as an image model and uses SD-Turbo's recommended sampling (1 step, no guidance); a request through OpenAI's API
gives a PNG of the size asked for, the same seed gives the same picture twice, and a different seed a different one;
kera image writes a PNG of the size asked for. The prompt improver (D-030, D-033): kera image --improve-prompt with a
small thinking chat model (Qwen3-0.6B, 0.6 GB; downloaded if missing), which answers as it does in a chat, its thinking
kept out of the rewrite: both texts shown, and, as in a chat, a new random seed each time, so asking again can give
another rewrite (nothing is kept to hand the same one back). (SmolLM2-360M, used before, never finishes an answer with
its own settings; Kera doesn't cut it short.)
"""
import argparse
import base64
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

URL = "https://huggingface.co/Green-Sky/SD-Turbo-GGUF/resolve/main/sd_turbo-f16-q8_0.gguf"
MODEL = "sd_turbo-f16-q8_0.gguf"
CHAT_URL = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
CHAT = "Qwen3-0.6B-Q8_0.gguf"


def png_size(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", data[16:24])


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--models", required=True, type=pathlib.Path)
    ap.add_argument("--port", type=int, default=8680)
    a = ap.parse_args()
    a.models.mkdir(parents=True, exist_ok=True)
    model = a.models / MODEL
    if not model.is_file():
        print(f"downloading {MODEL}")
        urllib.request.urlretrieve(URL, model)
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-image-"))
    env = dict(os.environ, KERA_HOME=str(work / "home"))
    failures = []

    def check(ok, what, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what + (f": {detail}" if detail else ""))

    log = work / "serve.log"
    p = subprocess.Popen([a.kera, "serve", "-m", str(model), "--port", str(a.port)], stdout=open(log, "w"),
                         stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(600):
            if p.poll() is not None:
                sys.exit("FAIL: kera serve stopped:\n" + log.read_text()[-3000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        text = log.read_text()
        check("an image model" in text, "kera serve recognises an image model")
        check("SD-Turbo" in text and "1 step" in text, "and uses SD-Turbo's recommended sampling")

        def picture(seed):
            t0 = time.time()
            # OpenAI's API has no seed: the engine reads its own settings from the prompt (sd_cpp_extra_args)
            prompt = 'a red apple on a wooden table, photo<sd_cpp_extra_args>{"seed": %d}</sd_cpp_extra_args>' % seed
            s, j = post(a.port, "/v1/images/generations", {"prompt": prompt, "n": 1, "size": "256x256"})
            if s != 200:
                return None, f"status {s}: {str(j)[:300]}", 0
            return base64.b64decode(j["data"][0]["b64_json"]), "", time.time() - t0

        one, err, took = picture(7)
        check(one is not None and png_size(one) == (256, 256), f"OpenAI's API: a 256x256 PNG ({took:.1f} s)", err)
        two, err, _ = picture(7)
        check(two is not None and two == one, "the same seed gives the same picture", err)
        three, err, _ = picture(8)
        check(three is not None and three != one, "another seed gives another picture", err)
        if one:
            (work / "served.png").write_bytes(one)
    finally:
        p.terminate()
        p.wait(timeout=60)

    out = work / "cli.png"
    r = subprocess.run([a.kera, "image", "-m", str(model), "-p", "a lighthouse on a cliff", "-W", "256", "-H", "256",
                        "-s", "3", "-o", str(out)], capture_output=True, text=True, env=env, timeout=900)
    ok = r.returncode == 0 and out.is_file() and png_size(out.read_bytes()) == (256, 256)
    check(ok, "kera image writes a 256x256 PNG", (r.stdout + r.stderr)[-1500:])

    # the prompt improver: opt-in, shown, and asked like a chat each time (a new random seed, nothing remembered)
    chat = a.models / CHAT
    if not chat.is_file():
        print(f"downloading {CHAT}")
        urllib.request.urlretrieve(CHAT_URL, chat)
    rewrites = []
    pictures = []
    for i in range(3):
        out = work / f"improved-{i}.png"
        r = subprocess.run([a.kera, "image", "-m", str(model), "-p", "a red fox in snow", "--improve-prompt", str(chat),
                            "-W", "256", "-H", "256", "-s", "5", "-o", str(out)], capture_output=True, text=True,
                           env=env, timeout=900)
        said = [l for l in r.stderr.splitlines() if l.startswith("Kera: improved: ")]
        rewrites.append(said[0] if said else "")
        pictures.append(out.read_bytes() if out.is_file() else b"")
        check(r.returncode == 0 and said and "Kera: your description: a red fox in snow" in r.stderr and
              png_size(pictures[-1]) == (256, 256), f"kera image --improve-prompt shows both and draws (run {i + 1})",
              (r.stdout + r.stderr)[-1500:])
    # three answers from a sampling model, each with a new random seed: all three the same would mean a fixed seed
    # or a remembered answer
    check(all(rewrites) and len(set(rewrites)) > 1, "asked again, the model answers again (a new random seed each time)",
          "\n".join(rewrites))
    check(all("<think>" not in r and "</think>" not in r for r in rewrites),
          "the model's thinking is kept out of the rewrite")
    check(not (work / "home" / "prompt-improver.json").exists(), "no rewrite is kept to be handed back")

    if failures:
        print(f"\nFAIL ({work}):\n  " + "\n  ".join(failures))
        print(log.read_text()[-3000:])
        sys.exit(1)
    print(f"\nPASS: kera serves and makes pictures with an image model (pictures in {work})")


if __name__ == "__main__":
    main()
