"""Models that see images (D-027): `kera serve -m model.gguf` finds the image reader next to the model by itself, and
answers about pictures through OpenAI's and Anthropic's APIs.

  python tests/integration/test_vision.py --kera build/bin/kera --models DIR

DIR holds SmolVLM-256M-Instruct-Q8_0.gguf and mmproj-SmolVLM-256M-Instruct-Q8_0.gguf (ggml-org's, 280 MB; downloaded
if missing). The picture is a red square with a blue disc in it; the tiny model must name a colour that is in it.
Also checked: --no-mmproj leaves the image reader out, and a text-only question still works with it in; a reader
turned off or named through a LLAMA_ARG_* variable is respected (hook 8).
"""
import argparse
import base64
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib

REPO = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/"
MODEL = "SmolVLM-256M-Instruct-Q8_0.gguf"
READER = "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"


def png(width, height, pixel):
    """A PNG made with the standard library: pixel(x, y) -> (r, g, b)"""
    rows = b"".join(b"\x00" + bytes(c for x in range(width) for c in pixel(x, y)) for y in range(height))

    def chunk(kind, data):
        return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")
    head = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def picture():
    def px(x, y):
        return (30, 60, 220) if (x - 128) ** 2 + (y - 128) ** 2 < 64 ** 2 else (220, 30, 30)
    return base64.b64encode(png(256, 256, px)).decode()


def start(kera, model, port, extra, log, env_extra=None):
    env = dict(os.environ, KERA_HOME=tempfile.mkdtemp(prefix="kera-vision-"), **(env_extra or {}))
    p = subprocess.Popen([kera, "serve", "-m", model, "--port", str(port), "-c", "4096", "--no-warmup", *extra],
                         stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    for _ in range(600):
        if p.poll() is not None:
            sys.exit(f"FAIL: kera serve stopped:\n" + pathlib.Path(log).read_text()[-3000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return p
        except Exception:
            time.sleep(0.25)
    sys.exit("FAIL: kera serve didn't start")


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--models", required=True, type=pathlib.Path)
    ap.add_argument("--port", type=int, default=8660)
    a = ap.parse_args()
    a.models.mkdir(parents=True, exist_ok=True)
    for f in (MODEL, READER):
        if not (a.models / f).is_file():
            print(f"downloading {f}")
            urllib.request.urlretrieve(REPO + f, a.models / f)
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-vision-logs-"))
    failures = []

    def check(ok, what, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what + (f": {detail}" if detail else ""))

    img = picture()
    colours = ("red", "blue", "purple", "pink", "magenta", "orange")
    p = start(a.kera, str(a.models / MODEL), a.port, [], work / "serve.log")
    try:
        log = (work / "serve.log").read_text()
        check("the model can see images" in log and READER in log, "the image reader is found next to the model")
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/props", timeout=10) as r:
            props = json.loads(r.read())
        check(bool(props.get("modalities", {}).get("vision")), "the server says it takes images")
        s, j = post(a.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "What colours are in this picture? Answer in a few words."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}],
            "max_tokens": 24, "temperature": 0})
        text = j["choices"][0]["message"]["content"] if s == 200 else str(j)
        check(s == 200 and any(c in text.lower() for c in colours), f"OpenAI's API: a colour in the picture ({text!r})", text)
        s, j = post(a.port, "/v1/messages", {"max_tokens": 24, "temperature": 0, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
            {"type": "text", "text": "What colours are in this picture? Answer in a few words."}]}]})
        text = "".join(c.get("text", "") for c in j.get("content", [])) if s == 200 else str(j)
        check(s == 200 and any(c in text.lower() for c in colours), f"Anthropic's API: a colour in the picture ({text!r})", text)
        s, j = post(a.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hello."}],
                                                     "max_tokens": 8, "temperature": 0})
        check(s == 200 and j["choices"][0]["message"]["content"].strip() != "", "a question without a picture")
    finally:
        p.terminate()
        p.wait(timeout=60)

    p = start(a.kera, str(a.models / MODEL), a.port + 1, ["--no-mmproj"], work / "serve-off.log")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port + 1}/props", timeout=10) as r:
            props = json.loads(r.read())
        check(not props.get("modalities", {}).get("vision") and "the model can see images" not in
              (work / "serve-off.log").read_text(), "--no-mmproj leaves the image reader out")
    finally:
        p.terminate()
        p.wait(timeout=60)

    # llama.cpp also takes its settings from LLAMA_ARG_* variables (and config.ini): Kera's reader is added only where
    # none of them names one or turns it off (hook 8)
    mine = work / "my-reader.gguf"
    shutil.copy2(a.models / READER, mine)
    for n, (env_extra, want_vision, what) in enumerate((
            ({"LLAMA_ARG_MMPROJ_AUTO": "0"}, False, "LLAMA_ARG_MMPROJ_AUTO=0 leaves the image reader out"),
            ({"LLAMA_ARG_MMPROJ": str(mine)}, True, "LLAMA_ARG_MMPROJ's reader is used, not Kera's pick"))):
        log = work / f"serve-env{n}.log"
        p = start(a.kera, str(a.models / MODEL), a.port + 2 + n, [], log, env_extra)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{a.port + 2 + n}/props", timeout=10) as r:
                vision = bool(json.loads(r.read()).get("modalities", {}).get("vision"))
            text = log.read_text(errors="replace")
            check(vision == want_vision and "not adding" in text and "the model can see images" not in text and
                  (not want_vision or str(mine) in text), what, text[-1500:])
        finally:
            p.terminate()
            p.wait(timeout=60)

    if failures:
        print(f"\nFAIL ({work}):\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nPASS: the model sees pictures through both APIs, with its image reader found by itself")


if __name__ == "__main__":
    main()
