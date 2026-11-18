"""Models from Ollama's store run the way Ollama runs them (rule one, D-033; docs/NEGATIVE_RESULTS.md N-018).

  python tests/integration/test_ollama_models.py --kera build/bin/kera --model tiny/qwen3_dense.gguf

Builds a small Ollama store in a temporary folder: a tiny model as a blob, with a manifest whose layers give its
Modelfile's SYSTEM text, MESSAGE pair and PARAMETERs, as `ollama create` writes them. Checked through `kera serve -m
<blob>`, the way Kera's app runs an Ollama model:
- a blob whose file has no chat template (Ollama keeps it in its own Go form) is refused with a plain message, never
  served in llama.cpp's fallback format (ChatML), which the model wasn't trained on;
- with a template, the chat gets the Modelfile's system text where it has none of its own (and keeps its own where it
  has one), then the example messages, as Ollama adds them;
- requests that set no sampling get the Modelfile's settings and stop words; a request's own values win, and so do the
  user's own server settings.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
TEMPLATE = ROOT / "third_party" / "llama.cpp" / "models" / "templates" / "Qwen-Qwen3-0.6B.jinja"
SYSTEM = "You are a lighthouse keeper. Answer in one short sentence."
MESSAGES = [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "The keeper of the light."}]
PARAMS = {"temperature": 0.42, "top_k": 7, "top_p": 0.77, "repeat_penalty": 1.05, "stop": ["LIGHTHOUSE-END"],
          "num_ctx": 512}


def gguf_copy(src, dst, template=None):
    """A copy of a GGUF, with tokenizer.chat_template set to template (or left out)."""
    import gguf
    r = gguf.GGUFReader(src)
    w = gguf.GGUFWriter(dst, arch=r.get_field(gguf.Keys.General.ARCHITECTURE).contents(), endianess=r.endianess)
    align = r.get_field(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        w.data_alignment = align.contents()
    for f in r.fields.values():
        if f.name in (gguf.Keys.General.ARCHITECTURE, gguf.Keys.Tokenizer.CHAT_TEMPLATE) or f.name.startswith("GGUF."):
            continue
        sub = f.types[-1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), f.types[0], sub_type=sub)
    if template is not None:
        w.add_key_value(gguf.Keys.Tokenizer.CHAT_TEMPLATE, template, gguf.GGUFValueType.STRING)
    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()


def add_blob(store, data):
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    path = store / "blobs" / digest.replace(":", "-")
    path.write_bytes(data)
    return digest, path


def make_store(store, model_file, name):
    """An Ollama store holding model_file as name:latest, with SYSTEM, MESSAGE and PARAMETER layers."""
    (store / "blobs").mkdir(parents=True, exist_ok=True)
    model_digest, blob = add_blob(store, pathlib.Path(model_file).read_bytes())
    layers = [{"mediaType": "application/vnd.ollama.image.model", "digest": model_digest}]
    for kind, data in (("system", SYSTEM.encode()), ("messages", json.dumps(MESSAGES).encode()),
                       ("params", json.dumps(PARAMS).encode())):
        digest, _ = add_blob(store, data)
        layers.append({"mediaType": f"application/vnd.ollama.image.{kind}", "digest": digest})
    manifest = store / "manifests" / "registry.ollama.ai" / "library" / name / "latest"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schemaVersion": 2, "layers": layers}))
    return blob


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def slot_params(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=30) as r:
        return json.loads(r.read())[0]["params"]


def serve(kera, blob, port, home, extra=()):
    log = open(home / f"serve-{time.time():.0f}.log", "w")
    p = subprocess.Popen([kera, "serve", "-m", str(blob), "--port", str(port), "-c", "1024", "-np", "1", "--no-warmup",
                          "--no-kv-store", *extra], stdout=log, stderr=subprocess.STDOUT,
                         env=dict(os.environ, KERA_HOME=str(home), LLAMA_SERVER_SLOTS_DEBUG="1"))  # /slots with stop words
    for _ in range(240):
        if p.poll() is not None:
            return p
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return p
        except Exception:
            time.sleep(0.25)
    p.kill()
    raise SystemExit("the server didn't start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--model", required=True, help="a tiny chat model (tiny/qwen3_dense.gguf)")
    ap.add_argument("--port", type=int, default=8441)
    a = ap.parse_args()
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-ollama-"))
    home = work / "home"
    home.mkdir()
    failures = []

    def check(ok, what, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what + (f": {detail}" if detail else ""))

    # a blob with no chat template: refused, not served as ChatML
    bare = work / "bare.gguf"
    gguf_copy(a.model, bare, None)
    blob = make_store(work / "store-bare", bare, "bare")
    r = subprocess.run([a.kera, "serve", "-m", str(blob), "--port", str(a.port), "--no-kv-store"], capture_output=True,
                       text=True, timeout=300, env=dict(os.environ, KERA_HOME=str(home)))
    check(r.returncode == 2 and "chat format" in r.stderr and "bare:latest" in r.stderr,
          "a blob without a chat template is refused with a plain message", (r.stdout + r.stderr)[-800:])

    # a blob with one: the Modelfile's messages and settings, as Ollama gives them
    templated = work / "templated.gguf"
    gguf_copy(a.model, templated, TEMPLATE.read_text())
    blob = make_store(work / "store", templated, "keeper")
    p = serve(a.kera, blob, a.port, home)
    try:
        check(p.poll() is None, "a blob with its own chat template is served")
        prompt = post(a.port, "/apply-template", {"messages": [{"role": "user", "content": "Hello"}]})["prompt"]
        check(SYSTEM in prompt and "The keeper of the light." in prompt and prompt.find(SYSTEM) < prompt.find("Hello"),
              "the chat gets the Modelfile's system text, then its example messages", prompt[:600])
        own = post(a.port, "/apply-template", {"messages": [{"role": "system", "content": "Be brief."},
                                                            {"role": "user", "content": "Hello"}]})["prompt"]
        check(SYSTEM not in own and "Be brief." in own, "a chat with its own system message keeps it", own[:600])
        post(a.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2})
        s = slot_params(a.port)
        got = (round(s["temperature"], 3), s["top_k"], round(s["top_p"], 3), round(s["repeat_penalty"], 3))
        check(got == (0.42, 7, 0.77, 1.05), "a request that sets none gets the Modelfile's settings", str(got))
        check("LIGHTHOUSE-END" in s.get("stop", []), "and its stop words", str(s.get("stop")))
        post(a.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2,
                                              "temperature": 0.9})
        s = slot_params(a.port)
        check(round(s["temperature"], 3) == 0.9 and s["top_k"] == 7, "a request's own value wins",
              str((s["temperature"], s["top_k"])))
    finally:
        p.terminate()
        p.wait(timeout=30)
    p = serve(a.kera, blob, a.port, home, ["--temp", "0.3"])
    try:
        post(a.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2})
        s = slot_params(a.port)
        check(round(s["temperature"], 3) == 0.3 and s["top_k"] == 7, "the user's own --temp wins over the Modelfile's",
              str((s["temperature"], s["top_k"])))
    finally:
        p.terminate()
        p.wait(timeout=30)

    if failures:
        print(f"\nFAIL ({work}):\n  " + "\n  ".join(failures))
        for f in sorted(home.glob("serve-*.log")):
            print(f.read_text()[-2000:])
        sys.exit(1)
    shutil.rmtree(work, ignore_errors=True)
    print("\nPASS: Ollama models run with their Modelfile, and never in a chat format they weren't trained on")


if __name__ == "__main__":
    main()
