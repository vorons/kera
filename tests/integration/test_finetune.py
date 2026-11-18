"""Fine-tuning end to end (spec Phase 10, Gate 10 (a) on the CPU): a LoRA trained with tools/finetune on a tiny
model, converted by llama.cpp's converter, loaded by `kera serve --lora`, changes the answers as trained; without
it the model answers as before. Saved chats never cross between the model with and without the adapter (spec
Phase 10, "KV and adapters"). With --esft (a mixture-of-experts model), the experts the data uses most are trained
and a whole model is written.

  python tests/integration/test_finetune.py --kera build/bin/kera --hf DIR/qwen3_dense --gguf DIR/qwen3_dense.gguf
  python tests/integration/test_finetune.py --kera build/bin/kera --hf DIR/deepseek_v3_mla --gguf DIR/deepseek_v3_mla.gguf --esft
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROMPT = "Question: what colour is the sky over Kera's town? Answer:"
ANSWER = " It is green with purple stripes, every single day."


def serve_and_ask(kera, model, port, extra, home, n_predict):
    log = pathlib.Path(home) / f"serve-{port}.log"
    env = dict(os.environ, KERA_HOME=home)
    with open(log, "w") as lf:
        p = subprocess.Popen([kera, "serve", "-m", model, "--port", str(port), "-c", "512", "-np", "1", "--no-warmup",
                              *extra], stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(600):
                if p.poll() is not None:
                    sys.exit(f"FAIL: the server stopped:\n{log.read_text()[-3000:]}")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except Exception:
                    time.sleep(0.25)
            body = {"prompt": PROMPT, "n_predict": n_predict, "temperature": 0}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        finally:
            p.terminate()
            p.wait(timeout=60)
    text = log.read_text(errors="replace")
    m = re.search(r"saving chat states in (\S+)", text)
    return r["content"], (m.group(1) if m else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--hf", required=True, help="the tiny model in Hugging Face's format")
    ap.add_argument("--gguf", required=True, help="the same model as GGUF")
    ap.add_argument("--esft", action="store_true")
    ap.add_argument("--port", type=int, default=8520)
    args = ap.parse_args()
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-finetune-"))
    data = work / "train.jsonl"
    with open(data, "w") as f:
        for _ in range(8):
            f.write(json.dumps({"prompt": PROMPT, "completion": ANSWER}) + "\n")
    out = work / "out"
    cmd = [sys.executable, str(ROOT / "tools" / "finetune" / "kera_finetune.py"), "--model", args.hf, "--data", str(data),
           "--out", str(out), "--trainer", "peft", "--steps", "60", "--batch", "2", "--lr", "3e-3", "--rank", "16",
           "--alpha", "32", "--outtype", "f32"] + (["--esft", "--esft-share", "0.6"] if args.esft else [])
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        return 1
    print(f"trained in {time.time() - t0:.0f} s")

    home = tempfile.mkdtemp(prefix="kera-ft-home-")
    n = 12
    base, base_dir = serve_and_ask(args.kera, args.gguf, args.port, ["--no-auto-guess"], home, n)
    if args.esft:
        model = out / "model.gguf"
        experts = json.loads((out / "esft_experts.json").read_text())["experts"]
        print(f"ESFT chose experts {experts}")
        if not model.exists() or not experts:
            print("FAIL: ESFT wrote no model or chose no experts")
            return 1
        # named after its base, so kera serve recognises the model and uses its card's settings, as with a LoRA
        sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
        import gguf
        name = gguf.GGUFReader(str(model)).get_field(gguf.Keys.General.NAME).contents()
        base_name = gguf.Metadata.load(None, pathlib.Path(args.hf).resolve(), None, 0).name or pathlib.Path(args.hf).name
        print(f"ESFT model's name: {name!r} (base: {base_name!r})")
        if not name.startswith(base_name):
            print("FAIL: the ESFT model isn't named after its base")
            return 1
        tuned, tuned_dir = serve_and_ask(args.kera, str(model), args.port + 1, ["--no-auto-guess"], home, n)
    else:
        adapter = out / "adapter.gguf"
        if not adapter.exists():
            print("FAIL: no adapter.gguf")
            return 1
        tuned, tuned_dir = serve_and_ask(args.kera, args.gguf, args.port + 1, ["--no-auto-guess", "--lora", str(adapter)],
                                         home, n)
    print(f"without fine-tuning: {base!r}\nwith fine-tuning:    {tuned!r}\ntrained answer:      {ANSWER!r}")
    ok = True
    if not ANSWER.startswith(tuned[: max(8, len(tuned) // 2)]) and not tuned.startswith(ANSWER[:20]):
        print("FAIL: the fine-tuned model doesn't give the trained answer")
        ok = False
    if base.startswith(ANSWER[:20]):
        print("FAIL: the model already gave the trained answer before fine-tuning")
        ok = False
    if not args.esft and (base_dir is None or tuned_dir is None or base_dir == tuned_dir):
        print(f"FAIL: saved chats share a folder with and without the adapter ({base_dir}, {tuned_dir})")
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
