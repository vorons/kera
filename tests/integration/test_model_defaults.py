"""The model card's sampling per request (spec 4c item 2; hook 4; docs/DECISIONS.md D-013).

  python tests/integration/test_model_defaults.py --kera build/bin/kera --model tiny/nemotron_h_moe.gguf

Runs `kera serve` with a Nemotron-H model and Nemotron 3 Nano's own chat template, sends chat requests
that set no sampling, and reads back what the server used (GET /slots): thinking on gives temperature 1.0
and top_p 1.0, thinking off greedy, tools 0.6 and 0.95, as NVIDIA's card says. A request's own values
always win, and --no-model-defaults leaves the server's defaults.

The card is only the card's own model's (rule one, D-033): the tiny model gets Nemotron 3 Nano's name in a copy;
under its own name it is another model and keeps the server's defaults. A file whose own defaults
(general.sampling) differ from the card's keeps them, and a value the user sets on the command line wins over the
card, as in stock llama.cpp.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
TEMPLATE = ROOT / "third_party" / "llama.cpp" / "models" / "templates" / "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "The weather in a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def used(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=30) as r:
        p = json.loads(r.read())[0]["params"]
    return round(p["temperature"], 3), round(p["top_p"], 3)


def renamed_copy(src, dst, name, sampling=None):
    """A copy of a GGUF with another general.name and, if given, its own general.sampling values."""
    import gguf
    r = gguf.GGUFReader(src)
    w = gguf.GGUFWriter(dst, arch=r.get_field(gguf.Keys.General.ARCHITECTURE).contents(), endianess=r.endianess)
    align = r.get_field(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        w.data_alignment = align.contents()
    for f in r.fields.values():
        if f.name in (gguf.Keys.General.ARCHITECTURE, gguf.Keys.General.NAME) or f.name.startswith("GGUF."):
            continue
        sub = f.types[-1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), f.types[0], sub_type=sub)
    w.add_key_value(gguf.Keys.General.NAME, name, gguf.GGUFValueType.STRING)
    for k, v in (sampling or {}).items():
        w.add_key_value(f"general.sampling.{k}", v,
                        gguf.GGUFValueType.INT32 if isinstance(v, int) else gguf.GGUFValueType.FLOAT32)
    for tensor in r.tensors:
        w.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for tensor in r.tensors:
        w.write_tensor_data(tensor.data, tensor_endianess=r.endianess)
    w.close()
    return str(dst)


def serve(kera, model, port, extra, template=TEMPLATE):
    # "--model-from-env": the model given through LLAMA_ARG_MODEL instead of -m, as llama.cpp also takes it
    from_env = "--model-from-env" in extra
    extra = [e for e in extra if e != "--model-from-env"]
    env = dict(os.environ, LLAMA_ARG_MODEL=str(model)) if from_env else None
    proc = subprocess.Popen([kera, "serve", *([] if from_env else ["-m", model]), "--port", str(port), "-c", "1024",
                             "-np", "1", "--jinja", *([] if template is None else ["--chat-template-file", str(template)]),
                             "--no-warmup", "--no-kv-store", *extra],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, env=env)
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise SystemExit("the server didn't start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8433)
    ap.add_argument("--qwen-model", help="tiny/qwen3_dense.gguf: also checks Qwen3's card (hybrid and 2507 Instruct)")
    ap.add_argument("--listed-model", help="qwen2.5-0.5b-instruct-q4_k_m.gguf (Qwen's own file, which carries no sampling "
                                           "of its own): also checks the makers' values from Kera's built-in list")
    args = ap.parse_args()
    msg = [{"role": "user", "content": "What is an SSD?"}]
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-defaults-"))
    nano = renamed_copy(args.model, work / "nano.gguf", "NVIDIA Nemotron 3 Nano tiny")
    own = renamed_copy(args.model, work / "own.gguf", "NVIDIA Nemotron 3 Nano tiny", {"temp": 0.3, "top_p": 0.7})
    cases = [
        ("thinking on (the default)", {"messages": msg}, (1.0, 1.0)),
        ("thinking off", {"messages": msg, "chat_template_kwargs": {"enable_thinking": False}}, (0.0, None)),
        ("tools offered", {"messages": msg, "tools": TOOLS}, (0.6, 0.95)),
        ("the request's own temperature", {"messages": msg, "temperature": 0.3}, (0.3, 1.0)),
    ]
    failed = 0
    proc = serve(args.kera, nano, args.port, [])
    try:
        for name, body, (want_t, want_p) in cases:
            post(args.port, dict(body, max_tokens=4))
            t, p = used(args.port)
            ok = t == want_t and (want_p is None or p == want_p)
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name}: temperature {t}, top_p {p}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    # each: the server's options, the request, what the server must use, why
    others = [
        (nano, ["--no-model-defaults", "--temp", "0.8", "--top-p", "0.9"], {"tools": TOOLS}, (0.8, 0.9),
         "--no-model-defaults keeps the server's own"),
        (nano, ["--temp", "0.5"], {}, (0.5, 1.0), "the command line's temperature wins over the card's; top_p is the card's"),
        (nano, ["--top-p", "0.6"], {"tools": TOOLS}, (0.6, 0.6), "the command line's top_p wins over the card's"),
        (own, [], {}, (0.3, 0.7), "a file with its own sampling (unlike the card's) keeps it"),
        (own, [], {"chat_template_kwargs": {"enable_thinking": False}}, (0.3, 0.7), "for every kind of request"),
        (args.model, [], {}, (0.8, 0.95), "another model of the same design (not the card's by name): llama.cpp's own"),
        (nano, ["--model-from-env"], {}, (1.0, 1.0),
         "a model given through LLAMA_ARG_MODEL, not -m, gets the card's too (filled in once llama.cpp has the file)"),
    ]
    if args.qwen_model:
        hybrid = renamed_copy(args.qwen_model, work / "qwen3.gguf", "Qwen3-0.6B")
        instruct = renamed_copy(args.qwen_model, work / "qwen3-2507.gguf", "Qwen3-4B-Instruct-2507")
        hybrid_t = ROOT / "third_party" / "llama.cpp" / "models" / "templates" / "Qwen-Qwen3-0.6B.jinja"
        instruct_t = ROOT / "tests" / "integration" / "templates" / "Qwen3-4B-Instruct-2507.jinja"
        think = {"chat_template_kwargs": {"enable_thinking": True}}
        no_think = {"chat_template_kwargs": {"enable_thinking": False}}
        others += [
            (hybrid, [], {}, (0.6, 0.95), "Qwen3 thinking (its default): the card's thinking values", hybrid_t),
            (hybrid, [], no_think, (0.7, 0.8), "Qwen3 thinking off: the card's other values", hybrid_t),
            (instruct, [], {}, (0.7, 0.8), "Qwen3-2507 Instruct: its card's only values", instruct_t),
            (instruct, [], think, (0.7, 0.8), "and still when a request asks it to think, which it can't", instruct_t),
        ]
    if args.listed_model:
        # a model with no card read by hand and no sampling in its file: its makers' generation_config.json values
        # from the built-in list (D-036), found by its name and shape; its own chat template
        mine = renamed_copy(args.listed_model, work / "mine.gguf", "My Assistant 0.5B")
        own_t = renamed_copy(args.listed_model, work / "own-temp.gguf", "qwen2.5-0.5b-instruct", {"temp": 0.3})
        others += [
            (args.listed_model, [], {}, (0.7, 0.8), "Qwen2.5-0.5B-Instruct: its makers' generation_config.json", None),
            (args.listed_model, [], {"temperature": 0.3}, (0.3, 0.8), "the request's own temperature wins", None),
            (args.listed_model, ["--temp", "0.5"], {}, (0.5, 0.8), "the command line's temperature wins", None),
            (args.listed_model, ["--no-model-defaults"], {}, (0.8, 0.95), "--no-model-defaults: llama.cpp's own", None),
            (mine, [], {}, (0.8, 0.95), "the same weights under another name (a fine-tune): llama.cpp's own", None),
            (own_t, [], {}, (0.3, 0.95), "a file with sampling of its own keeps it, and gets nothing from the list", None),
        ]
    for model, extra, body, want, name, *template in others:
        proc = serve(args.kera, model, args.port, extra, *template)
        try:
            post(args.port, dict({"messages": msg, "max_tokens": 4}, **body))
            t, p = used(args.port)
            ok = (t, p) == want
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name}: temperature {t}, top_p {p}")
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    print("PASS" if failed == 0 else f"FAILED ({failed})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
