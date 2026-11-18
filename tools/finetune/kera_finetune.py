"""Fine-tune a model and turn the result into a GGUF file Kera can load (spec Phase 10, M11).

Kera doesn't write a trainer (spec Phase 10). It wraps MLX-LM's LoRA on a Mac with Apple silicon, and Hugging Face
PEFT everywhere else (NVIDIA, AMD with PyTorch for ROCm, or the CPU for small models), then uses llama.cpp's own
converter for the GGUF:

  python tools/finetune/kera_finetune.py --model HF_MODEL_DIR --data train.jsonl --out OUT_DIR
  kera serve -m model.gguf --lora OUT_DIR/adapter.gguf

HF_MODEL_DIR is the model in Hugging Face's format (config.json, tokenizer and safetensors): the same model the GGUF
was made from. The data is JSON lines, one example per line, in any of these shapes:

  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  {"prompt": "...", "completion": "..."}
  {"text": "..."}

Only the answers are learned (the assistant's turns, or the completion); plain text is learned whole. With a model
that has a chat template, a prompt and completion are one user turn and one assistant turn in its chat format, as
MLX-LM makes them, so the fine-tune learns the format its chats use, with either trainer.

--esft (mixture-of-experts models): expert-specialised fine-tuning (ESFT, DeepSeek 2024). The training data goes
through the model's routers first; only the experts it uses most are trained, plus a LoRA on attention. The chosen
experts are listed in OUT_DIR/esft_experts.json. A Hugging Face model keeps each layer's experts in one tensor, which
a LoRA can't cover expert by expert, so ESFT writes a whole fine-tuned model (OUT_DIR/model.gguf) instead of an
adapter.

Needs: torch, transformers, peft, safetensors (pip install -r tools/finetune/requirements.txt), or mlx-lm on a Mac.
"""
import argparse
import json
import os
import pathlib
import platform
import random
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def llama_dir():
    """llama.cpp's converters: next to this file in a Kera download, or in the source tree"""
    for c in (HERE / "llama.cpp", HERE.parents[1] / "third_party" / "llama.cpp"):
        if (c / "convert_lora_to_gguf.py").exists():
            return c
    sys.exit("can't find llama.cpp's converters (convert_lora_to_gguf.py)")


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------

def read_examples(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not any(k in r for k in ("messages", "prompt", "text")):
                sys.exit(f"{path} line {n}: expected \"messages\", \"prompt\"/\"completion\" or \"text\"")
            rows.append(r)
    if not rows:
        sys.exit(f"{path} has no examples")
    return rows


def plain_chat(messages, add_generation_prompt=False):
    """For tokenizers without a chat template"""
    out = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages)
    return out + ("Assistant:" if add_generation_prompt else "")


def encode(tok, row, max_len):
    """Token ids and labels (-100 where nothing is learned) for one example"""
    def ids(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    if "text" in row and "messages" not in row and "prompt" not in row:
        x = ids(row["text"])
        return x[:max_len], x[:max_len]
    template = getattr(tok, "chat_template", None)
    if "prompt" in row and "messages" not in row and template:
        # as MLX-LM's CompletionsDataset does: a user turn and an assistant turn in the model's own chat format, which is
        # what kera serve's chat sends (a raw prompt + completion would teach a format the chat never uses)
        row = {"messages": [{"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row.get("completion", "")}],
               **({"tools": row["tools"]} if row.get("tools") else {})}
    if "prompt" in row:
        p, c = ids(row["prompt"]), ids(row.get("completion", "")) + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
        x = p + c
        return x[:max_len], ([-100] * len(p) + c)[:max_len]
    msgs = row["messages"]
    tools = {"tools": row["tools"]} if row.get("tools") else {}
    render = (lambda m, g=False: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=g, **tools)) \
        if template else plain_chat
    x, labels = [], []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = ids(render(msgs[:i], True))
        upto = ids(render(msgs[:i + 1]))
        if upto[:len(x)] != x:  # the template re-renders history differently: learn this turn on its own
            x, labels = [], []
        new_prompt = prefix[len(x):]
        answer = upto[len(prefix):]
        x += new_prompt + answer
        labels += [-100] * len(new_prompt) + answer
    if not x:  # no assistant turn: learn the whole conversation
        x = ids(render(msgs))
        labels = list(x)
    return x[:max_len], labels[:max_len]


# ---------------------------------------------------------------------------------------------------------------
# PEFT (Hugging Face)
# ---------------------------------------------------------------------------------------------------------------

def pick_device(torch):
    if torch.cuda.is_available():  # NVIDIA, or AMD with PyTorch for ROCm
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lora_targets(model):
    """Every linear layer of attention and the dense MLPs; not routers, fused experts or the output layer"""
    import torch
    names = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or ".layers." not in name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("gate", "router") or ".experts." in name:
            continue
        names.add(leaf)
    return sorted(names)


def esft_select(model, batches, share, torch):
    """Expert usage of the training data, per MoE layer (ESFT): the experts whose share of the routing adds up to
    `share`, most used first"""
    counts = {}
    hooks = []

    def make_hook(layer):
        def hook(mod, inp, out):
            # routers return (weights, indices) or logits; take the chosen experts
            idx = None
            if isinstance(out, tuple):
                for t in out:
                    if torch.is_tensor(t) and t.dtype in (torch.int32, torch.int64):
                        idx = t
                        break
            if idx is None:
                logits = out[0] if isinstance(out, tuple) else out
                k = getattr(model.config, "num_experts_per_tok", 2)
                idx = torch.topk(logits.float(), k, dim=-1).indices
            c = counts.setdefault(layer, {})
            for e in idx.reshape(-1).tolist():
                c[e] = c.get(e, 0) + 1
        return hook

    for name, mod in model.named_modules():
        # transformers' routers (e.g. DeepseekV3TopkRouter) return the chosen experts; plain linear gates their logits
        if ".layers." in name and (type(mod).__name__.endswith("Router") or name.endswith((".mlp.gate", ".mlp.router"))):
            layer = int(name.split(".layers.")[1].split(".")[0])
            hooks.append(mod.register_forward_hook(make_hook(layer)))
    with torch.no_grad():
        for x, _ in batches:
            model(input_ids=x)
    for h in hooks:
        h.remove()
    chosen = {}
    for layer, c in counts.items():
        total = sum(c.values())
        run = 0
        for e, n in sorted(c.items(), key=lambda kv: -kv[1]):
            chosen.setdefault(layer, []).append(e)
            run += n
            if run >= share * total:
                break
    return chosen, counts


def train_peft(args, rows):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or pick_device(torch)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    data = [encode(tok, r, args.max_len) for r in rows]
    cut = sum(1 for x, _ in data if len(x) >= args.max_len)
    if cut:
        print(f"note: {cut} of {len(data)} examples are {args.max_len} tokens or longer and are cut there, so their "
              f"ends aren't learned; --max-len sets the limit", flush=True)
    data = [(x, y) for x, y in data if any(v != -100 for v in y)]
    if not data:
        sys.exit("nothing to learn: every example's answer is empty")
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    def batch(items):
        n = max(len(x) for x, _ in items)
        ids = torch.full((len(items), n), pad, dtype=torch.long)
        lab = torch.full((len(items), n), -100, dtype=torch.long)
        att = torch.zeros((len(items), n), dtype=torch.long)
        for i, (x, y) in enumerate(items):
            ids[i, :len(x)] = torch.tensor(x)
            lab[i, :len(y)] = torch.tensor(y)
            att[i, :len(x)] = 1
        return ids.to(device), lab.to(device), att.to(device)

    esft = None
    if args.esft:
        model.eval()
        chosen, counts = esft_select(model, [(batch([d])[0], None) for d in data[:256]], args.esft_share, torch)
        if not chosen:
            sys.exit("--esft: no expert routers found; is this a mixture-of-experts model?")
        esft = {"share": args.esft_share, "experts": {str(k): v for k, v in sorted(chosen.items())},
                "usage": {str(k): {str(e): n for e, n in sorted(c.items())} for k, c in sorted(counts.items())}}
        print(f"ESFT: training {sum(len(v) for v in chosen.values())} of "
              f"{sum(len(c) for c in counts.values())} used experts over {len(chosen)} layers")

    targets = lora_targets(model)
    if esft:
        targets = [t for t in targets if t in ("q_proj", "k_proj", "v_proj", "o_proj", "q_a_proj", "q_b_proj",
                                               "kv_a_proj_with_mqa", "kv_b_proj", "qkv_proj")] or targets
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, target_modules=targets, bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    masks = []
    if esft:
        # the chosen experts' slices of the fused expert tensors learn too; the rest stays as it was
        for name, p in model.named_parameters():
            if ".experts." in name and p.dim() == 3 and ".layers." in name:
                layer = int(name.split(".layers.")[1].split(".")[0])
                keep = chosen.get(layer, [])
                if keep:
                    p.requires_grad_(True)
                    m = torch.zeros(p.shape[0], 1, 1, dtype=p.dtype, device=p.device)
                    m[keep] = 1
                    masks.append((p, m))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"training {sum(p.numel() for p in params) / 1e6:.2f}M parameters ({', '.join(targets)}"
          f"{' and the chosen experts' if masks else ''}) on {device}, {len(data)} examples")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    model.train()
    t0 = time.time()
    order = []
    for step in range(1, args.steps + 1):
        items = []
        while len(items) < args.batch:
            if not order:
                order = list(range(len(data)))
                random.shuffle(order)
            items.append(data[order.pop()])
        ids, lab, att = batch(items)
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for g in opt.param_groups:
            g["lr"] = lr
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        for p, m in masks:
            if p.grad is not None:
                p.grad.mul_(m)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"  step {step}/{args.steps}: loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)

    out = pathlib.Path(args.out)
    if esft:
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "hf", safe_serialization=True)
        tok.save_pretrained(out / "hf")
        (out / "esft_experts.json").write_text(json.dumps(esft, indent=1) + "\n")
        return "model", out / "hf"
    model.save_pretrained(out / "peft")
    return "lora", out / "peft"


# ---------------------------------------------------------------------------------------------------------------
# MLX-LM (Mac with Apple silicon)
# ---------------------------------------------------------------------------------------------------------------

def have_mlx():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_lm  # noqa: F401
        return True
    except Exception:
        return False


def mlx_to_peft(mlx_dir, peft_dir, base):
    """MLX-LM's LoRA (x + scale * (x @ a) @ b, a: [in, r], b: [r, out]) in PEFT's layout
    (x + alpha / r * B(A(x)), A: [r, in], B: [out, r])"""
    from safetensors.numpy import load_file, save_file
    cfg = json.loads((mlx_dir / "adapter_config.json").read_text())
    lp = cfg.get("lora_parameters", {})
    rank, scale = int(lp.get("rank", 8)), float(lp.get("scale", 20.0))
    w = load_file(str(mlx_dir / "adapters.safetensors"))
    out, modules = {}, set()
    for k, v in w.items():
        if not k.endswith((".lora_a", ".lora_b")):
            continue
        mod = k.rsplit(".", 1)[0]
        modules.add(mod.rsplit(".", 1)[-1])
        name = "lora_A" if k.endswith(".lora_a") else "lora_B"
        out[f"base_model.model.{mod}.{name}.weight"] = v.T.copy()
    peft_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(peft_dir / "adapter_model.safetensors"))
    (peft_dir / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": scale * rank, "lora_dropout": 0.0,
        "target_modules": sorted(modules), "bias": "none", "base_model_name_or_path": str(base)}, indent=1) + "\n")


def mlx_command(args, rows, data):
    """MLX-LM's lora command and its config: the same LoRA as PEFT's (rank, and scale = alpha / rank: MLX-LM's own
    default scale is 20, so the same data trained about 10 times harder on a Mac, and forgot more of the base model)
    and the same longest example (--max-len; MLX-LM's own default is 2048, PEFT's here 1024)"""
    config = {"lora_parameters": {"rank": args.rank, "scale": args.alpha / args.rank, "dropout": 0.0}}
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", args.model, "--train", "--data", str(data),
           "--iters", str(args.steps), "--batch-size", str(args.batch), "--learning-rate", str(args.lr),
           "--num-layers", "-1", "--adapter-path", str(pathlib.Path(args.out) / "mlx"), "--seed", str(args.seed),
           "--max-seq-length", str(args.max_len), "--config", str(data / "lora.yaml")]
    if any("prompt" in r or "messages" in r for r in rows):
        cmd.append("--mask-prompt")
    return cmd, config


def train_mlx(args, rows):
    out = pathlib.Path(args.out)
    data = out / "mlx-data"
    data.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in rows]
    (data / "train.jsonl").write_text("\n".join(lines) + "\n")
    (data / "valid.jsonl").write_text("\n".join(lines[: max(1, len(lines) // 10)]) + "\n")
    cmd, config = mlx_command(args, rows, data)
    (data / "lora.yaml").write_text(json.dumps(config) + "\n")  # JSON is YAML
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    mlx_to_peft(out / "mlx", out / "peft", args.model)
    return "lora", out / "peft"


# ---------------------------------------------------------------------------------------------------------------

def base_model_name(model_dir, env):
    """The name llama.cpp's converter gives the base model (its card, its config's name, else its folder's name)"""
    code = ("import gguf, pathlib, sys; p = pathlib.Path(sys.argv[1]).resolve(); "
            "print(gguf.Metadata.load(None, p, None, 0).name or p.name)")
    r = subprocess.run([sys.executable, "-c", code, str(model_dir)], env=env, capture_output=True, text=True)
    name = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
    return name or pathlib.Path(model_dir).resolve().name


def to_gguf(kind, path, args):
    llama = llama_dir()
    env = dict(os.environ, PYTHONPATH=str(llama / "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = pathlib.Path(args.out)
    if kind == "lora":
        target = out / "adapter.gguf"
        cmd = [sys.executable, str(llama / "convert_lora_to_gguf.py"), str(path), "--base", args.model,
               "--outtype", args.outtype, "--outfile", str(target)]
    else:
        target = out / "model.gguf"
        # named after the base model (saved in a folder called "hf", it would be named "Hf"), so kera serve knows the
        # model and uses its card's settings, as for a LoRA on the base (D-013)
        cmd = [sys.executable, str(llama / "convert_hf_to_gguf.py"), str(path), "--outtype", args.outtype,
               "--outfile", str(target), "--model-name", base_model_name(args.model, env) + " ESFT"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"llama.cpp's converter failed:\n{r.stderr[-3000:]}")
    return target


# The default length (the owner's choice, 9 Nov): about three passes over the data, as Hugging Face's own Trainer
# defaults to (num_train_epochs 3). A fixed number trained 10 examples 80 times over, which makes a model repeat them
# and forget more of what it knew, and covered only part of a big set. At least a few steps, so a tiny set still
# trains; at most a few thousand, so a huge one doesn't run for days (--steps sets any number).
PASSES, MIN_STEPS, MAX_STEPS = 3, 10, 3000


def default_steps(n_examples, batch):
    return max(MIN_STEPS, min(MAX_STEPS, -(-PASSES * max(1, n_examples) // max(1, batch))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the base model in Hugging Face's format (a folder)")
    ap.add_argument("--data", required=True, help="training examples, JSON lines")
    ap.add_argument("--out", required=True, help="folder for the results")
    ap.add_argument("--trainer", choices=["auto", "peft", "mlx"], default="auto",
                    help="auto: MLX-LM on a Mac with Apple silicon when installed, else PEFT")
    ap.add_argument("--steps", type=int, default=None,
                    help=f"training steps (default: about {PASSES} passes over the data, {MIN_STEPS} to {MAX_STEPS} steps)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--esft", action="store_true", help="MoE models: train the experts the data uses most, plus attention")
    ap.add_argument("--esft-share", type=float, default=0.5, help="train experts until they cover this share of the routing")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    ap.add_argument("--device", default="", help="cuda, mps or cpu (default: the fastest found)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_examples(args.data)
    if args.steps is None:
        args.steps = default_steps(len(rows), args.batch)
        passes = args.steps * args.batch / max(1, len(rows))
        print(f"{args.steps} steps: about {passes:.1f} passes over the {len(rows)} examples (--steps sets another number)")
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer = args.trainer
    if trainer == "auto":
        trainer = "mlx" if have_mlx() and not args.esft else "peft"
    print(f"Kera fine-tuning: {len(rows)} examples, {trainer.upper()} trainer, {args.steps} steps")
    kind, path = train_mlx(args, rows) if trainer == "mlx" else train_peft(args, rows)
    target = to_gguf(kind, path, args)
    meta = {"base": str(pathlib.Path(args.model).resolve()), "data": str(pathlib.Path(args.data).resolve()),
            "examples": len(rows), "trainer": trainer, "steps": args.steps, "rank": args.rank, "kind": kind,
            "gguf": target.name, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    (pathlib.Path(args.out) / "kera_finetune.json").write_text(json.dumps(meta, indent=1) + "\n")
    if kind == "lora":
        print(f"done: {target}\n  use it with: kera serve -m <the base model's GGUF> --lora {target}")
    else:
        print(f"done: {target} (a whole model: ESFT changed some experts)\n  use it with: kera serve -m {target}")


if __name__ == "__main__":
    main()
