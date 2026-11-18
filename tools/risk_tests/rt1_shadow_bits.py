"""M1 risk test 1: Shadow Bits go/no-go, and which version (spec P5 item 1, decides M7).

Four versions on a real dense model, all measured on the same 200 prompts:
  sb4      SB4 as written: 2-bit base + 2-bit correction, the base from Appendix A's grid search (alpha 0.2)
  bn2      SB4 with BitNest's base: the same 2+2 layout, the base from GPTQ on calibration text, the
           correction a closed-form rounding of what's left (BitNest, arXiv 2610.02800, without its rotations)
  bn4      4+4: BitNest's own layout: GPTQ 4-bit base (group 128) + 4-bit correction = 8-bit full model
  draft    no nesting: plain 4-bit weights plus a small draft model from the same family
For each: the per-token acceptance (how often the draft's top word is the full model's top word, on text
the 16-bit model wrote: what a greedy check accepts) and the full model's KL divergence from the 16-bit
model, next to its plain counterpart's (4-bit, or 8-bit for 4+4).

  python tools/risk_tests/rt1_shadow_bits.py --model DIR --draft DIR --work DIR [--prompts 200] [--new 64]

Each step saves its result in --work, so a run can be resumed. Runs on the CPU in 32-bit floats.
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))
import shadow_bits as sb  # noqa: E402

TOPK = 64          # reference log-probabilities kept per position (the rest is one "other" bucket)
SHRINKS = torch.linspace(0.5, 1.0, 26)


# ---------------------------------------------------------------------------------------------------
# Quantizers (torch versions of reference/shadow_bits.py, checked against it in self_test)
# ---------------------------------------------------------------------------------------------------

def f16(x):
    return x.to(torch.float16).to(torch.float32)


def sb4_encode(W, group=64, alpha=0.2):
    """Appendix A's encode_sb4: returns (base, full) as dense float tensors."""
    G = W.reshape(-1, group)
    var = G.var(1, unbiased=False, keepdim=True) + 1e-12
    lo0, hi0 = G.min(1, keepdim=True).values, G.max(1, keepdim=True).values
    best = None
    for c in SHRINKS:
        centre, half = (lo0 + hi0) / 2, (hi0 - lo0) / 2 * c
        lo, hi = centre - half, centre + half
        m = f16(lo)
        s = f16(torch.clamp((hi - lo) / 3, min=1e-8))
        q_hi = torch.clamp(torch.round((G - m) / s), 0, 3)
        base = m + s * q_hi
        q_lo = torch.clamp(torch.round((G - base) / (s / 4) + 1.5), 0, 3)
        full = base + (s / 4) * (q_lo - 1.5)
        score = alpha * ((G - base) ** 2).mean(1, keepdim=True) / var + (1 - alpha) * ((G - full) ** 2).mean(1, keepdim=True) / var
        if best is None:
            best = [score, base, full]
        else:
            better = score < best[0]
            best = [torch.where(better, a, b) for a, b in zip([score, base, full], best)]
    return best[1].reshape(W.shape), best[2].reshape(W.shape)


def rtn(W, bits, group):
    """reference rtn(): plain uniform quantizer with the best clipping per group."""
    G = W.reshape(-1, group)
    levels = 2 ** bits - 1
    lo0, hi0 = G.min(1, keepdim=True).values, G.max(1, keepdim=True).values
    best_err, best_q = None, None
    for c in SHRINKS:
        centre, half = (lo0 + hi0) / 2, (hi0 - lo0) / 2 * c
        lo, hi = centre - half, centre + half
        m = f16(lo)
        s = f16(torch.clamp((hi - lo) / levels, min=1e-8))
        q = m + s * torch.clamp(torch.round((G - m) / s), 0, levels)
        err = ((G - q) ** 2).mean(1, keepdim=True)
        if best_err is None:
            best_err, best_q = err, q
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q = torch.where(better, q, best_q)
    return best_q.reshape(W.shape)


def group_params_asym(Wg, levels):
    """Offset and scale for one group of columns (rows x group), with the best clipping (as rtn)."""
    lo0, hi0 = Wg.min(1, keepdim=True).values, Wg.max(1, keepdim=True).values
    best = None
    for c in SHRINKS:
        centre, half = (lo0 + hi0) / 2, (hi0 - lo0) / 2 * c
        m = f16(centre - half)
        s = f16(torch.clamp(2 * half / levels, min=1e-8))
        q = m + s * torch.clamp(torch.round((Wg - m) / s), 0, levels)
        err = ((Wg - q) ** 2).sum(1, keepdim=True)
        if best is None:
            best = [err, m, s]
        else:
            better = err < best[0]
            best = [torch.where(better, a, b) for a, b in zip([err, m, s], best)]
    return best[1], best[2]


def group_params_sym(Wg, qmax):
    amax = Wg.abs().max(1, keepdim=True).values
    best = None
    for c in SHRINKS:
        s = f16(torch.clamp(amax * c / qmax, min=1e-8))
        q = s * torch.clamp(torch.round(Wg / s), -qmax - 1, qmax)
        err = ((Wg - q) ** 2).sum(1, keepdim=True)
        if best is None:
            best = [err, s]
        else:
            better = err < best[0]
            best = [torch.where(better, a, b) for a, b in zip([err, s], best)]
    return best[1]


def gptq(W, H, mode, group, blocksize=128, damp=0.01):
    """GPTQ (Frantar et al. 2022) with per-group parameters set when each group starts.
    mode "asym2": codes 0..3 with offset m and scale s (the SB4 base layout).
    mode "sym4":  codes -8..7 with scale s (BitNest's base).
    Returns the base weights and, per column, the group's (m, s) so a correction can be added."""
    W = W.clone().float()
    rows, cols = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    M = torch.zeros(rows, cols // group)
    S = torch.zeros(rows, cols // group)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1, Q1, Err1 = W[:, i1:i2].clone(), torch.zeros(rows, i2 - i1), torch.zeros(rows, i2 - i1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:
                g = col // group
                Wg = W1[:, i:i + group]          # groups never cross a block: the block size is a multiple of it
                if mode == "asym2":
                    m, s = group_params_asym(Wg, 3)
                    M[:, g], S[:, g] = m[:, 0], s[:, 0]
                else:
                    S[:, g] = group_params_sym(Wg, 7)[:, 0]
            g = col // group
            w = W1[:, i]
            if mode == "asym2":
                q = M[:, g] + S[:, g] * torch.clamp(torch.round((w - M[:, g]) / S[:, g]), 0, 3)
            else:
                q = S[:, g] * torch.clamp(torch.round(w / S[:, g]), -8, 7)
            Q1[:, i] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, M, S


def bn2_full(W, base, M, S, group):
    """BitNest-style correction on the SB4 layout: round what's left to the 2-bit grid at s/4."""
    s = S.repeat_interleave(group, dim=1)
    q_lo = torch.clamp(torch.round((W - base) / (s / 4) + 1.5), 0, 3)
    return base + (s / 4) * (q_lo - 1.5)


def bn4_full(W, base, S, group):
    """BitNest's refinement: q_r = round((W - base) / (s/16)) in -8..7 (0..7 where the base code is -8)."""
    s = S.repeat_interleave(group, dim=1)
    q4 = torch.round(base / s)
    q_r = torch.round((W - base) / (s / 16))
    q_r = torch.where(q4 <= -8, torch.clamp(q_r, 0, 7), torch.clamp(q_r, -8, 7))
    return (16 * q4 + q_r) * (s / 16)


def self_test():
    """The torch quantizers must match reference/shadow_bits.py on random weights."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 128)).astype(np.float32)
    ref = sb.encode_sb4(w, group=64, alpha=0.2)
    base, full = sb4_encode(torch.from_numpy(w), 64, 0.2)
    assert np.allclose(base.numpy(), sb.decode_base(ref), atol=1e-6), "sb4 base differs from the reference"
    assert np.allclose(full.numpy(), sb.decode_full(ref), atol=1e-6), "sb4 full differs from the reference"
    assert np.allclose(rtn(torch.from_numpy(w), 4, 64).numpy(), sb.rtn(w, 4, 64), atol=1e-6), "rtn differs"
    print("quantizers match reference/shadow_bits.py")


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------

def load_prompts(parquet, n, seed=1):
    """The evaluation prompts (English opening messages), and calibration text for GPTQ: English assistant
    answers from the other conversations (oasst1's validation split has only about 150 suitable openers,
    all used for evaluation)."""
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    roots = [r for r in rows if r["parent_id"] is None and r["lang"] == "en" and 40 <= len(r["text"]) <= 1200]
    random.Random(seed).shuffle(roots)
    evaluated = roots[:n]
    trees = {r["message_tree_id"] for r in evaluated}
    calib = [r["text"] for r in rows if r["role"] == "assistant" and r["lang"] == "en" and len(r["text"]) >= 200
             and r["message_tree_id"] not in trees]
    random.Random(seed).shuffle(calib)
    return [r["text"] for r in evaluated], calib[:200]


def linear_layers(model):
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            out.append((name, mod))
    return out


@torch.no_grad()
def generate_eval(model, tok, prompts, n_new, batch=16):
    """The 16-bit model's own greedy answers: the text every version is measured on."""
    tok.padding_side = "left"
    seqs = []
    for i in range(0, len(prompts), batch):
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False) for p in prompts[i:i + batch]]
        enc = tok(msgs, return_tensors="pt", padding=True)
        out = model.generate(**enc, max_new_tokens=n_new, do_sample=False)
        stop = {tok.eos_token_id, tok.pad_token_id} | set(getattr(model.generation_config, "eos_token_id", None) or [])
        for j in range(out.shape[0]):
            n_pad = int((enc["attention_mask"][j] == 0).sum().item())
            n_prompt = int(enc["attention_mask"][j].sum().item())
            ids = out[j][n_pad:].tolist()                      # drop the left padding
            answer = ids[n_prompt:]
            for k, x in enumerate(answer):                     # keep up to and including the end-of-text word
                if x in stop:
                    answer = answer[:k + 1]
                    break
            if answer:
                seqs.append({"ids": ids[:n_prompt] + answer, "n_prompt": n_prompt})
        print(f"  generated {len(seqs)}/{len(prompts)}", flush=True)
    return seqs


@torch.no_grad()
def eval_logprobs(model, seqs, batch=8):
    """Yields (seq index, log-probabilities [n_new, vocab]) for the answer positions of every sequence."""
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        L = max(len(s["ids"]) for s in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        mask = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, s in enumerate(chunk):           # right padding: positions keep their meaning
            ids[j, :len(s["ids"])] = torch.tensor(s["ids"])
            mask[j, :len(s["ids"])] = 1
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        for j, s in enumerate(chunk):
            a, b = s["n_prompt"] - 1, len(s["ids"]) - 1      # positions predicting the answer tokens
            yield i + j, torch.log_softmax(logits[j, a:b], dim=-1)


# ---------------------------------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------------------------------

def load_model(path):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    m.eval()
    return m


def step_reference(args, work, tok):
    f = work / "reference.pt"
    if f.exists():
        return torch.load(f)
    model = load_model(args.model)
    prompts, calib = load_prompts(args.oasst, args.prompts)
    print("generating the 16-bit model's answers...")
    seqs = generate_eval(model, tok, prompts, args.new)
    ref = {"seqs": seqs, "calib": calib, "top_ids": [], "top_lp": [], "argmax": []}
    for i, lp in eval_logprobs(model, seqs):
        v, ix = torch.topk(lp, TOPK, dim=-1)
        ref["top_ids"].append(ix.to(torch.int32))
        ref["top_lp"].append(v)
        ref["argmax"].append(ix[:, 0].clone())
    torch.save(ref, f)
    return ref


@torch.no_grad()
def calibration_hessians_and_quantize(model, tok, calib_texts, quantize_fn, n_seq=32, seq_len=256):
    """Sequential GPTQ: for each decoder layer, collect each linear layer's input Hessian on calibration
    text that has passed through the already-quantized earlier layers, then quantize that layer."""
    enc = [tok(t, return_tensors="pt")["input_ids"][0] for t in calib_texts]
    flat = torch.cat(enc)
    n_seq = min(n_seq, flat.numel() // seq_len)
    batch = flat[: n_seq * seq_len].reshape(n_seq, seq_len)
    layers = model.model.layers
    cache = {}

    class Caught(Exception):
        pass

    def catch(mod, args_, kwargs):
        cache["x"] = args_[0] if args_ else kwargs.pop("hidden_states")
        cache["kwargs"] = kwargs
        raise Caught

    h = layers[0].register_forward_pre_hook(catch, with_kwargs=True)
    try:
        model(input_ids=batch, use_cache=False)
    except Caught:
        pass
    h.remove()
    x, kwargs = cache["x"], cache["kwargs"]
    assert all(blk % g == 0 for blk, g in ((128, 64), (128, 128)))
    for li, layer in enumerate(layers):
        lins = {n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
        H = {n: torch.zeros(m.in_features, m.in_features, dtype=torch.float64) for n, m in lins.items()}
        cnt = {n: 0 for n in lins}

        def make_hook(name):
            def hook(mod, inp, out):
                a = inp[0].reshape(-1, inp[0].shape[-1]).double()
                H[name] += a.T @ a
                cnt[name] += a.shape[0]
            return hook

        hooks = [m.register_forward_hook(make_hook(n)) for n, m in lins.items()]
        layer(x, **kwargs)
        for hk in hooks:
            hk.remove()
        for n, m in lins.items():
            Hn = (2 * H[n] / cnt[n]).float()
            m.weight.data = quantize_fn(m.weight.data, Hn)
        out = layer(x, **kwargs)                 # the next layer's input, through this layer's quantized weights
        x = out[0] if isinstance(out, tuple) else out
        print(f"  layer {li + 1}/{len(layers)} quantized", flush=True)


def build_variant(args, name, tok, ref):
    """Returns a model with the variant's weights."""
    if name == "draft4":
        model = load_model(args.draft)
        for _, mod in linear_layers(model):
            mod.weight.data = rtn(mod.weight.data, 4, 64)
        return model
    model = load_model(args.model)
    lins = linear_layers(model)
    if name in ("plain4", "plain8", "sb4_base", "sb4_full"):
        for _, mod in lins:
            W = mod.weight.data
            if name == "plain4":
                mod.weight.data = rtn(W, 4, 64)
            elif name == "plain8":
                mod.weight.data = rtn(W, 8, 128)
            else:
                base, full = sb4_encode(W, 64, 0.2)
                mod.weight.data = base if name == "sb4_base" else full
        return model
    # GPTQ variants: bn2_* (2-bit base, group 64) and bn4_* (4-bit base, group 128)
    group = 64 if name.startswith("bn2") else 128
    mode = "asym2" if name.startswith("bn2") else "sym4"
    want_full = name.endswith("_full")

    def qfn(W, H):
        base, M, S = gptq(W, H, mode, group)
        if not want_full:
            return base
        return bn2_full(W, base, M, S, group) if mode == "asym2" else bn4_full(W, base, S, group)

    calibration_hessians_and_quantize(model, tok, ref["calib"], qfn)
    return model


def step_variant(args, work, tok, ref, name):
    f = work / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    t0 = time.time()
    print(f"{name}: quantizing...", flush=True)
    model = build_variant(args, name, tok, ref)
    print(f"{name}: measuring...", flush=True)
    argmax, kls = [], []
    for i, lp in eval_logprobs(model, ref["seqs"]):
        argmax.append(lp.argmax(-1).tolist())
        ids, rlp = ref["top_ids"][i].long(), ref["top_lp"][i]
        p = rlp.exp()
        q_top = torch.gather(lp, 1, ids)
        r_rest = torch.clamp(1 - p.sum(-1), min=1e-12)
        q_rest = torch.clamp(1 - q_top.exp().sum(-1), min=1e-12)
        kl = (p * (rlp - q_top)).sum(-1) + r_rest * (torch.log(r_rest) - torch.log(q_rest))
        kls.extend(kl.tolist())
    agree = [a == r for am, ra in zip(argmax, ref["argmax"]) for a, r in zip(am, ra.tolist())]
    res = {"name": name, "kl_mean": float(np.mean(kls)), "kl_p99": float(np.percentile(kls, 99)),
           "argmax": argmax, "top1_agree_16bit": float(np.mean(agree)), "seconds": round(time.time() - t0, 1)}
    f.write_text(json.dumps(res))
    del model
    return res


def acceptance(draft, target):
    pairs = [(d, t) for dm, tm in zip(draft["argmax"], target["argmax"]) for d, t in zip(dm, tm)]
    return float(np.mean([d == t for d, t in pairs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the dense model (Hugging Face folder)")
    ap.add_argument("--draft", required=True, help="a small model from the same family")
    ap.add_argument("--oasst", required=True, help="OpenAssistant oasst1 parquet file (prompts and calibration text)")
    ap.add_argument("--work", required=True, type=pathlib.Path)
    ap.add_argument("--prompts", type=int, default=200)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.work.mkdir(parents=True, exist_ok=True)
    self_test()
    tok = AutoTokenizer.from_pretrained(args.model)
    ref = step_reference(args, args.work, tok)
    if not ref["calib"]:  # a reference saved before calibration text came from assistant answers
        ref["calib"] = load_prompts(args.oasst, args.prompts)[1]
    print(f"calibration: {len(ref['calib'])} texts", flush=True)
    print(f"reference: {len(ref['seqs'])} prompts, {sum(len(t) for t in ref['argmax'])} answer tokens")
    res = {}
    for name in ["plain4", "plain8", "sb4_base", "sb4_full", "bn2_base", "bn2_full", "bn4_base", "bn4_full", "draft4"]:
        res[name] = step_variant(args, args.work, tok, ref, name)
        print(f"  {name}: KL {res[name]['kl_mean']:.4f}, top word = 16-bit's {res[name]['top1_agree_16bit']:.1%} "
              f"({res[name]['seconds']} s)", flush=True)

    versions = {
        "sb4":   {"draft": "sb4_base", "full": "sb4_full", "plain": "plain4"},
        "bn2":   {"draft": "bn2_base", "full": "bn2_full", "plain": "plain4"},
        "bn4":   {"draft": "bn4_base", "full": "bn4_full", "plain": "plain8"},
        "draft": {"draft": "draft4",   "full": "plain4",   "plain": "plain4"},
    }
    summary = {}
    print("\nversion  acceptance  KL(full)  KL(plain)  KL ratio")
    for v, d in versions.items():
        a = acceptance(res[d["draft"]], res[d["full"]])
        klf, klp = res[d["full"]]["kl_mean"], res[d["plain"]]["kl_mean"]
        summary[v] = {"acceptance": a, "kl_full": klf, "kl_plain": klp, "kl_ratio": klf / klp if klp > 0 else math.inf,
                      "draft": d["draft"], "full": d["full"], "plain": d["plain"]}
        print(f"{v:7s}  {a:10.3f}  {klf:8.4f}  {klp:9.4f}  {klf / klp:8.3f}")
    (args.work / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
