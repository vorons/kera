"""Analyses routing traces from `kera trace` (M1 risk test 2, spec P5 item 2).

  python tools/risk_tests/rt2_analyze.py TRACE_DIR MODEL.gguf [--json OUT.json]

Computes, per the spec:
  1. hit rate against cache size: an expert cache per layer holding a share of the layer's experts,
     with plain LRU and with "learned pins" (the most used experts, learned on the first half of the
     traces and tested on the second half, as Phase 4 does);
  2. how predictable the next MoE layer's experts are: the next layer's own router applied to the
     residual stream one layer early (next-layer lookahead), against two simple baselines;
  3. union growth: distinct experts per layer for k consecutive words, k = 1..8 (what each extra
     guessed word adds, Phase 6);
  4. how many experts a prompt of a given length touches.
"""
import argparse
import json
import pathlib
import sys
from collections import OrderedDict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402


def load(trace_dir):
    d = pathlib.Path(trace_dir)
    meta = json.loads((d / "meta.json").read_text())
    topk = {L: np.load(d / f"topk_L{L}.npy") for L in meta["moe_layers"]}
    hidden = {}
    for L in meta["moe_layers"]:
        p = d / f"hidden_L{L}.npy"
        if p.exists():
            hidden[L] = np.load(p)
    return meta, topk, hidden


def gguf_info(path):
    r = gguf.GGUFReader(path)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()

    def field(name, default=None):
        f = r.fields.get(name)
        return default if f is None else f.parts[-1][0]

    tensors = {t.name: t for t in r.tensors}

    def tensor(name):
        t = tensors.get(name)
        if t is None:
            return None
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            a = np.array(t.data, dtype=np.float32)
        else:
            a = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        return a.reshape([int(x) for x in reversed(t.shape)])

    return {
        "n_expert": int(field(f"{arch}.expert_count")),
        "eps": float(field(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)),
        "tensor": tensor,
    }


def segments_of(meta):
    return [(s["start"], s["start"] + s["length"]) for s in meta["segments"]]


def hit_rates(topk, n_expert, fractions, layers, split):
    """Per-layer expert caches over the whole trace in order. Returns {policy: {fraction: hit rate}}.
    LRU runs over all tokens; pins are learned on tokens [0, split) and measured on [split, end)."""
    out = {"lru": {}, "lru_test_half": {}, "pins_learned": {}, "pins_oracle": {}}
    for f in fractions:
        cap = max(1, int(round(f * n_expert)))
        hits = total = hits_t = total_t = 0
        p_hits = p_total = o_hits = 0
        for L in layers:
            t = topk[L]
            lru = OrderedDict()
            for i, row in enumerate(t):
                for e in row:
                    e = int(e)
                    hit = e in lru
                    if hit:
                        lru.move_to_end(e)
                    else:
                        lru[e] = True
                        if len(lru) > cap:
                            lru.popitem(last=False)
                    hits += hit
                    total += 1
                    if i >= split:
                        hits_t += hit
                        total_t += 1
            train_counts = np.bincount(t[:split].ravel(), minlength=n_expert)
            pins = set(np.argsort(-train_counts, kind="stable")[:cap].tolist())
            test = t[split:].ravel()
            p_hits += sum(1 for e in test if int(e) in pins)
            p_total += len(test)
            test_counts = np.bincount(test, minlength=n_expert)
            o_hits += int(np.sort(test_counts)[::-1][:cap].sum())
        out["lru"][f] = hits / total
        out["lru_test_half"][f] = hits_t / total_t
        out["pins_learned"][f] = p_hits / p_total
        out["pins_oracle"][f] = o_hits / p_total
    return out


def hybrid_rates(topk, n_expert, fractions, layers, split, pin_shares=(0.1, 0.3, 0.5)):
    """Pins plus LRU, as Phase 4 runs them: a share of each layer's cache pinned to the experts used most
    in the first half, the rest LRU; hit rate on the second half. {share: {fraction: hit rate}}."""
    out = {}
    for share in pin_shares:
        out[share] = {}
        for f in fractions:
            cap = max(1, int(round(f * n_expert)))
            n_pin = int(round(share * cap))
            hits_t = total_t = 0
            for L in layers:
                t = topk[L]
                train_counts = np.bincount(t[:split].ravel(), minlength=n_expert)
                pins = set(np.argsort(-train_counts, kind="stable")[:n_pin].tolist())
                lru = OrderedDict()
                cap_lru = cap - n_pin
                for i, row in enumerate(t):
                    for e in row:
                        e = int(e)
                        if e in pins:
                            hit = True
                        else:
                            hit = e in lru
                            if hit:
                                lru.move_to_end(e)
                            elif cap_lru > 0:
                                lru[e] = True
                                if len(lru) > cap_lru:
                                    lru.popitem(last=False)
                        if i >= split:
                            hits_t += hit
                            total_t += 1
            out[share][f] = hits_t / total_t
    return out


def prompt_pin_rates(topk, n_expert, fractions, layers, segs, pin_shares=(0.1, 0.3, 0.5)):
    """Prompt pins (Phase 4 item 3): each conversation's first half plays the prompt and its second half
    the reply; during the reply, a share of the cache keeps the experts the prompt used most, the rest is
    LRU. Hit rate over the replies, against LRU alone over the same tokens. {share: {fraction: rate}}."""
    out = {0.0: {}}
    for share in pin_shares:
        out[share] = {}
    for f in fractions:
        cap = max(1, int(round(f * n_expert)))
        for share in out:
            n_pin = int(round(share * cap))
            hits = total = 0
            for L in layers:
                t = topk[L]
                lru = OrderedDict()
                for a, b in segs:
                    m = a + (b - a) // 2
                    counts = np.bincount(t[a:m].ravel(), minlength=n_expert)
                    pins = set(np.argsort(-counts, kind="stable")[:n_pin].tolist()) if n_pin > 0 else set()
                    for i in range(a, b):
                        reply = i >= m
                        cap_lru = cap - (n_pin if reply else 0)
                        for e in t[i]:
                            e = int(e)
                            if reply and e in pins:
                                hit = True
                            else:
                                hit = e in lru
                                if hit:
                                    lru.move_to_end(e)
                                elif cap_lru > 0:
                                    lru[e] = True
                            while len(lru) > max(cap_lru, 0):
                                lru.popitem(last=False)
                            if reply:
                                hits += hit
                                total += 1
            out[share][f] = hits / total
    return out


def union_growth(topk, layers, segs, kmax=8):
    """Mean distinct experts per layer over k consecutive tokens (windows inside one conversation)."""
    res = {}
    for k in range(1, kmax + 1):
        vals = []
        for L in layers:
            t = topk[L]
            for a, b in segs:
                for i in range(a, b - k + 1, max(1, k)):
                    vals.append(len(np.unique(t[i:i + k])))
        res[k] = float(np.mean(vals))
    return res


def touched_by_prompt(topk, n_expert, layers, segs, lengths):
    """Share of a layer's experts that a prompt of P tokens uses at least once."""
    res = {}
    for P in lengths:
        vals = []
        for L in layers:
            t = topk[L]
            for a, b in segs:
                for i in range(a, b - P + 1, P):
                    vals.append(len(np.unique(t[i:i + P])) / n_expert)
        if vals:
            res[P] = float(np.mean(vals))
    return res


def lookahead(meta, topk, hidden, info):
    """Predict layer L2's experts from the residual stream after the previous MoE layer L1, with L2's
    own norm and router (sigmoid + selection bias, as Nemotron-H routes). Recall = share of the
    actually chosen experts that are in the predicted set (k and 2k candidates)."""
    layers = meta["moe_layers"]
    n_used = meta["n_expert_used"]
    every = meta["hidden_every"]
    res = []
    for L1, L2 in zip(layers, layers[1:]):
        if L1 not in hidden:
            continue
        h = hidden[L1]
        idx = np.arange(h.shape[0]) * every
        idx = idx[idx < topk[L2].shape[0]]
        h = h[: len(idx)]
        norm_w = info["tensor"](f"blk.{L2}.attn_norm.weight")
        gate = info["tensor"](f"blk.{L2}.ffn_gate_inp.weight")       # [n_expert, n_embd]
        bias = info["tensor"](f"blk.{L2}.exp_probs_b.bias")
        if norm_w is None or gate is None:
            continue
        x = h / np.sqrt(np.mean(h.astype(np.float64) ** 2, axis=1, keepdims=True) + info["eps"]) * norm_w
        logits = x @ gate.T
        sel = 1.0 / (1.0 + np.exp(-logits)) + (bias if bias is not None else 0.0)
        order = np.argsort(-sel, axis=1)
        actual = topk[L2][idx]
        rec_k = np.mean([len(set(order[i, :n_used]) & set(actual[i])) / n_used for i in range(len(idx))])
        rec_2k = np.mean([len(set(order[i, :2 * n_used]) & set(actual[i])) / n_used for i in range(len(idx))])
        # baselines: the layer's most used experts, and the previous token's experts at L2
        freq = np.argsort(-np.bincount(topk[L2].ravel(), minlength=info["n_expert"]))[:n_used]
        base_freq = np.mean([len(set(freq) & set(actual[i])) / n_used for i in range(len(idx))])
        prev = [len(set(topk[L2][j - 1]) & set(topk[L2][j])) / n_used for j in idx if j > 0]
        res.append({"from": L1, "to": L2, "recall_k": float(rec_k), "recall_2k": float(rec_2k),
                    "baseline_most_used": float(base_freq), "baseline_previous_token": float(np.mean(prev)),
                    "samples": int(len(idx))})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("model")
    ap.add_argument("--json")
    args = ap.parse_args()
    meta, topk, hidden = load(args.trace_dir)
    info = gguf_info(args.model)
    layers = meta["moe_layers"]
    n_tok = topk[layers[0]].shape[0]
    segs = segments_of(meta)
    n_expert, n_used = info["n_expert"], meta["n_expert_used"]
    split = segs[len(segs) // 2][0] if len(segs) > 1 else n_tok // 2
    print(f"{meta['model']}: {n_tok} tokens, {len(segs)} conversations, {len(layers)} MoE layers, "
          f"{n_used} of {n_expert} experts per token")

    fractions = [0.05, 0.10, 0.25, 0.50, 0.75]
    hr = hit_rates(topk, n_expert, fractions, layers, split)
    print("\nHit rate against cache size (share of each layer's experts kept in memory):")
    print("  cache   uniform guess   LRU (all)   LRU (2nd half)   pins learned on 1st half   best possible pins")
    for f in fractions:
        print(f"  {f:5.0%}   {f:13.0%}   {hr['lru'][f]:9.1%}   {hr['lru_test_half'][f]:14.1%}   "
              f"{hr['pins_learned'][f]:24.1%}   {hr['pins_oracle'][f]:18.1%}")

    hy = hybrid_rates(topk, n_expert, fractions, layers, split)
    print("\nPins plus LRU (Phase 4 as it would run: part of each cache pinned, the rest LRU; 2nd half):")
    print("  cache   LRU alone   " + "   ".join(f"{s:.0%} pinned" for s in hy))
    for f in fractions:
        print(f"  {f:5.0%}   {hr['lru_test_half'][f]:9.1%}   " + "   ".join(f"{hy[s][f]:10.1%}" for s in hy))

    pp = prompt_pin_rates(topk, n_expert, fractions, layers, segs)
    print("\nPrompt pins (each conversation's 1st half is the prompt; hit rate over the 2nd half, the reply):")
    print("  cache   LRU alone   " + "   ".join(f"{s:.0%} pinned" for s in pp if s > 0))
    for f in fractions:
        print(f"  {f:5.0%}   {pp[0.0][f]:9.1%}   " + "   ".join(f"{pp[s][f]:10.1%}" for s in pp if s > 0))

    ug = union_growth(topk, layers, segs)
    print("\nUnion growth (distinct experts per layer for k consecutive words):")
    for k, v in ug.items():
        extra = v - ug[k - 1] if k > 1 else v
        print(f"  k={k}: {v:6.2f} experts ({v / n_used:4.2f}x one word); this word adds {extra:5.2f} ({extra / n_used:4.0%} of a word's set)")

    tp = touched_by_prompt(topk, n_expert, layers, segs, [16, 64, 256, 1024, 4096])
    print("\nShare of each layer's experts a prompt uses:")
    for P, v in tp.items():
        print(f"  {P:5d} tokens: {v:5.1%}")

    la = lookahead(meta, topk, hidden, info) if hidden else []
    if la:
        print("\nNext-layer lookahead (predict the next MoE layer's experts one layer early):")
        rk = np.mean([x["recall_k"] for x in la])
        r2k = np.mean([x["recall_2k"] for x in la])
        bf = np.mean([x["baseline_most_used"] for x in la])
        bp = np.mean([x["baseline_previous_token"] for x in la])
        print(f"  next layer's router, {n_used} guesses: {rk:.1%} of the chosen experts found; {2 * n_used} guesses: {r2k:.1%}")
        print(f"  baselines: the layer's most used experts {bf:.1%}; the previous word's experts {bp:.1%}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({
            "model": meta["model"], "tokens": n_tok, "conversations": len(segs), "moe_layers": len(layers),
            "n_expert": n_expert, "n_expert_used": n_used,
            "hit_rate": {p: {str(f): round(v, 4) for f, v in d.items()} for p, d in hr.items()},
            "pins_plus_lru": {str(sh): {str(f): round(v, 4) for f, v in d.items()} for sh, d in hy.items()},
            "prompt_pins": {str(sh): {str(f): round(v, 4) for f, v in d.items()} for sh, d in pp.items()},
            "union_growth": {str(k): round(v, 3) for k, v in ug.items()},
            "prompt_touches": {str(P): round(v, 4) for P, v in tp.items()},
            "lookahead": la,
        }, indent=1) + "\n")


if __name__ == "__main__":
    main()
