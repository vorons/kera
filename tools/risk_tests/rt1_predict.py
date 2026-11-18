"""M1 risk test 1, last step: the planner's predicted tokens per second for each Shadow Bits version (spec P5 item 1).

Reads the results rt1_shadow_bits.py saved in --work and, for a dense model bigger than memory, predicts the speed of
each version with Appendix C's machine figures and Appendix B's choose_k (the best k from 0 to 8):

  nested (sb4, bn2: 2+2; bn4: 4+4)  the base plane stays in memory and writes the guesses; a check reads the base
                                    from memory and the correction plane from the SSD (spare memory caches part of it)
  draft (no nesting)                ordinary weights stream from the SSD (memory caches what fits next to the draft);
                                    a small model from the same family writes the guesses

Both read about the same bytes per check (base + correction = the full model); what differs is the guesser: the base
is big and slow to read for each guess, the small model is fast but may guess worse. Acceptance is per token, as the
draft's top word matching the full model's on the same text (what a greedy check accepts).

  python tools/risk_tests/rt1_predict.py --work DIR [--draft-b 0.6]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))
from controller import choose_k  # noqa: E402

# Appendix C's machines: memory a model may use, memory read speed, SSD read speed
MACHINES = {
    "Mac": dict(usable_gb=10, ram_gb_s=100, ssd_gb_s=2.0),
    "Minisforum": dict(usable_gb=50, ram_gb_s=210, ssd_gb_s=5.0),
}
# bits per weight, scales included (fp16 offset and scale per group of 64 for 2-bit and plain 4-bit; one fp16 scale
# per 128 for BitNest's 4-bit base; plain 8-bit has an fp16 offset and scale per 128). The correction planes reuse
# the base's scales (s/4 for 2+2, s/16 for 4+4), so they add only their codes.
BITS = {
    "plain4": 4.5, "plain8": 8.25, "draft4": 4.5,
    "sb4_base": 2.5, "sb4_corr": 2.0, "bn2_base": 2.5, "bn2_corr": 2.0, "bn4_base": 4.125, "bn4_corr": 4.0,
}
# (machine, dense model size in billions): models bigger than that machine's memory at the version's full precision
CASES = [("Mac", 32), ("Minisforum", 123), ("Minisforum", 70)]


def acceptance(draft, target):
    pairs = [(d, t) for dm, tm in zip(draft["argmax"], target["argmax"]) for d, t in zip(dm, tm)]
    return float(np.mean([d == t for d, t in pairs]))


def nested_tok_s(a, b, base_bits, corr_bits, m):
    """Returns (tok/s, k) or None if the base plane doesn't fit."""
    base_gb, corr_gb = b * base_bits / 8, b * corr_bits / 8
    if base_gb > m["usable_gb"]:
        return None
    cached = min(m["usable_gb"] - base_gb, corr_gb)
    ram_gb, ssd_gb = base_gb + cached, corr_gb - cached
    k, rate = choose_k(a, base_gb / m["ram_gb_s"], ram_gb / m["ram_gb_s"], lambda k: ssd_gb * 1e9, m["ssd_gb_s"])
    return rate, k


def separate_tok_s(a, b, full_bits, draft_b, m):
    """Plain weights streamed, with a separate small draft model kept in memory."""
    full_gb, draft_gb = b * full_bits / 8, draft_b * BITS["draft4"] / 8
    cached = min(m["usable_gb"] - draft_gb, full_gb)
    k, rate = choose_k(a, draft_gb / m["ram_gb_s"], cached / m["ram_gb_s"], lambda k: (full_gb - cached) * 1e9,
                       m["ssd_gb_s"])
    return rate, k


def plain_tok_s(b, full_bits, m):
    """No guessing: every word reads the whole model."""
    full_gb = b * full_bits / 8
    cached = min(m["usable_gb"], full_gb)
    return 1.0 / (cached / m["ram_gb_s"] + (full_gb - cached) / m["ssd_gb_s"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=pathlib.Path)
    ap.add_argument("--draft-b", type=float, default=0.6, help="the separate draft model's size in billions")
    ap.add_argument("--out", type=pathlib.Path, help="where to save the table as JSON (default: WORK/predict.json)")
    args = ap.parse_args()
    res = {p.stem: json.loads(p.read_text()) for p in args.work.glob("*.json")
           if p.stem not in ("summary", "predict")}
    need = ["plain4", "plain8", "sb4_base", "sb4_full", "bn2_base", "bn2_full", "bn4_base", "bn4_full", "draft4"]
    missing = [n for n in need if n not in res]
    if missing:
        sys.exit(f"not measured yet: {', '.join(missing)}")

    versions = {  # draft, full, plain counterpart (same precision), the full model's bits
        "sb4": ("sb4_base", "sb4_full", "plain4", 4.5),
        "bn2": ("bn2_base", "bn2_full", "plain4", 4.5),
        "bn4": ("bn4_base", "bn4_full", "plain8", 8.125),
        "draft (for 4-bit)": ("draft4", "plain4", "plain4", 4.5),
        "draft (for 8-bit)": ("draft4", "plain8", "plain8", 8.25),
    }
    out = {"versions": {}, "speed": []}
    print("version            acceptance   KL(full)   KL(plain)   KL ratio   full = 16-bit's top word")
    for v, (d, f, p, _) in versions.items():
        a = acceptance(res[d], res[f])
        klf, klp = res[f]["kl_mean"], res[p]["kl_mean"]
        out["versions"][v] = {"draft": d, "full": f, "plain": p, "acceptance": a, "kl_full": klf, "kl_plain": klp,
                              "kl_ratio": klf / klp, "full_top1_16bit": res[f]["top1_agree_16bit"]}
        print(f"{v:18s} {a:10.3f} {klf:10.4f} {klp:11.4f} {klf / klp:10.3f}   {res[f]['top1_agree_16bit']:.1%}")

    print(f"\npredicted tokens per second (choose_k, k up to 8; separate draft: {args.draft_b}B at 4.5 bits)")
    for machine, b in CASES:
        m = MACHINES[machine]
        row = {"machine": machine, "model_b": b, **m}
        for v, (d, f, p, bits) in versions.items():
            a = out["versions"][v]["acceptance"]
            if v.startswith("draft"):
                r = separate_tok_s(a, b, bits, args.draft_b, m)
            else:
                name = d.split("_")[0]
                r = nested_tok_s(a, b, BITS[name + "_base"], BITS[name + "_corr"], m)
            row[v] = None if r is None else {"tok_s": round(r[0], 3), "k": r[1]}
        row["plain 4-bit, no guessing"] = round(plain_tok_s(b, 4.5, m), 3)
        row["plain 8-bit, no guessing"] = round(plain_tok_s(b, 8.25, m), 3)
        out["speed"].append(row)
        cells = []
        for v in versions:
            cells.append(f"{v}: " + ("base doesn't fit" if row[v] is None else f"{row[v]['tok_s']:.2f} (k={row[v]['k']})"))
        print(f"dense {b}B on the {machine}: " + "; ".join(cells) +
              f"; no guessing 4-bit {row['plain 4-bit, no guessing']:.2f}, 8-bit {row['plain 8-bit, no guessing']:.2f}")
    (args.out or args.work / "predict.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
