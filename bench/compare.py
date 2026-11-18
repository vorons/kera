"""Compare two sets of bench results (kera bench JSON files) and flag anything more than 3% slower.

  python bench/compare.py BEFORE_DIR AFTER_DIR [--threshold 3] [--summary FILE]

Results are matched by model hash and prompt. A model counts as slower if its prompt speed or
writing speed dropped by more than the threshold on any prompt (spec P2: the nightly bench fails,
and opens an issue, if any bench model gets more than 3% slower). Prints a Markdown table (and
appends it to --summary, e.g. $GITHUB_STEP_SUMMARY). Exits 1 if anything got slower.
"""
import argparse
import json
import pathlib
import sys


def load(folder):
    out = {}
    for f in sorted(pathlib.Path(folder).glob("*.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if j.get("schema") != "kera-bench-1":
            continue
        for r in j.get("results", []):
            out[(j["model"]["hash"], r["prompt_id"])] = (j, r)
    return out


def pct(before, after):
    return 100.0 * (after - before) / before if before else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--threshold", type=float, default=3.0, help="percent slower that counts as a slowdown")
    ap.add_argument("--summary", help="append the table to this file")
    args = ap.parse_args()

    before, after = load(args.before), load(args.after)
    rows = ["| Model | Prompt | Prompt speed (tok/s) | Writing speed (tok/s) | First word (ms) | |",
            "| --- | --- | --- | --- | --- | --- |"]
    slower = []
    for key in sorted(after):
        j, r = after[key]
        name = j["model"]["file"]
        if key not in before:
            rows.append(f"| {name} | {r['prompt_id']} | {r['prefill_tok_s']:.1f} | {r['decode_tok_s']:.2f} | "
                        f"{r['ttft_ms']:.0f} | new |")
            continue
        _, b = before[key]
        d_pre, d_dec, d_ttft = pct(b["prefill_tok_s"], r["prefill_tok_s"]), pct(b["decode_tok_s"], r["decode_tok_s"]), \
            pct(b["ttft_ms"], r["ttft_ms"])
        bad = d_pre < -args.threshold or d_dec < -args.threshold
        if bad:
            slower.append(f"{name} ({r['prompt_id']})")
        rows.append(f"| {name} | {r['prompt_id']} | {b['prefill_tok_s']:.1f} → {r['prefill_tok_s']:.1f} ({d_pre:+.1f}%) | "
                    f"{b['decode_tok_s']:.2f} → {r['decode_tok_s']:.2f} ({d_dec:+.1f}%) | "
                    f"{b['ttft_ms']:.0f} → {r['ttft_ms']:.0f} ({d_ttft:+.1f}%) | {'**slower**' if bad else 'ok'} |")
    text = "\n".join(rows) + "\n"
    if slower:
        text += f"\nMore than {args.threshold:g}% slower: " + ", ".join(slower) + "\n"
    elif not after:
        text = "No bench results to compare.\n"
    print(text)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(text)
    return 1 if slower else 0


if __name__ == "__main__":
    sys.exit(main())
