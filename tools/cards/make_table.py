"""Writes src/tune/model_cards.inc from tools/cards/model_cards.json (made by tools/cards/collect.py): the chat models
whose makers' own default sampling Kera knows (D-036).

  python tools/cards/make_table.py            # rewrite src/tune/model_cards.inc
  python tools/cards/make_table.py --check    # fail if it is out of date (CI)

A model is found by its repository's name and its shape (layers, width) together. Left out: names that two
repositories share with different values, test models, values no sampler takes, and another organisation's copy of a
model under the model's own name (only the makers' own file counts, or a copy proven to hold the same bytes).
"""
import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "tools" / "cards" / "model_cards.json"
OUT = ROOT / "src" / "tune" / "model_cards.inc"


def key(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def f(v):
    return "-1.0f" if v is None else f"{float(v):g}f" if "." in f"{float(v):g}" or "e" in f"{float(v):g}" else f"{float(v):g}.0f"


def table():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    by_key = {}
    for m in data["models"]:
        org, name = m["repo"].split("/")
        s = m["sampling"]
        # values no sampler takes (a temperature of 0 with do_sample, a top_p outside (0, 1], ...): not a recommendation
        if not (0 < s.get("temp", 1) <= 5 and 0 < s.get("top_p", 1) <= 1 and 0 <= s.get("top_k", 0) <= 1000 and
                0 <= s.get("min_p", 0) < 1 and 0.5 <= s.get("repeat", 1) <= 2):
            continue
        if re.search(r"test|tiny-random|dummy", m["repo"], re.I) or m["layers"] < 4:
            continue
        # another organisation's copy of a model under the model's own name (its card names the original as its
        # base): its file isn't the makers', and may differ from theirs. Not known (None): left out too.
        base = m.get("base")
        if base is None or any(b.split("/")[-1].lower() == name.lower() and b.lower() != m["repo"].lower() for b in base):
            continue
        k = (key(name), m["layers"], m["width"])
        v = (s.get("temp"), s.get("top_p"), s.get("top_k"), s.get("min_p"), s.get("repeat"))
        by_key.setdefault(k, []).append((v, m["repo"]))
    rows = []
    for k in sorted(by_key):
        values = {v for v, _ in by_key[k]}
        if len(values) != 1:
            continue  # the same name and shape with different settings: not known which a file is
        v = values.pop()
        repo = sorted(r for _, r in by_key[k])[0]
        rows.append('    { "%s", %d, %d, %s, %s, %d, %s, %s, "%s" },' % (
            k[0], k[1], k[2], f(v[0]), f(v[1]), -1 if v[2] is None else int(v[2]), f(v[3]), f(v[4]), repo))
    head = ("// Made by tools/cards/make_table.py from tools/cards/model_cards.json (collected %s): do not edit.\n"
            "// name, layers, width, temperature, top_p, top_k, min_p, repetition penalty (< 0: the makers name none), repository\n"
            % data.get("collected", "?"))
    return head + "\n".join(rows) + "\n", len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text, n = table()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            sys.exit("src/tune/model_cards.inc is out of date: run python tools/cards/make_table.py")
        print("src/tune/model_cards.inc is up to date")
        return
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({n} models)")


if __name__ == "__main__":
    main()
