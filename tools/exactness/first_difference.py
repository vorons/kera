"""Where do two runs of a model part ways? Compares the tensors `kera exactness ... --dump-all DIR` saved from two
runs (say the CPU and a graphics chip) and lists the first operations whose results differ by more than --rel,
in the order the graph computed them, with the operation and shape (docs/NEGATIVE_RESULTS.md N-010).

  python tools/exactness/first_difference.py CPU_DIR GPU_DIR [--rel 0.01] [--show 15] [--kv 256:52]

When both runs computed the same operations in the same order, tensors are matched by their place in it; set
GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 for Vulkan, which otherwise reorders the graph. Otherwise each is matched by its name
and how many times that name came before it (ambiguous where names repeat). --kv SIZE:USED compares only the first
USED entries of any dimension of SIZE: the memory of the text has SIZE slots but holds USED words, and the slots
never filled hold whatever was there. The difference is relative: the size of (b - a) over the size of a. A
difference that starts at one operation and grows after it points at that operation; one that is there from the
first layer and stays the same size is rounding.
"""
import argparse
import math
import pathlib
import struct


def read_index(d):
    rows, seen = [], {}
    for line in (pathlib.Path(d) / "index.tsv").read_text(encoding="utf-8").splitlines():
        n, name, op, *ne = line.split("\t")
        k = seen.get(name, 0)
        seen[name] = k + 1
        rows.append({"file": pathlib.Path(d) / f"{int(n):05d}.f32", "key": (name, k), "name": name, "op": op,
                     "ne": [int(x) for x in ne], "shape": "x".join(x for x in ne if x != "1") or "1"})
    return rows


def used_part(v, ne, kv):
    """Only the first USED entries along every dimension of SIZE (kv = (SIZE, USED)), else everything"""
    if not kv or kv[0] not in ne:
        return v
    keep = [kv[1] if n == kv[0] else n for n in ne]
    out = []
    for i3 in range(keep[3]):
        for i2 in range(keep[2]):
            for i1 in range(keep[1]):
                base = ((i3 * ne[2] + i2) * ne[1] + i1) * ne[0]
                out.extend(v[base:base + keep[0]])
    return out


def floats(path):
    b = path.read_bytes()
    return struct.unpack(f"<{len(b) // 4}f", b)


def compare(a, b):
    """(relative difference, largest absolute difference, values that are NaN or infinite in b but not in a)"""
    if len(a) != len(b):
        return math.inf, math.inf, 0
    num = den = big = 0.0
    bad = 0
    for x, y in zip(a, b):
        if not math.isfinite(y) and math.isfinite(x):
            bad += 1
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        num += (y - x) ** 2
        den += x * x
        big = max(big, abs(y - x))
    return math.sqrt(num) / (math.sqrt(den) + 1e-30), big, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="the run to trust (e.g. the CPU)")
    ap.add_argument("b", help="the run to check")
    ap.add_argument("--rel", type=float, default=0.01, help="relative difference that counts as parting ways")
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--kv", help="SIZE:USED, e.g. 256:52")
    args = ap.parse_args()
    kv = tuple(int(x) for x in args.kv.split(":")) if args.kv else None

    a_rows = read_index(args.a)
    b_list = read_index(args.b)
    same_order = [(r["name"], r["op"]) for r in a_rows] == [(r["name"], r["op"]) for r in b_list]
    print("matching by place in the graph" if same_order else "matching by name (the two runs computed in other orders)")
    b_rows = {r["key"]: r for r in b_list}
    shown, worst = 0, []
    for i, r in enumerate(a_rows):
        o = b_list[i] if same_order else b_rows.get(r["key"])
        if o is None:
            continue
        rel, big, bad = compare(used_part(floats(r["file"]), r["ne"], kv), used_part(floats(o["file"]), o["ne"], kv))
        worst.append((rel, r, big, bad))
        if (rel > args.rel or bad) and shown < args.show:
            print(f"{r['name']:32s} {r['op']:16s} {r['shape']:>18s}  relative {rel:9.2e}  largest {big:9.2e}"
                  + (f"  {bad} not finite" if bad else ""))
            shown += 1
    if shown == 0:
        print(f"no operation differs by more than {args.rel:g} (relative) in {len(worst)} matched tensors")
    print("\nlargest relative differences:")
    for rel, r, big, bad in sorted(worst, key=lambda w: -w[0])[:8]:
        print(f"  {r['name']:32s} {r['op']:16s} {r['shape']:>18s}  relative {rel:9.2e}")


if __name__ == "__main__":
    main()
