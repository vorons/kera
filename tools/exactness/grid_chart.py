"""A picture of many `kera exactness` runs: one row per tiny model, one column per back-end and setup, each cell saying
whether the run matched the reference (and how: exactly, at a near tie inside the model, or how many of several runs).

  python tools/exactness/grid_chart.py runs.tsv --out grid.png [--title "..."] [--note "..."]

runs.tsv has one run per line, tab-separated: back-end, setup, model, the run's PASS/FAIL line, and optionally the
"inside the model" line. Setups named like "streamed-3" are repeats of "streamed" and are counted together. Uses
tools/compare/chart.py's colours. Needs matplotlib.
"""
import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compare"))
from chart import GOLD, INK, LIGHT, MUTED, RULE, plt  # noqa: E402

BAD = "#F2555A"
BACKEND = {"cpu": "CPU", "vulkan": "Vulkan", "hip": "ROCm", "vulkan-software": "Vulkan (software)"}
SETUP = {"memory": "in memory", "streamed": "streamed", "streamed-kera": "streamed", "streamed-os": "streamed (OS cache)",
         "word-by-word-evicting": "word by word, evicting", "streamed-check": "streamed, checked"}
MODEL = {"qwen3_dense": "Qwen3 style (dense)", "deepseek_v3_mla": "DeepSeek-V3 style (MLA)",
         "glm_moe_dsa": "GLM-5 style (sparse attention)", "glm_moe_dsa_mtp": "GLM-5 style with MTP",
         "nemotron_h_moe": "Nemotron-H (hybrid)", "nemotron_h_latent_moe": "Nemotron LatentMoE"}


def verdict(line, inner):
    if line.startswith("PASS"):
        ties = lambda n: f"{n} near tie" + ("s" if n != "1" else "")
        m = re.search(r"taken at (\d+) near tie", line + " " + inner)  # inside the model: the CPU's choice was taken
        if m:
            return ("pass", f"exact after\n{ties(m.group(1))}")
        m = re.search(r"steps exact, (\d+) near-ties", line)  # in the answer: either word was allowed
        if m and m.group(1) != "0":
            return ("pass", f"exact but for\n{ties(m.group(1))}")
        return ("pass", "exact")
    if line.startswith("FAIL"):
        m = re.search(r"(\d+)/(\d+) steps exact", line)
        return ("fail", f"wrong\n({m.group(1)}/{m.group(2)} words)" if m else "wrong")
    if "no routed experts" in line or "doesn't stream" in line:
        return ("na", "no experts\nto stream")
    if "llama_decode failed" in line:
        return ("fail", "stopped\n(error)")
    return ("na", "-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("--out", default="grid.png")
    ap.add_argument("--title", default="Does Kera give the reference's answers?")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    cells = collections.OrderedDict()  # (backend, setup) -> model -> [verdicts]
    models = []
    for raw in open(args.runs, encoding="utf-8-sig"):
        f = raw.rstrip("\r\n").split("\t")
        if len(f) < 4:
            continue
        backend, setup, model, line = f[0], re.sub(r"-\d+$", "", f[1]), f[2], f[3]
        inner = f[4] if len(f) > 4 else ""
        cells.setdefault((backend, setup), {}).setdefault(model, []).append(verdict(line, inner))
        if model not in models:
            models.append(model)
    cols = list(cells)

    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "text.color": LIGHT, "font.size": 10})
    w = max(2.2 + 1.45 * len(cols), 10.5)  # room for the title and note when there are few columns
    h = 2.3 + 0.62 * len(models)
    fig = plt.figure(figsize=(w, h), facecolor=INK)
    fig.text(0.3 / w, 1 - 0.45 / h, args.title, fontsize=17, fontweight="bold", color=LIGHT, va="top")
    fig.text(0.3 / w, 1 - 0.85 / h, args.note, fontsize=9.5, color=MUTED, va="top", linespacing=1.45)
    ax = fig.add_axes([2.2 / w, 0.25 / h, 1.45 * len(cols) / w, (h - 2.05) / h])
    ax.set_facecolor(INK)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(models), -1.1)
    ax.axis("off")
    for j, (backend, setup) in enumerate(cols):
        ax.text(j + 0.5, -0.75, BACKEND.get(backend, backend), ha="center", va="center", fontsize=10.5, color=LIGHT,
                fontweight="bold")
        ax.text(j + 0.5, -0.35, SETUP.get(setup, setup), ha="center", va="center", fontsize=9, color=MUTED)
    for i, m in enumerate(models):
        ax.text(-0.08, i + 0.5, MODEL.get(m, m), ha="right", va="center", fontsize=10, color=LIGHT)
        for j, col in enumerate(cols):
            vs = cells[col].get(m, [])
            if not vs:
                continue
            kinds = [k for k, _ in vs]
            if len(vs) > 1:  # repeats: how many matched
                good = kinds.count("pass")
                kind = "pass" if good == len(vs) else "fail"
                text = f"{good} of {len(vs)}\nruns exact"
            else:
                kind, text = vs[0]
            face = {"pass": "#1A2436", "fail": "#3A1E26", "na": INK}[kind]
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, facecolor=face, edgecolor=RULE, linewidth=0.8))
            color = {"pass": GOLD, "fail": BAD, "na": MUTED}[kind]
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=9, color=color,
                    fontweight="bold" if kind != "na" else "normal", linespacing=1.2)
    fig.savefig(args.out, dpi=150, facecolor=INK)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
