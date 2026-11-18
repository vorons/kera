"""One picture of the Minisforum comparisons: Kera, stock llama.cpp and Ollama on the same model, at a glance.

  python tools/compare/overview.py --usual usual.json [--ollama ollama.json] --long long.json --out overview.png \
                                   [--label "KEY=NAME" ...] [--long-label "KEY=NAME" ...] [--title "..."] [--note "..."]

--usual: a tools/compare/compare.py result of the usual test, or several runs merged into one file under keys of their
own; --ollama: another one to take Ollama from (its settings must match); --long: a --long result, likewise. Each run
found under "kera", "kera_rocm", "kera_vulkan", "stock" or "ollama" gets a bar, named as below unless --label (for
--usual) or --long-label (for --long) says otherwise: e.g. --long-label "kera=Kera, ROCm add-on" for a file whose
"kera" ran on ROCm. Draws: writing speed (the average over the prompts), the median wait before a reply in the
follow-up chat, the first reply after a restart, and for the longest document the wait for the first word, the writing
speed and the wait for a follow-up. Uses chart.py's colours. Needs matplotlib.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart import GOLD, GREY, INK, LIGHT, MUTED, plt, style  # noqa: E402

KERA_ALT = "#C99A14"  # a darker gold for Kera's other runs
RUNS = {"kera": "Kera", "kera_rocm": "Kera, ROCm add-on", "kera_vulkan": "Kera, Vulkan", "stock": "llama.cpp",
        "ollama": "Ollama"}


def panel(ax, title, rows, unit, fmt):
    """rows: (label, value, colour); the first row is drawn at the top"""
    y = list(range(len(rows)))
    vals = [v for _, v, _ in rows]
    ax.barh(y, vals, 0.58, color=[c for _, _, c in rows])
    top = max(vals)
    for i, (_, v, c) in enumerate(rows):
        kera = c in (GOLD, KERA_ALT)
        ax.text(v + top * 0.015, i, fmt(v), va="center", fontsize=10.5, color=c if kera else MUTED,
                fontweight="bold" if kera else "normal")
    ax.set_yticks(y, [label for label, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.22)
    ax.set_xlabel(unit)
    style(ax, "")
    # the title from the page's left margin, where the labels start, so long ones fit
    left = ax.get_figure().transFigure.transform((0.04, 0))[0]
    x = ax.transAxes.inverted().transform((left, 0))[0]
    ax.text(x, 1.1, title, transform=ax.transAxes, fontsize=12.5, color=LIGHT, ha="left", va="bottom")


def runs(d, labels):
    """(key, name, colour) for each run in a result, Kera's first: the first in gold, the others darker"""
    names = dict(RUNS, **labels)
    keys = [key for key in RUNS if key in d]
    keras = [key for key in keys if key.startswith("kera")]
    colour = lambda key: (GOLD if key == keras[0] else KERA_ALT) if key in keras else GREY
    return [(key, names[key], colour(key)) for key in keras + [key for key in keys if key not in keras]]


def labels(pairs):
    out = {}
    for p in pairs:
        key, sep, name = p.partition("=")
        if not sep or key not in RUNS:
            raise SystemExit(f"--label wants KEY=NAME with KEY one of {', '.join(RUNS)}: {p}")
        out[key] = name
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usual", required=True)
    ap.add_argument("--ollama")
    ap.add_argument("--long", required=True)
    ap.add_argument("--label", action="append", default=[], help="KEY=NAME for a run in --usual")
    ap.add_argument("--long-label", action="append", default=[], help="KEY=NAME for a run in --long")
    ap.add_argument("--out", default="overview.png")
    ap.add_argument("--title", default="Kera, llama.cpp and Ollama at a glance")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    u = json.load(open(args.usual, encoding="utf-8"))
    lg = json.load(open(args.long, encoding="utf-8"))
    if args.ollama:
        o = json.load(open(args.ollama, encoding="utf-8"))
        same = ("ngl", "ctx", "runs", "max_tokens", "turns", "turn_words")
        diff = [s for s in same if u["settings"].get(s) != o["settings"].get(s)]
        if diff:
            raise SystemExit(f"the usual results were measured with different settings: {', '.join(diff)}")
        u = dict(u, ollama=o["ollama"])
    usual, long_runs = runs(u, labels(args.label)), runs(lg, labels(args.long_label))

    prompts = [p for p in u["kera"]["writing"] if all(p in u[key]["writing"] for key, _, _ in usual)]
    avg = lambda e: statistics.mean(e["writing"][p]["words_per_s"] for p in prompts)
    med_wait = lambda e: statistics.median(e["follow_up_waits_s"])
    restart = lambda e: e["after_restart"]["first_word_s"]
    longest = lambda e: e["long"][-1]
    words = max(longest(lg[key])["prompt_tokens"] for key, _, _ in long_runs)
    chat = len(u["kera"]["follow_up_waits_s"])

    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "text.color": LIGHT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": LIGHT, "axes.edgecolor": INK, "font.size": 11})
    rows_in = lambda n: 0.55 + 0.36 * n  # inches for a panel of n bars, with its title and axis
    heights = [rows_in(len(usual))] * 3 + [rows_in(len(long_runs))] * 3
    top_in, bottom_in = 1.9, 0.6
    fig = plt.figure(figsize=(8, top_in + bottom_in + sum(heights) * 1.55), facecolor=INK)
    h = fig.get_figheight()
    fig.text(0.04, 1 - 0.55 / h, args.title, fontsize=19, fontweight="bold", color=LIGHT)
    fig.text(0.04, 1 - 0.8 / h, args.note, fontsize=10, color=MUTED, va="top", linespacing=1.45)
    grid = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.95, left=0.3, right=0.93,
                            top=1 - top_in / h, bottom=bottom_in / h)

    s = lambda v: f"{v:.2f} s" if v < 1 else (f"{v:.1f} s" if v < 10 else f"{v:.0f} s")
    rate = lambda v: f"{v:.0f}"
    bars = lambda d, keys, get: [(name, get(d[key]), c) for key, name, c in keys]
    panel(fig.add_subplot(grid[0]), f"Writing speed, everyday prompts (average of {len(prompts)}; higher is better)",
          bars(u, usual, avg), "words per second", rate)
    panel(fig.add_subplot(grid[1]), f"Wait before each reply in a {chat}-message chat (median; lower is better)",
          bars(u, usual, med_wait), "seconds", s)
    panel(fig.add_subplot(grid[2]), "First reply after a restart (lower is better)", bars(u, usual, restart),
          "seconds", s)
    doc = f"{words / 1000:.0f}K-word"
    panel(fig.add_subplot(grid[3]), f"A {doc} document: wait for the first word (lower is better)",
          bars(lg, long_runs, lambda e: longest(e)["first_word_s"]), "seconds", s)
    panel(fig.add_subplot(grid[4]), f"Writing speed with that {doc} document (higher is better)",
          bars(lg, long_runs, lambda e: longest(e)["words_per_s"]), "words per second", rate)
    panel(fig.add_subplot(grid[5]), f"A follow-up question in that {doc} chat (lower is better)",
          bars(lg, long_runs, lambda e: longest(e)["follow_up_s"]), "seconds", s)
    fig.savefig(args.out, dpi=150, facecolor=INK)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
