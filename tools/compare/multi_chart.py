"""A picture of a tools/compare/multi_user.py result (several people at once), in Kera's colours.

  python tools/compare/multi_chart.py multi.json [more.json ...] --out multi.png [--title "..."] [--note "..."]

Three panels, one per test in the files: each person's writing speed and all the words written per second as more
people chat at once; the others' writing while one person's long document is read, against before it came; and how
long 4 agents with one long start waited for their first word. Several files are merged (e.g. a second run with one
more engine); an engine in two files is taken from the last. Needs matplotlib (not used anywhere else in Kera).
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, GOLD, LIGHT, GREY, MUTED, RULE = "#0D1524", "#FFC21A", "#F2F4F7", "#7C8799", "#B8C0CC", "#2A3447"  # BRAND.md
ENGINES = ["stock", "ollama", "kera", "kera-mu", "kera-mu-ng"]
LABEL = {"stock": "llama.cpp", "kera": "Kera", "kera-mu": "Kera --multi-user",
         "kera-mu-ng": "Kera --multi-user, no guessing", "ollama": "Ollama"}
COLOUR = {"stock": GREY, "ollama": "#5B8DEF", "kera": "#B38612", "kera-mu": GOLD, "kera-mu-ng": "#FFE08A"}


def style(ax, title, grid_axis="y"):
    ax.set_facecolor(INK)
    ax.set_title(title, loc="left", fontsize=12.5, color=LIGHT, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis=grid_axis, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def bars(ax, groups, series, fmt, unit):
    """groups: labels on the x axis; series: [(engine, [value per group])]"""
    n = len(series)
    w = 0.8 / max(n, 1)
    top = max([v for _, vals in series for v in vals] + [1e-9])
    for i, (e, vals) in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        ax.bar(xs, vals, w * 0.92, color=COLOUR[e], label=LABEL[e])
        for x, v in zip(xs, vals):
            ax.text(x, v + top * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=8.5,
                    color=GOLD if e == "kera-mu" else MUTED, fontweight="bold" if e == "kera-mu" else "normal")
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel(unit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Several people at once")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    res = {}
    model = ""
    for f in args.files:
        j = json.load(open(f, encoding="utf-8"))
        model = model or j.get("model", "")
        for e in ENGINES:
            if e in j:
                res[e] = j[e]
    engines = [e for e in ENGINES if e in res]
    panels = [t for t in ("chats", "document", "agents") if any(t in res[e] for e in engines)]

    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "text.color": LIGHT, "axes.labelcolor": MUTED,
                         "xtick.color": LIGHT, "ytick.color": MUTED, "axes.edgecolor": INK, "font.size": 11})
    rows = {"chats": 2, "document": 1, "agents": 1}
    heights = [3.0 for p in panels for _ in range(rows[p])]
    h = 2.7 + 2.9 * len(heights)  # inches: the header (title, note, legend), then the panels
    fig = plt.figure(figsize=(9, h), facecolor=INK)
    grid = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.75, left=0.1, right=0.97,
                            top=1 - 2.1 / h, bottom=0.9 / h)
    fig.text(0.04, 1 - 0.42 / h, args.title, fontsize=19, fontweight="bold", color=LIGHT)
    fig.text(0.04, 1 - 0.78 / h, args.note or model, fontsize=10, color=MUTED, va="top", linespacing=1.45)
    r = 0
    legend = None
    for p in panels:
        have = [e for e in engines if p in res[e]]
        if p == "chats":
            people = [c["people"] for c in res[have[0]]["chats"]]
            groups = [f"{n} at once" for n in people]
            ax = fig.add_subplot(grid[r])
            bars(ax, groups, [(e, [c["words_per_s_each"] for c in res[e]["chats"]]) for e in have], "{:.0f}", "words/s")
            style(ax, "Each person's writing speed (higher is better)")
            ax = fig.add_subplot(grid[r + 1])
            bars(ax, groups, [(e, [c["words_per_s_all"] for c in res[e]["chats"]]) for e in have], "{:.0f}", "words/s")
            style(ax, "All the words written per second, together (higher is better)")
            r += 2
        elif p == "document":
            ax = fig.add_subplot(grid[r])
            groups = ["the others' writing\nbefore it came", "the others' writing\nwhile it was read"]
            bars(ax, groups, [(e, [res[e]["document"]["chat_words_per_s_before"],
                                   res[e]["document"]["chat_words_per_s_during"]]) for e in have], "{:.1f}", "words/s")
            n = res[have[0]]["document"]["document_tokens"]
            style(ax, f"One person sends a {n:,}-word document while 3 others chat (higher is better)")
            r += 1
        else:
            ax = fig.add_subplot(grid[r])
            groups = ["median wait for the first word", "longest wait"]
            waits = [(e, [res[e]["agents"]["first_word_s_median"], res[e]["agents"]["first_word_s_max"]]) for e in have]
            short = max(v for _, vals in waits for v in vals) < 10
            bars(ax, groups, waits, "{:.2f} s" if short else "{:.1f} s", "seconds")
            style(ax, "4 agents with one long start, sent at the same moment (lower is better)")
            r += 1
        if legend is None:
            legend = ax.get_legend_handles_labels()
    if legend is not None:
        fig.legend(*legend, loc="upper left", bbox_to_anchor=(0.035, 1 - 1.2 / h), ncol=min(len(engines), 5),
                   frameon=False, labelcolor=LIGHT, fontsize=9.5)
    fig.savefig(args.out, dpi=150, facecolor=INK)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
