"""A picture of a tools/compare/compare.py result: Kera against one other engine, in Kera's colours.

  python tools/compare/chart.py compare.json --out chart.png [--against ollama] [--title "..."] [--note "..."]
                                [--kera kera_rocm] [--kera-label "Kera on ROCm"]
                                [--also kera_vulkan --also-label "Kera on Vulkan"] [--against-from other.json]

The usual test: words written per second for each prompt, the wait before each reply in the follow-up chat, and the
first reply after a restart. The long-context test (--long): for each document length, the wait for the first word,
the writing speed and the wait for a follow-up. --kera picks which Kera run in the file to draw ("kera" unless
several were merged into one file); --also draws one more from the same file, a dashed line (the long test only:
e.g. Kera on another back-end). --against-from takes the other engine's results from another compare.py result of
the same test (its settings must match): e.g. a newer Kera against Ollama measured earlier.
Needs matplotlib (not used anywhere else in Kera).
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, GOLD, LIGHT, GREY, MUTED, RULE = "#0D1524", "#FFC21A", "#F2F4F7", "#7C8799", "#B8C0CC", "#2A3447"  # BRAND.md
LABEL = {"stock": "llama.cpp", "kera": "Kera", "ollama": "Ollama"}
PROMPT = {"code": "Code", "long_context": "Long document\n(4,000 words)", "short_chat": "Short chat",
          "tool_call": "Tool call", "code_rewrite": "Code rewrite"}


def style(ax, title, grid_axis="x"):
    ax.set_facecolor(INK)
    ax.set_title(title, loc="left", fontsize=12.5, color=LIGHT, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis=grid_axis, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def draw_usual(fig, o, k, name, k_label="Kera"):
    grid = fig.add_gridspec(3, 1, height_ratios=[3.2, 2.2, 1.1], hspace=0.66, left=0.2, right=0.93, top=0.835, bottom=0.1)

    # 1. writing speed
    ax = fig.add_subplot(grid[0])
    ids = [p for p in PROMPT if p in k["writing"]]
    y = range(len(ids))
    ov = [o["writing"][p]["words_per_s"] for p in ids]
    kv = [k["writing"][p]["words_per_s"] for p in ids]
    h = 0.36
    ax.barh([i - h / 2 for i in y], ov, h, color=GREY, label=name)
    ax.barh([i + h / 2 for i in y], kv, h, color=GOLD, label=k_label)
    top = max(ov + kv)
    for i, (a, b) in enumerate(zip(ov, kv)):
        ax.text(a + top * 0.01, i - h / 2, f"{a:.0f}", va="center", fontsize=10, color=MUTED)
        ax.text(b + top * 0.01, i + h / 2, f"{b:.0f}  ({b / a:.2f}×)" if a else f"{b:.0f}", va="center", fontsize=10,
                color=GOLD, fontweight="bold")
    ax.set_yticks(list(y), [PROMPT[p] for p in ids])
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.28)
    ax.set_xlabel("words per second")
    style(ax, "Writing speed (higher is better)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, labelcolor=LIGHT, fontsize=10)

    # 2. follow-up waits
    ax = fig.add_subplot(grid[1])
    ow, kw = o["follow_up_waits_s"], k["follow_up_waits_s"]
    n = range(1, len(kw) + 1)
    ax.plot(n, ow, color=GREY, linewidth=2.2, marker="o", markersize=4, label=name)
    ax.plot(n, kw, color=GOLD, linewidth=2.6, marker="o", markersize=4, label=k_label)
    ax.set_xticks(list(n))
    ax.tick_params(axis="y", colors=MUTED)
    ax.set_xlabel("message in the chat")
    ax.set_ylabel("seconds")
    ax.set_ylim(0, max(ow + kw) * 1.15)
    style(ax, "Wait before each reply as a chat grows (lower is better)", "y")
    ax.legend(loc="best", frameon=False, labelcolor=LIGHT, fontsize=10)  # wherever the lines leave room

    # 3. restart
    ax = fig.add_subplot(grid[2])
    orr, kr = o["after_restart"], k["after_restart"]
    vals = [orr["first_word_s"], kr["first_word_s"]]
    ax.barh([0, 1], vals, 0.55, color=[GREY, GOLD])
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.1f} s", va="center", fontsize=10, color=[MUTED, GOLD][i],
                fontweight="bold" if i else "normal")
    ax.set_yticks([0, 1], [name, k_label])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("seconds to the first word")
    style(ax, "First reply after a restart (lower is better)")
    fig.text(0.04, 0.02, f"After the restart {k_label} read {kr['prompt_read']:,} words and reused {kr['prompt_cached']:,} "
             f"saved on the SSD; {name} read {orr['prompt_read']:,}.", fontsize=9.5, color=MUTED)


def draw_long(fig, o, k, name, k_label="Kera", extra=None, extra_label=""):
    rows = list(zip(o["long"], k["long"]))
    size = [max(a["prompt_tokens"], b["prompt_tokens"]) for a, b in rows]
    extra_rows = extra["long"][:len(rows)] if extra else []
    x = list(range(len(rows)))
    grid = fig.add_gridspec(3, 1, hspace=0.6, left=0.13, right=0.95, top=0.815, bottom=0.07)
    panels = [("first_word_s", "Wait for the first word: reading the document (lower is better)", "seconds"),
              ("words_per_s", "Writing speed with that much context (higher is better)", "words per second"),
              ("follow_up_s", "Wait for a follow-up question in that chat (lower is better)", "seconds")]
    fmt = lambda v: f"{v:.2f}" if v < 1 else (f"{v:.1f}" if v < 10 else f"{v:.0f}")
    for i, (key, title, unit) in enumerate(panels):
        ax = fig.add_subplot(grid[i])
        ov, kv = [a[key] for a, _ in rows], [b[key] for _, b in rows]
        ev = [r[key] for r in extra_rows]
        top = max(ov + kv + ev) * 1.25
        ax.plot(x, ov, color=GREY, linewidth=2.2, marker="o", markersize=5, label=name)
        ax.plot(x, kv, color=GOLD, linewidth=2.6, marker="o", markersize=5, label=k_label)
        if ev:
            ax.plot(x[:len(ev)], ev, color=LIGHT, linewidth=1.6, linestyle=(0, (4, 3)), marker="o", markersize=3.5,
                    label=extra_label)
        # at each length: the highest value's label above its point, the lowest one's below, and a third one's to
        # the right, on a patch of background so the lines don't run through it
        looks = {"other": (MUTED, "normal", 9.5), "kera": (GOLD, "bold", 9.5), "extra": (LIGHT, "normal", 8.5)}
        for j, xi in enumerate(x):
            pts = sorted([(ov[j], "other"), (kv[j], "kera")] + ([(ev[j], "extra")] if j < len(ev) else []),
                         key=lambda p: (p[0], p[1] == "kera"))
            for n, (v, who) in enumerate(pts):
                color, weight, size_pt = looks[who]
                if n == len(pts) - 1:
                    where = dict(xytext=(0, 8), ha="center")
                elif n == 0:
                    where = dict(xytext=(0, -16), ha="center")
                else:
                    where = dict(xytext=(9, 0), ha="left", va="center",
                                 bbox=dict(boxstyle="square,pad=0.15", fc=INK, ec="none"))
                ax.annotate(fmt(v), (xi, v), textcoords="offset points", fontsize=size_pt, color=color,
                            fontweight=weight, **where)
        ax.set_xticks(x, [f"{n / 1000:.0f}K" for n in size])
        ax.set_xlim(-0.4, len(x) - 0.6)
        ax.set_ylim(-top * 0.1, top)  # room for labels under the lowest points
        ax.tick_params(axis="y", colors=MUTED)
        ax.set_ylabel(unit)
        if i == len(panels) - 1:
            ax.set_xlabel("length of the document in the chat (words)")
        style(ax, title, "y")
    fig.legend(handles=ax.get_lines(), loc="upper right", bbox_to_anchor=(0.95, 0.962), ncol=len(ax.get_lines()),
               frameon=False, labelcolor=LIGHT, fontsize=10.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result")
    ap.add_argument("--out", default="chart.png")
    ap.add_argument("--against", help="the other engine: stock or ollama (default: the one in the result)")
    ap.add_argument("--title", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--kera", default="kera", help="which Kera run in the file to draw, by its key")
    ap.add_argument("--kera-label", default="Kera", help="the name for Kera's line")
    ap.add_argument("--also", help="another Kera run in the same file, by its key (the long test only)")
    ap.add_argument("--also-label", default="", help="its name")
    ap.add_argument("--against-from", help="another compare.py result of the same test to take the other engine from")
    args = ap.parse_args()
    d = json.load(open(args.result, encoding="utf-8"))
    src = d
    if args.against_from:
        src = json.load(open(args.against_from, encoding="utf-8"))
        same = ("ngl", "ctx", "runs", "max_tokens", "turns", "turn_words")
        diff = [s for s in same if src["settings"].get(s) != d["settings"].get(s)]
        if diff:
            raise SystemExit(f"the two results were measured with different settings: {', '.join(diff)}")
    other = args.against or next(e for e in ("ollama", "stock") if e in src)
    o, k, name = src[other], d[args.kera], LABEL[other]

    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "text.color": LIGHT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": LIGHT, "axes.edgecolor": INK, "font.size": 11})
    fig = plt.figure(figsize=(8, 11.5), facecolor=INK)
    fig.text(0.04, 0.955, args.title or f"Kera against {name}", fontsize=19, fontweight="bold", color=LIGHT)
    fig.text(0.04, 0.928, args.note, fontsize=10, color=MUTED, va="top", linespacing=1.45)
    if "long" in k:
        draw_long(fig, o, k, name, args.kera_label, d[args.also] if args.also else None, args.also_label or args.also)
    else:
        draw_usual(fig, o, k, name, args.kera_label)
    fig.savefig(args.out, dpi=150, facecolor=INK)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
