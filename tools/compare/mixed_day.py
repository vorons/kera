"""A working day of messages on one model, on Vulkan, on ROCm, and with Kera choosing per message (kera serve
--backend auto, docs/DECISIONS.md D-024's gate).

  python tools/compare/mixed_day.py --model MODEL.gguf --kera PATH/kera [--engines vulkan,hip,auto] [--ctx 131072]
                                    [--doc 64000] [--chats 5] [--follow-ups 3] [--max-tokens 300]
                                    [--chars-per-token 3.0] [--out mixed.json] [--quick]

One person, in order, the same on every engine: --chats short chats (two messages each), then a long document (--doc
tokens of llama.cpp's own sources) with a question and --follow-ups more questions about it, then --chats more short
chats. Each message: the wait for its first word, its writing speed, how long it took in all, and which back-end
answered it (kera serve says so in the X-Kera-Backend header). The day's total time per engine, and by kind of
message, is what the gate compares: choosing per message should take less time in all than either back-end alone, and
be within 10% of the faster one for each kind of message.

Answers are greedy with thinking off. Only the Python standard library is used, so it runs on the machine being
measured as it is. --quick: small sizes, to check the tool on the tiny models (with KERA_AUTO_BACKENDS=cpu,cpu).
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import compare  # noqa: E402  (the servers and documents of the one-person comparison)

LABEL = {"vulkan": "Kera on Vulkan", "hip": "Kera on ROCm", "auto": "Kera choosing per message"}
CHATS = [
    "Explain how a refrigerator keeps food cold.", "Describe the history of the printing press.",
    "How do vaccines teach the body to fight a disease?", "Write a short story about a lighthouse keeper.",
    "What makes a good night's sleep, and why does it matter?", "Explain how bridges carry heavy loads.",
    "Describe a day in the life of a honeybee.", "How does compound interest work? Give an example.",
    "Write a letter from a sailor to their family.", "Explain why the sky is blue and sunsets are red.",
]
FOLLOW_UPS = [
    "Which function in these files loads a model, and what does it check first?",
    "List three places where these files allocate memory, with the function names.",
    "Which of these files handles the vocabulary, and how does it split text into tokens?",
    "Summarise in five bullet points what these files do.",
]


def ask(port, messages, max_tokens):
    """One streamed, greedy answer: its text, the wait for its first word, its writing speed, the whole time, and the
    back-end that answered"""
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0, "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, stamps, parts, timings = None, [], [], {}
    with urllib.request.urlopen(req, timeout=compare.STALL_S) as r:
        backend = r.headers.get("X-Kera-Backend")
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            timings = j.get("timings") or timings
            for c in j.get("choices", []):
                d = c.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    now = time.time()
                    first = now if first is None else first
                    stamps.append(now)
                    parts.append(piece)
    end = time.time()
    words = len(stamps)
    return {"text": "".join(parts), "first_word_s": (first or end) - t0, "total_s": end - t0, "backend": backend,
            "words_per_s": (words - 1) / (stamps[-1] - stamps[0]) if words > 2 and stamps[-1] > stamps[0] else 0,
            "read": timings.get("prompt_n", 0), "cached": timings.get("cache_n", 0), "written": timings.get("predicted_n", 0)}


def day(port, args):
    out = []

    def chat(i, kind):
        m = [{"role": "user", "content": CHATS[i % len(CHATS)]}]
        for turn in range(2):
            r = ask(port, m, args.max_tokens)
            out.append({"kind": kind, **{k: v for k, v in r.items() if k != "text"}})
            print(f"    {kind}: {r['total_s']:.1f} s on {r['backend']}", flush=True)
            m += [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": "Tell me more, briefly."}]

    for i in range(args.chats):
        chat(i, "chat")
    doc = compare.long_document(int(args.doc * args.chars_per_token), "mixed-day")
    m = [{"role": "user", "content": doc + "\n\nWhat is this project, in two sentences?"}]
    r = ask(port, m, args.max_tokens)
    out.append({"kind": "document", **{k: v for k, v in r.items() if k != "text"}})
    print(f"    document: {r['total_s']:.1f} s on {r['backend']}", flush=True)
    m.append({"role": "assistant", "content": r["text"]})
    for q in FOLLOW_UPS[:args.follow_ups]:
        m.append({"role": "user", "content": q})
        r = ask(port, m, args.max_tokens)
        out.append({"kind": "follow-up", **{k: v for k, v in r.items() if k != "text"}})
        print(f"    follow-up: {r['total_s']:.1f} s on {r['backend']}", flush=True)
        m.append({"role": "assistant", "content": r["text"]})
    for i in range(args.chats):
        chat(args.chats + i, "chat after")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kera", required=True)
    ap.add_argument("--llama-server", help="not used (scripts/compare-run.ps1 passes it to every tool)")
    ap.add_argument("--engines", default="vulkan,hip,auto")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--doc", type=int, default=64000, help="the document's length in tokens")
    ap.add_argument("--chats", type=int, default=5, help="short chats before the document, and as many after")
    ap.add_argument("--follow-ups", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--chars-per-token", type=float, default=3.0)
    ap.add_argument("--extra", default="", help="more options for kera serve")
    ap.add_argument("--port", type=int, default=8601)
    ap.add_argument("--out", default="mixed.json")
    ap.add_argument("--quick", action="store_true", help="small sizes, to check the tool on the tiny models")
    args = ap.parse_args()
    if args.quick:
        args.ctx, args.doc, args.chats, args.follow_ups, args.max_tokens = 1024, 100, 2, 2, 16
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e in engines:
        if e not in LABEL:
            ap.error(f"--engines takes {', '.join(LABEL)}")
    compare.STALL_S = 3600

    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-mixed-"))
    env = dict(os.environ, KERA_HOME=str(work / "kera-home"))
    subprocess.run([args.kera, "setup", "--quick"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    common = ["-m", args.model, "-c", str(args.ctx), "-ngl", args.ngl] + args.extra.split()
    out = {"model": pathlib.Path(args.model).name, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "settings": {k: v for k, v in vars(args).items()}}
    for i, e in enumerate(engines):
        print(f"{LABEL[e]}:", flush=True)
        # every engine starts with no saved chats, and auto measures the back-ends first (kept in the same Kera folder)
        srv = compare.server(LABEL[e], [args.kera, "serve", "--backend", e] + common, args.port + i, work / f"{e}.log", env)
        res = {"load_seconds": srv.start()}
        try:
            res["messages"] = day(args.port + i, args)
        finally:
            srv.stop()
        out[e] = res
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    kinds = ["chat", "document", "follow-up", "chat after"]
    print("\n| Seconds in all | " + " | ".join(LABEL[e] for e in engines) + " |\n| --- |" + " --- |" * len(engines))
    for k in kinds + ["the whole day"]:
        cells = []
        for e in engines:
            ms = [m for m in out[e]["messages"] if k == "the whole day" or m["kind"] == k]
            cells.append(f"{sum(m['total_s'] for m in ms):.1f}")
        print(f"| {k} | " + " | ".join(cells) + " |")
    if "auto" in engines:
        went = {}
        for m in out["auto"]["messages"]:
            went.setdefault(m["kind"], []).append(m["backend"] or "?")
        print("\nWhere Kera sent each message: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in went.items()))
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nsaved {args.out} (server logs in {work})")


if __name__ == "__main__":
    main()
