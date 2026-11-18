"""Several people at once on one model and machine (docs/DECISIONS.md D-023, M13's gate): stock llama.cpp, Kera,
Kera with --multi-user and Ollama, each serving the same people the same requests.

  python tools/compare/multi_user.py --model MODEL.gguf --llama-server PATH/llama-server --kera PATH/kera
                                     [--engines stock,kera,kera-mu,kera-mu-ng,ollama] [--ollama NAME --ollama-url URL]
                                     [--users 1,2,4,8] [--slots 4] [--ctx 65536] [--ngl 99] [--tests chats,document,agents]
                                     [--doc 32000] [--start 8000] [--chars-per-token 3.0] [--out multi.json]

Every llama.cpp-based engine gets --slots chats at once (-np) in one memory for all of them (-kvu); Ollama gets
OLLAMA_NUM_PARALLEL where it is started (scripts/ollama-private.ps1 -Parallel). Answers are greedy, thinking off.
Three tests, each with every engine:

  chats      N people (each of --users) chatting at once, 3 messages each, back to back, answers of up to 300 words:
             the wait for each answer's first word, each answer's writing speed, and the words written per second by
             all of them together.
  document   3 people chatting while a 4th sends a long document (--doc tokens of llama.cpp's own sources) and a
             question: the others' writing speed while it is read, against before it came; the waits for their new
             messages meanwhile; and the document's wait for its first word, against the same length alone on the
             idle server.
  agents     4 people with the same long start (--start tokens: an agent's instructions and the files it works on)
             and their own question, all sent at the same moment: each one's wait for the first word, and how much of
             the start each one read.

Results: a table per test and --out (JSON). Only the Python standard library is used, so it runs on the machine being
measured as it is. kera-mu-ng is Kera --multi-user without guessing ahead (--no-auto-guess): whether guessing still
pays when several chats share the chip. --quick: small sizes, to check the tool itself on the tiny models.
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import compare  # noqa: E402  (the servers and documents of the one-person comparison)

LABEL = {"stock": "llama.cpp", "kera": "Kera", "kera-mu": "Kera --multi-user",
         "kera-mu-ng": "Kera --multi-user, no guessing", "ollama": "Ollama"}
SYSTEM = "You are a helpful assistant."
TOPICS = [
    "Explain how a refrigerator keeps food cold.", "Describe the history of the printing press.",
    "How do vaccines teach the body to fight a disease?", "Write a short story about a lighthouse keeper.",
    "What makes a good night's sleep, and why does it matter?", "Explain how bridges carry heavy loads.",
    "Describe a day in the life of a honeybee.", "How does compound interest work? Give an example.",
    "Write a letter from a sailor to their family.", "Explain why the sky is blue and sunsets are red.",
    "How do computers store numbers and text?", "Describe how to plan a vegetable garden.",
    "What causes the seasons on Earth?", "Write a short story about a lost dog finding its way home.",
    "Explain how a bicycle stays upright.", "Describe the water cycle in detail.",
]
AGENT_QUESTIONS = [
    "Which function in these files loads a model, and what does it check first?",
    "List three places where these files allocate memory, with the function names.",
    "Which of these files handles the vocabulary, and how does it split text into tokens?",
    "Find a function that copies tensors between devices and say what it does.",
]


def question(person, turn):
    return TOPICS[(person * 5 + turn) % len(TOPICS)] + " Answer in about 300 words."


class reply:
    """One streamed answer: when it was sent, when each piece of it came, its text and what the server counted"""

    def __init__(self):
        self.sent, self.first, self.pieces, self.text, self.written, self.read, self.cached = time.time(), None, [], "", 0, 0, 0

    def wait_s(self):
        return (self.first if self.first is not None else time.time()) - self.sent

    def words_per_s(self):
        n = max(self.written, len(self.pieces))
        if n < 2 or self.first is None or self.pieces[-1] <= self.first:
            return 0.0
        return (n - 1) / (self.pieces[-1] - self.first)


def stream_llama(port, messages, max_tokens):
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0, "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r, parts = reply(), []
    with urllib.request.urlopen(req, timeout=compare.STALL_S) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            t = j.get("timings")
            if t:
                r.written, r.read, r.cached = t.get("predicted_n", 0), t.get("prompt_n", 0), t.get("cache_n", 0)
            for c in j.get("choices", []):
                d = c.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    now = time.time()
                    r.first = now if r.first is None else r.first
                    r.pieces.append(now)
                    parts.append(piece)
    r.text = "".join(parts)
    return r


def stream_ollama(url, model, ctx, messages, max_tokens, think_off=True):
    body = {"model": model, "messages": messages, "stream": True, "keep_alive": "30m",
            "options": {"temperature": 0, "top_k": 1, "num_predict": max_tokens, "num_ctx": ctx}}
    if think_off:
        body["think"] = False
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r, parts = reply(), []
    try:
        resp = urllib.request.urlopen(req, timeout=compare.STALL_S)
    except urllib.error.HTTPError as e:
        if think_off and "think" in e.read().decode("utf-8", "replace"):
            return stream_ollama(url, model, ctx, messages, max_tokens, think_off=False)
        raise
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            j = json.loads(line)
            if "error" in j:
                raise RuntimeError(f"Ollama: {j['error']}")
            m = j.get("message") or {}
            piece = (m.get("content") or "") + (m.get("thinking") or "")
            if piece:
                now = time.time()
                r.first = now if r.first is None else r.first
                r.pieces.append(now)
                parts.append(piece)
            if j.get("done"):
                r.written, r.read = j.get("eval_count", 0), j.get("prompt_eval_count", 0)
    r.text = "".join(parts)
    return r


def median(xs):
    return statistics.median(xs) if xs else 0.0


def p90(xs):
    return sorted(xs)[min(len(xs) - 1, int(0.9 * len(xs)))] if xs else 0.0


def together(n, fn):
    """Runs fn(i) for i in 0..n-1 in threads that start at the same moment; the seconds it all took"""
    go = threading.Barrier(n)
    errors = []

    def one(i):
        go.wait()
        try:
            fn(i)
        except Exception as e:  # noqa: BLE001 (a failed request fails the test, below)
            errors.append(e)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return time.time() - t0


def test_chats(ask, people, turns, max_tokens):
    replies, lock = [], threading.Lock()

    def person(p):
        msgs = [{"role": "system", "content": SYSTEM}]
        for t in range(turns):
            msgs.append({"role": "user", "content": question(p, t)})
            r = ask(msgs, max_tokens)
            msgs.append({"role": "assistant", "content": r.text})
            with lock:
                replies.append(r)

    wall = together(people, person)
    waits = [r.wait_s() for r in replies]
    return {"people": people, "replies": len(replies), "seconds": wall,
            "words_per_s_all": sum(max(r.written, len(r.pieces)) for r in replies) / wall,
            "words_per_s_each": median([r.words_per_s() for r in replies]),
            "first_word_s": median(waits), "first_word_s_p90": p90(waits)}


def test_document(ask, args):
    """3 people chatting, then a 4th person's long document; the document alone first, on the idle server"""
    def doc(tag):
        text = compare.long_document(int(args.doc * args.chars_per_token), tag)
        return [{"role": "user", "content": text + "\n\n" + compare.LONG_ASK}]

    alone = ask(doc("alone"), 32)
    stop, replies, lock = threading.Event(), [], threading.Lock()

    def person(p):
        msgs, t = [{"role": "system", "content": SYSTEM}], 0
        while not stop.is_set():
            msgs.append({"role": "user", "content": question(p, t)})
            r = ask(msgs, args.max_tokens)
            msgs.append({"role": "assistant", "content": r.text})
            with lock:
                replies.append(r)
            msgs = msgs[:1] + msgs[-4:]  # the chat stays short: the document is the long prompt here
            t += 1

    threads = [threading.Thread(target=person, args=(p,)) for p in range(3)]
    t0 = time.time()
    for t in threads:
        t.start()
    time.sleep(args.warm)
    busy = ask(doc("busy"), 32)
    stop.set()
    for t in threads:
        t.join()
    a, b = busy.sent, busy.first or time.time()

    def rate(lo, hi):
        n = sum(1 for r in replies for x in r.pieces if lo <= x < hi)
        return n / max(hi - lo, 1e-9) / 3

    during = [r.wait_s() for r in replies if a <= r.sent < b]
    return {"document_tokens": busy.read or alone.read, "alone_first_word_s": alone.wait_s(),
            "busy_first_word_s": busy.wait_s(), "chat_words_per_s_before": rate(t0 + args.warm / 3, a),
            "chat_words_per_s_during": rate(a, b), "chat_first_word_s_during": median(during),
            "chat_replies_during": len(during)}


def test_agents(ask, args):
    start = compare.long_document(int(args.start * args.chars_per_token), "agents")
    out = [None] * len(AGENT_QUESTIONS)

    def agent(i):
        msgs = [{"role": "system", "content": start + "\n\nYou are a coding agent working on this project."},
                {"role": "user", "content": AGENT_QUESTIONS[i]}]
        out[i] = ask(msgs, 32)

    together(len(AGENT_QUESTIONS), agent)
    waits = [r.wait_s() for r in out]
    return {"agents": len(out), "first_word_s": waits, "first_word_s_median": median(waits),
            "first_word_s_max": max(waits), "read": [r.read for r in out], "shared": [r.cached for r in out]}


def run_engine(name, srv, ask, args, tests):
    res = {"load_seconds": srv.start()}
    try:
        if "chats" in tests:
            res["chats"] = []
            for n in args.users:
                print(f"  chats, {n} at once ...", flush=True)
                res["chats"].append(test_chats(ask, n, args.turns, args.max_tokens))
        if "document" in tests:
            print("  a long document while 3 others chat ...", flush=True)
            res["document"] = test_document(ask, args)
        if "agents" in tests:
            print(f"  {len(AGENT_QUESTIONS)} agents with one start at once ...", flush=True)
            res["agents"] = test_agents(ask, args)
    finally:
        srv.stop()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-server", help="llama.cpp's own llama-server (stock)")
    ap.add_argument("--kera", help="the kera program")
    ap.add_argument("--engines", default="stock,kera,kera-mu", help="in order: stock, kera, kera-mu, kera-mu-ng, ollama")
    ap.add_argument("--ollama", help="Ollama's name for the same model, e.g. nemotron-3-nano:30b")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11500")
    ap.add_argument("--tests", default="chats,document,agents")
    ap.add_argument("--users", default="1,2,4,8", help="for chats: how many people at once, each in turn")
    ap.add_argument("--slots", type=int, default=4, help="chats each server takes at once (llama.cpp's -np)")
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--ctx", type=int, default=65536, help="the memory for all the chats together")
    ap.add_argument("--ollama-ctx", type=int, help="Ollama's num_ctx, per chat (default: --ctx / --slots)")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--doc", type=int, default=32000, help="for document: its length in tokens")
    ap.add_argument("--start", type=int, default=8000, help="for agents: the shared start's length in tokens")
    ap.add_argument("--warm", type=float, default=20, help="for document: seconds the others chat before it comes")
    ap.add_argument("--chars-per-token", type=float, default=3.0)
    ap.add_argument("--extra", default="", help="more options for the llama.cpp-based servers")
    ap.add_argument("--port", type=int, default=8591)
    ap.add_argument("--out", default="multi.json")
    ap.add_argument("--quick", action="store_true", help="small sizes, to check the tool on the tiny models")
    args = ap.parse_args()
    if args.quick:
        args.users, args.turns, args.max_tokens, args.doc, args.start, args.warm = "1,2", 2, 24, 300, 200, 2
    args.users = [int(x) for x in str(args.users).split(",") if x]
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e in engines:
        if e not in LABEL:
            ap.error(f"--engines takes {', '.join(LABEL)}")
    for e, need in (("stock", args.llama_server), ("kera", args.kera), ("kera-mu", args.kera), ("kera-mu-ng", args.kera),
                    ("ollama", args.ollama)):
        if e in engines and not need:
            ap.error(f"--engines has {e}: give {'--llama-server' if e == 'stock' else '--ollama' if e == 'ollama' else '--kera'}")
    compare.STALL_S = 3600

    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-multi-"))
    common = ["-m", args.model, "-c", str(args.ctx), "-ngl", args.ngl, "-np", str(args.slots), "-kvu"] + args.extra.split()
    env = dict(os.environ, KERA_HOME=str(work / "kera-home"))
    if any(e.startswith("kera") for e in engines):
        subprocess.run([args.kera, "setup", "--quick"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = {"model": pathlib.Path(args.model).name, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "settings": {k: v for k, v in vars(args).items()}}
    for i, e in enumerate(engines):
        print(f"{LABEL[e]}:", flush=True)
        port = args.port + i
        if e == "stock":
            srv = compare.server("llama.cpp", [args.llama_server] + common, port, work / "stock.log")
        elif e.startswith("kera"):
            extra = {"kera": [], "kera-mu": ["--multi-user"], "kera-mu-ng": ["--multi-user", "--no-auto-guess"]}[e]
            srv = compare.server(LABEL[e], [args.kera, "serve"] + common + extra, port, work / f"{e}.log", env)
        else:
            ctx = args.ollama_ctx or args.ctx // args.slots
            srv = compare.ollama(args.ollama_url, args.ollama, ctx)
        if e == "ollama":
            ask = lambda m, n, c=srv.ctx: stream_ollama(args.ollama_url, args.ollama, c, m, n)  # noqa: E731
        else:
            ask = lambda m, n, p=port: stream_llama(p, m, n)  # noqa: E731
        out[e] = run_engine(e, srv, ask, args, tests)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    if "chats" in tests:
        print("\n| People at once | " + " | ".join(f"{LABEL[e]}: words/s each, all, first word" for e in engines) +
              " |\n| --- |" + " --- |" * len(engines))
        for k, n in enumerate(args.users):
            cells = [f"{out[e]['chats'][k]['words_per_s_each']:.1f}, {out[e]['chats'][k]['words_per_s_all']:.1f}, "
                     f"{out[e]['chats'][k]['first_word_s']:.2f} s" for e in engines]
            print(f"| {n} | " + " | ".join(cells) + " |")
    if "document" in tests:
        print("\n| A long document while 3 others chat | " + " | ".join(LABEL[e] for e in engines) +
              " |\n| --- |" + " --- |" * len(engines))
        d = {e: out[e]["document"] for e in engines}
        rows = [("the others' writing before it, words/s each", "chat_words_per_s_before", "{:.1f}"),
                ("the others' writing while it is read, words/s each", "chat_words_per_s_during", "{:.1f}"),
                ("their new messages meanwhile: wait for the first word", "chat_first_word_s_during", "{:.2f} s"),
                ("the document alone: wait for its first word", "alone_first_word_s", "{:.1f} s"),
                ("the document with the others chatting", "busy_first_word_s", "{:.1f} s")]
        for label, key, fmt in rows:
            print(f"| {label} | " + " | ".join(fmt.format(d[e][key]) for e in engines) + " |")
    if "agents" in tests:
        print(f"\n| {len(AGENT_QUESTIONS)} agents with one start at once | " + " | ".join(LABEL[e] for e in engines) +
              " |\n| --- |" + " --- |" * len(engines))
        a = {e: out[e]["agents"] for e in engines}
        print("| wait for the first word: median, longest | " +
              " | ".join(f"{a[e]['first_word_s_median']:.1f} s, {a[e]['first_word_s_max']:.1f} s" for e in engines) + " |")
        print("| tokens of the start shared, all agents together | " +
              " | ".join(f"{sum(a[e]['shared']):,}" for e in engines) + " |")
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nsaved {args.out} (server logs in {work})")


if __name__ == "__main__":
    main()
