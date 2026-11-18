"""Stock llama.cpp (or Ollama) against Kera, on the same model, machine and settings: is Kera faster, and are the
answers the same?

  python tools/compare/compare.py --model MODEL.gguf --llama-server PATH/llama-server --kera PATH/kera
                                  [--ngl 99] [--ctx 16384] [--runs 3] [--out compare.json]
                                  [--engines stock,kera] [--ollama NAME [--ollama-url URL]]
                                  [--long 8000,32000,64000,100000 --ctx 131072]

Runs each engine in turn (llama.cpp's own `llama-server`, then `kera serve` with its defaults) with the same model,
context size and graphics-chip layers, and sends both the same requests: greedy (temperature 0, top_k 1), thinking
off, answers of up to --max-tokens (256). These aren't the models' own default settings, so the speeds are for
greedy answers with thinking off.

  writing    the bench prompts (a short chat, code, a JSON tool call, a 4,000-token document) and a code rewrite
             (where copying from the chat guesses well): the words written per second and the wait for the first
             word, the median of --runs. Kera guesses ahead by itself and keeps it only where it pays.
  follow-up  a chat continued message by message, as chat apps send it (the whole history each time): the wait
             before each reply.
  restart    the server stopped and started again, then one more message in that chat: Kera keeps chats on the
             SSD, stock llama.cpp has to read the whole chat again.

No server is asked the same prompt twice. Kera's guesser (copying from the chat: llama.cpp's ngram-mod) keeps one
table for every request a server gets, so a prompt asked again on the same server lets it copy its own earlier
answer, and Kera looks faster than it is. (Before 9 Nov 2026 each prompt was asked --runs times in a row on one
server, and Kera's speeds came out too high.) So each run starts every engine afresh, sends it a short greeting to
warm up (the same for all, too short to copy from) and asks each prompt once; the follow-up chat runs on the last
run's server. Every run's numbers are kept in --out, with the words Kera guessed and how many of them it kept.

--engines picks which run and in what order: stock, kera and ollama. Ollama must already be running, with --ollama
naming its copy of the model (e.g. nemotron-3-nano:30b); it gets the same requests through its own API, with the same
context size. Starting it means loading the model and stopping it unloads it, so its restart is a model load and
then the whole chat read again (Ollama keeps no chats). Ollama can't be asked to read a prompt again, but each run
loads the model afresh, so every run's prompts are new to it.

--long runs a long-context test instead, at each size in tokens: a document of that length (llama.cpp's own source
files, real text that copying from the chat can't predict, with a first line of its own so nothing is reused from
a shorter one) and a question about it, then one follow-up in that chat. Each size gets a freshly started server,
warmed up the same way: the documents start with the same files, so an answer could repeat the start of the one
before, for Kera's guesser to copy. It measures the wait for the first word (reading the document), the speed of
writing with that much context, and the wait for the follow-up. Sizes are turned into characters with
--chars-per-token (measure it with llama.cpp's llama-tokenize for the model).

It also checks the answers are the same word for word (each engine's last run): guessing ahead never changes them,
except that checking several guessed words at once can round a near tie the other way (seen on the graphics chip and
on the CPU), as another batch size would (docs/NEGATIVE_RESULTS.md N-003). Tool calls' random ids aren't compared.
Ollama is another engine, so its answers aren't expected to match. Results: a table, plain-words verdicts and --out
(JSON). Only the Python standard library is used, so it runs on the machine being measured as it is.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "bench" / "prompts"
LABEL = {"stock": "llama.cpp", "kera": "Kera", "ollama": "Ollama"}

LONG_ASK = ("Above are source files from a C++ project. List the files, and say in one line what each one does.")
LONG_FOLLOW = "Which of these files is the longest?"

# a server that sends nothing for this long has stalled (--stall; a reply streams, but a long prompt is read first)
STALL_S = 7200

REWRITE = ("Here is a C++ source file. Rewrite it exactly as it is, but add a one-line comment above every function "
           "saying what it does. Reply with the whole file and nothing else.\n\n```cpp\n{code}\n```")

# sent to every freshly started server before what is measured, so no measured prompt pays for the first request's
# set-up; its few words are far too short for a guesser to copy anything from (copying needs a 24-word match)
WARM_UP = [{"role": "user", "content": "Say hello."}]

METHOD = ("each run on freshly started servers, warmed up with a short greeting, every prompt asked once per server; "
          "--long: a freshly started server for each size")


def warm_up(srv):
    srv.ask(WARM_UP, 8)


def guessed(t):
    """How many of the words guessed ahead were kept, from a server's timings or a run's record ("" if none)"""
    n = t.get("draft_n", 0)
    return f" ({t.get('draft_n_accepted', 0)} of {n} guessed words kept)" if n else ""


def bench_prompts():
    out = []
    for f in sorted(PROMPTS.glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        # the bench files list tools as {name, description, parameters}; the OpenAI interface wants them wrapped
        tools = [t if "type" in t else {"type": "function", "function": t} for t in j.get("tools") or []] or None
        out.append({"id": j["id"], "messages": j["messages"], "tools": tools})
    code = (PROMPTS / "code_rewrite_source.txt").read_text(encoding="utf-8")
    out.append({"id": "code_rewrite", "messages": [{"role": "user", "content": REWRITE.format(code=code)}], "tools": None})
    return out


class server:
    def __init__(self, name, cmd, port, log_path, env=None, wait_for=None):
        self.name, self.cmd, self.port, self.log_path, self.env, self.wait_for = name, cmd, port, log_path, env, wait_for
        self.p = None

    def start(self):
        start_at = pathlib.Path(self.log_path).stat().st_size if pathlib.Path(self.log_path).exists() else 0
        self.log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self.p = subprocess.Popen(self.cmd + ["--port", str(self.port)], stdout=self.log, stderr=subprocess.STDOUT,
                                  env=self.env)
        t0 = time.time()
        while time.time() - t0 < 1800:
            if self.p.poll() is not None:
                sys.exit(f"{self.name} stopped while loading (exit code {self.p.returncode}); see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            sys.exit(f"{self.name} didn't start within 30 minutes")
        took = time.time() - t0
        # Kera opens its saved chats once it has hashed the model file (in the background, after loading)
        while self.wait_for and time.time() - t0 < 1800:
            with open(self.log_path, "rb") as f:  # only what this start wrote
                f.seek(start_at)
                if self.wait_for.encode() in f.read():
                    break
            time.sleep(0.5)
        return took

    def stop(self):
        if self.p and self.p.poll() is None:
            time.sleep(5)  # Kera saves each chat when its reply ends, in the background: let the last one finish
            self.p.terminate()
            try:
                self.p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        if self.p:
            self.log.close()

    def ask(self, messages, max_tokens, tools=None, reuse=True):
        return ask(self.port, messages, max_tokens, tools, reuse)


class ollama:
    """An Ollama that is already running, sent the same requests through its own API (/api/chat)"""

    def __init__(self, url, model, ctx):
        self.name, self.url, self.model, self.ctx = "Ollama", url.rstrip("/"), model, ctx

    def _post(self, body):
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=STALL_S)

    def start(self):
        t0 = time.time()
        with self._post({"model": self.model, "messages": [], "keep_alive": "30m", "options": {"num_ctx": self.ctx}}) as r:
            r.read()
        return time.time() - t0

    def stop(self):
        # unload, so the model doesn't stay in memory after the test
        with self._post({"model": self.model, "messages": [], "keep_alive": 0}) as r:
            r.read()

    think_off = True  # "think": false, unless this Ollama refuses it for the model

    def ask(self, messages, max_tokens, tools=None, reuse=True):
        body = {"model": self.model, "messages": messages, "stream": True, "keep_alive": "30m",
                "options": {"temperature": 0, "top_k": 1, "num_predict": max_tokens, "num_ctx": self.ctx}}
        if tools:
            body["tools"] = tools
        if self.think_off:
            try:
                return self._stream(dict(body, think=False))
            except urllib.error.HTTPError as e:
                if "think" not in e.read().decode("utf-8", "replace"):
                    raise
                self.think_off = False
        return self._stream(body)

    def _stream(self, body):
        t0 = time.time()
        first, parts, done = None, [], {}
        with self._post(body) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                j = json.loads(line)
                if "error" in j:
                    raise RuntimeError(f"Ollama: {j['error']}")
                m = j.get("message") or {}
                piece = (m.get("content") or "") + (m.get("thinking") or "")
                if m.get("tool_calls"):
                    piece += json.dumps(m["tool_calls"])
                if piece:
                    if first is None:
                        first = time.time() - t0
                    parts.append(piece)
                if j.get("done"):
                    done = j
        per_s = lambda n, ns: n / (ns / 1e9) if ns else 0.0
        timings = {"predicted_n": done.get("eval_count", 0), "prompt_n": done.get("prompt_eval_count", 0), "cache_n": 0,
                   "predicted_per_second": per_s(done.get("eval_count", 0), done.get("eval_duration", 0)),
                   "prompt_per_second": per_s(done.get("prompt_eval_count", 0), done.get("prompt_eval_duration", 0))}
        return (first if first is not None else time.time() - t0), "".join(parts), timings


def ask(port, messages, max_tokens, tools=None, reuse=True):
    """One streamed chat request: (seconds to the first word, the answer, the server's timings). reuse=False asks
    the server to read the whole prompt again (nothing from its memory of earlier requests)"""
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0, "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": reuse}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, parts, timings = None, [], {}
    with urllib.request.urlopen(req, timeout=STALL_S) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            timings = j.get("timings", timings)
            for c in j.get("choices", []):
                d = c.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if d.get("tool_calls"):
                    piece += json.dumps(d["tool_calls"])
                if piece:
                    if first is None:
                        first = time.time() - t0
                    parts.append(piece)
    return (first if first is not None else time.time() - t0), "".join(parts), timings


def filler(seed, n):
    words = ("the of and to in is that for it as with was on be by at this are from or have an not they which one "
             "you were her all she there would their we him been has when who will more no if out so said what up "
             "its about into than them can only other new some could time these two may then do first any").split()
    return " ".join(words[(seed * 31 + k * 7 + (k * k) % 13) % len(words)] for k in range(n)) + "."


def long_document(n_chars, tag):
    """Real text for the long-context test: llama.cpp's own source files (not its Unicode tables), in name order,
    cut to n_chars, after a first line of its own"""
    src = ROOT / "third_party" / "llama.cpp" / "src"
    head = f"Project snapshot {tag}.\n\n"
    parts, n = [head], len(head)
    for f in sorted(src.glob("*.cpp")):
        if f.name.startswith("unicode-data"):
            continue
        t = f"// ===== {f.name}\n" + f.read_text(encoding="utf-8", errors="replace") + "\n"
        parts.append(t)
        n += len(t)
        if n >= n_chars:
            break
    if n < n_chars:
        sys.exit(f"only {n} characters of llama.cpp's sources here, {n_chars} asked for")
    return "".join(parts)[:n_chars]


def run_long(srv, args, sizes):
    """The long-context test: for each size a freshly started server (no earlier answer on it to copy from), a
    document of that size and a question, then one follow-up in that chat"""
    res = {"load_seconds": None, "long": []}
    for size in sizes:
        took = srv.start()
        if res["load_seconds"] is None:
            res["load_seconds"] = took
        try:
            warm_up(srv)
            doc = long_document(int(size * args.chars_per_token), f"{size}")
            messages = [{"role": "user", "content": doc + "\n\n" + LONG_ASK}]
            w, answer, t = srv.ask(messages, args.max_tokens)
            messages += [{"role": "assistant", "content": answer}, {"role": "user", "content": LONG_FOLLOW}]
            w2, _, t2 = srv.ask(messages, 32)
            row = {"size": size, "prompt_tokens": t.get("prompt_n", 0), "first_word_s": w,
                   "read_per_s": t.get("prompt_per_second", 0.0), "words_per_s": t.get("predicted_per_second", 0.0),
                   "draft_n": t.get("draft_n", 0), "draft_n_accepted": t.get("draft_n_accepted", 0),
                   "follow_up_s": w2, "follow_up_read": t2.get("prompt_n", 0)}
            res["long"].append(row)
            print(f"  {srv.name:14s} {row['prompt_tokens']:7d} words: first word after {w:6.2f} s "
                  f"({row['read_per_s']:5.0f} read/s), writes {row['words_per_s']:5.1f} words/s{guessed(row)}, "
                  f"follow-up after {w2:.2f} s", flush=True)
        finally:
            srv.stop()
    return res


def run_one(srv, args, prompts):
    """The everyday test: --runs runs, each on a freshly started server asked every prompt once (so Kera's guesser
    never sees a prompt it has answered before), then a chat continued message by message on the last run's server,
    then that chat after a restart"""
    each = {p["id"]: [] for p in prompts}
    loads = []
    for n in range(args.runs):
        loads.append(srv.start())
        try:
            warm_up(srv)
            for p in prompts:
                w, answer, t = srv.ask(p["messages"], args.max_tokens, p["tools"], reuse=False)
                r = {"first_word_s": w, "words_per_s": t.get("predicted_per_second", 0.0),
                     "read_per_s": t.get("prompt_per_second", 0.0), "words": t.get("predicted_n", 0),
                     "draft_n": t.get("draft_n", 0), "draft_n_accepted": t.get("draft_n_accepted", 0),
                     "answer": answer}
                each[p["id"]].append(r)
                print(f"  {srv.name:14s} run {n + 1} of {args.runs}: {p['id']:14s} writes {r['words_per_s']:7.1f} "
                      f"words/s{guessed(r)}, first word after {w:.2f} s", flush=True)

            if n == args.runs - 1:
                # a chat continued message by message, on the last run's server
                system = "You are a helpful assistant. " + filler(1, args.system_words)
                messages = [{"role": "system", "content": system}]
                waits = []
                for i in range(args.turns):
                    messages.append({"role": "user", "content": filler(100 + i, args.turn_words)})
                    w, answer, _ = srv.ask(messages, 32)
                    waits.append(w)
                    messages.append({"role": "assistant", "content": answer or filler(200 + i, 8)})
                print(f"  {srv.name:14s} follow-ups: median wait {statistics.median(waits[1:]):.2f} s over "
                      f"{len(waits)} messages", flush=True)
        finally:
            srv.stop()

    res = {"load_seconds": loads[0], "writing": {}}
    for p in prompts:
        runs = each[p["id"]]
        row = res["writing"][p["id"]] = {
            "first_word_s": statistics.median(r["first_word_s"] for r in runs),
            "words_per_s": statistics.median(r["words_per_s"] for r in runs),
            "read_per_s": statistics.median(r["read_per_s"] for r in runs),
            "answer": runs[-1]["answer"],
            "runs": runs,  # each run's numbers and answer, a fresh server each time
        }
        print(f"  {srv.name:14s} {p['id']:14s} writes {row['words_per_s']:7.1f} words/s, first word after "
              f"{row['first_word_s']:.2f} s (the median of {len(runs)} runs)", flush=True)
    res["follow_up_waits_s"] = waits

    # the same chat after a restart
    srv.start()
    try:
        messages.append({"role": "user", "content": filler(999, args.turn_words)})
        w, _, t = srv.ask(messages, 32)
        res["after_restart"] = {"first_word_s": w, "prompt_read": t.get("prompt_n", 0), "prompt_cached": t.get("cache_n", 0)}
        print(f"  {srv.name:14s} after a restart: first word after {w:.2f} s ({t.get('prompt_n', 0)} words read, "
              f"{t.get('cache_n', 0)} reused)", flush=True)
    finally:
        srv.stop()
    return res


def same_answer(a, b):
    """The same words, apart from the random ids servers give tool calls"""
    strip = lambda s: re.sub(r'"id": "[^"]*"', '"id": ""', s)
    return strip(a) == strip(b)


def same_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-server", help="llama.cpp's own llama-server (stock), for the baseline")
    ap.add_argument("--kera", help="the kera program")
    ap.add_argument("--engines", default="stock,kera", help="which run, in order: stock, kera, ollama")
    ap.add_argument("--ollama", help="Ollama's name for the same model, e.g. nemotron-3-nano:30b")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--ngl", default="99", help="layers on the graphics chip (both servers)")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--runs", type=int, default=3, help="runs of the everyday prompts, each on freshly started servers")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--turn-words", type=int, default=150)
    ap.add_argument("--system-words", type=int, default=300, help="the follow-up chat's system prompt")
    ap.add_argument("--extra", default="", help="more options for both servers, e.g. \"-fa on\"")
    ap.add_argument("--port", type=int, default=8571)
    ap.add_argument("--out", default="compare.json")
    ap.add_argument("--quick", action="store_true", help="short prompts and sizes, to check the tool itself (tiny models)")
    ap.add_argument("--long", help="the long-context test instead, at these sizes in tokens, e.g. 8000,32000,64000")
    ap.add_argument("--chars-per-token", type=float, default=3.0, help="for --long: the model's characters per token")
    ap.add_argument("--stall", type=float, help="seconds without a byte from a server before giving up "
                    "(default 600, or 3600 with --long)")
    args = ap.parse_args()
    if args.quick:
        args.runs, args.max_tokens, args.turns, args.turn_words, args.system_words = 1, 32, 4, 30, 30
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e, need in (("stock", args.llama_server), ("kera", args.kera), ("ollama", args.ollama)):
        if e in engines and not need:
            ap.error(f"--engines has {e}: give {'--llama-server' if e == 'stock' else '--' + e}")
    if not engines or any(e not in LABEL for e in engines):
        ap.error("--engines takes stock, kera and ollama")
    sizes = [int(x) for x in args.long.split(",")] if args.long else []
    global STALL_S
    STALL_S = args.stall or (3600 if sizes else 600)
    if any(n + args.max_tokens + 600 > args.ctx for n in sizes):
        ap.error(f"--long sizes must leave room for the answer in --ctx {args.ctx}")

    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-compare-"))
    common = ["-m", args.model, "-c", str(args.ctx), "-ngl", args.ngl, "-np", "1"] + args.extra.split()
    # Kera's saved chats and machine profile in a folder of their own, so the test never mixes with real chats
    env = dict(os.environ, KERA_HOME=str(work / "kera-home"))
    if "kera" in engines:
        subprocess.run([args.kera, "setup", "--quick"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    prompts = bench_prompts()
    if args.quick:
        prompts = [p for p in prompts if p["id"] == "short_chat"]
    out = {"model": pathlib.Path(args.model).name, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "method": METHOD,
           "settings": vars(args)}
    for e in engines:
        if e == "stock":
            print(f"Stock llama.cpp ({args.llama_server}):", flush=True)
            srv = server("llama.cpp", [args.llama_server] + common, args.port, work / "stock.log")
        elif e == "kera":
            print(f"Kera ({args.kera} serve):", flush=True)
            srv = server("Kera", [args.kera, "serve"] + common, args.port + 1, work / "kera.log", env,
                         wait_for="saving chat states in")
        else:
            print(f"Ollama ({args.ollama} at {args.ollama_url}):", flush=True)
            srv = ollama(args.ollama_url, args.ollama, args.ctx)
        try:
            out[e] = run_long(srv, args, sizes) if sizes else run_one(srv, args, prompts)
        except TimeoutError:
            # the server was stopped on the way out (run_one and run_long stop it whatever happens)
            sys.exit(f"{LABEL[e]} sent nothing for {STALL_S:.0f} s: it stalled (its log is in {work})")

    # Kera against the other engine (the first one that isn't Kera), or else the first two engines
    others = [e for e in engines if e != "kera"]
    a, b = (others[0], "kera") if "kera" in engines and others else (engines + [None])[:2]
    if b is None:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"saved {args.out} (server logs in {work})")
        shutil.rmtree(work / "kera-home", ignore_errors=True)
        return
    A, B = LABEL[a], LABEL[b]
    if sizes:
        print(f"\n| Document | {A}: first word | {B}: first word | {A}: writing | {B}: writing | {A}: follow-up | "
              f"{B}: follow-up |\n| --- | --- | --- | --- | --- | --- | --- |")
        for x, y in zip(out[a]["long"], out[b]["long"]):
            print(f"| {max(x['prompt_tokens'], y['prompt_tokens']):,} words | {x['first_word_s']:.2f} s | {y['first_word_s']:.2f} s | "
                  f"{x['words_per_s']:.1f} words/s | {y['words_per_s']:.1f} words/s | {x['follow_up_s']:.2f} s | "
                  f"{y['follow_up_s']:.2f} s |")
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"saved {args.out} (server logs in {work})")
        shutil.rmtree(work / "kera-home", ignore_errors=True)
        return
    print(f"\n| Test | {A} | {B} | {B} / {A} |\n| --- | --- | --- | --- |")
    ratios = []
    same = 0
    for p in prompts:
        s, k = out[a]["writing"][p["id"]], out[b]["writing"][p["id"]]
        r = k["words_per_s"] / s["words_per_s"] if s["words_per_s"] else 0
        ratios.append(r)
        print(f"| {p['id']}: words written per second | {s['words_per_s']:.1f} | {k['words_per_s']:.1f} | {r:.2f}x |")
        print(f"| {p['id']}: wait for the first word | {s['first_word_s']:.2f} s | {k['first_word_s']:.2f} s | |")
        if same_answer(s["answer"], k["answer"]):
            same += 1
        elif "ollama" not in (a, b):
            n = same_prefix(s["answer"], k["answer"])
            print(f"|   (answers differ from character {n} of {max(len(s['answer']), len(k['answer']))}) | | | |")
    sw, kw = out[a]["follow_up_waits_s"][1:], out[b]["follow_up_waits_s"][1:]
    print(f"| follow-up message: median wait | {statistics.median(sw):.2f} s | {statistics.median(kw):.2f} s | |")
    sr, kr = out[a]["after_restart"], out[b]["after_restart"]
    rr = sr["first_word_s"] / max(kr["first_word_s"], 1e-3)
    sooner = f"{rr:.1f}x sooner" if rr >= 1 else f"{1 / max(rr, 1e-3):.1f}x later"
    print(f"| after a restart: wait for the first word | {sr['first_word_s']:.2f} s | {kr['first_word_s']:.2f} s | "
          f"{sooner} |")
    geo = statistics.geometric_mean([r for r in ratios if r > 0]) if any(r > 0 for r in ratios) else 0
    out["verdict"] = {"against": a, "writing_speed_ratio": geo, "same_answers": f"{same}/{len(prompts)}",
                      "restart_wait_ratio": sr["first_word_s"] / max(kr["first_word_s"], 1e-3)}
    print(f"\nWriting: {B} {geo:.2f}x {A}'s speed on average (geometric mean over {len(prompts)} prompts). "
          + (f"Same answers: {same} of {len(prompts)}. " if "ollama" not in (a, b) else "")
          + f"After a restart {B}'s first word came {sooner}.")
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"saved {args.out} (server logs in {work})")
    shutil.rmtree(work / "kera-home", ignore_errors=True)


if __name__ == "__main__":
    main()
