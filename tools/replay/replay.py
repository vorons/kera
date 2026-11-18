"""Gate 5b's replay tests (spec Phase 5b item 4): how long chat apps and coding agents wait before each reply,
with Kera's reuse of what was already read, against reading everything again.

  python tools/replay/replay.py --serve "kera serve -m MODEL.gguf [options]" [--port 8080] [--quick]

It starts the server itself (so it can restart it), and runs:
  chat    a 30-message chat (the bench's chat texts, and real OpenAssistant turns when --oasst is given), every
          message sent the way chat apps send it: the whole history plus the new message. Measured with reuse
          on and off (off: each request asks the server not to reuse anything, "cache_prompt": false).
          Gate 5b (b): the median wait with reuse is at least 5x shorter.
  reopen  a long chat, the server stopped and started again (as when the app restarts the next day), then
          one more message: the saved chat comes back from the SSD. Against reading it again on a fresh
          server. Gate 5b (c): at least 20x faster (the spec's chat is 32k words: --reopen-words).
  agent   a coding-agent session: about 15k words of system prompt and tool definitions, then 20 turns of
          tool output and replies. Gate 5b (d): total waiting at least 3x lower with reuse.
  speed   writing speed after the long chat against after a short one. Gate 5b (e): within 5%.

The waits are measured as a chat app sees them: from sending the request to the first word streamed back
(thinking included). Results go to --out as JSON, and a summary with each gate's verdict is printed.
"""
import argparse
import json
import os
import pathlib
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]

WORDS = ("the of and to in is that for it as with was on be by at this are from or have an not they which one you "
         "were her all she there would their we him been has when who will more no if out so said what up its about "
         "into than them can only other new some could time these two may then do first any my now such like our over "
         "man me even most made after also did many before must through back years where much your way well down").split()


def text(seed, n):
    """n words of filler text, different for each seed (the model's answers don't matter here)"""
    return " ".join(WORDS[(seed * 31 + k * 7 + (k * k) % 13) % len(WORDS)] for k in range(n)) + "."


class server:
    def __init__(self, cmd, port, log):
        self.cmd = shlex.split(cmd) + ["--port", str(port)]
        self.port = port
        self.log = log
        self.p = None

    def start(self):
        self.p = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 1800:
            if self.p.poll() is not None:
                sys.exit(f"the server stopped while loading (exit code {self.p.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return time.time() - t0
            except Exception:
                time.sleep(0.5)
        sys.exit("the server didn't start within 30 minutes")

    def stop(self):
        if self.p and self.p.poll() is None:
            if os.name == "nt":
                # Windows can't send Ctrl+C to one process only; Kera saves each chat when its reply ends, in the
                # background, so give the last save a moment and stop it
                time.sleep(5)
                self.p.terminate()
            else:
                self.p.send_signal(signal.SIGINT)  # lets Kera finish writing saved chats
            try:
                self.p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()


def ask(port, messages, reuse, max_tokens, tools=None):
    """Send one chat request, streamed. Returns (seconds to the first word, the answer, the server's timings)."""
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0,
            "cache_prompt": reuse, "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    answer = []
    timings = {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            if "timings" in j:
                timings = j["timings"]
            for c in j.get("choices", []):
                d = c.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if d.get("tool_calls"):
                    piece += json.dumps(d["tool_calls"])
                if piece:
                    if first is None:
                        first = time.time() - t0
                    answer.append(d.get("content") or "")
    return (first if first is not None else time.time() - t0), "".join(answer), timings


def chat_replay(port, turns, words, reuse, max_tokens):
    messages = [{"role": "system", "content": "You are a helpful assistant. " + text(1, words // 2)}]
    waits = []
    for t in range(turns // 2):
        messages.append({"role": "user", "content": text(100 + t, words)})
        w, answer, _ = ask(port, messages, reuse, max_tokens)
        waits.append(w)
        messages.append({"role": "assistant", "content": answer or text(200 + t, 8)})
    return waits


def agent_session(system_words, turns):
    """An agent's long system prompt with tool definitions, then turns of tool output"""
    tools = []
    for k in range(12):
        tools.append({"type": "function", "function": {
            "name": f"tool_{k}", "description": text(300 + k, 60),
            "parameters": {"type": "object", "properties": {f"arg_{a}": {"type": "string", "description": text(400 + k * 5 + a, 20)}
                                                            for a in range(4)}}}})
    # real files from this repo as the tool outputs, as a coding agent reads them
    files = sorted(p for p in (ROOT / "src").rglob("*.cpp"))[: turns]
    system = "You are a coding agent working in a C++ repository. " + text(2, system_words)
    return system, tools, files


def agent_replay(port, system_words, turns, reuse, max_tokens, tool_words):
    system, tools, files = agent_session(system_words, turns)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "Read the source files one by one and summarise each."}]
    waits = []
    for t in range(turns):
        w, answer, _ = ask(port, messages, reuse, max_tokens, tools)
        waits.append(w)
        messages.append({"role": "assistant", "content": answer or "Next file."})
        content = files[t % len(files)].read_text(errors="replace") if files else text(500 + t, tool_words)
        messages.append({"role": "user", "content": "Tool output:\n" + " ".join(content.split()[:tool_words])})
    return waits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", required=True, help='how to start the server, e.g. "kera serve -m model.gguf -c 40000"')
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--quick", action="store_true", help="small sizes, to check the tool itself (tiny models)")
    ap.add_argument("--chat-turns", type=int, default=30)
    ap.add_argument("--chat-words", type=int, default=120, help="words per chat message")
    ap.add_argument("--reopen-words", type=int, default=24000, help="the long chat's length (the spec's 32k tokens)")
    ap.add_argument("--agent-words", type=int, default=11000, help="the agent's system prompt (about 15k tokens)")
    ap.add_argument("--agent-turns", type=int, default=20)
    ap.add_argument("--tool-words", type=int, default=400, help="words of tool output per turn")
    ap.add_argument("--max-tokens", type=int, default=32, help="words written per reply")
    ap.add_argument("--only", nargs="*", choices=["chat", "reopen", "agent", "speed"])
    ap.add_argument("--out", default="replay.json")
    ap.add_argument("--log", default="replay-server.log")
    args = ap.parse_args()
    if args.quick:
        args.chat_turns, args.chat_words, args.reopen_words = 10, 30, 400
        args.agent_words, args.agent_turns, args.tool_words = 300, 6, 60
        args.max_tokens = 8
    only = set(args.only or ["chat", "reopen", "agent", "speed"])
    log = open(args.log, "w")
    srv = server(args.serve, args.port, log)
    res = {"serve": args.serve, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "gates": {}}

    def fresh():
        srv.stop()
        return srv.start()

    try:
        res["load_s"] = fresh()
        if "chat" in only:
            on = chat_replay(args.port, args.chat_turns, args.chat_words, True, args.max_tokens)
            fresh()
            off = chat_replay(args.port, args.chat_turns, args.chat_words, False, args.max_tokens)
            m_on, m_off = statistics.median(on), statistics.median(off)
            res["chat"] = {"waits_reuse": on, "waits_no_reuse": off, "median_reuse": m_on, "median_no_reuse": m_off}
            res["gates"]["5b_b_chat"] = {"ratio": m_off / max(m_on, 1e-9), "need": 5.0}
            print(f"chat replay ({len(on)} replies): median wait {m_on:.2f} s with reuse, {m_off:.2f} s without "
                  f"({m_off / max(m_on, 1e-9):.1f}x; the gate asks for 5x)")
        if "reopen" in only or "speed" in only:
            fresh()
            long_chat = [{"role": "system", "content": text(3, args.reopen_words)},
                         {"role": "user", "content": text(4, 40)}]
            w_first, answer, t_long = ask(args.port, long_chat, True, args.max_tokens * 4)
            long_chat.append({"role": "assistant", "content": answer or "OK."})
            # writing speed on the long chat against a short one (gate (e))
            if "speed" in only:
                _, _, t_short = ask(args.port, [{"role": "user", "content": text(5, 40)}], False, args.max_tokens * 4)
                s_long, s_short = t_long.get("predicted_per_second", 0), t_short.get("predicted_per_second", 0)
                res["speed"] = {"long_tok_s": s_long, "short_tok_s": s_short}
                if s_short > 0:
                    res["gates"]["5b_e_speed"] = {"ratio": s_long / s_short, "need": 0.95}
                    print(f"writing speed: {s_long:.2f} words/s after the long chat, {s_short:.2f} after a short one "
                          f"({100 * s_long / s_short:.0f}%; the gate asks for at least 95%)")
            if "reopen" in only:
                srv.stop()                       # the app closes; saved chats are written
                srv.start()                      # and opens the next day
                long_chat.append({"role": "user", "content": text(6, 30)})
                w_saved, _, t_saved = ask(args.port, long_chat, True, args.max_tokens)
                fresh()
                w_reread, _, _ = ask(args.port, long_chat, False, args.max_tokens)
                res["reopen"] = {"wait_saved": w_saved, "wait_reread": w_reread, "reused": t_saved.get("cache_n"),
                                 "first_wait": w_first}
                res["gates"]["5b_c_reopen"] = {"ratio": w_reread / max(w_saved, 1e-9), "need": 20.0}
                print(f"reopening a {args.reopen_words}-word chat after a restart: first word after {w_saved:.2f} s "
                      f"({t_saved.get('cache_n')} tokens came back from the SSD), {w_reread:.2f} s re-reading it "
                      f"({w_reread / max(w_saved, 1e-9):.1f}x; the gate asks for 20x)")
        if "agent" in only:
            fresh()
            on = agent_replay(args.port, args.agent_words, args.agent_turns, True, args.max_tokens, args.tool_words)
            fresh()
            off = agent_replay(args.port, args.agent_words, args.agent_turns, False, args.max_tokens, args.tool_words)
            res["agent"] = {"waits_reuse": on, "waits_no_reuse": off, "total_reuse": sum(on), "total_no_reuse": sum(off)}
            res["gates"]["5b_d_agent"] = {"ratio": sum(off) / max(sum(on), 1e-9), "need": 3.0}
            print(f"agent replay ({len(on)} turns): {sum(on):.1f} s of waiting with reuse, {sum(off):.1f} s without "
                  f"({sum(off) / max(sum(on), 1e-9):.1f}x; the gate asks for 3x)")
    finally:
        srv.stop()
        log.close()
    for g in res["gates"].values():
        g["pass"] = g["ratio"] >= g["need"]
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
