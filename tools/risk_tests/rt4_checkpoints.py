"""M1 risk test 4: does llama.cpp's server re-read the whole chat on hybrid (Mamba + attention) models?

Runs `kera serve` (llama.cpp's server) on a Nemotron-H model with Nemotron 3 Nano's own chat template,
plays a long multi-turn chat the way a chat app does (send the whole history plus the new message),
and records for every turn how many prompt tokens were reused from the cache and how many had to be
read again, plus the server's checkpoint messages.

  python tools/risk_tests/rt4_checkpoints.py --kera build/bin/kera --model tiny/nemotron_h_moe.gguf \
      --template chat_template.jinja --turns 40 [--server-args "--checkpoint-min-step 0"] [--keep-thinking]

The speeds of a tiny model mean nothing; the token counts are what this measures. Nemotron 3 Nano's
template drops the thinking of earlier answers (truncate_history_thinking), so each new message changes
how the previous answer is written, and a hybrid model must go back to a saved checkpoint.
"""
import argparse
import json
import random
import re
import subprocess
import sys
import time
import urllib.request

WORDS = ("the of and to in is that for it as with was on be by at this are from or have an not they which "
         "you one had all their there were can more when what about some been would has into time could "
         "other these two may first then do any like my now over such our man me even most made after also "
         "did many before must through back years where much your way well down should because each just "
         "those people how too little state good very make world still own see men work long get here "
         "between both life being under never day same another know while last might us great old year "
         "off come since against go came right used take three file error line value function return test "
         "build output input model token memory cache server request result list table row column").split()


def text(rng, n_words):
    return " ".join(rng.choice(WORDS) for _ in range(n_words)) + "."


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True, help="chat template file (Jinja)")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=120, help="words the model writes per answer")
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--server-args", default="", help="extra llama-server options")
    ap.add_argument("--keep-thinking", action="store_true", help="truncate_history_thinking=false")
    ap.add_argument("--side-requests", action="store_true",
                    help="after each turn, send a short separate request (as chat apps do for titles and tags)")
    ap.add_argument("--trim-history", type=int, default=0,
                    help="keep only the system message and the last N messages (as some agents do)")
    ap.add_argument("--log", default="rt4_server.log")
    ap.add_argument("--json", help="write the per-turn results here")
    args = ap.parse_args()

    cmd = [args.kera, "serve", "-m", args.model, "--port", str(args.port), "-c", str(args.ctx), "--jinja",
           "--chat-template-file", args.template, "-lv", "4", "--no-warmup", "-np", "1"] + args.server_args.split()
    log = open(args.log, "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            sys.exit("server didn't start; see " + args.log)

        rng = random.Random(4)
        messages = [{"role": "system", "content": "You are a helpful coding assistant. " + text(rng, 120)}]
        rows = []
        for turn in range(args.turns):
            # agent-like turns: some short questions, some long tool outputs
            n_words = rng.choice([30, 60, 120, 400, 900])
            messages.append({"role": "user", "content": text(rng, n_words)})
            sent = messages if not args.trim_history else messages[:1] + messages[1:][-args.trim_history:]
            body = {"messages": sent, "max_tokens": args.max_tokens, "temperature": 0, "cache_prompt": True,
                    "chat_template_kwargs": {"truncate_history_thinking": not args.keep_thinking}}
            log_pos = log.tell()
            t0 = time.time()
            r = post(args.port, body)
            dt = time.time() - t0
            msg = r["choices"][0]["message"]
            # send back what a chat app would: the answer, and its thinking if the server split it out
            back = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("reasoning_content"):
                back["reasoning_content"] = msg["reasoning_content"]
            messages.append(back)
            tm = r.get("timings", {})
            if args.side_requests:
                post(args.port, {"messages": [{"role": "user", "content": "Write a five-word title for: " + text(rng, 20)}],
                                 "max_tokens": 16, "temperature": 0})
            log.flush()
            with open(args.log) as f:
                f.seek(log_pos)
                new_log = f.read()
            row = {
                "turn": turn + 1,
                "prompt_tokens": r["usage"]["prompt_tokens"],
                "reused": tm.get("cache_n"),
                "read_again": tm.get("prompt_n"),
                "full_reprocess": "forcing full prompt re-processing" in new_log,
                "restored": len(re.findall(r"restored context checkpoint", new_log)),
                "erased_too_close": len(re.findall(r"erasing context checkpoint too close", new_log)),
                "erased_old": len(re.findall(r"erasing old context checkpoint", new_log)),
                "created": len(re.findall(r"created context checkpoint", new_log)),
                "seconds": round(dt, 2),
            }
            rows.append(row)
            print(json.dumps(row))
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"command": cmd, "keep_thinking": args.keep_thinking, "side_requests": args.side_requests,
                           "trim_history": args.trim_history, "turns": rows}, f, indent=1)
        n_new = sum(r["read_again"] for r in rows)
        total = sum(r["prompt_tokens"] for r in rows)
        print(f"\nturns {len(rows)}, prompt tokens sent {total}, read again {n_new} "
              f"({100.0 * n_new / max(total, 1):.1f}%), full re-reads {sum(r['full_reprocess'] for r in rows)}")
    finally:
        srv.terminate()
        srv.wait(timeout=30)
        log.close()


if __name__ == "__main__":
    main()
