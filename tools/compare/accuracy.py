"""Are Kera's answers as accurate as another engine's? The same model, the same questions with known answers, through
each engine's OpenAI-style chat API, greedy (temperature 0), each answer scored right or wrong.

  python tools/compare/accuracy.py --engine kera=http://127.0.0.1:8700 --engine ollama=http://127.0.0.1:11500 \
      --model-name ollama=kera-acc --gsm8k gsm8k.parquet --arc arc.parquet --n 60 --out result.json

GSM8K (school maths word problems; the answer is a number) and ARC-Challenge (science, multiple choice) are read
from their Hugging Face parquet files (openai/gsm8k main/test, allenai/ai2_arc ARC-Challenge/test); the first N of
each are asked. Also reported: how often two engines gave word-for-word the same answer, and each engine's time.
Needs pandas and pyarrow.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

import pandas as pd

GSM_PROMPT = ("Solve this problem. Work it out step by step, then give the final answer on the last line as "
              "'Answer: <number>'.\n\n{q}")
ARC_PROMPT = "{q}\n\n{choices}\n\nReply with the letter of the correct answer only."


def ask(url, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_p": 1,
            "seed": 1, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    return j["choices"][0]["message"]["content"] or ""


def number(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"answer\s*[:：]?\s*\**\s*(-?\d+(?:\.\d+)?)", s, re.I)
    if not m:
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m[-1])
    return int(v) if v == int(v) else v


def letter(s):
    m = re.search(r"\b([A-E])\b", s.strip())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", action="append", required=True, help="name=base URL")
    ap.add_argument("--model-name", action="append", default=[], help="name=model (the API's model field)")
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--arc", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    engines = dict(e.split("=", 1) for e in a.engine)
    names = dict(m.split("=", 1) for m in a.model_name)

    gsm = pd.read_parquet(a.gsm8k).head(a.n)
    arc = pd.read_parquet(a.arc).head(a.n)
    items = []
    for _, r in gsm.iterrows():
        items.append({"set": "gsm8k", "prompt": GSM_PROMPT.format(q=r["question"]), "max": 512,
                      "gold": number(r["answer"].split("####")[-1])})
    for _, r in arc.iterrows():
        ch = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
        items.append({"set": "arc", "prompt": ARC_PROMPT.format(q=r["question"], choices=ch), "max": 8,
                      "gold": str(r["answerKey"])})

    result = {"n_per_set": a.n, "engines": {}, "items": []}
    answers = {}
    for name, url in engines.items():
        t0 = time.time()
        out = []
        for i, it in enumerate(items):
            text = ask(url, names.get(name, "model"), it["prompt"], it["max"])
            got = number(text) if it["set"] == "gsm8k" else letter(text)
            out.append({"text": text, "got": got, "right": got == it["gold"]})
            if (i + 1) % 20 == 0:
                print(f"{name}: {i + 1}/{len(items)}", flush=True)
        answers[name] = out
        secs = time.time() - t0
        summary = {"seconds": round(secs, 1)}
        for s in ("gsm8k", "arc"):
            idx = [i for i, it in enumerate(items) if it["set"] == s]
            summary[s] = {"right": sum(out[i]["right"] for i in idx), "of": len(idx)}
        result["engines"][name] = summary
        print(name, json.dumps(summary), flush=True)

    keys = list(answers)
    result["agreement"] = []
    for ia in range(len(keys)):
        for ib in range(ia + 1, len(keys)):
            x, y = answers[keys[ia]], answers[keys[ib]]
            ag = {"engines": [keys[ia], keys[ib]], "same_text": sum(p["text"] == q["text"] for p, q in zip(x, y)),
                  "same_answer": sum(p["got"] == q["got"] for p, q in zip(x, y)), "of": len(items)}
            result["agreement"].append(ag)
            print("agreement", json.dumps(ag), flush=True)
    for i, it in enumerate(items):
        result["items"].append({"set": it["set"], "gold": it["gold"],
                                **{k: {"got": v[i]["got"], "right": v[i]["right"], "text": v[i]["text"]}
                                   for k, v in answers.items()}})
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
