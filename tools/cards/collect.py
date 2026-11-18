"""Collects chat models' own default sampling from their makers' generation_config.json on Hugging Face (D-036).

  python tools/cards/collect.py [--pages 2] [--out tools/cards/model_cards.json]
  python tools/cards/make_table.py            # then: writes src/tune/model_cards.inc from that file

A model's generation_config.json holds the sampling its makers' own engines apply when a request sets none
(transformers' generate(), vLLM): temperature, top_p, top_k, min_p, repetition_penalty. llama.cpp's converter copies
them into a GGUF file (general.sampling.*), but files made before it did, and many made since, don't carry them. This
finds them for the models people run as GGUF files: the models that Hugging Face's most downloaded and most liked GGUF
repositories were made from, and its most downloaded and most liked chat models.

Only what was read is kept: the file's git blob id is stored with each entry. A gated model's file (Llama, Gemma,
Mistral) can't be read without its licence being accepted; its git blob id can, so the same file is taken from a public
copy of the repository and kept only if its bytes hash to that id. Run with no token: nothing here needs one.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://huggingface.co/api/models"
UA = {"User-Agent": "kera-cards/1"}


def get(url, tries=5):
    """(status, bytes, headers); waits out rate limits."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(min(60, 5 * (attempt + 1)))
                continue
            return e.code, b"", e.headers
        except Exception:
            time.sleep(3 * (attempt + 1))
    return 0, b"", {}


def listing(params, pages):
    url = API + "?" + urllib.parse.urlencode(params)
    out = []
    for _ in range(pages):
        status, body, headers = get(url)
        if status != 200:
            break
        out += json.loads(body)
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", "") or "")
        if not m:
            break
        url = m.group(1)
    return out


def blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def tree(repo):
    status, body, _ = get(f"{API}/{repo}/tree/main")
    if status != 200:
        return {}
    try:
        return {e["path"]: e.get("oid") for e in json.loads(body) if e.get("type") == "file"}
    except Exception:
        return {}


def read_file(repo, path, want_oid, mirrors):
    """The file's bytes, checked against its git blob id; from a public copy when the repository is gated."""
    status, body, _ = get(f"https://huggingface.co/{repo}/resolve/main/{path}")
    if status == 200 and (want_oid is None or blob_id(body) == want_oid):
        return body, repo
    if status in (401, 403) and want_oid:
        for m in mirrors():
            if tree(m).get(path) != want_oid:
                continue
            status, body, _ = get(f"https://huggingface.co/{m}/resolve/main/{path}")
            if status == 200 and blob_id(body) == want_oid:
                return body, m
    return None, None


def base_models(repo):
    """The repositories a model's card says it was made from (its base_model), as "org/name" strings."""
    status, body, _ = get(f"{API}/{repo}")
    if status != 200:
        return None
    try:
        base = (json.loads(body).get("cardData") or {}).get("base_model")
    except Exception:
        return None
    return [b for b in (base if isinstance(base, list) else [base]) if isinstance(b, str)]


def dims(cfg):
    """(layers, width) of the language model in a config.json, or None."""
    for part in (cfg, cfg.get("text_config"), cfg.get("llm_config"), cfg.get("language_config"),
                 cfg.get("thinker_config", {}).get("text_config") if isinstance(cfg.get("thinker_config"), dict) else None):
        if not isinstance(part, dict):
            continue
        layers = part.get("num_hidden_layers", part.get("n_layer", part.get("num_layers")))
        width = part.get("hidden_size", part.get("n_embd", part.get("d_model")))
        if isinstance(layers, int) and isinstance(width, int):
            return layers, width
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=2, help="pages of 1000 per listing")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("model_cards.json")))
    ap.add_argument("--limit", type=int, default=0, help="stop after this many models (a quick try)")
    a = ap.parse_args()

    repos = {}  # the makers' repository -> downloads of what led to it (for the order)

    def add(repo, weight):
        if isinstance(repo, str) and repo.count("/") == 1 and not repo.lower().endswith(("-gguf", "_gguf", ".gguf")):
            repos[repo] = max(repos.get(repo, 0), weight)

    for sort in ("downloads", "likes"):
        for m in listing({"filter": "gguf", "pipeline_tag": "text-generation", "sort": sort, "direction": -1,
                          "limit": 1000, "cardData": "true"}, a.pages):
            base = (m.get("cardData") or {}).get("base_model")
            for b in base if isinstance(base, list) else [base]:
                add(b, m.get("downloads", 0))
        for m in listing({"filter": "transformers", "pipeline_tag": "text-generation", "sort": sort, "direction": -1,
                          "limit": 1000}, a.pages):
            add(m.get("id"), m.get("downloads", 0))
    order = sorted(repos, key=lambda r: -repos[r])
    if a.limit:
        order = order[:a.limit]
    print(f"{len(order)} repositories to read", file=sys.stderr)

    entries, skipped = [], {}
    for n, repo in enumerate(order):
        if n % 50 == 0:
            print(f"  {n}/{len(order)}  kept {len(entries)}", file=sys.stderr)
        files = tree(repo)
        if "generation_config.json" not in files or "config.json" not in files:
            skipped[repo] = "no generation_config.json" if files else "not readable"
            continue
        name = repo.split("/")[1]
        found = []

        def mirrors():
            if not found:
                status, body, _ = get(API + "?" + urllib.parse.urlencode({"search": name, "limit": 40}))
                found.append([m["id"] for m in (json.loads(body) if status == 200 else [])
                              if m["id"] != repo and m["id"].split("/")[1].lower() == name.lower()][:8])
            return found[0]

        gen, gen_from = read_file(repo, "generation_config.json", files["generation_config.json"], mirrors)
        cfg, _ = read_file(repo, "config.json", files["config.json"], mirrors)
        if gen is None or cfg is None:
            skipped[repo] = "gated, and no public copy with the same file"
            continue
        try:
            g, c = json.loads(gen), json.loads(cfg)
        except Exception:
            skipped[repo] = "not JSON"
            continue
        d = dims(c)
        if d is None:
            skipped[repo] = "no layer count in config.json"
            continue
        if g.get("do_sample") is False:
            skipped[repo] = "do_sample false: no sampling of its own"
            continue
        s = {}
        for key, out_key in (("temperature", "temp"), ("top_p", "top_p"), ("top_k", "top_k"), ("min_p", "min_p"),
                             ("repetition_penalty", "repeat")):
            v = g.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                s[out_key] = v
        if not s:
            skipped[repo] = "no sampling values"
            continue
        entries.append({"repo": repo, "layers": d[0], "width": d[1], "sampling": s,
                        "blob": files["generation_config.json"], "read_from": gen_from, "base": base_models(repo)})
    out = {"_about": "Chat models' own default sampling, from their makers' generation_config.json on Hugging Face "
                     "(tools/cards/collect.py). blob: the file's git blob id in the makers' repository; read_from: "
                     "the repository its bytes were read from (a public copy with the same blob id when the makers' "
                     "is gated); base: the repositories its card says it was made from (a repository with the same "
                     "name as its base is another organisation's copy of that model, not its makers').",
           "collected": time.strftime("%Y-%m-%d", time.gmtime()), "models": entries, "skipped": skipped}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"kept {len(entries)} of {len(order)}; wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
