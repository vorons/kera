"""Streamed experts give the same answers (spec Phase 2 gate (a), section 6; decision D-012).

  python tests/integration/test_stream_exact.py --kera build/bin/kera --models DIR

For each tiny MoE model in DIR (built by tests/tiny_models/run_exactness.py), runs `kera exactness` word by
word without streaming, then with the routed experts streamed: Kera's cache with a budget too small to hold
one layer (so it evicts all the time), the same with next-layer guesses, and the OS's file cache. Every run
must pass the section 6 check, and the streamed runs must match the plain one bit for bit (the same largest
difference to the reference).
"""
import argparse
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE.parent / "tiny_models" / "golden"
DESIGNS = ["nemotron_h_moe", "nemotron_h_latent_moe", "glm_moe_dsa", "glm_moe_dsa_mtp", "deepseek_v3_mla"]
WAYS = {
    "plain": [],
    "Kera's cache, 0.1 MB": ["--stream", "kera", "--stream-budget-mb", "0.1"],
    "Kera's cache, 0.1 MB, guessing the next layer": ["--stream", "kera", "--stream-budget-mb", "0.1", "--stream-lookahead", "1"],
    "the OS's file cache": ["--stream", "os", "--stream-lookahead", "1"],
}


def run(kera, gguf, golden, extra):
    p = subprocess.run([kera, "exactness", str(gguf), str(golden), "--ubatch", "1", *extra],
                       capture_output=True, text=True)
    out = p.stdout
    m = re.search(r"largest top-5 log-prob difference ([0-9.e+-]+)", out)
    return p.returncode, (m.group(1) if m else None), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--models", required=True, type=pathlib.Path)
    args = ap.parse_args()
    failed = 0
    for d in DESIGNS:
        gguf = args.models / f"{d}.gguf"
        if not gguf.exists():
            print(f"FAIL {d}: no model at {gguf}")
            failed += 1
            continue
        results = {}
        for way, extra in WAYS.items():
            code, diff, out = run(args.kera, gguf, GOLDEN / f"{d}.json", extra)
            results[way] = diff
            if code != 0 or diff is None:
                print(f"FAIL {d} ({way}):\n{out}")
                failed += 1
        plain = results["plain"]
        same = all(v == plain for v in results.values())
        if not same:
            print(f"FAIL {d}: streamed runs differ from the plain run: {results}")
            failed += 1
        else:
            print(f"ok   {d}: {len(WAYS)} ways, all pass, identical (largest difference {plain})")
    print("PASS" if failed == 0 else f"FAILED ({failed})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
