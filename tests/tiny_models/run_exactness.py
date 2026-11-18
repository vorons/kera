"""Spec P2 exactness test: builds the tiny models, converts them with llama.cpp's converter, and checks
Kera's output on the CPU against the reference code's (golden/*.json), by the section 6 rules.

  python tests/tiny_models/run_exactness.py --kera build/bin/kera --work build/tiny

Needs the packages in requirements.txt (torch from its CPU index). Exits 1 if any design fails.
"""
import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
LLAMA = HERE.parent.parent / "third_party" / "llama.cpp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True, help="the kera program")
    ap.add_argument("--work", required=True, type=pathlib.Path, help="folder for the models (rebuilt each run)")
    ap.add_argument("--only", nargs="*", help="only these designs")
    ap.add_argument("--gpu", action="store_true", help="run Kera on the graphics chip (skips if there is none)")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    import make_tiny_models

    designs = [d for d in make_tiny_models.designs() if not args.only or d in args.only]
    subprocess.run([sys.executable, str(HERE / "make_tiny_models.py"), "--out", str(args.work), "--only", *designs],
                   check=True)

    env = dict(os.environ, PYTHONPATH=str(LLAMA / "gguf-py"))
    failed = []
    for d in designs:
        gguf = args.work / f"{d}.gguf"
        conv = subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(args.work / d),
                               "--outtype", "f32", "--outfile", str(gguf)], env=env, capture_output=True, text=True)
        if conv.returncode != 0:
            print(f"FAIL {d}: llama.cpp's converter failed\n{conv.stderr[-3000:]}")
            failed.append(d)
            continue
        run = subprocess.run([args.kera, "exactness", str(gguf), str(HERE / "golden" / f"{d}.json")]
                             + (["--gpu"] if args.gpu else []), capture_output=True, text=True)
        print(run.stdout.strip())
        if run.returncode != 0:
            print(run.stderr[-3000:])
            failed.append(d)

    print(f"\n{len(designs) - len(failed)}/{len(designs)} designs exact" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
