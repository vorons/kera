"""Writes tests/data/controller_vectors.json: decisions of the reference controller (spec Appendix B,
reference/controller.py) that the C++ port in src/spec/controller.cpp must reproduce exactly.

Run from the repo root: python tests/python/gen_controller_vectors.py"""
import json
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))

import controller  # noqa: E402


def choose_k_cases():
    rng = random.Random(7)
    cases = []
    for _ in range(200):
        a = rng.choice([0.0, 0.3, 0.55, 0.7, 0.8, 0.9, 0.97, 1.0])
        t_draft = rng.choice([0.0005, 0.002, 0.01, 0.05])
        t_check = rng.choice([0.01, 0.05, 0.2, 1.0])
        base = rng.choice([0.0, 1e8, 1e9, 8e9])
        growth = rng.choice([0.0, 0.2, 0.5, 0.8, 0.9])
        ssd = rng.choice([2.0, 5.0, 11.0])
        k_max = rng.choice([4, 8])
        moe_ram = rng.random() < 0.4
        ram_base = rng.choice([1e8, 1e9])
        ram_growth = rng.choice([0.1, 0.5])
        ram = rng.choice([100.0, 210.0])

        def new_bytes(k, base=base, growth=growth):
            return base * (1 + growth * k)

        def new_ram_bytes(k, ram_base=ram_base, ram_growth=ram_growth):
            return ram_base * ram_growth * k

        k, rate = controller.choose_k(a, t_draft, t_check, new_bytes, ssd, k_max=k_max,
                                      new_ram_bytes=new_ram_bytes if moe_ram else None,
                                      ram_gb_s=ram if moe_ram else None)
        cases.append(dict(a=a, t_draft=t_draft, t_check=t_check, base=base, growth=growth, ssd=ssd, k_max=k_max,
                          moe_ram=moe_ram, ram_base=ram_base, ram_growth=ram_growth, ram=ram, k=k, rate=rate))
    return cases


def ktuner_trace():
    """A deterministic run: speed depends on k, with noise from a fixed seed, and the best k changes
    halfway through. Stores every k used, so the C++ port can be checked step by step."""
    rng = random.Random(11)
    tuner = controller.KTuner(k_start=3, k_max=8, trial_tokens=32, period=128, max_period=2048)
    ks, tokens_l, seconds_l = [], [], []
    best_k = 2
    for i in range(6000):
        if i == 3000:
            best_k = 0  # the workload changes: guessing stops paying
        k = tuner.next_k()
        tokens = 1 + (min(k, 4) if k else 0)
        seconds = round(0.01 + 0.004 * k + (0.02 if k != best_k else 0.0) + rng.random() * 1e-4, 9)
        tuner.record(tokens, seconds)
        ks.append(k)
        tokens_l.append(tokens)
        seconds_l.append(seconds)
    return dict(k_start=3, k_max=8, trial_tokens=32, period=128, max_period=2048,
                skip_zero_gain=1.2, zero_every=8, ks=ks, tokens=tokens_l, seconds=seconds_l, final_k=tuner.k)


def acceptance_trace():
    rng = random.Random(3)
    t = controller.AcceptanceTracker()
    out = []
    for _ in range(500):
        guessed = rng.choice([0, 1, 2, 4, 8])
        accepted = rng.randint(0, guessed) if guessed else 0
        out.append(dict(guessed=guessed, accepted=accepted, a=t.update(guessed, accepted)))
    return out


if __name__ == "__main__":
    data = dict(choose_k=choose_k_cases(), ktuner=ktuner_trace(), acceptance=acceptance_trace())
    out = ROOT / "tests" / "data" / "controller_vectors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    assert all(math.isfinite(c["rate"]) for c in data["choose_k"])
