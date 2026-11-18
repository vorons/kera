"""Checks that the reference code in reference/ (spec Appendices A-C, copied verbatim) still gives the
numbers the spec quotes. Run: python -m pytest tests/python"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))

import controller  # noqa: E402
import planner  # noqa: E402
import shadow_bits as sb  # noqa: E402


# ---- Appendix C: planner example output ---------------------------------------------------------

MAC = dict(usable_ram_gb=10, ram_gb_s=100, ssd_gb_s=2.0)
MINI = dict(usable_ram_gb=50, ram_gb_s=210, ssd_gb_s=5.0)
SPARK = dict(usable_ram_gb=110, ram_gb_s=231, ssd_gb_s=11.0)


@pytest.mark.parametrize("args,kwargs,expected", [
    ((32, 32, 32, 4.5), MAC, 0.24),
    ((70, 70, 70, 4.5), MINI, 5.33),
    ((120, 7.5, 3, 4.5), dict(hit_rate=0.5, **MAC), 1.48),
    ((35, 4.5, 2, 4.5), dict(hit_rate=0.75, **MAC), 4.97),
    ((744, 40, 21, 4.5), dict(hit_rate=0.82, **SPARK), 3.67),
    ((744, 40, 21, 4.5), dict(hit_rate=0.82, overlap=True, **SPARK), 5.28),
])
def test_planner_examples(args, kwargs, expected):
    assert round(planner.ceiling_tok_s(*args, **kwargs), 2) == expected


# ---- Appendix A: Shadow Bits ---------------------------------------------------------------------

def _rel(w, q):
    return float(((w - q) ** 2).mean() / w.var())


@pytest.mark.parametrize("name,make,tol,expected", [
    # spec table: plain 2-bit, plain 4-bit, SB4 base, SB4 full (alpha 0.2), relative to the weights' spread
    ("gaussian", lambda r, n: r.standard_normal(n), 0.0005, (0.1108, 0.0073, 0.1170, 0.0083)),
    ("laplace", lambda r, n: r.laplace(size=n), 0.0005, (0.1628, 0.0114, 0.1760, 0.0132)),
    ("student-t", lambda r, n: r.standard_t(4, size=n), 0.003, (0.1760, 0.0126, 0.1910, 0.0149)),
])
def test_sb4_synthetic_table(name, make, tol, expected):
    w = make(np.random.default_rng(0), 1_280_000).astype(np.float32)
    enc = sb.encode_sb4(w, group=64, alpha=0.2)
    got = (_rel(w, sb.rtn(w, 2)), _rel(w, sb.rtn(w, 4)), _rel(w, sb.decode_base(enc)), _rel(w, sb.decode_full(enc)))
    for g, e in zip(got, expected):
        assert abs(g - e) <= tol, f"{name}: got {got}, spec says {expected}"


def test_sb4_merge_is_exactly_plain_4bit():
    w = np.random.default_rng(1).standard_normal(64 * 1000).astype(np.float32)
    enc = sb.encode_sb4(w, group=64, alpha=0.2)
    q4, scale, offset = sb.as_plain_q4(enc)
    assert q4.min() >= 0 and q4.max() <= 15
    merged = (offset + scale * q4).reshape(w.shape)
    assert np.array_equal(merged, sb.decode_full(enc))


def test_sb4_packing_round_trips():
    q = np.random.default_rng(2).integers(0, 4, size=64 * 100).astype(np.uint8)
    packed = sb.pack2(q)
    assert packed.size == q.size // 4
    assert np.array_equal(sb.unpack2(packed), q)


def test_sb4_alpha_tradeoff_gaussian():
    # spec: alpha 1.0 -> base 0.1108 / full 0.0141; 0.5 -> 0.1122 / 0.0106; 0.0 -> 0.1274 / 0.0073
    w = np.random.default_rng(0).standard_normal(1_280_000).astype(np.float32)
    for alpha, base_e, full_e in ((1.0, 0.1108, 0.0141), (0.5, 0.1122, 0.0106), (0.0, 0.1274, 0.0073)):
        enc = sb.encode_sb4(w, group=64, alpha=alpha)
        assert abs(_rel(w, sb.decode_base(enc)) - base_e) <= 0.0006
        assert abs(_rel(w, sb.decode_full(enc)) - full_e) <= 0.0006


# ---- Appendix B: controller ----------------------------------------------------------------------

def test_tokens_per_check():
    assert controller.tokens_per_check(1.0, 4) == 5.0
    assert abs(controller.tokens_per_check(0.5, 1) - 1.5) < 1e-12


def test_choose_k_dense_guessing_pays():
    # dense model streamed from a 2 GB/s SSD: every check reads the same bytes, so guessing pays
    k, rate = controller.choose_k(0.75, 0.05, 1.0, lambda k: 8e9, 2.0)
    assert k == 8 and rate > 1.0 / (1.0 + 4.0)


def test_choose_k_moe_backfires():
    # each guess pulls in many new experts: no guessing
    k, _ = controller.choose_k(0.6, 0.01, 0.2, lambda k: 1e9 * (1 + 0.9 * k), 2.0)
    assert k == 0


def test_acceptance_counts_only_tested_guesses():
    t = controller.AcceptanceTracker(start=0.8, weight=1.0)
    t.update(4, 2)  # 2 accepted, then 1 rejection: 2 / 3
    assert abs(t.a - 2 / 3) < 1e-12


def test_ktuner_switches_off_when_guessing_loses():
    tuner = controller.KTuner(k_start=2, period=64, trial_tokens=16)
    for _ in range(4000):
        k = tuner.next_k()
        tokens = 1 if k == 0 else 1  # guessing produces nothing extra here
        seconds = 0.01 if k == 0 else 0.01 + 0.005 * k
        tuner.record(tokens, seconds)
    assert tuner.k == 0
