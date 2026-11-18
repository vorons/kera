"""Shadow Bits reference quantizer: SB4 = a 2-bit base plane + a 2-bit correction plane.

Per group of G weights (one shared fp16 scale s and fp16 offset m):
    base  = m + s * q_hi                              q_hi in {0,1,2,3}   -> a usable 2-bit model
    full  = base + (s/4) * (q_lo - 1.5)               q_lo in {0,1,2,3}
          = (m - 0.375*s) + (s/4) * (4*q_hi + q_lo)                       -> an ordinary uniform 4-bit grid
So the base plane alone is a 2-bit model (the draft), and base + correction is a plain
4-bit model with scale s/4 and offset m - 0.375*s. Full mode can therefore reuse normal
4-bit kernels: merge the planes into q4 = 4*q_hi + q_lo once, when an expert enters RAM.

Storage per group of 64: base plane 16 B + s,m 4 B = 20 B (2.5 bits/weight, lives in RAM)
                         correction plane 16 B   (2 bits/weight, lives on the SSD)
                         total 36 B = 4.5 bits/weight, the same as a typical 4-bit format.

alpha chooses what the scale is tuned for: 1.0 = best possible 2-bit base,
0.0 = best possible 4-bit full model, in between = a compromise.
"""
import numpy as np


def _f16(x):
    return x.astype(np.float16).astype(np.float32)


def _shrunk_range(W, c):
    lo, hi = W.min(1, keepdims=True), W.max(1, keepdims=True)
    centre, half = (lo + hi) / 2, (hi - lo) / 2 * c
    return centre - half, centre + half


def encode_sb4(w, group=64, alpha=0.5, shrinks=np.linspace(0.5, 1.0, 26)):
    W = np.asarray(w, dtype=np.float32).reshape(-1, group)
    var = W.var(1, keepdims=True) + 1e-12
    best = None
    for c in shrinks:
        lo, hi = _shrunk_range(W, c)
        m = _f16(lo)
        s = _f16(np.maximum((hi - lo) / 3, 1e-8))
        q_hi = np.clip(np.rint((W - m) / s), 0, 3)
        base = m + s * q_hi
        q_lo = np.clip(np.rint((W - base) / (s / 4) + 1.5), 0, 3)
        full = base + (s / 4) * (q_lo - 1.5)
        score = alpha * ((W - base) ** 2).mean(1, keepdims=True) / var \
            + (1 - alpha) * ((W - full) ** 2).mean(1, keepdims=True) / var
        cand = [score, m, s, q_hi, q_lo]
        if best is None:
            best = cand
        else:
            better = score < best[0]
            best = [np.where(better, a, b) for a, b in zip(cand, best)]
    _, m, s, q_hi, q_lo = best
    return {"m": m.astype(np.float16), "s": s.astype(np.float16),
            "q_hi": q_hi.astype(np.uint8), "q_lo": q_lo.astype(np.uint8),
            "shape": np.shape(w), "group": group}


def decode_base(sb):
    m, s = sb["m"].astype(np.float32), sb["s"].astype(np.float32)
    return (m + s * sb["q_hi"]).reshape(sb["shape"])


def decode_full(sb):
    m, s = sb["m"].astype(np.float32), sb["s"].astype(np.float32)
    return (m + s * sb["q_hi"] + (s / 4) * (sb["q_lo"] - 1.5)).reshape(sb["shape"])


def as_plain_q4(sb):
    """Merge both planes into an ordinary 4-bit layout: (codes 0..15, scale, offset)."""
    s = sb["s"].astype(np.float32)
    q4 = (4 * sb["q_hi"] + sb["q_lo"]).astype(np.uint8)
    return q4, s / 4, sb["m"].astype(np.float32) - 0.375 * s


def pack2(q):
    """Pack 2-bit codes, 4 per byte (the on-disk / in-RAM plane layout)."""
    q = q.reshape(-1, 4).astype(np.uint8)
    return (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(np.uint8)


def unpack2(b):
    b = b.astype(np.uint8)
    return np.stack([(b >> k) & 3 for k in (0, 2, 4, 6)], axis=1).reshape(-1)


def rtn(w, bits, group=64, shrinks=np.linspace(0.5, 1.0, 26)):
    """Baseline: plain round-to-nearest uniform quantizer with the best clipping per group."""
    W = np.asarray(w, dtype=np.float32).reshape(-1, group)
    levels = 2 ** bits - 1
    best_err, best_q = None, None
    for c in shrinks:
        lo, hi = _shrunk_range(W, c)
        m = _f16(lo)
        s = _f16(np.maximum((hi - lo) / levels, 1e-8))
        q = m + s * np.clip(np.rint((W - m) / s), 0, levels)
        err = ((W - q) ** 2).mean(1, keepdims=True)
        if best_err is None:
            best_err, best_q = err, q
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_q = np.where(better, q, best_q)
    return best_q.reshape(np.shape(w))
