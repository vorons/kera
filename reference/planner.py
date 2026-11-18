"""Speed ceiling: the best tokens/second physics allows for a model on a machine.

    without overlap: seconds per token = RAM time + SSD time
    with overlap:    seconds per token = max(RAM time, SSD time) + 0.15 * min(...)
      RAM time = GB read from RAM / RAM speed,  SSD time = GB read from SSD / SSD speed
Overlap means SSD reads happen while the maths runs (section 10). The 0.15 is the part that
never hides in practice; measure it and update it. Real speed always lands below the ceiling.
"""


def ceiling_tok_s(total_b, active_b, always_b, bits, usable_ram_gb, ram_gb_s, ssd_gb_s, hit_rate=None, overlap=False):
    """total_b: all weights (billions). active_b: weights used per token (= total_b for dense).
    always_b: weights used by every token (attention, shared experts, embeddings; = total_b for dense).
    bits: bits per weight including scales (4-bit formats are ~4.5). hit_rate: share of expert
    reads already in RAM (MoE only; None = estimate from how much of the model fits).
    overlap: True if SSD reads run while the maths runs."""
    gb = bits / 8
    always_gb = always_b * gb
    per_token_gb = active_b * gb
    if total_b == active_b:  # dense: every weight every token
        from_ssd = max(total_b * gb - usable_ram_gb, 0)
    else:
        routed_gb = (active_b - always_b) * gb
        pool_gb = (total_b - always_b) * gb
        cache_gb = max(usable_ram_gb - always_gb, 0)
        h = hit_rate if hit_rate is not None else min(cache_gb / pool_gb, 1.0)
        from_ssd = routed_gb * (1 - h)
    ram_t, ssd_t = per_token_gb / ram_gb_s, from_ssd / ssd_gb_s
    if overlap:
        return 1.0 / (max(ram_t, ssd_t) + 0.15 * min(ram_t, ssd_t))
    return 1.0 / (ram_t + ssd_t)


if __name__ == "__main__":
    mac = dict(usable_ram_gb=10, ram_gb_s=100, ssd_gb_s=2.0)
    mini = dict(usable_ram_gb=50, ram_gb_s=210, ssd_gb_s=5.0)
    print("dense 32B, Mac:", round(ceiling_tok_s(32, 32, 32, 4.5, **mac), 2))
    print("dense 70B, Minisforum:", round(ceiling_tok_s(70, 70, 70, 4.5, **mini), 2))
    print("MoE 120B/5B active, Mac, 50% hits:", round(ceiling_tok_s(120, 7.5, 3, 4.5, hit_rate=0.5, **mac), 2))
    print("MoE 35B/3B active, Mac, 75% hits:", round(ceiling_tok_s(35, 4.5, 2, 4.5, hit_rate=0.75, **mac), 2))
    spark = dict(usable_ram_gb=110, ram_gb_s=231, ssd_gb_s=11.0)
    # GLM-5.3-like: 744B total, ~40B used per word, ~21B always used
    print("GLM-5.3-like on a DGX Spark, 82% hits, no overlap:", round(ceiling_tok_s(744, 40, 21, 4.5, hit_rate=0.82, **spark), 2))
    print("GLM-5.3-like on a DGX Spark, 82% hits, overlap:   ", round(ceiling_tok_s(744, 40, 21, 4.5, hit_rate=0.82, overlap=True, **spark), 2))
