"""Tiny random-weight copies of the target model designs (spec P2, section 2a).

For each design, builds a tiny model from its config with Hugging Face transformers (the official
reference code), fills every weight from a seeded numpy generator (so the weights don't depend on the
torch version), and saves it with a tokenizer. With --golden it also records the reference code's
greedy output, which the exactness test compares Kera against token for token.

  python make_tiny_models.py --out DIR            # write the models (the tests do this each run)
  python make_tiny_models.py --out DIR --golden   # also rewrite golden/*.json (after a deliberate change)

Prompts are token ids, not text, so the tokenizer only has to satisfy llama.cpp's converter. Every
design uses GPT-2's tokenizer (MIT licence, in tokenizer/) plus the special tokens GLM's converter
asks for, with the vocabulary padded to 50,264 (a multiple of 8, as Nemotron's converter wants).
"""
import argparse
import hashlib
import json
import pathlib
import shutil

import numpy as np
import torch
import transformers
from transformers import (AutoModelForCausalLM, DeepseekV3Config, GlmMoeDsaConfig, GPT2TokenizerFast, NemotronHConfig,
                          Qwen3Config)

HERE = pathlib.Path(__file__).resolve().parent
VOCAB = 50264
PROMPT = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 198, 1212, 318, 257, 1332, 286, 262,
          10717, 1080, 11, 290, 340, 815, 670, 13, 2061, 318, 1306, 30]  # 29 tokens of ordinary GPT-2 text
N_NEW = 24
GLM_SPECIAL = ["[gMASK]", "<|user|>", "<|observation|>", "<|assistant|>", "<sop>"]


def designs():
    """The P2 designs. Small, but each keeps the parts that make the design what it is."""
    common = dict(vocab_size=VOCAB, hidden_size=64, max_position_embeddings=512, tie_word_embeddings=False)
    return {
        # Nemotron 3 Nano style: hybrid Mamba-2 / MoE / attention (spec 2a)
        "nemotron_h_moe": NemotronHConfig(
            **common,
            layers_block_type=["mamba", "moe", "attention", "moe", "mamba", "moe"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            mamba_num_heads=8, mamba_head_dim=16, ssm_state_size=16, n_groups=2, conv_kernel=4, chunk_size=16,
            intermediate_size=128, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
            n_shared_experts=1, moe_shared_expert_intermediate_size=48, n_group=1, topk_group=1,
            norm_topk_prob=True, routed_scaling_factor=2.5, use_mamba_kernels=False,
        ),
        # Nemotron 3 Ultra style: the same hybrid with LatentMoE (experts work in a smaller latent space)
        "nemotron_h_latent_moe": NemotronHConfig(
            **common,
            layers_block_type=["mamba", "moe", "attention", "moe"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            mamba_num_heads=8, mamba_head_dim=16, ssm_state_size=16, n_groups=2, conv_kernel=4, chunk_size=16,
            intermediate_size=128, n_routed_experts=16, num_experts_per_tok=4, moe_intermediate_size=32,
            moe_latent_size=32, n_shared_experts=1, moe_shared_expert_intermediate_size=48, n_group=1,
            topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5, use_mamba_kernels=False,
        ),
        # GLM-5.x style: MLA attention plus the sparse-attention indexer (index_topk smaller than the prompt,
        # so the indexer really chooses), dense first layer, then MoE
        "glm_moe_dsa": GlmMoeDsaConfig(
            **common,
            num_hidden_layers=3, first_k_dense_replace=1, intermediate_size=128,
            num_attention_heads=2, num_key_value_heads=2, q_lora_rank=64, kv_lora_rank=64,
            # GLM-5's own head sizes (rope 64 = head_dim, indexer head 128): llama.cpp's indexer is built
            # around them (its Hadamard rotation matches the indexer head)
            qk_nope_head_dim=64, qk_rope_head_dim=64, head_dim=64, v_head_dim=64,
            n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, n_shared_experts=1,
            n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5,
            index_topk=8, index_head_dim=128, index_n_heads=8,
            # keys GLM-5's own config file carries that the converter reads
            scoring_func="sigmoid", topk_method="noaux_tc", rope_interleave=True, indexer_rope_interleave=True,
        ),
        # The GLM-5.x design with its multi-token-prediction (MTP) head: one more MoE block after the last
        # layer, which llama.cpp can guess the next words with (M4, spec 11.1). Its weights are glm_moe_dsa's
        # (WEIGHTS_OF), but its indexer keeps every position, as real GLM-5 does (2048) until a chat is longer:
        # with a top 8, any change in rounding (16-bit memory, flash attention, reading several words at once,
        # as a guess check does) can change which positions are attended to (docs/NEGATIVE_RESULTS.md N-003),
        # so guessing couldn't be checked word for word.
        "glm_moe_dsa_mtp": GlmMoeDsaConfig(
            **common,
            num_hidden_layers=3, first_k_dense_replace=1, intermediate_size=128,
            num_attention_heads=2, num_key_value_heads=2, q_lora_rank=64, kv_lora_rank=64,
            qk_nope_head_dim=64, qk_rope_head_dim=64, head_dim=64, v_head_dim=64,
            n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, n_shared_experts=1,
            n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5,
            index_topk=2048, index_head_dim=128, index_n_heads=8,
            scoring_func="sigmoid", topk_method="noaux_tc", rope_interleave=True, indexer_rope_interleave=True,
            num_nextn_predict_layers=1,
        ),
        # A plain dense model (Qwen3 style: grouped-query attention with q/k norms): what most fine-tuning
        # (M11, spec Phase 10) starts from, and the dense case for guessing ahead (M4)
        "qwen3_dense": Qwen3Config(
            **common,
            num_hidden_layers=3, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        ),
        # Kimi K2.6 / DeepSeek-V3 style: MLA attention, one dense layer, then MoE with a shared expert
        "deepseek_v3_mla": DeepseekV3Config(
            **common,
            num_hidden_layers=3, first_k_dense_replace=1, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=32,
            qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
            n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, n_shared_experts=1,
            n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5,
            # Real DeepSeek-V3 and Kimi config files carry these; transformers hard-codes the sigmoid and
            # doesn't write them, but llama.cpp's converter reads them (without them llama.cpp falls back
            # to DeepSeek-V2's softmax gating)
            scoring_func="sigmoid", topk_method="noaux_tc",
        ),
    }


# designs that reuse another design's weights
WEIGHTS_OF = {"glm_moe_dsa_mtp": "glm_moe_dsa"}


def seed_for(design, name):
    return int.from_bytes(hashlib.sha256(f"{design}/{name}".encode()).digest()[:8], "little")


def fill_weights(design, model):
    """Every parameter from a numpy generator seeded by its name: the same weights whatever torch does.
    Norm weights start near 1; Mamba's decay and step parameters get the values the reference uses."""
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            rng = np.random.default_rng(seed_for(design, name))
            shape = tuple(p.shape)
            if name.endswith("A_log"):
                v = np.log(np.arange(1, shape[0] + 1, dtype=np.float32))
            elif name.endswith(".D"):
                v = np.ones(shape, dtype=np.float32)
            elif name.endswith("dt_bias"):
                dt = np.exp(rng.uniform(np.log(1e-3), np.log(1e-1), size=shape)).astype(np.float32)
                v = dt + np.log(-np.expm1(-dt))  # inverse softplus, as the reference initialises it
            elif "norm" in name and p.dim() == 1:
                v = 1.0 + 0.1 * rng.standard_normal(shape).astype(np.float32)
            elif p.dim() == 1:
                v = 0.02 * rng.standard_normal(shape).astype(np.float32)
            else:
                fan_in = shape[-1]
                v = (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)
                if "indexer.weights_proj" in name:
                    # DSA's indexer sums ReLU'd scores with these per-head weights. Negative weights make
                    # many scores tie (often at exactly 0), and engines break top-k ties differently
                    # (even the reference does, by sequence length), so keep them positive: the top-k
                    # is then decided by distinct positive scores
                    v = np.abs(v)
            p.copy_(torch.from_numpy(np.ascontiguousarray(v)).to(p.dtype))
        # the output layer a little sharper, so the next word is rarely a near-tie
        head = model.get_output_embeddings()
        if head is not None:
            head.weight.mul_(16.0)


def use_original_dt_limit(model):
    """transformers' Nemotron-H code clamps Mamba's step size at time_step_min (0.001) when it reads a
    prompt, but not when it adds one word at a time, so the reference disagrees with itself (by up to 0.26
    in log-probability on these models). NVIDIA's original code, transformers' own Mamba-2 and llama.cpp
    all use time_step_limit, which is (0, inf): no clamp. Use that here too (docs/NEGATIVE_RESULTS.md)."""
    limit = getattr(model.config, "time_step_limit", None)
    if limit is None:
        return
    for module in model.modules():
        if hasattr(module, "time_step_limit"):
            module.time_step_limit = tuple(limit)


def make_tokenizer():
    tok = GPT2TokenizerFast(tokenizer_file=str(HERE / "tokenizer" / "tokenizer.json"))
    tok.add_special_tokens({"additional_special_tokens": GLM_SPECIAL})
    assert max(tok.get_vocab().values()) < VOCAB
    return tok


def add_mtp_block(design, cfg, out_dir):
    """Appends the MTP block the way real GLM-5 and DeepSeek-V3 checkpoints store it (transformers itself
    doesn't build it): model.layers.N with every tensor of an MoE layer, plus eh_proj (joins the token's
    embedding with the last hidden state), enorm, hnorm and shared_head.norm."""
    from safetensors.torch import load_file, save_file
    files = sorted(out_dir.glob("*.safetensors"))
    assert len(files) == 1, "tiny models are saved as one file"
    tensors = load_file(str(files[0]))
    n = cfg.num_hidden_layers
    last = f"model.layers.{n - 1}."
    h = cfg.hidden_size
    for name, t in sorted(tensors.items()):
        if name.startswith(last):
            tensors[f"model.layers.{n}." + name[len(last):]] = t.clone()
    extra = {f"model.layers.{n}.eh_proj.weight": (h, 2 * h), f"model.layers.{n}.enorm.weight": (h,),
             f"model.layers.{n}.hnorm.weight": (h,), f"model.layers.{n}.shared_head.norm.weight": (h,)}
    for name, shape in extra.items():
        rng = np.random.default_rng(seed_for(design, name))
        if len(shape) == 1:
            v = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            v = rng.standard_normal(shape) / np.sqrt(shape[-1])
        tensors[name] = torch.from_numpy(v.astype(np.float32))
    # the copied block gets its own weights, so it isn't a copy of the last layer
    for name in sorted(tensors):
        if name.startswith(f"model.layers.{n}.") and name not in extra:
            t = tensors[name]
            rng = np.random.default_rng(seed_for(design, name))
            if t.dim() == 1:
                v = (1.0 + 0.1 * rng.standard_normal(tuple(t.shape))) if "norm" in name else 0.02 * rng.standard_normal(tuple(t.shape))
            else:
                v = rng.standard_normal(tuple(t.shape)) / np.sqrt(t.shape[-1])
                if "indexer.weights_proj" in name:
                    v = np.abs(v)
            tensors[name] = torch.from_numpy(np.ascontiguousarray(v).astype(np.float32)).to(t.dtype)
    save_file(tensors, str(files[0]), metadata={"format": "pt"})


def build(design, cfg, out_dir):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    model.eval()
    fill_weights(WEIGHTS_OF.get(design, design), model)
    use_original_dt_limit(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    make_tokenizer().save_pretrained(out_dir)
    if getattr(cfg, "num_nextn_predict_layers", 0):
        add_mtp_block(design, cfg, out_dir)
    return model


def greedy_golden(design, model):
    """The reference code's greedy output for PROMPT. For every step it also keeps the top-5 next words
    with their log-probabilities (from one pass over the whole sequence, the way the checker reads it)
    and the probability gap between the top two (section 6: below 0.001 is a near-tie)."""
    ids = torch.tensor([PROMPT])
    out = []
    with torch.no_grad():
        for _ in range(N_NEW):
            nxt = int(torch.argmax(model(input_ids=ids, use_cache=False).logits[0, -1]))
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        full = torch.tensor([PROMPT + out[:-1]])
        logp = torch.log_softmax(model(input_ids=full, use_cache=False).logits[0].double(), dim=-1)[len(PROMPT) - 1:]
    steps = []
    for i in range(N_NEW):
        top = torch.topk(logp[i], 5)
        probs = torch.exp(top.values)
        steps.append({
            "token": out[i],
            "top_ids": [int(x) for x in top.indices],
            "top_logp": [round(float(x), 6) for x in top.values],
            "gap": round(float(probs[0] - probs[1]), 6),
        })
    return {
        "design": design,
        "architecture": model.config.architectures[0] if model.config.architectures else type(model).__name__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "prompt": PROMPT,
        "generated": out,
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--golden", action="store_true", help="also rewrite golden/*.json")
    ap.add_argument("--only", nargs="*", help="only these designs")
    args = ap.parse_args()
    for design, cfg in designs().items():
        if args.only and design not in args.only:
            continue
        d = args.out / design
        if d.exists():
            shutil.rmtree(d)
        model = build(design, cfg, d)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{design}: {n_params / 1e6:.1f}M parameters -> {d}")
        if args.golden:
            g = greedy_golden(design, model)
            path = HERE / "golden" / f"{design}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(g, indent=1) + "\n")
            print(f"  golden: {g['generated'][:8]}... smallest gap {min(s['gap'] for s in g['steps']):.4f}")


if __name__ == "__main__":
    main()
