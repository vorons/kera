"""MLX-LM's LoRA adapters become PEFT adapters that change a layer exactly as MLX does (tools/finetune, spec Phase 10).
MLX itself needs a Mac with Apple silicon, so this checks the conversion on made-up adapters. Run: python -m pytest tests/python"""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "finetune"))

safetensors_numpy = pytest.importorskip("safetensors.numpy")
import kera_finetune  # noqa: E402


def test_mlx_adapter_becomes_the_same_peft_change(tmp_path):
    rng = np.random.default_rng(0)
    rank, scale, d_in, d_out = 4, 20.0, 16, 24
    mlx = tmp_path / "mlx"
    mlx.mkdir()
    modules = ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"]
    w = {}
    for m in modules:
        w[m + ".lora_a"] = rng.standard_normal((d_in, rank)).astype(np.float32)   # MLX: [in, r]
        w[m + ".lora_b"] = rng.standard_normal((rank, d_out)).astype(np.float32)  # MLX: [r, out]
    safetensors_numpy.save_file(w, str(mlx / "adapters.safetensors"))
    (mlx / "adapter_config.json").write_text(json.dumps({"fine_tune_type": "lora", "num_layers": -1,
                                                          "lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0}}))
    peft = tmp_path / "peft"
    kera_finetune.mlx_to_peft(mlx, peft, "base-model")

    cfg = json.loads((peft / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == rank
    assert sorted(cfg["target_modules"]) == ["down_proj", "q_proj"]
    out = safetensors_numpy.load_file(str(peft / "adapter_model.safetensors"))
    x = rng.standard_normal((3, d_in)).astype(np.float32)
    for m in modules:
        a_peft = out[f"base_model.model.{m}.lora_A.weight"]   # PEFT: [r, in]
        b_peft = out[f"base_model.model.{m}.lora_B.weight"]   # PEFT: [out, r]
        assert a_peft.shape == (rank, d_in) and b_peft.shape == (d_out, rank)
        mlx_delta = scale * (x @ w[m + ".lora_a"]) @ w[m + ".lora_b"]
        peft_delta = (cfg["lora_alpha"] / cfg["r"]) * (x @ a_peft.T) @ b_peft.T
        np.testing.assert_allclose(peft_delta, mlx_delta, rtol=1e-5, atol=1e-5)


def test_examples_learn_only_the_answers():
    class Tok:  # one id per character, enough to check where the labels start
        eos_token_id = 0
        chat_template = None

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

    x, labels = kera_finetune.encode(Tok(), {"prompt": "ab", "completion": "cd"}, 100)
    assert x == [97, 98, 99, 100, 0] and labels == [-100, -100, 99, 100, 0]
    x, labels = kera_finetune.encode(Tok(), {"messages": [{"role": "user", "content": "hi"},
                                                          {"role": "assistant", "content": "yo"}]}, 1000)
    learned = "".join(chr(v) for v in labels if v != -100)
    assert learned.strip() == "yo"  # only the assistant's words
    x, labels = kera_finetune.encode(Tok(), {"text": "plain"}, 100)
    assert labels == x


def test_a_prompt_row_is_put_in_the_chat_format_like_its_messages_row():
    """With a chat template, {prompt, completion} is a user turn and an assistant turn (as MLX-LM makes it): the same
    tokens and the same learned part as the equivalent messages row, so PEFT and MLX learn the same format."""
    class Tok:
        eos_token_id = 0
        chat_template = "made up"

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
            out = "".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
            return out + ("<assistant>" if add_generation_prompt else "")

    as_prompt = kera_finetune.encode(Tok(), {"prompt": "ab", "completion": "cd"}, 1000)
    as_messages = kera_finetune.encode(Tok(), {"messages": [{"role": "user", "content": "ab"},
                                                            {"role": "assistant", "content": "cd"}]}, 1000)
    assert as_prompt == as_messages
    learned = "".join(chr(v) for v in as_prompt[1] if v != -100)
    assert learned == "cd</end>"  # the answer and the template's own end of turn


def test_mlx_trains_as_hard_as_peft(tmp_path):
    """MLX-LM gets PEFT's LoRA scale (alpha / rank) and longest example, not its own scale 20 and 2048 tokens: the
    same data trains the same on a Mac as elsewhere"""
    import argparse
    args = argparse.Namespace(model="m", out=str(tmp_path), steps=10, batch=2, lr=1e-4, seed=0, rank=8, alpha=16,
                              max_len=1024)
    cmd, config = kera_finetune.mlx_command(args, [{"prompt": "a", "completion": "b"}], tmp_path)
    assert config["lora_parameters"]["scale"] == 2.0 and config["lora_parameters"]["rank"] == 8
    assert cmd[cmd.index("--max-seq-length") + 1] == "1024" and "--config" in cmd and "--mask-prompt" in cmd


def test_the_default_length_is_about_three_passes():
    """--steps left out: about three passes over the data (Hugging Face's Trainer default), within 10 to 3,000 steps"""
    assert kera_finetune.default_steps(1000, 4) == 750           # 3 x 1000 / 4
    assert kera_finetune.default_steps(10, 4) == 10              # 8 steps would be too few to learn anything: 10
    assert kera_finetune.default_steps(100, 4) == 75
    assert kera_finetune.default_steps(1_000_000, 4) == 3000     # not days of training
