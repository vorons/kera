"""Picture and video models get their makers' settings, and only where the command line sets none (rule one, D-033;
docs/NEGATIVE_RESULTS.md N-018).

  python tests/integration/test_picture_settings.py --kera build/bin/kera

Runs `kera image` and `kera video` against a stand-in for the image engine that writes down the command Kera hands it,
so no model is needed. Checked: the step-distilled models get their cards' steps, CFG, sampler, schedule and size;
a picture to start from keeps the user's and the engine's steps and size; FLUX is told apart by its weights
(guidance-distilled or not) and gets its variant's guidance; Wan and LTX get their makers' CFG, flow shift and default
negative prompt; speed-up variants of Wan and unknown models get nothing; and the user's own values always win.
"""
import argparse
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
                "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
                "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def safetensors(path, names):
    """A tiny safetensors file holding one 4-byte tensor per name."""
    header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]} for i, n in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * len(names)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    a = ap.parse_args()
    if os.name == "nt":
        print("SKIP: the stand-in engine is a shell script")
        return
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-picture-settings-"))
    kera_src = pathlib.Path(a.kera).resolve()
    (work / "bin" / "image").mkdir(parents=True)
    kera = work / "bin" / kera_src.name
    shutil.copy2(kera_src, kera)  # kera finds its engines next to itself
    engine = work / "bin" / "image" / "sd-cli"
    engine.write_text('#!/bin/sh\nprintf \'%s\\0\' "$@" > "$KERA_TEST_ARGS"\n')
    engine.chmod(0o755)
    models = work / "models"
    models.mkdir()
    for name in ("sdxl_lightning_4step_unet.safetensors", "sdxl_lightning_1step_x0_unet.safetensors",
                 "sd_turbo.safetensors", "wan2.1_t2v_1.3B_fp16.safetensors", "wan2.1_i2v_480p_14B_fp16.safetensors",
                 "wan2.2_t2v_low_noise_14B_fp16.safetensors", "wan2.2_t2v_high_noise_14B_fp16.safetensors",
                 "Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf", "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
                 "Wan2.1-T2V-14B-lightx2v-4step.gguf", "ltx-2.3-22b-distilled.safetensors", "in.png",
                 "qwen-image-Q8_0.gguf", "qwen_image_fp8_e4m3fn.safetensors", "Qwen-Image-Edit-2509-Q4_K_M.gguf",
                 "qwen-image-lightning-8steps.safetensors", "my-qwen-image-finetune.safetensors",
                 "flux-3-image-klein-BF16.gguf", "flux-3-image-klein-base-Q8_0.gguf", "sd3.5_large.safetensors",
                 "sd3.5_large_turbo.safetensors", "sd3.5_medium-Q8_0.gguf", "sd_xl_base_1.0.safetensors",
                 "dreamshaperXL.safetensors", "z_image_bf16.safetensors", "chroma1-hd-Q8_0.gguf",
                 "hunyuanvideo1.5_720p_t2v_fp16.safetensors", "wan2.2_s2v_14B_fp8_scaled.safetensors"):
        (models / name).write_bytes(b"\0" * 16)
    flux_parts = ["double_blocks.0.img_attn.qkv.weight", "guidance_in.in_layer.weight"]
    safetensors(models / "flux1-dev.safetensors", flux_parts)
    safetensors(models / "flux1-kontext-dev.safetensors", flux_parts)
    safetensors(models / "flux-dev-de-distill.safetensors", flux_parts[:1])
    safetensors(models / "my-flux-dev-finetune.safetensors", flux_parts[:1])
    args_file = work / "args"
    env = dict(os.environ, KERA_HOME=str(work / "home"), KERA_TEST_ARGS=str(args_file),
               LD_LIBRARY_PATH=str(kera_src.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    failures = []

    def run(cmd, *rest):
        m = lambda n: str(models / n)  # noqa: E731
        argv = [str(kera), cmd]
        for w in rest:
            argv.append(m(w) if (models / w).exists() else w)
        if args_file.exists():
            args_file.unlink()
        r = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env)
        got = args_file.read_bytes().decode().split("\0")[:-1] if args_file.exists() else []
        return got, r.stderr

    def value(got, flag):
        return got[got.index(flag) + 1] if flag in got else None

    def check(ok, what, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what + (f": {detail}" if detail else ""))

    got, err = run("image", "-m", "sdxl_lightning_4step_unet.safetensors", "-p", "a cat")
    want = {"--steps": "4", "--cfg-scale": "1.0", "--sampling-method": "euler", "--scheduler": "sgm_uniform",
            "-W": "1024", "-H": "1024"}
    check(all(value(got, k) == v for k, v in want.items()) and "SDXL-Lightning" in err,
          "SDXL-Lightning: its card's 4 steps, no CFG, Euler with trailing steps, 1024x1024", f"{got} {err}")
    got, err = run("image", "-m", "sdxl_lightning_4step_unet.safetensors", "-p", "a cat", "-i", "in.png")
    check("--steps" not in got and "-W" not in got and value(got, "--cfg-scale") == "1.0",
          "from a picture: no steps and no size added (the engine would ignore the picture), CFG still the card's",
          str(got))
    got, err = run("image", "-m", "sdxl_lightning_1step_x0_unet.safetensors", "-p", "a cat")
    check("--steps" not in got and "--cfg-scale" not in got and "x0" in err,
          "SDXL-Lightning's 1-step x0 file: nothing added, and the user is told the engine can't run it as made",
          f"{got} {err}")
    got, err = run("image", "-m", "sd_turbo.safetensors", "-p", "a cat")
    check(value(got, "--steps") == "1" and value(got, "--cfg-scale") == "1.0" and "-W" not in got,
          "SD-Turbo: 1 step, no CFG, at the engine's own 512x512", str(got))
    got, err = run("image", "-m", "sd_turbo.safetensors", "-p", "a cat", "--cfg-scale", "2")
    check(got.count("--cfg-scale") == 1 and value(got, "--cfg-scale") == "2" and value(got, "--steps") == "1",
          "the user's own CFG wins", str(got))
    got, err = run("image", "--diffusion-model", "flux1-dev.safetensors", "-p", "a cat")
    check(value(got, "--cfg-scale") == "1.0" and "--guidance" not in got and value(got, "-W") == "1024" and
          value(got, "--steps") == "50",
          "FLUX.1-dev (guidance-distilled by its weights): no CFG, its own guidance, 1024x1024, its card's 50 steps",
          str(got))
    got, err = run("image", "--diffusion-model", "flux1-kontext-dev.safetensors", "-p", "a cat")
    check(value(got, "--guidance") == "2.5" and "Kontext" in err, "FLUX.1-Kontext-dev: its card's guidance 2.5",
          f"{got} {err}")
    got, err = run("image", "--diffusion-model", "flux-dev-de-distill.safetensors", "-p", "a cat")
    check(value(got, "--cfg-scale") == "3.5" and "-W" not in got and value(got, "--steps") == "28" and
          value(got, "-n") == "bad photo", "a de-distilled FLUX dev (no guidance weights): CFG kept on, at its maker's "
          "3.5, with its 28 steps and its negative prompt", str(got))
    got, err = run("image", "--diffusion-model", "my-flux-dev-finetune.safetensors", "-p", "a cat")
    check(not any(f in got for f in ("--cfg-scale", "--steps", "--guidance", "-W")),
          "an unknown FLUX dev without guidance weights: the engine's own settings", str(got))

    got, err = run("video", "--diffusion-model", "wan2.1_t2v_1.3B_fp16.safetensors", "-p", "a cat")
    check(value(got, "--cfg-scale") == "6.0" and value(got, "-n") == WAN_NEGATIVE and value(got, "--flow-shift") == "8"
          and value(got, "--steps") == "50", "Wan 2.1 T2V-1.3B: its makers' 50 steps, the README's guidance 6 and shift 8 "
          "for the 1.3B, and Wan's own default negative prompt", f"{got} {err}")
    got, err = run("video", "--diffusion-model", "wan2.1_t2v_1.3B_fp16.safetensors", "-p", "a cat", "-n", "blurry")
    check(got.count("-n") == 1 and value(got, "-n") == "blurry", "the user's own negative prompt wins", str(got))
    got, err = run("video", "--diffusion-model", "wan2.1_i2v_480p_14B_fp16.safetensors", "-p", "a cat", "-i",
                   "in.png", "-W", "832", "-H", "480")
    check(value(got, "--flow-shift") == "3" and value(got, "--cfg-scale") == "5.0" and value(got, "--steps") == "40",
          "Wan 2.1 I2V at 832x480: flow shift 3 and 40 steps, as its generate.py does (with its starting picture)",
          str(got))
    got, err = run("video", "--diffusion-model", "wan2.2_t2v_low_noise_14B_fp16.safetensors",
                   "--high-noise-diffusion-model", "wan2.2_t2v_high_noise_14B_fp16.safetensors", "-p", "a cat")
    check(value(got, "--cfg-scale") == "3.0" and value(got, "--high-noise-cfg-scale") == "4" and
          value(got, "--flow-shift") == "12" and "--moe-boundary" not in got and value(got, "--steps") == "40",
          "Wan 2.2 T2V-A14B: CFG 3 and 4 (high noise), flow shift 12, 40 steps", str(got))
    got, err = run("video", "--diffusion-model", "Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf", "--high-noise-diffusion-model",
                   "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf", "-p", "a cat", "-i", "in.png")
    check(value(got, "--cfg-scale") == "3.5" and value(got, "--high-noise-cfg-scale") == "3.5" and
          value(got, "--moe-boundary") == "0.9" and "--flow-shift" not in got,
          "Wan 2.2 I2V-A14B: CFG 3.5 for both experts, boundary 0.9", str(got))
    got, err = run("video", "--diffusion-model", "Wan2.1-T2V-14B-lightx2v-4step.gguf", "-p", "a cat")
    check(not any(f in got for f in ("--cfg-scale", "-n", "--steps", "--flow-shift")),
          "a step-distilled Wan: nothing added (its settings are its own)", str(got))
    got, err = run("video", "--diffusion-model", "ltx-2.3-22b-distilled.safetensors", "-p", "a cat")
    check(value(got, "--steps") == "8" and value(got, "--cfg-scale") == "1.0" and value(got, "--fps") == "24" and
          "-n" not in got and "--extra-sample-args" not in got,
          "LTX 2.3 distilled: its card's 8 steps without CFG, 24 frames a second", str(got))
    got, err = run("video", "--diffusion-model", "ltx-2.5-22b-dev-transformer-bf16.safetensors", "-p", "a cat")
    check(value(got, "--steps") == "20" and value(got, "--cfg-scale") == "3.0" and value(got, "--fps") == "24" and
          "-n" not in got and "--extra-sample-args" not in got,
          "LTX 2.5 dev: 20 steps, CFG 3, 24 frames a second; no negative prompt or guider added by itself", str(got))
    got, err = run("video", "--diffusion-model", "ltx-2.3-22b-dev.safetensors", "-p", "a cat", "--fps", "30")
    check(value(got, "--steps") == "30" and got.count("--fps") == 1 and value(got, "--fps") == "30",
          "LTX 2.3 dev: 30 steps, and the user's own frame rate wins", str(got))
    got, err = run("video", "--diffusion-model", "Wan2.1-T2V-1.3B-Q8_0.gguf", "-p", "a cat")
    check("--fps" not in got, "Wan: the engine's own 16 frames a second, nothing added", str(got))

    # the image engine's other recipes, with their makers' step counts, by the makers' file names (D-036)
    got, err = run("image", "--diffusion-model", "qwen-image-Q8_0.gguf", "-p", "a cat")
    check(value(got, "--steps") == "50" and value(got, "--cfg-scale") == "4.0" and value(got, "-W") == "1024" and
          "--flow-shift" not in got and "Qwen-Image" in err,
          "Qwen-Image: its card's 50 steps and CFG 4 at 1024x1024, whatever the file's precision", f"{got} {err}")
    got, err = run("image", "--diffusion-model", "qwen_image_fp8_e4m3fn.safetensors", "-p", "a cat", "--steps", "30")
    check(got.count("--steps") == 1 and value(got, "--steps") == "30" and value(got, "--cfg-scale") == "4.0",
          "the same model as an 8-bit safetensors file, and the user's own steps win", str(got))
    got, err = run("image", "--diffusion-model", "Qwen-Image-Edit-2509-Q4_K_M.gguf", "-p", "make it red", "-r", "in.png")
    check(value(got, "--steps") == "40" and "-W" not in got,
          "Qwen-Image-Edit-2509: its card's 40 steps with a picture too (it is made for editing), the picture's size",
          str(got))
    for name, what in (("qwen-image-lightning-8steps.safetensors", "a speed-up variant"),
                       ("my-qwen-image-finetune.safetensors", "a fine-tune under another name"),
                       ("sd3.5_large_turbo.safetensors", "a turbo model with no entry of its own"),
                       ("dreamshaperXL.safetensors", "a community model")):
        got, err = run("image", "--diffusion-model", name, "-p", "a cat")
        check(not any(f in got for f in ("--cfg-scale", "--steps", "-W", "--flow-shift")),
              f"{what} ({name}): the engine's own settings", str(got))
    got, err = run("image", "--diffusion-model", "flux-3-image-klein-BF16.gguf", "-p", "a cat")
    check(value(got, "--steps") == "16" and value(got, "--cfg-scale") == "1.0" and value(got, "-W") == "1024",
          "FLUX 3 klein: the engine's recipe (16 steps, no CFG)", str(got))
    got, err = run("image", "--diffusion-model", "flux-3-image-klein-base-Q8_0.gguf", "-p", "a cat")
    check(value(got, "--steps") == "50" and value(got, "--cfg-scale") == "4.0",
          "FLUX 3 klein base: its makers' 50 steps with CFG 4", str(got))
    got, err = run("image", "-m", "sd3.5_large.safetensors", "-p", "a cat")
    check(value(got, "--steps") == "28" and value(got, "--cfg-scale") == "3.5", "SD 3.5 Large: 28 steps, CFG 3.5",
          str(got))
    got, err = run("image", "--diffusion-model", "sd3.5_medium-Q8_0.gguf", "-p", "a cat")
    check(value(got, "--steps") == "40" and value(got, "--cfg-scale") == "4.5", "SD 3.5 Medium: 40 steps, CFG 4.5",
          str(got))
    got, err = run("image", "-m", "sd_xl_base_1.0.safetensors", "-p", "a cat")
    check(value(got, "--steps") == "50" and value(got, "--cfg-scale") == "5.0" and value(got, "-W") == "1024",
          "SDXL base 1.0: 50 steps, guidance 5, 1024x1024", str(got))
    got, err = run("image", "--diffusion-model", "z_image_bf16.safetensors", "-p", "a cat")
    check(value(got, "--steps") == "50" and value(got, "--cfg-scale") == "4.0" and value(got, "-W") == "720" and
          value(got, "-H") == "1280", "Z-Image: 50 steps, CFG 4, its card's 720x1280", str(got))
    got, err = run("image", "--diffusion-model", "chroma1-hd-Q8_0.gguf", "-p", "a cat")
    check(value(got, "--steps") == "40" and value(got, "--cfg-scale") == "3.0", "Chroma1-HD: 40 steps, CFG 3", str(got))
    got, err = run("video", "--diffusion-model", "hunyuanvideo1.5_720p_t2v_fp16.safetensors", "-p", "a cat")
    check(value(got, "--steps") == "50" and value(got, "--cfg-scale") == "6.0" and value(got, "--flow-shift") == "9"
          and "-W" not in got, "HunyuanVideo 1.5 720p T2V: 50 steps, CFG 6, flow shift 9 (the size stays the user's)",
          str(got))
    got, err = run("video", "--diffusion-model", "wan2.2_s2v_14B_fp8_scaled.safetensors", "-p", "a cat", "-i", "in.png")
    check(value(got, "--steps") == "40" and value(got, "--cfg-scale") == "4.5" and value(got, "--flow-shift") == "3",
          "Wan 2.2 S2V-14B: its config's 40 steps, guidance 4.5, shift 3", str(got))

    if failures:
        print(f"\nFAIL ({work}):\n  " + "\n  ".join(failures))
        sys.exit(1)
    shutil.rmtree(work, ignore_errors=True)
    print("\nPASS: picture and video models get their makers' settings, only where the command line sets none")


if __name__ == "__main__":
    main()
