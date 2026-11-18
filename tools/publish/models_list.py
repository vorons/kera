"""Writes docs/MODELS.md: the models Kera runs, and the ones it knows the makers' own settings for.

  python tools/publish/models_list.py            # rewrite docs/MODELS.md
  python tools/publish/models_list.py --check    # fail if it is out of date (CI)

The picture and video table comes from app/image_recipes.inc, the chat list from src/tune/model_cards.inc; the rest is
written here by hand from the code it describes (app/image_cli.cpp, app/media_cli.cpp, app/media3d_cli.cpp,
src/tune/model_defaults.cpp). The Hugging Face page carries the same file (tools/publish/huggingface.py).
"""
import argparse
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "MODELS.md"

HEAD = """# Models Kera runs

Kera runs a model on your own computer from its file. It doesn't ship or download models: you get the file (most
come from Hugging Face) and name it, or put it in Kera's folder for that kind of model.

Two things are listed here. **Runs**: the kinds of model each of Kera's engines can run. **Makers' settings known**:
the models Kera recognises and runs the way their makers run them where you set nothing (steps, guidance, sampling).
A model that isn't recognised still runs, with its engine's own defaults; what you set yourself always wins.

"Run here" marks what was actually run with Kera in its tests or on its owner's machines. The rest is recognised
from its makers' published settings and hasn't been run by Kera's authors.

## Chat (`kera serve`, `kera cli`, the app)

**Runs:** any chat model in GGUF form that llama.cpp (build b12500) runs, with the model's own chat template; models
from Ollama's store with their Modelfile; models that see images, with their image reader (mmproj) found by itself;
MoE models larger than memory, with their experts streamed from the SSD.

**Makers' settings known:**

- From the model's card, read by hand, with separate values for thinking, no thinking and tool calls: NVIDIA
  Nemotron 3 Nano 30B-A3B (run here), Qwen3 with its thinking switch and the 2507 Instruct and Thinking releases
  (run here: Qwen3-0.6B), Qwen3-Coder.
- From the file itself: any GGUF file that carries its makers' sampling (`general.sampling.*`), as newer files do.
- From Kera's built-in list, for a file that carries none: the @CHAT_COUNT@ models at the end of this page (run here:
  Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct). The values are each model's own `generation_config.json`,
  collected from Hugging Face on @CHAT_DATE@.

Not in the list: Llama 3.1 8B, Gemma 3 and Mistral's models, whose makers' pages are locked and have no identical
public copy. A recent file of those usually carries its settings itself.

## Pictures (`kera image`, `kera serve` with an image model)

**Runs:** the models stable-diffusion.cpp runs, as GGUF or safetensors files: Stable Diffusion 1.x, 2.x, XL, 3 and
3.5, FLUX.1, FLUX.2, FLUX 3, Qwen-Image, Z-Image, Chroma, HiDream and the others in the table below.

**Makers' settings known:**

| Model | Steps | Guidance | Notes |
| --- | --- | --- | --- |
| SD-Turbo, SDXL-Turbo | 1 | none | run here: SD-Turbo |
| SDXL-Lightning (1, 2, 4, 8 step) | the file's | none | Euler, trailing steps, 1024x1024 |
| FLUX.1-schnell | 4 | none | 1024x1024 |
| FLUX.1-dev | 50 | 3.5 (built in) | told apart from de-distilled models by its weights |
| FLUX.1-Kontext-dev | 28 | 2.5 | editing |
| FLUX.1-Fill-dev, FLUX.1-Canny-dev | 50 | 30 | |
| FLUX.1-Depth-dev | 30 | 10 | |
| FLUX.1-Krea-dev | 28 | 4.5 | |
| FLUX.1-dev de-distilled | 28 | CFG 3.5 | with its maker's negative prompt |
| FLUX.2-dev | 50 | 4 | |
| Z-Image-Turbo | 9 | none | 1024x1024 |
@IMAGE_ROWS@

A speed-up variant (Lightning, Turbo or distilled versions without a row of their own), a fine-tune under another
name or a community model keeps the engine's own settings.

## Video (`kera video`)

**Runs:** the video models stable-diffusion.cpp runs.

**Makers' settings known:**

| Model | Steps | Guidance | Flow shift |
| --- | --- | --- | --- |
| Wan 2.1 T2V-14B | 50 | 5 | the engine's |
| Wan 2.1 T2V-1.3B | 50 | 6 | 8 |
| Wan 2.1 I2V-14B | 40 | 5 | 3 at 480p |
| Wan 2.1 FLF2V-14B, VACE | 50 | 5 | 16 |
| Wan 2.2 T2V-A14B | 40 | 3 and 4 (high noise) | 12 |
| Wan 2.2 I2V-A14B | 40 | 3.5 | the engine's |
| Wan 2.2 TI2V-5B | 50 | 5 | the engine's |
| Wan 2.2 S2V-14B | 40 | 4.5 | 3 |
| LTX 2.3 and 2.5 | 30 | 3 | the engine's |
| LTX 2.3 and 2.5 distilled | 8 | none | the engine's |
@VIDEO_ROWS@

Wan models also get Wan's own default negative prompt. No video model has been run here: they are too big for the
machines Kera's tests run on.

## Music (`kera music`, `kera serve --music`)

**Runs:** ACE-Step 1.5 as GGUF files: the sound models (turbo and its variants, SFT, base, and the XL versions of
each), the planners (0.6B, 1.7B, 4B), its text encoder and VAE.

**Makers' settings known:** turbo models: 8 steps, shift 3, ACE-Step's correction step; SFT and base models: 50
steps, guidance 7, shift 3. With several models in the folder Kera takes the ones ACE-Step's guide recommends for
the computer's memory, at the highest precision that fits. Run here: turbo, SFT, base and XL SFT with the 0.6B and 4B
planners.

## Speech (`kera speak`)

**Runs:** Qwen3-TTS Base (ten languages, a voice copied from a short sample) and Pocket TTS, with llama.cpp's speech
tool. **Makers' settings:** the files carry them (Qwen3-TTS: temperature 0.9, top_k 50, top_p 1.0). Run here:
Qwen3-TTS 1.7B Base. Not supported: voices described in words (Qwen3-TTS VoiceDesign).

## 3D models (`kera 3d`)

**Runs:** TRELLIS.2 (a 3D model from a picture, or from text through `kera image`), with BiRefNet to cut out the
background. **Makers' settings:** TRELLIS.2's own 12 steps and guidance. Not run here with real weights.

## Sound effects (`kera sfx`)

**Runs:** MOSS-SoundEffect v2 and Stable Audio 3 Small SFX. **Makers' settings:** the models' own steps (100 and 8)
and lengths; the full-precision file first when the folder has several. Run on the owner's DGX Spark:
MOSS-SoundEffect v2.

## The chat list

The models whose makers' default sampling Kera carries, by organisation. A file is matched by the model's name and
its shape together, so a fine-tune under another name isn't.

@CHAT_LIST@
"""


def num(v):
    return re.sub(r"\.0$", "", v)


def image_rows():
    images, videos = [], []
    for line in (ROOT / "app" / "image_recipes.inc").read_text(encoding="utf-8").splitlines():
        m = re.match(r'\s*\{ "([^"]+)", "[^"]*", (\d+), ([-\d.]+), ([-\d.]+), ([-\d.]+), (\d+), (\d+), .*, (true|false), (true|false) \},', line)
        if not m:
            continue
        family, steps, cfg, guidance, shift, w, h, _with_picture, video = m.groups()
        guide = "none" if cfg == "1" and guidance == "-1" else f"CFG {num(cfg)}" if guidance == "-1" else f"{num(guidance)} (built in)"
        if video == "true":
            videos.append(f"| {family} | {steps} | {guide.replace('CFG ', '')} | {num(shift) if shift != '-1' else 'the engine\'s'} |")
        else:
            notes = f"{w}x{h}" if w != "0" else ""
            images.append(f"| {family} | {steps} | {guide} | {notes} |")
    return "\n".join(images), "\n".join(videos)


def chat_list():
    text = (ROOT / "src" / "tune" / "model_cards.inc").read_text(encoding="utf-8")
    date = re.search(r"collected (\d{4}-\d{2}-\d{2})", text).group(1)
    by_org = collections.defaultdict(list)
    for repo in re.findall(r'"([^"/]+/[^"]+)" \},', text):
        org, name = repo.split("/", 1)
        by_org[org].append(name)
    out = []
    for org in sorted(by_org, key=lambda o: (-len(by_org[o]), o.lower())):
        out.append(f"**{org}** ({len(by_org[org])}): " + ", ".join(sorted(by_org[org], key=str.lower)))
    return "\n\n".join(out), sum(len(v) for v in by_org.values()), date


def page():
    images, videos = image_rows()
    chats, count, date = chat_list()
    return (HEAD.replace("@IMAGE_ROWS@", images).replace("@VIDEO_ROWS@", videos).replace("@CHAT_LIST@", chats)
            .replace("@CHAT_COUNT@", str(count)).replace("@CHAT_DATE@", date))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = page()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            sys.exit("docs/MODELS.md is out of date: run python tools/publish/models_list.py")
        print("docs/MODELS.md is up to date")
        return
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
