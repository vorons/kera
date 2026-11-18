---
license: apache-2.0
tags:
  - llama.cpp
  - gguf
  - local-ai
  - inference
  - mixture-of-experts
  - vulkan
  - rocm
  - metal
  - text-to-image
  - text-to-speech
  - text-to-audio
  - text-to-video
  - cuda
  - dgx-spark
---

# Kera

**A local AI engine and app, built on llama.cpp, that runs big models fast on your own computer.**

By Kera Labs. Version @VERSION@: it works and is measured on real hardware, but not every goal is reached
yet (below).

Kera runs the same GGUF model files as llama.cpp and Ollama, and gives the model's own answers: it has no modes that
trade quality for speed. Checking several guessed words at once can round a near tie the other way (seen on the
graphics chip and on the CPU), as another batch size would, so a reply may read differently from stock llama.cpp's
(`docs/QUALITY.md` in the source). What it adds:

- **It writes faster.** It guesses the next few words ahead by itself where that pays (copying from the chat, the
  model's own MTP head), measures whether each guesser pays, and drops the ones that don't.
- **It picks the fastest back-end for each message.** On an AMD machine with the optional ROCm add-on,
  `kera serve --backend auto` measures Vulkan and ROCm on each model and sends every message to the one that
  finishes it sooner: chats to Vulkan, long documents (and their follow-ups) to ROCm.
- **Chats survive restarts.** Each chat's state is kept on the SSD, so a follow-up or a restart doesn't read the
  whole conversation again.
- **Models bigger than memory.** A mixture-of-experts model that doesn't fit is streamed from the SSD, expert by
  expert, with the experts it uses most kept in memory.
- **Several people at once.** `kera serve --multi-user` reads a new message before the rest of someone's long
  document, keeps everyone writing while a long prompt is read, and reads a start several chats share only once.
- **It explains your computer.** The first time it starts, it measures your drive, memory and graphics chip in about
  30 seconds and says in plain words what you can run.
- **OpenAI- and Anthropic-compatible** server (`kera serve`), so existing chat apps and agents work with it.
- **Pictures, music, speech and video, on your computer too** (0.1.1). Vision models get their image reader by
  themselves; `kera image` and `kera serve` run image models (SD, SDXL, SD3.5, FLUX, Qwen-Image, Z-Image) through
  stable-diffusion.cpp, and `kera video` video models (Wan, LTX-Video); `kera music` makes songs with ACE-Step 1.5
  through acestep.cpp, lyrics included; `kera speak` turns text into speech with Qwen3-TTS. Each engine comes in the
  download, and the models are yours to download.
- **3D models and sound effects; faster pictures and music** (0.1.2). `kera 3d` turns a picture (or a description,
  through a picture) into a textured 3D model (`.glb`) with TRELLIS.2 through trellis.cpp; `kera sfx` makes sound
  effects with MOSS-SoundEffect through audio.cpp. On the processor, pictures are made 1.85 times and songs 1.6 times
  as fast as in 0.1.1. The pictures were checked against the model's reference code (as close as before); the songs
  against 0.1.1's (the same song from the same codes; no reference code was run for music). The same seed gives the
  same picture and the same song. `--improve-prompt` (opt-in) lets a chat model rewrite a short description in more
  detail first, and shows both.
- **The model's own answers, the models' own settings** (0.1.3). Kera never changes how a model responds; it only
  makes things faster. At its defaults it answers exactly as stock llama.cpp does, and this is tested on every
  change. The modes that traded quality for speed are removed. Chat, picture and video models run with their makers'
  own settings where you set none (FLUX, Z-Image, SDXL-Lightning, Wan, LTX, Qwen3, Nemotron 3 Nano), and what you set
  yourself always wins.
- **Every model at its makers' own settings** (0.1.4). Music: ACE-Step's base and SFT models run at ACE-Step's own
  guidance and shift, and of a folder's models Kera takes the ones ACE-Step's guide recommends for the computer's
  memory (on a DGX Spark, Whisper heard 95% of a song's given words, against 75% for what 0.1.3 took by itself).
  Pictures and video: 53 more models get their makers' settings and step counts (Qwen-Image, FLUX 3 klein, Stable
  Diffusion 3.5, SDXL, Z-Image, Chroma, HunyuanVideo 1.5 and others). Chat: a file with no sampling settings of its
  own gets its makers' defaults from a built-in list of 680 models. Kera says when a full-precision copy of a model
  is next to the file you named.

## Measured

On a Minisforum mini PC (AMD Ryzen AI Max, Radeon 8060S graphics, Windows), with Nemotron 3 Nano 30B-A3B (Ollama's
own 22.8 GB file, so every engine ran the same weights), in October 2026 with an earlier version of Kera (preview 3).
That machine is no longer available, so these can't be measured again:

| | Kera, Vulkan | Kera, ROCm add-on | llama.cpp | Ollama |
| --- | --- | --- | --- | --- |
| First word in a chat | 0.36 s | **0.32 s** | 0.38 s | 1.1 s |
| First word after a restart | **0.79 s** | 0.87 s | 3.0 s | 3.2 s |
| A 95,000-word document: first word | 296 s | **138 s** | 303 s | 144 s |
| Follow-up question on it | 0.43 s | **0.33 s** | 0.41 s | 2.2 s |

Every engine answered greedy (temperature 0, top_k 1), with thinking off and answers of up to 256 tokens; Nemotron
3 Nano's own default (thinking on) wasn't measured. **Kera's everyday writing speed isn't shown:** it was measured
with a flawed method (each prompt was asked three times in a row on one server, and from the second time Kera's
guesser could copy its own earlier answer), which made Kera look faster than it is, and it couldn't be measured
again before the machine went. The waits for the first word above aren't affected by that.

![Kera against llama.cpp and Ollama as a document gets longer](images/long-documents.png)

Every number has its source, settings and date in `docs/BENCHMARKS.md` in the source.

### As accurate as the usual way

Speed doesn't cost accuracy. One model file (Qwen2.5-1.5B-Instruct) through Kera and through Ollama 0.40.15,
200 questions with known answers (100 school maths problems from GSM8K, 100 science questions from
ARC-Challenge), greedy: **Kera 149 right, Ollama 147**; both give the model the same prompt, token for token.
Pictures made with SD-Turbo through Kera and through the model's own reference code (Hugging Face diffusers),
with the same seed and weights, look identical (46.7 dB alike). Details: `docs/QUALITY.md` in the source.

## Supported models

Kera runs models from their files on your own computer; it ships none. The full list, with each model's settings,
is in [MODELS.md](MODELS.md).

| Kind | Runs | Run the way their makers run them, by themselves |
| --- | --- | --- |
| Chat | any GGUF chat model llama.cpp runs, models from Ollama's store, models that see images | Nemotron 3 Nano, Qwen3, Qwen3-Coder (from their cards); any file that carries its own settings; 680 more from a built-in list (Qwen2.5, DeepSeek, Llama 3.2, GLM, Nemotron and others) |
| Pictures | Stable Diffusion 1.x to 3.5, SDXL, FLUX.1, FLUX.2, FLUX 3, Qwen-Image, Z-Image, Chroma and more | SD-Turbo, SDXL-Lightning, FLUX.1 (schnell, dev, Kontext, Fill, Canny, Depth, Krea), FLUX.2, FLUX 3 klein, Qwen-Image and its edit models, Z-Image, SD 3 and 3.5, SDXL base, Chroma, and others |
| Video | Wan 2.1 and 2.2, LTX 2.3 and 2.5, HunyuanVideo 1.5 | all of those |
| Music | ACE-Step 1.5: turbo, SFT, base and their XL versions, with the 0.6B, 1.7B and 4B planners | all of those; with several in the folder, the ones ACE-Step recommends for your memory |
| Speech | Qwen3-TTS Base (ten languages, a voice copied from a sample), Pocket TTS | both (their files carry their settings) |
| 3D | TRELLIS.2, from a picture or from text | TRELLIS.2 |
| Sound effects | MOSS-SoundEffect v2, Stable Audio 3 Small SFX | both |

A model Kera doesn't recognise still runs, with its engine's own defaults. What you set yourself always wins.

## Downloads

| File | For |
| --- | --- |
| `downloads/kera-@VERSION@-windows-x64.zip` | Windows 10 and 11 (Vulkan: AMD, NVIDIA and Intel graphics, or the CPU) |
| `downloads/kera-@VERSION@-windows-x64-hip-addon.zip` | Optional: AMD's ROCm for Radeon 8060S (gfx1151), unzipped into the same folder |
| `downloads/kera-@VERSION@-linux-x64.tar.gz` | Linux (Vulkan or the CPU) |
| `downloads/kera-@VERSION@-macos-arm64.zip` | Mac with Apple silicon (Metal), macOS 13.3 or later |
| `downloads/kera-@SPARK_VERSION@-linux-arm64-cuda.tar.gz` | NVIDIA DGX Spark and other GB10 machines (Linux ARM64, CUDA 13) |
| `downloads/Kera-vs-Ollama.dmg` | A Mac app that races Kera against Ollama on your Mac, live, with one button |
| `source/kera-@VERSION@-source.tar.gz` | The source code |@SPARK_SOURCE@

@SPARK_NOTE@**Try it:** unzip, run `kera` (a page opens in your browser and explains your computer), then
`kera serve -m model.gguf` for chat apps (`http://127.0.0.1:8080`). `kera serve --help` lists Kera's options.

No model weights are included: Kera runs GGUF files you download (from Hugging Face, or the ones Ollama already
has), each under its own licence.

## Status

This is a preview. Kera's goals each have a gate measured on real hardware; the Windows machine above has passed
some of its measurements, and the rest (the Mac, NVIDIA machines and the bigger models) are still to be measured.
The Mac version builds and passes its tests, but hasn't been measured on a Mac yet.

## Licence

Kera's own code: Apache License 2.0, copyright Kera Labs (`LICENSE`). It is built on llama.cpp and ggml (MIT) and
bundles other open-source code under its own licences (`NOTICE`, and `THIRD_PARTY_LICENSES.txt` in each download).
