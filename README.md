<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand/png/kera-logo-on-dark-2000.png">
  <img alt="Kera" src="assets/brand/png/kera-logo-on-light-2000.png" width="360">
</picture>

**Kera runs AI models on your own computer as fast as its hardware allows,
including models too big to fit in its memory.**

You download the app and open it. It measures your computer by itself (how fast
the SSD, the memory and the graphics chip are), tells you which models will run
and how fast, and tunes its own settings. Nobody has to measure anything by hand
or edit a settings file.

## Status

Kera is being built. The current state, milestone by milestone, is in
[`docs/STATUS.md`](docs/STATUS.md). This page only quotes speeds that have been
measured; there are none to show yet.

- **Which models it runs**, and which it runs the way their makers run them by itself, is in
  [`docs/MODELS.md`](docs/MODELS.md): chat, pictures, video, music, speech, 3D and sound effects.
- **Windows comes first** (AMD Ryzen AI Max machines), then Mac, then NVIDIA.
- Downloads appear on this repo's **Releases** page: Windows (with an optional
  ROCm add-on), Linux, Mac, and the NVIDIA DGX Spark (Linux ARM64 with CUDA).

## How it is built

- The compute core is [llama.cpp](https://github.com/ggml-org/llama.cpp)
  (MIT licence), copied into `third_party/llama.cpp`.
- Kera adds the parts nobody has built together yet: a weight store that
  streams parts of the model from the SSD, smart caching and reading ahead,
  saved chat memory for faster replies, a controller that decides when
  guessing ahead pays, and a planner that predicts the best possible speed.
- Answers are the model's own: Kera has no modes that trade quality for
  speed. Checking several guessed words at once can round a near tie the
  other way (seen on the graphics chip and on the CPU), as another batch size
  would, so a reply may read differently from stock llama.cpp's. The checks
  are in [`docs/QUALITY.md`](docs/QUALITY.md).
- On an AMD machine with the optional ROCm add-on, `kera serve --backend auto`
  measures Vulkan and ROCm on each model once and sends each message to the
  one that finishes it sooner: chats to Vulkan, long documents (and their
  follow-ups) to ROCm. With memory for the model twice both stay loaded;
  otherwise Kera swaps them when that pays.
- Several people or agents can share one model: `kera serve --multi-user` reads
  a new message before the rest of someone's long document, keeps everyone
  writing while a long prompt is read, and reads a start that several chats
  share only once.
- Models that see images: `kera serve -m model.gguf` finds a vision model's
  image reader by itself (next to it, or in Ollama's store), so apps can send
  it pictures. The app lists your Ollama models by name.
- Models that make images: `kera image` makes a picture from text, and
  `kera serve -m image-model` serves one with OpenAI's images API (SD, SDXL,
  SD3.5, FLUX, Qwen-Image, Z-Image and more, through stable-diffusion.cpp).
  `kera video` makes short videos (Wan 2.1/2.2, LTX-Video) the same way.
- Music: `kera music -p "upbeat pop rock" --duration 60 -o song.mp3` writes the
  lyrics and makes the song (ACE-Step 1.5, through acestep.cpp), and
  `kera serve --music DIR` opens a page for making songs.
- Speech: `kera speak -m Qwen3-TTS-...gguf -p "Hello" -o hello.wav`, with
  llama.cpp's own speech models; the audio decoder is found by itself.
- `kera finetune` trains a small adapter (a LoRA) on your own examples.
- The full plan is in [`docs/KERA_BUILD_SPEC.md`](docs/KERA_BUILD_SPEC.md).

Build and test commands are in [`AGENTS.md`](AGENTS.md).

## Licence

Kera's own code: Apache License 2.0, copyright Kera Labs (see [`LICENSE`](LICENSE)). Third-party
code keeps its own licence (see [`NOTICE`](NOTICE)). Model weights are never
bundled; each model has its own licence.
