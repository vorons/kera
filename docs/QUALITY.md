# Quality: Kera gives the model's own output

Kera never changes how a model responds; it only makes it faster (rule number one, D-033). It has no modes that
trade quality for speed, not even ones you have to turn on: the three opt-in lossy modes this page used to list were
removed on 9 Nov 2026 (D-034; what they measured is in `docs/NEGATIVE_RESULTS.md` N-017).

What a speed-up may change: nothing, or the last bits of a sum done in another order (rounding), and only after its
output has been checked against the model's own reference code. This page keeps those checks.

## Checked against stock llama.cpp: same answers, same memory (9 Nov 2026)

The owner asked for proof that nothing in Kera makes a model hallucinate or forget more. Measured in the cloud (4 CPU
cores) against stock llama.cpp b11619 built the same way, with greedy decoding so any difference shows.

| Check | Result |
| --- | --- |
| Kera as it ships (guessing ahead, saved chats, the card's sampling) vs stock `llama-server`: Qwen3-0.6B, the same 899-token prompt, 150 words | **Identical**: every word, every log-probability (difference 0.0) |
| Long-document recall: Qwen2.5-1.5B, a 5,328-token text with 5 facts at 8 to 93% depth; 5 questions and a 6-turn chat (prompts up to 5,542 tokens) | **11/11 answers identical** to stock, word for word, and all right |
| A chat restored from the SSD after a restart (5,541 of 5,542 tokens from the SSD; and a thinking model, 3 turns) | **The same answers** as without the restart; equal to stock continuing in one process (difference 0.0) |
| A saved chat with flipped bytes, cut in half, zeroed or empty (6 kinds of damage) | **All caught**: the chat is read again, the same answer, the file removed |
| A hybrid (Mamba) chat changed after its last checkpoint; a sliding-window chat | Only a checkpoint before the change is used; equal to stock's own cache (difference 0.0) |
| A prompt longer than the context; an answer reaching the context's end | Refused and stopped exactly as stock (nothing dropped from earlier in the chat) |
| A saved chat under another context size on a LongRoPE model, with a setting from an `LLAMA_ARG_*` variable, or after another adapter scale | **Was wrong** (the restored chat's words differed from a fresh read); **fixed** (D-033): each now reads the chat again and gives stock's answer exactly |
| The card's sampling with `--temp` (or `--top-p`) on the command line; a fine-tune of the same design; a file with its own sampling | **Was wrong** (the card's values replaced the user's and the file's); **fixed** (D-033): `tests/integration/test_model_defaults.py`, 14/14 with Qwen3 hybrid and 2507 Instruct |
| 20 short factual and reasoning questions, greedy, Kera as it ships vs stock: Qwen2.5-1.5B, Qwen3-0.6B thinking on and off | **60/60 identical**, every log-probability, thinking text included; guessing ahead was active (86 of 144 guesses taken on one prompt) |
| Guessing ahead with sampling (temperature 0.8, a fixed seed) | Each word is still drawn from the model's own probabilities (as stock llama.cpp does when it guesses); with a fixed seed the exact text can differ from a run without guessing, since the random numbers are used in another order |
| Pictures through a vision model (SmolVLM-256M, its own reader found by Kera) vs stock with `--mmproj` | **21/21 identical** answers and log-probabilities (8 drawn pictures, a 4-turn chat, Anthropic's API) |
| Models from Ollama's store | **Were wrong:** run without their Modelfile, and in ChatML when the file has no template (Gemma 3 270M: 2/24 right, against 23/24 in its own format); **fixed** (D-033): `tests/integration/test_ollama_models.py` |
| A lone image reader beside a model that isn't its own | **Was wrong** (a text model described pictures it can't see); **fixed**: taken only on evidence it is the model's |
| Word probabilities (`logprobs`) with guessing ahead | **Were wrong** (1.0 for every guessed word); **fixed** with llama.cpp's own later fix |
| `kera serve` with no options against llama.cpp's `llama-server` with no options, the same build, SmolVLM-256M Q8_0: a chat question and a list the model goes on copying (now in CI, `tests/integration/test_stock_defaults.py`) | **Identical**: every word and log-probability (difference 0), 24 of 24 guessed words included |
| Chats saved and restored, and several chats at once, with the graphics chip's batched memory copy forced on (`KERA_IO_SPAN_TEST=1`) on five tiny designs (dense, DeepSeek-style, GLM sparse attention, two hybrids) | **The same answers** as without it, and as a fresh read |
| Picture and video models' settings: SDXL-Lightning, SD-Turbo, FLUX dev (distilled, de-distilled, Kontext, unknown), Wan 2.1 and 2.2, LTX 2.3 (`tests/integration/test_picture_settings.py`, 16 cases) | Each gets its makers' settings only where the command line sets none; a picture to start from keeps the user's steps; speed-up variants and unknown models get nothing |
| **On a graphics chip** (the owner's DGX Spark, NVIDIA GB10, CUDA 13, 9 Nov): `kera serve` with no options against `llama-server` with no options, SmolVLM-256M Q8_0 | A chat question: **identical** (32 words, log-probabilities equal). A list the model copies: **the same 96 words**, 24 of 24 guessed words included; their log-probabilities within 0.042, where the chip checks guessed words in a batch and rounds differently from one at a time |
| On the GB10: chats saved and restored (five designs and Qwen3-0.6B), several chats at once (three designs), images through a vision model, pictures | **All the same answers** as a fresh read and as one chat alone: the chip's own batched memory copy (`llama_io_pieces_transfer`), which the CPU only reaches when forced, gives the same memory |
| On the GB10: the tiny designs against the reference code, with the stricter tie rule (9 Nov) | 5 of 6 pass. Nemotron's latent-MoE design picks another word at one of 24 steps, where the reference's top two are 1.3 points apart; every choice inside the model was the CPU's, and the log-probabilities are within the chip's limit (0.12 of 0.25): the chip's own rounding, which Kera doesn't change |
| On the GB10: guessing ahead on the tiny GLM design with its MTP head (exact maths) | llama.cpp's fixed guess length: identical. Kera's own choice: from word 239 of 495 another word, where the two were 0.282 and 0.279 likely (0.3 points apart). Checking a guess computes each word in a batch of two even when the guess is wrong (here 0 of 384 were right), and on the chip that rounds differently; a random tiny model's routing and sparse attention carry it to a near tie. On the CPU with exact maths the same test is identical (CI) |
| Guessing ahead with penalties, DRY and a logit bias, against `--no-auto-guess` (Qwen3-0.6B Q8_0, this machine's `-march=native` build, 10 Nov; N-019) | The sampler ends in the same state as writing word by word (read in the code, checked by hand at each split). **Three replies still differ, and not at near ties** (candidates 4.2 and 2.0 points apart): on this processor an 8-bit file's own numbers move by up to 0.3 nats with the number of words computed at once, with guessing off too. A 32-bit copy of the weights doesn't. The owner's choice is in `docs/STATUS.md` |
| **Music at ACE-Step's own settings (0.1.4, D-036), on the owner's DGX Spark (GB10, CUDA; 10 Nov).** One folder with XL SFT (BF16 and Q8_0), the base and turbo models (Q8_0) and the 4B (BF16) and 0.6B planners; three sets of lyrics, two seeds, 45 s songs; the words Whisper (medium.en) hears, counted with `tools/compare/lyrics.py` | **As 0.1.4 runs by itself** (it takes XL SFT BF16 and the 4B planner for 131 GB of graphics memory, 50 steps, guidance 7, shift 3): **288 of 302 words, 95%.** The same models with guidance and shift put back to 1: 279, 92%. **What 0.1.3 took by itself** in that folder (the base model and the 0.6B planner, the first by name, guidance 1, shift 1): 228, 75% (71% with the words it added counted). An earlier session's measurement there (9 Nov, the settings given by hand at 64 steps): 97% against 92% |
| ACE-Step's correction step (DCW) for turbo models, on the Spark: the same nine songs (three lyrics, three seeds, the same planner codes) with it on, as 0.1.4 runs turbo, and off | On: 379 of 453 words, 84%. Off: 369, 81%. Not worse for the words; single songs vary far more (33% to 98%) than the two settings differ |
| 0.1.4's code on the Spark's chip (10 Nov), rebuilt there with every engine on CUDA | Unit tests 20/20; the chat defaults test 21/21 with Qwen's own Qwen2.5-0.5B file (its makers' values from the list); the picture settings test; `kera serve` against `llama-server` with no options: the same 32 and 96 words, 24 of 24 guessed, log-probabilities within 0.042 (three runs); saved chats, several chats at once, vision and pictures pass. Two tests failed on a first run with the driver's "out of memory" while 30 GB of models were downloading into the shared memory, and one song the same way; all passed when run again |
| A chat model's own default sampling from the built-in list of 680 models (0.1.4): Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct Q4_K_M, Qwen's own files, which carry no sampling of their own (the second in CI, `tests/integration/test_model_defaults.py`) | With no options: temperature 0.7, top_p 0.8, top_k 20, repetition penalty 1.1, its makers' `generation_config.json`; min_p stays llama.cpp's. A request's temperature (0.2) and `--temp 0.3` win; `--no-model-defaults` gives llama.cpp's own 0.8 / 0.95 / 40; the same weights under another name, and a file with sampling of its own, get nothing from the list |
| Picture and video models' settings from the image engine's recipes with their makers' step counts (0.1.4; `tests/integration/test_picture_settings.py`, 36 cases) | Qwen-Image, Qwen-Image-Edit-2509, FLUX 3 klein and klein base, SD 3.5 Large and Medium, SDXL base, Z-Image, Chroma1-HD, HunyuanVideo 1.5, Wan 2.2 S2V each get theirs, whatever the file's precision; a speed-up variant, a fine-tune under another name, a turbo model with no entry and a community model get nothing |

The first row needed a clean build: this machine's earlier build had CPU kernels from the compiler cache, compiled for
another processor, which alone made Kera and stock differ by rounding from the first word (N-018). Downloads aren't
affected.

## Kera against other engines

The owner asked whether a model run through Ollama gives better or more accurate answers than through Kera
(7 Nov 2026). Measured in the cloud (4 CPU cores) the same day.

**Answers** (`tools/compare/accuracy.py`, `bench/results/cloud/2026-11-07-accuracy-kera-vs-ollama.json`): one file,
Qwen2.5-1.5B-Instruct Q4_K_M, loaded by Kera (as it ships, guessing on), by Kera with `--no-auto-guess`, and by Ollama
0.40.15 (imported with `FROM` the same file; it runs llama.cpp's server inside). Greedy, through each one's OpenAI API,
the first 100 GSM8K test problems and the first 100 ARC-Challenge test questions, each scored right or wrong.

| | GSM8K (maths) | ARC-Challenge (science) | Total | Time |
| --- | --- | --- | --- | --- |
| Kera | 75 | 74 | **149** | 21 min |
| Kera, guessing off | 74 | 74 | 148 | 45 min (shared the CPU with the picture test) |
| Ollama 0.40.15 | 73 | 74 | 147 | 25 min |

- The same accuracy: Kera was right on 11 questions Ollama got wrong, Ollama on 9 Kera got wrong.
- Both read the same prompt, token for token (the model's own chat template; 123 tokens for the first problem).
  Their replies were word for word the same on 96 of 200 and gave the same answer on 170: they part at a word the
  model rates almost equal to another, where the two builds of llama.cpp's maths round differently (different
  versions and kernels), and a long worked answer has many such words. Kera with and without guessing gave the same
  reply word for word on 171 of 200 (all 29 differences are in worked maths answers, where a near tie took the other
  word) and the same final answer on 194: six answers changed, one from wrong to right and five from one wrong number
  to another (75 right against 74 on the maths problems). Measured with the build later found to have stale CPU
  kernels (N-018); to be measured again.

**Pictures** (`bench/results/cloud/2026-11-07-images-kera-vs-diffusers.json` and `.jpg`): SD-Turbo, 6 prompts, 512x512,
one step, the same seed and the same starting noise (`--rng cpu`, PyTorch's), through `kera image` and through
Hugging Face diffusers, the model's reference code. Kera's image engine is stable-diffusion.cpp, unchanged.

| | Alike with the reference (PSNR) | Fits the prompt (CLIP score) |
| --- | --- | --- |
| Reference (diffusers, 16-bit weights) | | 33.56 |
| Kera, `sd_turbo-f16.gguf` (16-bit) | **46.7 dB** (looks identical) | **33.51** |
| Kera, `sd_turbo-f16-q8_0.gguf` (8-bit, the usual download) | 27.2 dB (fine texture differs) | 33.08 |
| For scale: the reference with another seed | 11.8 dB (another picture) | |

The 8-bit file's small loss is the file's, not Kera's: with the same weights the pictures match.

**0.1.2's faster engines** (D-030, `bench/results/cloud/2026-11-08-engines-tiled-multiply.json`): the image and music
engines now use ggml's tiled matrix multiply on the CPU (pictures 1.85 times as fast, songs 1.6 times). The same six
pictures, against the same reference:

| | Before (0.1.1) | After (0.1.2) |
| --- | --- | --- |
| `sd_turbo-f16.gguf` (16-bit), alike with the reference | 46.70 dB | **46.92 dB** |
| `sd_turbo-f16-q8_0.gguf` (8-bit), alike with the reference | 27.25 dB | 27.21 dB |
| 0.1.2 against 0.1.1, 16-bit | | 56.0 dB (the same picture, rounding only) |

Music has no reference run here. The song made from the same codes and seed by both builds: 2.1 dB apart in its
spectrum, against 21.5 dB for another seed, the same loudness (-16.1 dB). Rounding only, at the same precision.
