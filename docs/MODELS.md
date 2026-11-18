# Models Kera runs

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
- From Kera's built-in list, for a file that carries none: the 680 models at the end of this page (run here:
  Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct). The values are each model's own `generation_config.json`,
  collected from Hugging Face on 2026-11-10.

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
| Qwen-Image | 50 | CFG 4 | 1024x1024 |
| Qwen-Image-Edit | 50 | CFG 4 |  |
| Qwen-Image-Edit-2509 | 40 | CFG 4 |  |
| Qwen-Image-Edit-2511 | 40 | CFG 4 |  |
| Qwen-Image-Layered | 50 | CFG 4 | 640x640 |
| Qwen-Image-2.1 | 40 | none | 1024x1024 |
| Qwen-Image-2.1-Turbo | 8 | none | 1024x1024 |
| Z-Image | 50 | CFG 4 | 720x1280 |
| Stable Diffusion 3 Medium | 28 | CFG 7 | 1024x1024 |
| Stable Diffusion 3.5 Large | 28 | CFG 3.5 | 1024x1024 |
| Stable Diffusion 3.5 Medium | 40 | CFG 4.5 | 1024x1024 |
| Stable Diffusion XL base 1.0 | 50 | CFG 5 | 1024x1024 |
| Stable Diffusion 1.x | 50 | CFG 7.5 | 512x512 |
| Stable Diffusion 2.x (512) | 50 | CFG 7.5 | 512x512 |
| Stable Diffusion 2.x (768) | 50 | CFG 7.5 | 768x768 |
| Chroma1-HD | 40 | CFG 3 | 1024x1024 |
| Chroma1-Radiance | 50 | CFG 3.5 | 1024x1024 |
| FLUX.2 klein base 4B | 50 | CFG 4 | 1024x1024 |
| FLUX.2 klein base 9B | 50 | CFG 4 | 1024x1024 |
| FLUX.2 klein 4B | 4 | none | 1024x1024 |
| FLUX.2 klein 9B | 4 | none | 1024x1024 |
| FLUX 3 klein base | 50 | CFG 4 | 1024x1024 |
| FLUX 3 klein | 16 | none | 1024x1024 |
| ERNIE-Image | 50 | CFG 4 | 1024x1024 |
| ERNIE-Image-Turbo | 8 | none | 1024x1024 |
| LongCat-Image | 50 | CFG 4 | 1344x768 |
| LongCat-Image-Edit | 50 | CFG 4.5 |  |
| Ovis-Image-7B | 50 | CFG 5 | 1024x1024 |
| Krea 2 Raw | 52 | CFG 3.5 | 1024x1024 |
| Krea 2 Turbo | 8 | none | 1024x1024 |
| Fibo 2.0 dev | 50 | CFG 5 | 1024x1024 |
| Boogu-Image 0.1 Base | 50 | CFG 4 | 1024x1024 |
| Boogu-Image 0.1 Edit | 50 | CFG 5 | 1024x1024 |
| Flex.2-preview | 50 | 3.5 (built in) | 1024x1024 |
| Iris 3B | 100 | CFG 3 | 1024x1024 |
| MOSS-Image-Base | 50 | CFG 3 | 1024x1024 |
| MOSS-Image-SFT | 50 | CFG 3 | 1024x1024 |
| SeFi-Image 5B Base | 50 | CFG 4 |  |
| SenseNova-U1.5 8B MoT | 50 | CFG 4 | 1024x1024 |
| LLaDA-Image | 50 | CFG 5 | 1024x1024 |
| LLaDA-Image-Turbo | 4 | none | 1024x1024 |
| Lens | 20 | CFG 5 | 1024x1024 |
| Lens-Turbo | 4 | none | 1024x1024 |
| Mage-Flow 4B | 20 | CFG 5 | 1024x1024 |
| Mage-Flow-Turbo | 4 | none | 1024x1024 |
| Mage-Flow-Edit 4B | 30 | CFG 5 | 1024x1024 |
| PixArt-alpha 1024 | 20 | CFG 4.5 | 1024x1024 |
| PixArt-alpha 512 | 20 | CFG 4.5 | 512x512 |
| PixArt-Sigma 1024 | 20 | CFG 4.5 | 1024x1024 |

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
| HunyuanVideo 1.5 (720p T2V) | 50 | 6 | 9 |
| HunyuanVideo 1.5 (720p I2V) | 50 | 6 | 7 |
| LingBot-Video dense 1.3B | 40 | 3 | 3 |
| LingBot-Video MoE 30B-A3B | 40 | 3 | 3 |

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

**Qwen** (102): CodeQwen1.5-7B-Chat, Qwen-14B, Qwen-14B-Chat, Qwen-72B, Qwen-72B-Chat, Qwen-7B, Qwen-7B-Chat, Qwen-AgentWorld-35B-A3B, Qwen-Audio, Qwen-VL, Qwen-VL-Chat, Qwen1.5-0.5B-Chat, Qwen1.5-1.8B-Chat, Qwen1.5-32B-Chat, Qwen1.5-4B-Chat, Qwen1.5-72B-Chat, Qwen1.5-7B-Chat, Qwen1.5-MoE-A2.7B-Chat, Qwen2-0.5B-Instruct, Qwen2-1.5B-Instruct, Qwen2-57B-A14B-Instruct, Qwen2-72B-Instruct, Qwen2-7B-Instruct, Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-1.5B-Instruct-AWQ, Qwen2.5-14B-Instruct, Qwen2.5-14B-Instruct-1M, Qwen2.5-14B-Instruct-AWQ, Qwen2.5-14B-Instruct-GPTQ-Int4, Qwen2.5-32B-Instruct, Qwen2.5-32B-Instruct-AWQ, Qwen2.5-32B-Instruct-GPTQ-Int4, Qwen2.5-3B-Instruct, Qwen2.5-3B-Instruct-AWQ, Qwen2.5-72B-Instruct, Qwen2.5-72B-Instruct-AWQ, Qwen2.5-7B-Instruct, Qwen2.5-7B-Instruct-1M, Qwen2.5-7B-Instruct-AWQ, Qwen2.5-7B-Instruct-GPTQ-Int4, Qwen2.5-Coder-0.5B-Instruct, Qwen2.5-Coder-1.5B-Instruct, Qwen2.5-Coder-14B-Instruct, Qwen2.5-Coder-14B-Instruct-AWQ, Qwen2.5-Coder-32B-Instruct, Qwen2.5-Coder-32B-Instruct-AWQ, Qwen2.5-Coder-3B-Instruct, Qwen2.5-Coder-3B-Instruct-AWQ, Qwen2.5-Coder-7B-Instruct, Qwen2.5-Coder-7B-Instruct-AWQ, Qwen2.5-Coder-7B-Instruct-GPTQ-Int4, Qwen2.5-VL-32B-Instruct, Qwen3-14B, Qwen3-14B-AWQ, Qwen3-14B-FP8, Qwen3-235B-A22B, Qwen3-235B-A22B-Instruct-2507, Qwen3-235B-A22B-Instruct-2507-FP8, Qwen3-235B-A22B-Thinking-2507, Qwen3-235B-A22B-Thinking-2507-FP8, Qwen3-30B-A3B, Qwen3-30B-A3B-FP8, Qwen3-30B-A3B-GPTQ-Int4, Qwen3-30B-A3B-Instruct-2507, Qwen3-30B-A3B-Instruct-2507-FP8, Qwen3-30B-A3B-Thinking-2507, Qwen3-32B, Qwen3-32B-AWQ, Qwen3-32B-FP8, Qwen3-4B, Qwen3-4B-AWQ, Qwen3-4B-FP8, Qwen3-4B-Instruct-2507, Qwen3-4B-Instruct-2507-FP8, Qwen3-4B-Thinking-2507, Qwen3-4B-Thinking-2507-FP8, Qwen3-8B, Qwen3-8B-AWQ, Qwen3-8B-FP8, Qwen3-Coder-30B-A3B-Instruct, Qwen3-Coder-30B-A3B-Instruct-FP8, Qwen3-Coder-480B-A35B-Instruct, Qwen3-Coder-480B-A35B-Instruct-FP8, Qwen3-Coder-Next, Qwen3-Coder-Next-FP8, Qwen3-Next-80B-A3B-Instruct, Qwen3-Next-80B-A3B-Instruct-FP8, Qwen3-Next-80B-A3B-Thinking, Qwen3-VL-8B-Instruct, Qwen3.5-122B-A10B, Qwen3.5-35B-A3B, Qwen3.5-397B-A17B, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-2.4T-A95B, Qwen3.8-2.4T-A95B-FP8, Qwen3.8-27B, Qwen3.8-27B-FP8, Qwen3.8-Flash-Next, QwQ-32B, QwQ-32B-Preview

**deepseek-ai** (34): DeepSeek-Coder-V2-Instruct, DeepSeek-Coder-V2-Lite-Base, DeepSeek-Coder-V2-Lite-Instruct, deepseek-llm-67b-chat, deepseek-llm-7b-chat, DeepSeek-Math-V2, DeepSeek-R1, DeepSeek-R1-0528, DeepSeek-R1-Distill-Llama-70B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-14B, DeepSeek-R1-Distill-Qwen-32B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Zero, DeepSeek-V2, DeepSeek-V2-Chat, DeepSeek-V2-Chat-0628, DeepSeek-V2-Lite, DeepSeek-V2-Lite-Chat, DeepSeek-V2.5, DeepSeek-V2.5-1210, DeepSeek-V3.1, DeepSeek-V3.1-Base, DeepSeek-V3.1-Terminus, DeepSeek-V3.2, DeepSeek-V3.2-Exp, DeepSeek-V3.2-Speciale, DeepSeek-V4-Flash, DeepSeek-V4-Flash-0731, DeepSeek-V4-Flash-DSpark, DeepSeek-V4-Pro, DeepSeek-V4-Pro-0813, DeepSeek-V4-Pro-DSpark

**unsloth** (33): deepseek-r1-distill-qwen-1.5b-unsloth-bnb-4bit, llama-2-7b, llama-2-7b-chat, llama-3-8b-bnb-4bit, llama-3-8b-Instruct, llama-3-8b-Instruct-bnb-4bit, Llama-3.2-1B-Instruct-bnb-4bit, Llama-3.2-1B-Instruct-unsloth-bnb-4bit, Llama-3.2-3B-Instruct-bnb-4bit, Llama-3.2-3B-Instruct-unsloth-bnb-4bit, Meta-Llama-3.1-8B-Instruct-bnb-4bit, Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit, Qwen2.5-0.5B-Instruct-bnb-4bit, Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit, Qwen2.5-1.5B-Instruct-bnb-4bit, Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit, Qwen2.5-14B-Instruct-bnb-4bit, Qwen2.5-32B-Instruct-bnb-4bit, Qwen2.5-3B-Instruct-bnb-4bit, Qwen2.5-3B-Instruct-unsloth-bnb-4bit, Qwen2.5-7B-Instruct-bnb-4bit, Qwen2.5-7B-Instruct-unsloth-bnb-4bit, Qwen2.5-Coder-1.5B-Instruct-bnb-4bit, Qwen2.5-Coder-14B-Instruct-bnb-4bit, Qwen2.5-Coder-3B-Instruct-bnb-4bit, Qwen2.5-Coder-7B-Instruct-bnb-4bit, Qwen3-0.6B-unsloth-bnb-4bit, Qwen3-1.7B-unsloth-bnb-4bit, Qwen3-14B-unsloth-bnb-4bit, Qwen3-4B-Instruct-2507-bnb-4bit, Qwen3-4B-Instruct-2507-unsloth-bnb-4bit, Qwen3-4B-unsloth-bnb-4bit, Qwen3.8-27B-NVFP4

**lmstudio-community** (30): ERNIE-4.5-21B-A3B-MLX-4bit, ERNIE-4.5-21B-A3B-MLX-6bit, ERNIE-4.5-21B-A3B-MLX-8bit, GLM-4.7-Flash-MLX-6bit, GLM-4.7-Flash-MLX-8bit, MiniMax-M2.5-MLX-4bit, MiniMax-M2.5-MLX-6bit, MiniMax-M2.5-MLX-8bit, NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit, NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-5bit, NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-6bit, NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-8bit, Qwen3-30B-A3B-Instruct-2507-MLX-4bit, Qwen3-30B-A3B-Instruct-2507-MLX-6bit, Qwen3-30B-A3B-Instruct-2507-MLX-8bit, Qwen3-4B-Instruct-2507-MLX-4bit, Qwen3-4B-Instruct-2507-MLX-5bit, Qwen3-4B-Instruct-2507-MLX-6bit, Qwen3-4B-Instruct-2507-MLX-8bit, Qwen3-4B-Thinking-2507-MLX-4bit, Qwen3-4B-Thinking-2507-MLX-6bit, Qwen3-4B-Thinking-2507-MLX-8bit, Qwen3-Coder-30B-A3B-Instruct-MLX-4bit, Qwen3-Coder-30B-A3B-Instruct-MLX-5bit, Qwen3-Coder-30B-A3B-Instruct-MLX-6bit, Qwen3-Coder-30B-A3B-Instruct-MLX-8bit, Seed-OSS-36B-Instruct-MLX-4bit, Seed-OSS-36B-Instruct-MLX-5bit, Seed-OSS-36B-Instruct-MLX-6bit, Seed-OSS-36B-Instruct-MLX-8bit

**nvidia** (19): Llama-3.1-8B-Instruct-FP8, Llama-3.1-Nemotron-Nano-8B-v1, Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16, Nemotron-Orchestrator-8B, NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8, NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4, NVIDIA-Nemotron-3-Nano-4B-BF16, NVIDIA-Nemotron-3-Nano-4B-FP8, NVIDIA-Nemotron-3-Super-120B-A12B-BF16, NVIDIA-Nemotron-3-Super-120B-A12B-FP8, NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4, NVIDIA-Nemotron-3.5-Bolt-E4B-BF16, NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16, Orchestrator-8B, Qwen3.8-Flash-Next-FP4-Local, Qwen3.8-Flash-Next-NVFP4

**huihui-ai** (16): DeepSeek-R1-Distill-Llama-70B-abliterated, DeepSeek-R1-Distill-Qwen-14B-abliterated-v2, DeepSeek-R1-Distill-Qwen-32B-abliterated, Huihui-gemma-4-12B-agentic-fable5-abliterated, Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated, Huihui-Ornith-1.5-35B-A3B-abliterated, Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated, Huihui-Qwen3-8B-abliterated-v2, Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated, Huihui-Qwen3-Coder-Next-abliterated, Huihui-Qwen3.8-27B-abliterated, Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated, Huihui-Qwen3.8-Flash-Next-abliterated, Llama-3.3-70B-Instruct-abliterated, QwQ-32B-abliterated, QwQ-32B-Preview-abliterated

**zai-org** (16): cogvlm2-llama3-chat-19B, glm-4-9b, GLM-4.6, GLM-4.6V-Flash, GLM-4.7, GLM-4.7-Flash, GLM-5, GLM-5-FP8, GLM-5.1, GLM-5.1-FP8, GLM-5.2, GLM-5.2-FP8, GLM-5.3, GLM-5.3-BF16, GLM-5.3-Flash, GLM-5.3-Flash-BF16

**ornith-ai** (14): Ornith-1.0-35B-FP8, Ornith-1.0-397B, Ornith-1.0-397B-FP8, Ornith-1.5-122B, Ornith-1.5-122B-FP8, Ornith-1.5-122B-MLX, Ornith-1.5-122B-MLX-4bit, Ornith-1.5-122B-MLX-8bit, Ornith-1.5-122B-NVFP4, Ornith-1.5-35B-A3B, Ornith-1.5-35B-A3B-FP8, Ornith-1.5-397B, Ornith-1.5-397B-FP8, Ornith-1.5-397B-NVFP4

**NousResearch** (12): DeepHermes-3-Llama-3-8B-Preview, Hermes-3-Llama-3.1-8B, Hermes-3-Llama-3.2-3B, Hermes-4-70B, Llama-2-7b-chat-hf, Llama-2-7b-hf, Meta-Llama-3-8B, Meta-Llama-3-8B-Instruct, Meta-Llama-3.1-8B, Meta-Llama-3.1-8B-Instruct, nomos-1, Nous-Hermes-Llama2-13b

**openbmb** (10): MiniCPM-2B-sft-bf16, MiniCPM-2B-sft-fp32, MiniCPM-V-4.7-35B-A3B, MiniCPM3-4B, MiniCPM4-8B, MiniCPM4.1-8B, MiniCPM5-1B, MiniCPM5-1B-MLX, MiniCPM5-2B, MiniCPM5-2B-MLX

**google** (8): gemma-4-12B, gemma-4-12B-it, gemma-4-26B-A4B, gemma-4-26B-A4B-it, gemma-4-26B-A4B-it-assistant, gemma-4-31B-it, gemma-4-E2B-it, gemma-4-E4B-it

**meta-llama** (8): Llama-2-70b-chat-hf, Llama-3.1-405B-FP8, Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-4-Scout-17B-16E-Instruct, Llama-Guard-3-1B, Meta-Llama-3-70B, Meta-Llama-3.1-70B-Instruct

**ReadyArt** (8): gemma-4-31B-it-scotoma-2, Omega-Arsenal-12B, Omega-Arsenal-31B-Final, Wicked-Stuff-12B, Wicked-Stuff-26B-A4B-v2.1-FINAL, Wicked-Stuff-26B-A4B-v3.0-Final, Wicked-Stuff-31B-Final, Wicked-Tutu-31B-Final

**allenai** (7): Llama-3.1-Tulu-3-8B, Olmo-3-32B-Think, Olmo-3-7B-Instruct, Olmo-3-7B-Instruct-DPO, Olmo-3-7B-Instruct-SFT, Olmo-3-7B-Think, Olmo-3.1-32B-Instruct

**cognitivecomputations** (7): dolphin-2.9.4-llama3.1-8b, Dolphin-Mistral-24B-Venice-Edition, Dolphin3.0-Llama3.1-8B, Dolphin3.0-Llama3.2-1B, Dolphin3.0-Llama3.2-3B, Dolphin3.0-Mistral-24B, Dolphin3.0-R1-Mistral-24B

**LiquidAI** (7): LFM2.5-2.6B, LFM2.5-230M, LFM2.5-24B-A2B, LFM2.5-24B-A2B-FP8, LFM2.5-8B-A1B, LFM3-1.7B, LFM3-1.7B-NVFP4

**llmfan46** (6): gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-uncensored-heretic, gemma-4-12B-coder-fable5-composer2.5-v1-uncensored-heretic, MiniMax-M2.7-BF16-ultra-uncensored-heretic, Ornith-1.0-35B-uncensored-heretic, Qwen3-Coder-Next-Uncensored-Heretic, Qwen3.8-27B-Uncensored-Heretic-Native-MTP-Preserved

**prism-ml** (6): Bonsai-1.7B-unpacked, Bonsai-4B-unpacked, Bonsai-8B-unpacked, Ternary-Bonsai-1.7B-unpacked, Ternary-Bonsai-4B-unpacked, Ternary-Bonsai-8B-unpacked

**QuantTrio** (6): GLM-4.7-Flash-AWQ, Qwen3-30B-A3B-Thinking-2507-AWQ, Qwen3-Coder-30B-A3B-Instruct-AWQ, Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8, Qwen3-VL-30B-A3B-Instruct-AWQ, Qwen3-VL-30B-A3B-Thinking-AWQ

**DavidAU** (5): Qwen3-30B-A1.5B-High-Speed, Qwen3-4B-Claude-Sonnet-4-Reasoning-Distill-Heretic-Abliterated-Heretic-Abliterated, Qwen3-8B-192k-Context-6X-Josiefied-Uncensored, Qwen3-8B-64k-Context-2X-Josiefied-Uncensored, Qwen3.6-12B-IQ-Ultra-Heretic-Uncensored-Thinking-V2-Hightop

**LGAI-EXAONE** (5): EXAONE-Deep-2.4B, EXAONE-Deep-32B, EXAONE-Deep-7.8B, K-EXAONE-2.0-750B-A37B, K-EXAONE-236B-A23B

**MiniMaxAI** (5): MiniMax-M2, MiniMax-M2.1, MiniMax-M2.5, MiniMax-M2.7, MiniMax-M3.1-Flash

**mlabonne** (5): Meta-Llama-3-120B-Instruct, Meta-Llama-3.1-8B-Instruct-abliterated, NeuralDaredevil-8B-abliterated, Qwen3-4B-abliterated, Qwen3-8B-abliterated

**tencent** (5): Hunyuan-0.5B-Instruct, Hunyuan-1.8B-Instruct, Hunyuan-7B-Instruct, Hunyuan-A13B-Instruct, Youtu-LLM-2B

**TrevorJS** (5): gemma-4-12B-it-uncensored, gemma-4-26B-A4B-it-uncensored, gemma-4-31B-it-uncensored, gemma-4-E2B-it-uncensored, gemma-4-E4B-it-uncensored

**CausalLM** (4): 14B, 7B, 7B-DPO-alpha, miniG

**cyankiwi** (4): GLM-4.7-Flash-AWQ-4bit, Qwen3-30B-A3B-Instruct-2507-AWQ-4bit, Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit, Qwen3-Next-80B-A3B-Instruct-AWQ-4bit

**FINAL-Bench** (4): Darwin-180B-RSI-R3, Darwin-27B-RSI, Darwin-28B-Coder, Darwin-35B-A3B-Opus

**JackKozmo29** (4): codeguard-jev-style-1.5b, depguard-jev-style-1.5b, mailguard-jev-style-1.5b, sevclassifier-jev-style-1.5b

**PrimeIntellect** (4): INTELLECT-2, INTELLECT-3, Qwen3-0.6B, Qwen3-1.7B

**RedHatAI** (4): DeepSeek-V2.5-1210-FP8, Meta-Llama-3.1-70B-Instruct-FP8, Meta-Llama-3.1-8B-Instruct-FP8, Qwen3-30B-A3B-Instruct-2507-quantized.w4a16

**Sao10K** (4): 72B-Qwen2.5-Kunou-v1, L3-70B-Euryale-v2.1, L3.3-70B-Euryale-v2.3, Llama-3.1-8B-Stheno-v3.4

**arcee-ai** (3): Trinity-Large-Preview, Trinity-Large-Thinking, Trinity-Mini

**baichuan-inc** (3): Baichuan-13B-Chat, Baichuan2-13B-Chat, Baichuan2-7B-Chat

**fableforge-ai** (3): NEXUS-Finance, NEXUS-Legal, NEXUS-Medical

**Gryphe** (3): Gemma-4-31B-StyleTune, MythoMax-L2-13b, Pantheon-Reasoning-26B-A4B-1.1-V2

**h2oai** (3): h2ogpt-4096-llama2-70b-chat, h2ovl-mississippi-2b, h2ovl-mississippi-800m

**ibm-granite** (3): granite-4.2-30b, granite-4.2-3b, granite-4.2-8b

**jinaai** (3): reader-lm-0.5b, reader-lm-1.5b, ReaderLM-v2

**lmsys** (3): vicuna-13b-v1.5, vicuna-13b-v1.5-16k, vicuna-7b-v1.5

**ManniX-ITA** (3): JackOD-9B-Coder, Qwen3.6-27B-A3B-Coder, Qwen3.6-27B-A3B-CoderX

**Menlo** (3): Jan-nano-128k, Jan-v3-4B-base-instruct, Lucy-128k

**microsoft** (3): bitnet-b1.58-2B-4T, Phi-4-reasoning, Phi-4-reasoning-plus

**miromind-ai** (3): MiroThinker-1.7, MiroThinker-v1.5-235B, MiroThinker-v1.5-30B

**OBLITERATUS** (3): Gemma-4-12B-OBLITERATED, Qwen3.6-27B-OBLITERATED, Qwen3.8-27B-OBLITERATED

**openchat** (3): openchat-3.5-0106, openchat-3.5-1210, openchat_3.5

**poolside** (3): Laguna-S-2.1, Laguna-XS-2.1, Laguna-XS.2

**01-ai** (2): Yi-34B-Chat, Yi-6B-Chat

**0bserverx** (2): Qwen3.8-27B-Heretic-Abliterated-Uncensored, RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored

**0xSero** (2): DeepSeek-V4-Flash-162B, Qwen3.6-28B-REAP20-A3B

**AEON-7** (2): Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16, Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16

**agentica-org** (2): DeepCoder-14B-Preview, DeepSWE-Preview

**Aleph-Alpha** (2): Kolibri-1, Kolibri-1-BF16

**baidu** (2): ERNIE-4.5-21B-A3B-PT, ERNIE-4.5-21B-A3B-Thinking

**Blackfrost-AI** (2): CYBER-FROST-3.8-BF16, RED-SNOW-5.3-FLASH-BF16

**dealignai** (2): GLM-5.3-UNCENSORED-FP8, Qwen3.8-Flash-Next-UNCENSORED-FP8

**dongfangshuo** (2): Dongfangshuo-Qwen3.8-27B-NSFW, Dongfangshuo-Qwen3.8-27B-NSFW-Novel

**elyza** (2): ELYZA-Shortcut-1.0-Qwen-7B, Llama-3-ELYZA-JP-8B

**failspy** (2): llama-3-70B-Instruct-abliterated, Meta-Llama-3-8B-Instruct-abliterated-v3

**GnLOLot** (2): MiniCPM5-1B-Claude-Opus-Fable5-Thinking, MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking

**Goekdeniz-Guelmez** (2): Josiefied-Qwen3-14B-abliterated-v3, Josiefied-Qwen3-8B-abliterated-v1

**gradientai** (2): Llama-3-8B-Instruct-262k, Llama-3-8B-Instruct-Gradient-1048k

**Groq** (2): Llama-3-Groq-70B-Tool-Use, Llama-3-Groq-8B-Tool-Use

**haoranxu** (2): ALMA-13B-R, ALMA-7B-R

**hotdogs** (2): Qwen3.8-27B-abliterated, Qwen3.8-27B-abliterated-cyber-preview

**hugging-quants** (2): Meta-Llama-3.1-70B-Instruct-AWQ-INT4, Meta-Llama-3.1-8B-Instruct-AWQ-INT4

**HuggingFaceTB** (2): SmolLM-135M-Instruct, SmolLM3-3B

**IlyaGusev** (2): gemma-2-9b-it-abliterated, saiga_llama3_8b

**janhq** (2): Jan-code-4b, Jan-v1-4B

**K-intelligence** (2): Midm-2.0-Base-Instruct, Midm-2.0-Mini-Instruct

**Kwaipilot** (2): KAT-Coder-V2.5-Dev, KAT-Dev

**LocoreMind** (2): LocoOperator-4B, LocoTrainer-4B

**Lythri** (2): Lythri-4B-A2B, Lythri-7B-A4B

**mattshumer** (2): ref_70_e3, Reflection-Llama-3.1-70B

**MaziyarPanahi** (2): calme-2.4-llama3-70b, Llama-3-8B-Instruct-64k

**mistralai** (2): Devstral-Small-2-24B-Instruct-2512, Mistral-Small-24B-Instruct-2501

**Motif-Technologies** (2): Motif-3, Motif-3-Beta

**Nexusflow** (2): Athene-70B, Athene-V2-Chat

**open-r1** (2): OlympicCoder-32B, OlympicCoder-7B

**Orenguteng** (2): Llama-3.1-8B-Lexi-Uncensored, Llama-3.1-8B-Lexi-Uncensored-V2

**sarvamai** (2): sarvam-1, sarvam-m

**SC117** (2): Laguna-S-2.1-Uncensored, Ornith-1.0-35B-Heretic-MTP

**skt** (2): A.X-4.0, A.X-K1

**stelterlab** (2): Mistral-Small-24B-Instruct-2501-AWQ, Qwen3-30B-A3B-Instruct-2507-AWQ

**t-tech** (2): T-lite-it-2.1, T-Search

**Tesslate** (2): UIGEN-T2-7B, UIGEN-X-32B-0727

**THUDM** (2): glm-4-9b-chat, LongWriter-llama3.1-8b

**tvall43** (2): Qwen3.6-14B-A3B-FableVibes, Qwen3.6-14B-A3B-VibeForged-v2

**upstage** (2): SOLAR-0-70b-16bit, Solar-Open-100B

**webAI-Official** (2): TwIL-8B, TwIL-LM3

**WizardLMTeam** (2): WizardLM-13B-V1.2, WizardLM-70B-V1.0

**XingChen-AGI** (2): Xing4.0-29B-A4B, Xing4.0-29B-Intent

**26B-Suite** (1): Gemory-26B-A4B-SOMPOA

**a-m-team** (1): AM-Thinking-v1

**agentscope-ai** (1): CoPaw-Flash-9B

**AiCloser** (1): Qwen2.5-32B-AGI

**AIDC-AI** (1): Marco-o1

**aifeifei798** (1): DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored

**ajgazin** (1): Swift-1.5-Qwen3.8-27B-Uncensored-MTP

**akjindal53244** (1): Llama-3.1-Storm-8B

**Alibaba-Apsara** (1): DASD-4B-Thinking

**AliesTaha** (1): fable-traces

**all-hands** (1): openhands-lm-32b-v0.1

**allura-forge** (1): Llama-3.3-8B-Instruct

**allura-org** (1): Qwen3.6-35B-A3B-Anko

**Altworld** (1): Hemmingway-1

**amd** (1): Instella-MoE-16B-A3B-Think

**anthracite-org** (1): magnum-v1-72b

**apetersson** (1): DeepSeek-V4-Flash-0731-Abliterated-FP8

**Apexintelligence-AI** (1): Apex-35B-V1.0

**apodex** (1): Apodex-1.1-mini

**ArliAI** (1): GLM-4.6-Derestricted-v3

**AyoubChLin** (1): lfm2.5-2.6b-fable5-coding-agent

**badtheorylabs** (1): BTL-4

**Bahushruth** (1): Qwen3.6-35B-A3B-abliterated-v4

**bg-digitalservices** (1): Gemma-4-26B-A4B-it-NVFP4

**Bllossom** (1): llama-3.2-Korean-Bllossom-3B

**Blockway** (1): Agens-Volundr-32B-Preview

**bottlecapai** (1): ThinkingCap-Qwen3.6-27B

**braindao** (1): gemma-3-4b-it-uncensored-v2

**BSC-LT** (1): salamandra-7b-instruct

**bytedance-research** (1): ChatTS-14B

**ByteDance-Seed** (1): Seed-OSS-36B-Instruct

**bytkim** (1): Qwen3.8-27B-pi

**Caldalis** (1): MiniCPM5-2B-Math

**casperhansen** (1): llama-3.3-70b-instruct-awq

**cerebras** (1): Qwen3-Coder-REAP-25B-A3B

**chbae624** (1): vllm-translategemma-12b-it

**chuanli11** (1): Llama-3.2-3B-Instruct-uncensored

**Chunjiang-Intelligence** (1): DeepSeek-v4-Fable

**culturerevolt** (1): gemma-4-12b-heretic-abliterated

**cyberagent** (1): DeepSeek-R1-Distill-Qwen-14B-Japanese

**deadbydawn101** (1): gemma-4-E4B-Agentic-Opus-Reasoning-GeminiCLI-mlx-4bit

**deepreinforce-ai** (1): Ornith-1.0-35B

**dervig** (1): m51Lab-MiniMax-M2.7-REAP-139B-A10B

**diabolic6045** (1): open-llama-3.2-1B-Instruct

**dima6312** (1): Flint-40B-A2.5B-abliterated

**dots-studio** (1): dots.llm1.inst

**dphn** (1): Dolphin-X1-8B

**DuoNeural** (1): Gemma4-12B-IT-Abliterated

**ElementMerc** (1): qwen2.5-1.5b-uncensored

**empero-ai** (1): Qwen3.8-35B-A3B-Distill

**EschaLabs** (1): Qwen3.8-27B-Escha-W2

**EssentialAI** (1): rnj-1-instruct

**ewinregirgojr** (1): MiniCPM5-1B-Agentic-Tooluse-Merged-FP16

**facebook** (1): MobileLLM-Pro

**farbodtavakkoli** (1): OTel-2.0-LLM-31B-IT

**fdtn-ai** (1): Foundation-Sec-8B-Reasoning

**FenomAI** (1): Ornith-1.5-35B-A3B-NVFP4

**Fortytwo-Network** (1): Strand-Rust-Coder-14B-v1

**Fredithefish** (1): Guanaco-13B-Uncensored

**grumd0** (1): Nemotron-3.5-Bolt-E4B-Heretic

**gugujiji** (1): novel-humanize-4b

**happzy2633** (1): qwen2.5-7b-ins-v3

**HemmingwayAI** (1): Hemmingway-1.5

**hiwaifu-research** (1): WaifuGemma4-26b-a4b-v1

**HuggingFaceBio** (1): Carbon-X

**ibnzterrell** (1): Meta-Llama-3.3-70B-Instruct-AWQ-INT4

**igorls** (1): gemma-4-12B-it-heretic

**inclusionAI** (1): Ling-3.0-tiny

**Inferact** (1): GLM-5.3-NVFP4

**inference-net** (1): Schematron-3B

**inference-optimization** (1): Qwen3-1.6B-A0.9B

**IQuestLab** (1): IQuest-Q1

**Jab1718** (1): qwen3.8-flash-coder-85gb-bf16

**jialinyyzz** (1): humanizer

**Jiunsong** (1): supergemma4-e4b-abliterated

**JonathanColetti** (1): Qwen3.8-27B-Uncensored

**jtatman** (1): llama-3.2-1b-lewd-mental-occult

**kai-os** (1): Carnice-27b

**katanemo** (1): Arch-Router-1.5B

**kawaimasa** (1): Wanabi-Gemma4-31B

**KRAFTON** (1): Raon-1-mini-preview

**LilaRest** (1): gemma-4-31B-it-NVFP4-turbo

**lmms-lab** (1): llava-onevision-qwen2-7b-ov

**logic65** (1): Whittle-Qwen-3.8-35B-A3B

**Luni** (1): Gemma-4-31B-Heretic

**Ma7ee7** (1): Qwen3.8_4B_Distilled

**Madras1** (1): Gemma-4-E4B-Abliterated-Uncensored

**meituan-longcat** (1): LongCat-Flash-Lite

**MemTensor** (1): MemT-1.0-Flash

**migtissera** (1): Tess-2.0-Llama-3-8B

**mikeinnyc** (1): Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16

**MLP-KTLim** (1): llama-3-Korean-Bllossom-8B

**Nanbeige** (1): Nanbeige4.2-3B

**NaniDAO** (1): deepseek-r1-qwen-2.5-32B-ablated

**neopolita** (1): Qwen3.6-11B-A3B-Niwaki-4bit-mlx

**NovaSky-AI** (1): Sky-T1-32B-Preview

**NTQAI** (1): Nxcode-CQ-7B-orpo

**null-space** (1): Qwen3-235B-A22B-abliterated-FP8

**Olafangensan** (1): GLM-4.7-Flash-heretic

**open-thoughts** (1): OpenThinker-32B

**OpenOneRec** (1): OneRec-1.7B

**OpenVINO** (1): Qwen2.5-Coder-7B-Instruct-int8-ov

**orcarouter** (1): GLM-5.3-Flash-Uncensored-FP8

**org2ai** (1): Wald-26B-A4B

**Orion-zhen** (1): Qwen2.5-7B-Instruct-Uncensored

**OrionLLM** (1): GRM-2.6-Plus

**oyildirim** (1): CyberStrike-OffSec-35B

**p-e-w** (1): Qwen3-4B-Instruct-2507-heretic

**phamhai** (1): Llama-3.2-1B-CyberFrog

**PowerInfer** (1): SmallThinker-3B-Preview

**PygmalionAI** (1): pygmalion-2-7b

**qnguyen3** (1): nanoLLaVA

**QuixiAI** (1): Qwen3-30B-A3B-AWQ

**RecursiveSI** (1): flint

**ReliquaryForge** (1): Teutonic-I-Reliquary-RL

**RootMonsteR** (1): Qwen3-14B-Abliterated

**sabicap** (1): sabi-noema

**Salesforce** (1): Llama-xLAM-2-8b-fc-r

**seedboxai** (1): KafkaLM-70B-German-V0.1

**sensenova** (1): SenseNova-Loop-9B-Preview

**shawnw3i** (1): Qwopus3.8-27B-Flash-V2-abliterated

**stabilityai** (1): StableBeluga2

**TencentARC** (1): LLaMA-Pro-8B

**TheBloke** (1): MythoMax-L2-13B-GPTQ

**THU-KEG** (1): LongWriter-Zero-32B

**Tongyi-Zhiwen** (1): QwenLong-L1.5-30B-A3B

**trohrbaugh** (1): gemma-4-26B-A4B-it-heretic-ara-v2

**TrueRealRaven** (1): gemma-4-E2B-it-heretic-v1

**ukisai** (1): Swift-1.5-Qwen3.8-27b

**Vortecks** (1): LFM3-1.7B-Heretic-Abliterated

**whitecircle** (1): GLM-4.7-Flash-Coder

**XCurOS** (1): XCurOS0.1-8B-Instruct

**XiaomiMiMo** (1): MiMo-V2.6-Distill-Qwen-9B

**Xwin-LM** (1): Xwin-LM-70B-V0.1

**ykarout** (1): Qwen3.5-9b-Opus-Openclaw-Distilled

**Youssofal** (1): MiniMax-M2.7-abliterated-BF16

**YuYu1015** (1): Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-NVFP4

**Zyphra** (1): ZR1-1.5B
