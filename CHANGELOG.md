# Changelog

What changed in each version of Kera, newest first. The full notes for each release are in `docs/releases/`, and
the reasons behind each change in `docs/DECISIONS.md` (the D-numbers below). Kera gives the model's own answers;
the lossy modes the 0.1.0 previews had (opt-in) were removed in 0.1.3.

## Unreleased

## 0.1.5.3 (2026-11-18)

Patch release: llama.cpp b11619 -> b12500, with Kera's eight hooks carried over (saved chats, guess controller,
streamed experts, per-request model defaults, reply shapes, chat sharing and late parameter filling). Kera's own
answer path is unchanged: its IO batching and two fixes now in upstream were dropped in favour of upstream's.
The N-009 greedy replay guard stays. Saved chats were adapted to upstream's shared-pointer checkpoints and new
state-data type. Linux CI is green; graphics-chip checks and speed measurements still need the owner's hardware.

## 0.1.5.2 (2026-11-11)

LTX video follows what you ask for (D-042). The DGX Spark download comes first, built on a DGX Spark; the other
downloads stay at 0.1.4 until they are built.

- **A long LTX clip no longer comes out as something else.** The owner asked for a sea turtle over a reef and got
  two people talking on a stage. The cause was the step schedule: the image engine shifted it by the clip's size,
  where Lightricks' own pipelines give every clip the same one, so a 5-second clip was run far outside what the
  model is made for. The same clip, seed and settings now give the turtle.
- **Lightricks' own guidance for LTX is in the image engine**, for those who want it: beside the text guidance, their
  three other terms and a separate scale for the sound (`kera video --help` gives the line). It is not on by
  itself yet: it takes four passes of the model a step where plain guidance takes two, and it has not been checked
  on a long clip. The defaults are as in 0.1.5.1 (LTX 2.5: 20 steps, CFG 3, 24 frames a second).
- `kera serve --sfx` refuses a `--port` that is not a number from 1 to 65535 (it went into a file name as typed),
  and `kera music --guidance` and `--shift` refuse a word (it was read as 0, which turned the guidance off without
  saying so). Found in a scan of the code added since 0.1.3.
- The LTX test can now check that the words reach the video (a snowfield must come out brighter than a cave), and
  the long clip is part of the release check: this fault only showed there.

## 0.1.5.1 (2026-11-11)

0.1.5 with LTX 2.5 video made to work. The DGX Spark download comes first, built on a DGX Spark (D-041); the other
downloads stay at 0.1.4 until they are built.

**LTX 2.5 video works** (the owner's report from a DGX Spark):

- It no longer stops at the first step. Some of its small weights are stored as bf16, which the image engine could
  not add to its working values; they are read as 32-bit floats, which holds the same numbers exactly. No converted
  copy of the model is needed.
- Its video is no longer blank. The text reader (Gemma 4) gave values that are not numbers when its fast attention
  ran on the graphics chip, so every frame was white and the sound at full scale. It runs without it there; the
  video model keeps its own.
- LTX videos are written at 24 frames a second, which is what LTX makes (they were written at 16 and played slow).
  `--fps` sets another.
- LTX 2.5 takes 20 steps where you set none (it took 30): the image engine's own recipe, from Lightricks' code.
- A test now makes a real LTX video and looks at the frames (`tests/integration/test_ltx_video.py`; it needs the
  model's files, so it runs on the Spark, not on GitHub).

## 0.1.5 (2026-11-10)

Speed only: every output is the same, byte for byte (D-039, D-040). Built on a DGX Spark for the DGX Spark
(D-041) and used there, but not published: 0.1.5.1 took its place.

- Music loads its models about 15 times sooner on a graphics chip that shares the computer's memory: on a DGX
  Spark a 60 s song from `kera music` takes 43 s instead of 76 (24 s instead of about 50 with a turbo model), and
  the first song from `kera serve --music` 42 s instead of 81. The songs are the same files, byte for byte (D-040).
  FLUX 3 pictures, edits and widened pictures were measured too (23 s, 50 s and 67 s there); nothing was found that
  is faster without changing the picture.
- `kera serve --sfx DIR` keeps the sound-effect model loaded: on a DGX Spark a 10 s sound takes 32 s instead of 39,
  and it is the same sound, sample for sample, as `kera sfx` makes (D-039). Video was measured too; nothing was found
  there that is faster without changing the output.

## 0.1.4 (2026-11-10)

Every model runs with its makers' own settings by default (D-036). What you set still wins.

**Music**

- ACE-Step's base and SFT models (XL too) run at ACE-Step's own guidance 7 and shift 3 where you set none. They ran
  without guidance at shift 1, which made sung words less clear. `kera music --guidance` and `--shift` set
  others.
- Turbo models get ACE-Step's own correction step (DCW), which ACE-Step turns on for them by default.
- With several models in the music folder, Kera uses the planner and the sound model ACE-Step's guide recommends
  for the computer's memory, at the highest precision that fits (it took the first by file name: the base model, the
  smallest planner, the 4-bit file). `--lm` and `--dit` still choose.
- Measured on a DGX Spark, the words Whisper hears in songs with given lyrics: 0.1.4 by itself 95%, what 0.1.3 took
  by itself in the same folder 75% (`docs/QUALITY.md`).

**Pictures and video**

- 53 more models get their makers' settings by themselves: Qwen-Image and its edit models, Z-Image, FLUX.2 klein,
  FLUX 3 klein, Stable Diffusion 3 and 3.5, SDXL base, Stable Diffusion 1.x and 2.x, Chroma, HunyuanVideo 1.5 and
  others. They ran at the image engine's generic 20 steps and guidance 7.
- Models whose makers use more steps now take them (Qwen-Image 50, SD 3.5 Medium 40, SDXL base 50, HunyuanVideo 1.5
  50), so they draw as their makers intend and take longer. `--steps` sets fewer.
- Wan 2.2 S2V gets its makers' 40 steps, guidance 4.5 and flow shift 3.

**Chat**

- A model file that carries no sampling settings of its own now gets its makers' own defaults (temperature, top_p,
  top_k, repetition penalty) from a list of 680 models built into Kera, collected from the makers' pages on Hugging
  Face. The model is found by its name and shape, so a fine-tune under another name is left alone. `--no-model-defaults` turns it
  off.

**Precision**

- Kera says when the same model at full precision (the model's own weights) is next to the file you named and fits
  this computer's memory.

## 0.1.3 (2026-11-09)

Rule number one: Kera never changes how a model responds, it only makes things faster. The places found where Kera
did change a model's output are fixed, the lossy modes are gone, and chat, picture and video models run with their
makers' own settings.

**Known, not fixed in this version** (fixed in 0.1.4)

- `kera music` with ACE-Step's base and SFT models runs at the music engine's own values (50 steps, no guidance,
  shift 1), not the ones ACE-Step recommends for them, and takes the first planner model in the folder by name. A
  check in another session found the sung words much less accurate that way. `--steps`, `--lm` and `--dit` can be
  set by hand; guidance and shift can't yet. The turbo model's 8 steps without guidance are the engine's values for
  it too; all of them are checked against ACE-Step's own for 0.1.4.
- Models' own settings come from a list of the models Kera knows by name and by their weights. A model not on the
  list runs with its engine's defaults. Finding every model's own defaults automatically is being worked on.

**Faster starts**

- The picture, music, 3D and sound-effect engines keep the graphics chip's built programs (shaders) in Kera's
  folder, as the chat engine has since preview 3, so the first pictures after a start are no longer slow while
  they are built (D-032). Not yet built for or tested on a graphics chip.
- `kera serve` with an image model makes one small throwaway picture after it starts, when the computer has a
  graphics chip, so the first real picture is at full speed. A picture made after it is byte for byte the same as
  one made without it. This may be replaced by warming while the model loads and keeping the engine running.

**Sound effects**

- MOSS-SoundEffect's text encodings are kept per model. A description used before (the same sound with another
  seed) skips loading its 2 GB text encoder, and the encoder is freed before the sound is made. The sound is the
  same. They are filed by the model file's contents, Kera's build and the back-end, so one is only ever reused by the
  same model on the same engine.

**Rule number one: Kera never changes how a model responds, it only makes things faster** (D-033)

- The prompt improver now asks the chat model through Kera's own chat server, exactly as a chat app does: the
  model's own template and settings, its thinking as usual (kept apart from the rewrite), and its whole answer,
  used as written. 0.1.2 used greedy sampling and its answer was cut at the first blank line; both are gone.
- It asks with a new random seed each time, as a chat does, so asking again can give another rewrite. 0.1.2 used a
  fixed seed and kept its rewrites to hand the same one back; both are gone.
- Faster without touching the answer: if Kera's app already has that chat model loaded, it is asked there, with no
  loading at all, and Kera's chat speed-ups apply (guessing ahead, and its instructions kept on the SSD so the next
  rewrite doesn't read them again).
- `kera 3d` and `kera sfx` no longer fix a seed the engine would pick at random (3D: Kera passed 42; Stable Audio:
  0). Without `--seed`, each engine uses its own default.
- `kera music --no-rewrite` is removed: it turned off the music model's own rewriting step. Kera still shows what
  the model wrote.
- **No modes that trade quality for speed** (D-034): `kera serve --lossy-kv8`, `--lossy-route` and `--lossy-reuse`
  are removed. They were never on by default; now they don't exist. Every speed-up Kera keeps gives the model's own
  answers.
- The model card's sampling no longer replaces what you set yourself: `--temp`, `--top-p`, `--top-k` and `--min-p`
  on `kera serve` (or llama.cpp's `LLAMA_ARG_*` variables) now win over the card, as in llama.cpp.
- The card is used only for the card's own models: Qwen's own Qwen3 releases and NVIDIA's Nemotron 3, by name. A
  fine-tune or distill of the same design (such as DeepSeek-R1-0528-Qwen3-8B), Nemotron-H and Nemotron Nano 2, and
  any file that carries its own sampling defaults, now run with those, as in llama.cpp.
- Saved chats come back only into the run that would compute the same memory: not after an update, not under
  another context size on models whose positions depend on it (Phi-3, Phi-4-mini), not with other settings given
  through `LLAMA_ARG_*` variables, and not after a request used other adapter scales. Before, such a chat could
  continue from memory computed differently.
- `kera 3d -p` draws the picture from your words as written; it used to add its own ("a single object, centered,
  plain white background").
- `kera music` lets the model choose the song's length unless you give `--duration` (it was always 30 s).
- `kera sfx` passes a length only when you give one, exactly as typed (it always sent 10 s, which isn't Stable
  Audio's own, and rounded your number itself).
- Z-Image-Turbo gets 9 steps, which is its card's 8 model calls in the image engine's count (it got 8); FLUX.1-dev
  keeps the engine's own step count.
- The sound and 3D help no longer suggest fewer steps or a lower resolution to go faster.
- **Models from Ollama's store** now run the way Ollama runs them: with their Modelfile's system prompt, example
  messages, settings, stop words and adapters. One whose file has no chat format of its own (Ollama keeps it in its own
  form) is refused with a plain message instead of being prompted in a format it wasn't trained on, which made it
  answer much worse.
- The Nemotron card is now only Nemotron 3 Nano 30B-A3B's: Nano Omni, Super, Ultra and the 3.5 releases keep their own
  settings. A Qwen3-2507 Instruct model keeps its card's settings even when an app asks it to think. Qwen's quantized
  Qwen3 files get the presence penalty Qwen's own pages recommend.
- A model is no longer given an image reader that isn't its own: a reader alone beside a model is used only when the
  files show it belongs to that model; otherwise Kera says so and the model stays text-only.
- With guessing ahead on, word probabilities (`logprobs`) are the model's own again; they were reported as certain for
  every guessed word (llama.cpp's own fix, ported).
- Pictures get their card's whole recipe where you set none: 1024x1024 for FLUX, Z-Image-Turbo and SDXL-Lightning,
  and SDXL-Lightning's Euler with trailing steps (its 2-step model ran as one jump). FLUX is told apart by its weights:
  a de-distilled "dev" model keeps its guidance (it was turned off), and Kontext, Fill, Canny, Depth, Krea and
  FLUX.2 get their own. With a picture to start from, a turbo model keeps your steps (with 1 step the engine ignored
  the picture).
- `kera video` uses Wan's and LTX's own settings (Wan ran at the engine's generic guidance 7 with no negative prompt):
  Wan's guidance, flow shift and its makers' default negative prompt, shown when used; LTX 2.3 distilled: 8 steps
  without guidance. Steps stay the engine's.
- `kera 3d` needs `birefnet.gguf`, the background remover TRELLIS.2 itself always uses (without it a drawn picture
  was cut out by brightness); `--seed 0` is refused, since the engine reads it as "pick one".
- `kera sfx` takes the full-precision MOSS file when both are in the folder (it took the 8-bit one), and `kera speak`
  and vision models take the reader closest in precision to the model, as llama.cpp's own rule does.
- `kera music` writes WAV unless you ask for `.mp3` (it wrote 128 kbps MP3); `kera serve --music` keeps models loaded
  only when the folder has one of each kind; a lyrics file keeps its words (Windows line ends, a byte-order mark,
  UTF-16 and Windows code pages are handled; they reached the model as stray marks).
- Fine-tuning puts prompt/completion examples in the model's chat format, as MLX-LM does on a Mac, so a fine-tune
  answers in chats the way it learned; MLX now trains as hard as PEFT (MLX-LM's own default trained about 10 times
  harder, and forgot more); an ESFT model keeps its base's name, so Kera still uses its card's settings; examples cut
  at `--max-len` are counted and said.
- `kera wait` no longer suggests turning thinking off, and times the whole reply (it stopped at 256 words).
- An image reader you name or turn off through `LLAMA_ARG_*` variables or llama.cpp's `config.ini` is respected:
  Kera now fills in its own pick after llama.cpp has read every setting (a new llama.cpp hook, 8).
- Streamed experts: a failed read ends the request with an error instead of crashing the server, and a streaming set-up
  that fails stops the load instead of answering from empty weights.
- The picture, music and 3D engines use ggml's tiled multiply on x86-64 only, where it was checked; on ARM it adds
  in half-precision lanes with about twice stock's rounding.
- Qwen3-Coder gets its card's repetition penalty (1.05). A model file replaced in place by another of the same size
  and date is recognised as changed, so its saved chats aren't continued. The picture engine's remembered choice of a
  faster method still checks the method's numbers against the reference at each start.
- The published speeds (15-16 Oct) are marked too high until measured again: each prompt was asked three times on one
  server, so Kera's guesser could copy its own earlier answer. `tools/compare` now asks each prompt once per server.
- **The models' own settings, everywhere** (the owner's answers, D-035): FLUX, FLUX.2, Wan and LTX models take their
  makers' step counts (FLUX.1-dev 50, Wan 2.1 50, Wan 2.2 40, LTX 30...) instead of the engine's 20, so they draw as
  their makers do, and take 2 to 2.5 times as long (`--steps` sets fewer). Nemotron 3 Nano and Qwen3-Coder sample as
  their makers' engines do, without llama.cpp's extra cut of rare words (top_k 40, min_p 0.05). `kera cli`, and a model
  given with `-hf` or `LLAMA_ARG_MODEL`, get the card's settings as `kera serve -m` does. Fine-tuning defaults to about
  three passes over your data instead of a fixed 200 steps.

**Fixed** (a fourth review)

- With `--backend auto`, a message waiting for the other back-end to load is answered after 90 seconds instead of
  failing after 10 minutes while others keep the loaded one busy.
- `kera 3d -p` no longer overwrites a picture already next to the model.
- `kera 3d --seed` above 2,147,483,647 gave the same model for every seed on Windows: such seeds are now refused.
- `kera 3d` and `kera sfx` no longer stop when the temporary folder is missing, and clean up folders a stopped run
  left behind.
- Windows users with an accented or non-Latin user name: the picture engine's saved choices and the sound engine's
  text cache now work.
- A probe the picture engine couldn't finish is remembered too, so a later start can't change the picture for the
  same seed.
- The prompt improver uses the phrase style for SD and SDXL `.safetensors` files, as for their GGUFs.
- Starting Kera again opens the running one only if it belongs to the same user (on a shared computer, another
  user's Kera could be opened).

## 0.1.2 (2026-11-08)

3D models, sound effects, faster pictures and music, a prompt improver, 22 bugs fixed.

**New**

- **3D models** (`kera 3d`): a textured 3D model (`.glb`) from a picture, or from a description (Kera draws the
  picture first). Microsoft's TRELLIS.2 through trellis.cpp (D-031).
- **Sound effects** (`kera sfx -p "door creak" --duration 4 -o door.wav`): MOSS-SoundEffect v2 through audio.cpp,
  48 kHz, up to 30 s (D-031).
- **Prompt improver** (`kera image|video ... --improve-prompt CHAT_MODEL.gguf`, only when asked): a chat model
  rewrites a short description in more detail, keeping everything it says. Both are shown (D-030).
- `kera music` shows how its model rewrote the description; `--no-rewrite` turns that off.

**Faster, with the same quality** (D-030)

- Pictures are made 1.85 times and songs 1.6 times as fast on the processor: a SD-Turbo picture in 20.8 s instead
  of 38.5 s, a 10 s song in 35.7 s instead of 57.8 s. Pictures are as close to the model's reference code as before
  (46.9 dB against 46.7 dB); songs were checked against 0.1.1's (the same song from the same codes; no reference
  code was run for music).
- The picture server loads its model when it starts; the music server keeps its models loaded between songs on
  computers with 32 GB or more.

**Fixed** (N-016)

- The same seed now gives the same song, and the same picture on every graphics chip.
- Starting Kera a second time opens the one already running instead of loading a second copy of the model (the
  "only one user can get in" report).
- With `--backend auto`, one long document no longer makes everyone else wait.
- An ordinary NVIDIA graphics card is no longer taken for a shared-memory chip like the DGX Spark's.
- 18 more: the app's model answers web pages on this computer only; Ctrl+C no longer loses chats being saved;
  `kera finetune` on Windows no longer goes through the command shell; and others.

## 0.1.1.1 (2026-11-08)

A download for the NVIDIA DGX Spark (D-029).

- **The DGX Spark download** (Linux on ARM with CUDA), built for the GB10's graphics chip: chat, pictures, music,
  video and speech all use it.
- Setup recognises a Spark: it treats the 128 GB as shared, lets a model use about 110 GB, and says in plain words
  what would free more memory, changing no system settings itself.
- Models are read into memory in whole rather than memory-mapped on a Spark (faster there).

## 0.1.1 (2026-11-07)

Pictures, music, speech and video; faster engines; 45 bugs fixed.

**New**

- **Models that see images**: `kera serve` finds a vision model's image reader by itself, next to the model or in
  Ollama's store (D-027).
- **Models that make images**: stable-diffusion.cpp in the download. `kera image`, and `kera serve` with OpenAI's
  images API. SD 1.x/2.x/XL, SD3.5, FLUX, Qwen-Image, Z-Image and more (D-027).
- **Music**: ACE-Step 1.5 through acestep.cpp. `kera music`, lyrics written or yours; `kera serve --music` opens
  its own page (D-028).
- **Speech**: `kera speak` with Qwen3-TTS (ten languages, a voice copied from a short sample) or Pocket TTS.
- **Video**: `kera video` with Wan 2.1/2.2, LTX-Video and others.
- Faster first load: as quick as stock llama.cpp from a cold start.

**Faster**

- Music 1.8 times as fast (the music engine now uses every core).
- The picture and music engines pick the best code for the processor when they start.

**Checked**

- Answers as accurate as Ollama's on the same model file (149 against 147 of 200 questions).
- Pictures identical to the model's reference code with the same weights.

**Fixed** (N-014): 45 defects, among them a saved chat continued under different long-context settings, a vision
model given another model's image reader, crashes on two quick clicks of Run, and Windows user names with accented
letters.

## 0.1.0 preview 4 (2026-11-04)

- **The back-end chosen for every message** (`kera serve --backend auto`): Vulkan for chats, the ROCm add-on for
  long documents, measured once per model (D-022, D-024). Off by default until measured.
- **Several people at once** (`kera serve --multi-user`): new messages aren't held up by someone's long document,
  and a start several chats share is read once (D-023). Off by default until measured.
- **Kera vs Ollama** for the Mac: a small app that races the two live on the same model (D-025).
- Open source: Apache 2.0, published on Hugging Face (D-026).

## 0.1.0 preview 3 (2026-10-15)

The first preview measured on the Minisforum.

- Saving a chat no longer holds up the next message (N-005).
- The ROCm add-on works (N-007), and reads a 95,000-word document more than twice as fast as Vulkan.
- Guessing ahead can no longer stop a reply for good (N-009).
- A new Kera folder or an update starts warm: the graphics chip's built shaders are kept (first answer 0.18 s
  instead of 3.2 s, N-011).
- Some Windows command lines and folder names stopped `kera serve` (N-008).
- Measured: Kera writes 1.71 times as fast as Ollama and 1.16 times as fast as stock llama.cpp on Nemotron 3 Nano.

## 0.1.0 preview 2 (2026-10-15)

- Guessing ahead picks the best method per chat and stops when it doesn't pay (M4), and works on models streamed
  from the SSD (M6).
- `kera finetune`: a LoRA adapter from your own examples (M11).
- Three opt-in lossy modes, each with its measured cost: `--lossy-kv8`, `--lossy-route`, `--lossy-reuse` (M12).
  (Removed later: D-034.)
- The optional ROCm add-on for Windows (M8).
- `kera serve` sizes llama.cpp's memory for recent chats to what the model leaves.

## 0.1.0 preview 1 (2026-10-14)

The first Windows download.

- The app: a page in the browser that measures the computer, explains it in plain words, and lists your models
  with how fast each should run; **Run** keeps a model loaded for chat apps.
- `kera serve` (OpenAI and Anthropic APIs) with chats saved on the SSD, so a restart doesn't read them again.
- `kera bench`, `kera wait` (where the wait before a reply goes) and `kera plan` (the speed prediction).
- Mixture-of-experts models bigger than memory stream their experts from the SSD, with the same answers.
- llama.cpp b11619 inside, with Vulkan.
