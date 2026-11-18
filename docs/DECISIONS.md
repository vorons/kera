# Decisions

Newest first. Each entry says what was decided, who decided it, and why.

## D-042: LTX runs on Lightricks' own schedule; their guidance is in the engine, not yet the default (2026-11-11)

The owner's bug 3 (11 Nov): with 0.1.5.1, "a sea turtle glides slowly over a bright coral reef" came out as two
people talking on a stage. The owner suspected the text was lost on its way to the video model. It was not.

- **What was measured on the Spark** (the owner's files, read only). The text reader at its full precision on the
  chip gives the turtle; so does the video model at its full precision; so do both together, at 33 frames with 8, 12
  and 20 steps, at 121 frames with 8 steps, and at 121 frames with 20 steps on another seed. One case gave the
  stage: 121 frames, 20 or 30 steps, seed 42, and it did so with the video model at 8 bits too, with the engine's
  fused kernels off, with Lightricks' negative prompt, with CFG unbatched (bit for bit the same clip), and, as a
  cartoon of two characters talking, with Lightricks' guider added. So: only long clips, whatever the guidance.
- **The cause: the step schedule.** LTX's schedule is shifted towards high noise by an amount that grows with a
  token count. Lightricks' pipelines call it without the latent (`self._scheduler.execute(steps=...)` in
  `ltx_pipelines/ti2vid_*.py`), so it takes its default of 4096 tokens and every clip gets shift 2.05. The image
  engine passed the clip's own token count: 1920 for 33 frames at 768x512 (shift 1.27), 6144 for 121 frames (shift
  2.78, past anything its makers run it with). Printed from the engine on the Spark and recomputed from
  Lightricks' formula. With 4096, the same clip, seed and settings give the turtle.
- **Decided for 0.1.5.2: the schedule is fixed, and the default settings are the ones that were checked.** The
  owner asked for the Spark download at once and without questions, and the Spark had no memory left for more long
  runs beside the owner's site. So:
  - **Lightricks' schedule** is in the image engine (`docs/UPSTREAM_PATCHES.md`): this is the fix for bug 3, checked
    on the owner's own clip;
  - **the defaults stay as in 0.1.5.1** (LTX 2.5: 20 steps, CFG 3, no negative prompt), which is what that check ran;
  - **Lightricks' guider is in the image engine but not on by itself** (`MultiModalGuiderParams`: beside text CFG, a
    pass with block 28's self-attention left out, a pass with sound and picture cut off from each other, a rescale,
    and CFG 7 for the sound). `kera video --help` gives the line that turns it on. It makes four passes a step
    where plain CFG makes two.
- **Still to do, by D-036's rule (the makers' own settings by default):** Lightricks run these models with that
  guider, their default negative prompt and 30 steps (`LTX_2_4_PARAMS` is `LTX_2_3_PARAMS` with another image
  setting, so 30 from LTX 2.3 on; the 20 came from the image engine's `ltx25-dev` recipe, which the owner's report
  took to be Lightricks' constants). None of the three has been run on a long clip with the fixed schedule: with the
  wrong schedule the guider gave a cartoon and the negative prompt changed nothing, which says nothing about them
  with the right one. They become the default when that run has been looked at.
- **Not done:** LTX 2.5 distilled still gets only the frame rate (its makers run it without guidance, on a fixed
  8-step schedule the engine does not have under that name). Lightricks' two-stage pipeline (a second pass at twice
  the size) is not in the engine. The guider's rescale counts every value where a first picture is given (theirs
  leaves the given frames out) unless the mask has the latent's own shape. Nothing here was compared with
  Lightricks' own program run on the same seed: it was read from their code and checked by eye on the Spark.

## D-041: The DGX Spark download is built on the DGX Spark, and comes first (2026-11-10)

Said by the owner (10 Nov: "currently the most important build is the Spark. So build the Spark version on the Spark
and do the other stuff as whenever"). It follows D-038 (the Spark is the owner's machine) and stands while GitHub's
machines do not run for the repository (`docs/STATUS.md`, item 15), and after.

- **How:** `packaging/linux/build-spark.sh VERSION`, in the agent's own `kera` folder on the Spark, no admin rights:
  the recipe of `.github/workflows/build-linux-arm64-cuda.yml` (the same options, the unit tests, the same package).
  Unlike GitHub's machine, the Spark has its graphics chip, so the download is also run on the chip before it is
  published.
- **The compiler is DGX OS's own, GCC 13**, where GitHub's build uses GCC 14. GCC 13 cannot compile ggml's two CPU
  variants for ARM chips with SME, so they are built only where the compiler can (`docs/UPSTREAM_PATCHES.md`). The
  GB10 has no SME and never loaded them: on a Spark the download is the same. On another ARM machine with SME the
  Spark-built download uses the next variant down for work done on the CPU.
- **Nothing of the machine in the download:** the programs are compiled with the build folders' names mapped away
  (`-ffile-prefix-map`), the build folder is taken out of each program's library search path
  (`packaging/strip_build_paths.py`: CMake adds it so that programs run from the build tree), and the script
  searches the package for the machine's folders and refuses to pack it if it finds them (it did, the first time). This is what stopped a Spark-built download before (0.1.4's Spark download came from GitHub).
- **A version can be published for the Spark alone:** `tools/publish/huggingface.py --spark-only OTHERS` puts up the
  Spark download and its source, and the page says which version the other downloads are. 0.1.5.1 is the first
  (0.1.5 was built and used on the Spark, not published).
- **The other downloads** (Windows, the ROCm add-on, Linux x64, Mac) follow when they can be built; they are not
  held back for, and do not hold back, the Spark's.

## D-040: Speed for pictures (FLUX 3) and music: the music models load 15 times sooner, the output the same (2026-11-10)

The owner's request (10 Nov: "now do the same for images and edits, expanding images with FLUX 3, and also music
generation ... we don't change quality, we only change speed"). Measured on the owner's DGX Spark, each part timed
(`docs/BENCHMARKS.md`), as for D-039.

- **Music (ACE-Step 1.5, a 60 s song with words): 76 s to 43 s with the XL SFT model, about 50 s to 24 s with
  turbo; the same file, byte for byte.** Of the 76 s, 35 s were loading the models, which `kera music` does for every
  song. The music engine copied each model's weights to the graphics chip straight from the memory-mapped file:
  one thread reading it page by page, at about 0.5 GB a second on the Spark (13.5 s for the 8 GB planner, 16 s for
  the 8 GB sound model). They now go through ordinary memory first, read by 8 threads: 0.5 s each. The bytes that
  reach the chip are the same, so the song is (`third_party/acestep.cpp/src/weight-ctx.h`, listed in
  `docs/UPSTREAM_PATCHES.md`). On the CPU nothing changes (the engine's own copy is used there as before). It is the
  same thing the setting `gpu.shared_no_mmap` does for chat models on the Spark (spec 10.6), and what the picture
  engine does on every machine. Measured on the Spark only: on other graphics chips it is the same data path the
  picture engine already uses, and is not yet timed there.
- **`kera serve --music`** (the models kept loaded, 0.1.1) gains on its first song only: 81 s to 42 s; later songs
  were already free of loading (38 s, and 20 s for turbo). What is left in a song is the models' own work: the planner
  writing the song's plan (17 s), the 50 steps (19 s; turbo's 8: under 1 s) and the decoding (2 s).
- **Pictures (FLUX 3 klein, BF16, the makers' 16 steps at 1024x1024): nothing found that is free, nothing changed.**
  A picture takes 23 s (19 s the steps, 2 s the decoding, 1 s the words, 1 s loading: the picture engine already
  loads 20 GB in about a second). An edit of a picture takes 50 s (45 s the steps: the picture given is read by the
  model at every step) and widening one to 1536x1024 takes 67 s (61 s the steps). The engine's switches, each timed
  alone: its fast attention is already on by default (off: 43 s the steps of a picture instead of 19, and 123 s
  instead of 45 for an edit); direct convolutions and unbatched guidance give the same picture in the same time
  (FLUX 3 klein runs without guidance, so there is nothing to batch). `kera serve` with an image model already keeps
  it loaded (D-032). What is left is the transformer's own arithmetic on the chip.
- **Tried and taken out:** the same copy through memory in the sound-effect engine: no change (37.9 s against
  38.6 s, the same sound), so its loading time is not this, and the change is not kept.
- **Not done, by rule one:** fewer steps, cached or skipped steps, smaller or tiled decoding, lower precision.
- **The files:** FLUX 3 klein from its makers' GGUF conversion; its text reader (Qwen3-VL 4B, BF16) and its decoder
  (the FLUX.2 VAE) from their public copies, since the makers' own bundle of them needs a licence to be accepted on
  the account, which the agent's has not. Speed is what was measured, not the pictures against the makers' own.

## D-039: Speed for sound effects and video: only what leaves the output the same (2026-11-10)

The owner's request (10 Nov: "make some speed improvements to the video, sound effect ... we don't change quality, we
only change speed"). Measured first on the owner's DGX Spark, each part timed (`docs/BENCHMARKS.md`).

- **Sound effects (MOSS-SoundEffect v2, a 10 s sound, 39 to 41 s):** 32 s are the model's own 100 steps, already run
  as one pass per step for both halves of the guidance, with the work kept between steps, at the speed the PyTorch
  reference reaches on that machine. Nothing is changed there: going faster would mean other arithmetic. The other 7 to
  8 s are loading the model for every sound. **`kera serve --sfx DIR`** keeps it loaded (the sound engine's own
  server, now built and shipped): 32 s a sound, and the samples are the same as `kera sfx`'s for the same words and
  seed, byte for byte (checked on the Spark).
- **Video (Wan 2.1 T2V-1.3B, 33 frames at 832x480, the makers' 50 steps, about 6 minutes):** 97% is the steps, 15 s
  the decoding, 1 s the loading. The engine's switches were each timed alone on the chip: its fast attention is
  already on by default (off is three times slower); running the two halves of the guidance in one pass and its
  direct convolutions change neither the time nor the output for this model. **Nothing found that is free**, so
  nothing is changed for video. What is left is work inside the engine's CUDA code for the transformer, which is
  open-ended and not started.
- **Not done, by rule one:** fewer steps, cached or skipped steps, lower precision, tiled or smaller decoding, the
  engine's `--sage-attn` (quantized attention).

## D-038: The Minisforum is sold; the DGX Spark is the owner's machine (2026-11-10)

Said by the owner (10 Nov: "the amd is getting sold bc I have the spark"). It changes D-001's order (Windows on the
Minisforum first, then Mac, then NVIDIA) and D-002's "the Minisforum only runs finished packages".

- **Where Kera is measured from now on:** the owner's DGX Spark (the agent's own login, one `kera` folder, no admin
  rights, as since 9 Nov), then the Mac. The order of work becomes the Spark first, then Mac.
- **Windows and AMD:** the Windows download and its ROCm add-on keep being built and checked by GitHub, but nothing
  can be measured on an AMD Ryzen AI Max machine any more. What was measured there stays in the records with its
  date and Kera version (preview 3); it can't be repeated.
- **What can no longer be finished:** the gates and tasks that need that machine (Gate 0 opened by the owner on
  Windows, the new speed comparison with Ollama and stock llama.cpp, the choice between Vulkan and ROCm per model
  and per message, the multi-user benchmark there). They are marked so in `docs/STATUS.md`, not dropped silently.
- **The speeds measured there with the flawed method** (each prompt asked three times on one server: 1.71 times
  Ollama) can't be measured again once the machine is gone. Unless the owner gives one last session on it, they are
  taken off the public page and marked as not repeatable in `docs/BENCHMARKS.md`. **Done the same day:** the owner
  said the machine is gone, so the writing-speed row and its chart left the Hugging Face page; the waits for the
  first word, which the flaw doesn't touch, stay with a note that they can't be repeated.

## D-037: The working branch is merged into `main`, and the supported models are listed publicly (2026-11-10)

Decided by the owner (10 Nov: "the source on GitHub merge it bc urs is separate to mine", and "on Hugging Face add a
list of supported models"). It replaces D-007's "nothing is merged into `main` until a gate passes".

- **`main` is the working branch's code.** `main` held only the first commit, and everything else sat on the agent's
  branch, so the repository's front page showed none of it. `main` is moved forward to the branch (no merge commit:
  the branch contains `main`), after each release and whenever the owner asks. Work still happens on the branch, and
  the gates still decide what a milestone's status says; they no longer decide what `main` holds.
- **Never with a release marker on top:** the release workflow starts for any pushed head commit whose message
  carries `[release v...]`, on any branch. `main` is moved only to a commit without one, so a release isn't built
  twice.
- **`docs/MODELS.md`** lists the models Kera runs and the ones it runs at their makers' settings, made by
  `tools/publish/models_list.py` from the tables Kera is built with (checked in CI). The Hugging Face page carries
  the file and a summary of it. It says which models were actually run by Kera's authors and which are recognised
  from their makers' published settings only.

## D-036: Every model runs with its makers' own settings by default, at its own precision where Kera chooses (2026-11-10)

The owner's requests (9 and 10 Nov: "auto detect the models' defaults", "make Kera default to the best settings
always", and a timing chart from another session whose "Kera best" column was Kera 0.1.3 with the full-precision
model files and the makers' settings). It extends D-013, D-033 and D-035: where the user sets nothing, a model runs as
its makers run it. What the user sets always wins, and nothing here trades quality for speed (D-034). The choices
below marked "the agent's" were made without asking; the owner can change any of them.

- **Music (ACE-Step 1.5).** Where a request sets none, the base and SFT models (XL too) run at guidance 7 and shift
  3, as ACE-Step's own interface and server run them (`GenerationParams.guidance_scale = 7.0`; shift 3.0 in its
  Gradio interface and its HTTP API); the music engine's own "auto" ran them without guidance at shift 1. Steps stay
  50 (ACE-Step's table for base and SFT; its interface starts base at 32: the agent's choice of 50, the engine's own,
  the higher of the makers' two). Turbo models keep 8 steps and shift 3 and get ACE-Step's DCW correction ("double",
  0.05 low / 0.02 high), which its own pipeline turns on for turbo models by default (the agent's choice: ACE-Step
  labels the option experimental but ships it on; its Gradio page uses 0.02 / 0.06 when the planner writes the
  codes, its pipeline and server 0.05 / 0.02, the values used here). `kera music --guidance` and `--shift` set
  others; a request's `dcw_scaler` of 0 turns the correction off. Measured on the owner's DGX Spark (10 Nov,
  `docs/QUALITY.md`): the correction doesn't cost sung words (84% heard with it, 81% without, nine songs each), and
  0.1.4's own choices give 95% of the lyrics against 75% for what 0.1.3 took by itself in the same folder.
- **Which music models.** Of a folder's planners and sound models, a song that names none is made with the ones
  ACE-Step's own guide recommends for the computer's graphics memory ("Which Model Should I Choose?": XL SFT and the
  4B planner from 24 GB, XL turbo or XL SFT and the 1.7B planner from 20 GB, SFT or XL turbo from 16 GB, turbo below;
  the 1.7B planner from 12 GB, the 0.6B below), at the highest precision that fits. The engine took the first by file
  name: the base model, the 0.6B planner, the 4-bit file. On the CPU alone the planner follows the computer's memory
  and the sound model the guide's smallest row (turbo: 8 model calls a song instead of 100; the agent's choice).
- **Pictures and video.** 53 more models are recognised by their makers' file names (whatever the precision): each
  gets the image engine's own recipe for it (CFG, guidance, size, flow shift, sampler, schedule: the engine's authors
  reviewed those against the cards) with the makers' own step count instead of the recipes' cap of 20, read from the
  cards and reference code on 10 Nov (`tools/recipes/makers.json` holds each number's source and quote). Recognised
  by the exact name only: a speed-up variant, a fine-tune under another name or a community model keeps the engine's
  defaults. Left out where the engine can't run the makers' recipe (HiDream-O1's own scheduler, Ideogram 4's guidance
  schedule), where one recipe covers models with different counts, and for adapters. Qwen-Image's flow shift is left
  to the engine (the recipe's 3.0 isn't in the card). Wan 2.2 S2V gets its config's 40 steps, guidance 4.5, shift 3.
- **Chat.** A file that carries no sampling of its own (older GGUF files, and many newer ones) and isn't one of the
  cards read by hand gets its makers' `generation_config.json` values: what transformers and vLLM apply when a
  request sets none, and what llama.cpp's converter writes into a file today. They come from a list built into Kera,
  collected from Hugging Face by `tools/cards/collect.py` for the models people run as GGUF files (the models its
  most downloaded and most liked GGUF repositories were made from, and its most downloaded and liked chat models).
  A model is found by its name and its shape together (layers and width), so a fine-tune under its own name gets
  nothing. Only the values the makers name are set; the rest stay llama.cpp's, exactly as for a newly converted file.
  Gated models (Llama, Gemma, Mistral) are listed only where a public copy holds the same file, byte for byte, and
  another organisation's copy of a model under the model's own name doesn't count as its makers'. On 10 Nov: 1,969
  repositories looked at, 759 with sampling values, 680 models in the list (103 gated ones had no such copy).
- **Not looked up online (the agent's choice).** Kera asks no server about a model: its downloads are built without
  HTTPS, and a lookup would tell a website which models a person runs. The list is refreshed with each release.
- **Precision.** A model's own weights are its full-precision file (BF16, F16). Where Kera chooses among a folder's
  files (music, sound effects) it takes the highest precision that fits. Where the user names a file with `-m`, Kera
  uses that file and says when the same model at a higher precision is next to it and fits this computer's memory
  (the agent's choice, asked of the owner on 10 Nov: say it, or switch to it).
- **Speech, 3D, sound effects: checked, unchanged.** The Qwen3-TTS files carry their makers' values (temperature
  0.9, top_k 50, top_p 1.0) and llama.cpp uses them; the 3D engine's 12 steps and guidance are TRELLIS.2's own
  `pipeline.json`; `kera sfx` already ran the model's own 100 steps and took the full-precision file first.

## D-035: The models' own settings where a card says more, or less, than the engine (2026-11-09)

Decided by the owner (9 Nov, asked in the session with the trade-off of each; every answer the recommended one). It
extends D-013 and D-033: where the user sets nothing, a model runs as its makers run it.

- **Steps for pictures and video:** the card's own step count, not the image engine's 20 (its recipes cap every model
  at 20): FLUX.1-dev and its variants, FLUX.2-dev, Wan 2.1 and 2.2 and LTX dev models get their makers' counts (the
  numbers and their sources are in `app/image_cli.cpp`). A picture or video takes 2 to 2.5 times as long; `--steps`
  still sets any number. Step-distilled models keep theirs (turbo 1, schnell 4, Z-Image-Turbo 9, LTX distilled 8).
- **Values a card leaves out:** for Nemotron 3 Nano (top_k and min_p) and Qwen3-Coder (min_p), the makers' own serving
  engines apply none, so Kera turns llama.cpp's own top_k 40 and min_p 0.05 off for them. Only values the card names
  are compared with a file's own (D-013).
- **Fine-tuning's length:** about three passes over the data (Hugging Face's Trainer default), from 10 to 3,000 steps,
  instead of a fixed 200 (80 passes over 10 examples, which makes a model repeat them and forget more). `--steps`
  still sets any number.
- **Other ways in:** `kera cli`, and `kera serve` with a model llama.cpp finds by itself (`-hf`, `LLAMA_ARG_MODEL`, a
  preset), get the card's sampling as `kera serve -m` does: `kera cli` runs llama.cpp's server inside it, and hook 8
  fills the card in once llama.cpp has the file.

## D-034: Kera has no modes that trade quality for speed; the three lossy modes are removed (2026-11-09)

Decided by the owner (9 Nov, on "Anything that can lower quality is never on by default. The only such things in
Kera are the three opt-in modes...": "THIS SHOULD NOT BE A THING"). It overrides the spec's Phase 11 and M12 (lossy
modes as opt-in features), D-017 and D-019, and D-003's "lossy modes stay off by default".

- **Removed:** `kera serve --lossy-kv8` (the chat's memory in 8 bits), `--lossy-route MARGIN` (a cached expert used
  instead of one a word chose, for streamed MoE models) and `--lossy-reuse SHARE` with `--lossy-reuse-min` (a
  passage the chat had read before copied to its new place instead of read again), their settings (`lossy.*`),
  their code (`src/kvstore/reuse.cpp`, the routing in `src/stream/session.cpp`, llama.cpp's hook 5 with
  `llama_memory_seq_cp_shift`), their tests and `tools/quality/quality.py`. What they measured is kept in
  `docs/NEGATIVE_RESULTS.md` N-017. Hook 7 (sharing a chat's start at the same positions, exact) stays.
- **The rule from now on** (AGENTS.md, rule number one): no mode, option or setting that trades quality for speed is
  added, on or off by default. Ideas listed as possible opt-in modes (D-019's QuantSpec, D-030's TAESD, VAE tiling,
  step caches) won't be built. The engines' own command-line options still pass through as they do in the stock
  engines (Kera doesn't block llama.cpp's or stable-diffusion.cpp's own options); Kera neither adds nor suggests any
  that lower quality.
- **Checked after the removal:** unit tests, the six tiny designs against the reference code (6/6 exact),
  streaming, saved chats, guessing ahead, the model card's settings and several users at once: all pass.

## D-033: Rule number one: Kera never changes how a model responds; it only makes things faster (2026-11-09)

Decided by the owner (9 Nov: "NUMBER 1 RULE WE DO NOT MESS WITH HOW THE MODEL RESPONDS THINKING ETC WE ONLY DO
SPEED"), after the agent had made the prompt improver turn off or cap a thinking model's thinking, stop answers
early and trim them.

- **The rule:** nothing in Kera changes a model's thinking, answer, length, wording or sampling, by default or as
  an option Kera adds for it. A model runs with its own chat template and its own card's settings (D-013), and
  thinks as it does in a chat. Kera's work is speed: anything it adds must give the same output, or differ by
  rounding only, checked against the reference (as D-030's tiled multiply was). (The opt-in lossy modes this
  first allowed were removed the same day: D-034.)
- **What changed because of it (work since 0.1.2):**
  - The prompt improver asks the chat model through Kera's own chat server, as a chat app does: its template, its
    card's sampling, its thinking, and its whole answer, used as written (only blank space around it removed;
    its thinking is kept apart, as for a chat app). Removed: giving a thinking model an empty thinking part,
    capping its thinking, stopping at the first blank line, trimming a long answer, the greedy sampling and the
    `--improve-no-think` option. Speed is kept by other means: a model the app already has loaded is asked where
    it is (no loading), and the chat server's own speed-ups apply (guessing ahead; its instructions kept on the SSD,
    so the next rewrite doesn't read them again).
  - **Seeds are left random where the engine leaves them random** (the owner, 9 Nov: "THE SEED IS RANDOM"). The
    prompt improver had asked with a fixed seed and kept its rewrites to hand the same one back for the same words;
    a chat asks with a new random seed each time, so both went. `kera 3d` passed seed 42 where trellis.cpp picks a
    random one, and `kera sfx` passed 0 where Stable Audio picks a random one: now a seed is passed only when the
    user gives `--seed` (the picture engine's own default, 42, and MOSS-SoundEffect's, 0, are theirs and apply as
    they are). `kera music --seed N` still sets both its steps' seeds, as its help says, only when given.
  - `kera music --no-rewrite` is removed: it turned off the music model's own rewriting of the description (its
    chain-of-thought caption step). Kera still shows what the model wrote.
- **Checked and kept:** following a model card's own recommended settings where the user sets none (D-013 for
  chat, the turbo image models' steps in D-027); guessing ahead (each word is still drawn from the model's own
  probabilities, but checking several guessed words at once can round a near tie the other way, as another batch size
  or other chats at once do in stock llama.cpp, so a greedy reply or a seeded sample can read differently from stock's;
  `--no-auto-guess` gives stock's exactly; N-003, N-018); the media engines' tiled multiply (on x86-64, where it was
  checked; N-018) and remembered engine choices (rounding only, checked); a seed the user gives; the warm-up picture
  (the next picture is byte for byte the same).
- **Found by an audit of every place Kera touches generation (119 places, 15 checked in depth) and a deep check with
  experiments against the stock engines, and fixed (9 Nov):**
  - The model card's sampling (hook 4) replaced values the user set for the server (`--temp`, `--top-p`, `--top-k`,
    `--min-p`, their `LLAMA_ARG_*` variables or a preset) on every request that set none. Now it fills only fields
    that neither the request nor the user's server settings set (llama.cpp's own record of them,
    `user_sampling_config`), the same rule llama.cpp uses for the file's own `general.sampling` values.
  - It was chosen by the model's design alone, so every `qwen3`/`qwen3moe` file got Qwen's card and every Nemotron-H
    file Nemotron 3's: fine-tunes and distills (DeepSeek-R1-0528-Qwen3-8B), Nemotron-H and Nemotron Nano 2 too. Now
    only the card's own models match, by design and name (Qwen's releases named "Qwen3...", Nemotron 3 by name), and
    only if the file's own defaults (`general.sampling`, from its `generation_config.json`) are one of the card's;
    any other file keeps llama.cpp's own defaults, as in stock.
  - Saved chats (hook 1) could come back where the memory they hold isn't what this run computes: after an update
    (no engine version in their identity), under another context size on a LongRoPE model (Phi-3, Phi-4-mini,
    MiniCPM3 pick other rope factors), with options set through `LLAMA_ARG_*` variables (only the command line was
    read), after a request ran with other adapter scales, or with a split model's later parts changed. Shown on
    small models: the restored chat's words differed from a fresh read and from stock. Now their identity is Kera's
    and llama.cpp's version, every part of the model by content, the settings as llama.cpp parsed them (wherever
    they came from, with whether a LongRoPE model uses its long factors) and the contents of the files they name; a
    request or `POST /lora-adapters` that changes adapter scales turns the SSD tier off for the rest of the run.
    Rechecked: each case now reads the chat again and gives stock's answer exactly.
  - `kera 3d -p` added its own words to the description (", a single object, centered, whole object in view, plain
    white background") before the picture was drawn. Now the picture is drawn from the user's words as written; the
    help says the 3D step reads one whole object on a plain background best.
  - `kera music` set every song to 30 seconds where the music model chooses the length itself (acestep.cpp's default
    is 0, "unset"); a length is now passed only when the user gives `--duration`.
  - `kera sfx` always passed a length (10 s): MOSS-SoundEffect's own default, but not Stable Audio's (its file's
    longest), and it rounded the user's number itself where the engine rounds differently on some inputs (0.15:
    Kera 0.2, the engine 0.1). Now a length is passed only when given, as typed.
  - Z-Image-Turbo got 8 steps: its card's "8 model calls" are 9 steps in stable-diffusion.cpp's count (its own
    Z-Image-Turbo recipe uses 9); now 9. FLUX.1-dev keeps the engine's own step count (Kera had passed the same 20
    but called it the card's).
  - MOSS-SoundEffect's remembered text encodings (D-032) were filed by the model file's name and size: now by its
    contents, this build and the back-end, and only with the file's own weights.
  - The sound and 3D help no longer suggest fewer steps or a lower resolution for speed.
- **Found by the deep check's second round (experiments against the stock engines and real model files) and fixed:**
  - **Models from Ollama's store** ran without their Modelfile: Kera passed only the model blob (and its image reader),
    so the system prompt, settings, stop words and adapters were dropped, and a blob with no chat template of its own
    (Ollama keeps it in its own Go form: gpt-oss, Magistral, Gemma 3n, CodeLlama and others) was prompted in
    llama.cpp's fallback ChatML, a format it wasn't trained on. Measured on Gemma 3 270M's real weights: 23 of 24
    questions right with its own format, 2 of 24 with ChatML. Now `kera serve` (and so the app's Run) reads the whole
    manifest: the Modelfile's SYSTEM goes first where a chat has no system message and its MESSAGE pairs after it, as
    Ollama adds them; its PARAMETERs (temperature, top_k, top_p, min_p, the penalties, stop words) fill what neither a
    request nor the user sets, over the card's; its ADAPTERs are loaded. Ollama's runtime options (num_ctx,
    num_predict) are not applied: they would shorten the model's memory or its answers. A blob with no chat template
    is refused with a plain message, never served as ChatML; shipping the makers' own templates for those models is
    left for later (each needs a test against Ollama's rendering).
  - **Nemotron 3's card** matched every Nemotron 3 by name, but only Nemotron 3 Nano 30B-A3B has it: Super, Ultra and
    the 3.5 releases recommend 1.0/0.95 everywhere and Nano Omni 0.6/0.95 thinking. The three most downloaded Nano Omni
    GGUFs got Nano's 1.0/1.0. Now only Nano 30B-A3B's names match (not Omni), and a file's own defaults are compared with
    the card's plain settings only (its generation_config.json describes plain generation, not tool calls).
  - **Qwen3-2507 Instruct** can't think, but a request with `enable_thinking: true` gave it Qwen3's thinking values.
    Hook 4 now asks whether this request's prompt leaves the model a thinking section.
  - **Qwen's GGUF cards** for Qwen3 recommend presence penalty 1.5 for quantized files "to suppress repetitive outputs";
    Kera applied four of the five values. It now applies it to quantized Qwen3 files (not the 2507 releases, whose cards
    leave it optional, nor full-precision files), where neither a request nor the user sets one; `--presence-penalty`
    now counts as the user's own setting.
  - **Image readers:** a folder's only reader beside its only model was attached without any check, so a text model
    could get another model's reader and describe pictures it can't see (a blue square became "a lot of repeated
    symbols"), or a reader of the wrong width stopped the server. Now it is taken only on evidence it is this model's:
    its file name names no model ("mmproj-F16.gguf"), or its own name (general.name, written by today's converter) is
    the model's or one of the model's base models'. Otherwise the model stays text-only, as in stock llama.cpp, and
    Kera says which reader it passed over. (A width check before loading needs a llama.cpp helper; left for later.)
  - **A guessed word's probabilities** were reported as 1 with no alternatives (an upstream gap, fixed upstream on 20
    Oct after Kera's llama.cpp version; Kera guesses by default, stock doesn't). Upstream's fix is ported
    (docs/UPSTREAM_PATCHES.md); a request for post-sampling probabilities doesn't guess.
- **Found by the deep check's third round (each problem re-checked by an agent trying to disprove it; 30 of 31 real,
  none high) and fixed:**
  - **Pictures:** the card's size too (1024x1024 for FLUX, Z-Image-Turbo and SDXL-Lightning; the turbos keep 512);
    SDXL-Lightning's Euler with trailing steps (`sgm_uniform`), and its 1-step x0 file, which the engine can't run as
    made, is named rather than given settings; FLUX told apart by its weights (a `guidance_in` tensor means
    guidance-distilled: no CFG and the variant's own guidance, Kontext 2.5, Fill and Canny 30, Depth 10, Krea 4.5,
    FLUX.2 4; without it, de-distilled: CFG kept, 3.5 if recognised, else the engine's); steps and size only for text
    to picture (with a starting picture a turbo's 1 step makes the engine ignore it).
  - **Video:** Wan and LTX get their makers' settings, as the engine's own recipes give them (Wan 2.1: CFG 5; Wan 2.2
    T2V: CFG 3 and 4 for the high-noise half, flow shift 12; I2V: 3.5 and 3.5, expert boundary 0.9; TI2V-5B: CFG 5;
    FLF2V and VACE: flow shift 16; Wan 2.1 I2V at 832x480: flow shift 3; LTX 2.3 distilled: 8 steps, no CFG; LTX 2.3
    and 2.5: CFG 3), and Wan's own default negative prompt when the user gives none, shown. Speed-up variants of Wan
    (step-distilled, Fun, SkyReels...) keep the engine's. Steps stay the engine's (FLUX.1-dev's 50 and Wan's 40-50
    are the owner's call: they make a picture or video 2 to 2.5 times slower).
  - **3D and sound:** BiRefNet is needed (TRELLIS.2 always removes a background with it; without it a drawn picture
    was cut out by brightness), and a drawn picture asks for it outright; `--seed` takes 1 to 2147483647 (the engine
    reads 0 as "pick one"); `kera sfx` takes the full-precision MOSS file before the 8-bit one; its text cache is keyed
    by the engine's own build and the graphics chips. `kera speak` and vision models take the reader whose precision
    is nearest the model's (llama.cpp's own rule).
  - **Music:** `--keep-loaded` is added only for a folder with one LM and one DiT (with several, an unnamed request
    got the model used last); songs are written as WAV unless the user asks for MP3 (it was 128 kbps MP3).
  - **Fine-tuning:** a prompt/completion row is put in the model's chat format when it has one, as MLX-LM does, so
    both trainers learn the format the chats use; an ESFT model keeps its base's name ("... ESFT"), so Kera knows it.
  - **`kera wait`** no longer advises turning thinking off, and times the whole reply (no 256-word cap).
  - **Image readers** are filled in by llama.cpp hook 8 after it has read the command line, `LLAMA_ARG_*` variables
    and config.ini, so a reader named or turned off there is respected.
  - **Streaming:** a failed read stops the request with an error instead of crashing the server; a streaming setup that
    fails fails the load instead of answering from zeros; a short read of a shared page is caught.
  - **Engines on ARM:** the tiled multiply only on x86-64, where it was checked (on ARM it adds in 8 half-precision
    lanes, about twice stock's rounding).
  - **Checks and claims:** the GPU exactness check no longer lets swaps of scores up to 0.02 apart pass; `--kv q8_0`
    works there. The published speeds asked each prompt three times on one server, so Kera's guesser could copy its
    own earlier answer: the comparison tool now uses a fresh server per run and the numbers are marked until measured
    again. "Exactly the model's own answers", "guessing ahead (exact)", the accuracy page's "one answer changed" and
    "about twice as fast" for songs are corrected. CI now compares Kera with stock llama.cpp at the defaults on a real
    quantized model, guessed words included.
- **Also found:** this cloud machine's own build had CPU kernels from the compiler cache (ccache), compiled weeks
  earlier for another processor, which made Kera's answers differ from stock llama.cpp's by rounding from the first
  word (N-018). Not in any download (they pick CPU variants at run time). Rebuilt, Kera with its defaults (guessing
  ahead, saved chats, the card's sampling) gives exactly stock llama.cpp's greedy answer: Qwen3-0.6B, an 899-word
  prompt, 150 words, every log-probability identical.

## D-032: Engines that start warm, faster sound-effect loading, a fourth bug review (work in progress, 2026-11-09)

Asked for by the owner (8-9 Nov: sound effects working "with a lot of optimisation", "loading times faster", the
device warm from the first picture instead of after three or four). The design is the agent's; not yet measured
on a graphics chip.

- **Built programs kept:** the picture, music, 3D and sound engines' Vulkan back-ends get the pipeline cache file
  the chat engine has had since N-011 (one file per engine), so a start doesn't rebuild every shader. Not yet
  built for or tested on a graphics chip.
- **Warm-up:** `kera serve` with an image model makes one small throwaway picture after it starts, on computers
  with a graphics chip (setting `image.warmup`, on at the owner's request); the next picture is byte for byte the
  same as without it. To be replaced by warming while the model loads and keeping the engines running (proposed
  to the owner), and measured on the Minisforum or the Spark first.
- **Sound effects:** MOSS-SoundEffect's text encodings are kept per model in the Kera folder
  (`AUDIOCPP_TEXT_CACHE`, a change in audio.cpp): a description used before skips loading the 2 GB text encoder,
  and the encoder is freed before the sound is made. The same sound. A first real sound was made on the cloud CPU
  (4 s, 10 steps, 6 min; the 100 steps the model card uses take about an hour on 4 cores: a graphics chip is
  needed).
- **The fourth bug review:** fixes listed in `CHANGELOG.md` (Unreleased).

## D-031: Kera 0.1.2 makes 3D models and sound effects (2026-11-08)

Asked for by the owner (8 Nov, version 0.1.2: "support for making 3-D modelling", "images music sound effects", all
fast and with the same good quality). The design is the agent's; the choice of engines follows a research pass
(no Python at run time, permissive licences, a C++ engine on ggml that runs on every download's graphics chip).

- **3D:** a fourth engine, trellis.cpp (MIT: Microsoft's TRELLIS.2 in C++ on ggml), copied into `third_party/` at a
  pinned commit with its own ggml fork and built like the music engine into the downloads' `3d/` folder, without
  its optional quad retopology (CGAL, GPL) and WebP textures (PNG instead). `kera 3d -i picture.png -o model.glb`
  removes the background (BiRefNet, or by brightness without it), reads the picture with DINOv3, runs the three flow
  models, simplifies, unwraps and paints the mesh: a textured `.glb`. `kera 3d -p "text" --image-model M` draws the
  picture with the image engine first (from the user's words as written, D-033) and keeps it next to the model. `--resolution 512|1024`
  (1024 by default, as upstream), `--seed` (42 by default, so a run repeats; since D-033 none by default, so each run gets a new random seed as trellis.cpp's code does, and `--seed` takes 1 to 2147483647, since the engine reads 0 as "pick one"), `--no-texture`. The weights (about
  5 GB at Q4, huggingface.co/ilintar/trellis2-gguf) are the user's download in Kera's `models/3d`; DINOv3 comes
  under Meta's DINOv3 License, which the user accepts by downloading it. Kera ships none. Not chosen: Hunyuan3D
  (its licence excludes the EU, the UK and South Korea), Pixal3D (gated in the EU), and porting a model ourselves
  (weeks; a maintained port exists).
- **Sound effects:** a fifth engine, audio.cpp (Apache-2.0), with only its sound-effect models, built the same way
  into `sfx/`. `kera sfx -p "door creak" --duration 4 -o door.wav` uses MOSS-SoundEffect v2 (Apache-2.0, 48 kHz, up
  to 30 s, no sign-up) by default; Stable Audio 3 Small SFX also runs, under Stability AI's Community License (the
  help says so). The model is the user's download in Kera's `models/sfx`. ACE-Step (the music engine) makes music,
  not effects.
- **Both:** ggml's tiled multiply on (D-030); the physical core count as threads; each command's help names the
  exact files, where they come from and their licences, and a missing file is named before an engine starts.
  `KERA_3D` and `KERA_SFX` (on by default) build them; every download carries them with a `--help` check.
- **The same output, faster, for 3D:** what is exact is kept (models stay loaded within a run, a repeated seed
  repeats). FlashVDM-style decoding, "turbo" checkpoints and mesh decimation by default all change the model, so
  none is used; each could become an opt-in with a quality check.
- **Changes to upstream:** `docs/UPSTREAM_PATCHES.md` (a shim that lets trellis.cpp run with ggml's run-time-loaded
  CPU variants; library paths that work on macOS and the Spark; audio.cpp's build accepting the models left out).
- **Tests:** `tests/integration/test_3d_sfx.py` in CI (the commands, both engines starting, the messages for missing
  models and bad values). A real sound and a real 3D model aren't made in CI or here: the weights are 4 GB and
  more, and 3D needs a graphics chip; the test makes them when given the weights (`--sfx-model`, `--3d-models` and
  `--picture`).
- **Not a gate:** new abilities. A first real model and sound, and their speed, on the Minisforum (Vulkan), the Mac
  (Metal) and the Spark (CUDA) come with the next hardware runs.

## D-030: Kera 0.1.2: faster pictures and music with the same output, a prompt improver, and a third bug sweep (2026-11-08)

Asked for by the owner (8 Nov: prepare 0.1.2; much faster pictures, music and sound "with the same good quality",
"do not harm the quality of the output ever"; a prompt improver; "the drawing part of image making is slow"; fix all
the bugs, including someone's report that only one user could get in). The design is the agent's. 3D models and
sound effects are D-031.

- **The image and music engines get ggml's tiled matrix multiply** (`GGML_LLAMAFILE=ON`, llamafile's sgemm). llama.cpp
  turns it on for the chat engine; ggml on its own leaves it off, so the two engines built without it, and their
  convolutions ran one dot product at a time. On the cloud CPU, warm:
  - a SD-Turbo picture (512x512, 1 step) takes 20.8 s instead of 38.5 s; its VAE decode, the "drawing" step,
    takes 14.7 s instead of 29.8 s;
  - a 10 s song takes 35.7 s instead of 57.8 s; its VAE decode takes 9.0 s instead of 16.0 s.
  The sums are the same products added in another order, so the output changes by rounding only. Checked against
  the reference (`bench/results/cloud/2026-11-08-engines-tiled-multiply.json`): with the full-precision SD-Turbo
  file the six pictures are 46.9 dB from diffusers' against 46.7 dB before (closer), and 56 dB from the old build's;
  with the 8-bit file 27.2 dB either way. The song made from the same codes is 2.1 dB from the old one in its
  spectrum, against 21.5 dB for another seed, at the same loudness. N-015's "can't be made much faster" was wrong:
  this wasn't tried. It speeds up the CPU only (the parts that run there: everything on a machine without a
  graphics chip, the text encoders where they are put on the CPU); graphics chips are unchanged.
- **The same seed, the same picture:** the image engine picks flash attention and direct convolutions by timing both
  on the graphics chip each time it starts (Vulkan except NVIDIA RTX, and Metal), and a close race could go either
  way, changing the picture. Its first result is now remembered per device in the Kera folder
  (`image-engine-choices.txt`, a small change in the engine, `docs/UPSTREAM_PATCHES.md`). The CPU's choice was
  already fixed.
- **The same seed, the same song:** the music engine picked its language model's seed at random each time, so
  `kera music --seed N` gave different lyrics, key and codes each run. Kera now passes the seed to both steps.
- **The image server loads its model when it starts** (`--eager-load`), not on the first picture, so the first
  request takes as long as the next. It already keeps the model and the last 4 encoded prompts between requests.
- **The music server keeps its models loaded between songs** on computers with 32 GB or more (`--no-keep-loaded`
  frees them after each): about 5 s a song on the CPU (N-015), more where they go to a graphics chip, for 3-6 GB
  held. The same songs.
- **The prompt improver** (`kera image|video ... --improve-prompt CHAT_MODEL.gguf`, opt-in): a chat model rewrites
  the description into a fuller one, keeping everything it says, before the image engine sees it: sentences for
  the models that read them (FLUX, Z-Image, Qwen-Image, SD3, video models), comma-separated phrases for SD 1.x/2.x
  and SDXL. Kera's own instructions; greedy, so the same words and model give the same rewrite; remembered in the
  Kera folder (`prompt-improver.json`), so a repeat needs no chat model. Both texts are shown. It changes what is
  drawn, so it is never on by itself. `kera music` already rewrites its description with its own language model:
  Kera now shows what it wrote, and `--no-rewrite` turns that off.
  **Later (D-033):** the greedy sampling, and `--no-rewrite`, went: the chat model now answers through Kera's chat
  server exactly as in a chat, and the music model always rewrites as it was made to.
- **Not taken, because the output changes:** a smaller VAE (TAESD), VAE tiling (each tile is normalised on its
  own), the engines' step caches (EasyCache, cache-dit and the like), flash attention where the engine doesn't choose
  it, fewer steps. None of them will be added (D-034).
- **The third bug sweep:** N-016 (two reviews of every file; the one-user report traced to starting Kera twice).
- **Tests:** `tests/integration/test_image.py` checks the prompt improver (SmolLM2-360M: both texts shown, the same
  rewrite and picture twice, remembered). Unit and integration tests pass on the CPU.
- **Not a gate:** the speed-ups are measured on the cloud CPU; the Minisforum and Mac measurements come with the
  next hardware runs.

## D-029: Kera for the DGX Spark: a download, its setup and its tests, before there is a Spark (2026-11-07)

Asked for by the owner (7 Nov: "Update Kera for the Dgx spark"). This brings the NVIDIA part of the order (Windows,
Mac, then NVIDIA) forward for what can be built and checked without the machine. There is no Spark here, and buying
or renting one is the owner's call, so spec 10.11's steps S0-S8, which are measurements on a Spark, still wait for one
(STATUS, "Waiting for the owner", item 7).

- **A Spark download:** `kera-VERSION-linux-arm64-cuda.tar.gz`, built on GitHub's ARM machines in NVIDIA's CUDA
  13.0.3 image (`build-linux-arm64-cuda.yml`) for the GB10's chip only (sm_121a), with the CPU code in one copy per
  ARM generation (the best picked when it starts). The image and music engines are built with CUDA too. CUDA 13.0 is
  the oldest CUDA the GB10 ships with, so the download runs on any CUDA 13 driver; it uses DGX OS's own CUDA
  libraries (cudart, cuBLAS) instead of carrying about 1 GB of them, and runs on the CPU, saying why, without them.
  Full releases build it; a commit marked `[spark]` builds it in CI.
- **Setup knows a Spark:** llama.cpp's CUDA back-end reports the GB10 as a separate card (its "integrated" flag is off
  upstream after corrupted output, llama.cpp #15034), so Kera recognises NVIDIA chips that share the computer's memory
  itself: by name (GB10, Thor, Orin) or by a CUDA chip that may use more than three quarters of the memory
  (`nvidia_shares_memory`, tested). On a Spark a model may use about 110 GB (all but 10 GB, spec section 2's
  "about 110 GB, headless") instead of 80% of the memory.
- **Loading (spec 10.6):** on such a chip `kera serve` reads the model into memory instead of memory-mapping it
  (llama.cpp's `--load-mode none`): 56.3 against 44.7 tok/s was measured on a Spark (spec 10.6, which says to never
  memory-map there). It changes no answer. The setting `gpu.shared_no_mmap` is on by default because the spec
  prescribes it for the Spark; S0 on a real Spark confirms it (and tries `--load-mode dio`). Streamed experts are left
  as they are: Kera's cache reads them itself.
- **Setup advice (spec 10.6), in plain words, never changed by Kera:** DGX OS 7.5.0's KHO memory cut (about 112 of
  119.5 GiB; NVIDIA's update or `kho=off`), `vm.swappiness` above 10, a running desktop (2-3 GB), and NVIDIA's driver
  present while CUDA's libraries are missing.
- **Tests:** the setup rule is a unit test; `kera serve` on a GB10 profile was checked by hand (the message, then
  answers). A `[spark]` commit also runs Kera on an ARM CPU like the Spark's: unit tests, all six tiny model designs
  exact, streamed experts, saved chats and guessing (`ci.yml`, job `linux-arm`).
- **Built with GCC 14** (ggml's CPU variant for the newest ARM chips needs `+sme`, which GCC 13 rejects; Kera's own
  code builds with GCC 14 without a warning). The image engine is built **without CUTLASS** (NVIDIA's library for
  faster 8-bit maths in stable-diffusion.cpp, a submodule not copied in): adding it is for after S0, measured.
- **Checked for a first run on a Spark** (before release): the app listens on the Spark itself at port 8341 and
  answers only at that port, so the read-me's tunnel uses the same number on both sides (`ssh -L 8341:localhost:8341`;
  it first said 8080, which the page would have refused); `./kera --no-browser` prints the address. The CUDA parts
  look for CUDA's libraries in `/usr/local/cuda` too (RUNPATH), in case DGX OS doesn't put them on the library path,
  which would otherwise leave everything on the CPU; the Spark build checks this. Unchecked until a Spark: speed, and
  experts streamed from the SSD with CUDA (tested on the CPU, Vulkan and ROCm, never on CUDA).
- **Not built yet (they need the machine):** the I/O engine's 16 MiB-in-flight tuning (S1), the GB10 kernels (S2),
  the overlap schedule (S3), the cache brain (S4), and the heat cap (`nvidia-smi -lgc`, which needs admin rights).

## D-028: Kera 0.1.1 also makes music, speech and video; a speed pass and a second bug sweep (2026-11-07)

Asked for by the owner (7 Nov, before 0.1.1 was released: hold the release; "support music generator video generator
audio generator"; leave fine-tuning as it is; "a massive optimisation that everything faster"; "fix all crucial
bug"). The design is the agent's. Fine-tuning is unchanged.

- **Music:** a third engine, acestep.cpp (MIT: ACE-Step 1.5 in C++ on ggml), copied into `third_party/` at a pinned
  commit with its own ggml fork and built like the image engine, into the downloads' `music/` folder.
  `kera music -p "description" [--lyrics TEXT|@FILE] [--duration S] -o song.mp3|.wav` runs its two steps (a
  language model writes the lyrics, unless given, and the song's codes; a diffusion model and a VAE make 48 kHz
  stereo). `kera serve --music DIR` runs its server with its own page for making songs. The models (4 files, 3.3 GB
  at the smallest, from Hugging Face) are the user's download, in `~/.kera/models/music` by default.
- **Speech:** llama.cpp's own speech tool (with libmtmd), linked into kera: `kera speak -m MODEL -p TEXT -o out.wav`
  with Qwen3-TTS (ten languages, a voice copied from a short sample with `--tts-speaker-file`) or Pocket TTS. The
  model's audio decoder (its mmproj file) is found next to it, by the rule vision models use.
- **Video:** the image engine's video mode: `kera video` makes a short .avi with Wan 2.1/2.2, LTX-Video and the
  others stable-diffusion.cpp runs, with the model card's sampling where the command line sets none.
- **Speed, measured on the cloud CPU (4 cores, one thread each, AVX-512):**
  - The music engine used half the logical CPUs, which is half the cores where a core runs one thread (Apple
    Silicon, many laptops, virtual machines): Kera now tells it the real count. A 10 s song took 40 s instead of
    71 s (1.8x).
  - The image and music engines are now built with ggml's CPU variants in the downloads, as the chat engine is: the
    music model's diffusion step took 5.8 s instead of 9.9 s with AVX-512 used, and the engines no longer need AVX2
    to start. A picture took the same (sampling 6.0 s either way). The cost is 18 MB more per engine.
  - Tried and not taken (N-015): the image engine's direct convolutions (no change: 28.1 s against 28.2 s for SD-Turbo's
    VAE), its flash attention (1.5 s sooner but a different picture, so not lossless), and the music server's
    `--keep-loaded` (saves about 5 s of loading per song but holds 3 GB or more; left to the user until measured on the
    Minisforum, P7). On the CPU both VAEs are bound by arithmetic, about 2 trillion operations for 10 s of music or
    a 512x512 picture: the graphics chip is what makes them fast.
- **The second bug sweep:** N-014's second part.
- **Tests:** `tests/integration/test_media.py` (Qwen3-TTS 1.7B: a WAV with sound in it and the decoder found by
  itself; ACE-Step's smallest set: an 8 s stereo 48 kHz WAV and an MP3, `kera serve --music` with its page;
  `kera video --help`), in CI with the models cached.
- **Not a gate:** new abilities and CPU measurements; their speed on the Minisforum's graphics chip and the Mac is
  measured with the next hardware runs (the music and image engines on Vulkan, `--keep-loaded`).

## D-027: Kera 0.1.1: models that see and make images, faster first loads, and a bug sweep (2026-11-07)

Asked for by the owner (7 Nov: "version 0.1.1 ... new ways for running image models, faster model loading, scan
all the bugs, fix all the bugs"; both kinds of image model). The design is the agent's.

- **Models that see images:** llama.cpp runs a vision model as two files, the model and its image reader (a GGUF
  whose design is "clip"), and sees images only when told where the reader is. `kera serve -m model.gguf` now finds
  it (`src/planner/projector.h`): next to the model (the closest name if there are several), or, for a model in
  Ollama's store, the projector its manifest lists. `--no-mmproj` turns it off. The app lists Ollama's models by their
  Ollama names (used where they are) and marks the ones that see images. Images come through OpenAI's and
  Anthropic's APIs as before; chats with images aren't saved to the SSD (as before).
- **Models that make images:** a second engine, stable-diffusion.cpp (MIT, copied into `third_party/` at a pinned
  commit with its own ggml fork, built as a separate project into the downloads' `image/` folder so its ggml never
  meets llama.cpp's). `kera image ...` runs its command-line program; `kera serve -m MODEL` serves an image model with
  it (OpenAI's `/v1/images/generations` and `/v1/images/edits`, and AUTOMATIC1111's `/sdapi/v1/txt2img`), on the
  port and address options chat models use. An image model is recognised by its file: `.safetensors`/`.ckpt`, a GGUF
  naming a diffusion design, or a GGUF llama.cpp can't read (stable-diffusion.cpp's own allow longer tensor names).
  As D-013 does for language models, the model card's sampling applies where the command line sets none (SD-Turbo
  and SDXL-Turbo 1 step, FLUX.1-schnell 4, Z-Image-Turbo 9 (corrected from 8 on 9 Nov, D-033), all without guidance; FLUX.1-dev: the engine's own steps): without it a turbo
  model took 4 min 25 s and came out over-cooked on the cloud CPU, with it 33 s and right. It runs on Vulkan
  (Windows, Linux), Metal (Mac) or the CPU; the ROCm add-on isn't used for it yet.
- **Faster first load:** the first time Kera sees a model, it hashes the whole file for the saved chats' identity.
  That ran while the server loaded the model, so on a cold start the two shared the drive: a 4 GB model was ready
  after 3.5 s against stock llama.cpp's 2.6 s. The hash now waits until the server answers; Kera is ready in 2.6 s,
  as stock (on the Minisforum's 23 GB model, about 10 s sooner the first time). llama.cpp's direct-I/O load mode
  (`--load-mode dio`) is to be measured on the Minisforum's graphics chip before it is used by default.
- **The bug sweep:** N-014.
- **Tests:** `tests/integration/test_vision.py` (SmolVLM-256M, its image reader found by itself, a colour named
  through both APIs, `--no-mmproj`) and `tests/integration/test_image.py` (SD-Turbo at 256x256: recognised, the
  model card's sampling, a PNG of the size asked for, the same seed the same picture, `kera image`), in CI.
- **Not a gate:** these are new abilities, not speed claims; image generation's speed on the Minisforum and the Mac
  is measured with the next hardware runs.

## D-026: Kera is published on Hugging Face, open source under Apache 2.0 (2026-11-04)

Decided by the owner (4 Nov: publish Kera on Hugging Face, public, with the downloads and the source code, under
Apache 2.0, by Kera Labs).

- **What:** a public Hugging Face model repo under a Kera Labs organisation, with each release's downloads (Windows,
  its ROCm add-on, Linux, the Kera-vs-Ollama Mac app), the source code of the commit it was built from, and a page
  (`tools/publish/huggingface_README.md`) with the measured results so far. `tools/publish/huggingface.py` builds and
  uploads it with a token the owner keeps in the cloud environment (`HF_TOKEN`), never in the repo.
- **The licence** changes from "all rights reserved" (D-009's choice, which left the owner free to choose later) to
  the Apache License 2.0, copyright Kera Labs: anyone may use, change and distribute Kera, with a patent grant and
  the notices kept. Third-party code keeps its own licences (`NOTICE`).
- **The source is a snapshot, not the git history:** an old commit holds a path with the owner's Mac user name
  (waiting for the owner, the git-history item), so the history isn't published.
- **The GitHub repo was private when this was decided** (D-004): only the Hugging Face copy was public then.
  **Superseded on 18 Nov 2026:** the owner made the GitHub fork public and cancelled D-004.
- **Model weights are never published** with Kera; each model keeps its own licence.

## D-025: The owner's MacBook for a Kera-vs-Ollama test app (2026-10-29)

Asked for by the owner (29 Oct): use the 128 GB MacBook Pro, with a small Mac app to open and press "Start test",
which downloads a mixture-of-experts model and races Kera against Ollama for 10 minutes, live, on context, coding and
reasoning. This lifts, for that MacBook, the rule that the agent touches no hardware but the Minisforum (owner
decision 6); the other never-do items stand.

- **What:** `tools/compare/mac-app`, a SwiftUI app built by `.github/workflows/mac-test-app.yml` on GitHub's Mac
  machines (only by hand or for a commit marked `[macapp]`, D-011) with `kera` inside it, signed ad hoc. Start test:
  the Ollama already running (else the installed one, else Ollama's own download into the app's folder, on a
  private port), the model pulled through Ollama (Nemotron 3 Nano 30B or Qwen3 30B-A3B), and `kera serve` started on
  Ollama's own copy of the file, so both run the same weights. Then, for 3, 10 or 20 minutes, both get the same
  messages one at a time, taking turns at going first: a chat message, code, a sum (the answer checked), a 16K-token
  log with a question (checked) and a follow-up (checked); greedy, thinking off, a 32K-token memory each.
- **Measured the same way on both:** the wait for the first token on the app's own clock; writing speed by each
  engine's own count (llama.cpp's timings, Ollama's eval counts); results saved as JSON with both servers' logs.
- **Not a gate:** it shows the owner the difference on their own Mac. The Mac's gates (spec Phase 9, M9 and M10's
  Metal work) still need their own measurements.

## D-024: kera serve --backend auto chooses the back-end per message, not per model (M8) (2026-10-18)

Asked for by the owner (18 Oct, on the chart of the 95K-word document: Kera should detect which back-end is best for
every message and use it); the design is the agent's.

- **What:** `kera serve --backend auto` (or no `--backend` once the setting `gpu.auto_backend` is on) runs the model
  as one engine per back-end, each its own `kera serve` on a private port, and answers on the asked-for port itself.
  Each request goes to the engine with the shorter expected time: D-022's speed model for what the request must read
  and write, less what that engine already holds, plus half the remaining time of the work it has in hand. The
  router remembers each engine's prompts as chained hashes of 1 KB pieces, so a follow-up stays where its document
  was read, and an assistant's message right after a prompt it served counts as held there (it wrote it). Tokens
  are estimated from the prompt's bytes, the bytes per token learned from the replies to long prompts. Each response
  says which back-end answered (`X-Kera-Backend`). `--backend auto-model` keeps D-022's one choice per model.
- **Two loaded, or one at a time:** with memory free now for two copies (2 × (the file + 2.5 GB) + 4 GB, by what the
  OS can hand out without paging anything, so the PC's other programs keep theirs), both stay loaded. Otherwise one
  is loaded at a time, and the other is loaded in its place, once the loaded one has finished what it is answering,
  when a request is expected to finish sooner there even counting the loading (a long document goes to ROCm at once),
  or once the time the other would have saved since the last change adds up to its loading time (a run of chats
  brings Vulkan back). Loading times are measured as the engines load. On the Minisforum, where the owner's services
  use about half of its 62 GB, Nemotron 3 Nano gets one at a time.
- **Why not one process with both back-ends:** llama.cpp's server holds a model on one set of devices, and each
  back-end keeps its own copy of the weights and its own chat memory, so a reply can't move between them halfway.
  Sharing one copy of the weights between two engines (both back-ends can read host memory on this chip) would make
  two loaded engines cost one model's memory; it needs a change to llama.cpp's ROCm back-end, which can't take a
  buffer from host memory today, and a measurement of how fast weights read that way are. That is the next step if
  loading one at a time proves too slow.
- **Answers:** each engine is a whole `kera serve` (saved chats kept per back-end, D-022), so every answer is that
  back-end's own; a chat that moves to the other back-end is read again there.
- **Checked in the cloud:** the policy's unit tests, with speeds shaped like the Minisforum's (chats to Vulkan, a
  127K-token document to ROCm, its follow-up staying there, a busy engine counting against itself; one at a time:
  ROCm loaded for the document but not for a medium prompt or while Vulkan is busy, and Vulkan back after about 20
  chats, not while ROCm is answering someone), and an end-to-end test on two tiny models in both ways (the CPU twice,
  with set speeds): each message to the engine expected, answers word for word a plain `kera serve`'s through
  OpenAI's, Anthropic's and llama.cpp's APIs, streamed and not, refusals passed on as sent, and the engines stopping
  with `kera serve`, also when it is killed outright.
- **Gate, to turn the setting on** (with D-022's): on the Minisforum with Nemotron 3 Nano, `tools/compare/mixed_day.py`
  (5 short chats, a 64K-token document with 3 follow-ups, 5 more chats): choosing per message takes less time over
  the day than Vulkan alone and than ROCm alone, and is within 10% of the faster one for each kind of message, with
  the owner's services keeping their memory.

## D-023: Serving several people at once is now a goal (2026-10-17)

Decided by the owner (17 Oct: add vLLM's benefits for serving several people at once, "so Kera has it all"). This
reverses one of the spec's non-goals ("Serving many users at once on datacenter GPUs. vLLM, SGLang and TensorRT-LLM
own that") and the first item of section 12.4 ("Server tricks that only pay with many users").

- **What llama.cpp's server already does** (b11619, read in its code): four chats at once by default (`-np` left at
  auto), in one shared chat memory, with continuous batching (a new chat joins the running batch at the next step);
  a new request goes to the free slot whose last chat shares the longest start with its prompt; recent chats'
  states are kept in RAM; several answers per request (`n`) share their prompt; Prometheus metrics with `--metrics`.
- **What vLLM and SGLang add that can matter on one machine:** a start several running chats share (a system prompt,
  a document many people ask about) read once, not once per chat (vLLM's automatic prefix caching, SGLang's
  RadixAttention); long prompts read in pieces between the other chats' words (chunked prefill), so one person's
  long document doesn't stop everyone else's writing; scheduling (priorities; guessing that backs off when many
  chats share the chip, since each guess check then costs more); and the load visible (metrics).
- **Rules that still hold:** each piece is a setting that stays off until its gate passes on the Minisforum (P7),
  measured against Ollama and stock llama.cpp serving the same number of people; answers follow section 6's rule
  for graphics chips (batching chats together can round a near tie the other way, as guessing can, N-003), and
  that is measured too.
- **Built (17 Oct), off until measured:** `kera serve --multi-user`, or each piece by its setting (llama.cpp hook 7,
  `src/planner/chat_share.h`):
  - the prompt with the fewest tokens left is read first (`serve.short_first`), so a new message goes before the
    rest of someone's long document;
  - while others write, prompt reading goes in steps of about half a second (`serve.read_step_s`, sized from the
    measured time per token), and a long prompt takes at most half the chip's time (`serve.read_share`); a new
    message isn't held to that share, only to the step size. With nobody else writing, prompts get whole batches
    as upstream. Off for streamed experts, where every step reads them from the SSD again;
  - a start a new chat has in common with another chat's memory is shared, not read again (`serve.share_prefix`):
    the two chats point at the same cells. When several chats with the same start come at once (agents with one
    system prompt), one reads it and the others wait for that, then share it. In a hybrid model like Nemotron 3 Nano
    the recurrent part of the memory changes with every word, so a chat shares only up to a checkpoint the other
    made (at a user message's start, or a few tokens before its prompt's end) and takes the recurrent state saved
    there. Not with sliding windows (gpt-oss, Gemma) or a draft model or MTP head guessing;
  - a request's `priority` (lower is sooner, as vLLM's): of the requests waiting for a free slot, and of the
    prompts waiting to be read, a lower one goes first, so a person chatting can go ahead of a background agent;
  - llama.cpp's load and speed counters at `/metrics` (`serve.metrics`).
- **Checked in the cloud** (`tests/integration/test_multi_user.py`, two servers with and without `--multi-user`, the
  tiny models): the second chat's 300-token start shared (all of it on the dense, DeepSeek-style and GLM-style models,
  294 tokens up to the checkpoint on the two hybrid ones), of three chats sent at once with one start two shared it,
  the new message answered first, of two requests waiting for the one slot the one with the lower priority answered
  first (in turn without `--multi-user`), and every answer word for word the same, with guessing on and off.
- **Gate, to turn each on by default:** on the Minisforum with 1, 2, 4 and 8 people, against stock llama.cpp and
  Ollama serving as many: while one person's long document is read, the others keep at least 40% of their writing
  speed and the document takes at most 2.5 times as long as alone; with no long prompts, no slower than without
  (within 3%); and a second chat with the same long start answers sooner than reading it would allow.
- **Not built yet:** guessing that backs off when many chats write (the Minisforum run measures whether it pays
  first), and taking a running chat's memory away for a more urgent one (vLLM's preemption).

## D-022: kera serve measures Vulkan and the ROCm add-on per model and uses the faster (M8) (2026-10-17)

Asked for by the owner (17 Oct: Kera should find the fastest way and use it); the design is the agent's.

- **What:** `kera serve --backend auto`, or with no `--backend` once the setting `gpu.auto_backend` is on (off until
  the gate below passes, P7). With the ROCm add-on installed, each back-end is measured once per model by `kera
  calibrate`, in its own process so the two never hold the model at once: it reads 2,048 tokens into an empty chat
  and the rest of a 16K one, and writes 32 words after a short prompt and after the long one, with the options of
  `kera serve`'s command line that change speed (llama.cpp reads them and fits the model to the chip's memory as the
  server does). The results are kept in `backends.json` in Kera's folder, by model file, Kera and llama.cpp
  version, and those options. Every reply's shape is noted in `replies/` (llama.cpp hook 6: how much the chat held,
  read and wrote, and how long it took; counts and times only). At each start the back-end with the shorter
  expected time for the model's last 300 replies is used; until the model has served 20, for chats (a 200-word
  message after 1,500 held, 300 written) with every tenth reply a document half the context long (at most 64K).
- **Why measured, not rules:** on Nemotron 3 Nano (16 Oct), Vulkan reads a short prompt faster and writes 26% faster
  (57.3 against 45.4 words a second after a 7.7K document), but its reading slows with what the chat holds about
  ten times as fast as ROCm's, so ROCm wins for prompts over about 20K words (a 95K document: 296 s against 138 s).
  Where that line falls depends on the model's design (how many attention layers) and on its size against each
  back-end's memory (ROCm sees 25.7 GB, Vulkan 33.6 GB), which a measurement captures and rules would guess.
- **The speed model** (`src/planner/backend_pick.h`): reading n prompt tokens after d held takes n·(a + b·(d + n/2)),
  writing likewise, each fitted from two measurements. Fitted to the 7.7K and 95K documents of 16 Oct, it gives the
  64K one within 2% on both back-ends (148 s and 87 s predicted, 148 s and 85 s measured).
- **Left alone:** a model whose experts are streamed (the SSD sets its speed) keeps llama.cpp's choice, as does any
  command line naming `--device`.
- **Saved chats are kept per back-end** (ROCm's and the CPU's in folders of their own; Vulkan's keep theirs): the
  back-ends round differently in the last bits, so a chat saved on one is read again on another, not continued
  there, and its answers stay the ones that back-end gives.
- **Checked as it runs:** at the end of a run, `kera serve` says how long the replies took against what the
  measurement predicted.
- **Gate, to turn the setting on:** on the Minisforum with Nemotron 3 Nano, both back-ends measured; the predicted
  times for a chat and for the 7.7K, 64K and 95K documents within 15% of the measured ones on each back-end; and the
  choice matching the faster back-end both for chats and for long documents. Each back-end already passes the
  exactness checks, and this only picks one.

## D-021: The graphics chip's built shaders are kept in Kera's folder, on by default (M8) (2026-10-15)

Decided by the agent (P7: a setting goes on once measured).

- **What:** the setting `gpu.shader_cache` (1 = on) makes `kera` set `GGML_VK_PIPELINE_CACHE` to
  `<Kera folder>/vulkan-pipelines`, and llama.cpp's Vulkan back-end (Kera's patch) keeps every shader it builds in
  `vulkan-pipelines-Vulkan0.bin` there (a few hundred KB).
- **Why:** without it, a Kera in a new folder (every update) built its shaders again on its first requests: 3.2 s
  to the first word instead of 0.18 s on the Minisforum, and slower writing until they were built (N-011).
- **Safe to keep on:** Vulkan checks the file's header and ignores one from another driver or chip; the file is
  written whole and then renamed into place; it changes no result, only how soon the first ones come. The
  environment variable `KERA_SHADER_CACHE=0` turns it off (back to the driver's own cache only).

## D-020: kera serve sizes llama.cpp's RAM prompt cache to the memory the model leaves (2026-10-15)

Decided by the agent, after a measurement slowed to a crawl.

- llama.cpp's server keeps recent chats' states in RAM, up to 8 GB by default (`--cache-ram`), on top of the model.
  Measuring cache-aware routing on Nemotron 3 Nano (streamed, 4 GB of RAM for experts, the 15 GB cloud machine),
  that cache filled to its limit after about 20 minutes of chats (143 MB per saved state, mostly Mamba
  checkpoints). The model's always-used weights then no longer stayed in the OS file cache, and writing slowed from
  about 3 tokens a second to 0.6.
- Now, unless the command line sets `--cache-ram` (or `-cram`): 1 GB while the experts are streamed (the memory they
  don't use is only the margin left for the chat and the working buffers), and for a model that fits, half the
  memory it leaves free (the profile's usable memory less the file and 2 GB), between 512 MB and 8 GB. Kera's saved
  chats on the SSD (hook 1) keep the older chats, so going back to one still doesn't mean reading it all again.
- The routing measurement itself is still valid: answers don't depend on speed, and the expert cache kept its 4 GB
  budget throughout.

## D-019: Reusing passages a chat has already read (CacheBlend, Kera's way); QuantSpec not built (M12) (2026-10-15)

**Removed on 9 Nov 2026 (D-034): Kera has no lossy modes. Kept as the record.**

Decided by the agent (spec 4b item 11, Phase 11, 11.5; lossy modes stay off by default).

- **What it does** (`kera serve --lossy-reuse SHARE [--lossy-reuse-min N]`, settings `lossy.reuse` and
  `lossy.reuse_min_tokens`): when a new prompt repeats a passage of at least N tokens (256 by default) that the chat
  has already read (a document pasted again, a file a coding agent reads again), Kera copies that passage's memory
  to its new place instead of reading it again, all but its first SHARE (0.15 = 15%), which is read again. The copies are made by llama.cpp hook 5
  (`docs/UPSTREAM_PATCHES.md`): new memory cells, moved to the new positions with the same RoPE re-rotation
  llama.cpp uses when it shifts a chat.
- **Why the passage's start, not CacheBlend's chosen tokens:** CacheBlend (arXiv 2405.16444) re-reads about 15% of
  a reused document's tokens, the ones whose memory changed most, anywhere in it. llama.cpp writes each chat's memory
  strictly in order (a word can only be added right after the chat's last position), so a word in the middle of a
  copied passage can't be read again after the words behind it are in place. Re-reading the start of each passage
  fits that rule: the start takes in the passage's new surroundings, and everything after the passage still reads all
  of it. Its cost is measured, not assumed (`docs/QUALITY.md`).
- **Where the passage can come from:** what the same chat (server slot) keeps in memory before the point where the
  new prompt differs. Passages from other saved chats, and documents reordered in the part that changed, are not
  reused (they would need a second sequence to copy through; a later step if the measurements say it pays).
- **Where it is off by itself:** models whose memory can't move positions (Mamba hybrids such as Nemotron, sliding
  windows, M-RoPE), a draft model or MTP head in use (their own memory would miss the copied words), images,
  shared-prefix requests and aLoRA. Saved chats made with it are kept apart from ones made without it (it is part of
  the saved-chat identity), so its approximations never leak into exact runs.
- **QuantSpec (11.5) is not built as its own guesser.** It guesses with the model itself reading a 4-bit copy of the
  chat's memory. Its paper saves most of its bytes on 16-bit weights; Kera's models already use 4-bit weights, so only
  the memory part remains. Per guessed word the guess reads all the weights plus a quarter of the chat's memory, and
  a check of k guesses reads the weights and the memory once: with k = 4 and 90% acceptance it beats no guessing only
  when the chat's memory is bigger than the weights read per word (e.g. 32,000 words of chat on Qwen3-32B: memory
  about 8 GB against 18 GB of weights, so it loses; it pays on small models in very long chats). It would also read
  every prompt twice, which lengthens the wait before the first word. A rough version is already available through
  llama.cpp's options (`-md <the same file> -ctkd q4_0 -ctvd q4_0`), and Kera's guesser switch measures it and drops
  it when it doesn't pay; a native version (one memory read at two precisions, sharing the weights) needs new memory
  formats and kernels, measured on the Minisforum first.
- **KIVI (4- and 2-bit chat memory) is not built:** the spec asks for it only if 8-bit isn't enough, and 8-bit cost
  0.05% perplexity (`docs/QUALITY.md`).

## D-018: The ROCm (HIP) back-end is an opt-in add-on; Vulkan stays the default (M8) (2026-10-15)

Decided by the agent (spec Phase 8 item 1, M1 risk test 5).

- **The add-on:** `build-windows-hip.yml` builds llama.cpp's HIP back-end (`ggml-hip.dll`) for gfx1151 the way
  llama.cpp's own release does (AMD's ROCm for Windows from AMD's Python package index, only the back-end, from
  the same llama.cpp copy as the download), with AMD's runtime files (`amdhip64_7.dll`, `rocm_kpack.dll`,
  `amd_comgr.dll`) and their licences, in a `hip` folder. It ships as a separate zip with full milestone
  releases (and with any release or CI commit marked `[hip]`).
- **Opt-in:** the `hip` folder isn't scanned by llama.cpp, so unzipping the add-on changes nothing. `--backend hip`
  (on `kera serve` and `kera exactness`) loads it, puts the folder on Windows' DLL search path (AMD's driver keeps
  an older `amdhip64_7.dll` in System32, which would otherwise win), and uses only the ROCm device; `--backend
  vulkan` and `--backend cpu` pick the others. Two back-ends for one graphics chip would otherwise split the model
  between them.
- **Why not the default:** llama.cpp's own tests on gfx1151 needed `HIP_LAUNCH_BLOCKING=1` to get right answers
  from batched work (RT5). HIP becomes the default for prompt reading only after `kera exactness --gpu --backend
  hip` passes on the Minisforum and it measures faster (spec 11.4 reports ROCm reading prompts 2.6-3.6x faster on
  Strix Halo, Vulkan writing 5-10% faster).
- **The memory split** (Phase 8 items 3 and 6): auto-setup records how much memory Windows lets the graphics chip
  use; when that is under 60% of the computer's memory, the explanation says how to raise it (AMD Software's
  Variable Graphics Memory, or the BIOS's UMA frame buffer size). Kera never changes it itself.
- **Measured on the Minisforum (15 Oct, Nemotron 3 Nano):** the add-on now carries everything it needs (N-007).
  ROCm passes `kera exactness --gpu` on all six tiny models, within 0.001 of the reference; Vulkan rounds more
  coarsely: its log-probabilities are off by up to 0.12, and on the GLM-5-style design that flipped a near-tie in
  the sparse attention's choice of words, which changed 3 words (N-010). ROCm reads long
  documents faster (a 95K-word document in 136 s, against 303 s on Vulkan and Ollama's 150 s) but writes about a
  quarter slower (45 against 61-65 words a second) and reads short prompts more slowly. ROCm sees 25.7 GB (the
  BIOS's dedicated share) against Vulkan's 33.6 GB, so a bigger model fits only on Vulkan, unless llama.cpp's
  unified-memory option (`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`) works here (not tried).
- **So, for now:** Vulkan stays the default, since most chats are short and it writes faster; `--backend hip` is the
  advice for long documents, and where answers must match the reference most closely. Next (M8): choose the
  back-end when a model loads from the model's design, its size against each back-end's memory, and the context
  it's loaded with, as a setting that stays off until M8's gate measures it.

## D-017: Lossy modes are off by default and each carries its measured cost (M12) (2026-10-15)

**Removed on 9 Nov 2026 (D-034): Kera has no lossy modes. Kept as the record.**

Decided by the agent (spec Phase 11; the owner's rule: lossy modes stay off by default).

- **8-bit chat memory** (`--lossy-kv8`): llama.cpp's `q8_0` KV cache, as a named mode with its quality measured.
- **Cache-aware routing** (`--lossy-route MARGIN`, streamed MoE models only): the stream session also reads the
  router's scores, and below a word's top two experts swaps an uncached choice for a cached expert scored within
  MARGIN (relative). The changed choice is written back before the experts' weights are gathered, so the layer's
  maths follows it. On a tiny model with half its experts cached, 5% cut the experts read from the SSD by 18%
  and 20% by 38% (tiny random models exaggerate the quality cost; the real cost is measured on Nemotron 3 Nano).
  Measured there (`docs/QUALITY.md`): with a quarter of the experts in memory and a 5% margin, half the SSD reads per
  word, perplexity +0.1%, 5 fewer right of 200 questions and 1 fewer valid JSON answer of 50.
- **The quality harness** (`tools/quality/quality.py`) runs section 6's quality tests on the model with and without
  a mode: perplexity, KL divergence and top-word agreement (`kera perplexity`, which now takes Kera's streaming
  options), 50 JSON prompts, a 2,000-word answer, and 200 built-in questions. Results go in `docs/QUALITY.md`.
- Every lossy mode prints a line saying it is on and lossy when `kera serve` starts.
- **Not built:** Fast and Balanced (need SB4, M7), precision by importance (graphics kernels), KIVI 4/2-bit KV, and
  Phase 12's experiments (NPU, Apple's matrix unit, two machines), which need the hardware. CacheBlend came later
  (D-019).

## D-016: Fine-tuning wraps MLX-LM and PEFT; adapters become GGUF files (M11) (2026-10-14)

Decided by the agent (spec Phase 10: "don't write a trainer, wrap the existing ones").

- **`kera finetune`** runs `tools/finetune/kera_finetune.py` with the computer's own Python (it ships in each
  download's `finetune` folder, with llama.cpp's converters). It trains a LoRA adapter on JSON-lines examples
  (chat messages, prompt and completion, or plain text; only the answers are learned), then turns it into a
  GGUF adapter with llama.cpp's `convert_lora_to_gguf.py`, for `kera serve -m model.gguf --lora adapter.gguf`.
- **Trainers:** MLX-LM's LoRA on a Mac with Apple silicon when `mlx-lm` is installed (its adapters are converted to
  PEFT's layout first; a unit test checks the change to each layer is the same), Hugging Face PEFT everywhere
  else: on NVIDIA (CUDA), on AMD with PyTorch for ROCm where it exists, or on the CPU for small models.
  Unsloth (NVIDIA) isn't wrapped: PEFT covers the same adapters, and the owner has no NVIDIA machine yet.
- **MoE models, `--esft`:** the training data goes through the model's routers first; the experts it uses most
  (until they cover half the routing, `--esft-share`) are trained, plus a LoRA on attention. transformers keeps a
  layer's experts in one tensor, which a LoRA can't cover expert by expert, so the chosen experts' slices are
  trained directly and ESFT writes a whole model GGUF. A per-expert overlay for the streamed store (only the
  trained experts differ from the base file) is the natural next step if ESFT is used on big models.
- **Adapters and saved chats:** each adapter's contents (its hash, not its path) and scale are part of the
  saved-chat identity, so a retrained adapter never reuses chats saved under the old one.
- **Checked in CI on the CPU:** a LoRA trained on a tiny dense model in about 15 seconds makes `kera serve
  --lora` give exactly the trained answer, the model without it doesn't, and saved chats are kept apart; ESFT on
  a tiny MoE model does the same with a whole model. Gate 10 (a) itself needs the owner's Mac and the
  owner's own small dataset.
- **Not built yet:** training draft heads for guessing (EAGLE- or MTP-style) and calibrating the SB4 base (that
  waits on RT1).

## D-015: Guessing for streamed models (M6) (2026-10-14)

Decided by the agent (spec Phase 6, 10.8).

- **Same switch, measured on the streamed model:** the guesser switch (D-014) measures real words per second,
  so the SSD reads a guess check causes are part of what it measures. A check of k guesses reads every
  expert any of its k + 1 words uses (RT2: 4 words use 2.9x one word's experts), so when the experts are
  streamed, guessers start low (model guessers at 1 word, copying at 4) and the switch measures upward.
- **Union growth is logged** (Phase 6 item 4): for each check size, the distinct experts per layer against
  one word's, in the per-minute and end-of-run lines of `kera serve`.
- **new_bytes(k)** (Phase 6 item 2): `stream::session::predicted_new_bytes(k)`, the bytes a check of k
  guesses is expected to read: the experts it is expected to use that aren't in memory yet, from the measured
  union growth (before any check, RT2's curve). Shown in the per-minute line; the switch's measurements
  have the final say, as Appendix B's KTuner does over its formula.
- **Exact:** greedy output is the same with guessing on and off with the experts streamed, with the model's
  MTP head and with a draft model (`tests/integration/test_guess_controller.py`, Gate 6 (a)).
- **Not built, and why:**
  - *Feeding the guesser's expert choices to the prefetcher* (Phase 6 item 3): no available guesser predicts
    the big model's experts. An MTP head routes only its own block, and a separate draft model has other
    experts. The SB4 base (M7) would; RT1 decides whether it exists.
  - *Guessing while waiting for the SSD* (Phase 6 item 6, MoE-SpeQ): it needs a guesser that predicts experts
    (as above), and llama.cpp's server runs guessing and checking one after the other. A draft model on its
    own thread, guessing while a check waits for experts, can be tried once the Minisforum shows how long
    those waits are; its gate is 10% faster with exact output.
- **Dense models bigger than memory** stream through llama.cpp's mapped file. A check of k guesses reads
  their weights once for k + 1 words, so a draft model pays most there; the switch finds the length. Gate 6 (c)
  measures it on the Mac.

## D-014: The guesser switch, and which guessers each model gets (2026-10-14)

Decided by the agent (spec 11.1 item 1, Appendix B, M4).

- **One switch per chat** (`src/spec/kr_guess_switch.h`): llama.cpp asks its guessers in a fixed order, n-gram
  guessers first (they cost nothing to run), then draft models and heads. Hook 2 now asks Kera about each
  guesser at every step: a model guesser before it runs (how far may it guess; 0 = skip it), an n-gram
  guesser after it guessed (how many of its words to keep; 0 = drop them). Each guesser has its own
  Appendix B KTuner, which measures the words per second of its choices, "don't guess" included, in the
  steps where that guesser decides. So each learns whether it beats what happens without it.
- **Lengths:** model guessers choose 0 to 8 words (Appendix B's k_max), starting at 2; n-gram guessers
  choose from 0, 4, 8, 16, 32 or 64 words, starting at 8. Each guesser's first test runs at once rather
  than after 512 words. Low starts cost little when guessing doesn't pay, and the first test still finds
  a good length when it does.
- **When nothing pays:** just trying to guess costs a little each step (llama.cpp's server hands every
  guess to its task queue: 33% of a step on a tiny model, a few tenths of a millisecond). While every
  guesser's choice is 0 and none is testing, guessing is only tried every 16th step.
- **Defaults per model:** with no guessing options on the command line, `kera serve` adds copying from the
  chat (llama.cpp's `ngram-mod`, its own settings) and the model's own MTP head when the file has one
  (`spec.auto`, `--no-auto-guess`). A model guesser still costs time while loaded (an MTP head reads every
  word), so when its best choice settles at 0 after 1,000 steps, Kera remembers it for that model file
  (`guessers.json` in the Kera folder), and the next start leaves it out for 14 days.
- **Exactness** is checked with 32-bit memory and no flash attention, where checking several words rounds
  exactly like writing them one by one (N-003) on CI's tiny models; with llama.cpp's defaults, and with some
  quantized models' kernels (K-quants on CPUs whose matrix code changes with the batch size), a choice between two
  nearly equal words can flip, as with any change of batch size or other chats at once (N-018). `--no-auto-guess`
  gives stock llama.cpp's replies exactly.
- **Simulated** (`tests/cpp/test_guess_switch.cpp`): with a good n-gram guesser and an MTP-like head, the switch
  settles on 16 and 2 words (3.2x the speed without guessing); a slow, rarely right guesser goes to 0 and
  costs 0.5%, trials included; with nothing paying, it tries on 6% of steps and costs 1.5% where trying
  costs 3% of a step.
- **Not built:** SuffixDecoding and lookahead decoding (llama.cpp's n-gram guessers cover copying from the
  chat; SuffixDecoding is the candidate if agent sessions show they need more), and the SB4 base (M7).

## D-013: Per-model thinking and sampling defaults, from the model cards (2026-10-14)

Decided by the agent (spec 4c item 2, ground rule 5).

- `kera serve` uses the sampling a model's own card recommends for each kind of request, wherever the request
  sets none (hook 4). A request's own values always win, and `--no-model-defaults` (or the setting
  `chat.model_defaults` = 0) turns it off. Since 9 Nov (D-033): the user's own server settings win too, the card
  applies only to the card's own models (by design and name), and a file whose own `general.sampling` defaults
  aren't one of the card's keeps them.
- Only families whose cards were read on 14 Oct 2026 are listed; others keep the server's defaults, which
  come from the GGUF file's own `general.sampling` values when the file has them:
  - **Nemotron 3 Nano 30B-A3B** (corrected on 9 Nov: Super, Ultra, the 3.5 releases and Nano Omni have other cards and
    carry their own values; [NVIDIA's Nano card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)):
    reasoning on (the default) temperature 1.0 and top_p 1.0; reasoning off greedy; tool calling
    temperature 0.6 and top_p 0.95. Its tool calls use the `qwen3_coder` format, which llama.cpp parses.
  - **Qwen3** ([Qwen3-30B-A3B card](https://huggingface.co/Qwen/Qwen3-30B-A3B)): thinking 0.6 / 0.95 /
    top_k 20 / min_p 0; non-thinking 0.7 / 0.8 / 20 / 0. The 2507 Thinking release always gets the thinking
    values, and the 2507 Instruct release (it can't think) always the non-thinking ones. Since 9 Nov, from Qwen's own
    GGUF cards: presence penalty 1.5 for quantized files (not 2507, not full precision).
  - **Qwen3-Coder** ([card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)): 0.7 / 0.8 / top_k 20.
    Its repetition penalty (1.05) is left out: llama.cpp's penalty works over a window, not the whole text.
- Thinking stays on by default where the card says so (llama.cpp turns it on for templates that support
  it). The thinking text the chat app drops from its history doesn't force a re-read: RT4 showed the server's
  checkpoints handle it.
- Tool-call parsing for the target families (Nemotron 3, GLM, Kimi K2, DeepSeek, Qwen3, gpt-oss) is
  llama.cpp's own, tested with each model's real template by upstream's `test-chat`, which Kera's CI now
  builds and runs on the pinned copy.

## D-012: Streaming experts: read in place into a mirror of the model file, no repacked copy (2026-10-14)

Decided by the agent, after building M5's store and measuring it in the cloud.

- **How experts get into memory (v2, Kera's cache):** hook 3 (`docs/UPSTREAM_PATCHES.md`) puts each routed
  expert matrix of a MoE model in Kera's own buffer type: a *mirror* of the model's files, a reserved range
  of address space where the byte at file offset x sits at base + x. Nothing is in RAM until it is read.
  Before each MoE layer runs, the hook's callback sees the layer's chosen experts (llama.cpp's
  `ffn_moe_topk` tensor), and Kera reads the missing ones with direct reads (which skip the OS file cache)
  straight into their place in the mirror, giving the least recently used ones back to the OS to stay
  within the RAM budget. A layer's experts are locked until the next layer asks. The bytes, kernels and
  order of operations are llama.cpp's own, so the output is bit-identical: checked on all four tiny designs,
  word by word, with a cache too small to hold one layer.
- **v1 (the OS's file cache):** the experts stay mapped from the file, as lazy weights that are never read at
  load, and Kera asks the OS to start reading each layer's experts as the layer begins (and the next layer's
  guesses), from a helper thread. The auto-tuner can pick either per machine.
- **Why not the spec's repacker (Phase 2 step 1):** ggml's MoE kernels read all experts of one matrix from one
  3-D tensor with a fixed stride, so one block per expert would need new kernels in every back-end, or a copy
  after every read. Reading each expert's slice of each matrix in place costs 2-3 reads per expert instead of
  one; at 1-13 MB per read an NVMe drive runs at full speed anyway. It needs no second copy of a 60-130 GB
  model and works on any GGUF as downloaded. The index the spec asks for comes from the GGUF header at load.
- **The cost of reading in place:** an evicted expert's pages go back to the OS and the next expert's land in
  fresh ones, one page fault per 4 KB read (13,000-31,000 per word in the cloud test). They cost nothing while
  the maths waits for the SSD, but compete with it when reads overlap the maths on a CPU with few cores. If
  that shows on the Minisforum, the fix is fixed slots plus a repacked file (one aligned block per expert,
  read straight into reused memory). It gets measured there before it is built.
- **The graphics chip:** upstream's GPU expert cache (`--moe-cache-mib`) works on top unchanged; it uploads
  from the mirror. On the Minisforum the chip shares the RAM, so an upload is a copy within memory; reading
  straight into the GPU cache's memory is a later step, measured there.
- **Prompts:** a prompt uses nearly every expert (RT2: 256 words touch 88%), so a streamed model reads its
  prompt layer by layer in large batches, never small chunks (D-008).
- **Phase 3 (next-layer guesses):** built. The next MoE layer's own router, applied to this layer's router
  input rescaled by the two layers' norm weights, guesses 68-72% of its experts in use (RT2 predicted 73%).
  **Off by default** until Gate 3 passes on the Minisforum: on the 4-core cloud machine it halved the wait
  for the SSD but spent as much CPU (the guessing, and the extra reads' page faults) as it saved.
- **Phase 4 (pins):** built, off by default: they failed Gate 4 on RT2's traces (N-002).
- **Phase 5 (second SSD):** the cache reads from byte-identical copies of the model's files on other drives,
  sending each read to the least busy copy and falling back to the original on any error. Gate 5 needs a
  second drive; buying one is the owner's decision.
- **Measured in the cloud** (Nemotron 3 Nano Q4_0, 18 GB, on a VM with 15 GB of RAM, 4 cores and a 1.9 GB/s
  disk; cold, three different prompts, 64 words each): stock llama.cpp with mmap wrote 4.64, 3.14 and 3.90
  words a second; Kera's cache with 11 GB of RAM 5.64, 4.85 and 4.75 (22%, 54% and 22% faster), and read the
  prompts faster too (`docs/BENCHMARKS.md`). These are not the gates: those are on the Minisforum.

## D-011: Keeping GitHub's free minutes for what matters (2026-10-14)

Decided by the agent, following P8 ("keeping the minutes down") and the owner's "no plan changes".

- **Changed on 18 Nov 2026:** the GitHub fork is public, so the 2,000-minute cap below no longer applies to it.
  At the time, the repo was private, so GitHub's own machines drew on the free monthly allowance: 2,000 minutes on GitHub Free. Windows minutes count about double and Mac minutes about ten times *(verify on GitHub's billing page; GitHub has changed these before)*. The spec's plan of running Windows on every change and the Mac nightly assumed free runners on the owner's machines. The owner chose not to have those (D-002).
- **Every code push:** the Linux build, unit tests, reference-code tests, an app smoke test and the web page's type check (`ci.yml`). Docs-only pushes run nothing.
- **Windows build and download:** on every release tag, and on pushes whose commit message contains `[win]` (`build-windows.yml`).
- **Mac build and download:** on full milestone releases, on demand (`mac.yml`), and on pushes marked `[mac]`. It never runs on a schedule on GitHub's machines.
- **CUDA build-only check:** weekly instead of nightly (`nightly.yml`).
- **Benchmarks on the owner's machines:** only on runners the owner installs. The jobs stay off until a repository variable says a runner exists.
- Scheduled workflows (nightly, monthly upstream) only run from `main`, so they start once M0's gate passes and the work is merged.
- Speed-ups: ccache on every platform, a newer push cancels the older run, and build files are kept 3 days.

## D-010: Platform facts checked on 14 Oct 2026 (ground rule 5)

Checked by the agent, against llama.cpp release b11619's source and `--help`.

- **llama.cpp's graphics-memory expert cache is merged** (spec 12.1 item 9): `--moe-cache-mib N`, code in `src/llama-moe-cache.cpp`. It needs a GPU or integrated-GPU back-end and only serves batches of up to 32 tokens.
- **Checkpoint flags for hybrid models** (P5 risk test 4): `--checkpoint-min-step` (default 8,192 tokens, 0 = no minimum), `--ctx-checkpoints` (default 32) and `--cache-ram` (default 8,192 MiB) all exist under those names.
- **Expert placement flags:** `--override-tensor` (`-ot`) and `--n-cpu-moe` (`-ncmoe`) exist.

## D-009: Licence for Kera's own code (2026-10-14)

Decided by the agent as a safe default; licence changes are the owner's call.

- Kera's own code is "all rights reserved" under Kera Labs, the organisation name in the logo kit. That grants nobody anything, so the owner can choose any licence later, including selling Kera. (Replaced on 4 Nov 2026 by D-026: Apache 2.0, the owner's choice.)
- Third-party code keeps its own licence: llama.cpp is MIT, and Colibri code will be Apache-2.0 with its notices. Both are listed in `NOTICE`.

## D-008: Foundation: build on llama.cpp, hook into its expert cache (2026-10-14)

Decided by the agent after reading both codebases (Phase 0 decision, spec section 3).

- **Fork point:** llama.cpp release `b11619` (commit `9cf967a`, 14 Oct 2026), copied into `third_party/llama.cpp` with `git subtree --squash`.
- **Why llama.cpp and not Colibri's engine:** llama.cpp already has the Metal, Vulkan, HIP and CUDA kernels, the GGUF format, all five target model designs, guess-ahead (including MTP), LoRA loading and a server. Colibri is a separate engine of about 157,000 lines of C, with one source file per model family and its own kernels and weight formats, so every new model would need its own work there.
- **Where the streaming store hooks in:** release b11619 has a new MoE expert cache (`src/llama-moe-cache.cpp`, `--moe-cache-mib`). It does the following:
    - MoE layers whose experts stay in host memory get a cache of expert "slots" on the graphics chip.
    - During writing, a callback between graph splits reads which experts the router chose, evicts the least recently used ones, and uploads the missing experts.
    - The expert maths then reads from the slots.
    - Today it uploads by reading the memory-mapped model file, which is the slow "mmap and hope" path the spec warns about (section 8).

  Kera changes only the source of those bytes. A small hook, listed in `docs/UPSTREAM_PATCHES.md` once written, lets Kera's store hand over each expert. The store reads with parallel direct reads from the repacked file, keeps its own RAM cache and learned pins, and starts reads early from next-layer guesses and draft routes. The model loader is unchanged: expert tensors stay host-side and memory-mapped, but the fast path never touches the map.
- **Routing traces need no patch:** llama.cpp's evaluation callback (`cb_eval`) sees each layer's `ffn_moe_topk` tensor, the chosen experts (P5 risk test 2).
- **What Kera ports from Colibri:** its logic, not its engine, rewritten into Kera's `src/` with its Apache-2.0 notices. That covers:
    - the expert store and its lease rules (`c/expert_store.h`);
    - learned pins from routing history (`c/route_trace.h`, `.coli_usage`);
    - next-layer lookahead;
    - multi-disk mirrors (byte-identical copies, checked at start, falling back to the main drive on any error, with reads split by measured speed;
      as of 9 Nov only the reading side is built and only tests set copies: no option uses them, and the check at start isn't built,
      so they stay off until it is, N-018);
    - KV prefix reuse and keeping KV across restarts (`c/kv_prefix.h`);
    - the setup, plan and tune logic (`c/setup_*.py`, `c/resource_plan.py`, `c/autotune.py`), from Python into C++.
- **File layout for streamed experts:** the repacked GGUF from Phase 2. Each expert's matrices sit in one 16 KiB-aligned block, with an index, so llama.cpp's own quantization types and kernels keep working. The 16 KiB alignment also meets Windows' unbuffered-read rules. Colibri's own files are used only for the Colibri baseline in benchmarks.
- **Risks:**
    1. **No CPU-only mode.** Upstream's cache refuses to run without a GPU or integrated-GPU back-end. CPU-only streaming (the cloud CI, and the CPU path on the Minisforum) needs a second small hook that lets the CPU act as the cache device. Until then, CPU-only runs use cache mode v1: memory-mapped experts, with Kera warming the OS file cache by reading ahead.
    2. **Prompt reading skips the cache.** Batches over 32 tokens don't use it. For streamed models, prompt reading runs as one large batch, so the graph goes layer by layer across the whole prompt (4b item 3), with Kera streaming each layer's experts just ahead. It must never be small chunks (section 8).
    3. **A pause per MoE layer.** The callback makes the graphics chip wait at every MoE layer. The overhead is measured in M5.
    4. **Upstream churn.** The cache is one week old upstream and may change. The hooks stay tiny, and upstream is pulled monthly.
- **Fallback, per spec section 3:** Colibri's engine plus ggml kernels. It isn't needed now, because the upstream cache already gives the per-expert slot mechanism the spec asks for.

## D-007: How milestones whose gates need hardware are handled (2026-10-14)

Decided by the agent, following the owner's "don't stop" rule (D-003).

- Most gates need a measurement on a real machine: the Minisforum (reached over Tailscale, D-005), the Mac, or NVIDIA. When a milestone's code is built and every check this cloud machine can run passes, but its gate needs hardware that isn't reachable yet, it is marked **"built; gate waiting for hardware"** (or "waiting for the owner" when the owner has to set up access). It is never marked done.
- Later milestones that only need an earlier milestone's code go ahead. Ones that need its measurements wait.
- A milestone is merged into `main` and released only when its gate passes. Preview builds, such as the first Windows download that Gate 0 asks the owner to open, are published as GitHub pre-releases.

## D-006: No pull requests (2026-10-14)

Decided by the owner.

- The owner doesn't want pull requests. Work is committed and pushed straight to the working branch (`claude/clever-planck-ga06qs`), with the P7 checklist done and the gate result in each commit message.
- This replaces the spec's pull-request flow (P7, P8). The spec's text was updated directly to say so.
- Without pull requests, the spec's GitHub labels, milestones and per-step issues aren't used. `docs/STATUS.md` is the tracker.

## D-005: Reaching the Minisforum from cloud sessions (2026-10-14)

Decided by the agent, to carry out D-002.

- A cloud session joins the owner's tailnet with Tailscale in userspace mode,
  with its state kept in memory only. Each session shows up as a temporary
  device that disappears when the session ends.
- Measured from a cloud session on 2026-10-14 with Tailscale 1.104.1: the login
  server and the relay (DERP) servers can be reached through the session's
  proxy, but direct UDP is blocked, so all traffic goes through a relay. The
  nearest relay was Chicago at 33 ms.
- A relay is fine for sending commands and a finished package. It is too slow
  for moving models, so models are downloaded on the Minisforum itself.
- The Tailscale auth key comes from the environment variable `TS_AUTHKEY`, set
  in the cloud environment's settings. Never in the repo or the chat.
- Tailscale's own SSH server doesn't run on Windows. Running commands on the
  Minisforum therefore needs Windows' OpenSSH Server and a login key for the
  agent, read from `KERA_SSH_KEY_B64` and `KERA_SSH_TARGET`. Setting that up is
  an owner action (see `docs/STATUS.md`).
- First used on 2026-10-15, without those settings: the session joined with a
  login link the owner approved, and made its own key pair, whose public half
  the owner added with their own key. `scripts/remote.sh` does all of it. The
  Minisforum's SSH server starts PowerShell, so every command goes as
  `powershell -EncodedCommand`, which works under cmd.exe too.
- The Minisforum also serves other apps (websites, and an Ollama they use all
  the time). The agent works only in the Kera folder and leaves those alone; to
  compare with Ollama it starts a private copy on its own port
  (`scripts/ollama-private.ps1`), because loading a model in the machine's own
  Ollama pushes out the model the apps use.

## D-004: Nothing is pushed while the repo is public (2026-10-14; cancelled 18 Nov 2026)

**Cancelled by the owner on 18 Nov 2026.** Kera now lives in a public GitHub fork, so pushing to it is allowed.
The text below is kept as history only and no longer applies.

Decided by the agent, following the owner's "never make it public".

- When the first session started, `Vortexal2/Kera` was public (created
  2026-10-14 17:40 UTC). Pushing would have published the work.
- Every session checks the repo's visibility before it pushes. While it is
  public, commits stay local and the session says so in `docs/STATUS.md`.
- The owner made the repo private on 2026-10-14 at about 18:11 UTC. The first
  push followed.

## D-003: Standing permissions and limits (2026-10-14)

Decided by the owner. The full list is in `AGENTS.md`, which every session
reads first.

- Allowed: download models (keep 100 GB free, stay inside the Kera folder,
  never move or delete the owner's own models); install the GitHub runner
  labelled `amd` while the repo was private (it is public now, so no runners: P8); benchmark any time, closing other
  AI apps and reopening them afterwards; merge your own pull requests once the
  checks are green and the gate passed.
- Lossy modes stay off by default.
- Never: admin rights after the first setup, spending money, payment methods or
  plan changes, adding people, BIOS, driver or Windows changes, deleting outside
  the Kera folder, or hardware that isn't here (the Mac, NVIDIA, DGX Spark).
  These go under "Waiting for the owner" instead.
- If a gate fails twice, or a milestone takes more than twice its size: log it
  in `docs/NEGATIVE_RESULTS.md`, write the options in `docs/STATUS.md`, and move
  on to milestones that don't depend on it. Do M7 only if M1 says go.

## D-002: Code in GitHub, finished packages on the Minisforum (2026-10-14)

Decided by the owner.

- All code is written and kept in the GitHub repo. Builds and tests run on
  GitHub's hosted runners.
- The Minisforum only runs finished packages, reached over Tailscale. Nothing is
  built or developed on it, so the earlier plan to install build tools there
  with winget is dropped.
- The agent's reading, for the owner to confirm or correct: where a gate needs
  a measurement on the Minisforum (the F0 baselines, the streaming work), the
  session runs that milestone's finished package there, rather than waiting
  until every milestone is done.

## D-001: Windows first, Mac second (2026-10-14)

Decided by the owner. This replaces the spec's Mac-first order.

- The Minisforum (Windows, AMD) comes first. The first download is the Windows
  app, and the F0 baselines and the streaming work are built and measured on it
  first.
- The Mac comes after. Sessions do the parts GitHub's macOS runners can build
  and test, and list real-Mac measurements in `docs/STATUS.md` under "Waiting
  for the owner".
- Still to do: update `KERA_BUILD_SPEC.md` to this order through a pull
  request, once the spec is in the repo.
