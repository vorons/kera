# Negative results

Every failed gate, dropped feature and idea that measured slower goes here, so
nobody repeats it (ground rule 1). Newest first.

Each entry: the date, what was tried, on which machine and model, the numbers,
why it failed (if known), and what was decided.

## N-020: Clover-35B-A3B-Preview is a newer model than llama.cpp understands (2026-11-18)

**Tried.** `kera serve -m Clover-35B-A3B-Preview-Q4_0.gguf --stream kera --stream-budget-gb 3000` on this Linux
computer, with llama.cpp b12500 (the engine in 0.1.5.3).

**Result.** Loading stopped twice. First the loader asked for `clover_moe.expert_shared_count`, which this file
does not have. A local fallback that inferred one shared expert from the shared-expert tensors was tried and
removed again: with it, loading reached the tensors and stopped at `wrong number of tensors; expected 853, got 733`.

**Why.** The file is a newer or custom Clover variant. It has 40 layers, 256 experts with 8 used per word and one
shared-expert set per layer; it carries `clover_moe.moe_block_coreset_enabled`, `..._top_p` and `block_size`,
which b12500 does not read. Each of its 30 linear-attention layers has four tensors that neither b12500 nor the
current llama.cpp master names: `ssm_bd_history.weight`, `ssm_bd_history.bias`, `ssm_bd_logit_scale.weight` and
`ssm_conv1d.weight_right` (120 tensors in all). b12500's Clover graph has no operation for them. Telling the
loader to ignore the extra tensors would make the model run, but on a different computation than the file
describes, which rule one forbids.

**Decided.** No loader patch: this file needs a converter that exports it as the Clover variant llama.cpp
supports, or a llama.cpp that implements the four extra tensors. The local fallback patch was reverted.

## N-019: On the CPU, an 8-bit file's own numbers depend on how many words are computed at once (2026-11-10)

**What was asked.** After 0.1.3's checks, three replies still differed between `kera serve` as it ships (guessing
ahead on) and `--no-auto-guess`, on requests with penalties, DRY or a logit bias. Is that Kera's guessing mishandling
those settings, or rounding?

**Where.** The cloud machine (4 cores of a Xeon with AVX-512 and AMX, the build made with `-march=native`),
Qwen3-0.6B Q8_0, greedy, the same six requests. Two investigations by agents, one reading the code and one measuring.

**What was found.**

- **The sampler is right.** A guess check samples and accepts one word at a time in order and stops at the first
  word that isn't the guess; penalties, DRY, a logit bias, grammars, the thinking budget and a seeded draw each end in
  the state writing word by word reaches (read in the code; the penalty arithmetic at each split matched by hand).
- **The model's own numbers differ.** At the words where the replies split, the model's log-probabilities with
  guessing on leaned 0.16 and 0.22 nats away from those with it off. Those splits are **not near ties**: the two
  candidates were 4.2 and 2.0 percentage points apart after the request's own settings.
- **It is the batch, not the guessing.** The server with guessing off flips the same two words by itself when the
  last few words before them are read as one batch instead of one at a time (most batch sizes from 2 to 64 flip
  them). The deciding log-ratio moves by up to 0.32 nats with the batch size.
- **It is the 8-bit weights on the CPU.** Turning flash attention off or using 32-bit memory for the chat leaves the
  spread at 0.09 to 0.33 nats. A 32-bit copy of the same weights shrinks it 20 to 50 times (0.016 and 0.0045 nats),
  and both words then stay the same at every batch size. So the cause is ggml's CPU arithmetic for 8-bit weights,
  which rounds a row computed in a batch differently from a row computed alone.

**What it means.** With a quantized file on a CPU, anything that changes how many words are computed at once moves
the model's own numbers by more than the 0.1-point "near tie" the exactness rule assumes: guessing ahead, llama.cpp's
own speculative decoding, or the size of the batches a prompt is read in. The text is still the model's own (each
word is the most likely one by the numbers computed for it), but it is not always the text the same file gives word
by word. `tests/integration/test_stock_defaults.py` (SmolVLM-256M Q8_0 on GitHub's build, without `-march=native`)
still finds Kera identical to stock llama.cpp, guessed words included, so this depends on the processor's kernels.
Not measured: a graphics chip, 4-bit files, and whether it is the AMX kernels.

**Decided.** Nothing is changed in 0.1.4: the sampler needs no fix, and the choice is the owner's (rule one):
guessing ahead stays on by default as it is (the same kind of difference stock llama.cpp has between batch sizes),
or it goes off by default on the CPU for quantized files, where Kera would then give stock's text word for word at
the cost of the speed-up. It is item 12 under "Waiting for the owner" in `docs/STATUS.md`.

## N-018: Places where Kera changed what a model does, found by the rule-one audit (2026-11-09)

- **What:** after the owner's rule number one (D-033), an audit of every place Kera touches generation (119 places in
  the chat server, the hooks, the media commands and the engines; 15 checked in depth) and a deep check with
  experiments against stock llama.cpp found eleven places where Kera changed a model's output by default: the card's
  sampling over the user's own `--temp`/`--top-p`/`--top-k`/`--min-p` and over the file's own defaults, and on models
  that only share a design with the card's; saved chats restorable after an update, under another context size on
  LongRoPE models, with `LLAMA_ARG_*` settings or after other adapter scales; words added to `kera 3d -p`; a 30 s
  song length; a sound length and its rounding; Z-Image-Turbo one step short of its card; the MOSS text cache keyed
  by name and size.
- **Why it happened:** each was added for a reason (a nicer default, a quicker start, "what the 3D step reads best"),
  without asking whether the engine's own default or the user's setting would differ; and the saved-chat identity was
  built from the command line's text, not from the settings llama.cpp actually used.
- **What was decided:** all fixed (D-033 lists each); tests added for the card's rules; the saved-chat cases
  rechecked with the experiments that found them (each now reads the chat again and gives stock's answer exactly).
- **The second round** (a deep check with experiments against stock llama.cpp and real model files) found more, all
  fixed (D-033): Ollama models ran without their Modelfile, and one with no chat template in its file was prompted in
  ChatML (Gemma 3 270M, on its real weights: 23 of 24 questions right in its own format, 2 of 24 in ChatML, never
  stopping); Nemotron 3 Nano's card reached Nano Omni (a real, popular file) and the rest of the family by name;
  Qwen3-2507 Instruct got thinking values when asked to think; Qwen's GGUF cards' presence penalty was left out; a
  folder's lone image reader was attached to any model (a text model then described pictures it can't see); and
  guessed words were reported with probability 1 (an upstream gap, fixed upstream after Kera's version).
- **The third round** (the deep check's other 31 suspected problems, each re-checked by an agent that tried to
  disprove it: 30 real, 1 not, none high) found, all fixed or corrected (D-033): picture models drew at the engine's 512x512
  instead of their cards' 1024x1024; SDXL-Lightning got its steps but not Euler with trailing steps (its 2-step model
  ran as one jump); any FLUX file named "dev" got FLUX.1-dev's settings (CFG off for de-distilled models, Kontext,
  Fill, Krea and FLUX.2 without their guidance); a turbo's 1 step applied to picture-to-picture, where the engine then
  ignores the picture; `kera video` used none of Wan's or LTX's settings (CFG 7, no negative prompt); `kera 3d` cut a
  drawn picture out by brightness when BiRefNet was missing, not with the model's own remover; `kera sfx` took the
  8-bit MOSS file over the full-precision one beside it, and `kera speak` the 8-bit audio decoder; `kera serve
  --music` gave an unnamed request the model used last; songs came out as 128 kbps MP3; the sound text cache didn't
  change with the engine or the graphics chip; a fine-tune's prompt/completion rows were learned without the chat
  format the chats use (on Windows and Linux; MLX on a Mac adds it), and an ESFT model was named "Hf", so Kera didn't
  know it; `kera wait` advised turning thinking off; Kera's image reader overrode one named in `LLAMA_ARG_MMPROJ` or
  config.ini and was attached where those turned it off (a new llama.cpp hook 8 fills it in after they are read); a
  failed read of a streamed expert crashed the server, a streaming setup that failed left the experts as zeros, and a
  short read of a shared page went unnoticed; the engines' tiled multiply was on for ARM, where it adds in 8
  half-precision lanes; the GPU exactness check let swaps of scores 0.02 apart pass. The published speeds were measured
  asking each prompt three times on one server, so from the second time Kera's guesser could copy its own earlier
  answer: the tool now starts each run on a fresh server, and the published numbers are marked until measured again.
  Wording corrected: "exactly the model's own answers" (guessing can round a near tie the other way, on the CPU too),
  "guessing ahead (exact)", the accuracy page's "one answer changed" (six did, one of them to right), and "about twice
  as fast" for songs (1.6 times, checked against the old build, not reference code). New tests: Kera against stock
  llama.cpp at the defaults on a real quantized model, guessed words included (identical), and the picture and video
  settings (16 cases).
- **A testing trap found on the way:** llama.cpp's build uses the compiler cache (ccache) when present, and
  `-march=native` reads the same to it on any processor. This cloud machine's Kera build therefore had CPU kernels
  compiled weeks earlier on another processor (without AMX, AVX512-BF16, VBMI or AVX-VNNI), and Kera differed from a
  fresh stock build by rounding from the first word, which looked like a Kera change until the libraries were
  swapped one at a time. Comparisons with stock need the CPU kernels built without the cache
  (`CCACHE_DISABLE=1`). The downloads aren't affected: they pick CPU variants at run time.

## N-017: The three lossy modes were removed: Kera trades no quality for speed (2026-11-09)

- **What:** M12's opt-in lossy modes (spec Phase 11), built and measured on 15 Oct 2026, were dropped by the owner's
  decision (D-034): Kera keeps the model's own output, always. They were never on by default.
- **What they measured** (the cloud CPU; the full tables were in `docs/QUALITY.md` until this change):
  - `--lossy-kv8` (the chat's memory in 8 bits, Qwen3-1.7B): perplexity +0.05%, the same top word 97.4% of the
    time, 113 against 110 of 200 questions right; half the memory per token.
  - `--lossy-route 0.05` (Nemotron 3 Nano streamed with a quarter of its experts in memory): perplexity +0.10%, 163
    against 168 of 200 questions right and 47 against 48 JSON answers; 48% fewer SSD reads per word.
  - `--lossy-reuse 0.15` (Qwen3-1.7B, a 340-token document pasted again): 29 against 30 second answers right, 28 of 30
    identical; 77% fewer words read for the second message.
- **Why removed:** each one changed some answers, however rarely, and the owner's rule is that Kera only makes models
  faster. The exact speed-ups (saved chats, guessing ahead, streaming experts, sharing a chat's start, the back-end
  choice) stay.

## N-016: A third bug sweep before 0.1.2 found 23 defects, 22 fixed; the "only one user" report was Kera started twice (2026-11-08)

- **How:** two reviews of every file (the app, its server and router; the media commands, setup, planner, stores
  and build), each finding traced through the code, then checked and fixed by hand. N-014's fixes weren't re-reported.
- **The report** ("only one user can get in, then a bunch of lines"): nothing in Kera serves one user at a time (the
  page and router use thread pools, the chat engine 4 slots, and no lock is held during an answer). Starting Kera a
  second time did the damage: the second copy moved its page to the next port but first started a second copy of the
  last model (failing on a full machine, the page then showing "stopped" and the last 8 lines of the log), and
  emptied the first copy's log. Kera now opens the running one instead. Two more paths could look like it, both fixed:
  with `--backend auto` on one-at-a-time memory, one long document made every other request wait behind a back-end
  swap for up to 10 minutes, then fail; and on Windows a second server could listen on a port already in use, so
  requests went to either at random. Kera's page answers this computer only, by design: other devices can't reach it.
- **Fixed (22, some lines below hold several):**
  - a second `kera app` opens the running one and starts no second model; a page that had to move port starts none;
  - the router lets other requests through during a back-end swap (requests that aren't prompts at once, prompts
    after 30 s);
  - the router and llama.cpp's server (on Windows) refuse a port already in use (`docs/UPSTREAM_PATCHES.md`);
  - ordinary NVIDIA cards were taken for shared-memory chips like the Spark's (decimal GB compared with GiB, and a
    loose 0.75): an RTX 4090 in a 32 GB PC was told it shares memory and read models in whole. The size rule now
    applies on ARM Linux only;
  - `kera music --seed` gave a different song every run (D-030);
  - the image engine's timed choices could change the picture for the same seed (D-030);
  - the app's model answered every website open in the browser (llama.cpp's CORS default): now pages on this
    computer only;
  - Ctrl+C in the app's terminal reached the model twice (llama.cpp's server then quits without saving chats): the
    model runs in a process group of its own;
  - a ROCm back-end measured faster once but no longer loading stopped `kera serve`: it falls back;
  - `kera app --port 0` gave a page answering only "Forbidden";
  - `kera finetune` on Windows went through cmd.exe, which expanded `%NAME%` and could run what followed `&` after a
    quote: Python is now started directly;
  - the multi-user prompt chunk could overflow an int after many quick steps (undefined behaviour);
  - the saved-chat store could index a size of 2^64-1, delete another `kera serve`'s write in progress, and stop
    `kera serve` on a folder read error;
  - `kera plan` stopped the program on a model whose name or architecture isn't text;
  - `kera music` stopped on a missing TMP folder, ignored a failed write, and left folders behind after Ctrl+C;
  - `kera serve` of an image model passed on more chat options the image engine stops at (`-fa`, `--mlock`,
    `--api-key` ...): now dropped, with a line saying so;
  - SDXL-Lightning's 1-, 2- and 8-step files all got 4 steps, and `sdxl_lightning` wasn't recognised;
  - setup left an empty `.kera-speed-test` folder at the top of each drive it measured;
  - a build with several configurations (Visual Studio) couldn't find the engines; the desktop check assumed user
    1000.
- **Noted, not fixed (1):** a router engine's port is chosen when `kera serve` starts but bound at the first swap
  (something else could take it meanwhile; the engine is then marked failed). By design, not a defect: the app's page
  is reachable from this computer only.

## N-015: On the CPU, the image and music engines' VAEs can't be made much faster without changing their output (2026-11-07)

- **Tried** (D-028, the cloud's 4-core AVX-512 CPU, native builds): SD-Turbo at 512x512 in one step, and a 10 s
  ACE-Step song. The picture's VAE decode takes 28 s of its 35 s, the song's 15 s of its 40 s.
- **Numbers:** stable-diffusion.cpp's direct convolutions (`--conv-direct vae=on`): 28.1 s against 28.2 s, the same
  picture. Its flash attention: 33.8 s against 35.7 s, but a different picture (a different checksum), so not
  lossless and not a default (P7). The CPU variants (D-028): the same picture time (6.0 s sampling either way),
  the song's VAE 14.5 s against 15.6 s. The music server's `--keep-loaded` saves about 5 s of loading per song
  but holds the models (3 GB or more) between songs.
- **Why:** both VAEs are bound by arithmetic (about 2 trillion operations each), already spread over every core.
  Only fewer operations (a smaller, lossy decoder such as TAESD) or the graphics chip make them much faster.
- **What next:** measure both engines on the Minisforum's graphics chip (Vulkan) and the Mac (Metal), with and
  without `--keep-loaded`, before choosing any default.
- **Later (D-030):** wrong about "can't be made much faster": the engines were built without ggml's tiled matrix
  multiply, which halves both VAEs' time on this CPU with the same output quality. Not tried here.

## N-014: Two bug sweeps before 0.1.1 found 48 defects: 45 fixed, 3 noted; one could give wrong answers (all parts) (2026-11-07)

Five reviewers read the code in parallel (the server and back-end choice, the app, expert streaming, the saved
chats and planner, and Kera's changes to llama.cpp); each finding was traced through the code before it was fixed,
and each fix tested (unit tests, the integration tests, all six tiny models exact, streamed experts identical).

- **Could give wrong answers:** a saved chat's identity left out the options that change its memory's positions or
  numbers (RoPE and YaRN scaling, `--override-kv`, control vectors, lossy routing), so a chat saved under one setting
  could be continued under another. They are part of it now.
- **Saved chats:** a decode error or a failed slot restore saved a state whose memory no longer matched its tokens
  (after which that chat failed on every reload), and an erased slot was saved too; a chat waiting to be saved was
  dropped when another chat with the same start took its slot; the 20 GB disk budget applied to each model's folder,
  not to all of them (five models could take 100 GB); a failed save left its temporary file; a damaged file could
  ask for a 512 GB allocation.
- **The app:** two quick clicks on Run crashed it (two threads replaced one thread object); on Windows, closing its
  window left `kera serve` running with the model loaded; a second copy of the app shared the first one's port; a
  failed start said nothing; a failure shown couldn't be dismissed.
- **Windows with a non-ASCII user name:** the machine profile, model hashes, model folders and the kept Vulkan shaders
  couldn't be saved or read (narrow paths in the old code page); `kera finetune` built a broken command line.
- **The per-message back-end (D-024):** a chat app that went away didn't cancel the engine's work; Ctrl+C reached the
  engines twice (llama.cpp's server then quits without saving); a failed swap stopped kera serve; requests that
  aren't prompts weren't counted, so an engine could be swapped out under them; Windows swaps counted chats as held
  that weren't saved yet; a non-string role was a 500.
- **Expert streaming:** one failed read of a guessed expert stopped every later request until a restart (and leaked
  its pages); a guess that couldn't fit emptied the cache first; a short read counted as complete.
- **Smaller:** an invalid `--stream` value aborted instead of exiting; `--lossy-kv8` overrode a `-ctv` the command
  line set; the guess-length change applied with no controller set; `kera bench --cold` claimed to work on macOS.
- **Left as they are (noted):** strict request priorities can starve a high number under steady low-number traffic
  (as in vLLM); multipart requests (audio transcription) don't pass through the per-message back-end's router; on
  Windows a saved chat being read can't be deleted until it's closed (the file stays until the next start).
- **The second sweep** (D-028: the music, speech and video code, and the first sweep's fixes, read again by a fresh
  reviewer and by the agent) found 12 more, all fixed:
  - **Another model's image reader:** the reader was matched by its name's first letters, so a 3B model took its
    7B sibling's reader (llama.cpp then stopped) and a text model could take a vision model's. Now the names must
    agree, number formats aside, up to the end of the shorter one, and a size the longer gives must be in both.
  - **Windows with a non-ASCII user name, again:** the music engine couldn't open its models or requests, and
    `kera speak -o` couldn't write its WAV (both read paths in the old code page). The engines get a UTF-8 code page
    in their manifest, and the speech tool opens its file with `ggml_fopen`.
  - **Lost work:** `kera music -o folder/song.mp3`, with the folder missing, removed the finished song and reported
    success. Now it makes the folder and, if the song still can't be written, says where it was kept. A lyrics file
    that isn't UTF-8 crashed it, and a mistyped `@file` gave empty lyrics without a word.
  - **The downloads' engines** were built for AVX2 alone (no CPU variants): they wouldn't start on CPUs without it,
    and left AVX-512 unused.
  - **From the first sweep's fixes:** a GGUF chat model cut short in its download was taken for an image model; chat
    options (`-c`, `-ngl`, `--cache-type-k` ...) reached the image server, which stopped on them; a model runner
    waiting on `/health` didn't know the image server has none; a saved-chats folder another running Kera had just
    made could be removed under it, and a save then failed; a guessed expert that couldn't fit was refused even when
    it had to be loaded.
  - **Smaller:** `kera video --help` offered WebP output, which the build leaves out.

## N-012: Streamed experts sometimes gave wrong answers on the Minisforum: a page two experts share was read twice at once (M5, M8) (2026-10-15, found and fixed 2026-10-16)

- **Tried:** `kera exactness --gpu --stream kera --stream-budget-mb 1` on the tiny models on the Minisforum, many times
  per setup (builds 0c641e4 to 8b3fef6), then a run with every tensor saved (`--dump-all`) and Kera's check of each
  expert in memory against the file (`KERA_STREAM_CHECK=1`).
- **Numbers:** on ROCm up to 35% of streamed runs went wrong (in the diagnostic runs, 40 per setup: Nemotron-H style
  30-35%, GLM-5 style 15-23%, DeepSeek style 5-18%; in the full test 0-15% per model), on Vulkan 3 of 90 GLM-5-style
  runs; in memory never, on either chip. None of
  these changed it: copying the experts to the chip through its own pinned memory (`KERA_STREAM_COPY=stage`),
  keeping the mirror's pages once used (`KERA_STREAM_KEEP=1`), computing every expert on the chip or every one on the
  CPU (`GGML_OP_OFFLOAD_MIN_BATCH` 1 or 100000), HIP graphs, AMD's serializing switches, or where the expert choices
  were read. Reading every copy back from the chip (`KERA_STREAM_VERIFY=1`) found each one equal to the memory it
  came from. A bad run against a good one: the first wrong number was layer 1's expert down product, wrong exactly
  for the bytes of one 4 KB page of the mirror (the tensor starts 2,368 bytes into a page; experts 0 and 1 share
  that page). With the check on, every bad run said so right after the experts were read: "layer 1 just brought in:
  4 of 8 experts differ from the file (8162 bytes)", and no good run did.
- **Why:** Kera read each expert's part in whole pages, so the page two neighbouring parts share was read twice,
  often at the same time by two of its reading threads. The Minisforum's C: drive is BitLocker-encrypted; an
  unbuffered read lands there as ciphertext and is decrypted in the buffer it landed in, and two such reads of one
  page at once can leave it wrong (the likely mechanism: what was measured is the mirror differing from the file
  right after both reads). The chip then computed faithfully on the wrong weights. The cloud's plain disk never
  showed it, and how often it hits depends on timing, which is why the back-end seemed to matter. Found on the way
  (kept): a false eval callback now cancels the graph compute instead of only its split (`llama_decode()` returns 2),
  and a failed read is retried three times and reported with the OS's error code.
- **Decided:** a read now writes only its own part's bytes: a chunk's first or last page, when it holds a
  neighbour's bytes too, is read into the reading thread's own page and only the part's bytes are copied in
  (`src/stream/expert_cache.cpp`, with a unit test that fails on the old reader). Checked on the Minisforum with
  build 6cee91c: 200 of 200 ROCm streamed runs and 100 of 100 Vulkan ones exact over the five tiny MoE models, and
  20 checked runs with every expert equal to the file (`bench/results/minisforum/2026-10-16-n012-fixed-grid.png`).
  `kera serve --backend hip` streams experts again. The diagnostic switches stay, off by default.

## N-013: Streaming gpt-oss-120b crashed on Windows as soon as a prompt went to the graphics chip (M5, M8) (2026-10-15)

- **Tried:** `kera bench --stream kera --stream-budget-gb 14` on the Minisforum with gpt-oss-120b (63 GB, MXFP4,
  128 experts, 4 used per word), preview 3 on Vulkan: the first test of a streamed model there that doesn't fit.
- **Numbers:** Kera stopped about 100 s in, with no message; Windows logged an access violation (0xc0000005) in
  `memcpy` (VCRUNTIME140.dll). With llama.cpp's offloading of big batches to the graphics chip turned off
  (`GGML_OP_OFFLOAD_MIN_BATCH=100000`), the same run went on.
- **Why:** for a batch of 32 words or more, llama.cpp copies the experts a batch uses from the CPU's memory to the
  graphics chip, and after each run of experts copies 512 bytes more from the next expert ("so no NaNs are in the
  padding", for NVIDIA's kernels). Kera's streamed weights hold only the experts it brought in from the SSD, and on
  Windows the memory of the others is reserved but not readable, so reading that padding faulted. On Linux the same
  read gives zeros, which is why the cloud tests never saw it; the tiny models use all their experts, so a run of
  experts never ended before the last one.
- **Decided:** for a streamed weight, llama.cpp's expert copy pads with zeros instead (as good against NaNs), and
  warns once if a streamed weight would ever be copied whole (hook 3, `docs/UPSTREAM_PATCHES.md`). Checked on the
  Minisforum with build ed84fc4: the same run, offloading on, finished, writing 2.44 and 2.27 words a second
  (`docs/BENCHMARKS.md`).

## N-011: The first run of a new Kera folder on Vulkan was slow while the driver built shaders (2026-10-15)

- **Tried:** the usual `tools/compare` run with a Kera build unzipped into a new folder on the Minisforum.
- **Numbers:** the first request read 197 words in 7.9 s (0.35-0.41 s for stock llama.cpp and for the same build
  from a folder that had run before), and guess steps took about 325 ms each instead of about 20 ms, so the code
  prompt measured 0.49x stock's speed. Run again, the same build measured 1.05x, and every prompt was normal.
- **Why:** llama.cpp's Vulkan back-end builds each shader the first time a batch shape needs it, and AMD's driver
  keeps its cache of built shaders per program; a new program path starts empty, and guessing brings many new
  shapes. Stock llama.cpp's folder never changed.
- **Decided:** measurements on the Minisforum run twice after a new build and keep the second run (or use a folder
  that has run before). Worth fixing for users too, since every update starts cold. llama.cpp's Vulkan back-end
  builds each shader the first time it's needed and gives Vulkan no pipeline cache (`VK_NULL_HANDLE`), so only the
  driver's own per-program cache remembers them.
- **Fixed (build 24e2675):** a pipeline cache file in Kera's folder (`gpu.shader_cache`, a small llama.cpp patch),
  loaded at start, used for every shader built and saved as it grows. Three fresh copies of the build on the
  Minisforum, Nemotron 3 Nano, first request with no warm-up: with an empty cache file 3.36 s to the first word and
  49.7 words a second; with that file (304 KB) in another new folder 0.18 s and 59.6; without a file 3.20 s and
  48.4. On by default since (D-021).

## N-010: On the Minisforum's graphics chip, Vulkan's coarser rounding fails the tiny-model exactness check; ROCm passes (2026-10-15)

- **Tried:** `kera exactness MODEL GOLDEN --gpu --backend vulkan` and `--backend hip` on the six tiny test models
  (the reference code's own outputs, spec P2), Kera 1fc1eea on the Radeon 8060S.
- **Numbers:** ROCm passed all six: every step picked the reference's word, and the top-5 log-probabilities were
  within 0.001 of the reference, as on the CPU. Vulkan picked the right word at every step on five designs, but
  their log-probabilities were off by 0.07-0.12 (the limit is 0.02), and on the DeepSeek-style MLA model one of the
  top 5 was off by 24; on the GLM-5-style model (MoE with DeepSeek's sparse attention) it picked 3 wrong words of 24,
  none at a near-tie. With llama.cpp's cooperative-matrix maths turned off (`GGML_VK_DISABLE_COOPMAT=1`) the MLA
  model's worst difference fell to 0.09, but the GLM-5-style model then picked 4 wrong words; turning off 16-bit
  maths, fusion or sparse flash attention changed nothing.
- **Why:** Vulkan rounds more coarsely on this chip (about 4 parts in 10,000 on the sparse-attention scores; ROCm
  and the CPU agree with the reference to about 1 in a million). For the GLM-5-style model that was enough to flip a
  near-tie inside the attention, not in the word chosen: the model lets each word look at only its 8 best-scoring
  earlier words (the "indexer"), and for word 41 the 8th and 9th scored 1.153796 and 1.153789 on the CPU; Vulkan
  scored them the other way round and looked at another word, which changed 3 later words. Found with `kera
  exactness --dump-all` on the CPU and on Vulkan (with `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`, so both compute in the same
  order) and `tools/exactness/first_difference.py`: every operation up to that choice agreed within rounding. So no
  operation is broken. Kera's own changes aren't involved either: Kera on Vulkan writes the same words as stock
  llama.cpp on Vulkan.
- **Decided:** section 6's near-tie rule must also cover a top-k choice inside the model (sparse attention's
  indexer, a MoE router), where a near-tie can change more than one word; the exactness tool should trace a
  difference back to such a choice and accept it when it was a near-tie (to do). ROCm stays the more exact choice on
  this chip, and Vulkan the faster writer (D-018). The coarse rounding behind the 0.07-0.12 log-probability
  differences, and the 24 with cooperative matrices on the MLA design, is llama.cpp's Vulkan maths on this chip; worth
  a closer look upstream.
- **Then (the MLA model's 24, traced):** it is another near tie, in a MoE router this time. It reproduces on the
  cloud with Mesa's software Vulkan driver (24.04), with the experts in memory or streamed. Traced as above: every
  operation agrees within rounding up to layer 2's experts, where for word 1 the router scored expert 6 at 0.789534
  and expert 5 at 0.789414 on the CPU (the second and third best, 0.00012 apart); Vulkan scored them 0.789525 and
  0.789528 and used expert 5, which moved one of that word's top 5 log-probabilities by 24 (a word the model gives
  almost no chance). The other 23 words differ by 0.03-0.09, Vulkan's usual rounding. Streaming isn't involved: on
  the Minisforum Vulkan gives 24.04 streamed and in memory, every time.
- **Then (the to-do, done):** `kera exactness --gpu` now follows section 6's rule for a graphics chip: the words are
  judged with the near-tie rule, and the log-probabilities get a limit of 0.25 (0.02 stays for the CPU). When the
  check fails on a chip, the tool runs the CPU and keeps every choice inside the model (each MoE router's experts,
  each sparse-attention indexer's words), then runs the chip again, taking the CPU's choice wherever the chip chose
  differently and its scores for the swapped candidates were within 1% of the CPU's (only their order changed). That
  run must pass; a choice that differs with the chip's scores further off fails at once. On Mesa's software Vulkan
  all six tiny models pass: the DeepSeek-style one after its router near tie, the GLM-5-style one after two near
  ties in its indexer (the first 2.31794 against 2.31733), which had changed 3 words.

## N-009: Guessing could stop a reply for good on a MoE model on a graphics chip (2026-10-15)

- **Tried:** `kera serve --backend hip` on the Minisforum (Nemotron 3 Nano on ROCm), the usual `tools/compare` prompts.
- **Numbers:** the code-rewrite prompt stopped at its 179th word every time; the server kept answering status
  requests and one CPU core stayed busy, for 10 minutes until stopped. With `LLAMA_TRACE=1`: "accepted 5/16 draft
  tokens (restore checkpoint)", then "accepted 5/6 (restore checkpoint)" over and over, 560 times and counting.
- **Why:** when only part of a guess is right, a hybrid model like Nemotron-H can't drop the wrong words from its
  memory, so llama.cpp's server restores the state from before the guess and reads the accepted words again (a
  "replay"). With greedy decoding it checked those words again. On a MoE model on a graphics chip the result for one
  word depends on the other words in the batch (they share experts, which changes the rounding), so the replay
  flipped a near tie, the server restored and replayed again, and that flipped it back: a loop with no progress.
  llama.cpp already accepts replayed words without checking them when sampling with a temperature; not when greedy.
  It can happen on any graphics chip and any MoE model where a near tie falls just so; ROCm on this prompt hit it.
- **Decided:** a replay always accepts its words, greedy too (they were chosen when the guess was checked), listed in
  `docs/UPSTREAM_PATCHES.md`. Worth offering upstream. `tools/compare` now gives up when a server sends nothing for
  `--stall` seconds instead of waiting two hours.

## N-008: On Windows, `kera serve` could stop at once with "invalid argument: serve" (2026-10-15)

- **Tried:** `kera serve ... --no-auto-guess` on the Minisforum (to check where answers differ from stock's).
- **Numbers:** Kera stopped while loading, every time, with llama.cpp's "error: invalid argument: serve".
- **Why:** on Windows, llama.cpp's `common_params_parse()` swaps the options it is given for the program's own
  command line, read again as UTF-8, whenever the two have the same number of words. Kera's commands take their own
  options out and add others before passing the rest on, and with `--no-auto-guess` the count happened to match, so
  llama.cpp read the raw command line, `kera.exe serve ...`, and took `serve` for an option. Any mix of options could
  hit it by chance. The same code meant Kera itself never read its options as UTF-8 on Windows: a model under a
  folder with an accented or non-Latin name couldn't be opened, from the app too.
- **Decided:** llama.cpp swaps only when the words in plain ASCII match one for one (they read the same in any code
  page; listed in `docs/UPSTREAM_PATCHES.md`), and Kera's `main()` reads its own words as UTF-8 on Windows. Worth
  offering upstream.

## N-007: The ROCm add-on from preview 2 didn't load, then stopped at its first sum (M8) (2026-10-15)

- **Tried:** `kera serve --backend hip` on the Minisforum with preview 2's ROCm add-on unzipped next to `kera.exe`.
- **Numbers:** Windows refused `hip\ggml-hip.dll` with error 126 (a DLL it needs is missing); AMD's runtime DLLs in
  the add-on loaded fine. `ggml-hip.dll` imports `hipblas.dll`, which the add-on didn't carry.
- **Why:** the add-on copied llama.cpp's own Windows ROCm release, which ships the same four files and expects
  hipBLAS, rocBLAS and hipBLASLt from an installed HIP SDK ("resolve fine via PATH"). A normal PC has none. CI built
  the add-on but never loaded it, and has no AMD graphics chip, so nothing noticed.
- **Decided:** the add-on now carries every ROCm DLL `ggml-hip.dll` needs (found by following its imports), and CI
  loads it before packing, which needs no graphics chip. Kera now says "missing a file (error 126)" instead of
  blaming the driver.
- **Then:** that add-on loaded on the Minisforum and found the chip, but Kera stopped at the first question:
  rocBLAS couldn't find its kernels (`rocblas\library\TensileLibrary.dat` for gfx1151). AMD's packages keep a chip's
  kernels in a separate "device" package (`rocm-sdk-device-gfx1151`), which `rocm[libraries]` leaves out, so the copy
  step found nothing and said nothing. rocBLAS and hipBLASLt also look for their own code for the chip in
  `..\.kpack\blas_lib_gfx1151.kpack`, relative to their DLL. With those files copied in by hand, Kera ran Nemotron 3
  Nano on ROCm. CI now installs the device package, copies the kernels into `hip\rocblas` and `hip\hipblaslt` and the
  kpack into a `.kpack` folder next to `hip`, and fails if any is missing; Kera checks for them before loading the
  add-on. The add-on CI built that way (build 1fc1eea) ran every test on the Minisforum.

## N-006: With long documents, Kera on Vulkan reads more slowly than Ollama on ROCm (M8) (2026-10-15)

- **Tried:** the long-context test in `tools/compare` on the Minisforum: Nemotron 3 Nano, a document of 7.7K to 95K
  words and a question, Kera preview 2 (Vulkan, the default) against Ollama 0.30.9 (ROCm), 128K context.
- **Numbers:** Kera read 1,165 words a second at 7.7K but 671 at 34K, 436 at 64K and 316 at 95K; Ollama 875, 861,
  700 and 639. So the first word came later with Kera from about 30K words on: 303 s against 150 s at 95K. Kera
  still wrote faster at every length (42.4 against 36.0 words a second at 95K) and answered follow-ups 3.5-4.5
  times sooner.
- **Why (likely):** llama.cpp's Vulkan back-end slows down more than ROCm as attention covers more words on this
  chip. Not yet profiled.
- **Decided:** measure Kera's ROCm add-on on the same test once it loads (N-007); if it reads as fast as Ollama,
  Kera should pick ROCm for long prompts on this chip (M8's "choose Vulkan or ROCm by measured speed"). Vulkan stays
  the default until then.
- **Then (ROCm add-on, build 38e4543, guessing off):** Kera on ROCm read the same documents at 861, 846, 766 and 706
  words a second: the first word after 9.0 / 39.9 / 84 / 136 s, against Ollama's 9.1 / 39.4 / 92 / 150 s and Kera on
  Vulkan's 6.6 / 50.4 / 147 / 303 s. It wrote at 44.7-39.3 words a second (Ollama 40.7-36.0, Vulkan 58.5-42.4) and
  answered follow-ups in 0.21-0.32 s. So ROCm is the faster choice for documents over about 20K words, Vulkan for
  short ones and for writing. ROCm also passes the tiny-model exactness check on this chip, where Vulkan doesn't
  (N-010). Vulkan stays the default for now (D-018 has the plan); for long documents, `--backend hip`.

## N-005: Saving chats held up follow-ups and restarts on the Minisforum (M2) (2026-10-15)

- **Tried:** `kera serve` from preview 2 against stock llama.cpp b11619 on the Minisforum's graphics chip (Vulkan),
  with Nemotron 3 Nano 30B-A3B (Ollama's 22.8 GB file) and `tools/compare`: a 12-message chat sent the way chat
  apps send it (the whole history each time), then a restart and one more message.
- **Numbers:** stock's wait before each reply stayed at 0.29-0.32 s. Kera's grew from 0.30 s to 2.08 s by message
  12 (median 0.66 s). Both read the same 165 or so new words per message in the same time (about 270 ms); the
  difference is the time between the end of a reply and taking the next message: stock 15-25 ms, Kera 70 ms rising
  to 1.8 s as the chat grew. After a restart Kera took about 2 s to load 2,657 words of the chat from the SSD, then
  read the 165 new ones: its first word came at 2.99 s, stock's at 2.33 s after reading all 2,822. Against Ollama
  (0.63 s and 1.29 s), Kera still came out ahead.
- **Why:** the time goes to Kera's save at the end of each reply (`prompt_store_save()` in llama.cpp's server, a
  Kera hook): it copies the chat's whole state, and every context checkpoint llama.cpp keeps for it, in the
  server's main loop, so the next message waits. The copy grows with the chat, and a hybrid model like Nemotron-H
  keeps several checkpoints, each holding the model's recurrent state. Loading a saved chat reads all of it back.
  The cloud tests used tiny models, where this takes microseconds.
- **Decided:** fix it in M2 before Gate 5b: save in the background without holding the main loop, keep only what a
  resumed chat needs, then run the same test again. Until then Kera isn't ready for long chats on hybrid models.
- **Fixed (measured on the Minisforum, build 1fc1eea):** a chat is now saved once the server has been idle for
  250 ms (or before its slot is reused), the store gets only a chat's newest 2 checkpoints, and saving or loading a
  state copies a tensor's many small pieces in one go on a graphics chip. Follow-ups now wait 0.30 s like stock's
  0.31 s, and the first word after a restart comes in 0.6-0.65 s against stock's 2.4-3.5 s. Writing is 1.16 times
  stock's on average (1.30x on the chat, 1.40x on the rewrite, where copying from the chat pays). On the way, the
  first version of the batched copy also batched a hybrid model's few big recurrent-state rows, which sit far apart:
  every guess step took 140 ms and guessing turned itself off (the code prompt fell to 0.53x). Batching only many
  small pieces packed close together fixed that.

## N-004: Shadow Bits: no nested version is worth building; M7 is skipped (M1 risk test 1) (2026-10-15)

- **Tried:** spec P5's four versions, on Qwen3-1.7B in PyTorch on the cloud CPU (Qwen3-0.6B as the separate draft),
  on the 16-bit model's own answers to 149 OpenAssistant prompts (9,467 words): SB4 as written (2-bit base + 2-bit
  correction from Appendix A's grid search), SB4 with BitNest's base (GPTQ 2-bit base, the correction a rounding of
  what is left), BitNest's 4+4 (GPTQ 4-bit base inside an 8-bit model), and no nesting (plain 4-bit plus the small
  draft). Details and the full table: `docs/M1_RISK_TESTS.md`, RT1.
- **Numbers** (acceptance of the draft by the full model; KL of the full model from 16-bit, against its plain
  counterpart's): SB4 as written 0.000 and 0.668 against 0.609 (+10%): its 2-bit base alone writes nothing right
  (top word right 0% of the time). SB4 with BitNest's base 0.424, and 2.757 (4.5 times plain 4-bit): the 2-bit
  correction can't reach what GPTQ's base leaves over. 4+4: 0.846 (passes), but 0.086 against plain 8-bit's 0.0023
  (38 times): the 8-bit model inherits the 4-bit base's clipped range. No nesting: 0.663. The planner's predicted
  speed for a dense model bigger than memory: 32B on the Mac 0.68 tok/s with the separate draft, 0.38 at best
  nested; 123B on the Minisforum 0.70 against 0.36.
- **Why:** the go rule needs a nested version with acceptance of at least 0.70 and a full model within 3% of plain
  quality at the same size. The 2-bit bases are too poor to guess with, and building the full model on top of a
  base costs quality that plain weights of the same size don't lose. Even where the base guesses well (4+4), every
  guess reads the whole base from memory (36 GB for a dense 70B), while a 0.6B draft reads 0.3 GB, so the separate
  draft wins.
- **Decided:** M7 is not built (spec P5: "if the no-nesting version wins, skip M7"); so neither are the lossy
  Fast and Balanced modes (Phase 11), which are SB4's base alone. Separate draft models (M4, M6) are the way to
  guess for dense models bigger than memory.
- **What could change it:** this ran on a 1.7B model (the spec asks for 3-4B, or 8B on the Minisforum; 1.7B in
  32-bit floats was the largest that fit this machine's 15 GB). Bigger models lose less at 2 bits, but the gaps
  here are wide: a rerun with `tools/risk_tests/rt1_shadow_bits.py` and `rt1_predict.py` on an 8B model on the
  Minisforum would settle it.

## N-003: With llama.cpp's default maths, checking guesses can flip near-equal words on MLA and DSA models (2026-10-14)

- **Tried:** M4's guessing test (`tests/integration/test_guess_controller.py`): greedy output with guessing on
  must match guessing off word for word (section 6). Tiny GLM-5-style (MLA attention plus the DSA sparse
  indexer, with an MTP head) and DeepSeek-style (MLA) models, llama.cpp b11619 on the CPU, 495 words.
- **Numbers:** with llama.cpp's defaults (16-bit memory of the text, flash attention), llama.cpp's own
  guessing (fixed lengths, no Kera code involved) changed the GLM-5-style output from word 162 on, with a
  draft model or the model's MTP head, although every guess was accepted. Scoring a text in one pass and
  writing it word by word differ by up to 0.012 in log-probability on both designs with 16-bit memory, and
  by 0.00008 with 32-bit memory. With the indexer keeping its top 8 positions, those differences change
  which positions are attended to, and the log-probabilities then differ by up to 2.9. With the indexer
  keeping every position (as real GLM-5 does, top 2048, until a chat is longer than that), the first
  changed word was a choice between 37.9% and 37.2%. Without guessing, llama.cpp's server and a
  word-by-word run also differ (word 66) on this design, from flash attention alone.
- **Why:** reading several words at once (a guess check, or a bigger prompt batch) rounds differently from
  reading one at a time. Section 6's near-tie rule (top two within 0.1 percentage points) assumes
  smaller differences than 16-bit memory gives these designs, and a top-k choice inside the model turns
  small differences into different attention.
- **Decided:** guessing is judged word for word with 32-bit memory and no flash attention (D-014): then
  every setup matches, MTP and streaming included, on both designs. Kera keeps llama.cpp's defaults for
  normal use, as llama.cpp's own guessing does; the effect is the same as choosing another batch size.
  To measure on the Minisforum with real GLM-5 (whose indexer only chooses beyond 2048 words): how often
  guessing changes a word under the default maths, and whether 32-bit memory for the indexer alone is
  enough. The tiny GLM-5 model with an MTP head keeps every position in its indexer, so the test checks
  guessing itself.

## N-002: Learned expert pins lose to plain LRU (Phase 4, Gate 4) (2026-10-14)

- **Tried:** spec Phase 4, on routing traces of Nemotron 3 Nano 30B-A3B (Q4_0; 64 chats and an agent
  session, 38,558 words; M1 risk test 2). Pins learned from usage on the first half of the traces and
  tested on the second half (held-out prompts), in four forms: pins alone, pins filling 10%, 30% or 50%
  of each layer's cache with LRU for the rest, the best possible fixed pins chosen with hindsight, and
  "prompt pins" (while a reply is written, keep the experts its prompt used most).
- **Numbers:** with a quarter of each layer's experts in memory, plain LRU finds 62% of a word's
  experts; pins alone 34%; the best possible fixed pins 42%; 30% pins plus LRU 59%; prompt pins plus
  LRU 63%. At every cache size from 5% to 75%, nothing beat LRU by more than 0.2 points, and Gate 4
  asks for 10 (full table in `docs/M1_RISK_TESTS.md`, RT2).
- **Why:** a word's experts are mostly the ones the last few words used. A fixed set of popular experts
  can't follow that, and every pinned slot is one LRU can't use.
- **Decided:** pins and prompt pins are off by default in Kera's expert cache (`stream::config`). The
  code and the per-model usage log stay, so the gate can be re-run on the 120B-class model on the
  Minisforum, where the spec's gate is defined; if it fails there too, the code goes.

## N-001: Nemotron-H tiny models "failed" because the reference disagrees with itself (2026-10-14)

- **Tried:** the P2 exactness test on the two Nemotron designs (hybrid Mamba-2/MoE/attention, and
  LatentMoE), llama.cpp b11619 on the CPU against transformers 5.19.0, in the cloud.
- **Numbers:** every word matched (24/24), but the top-5 log-probabilities differed by up to 0.264
  (limit 0.02). DeepSeek-V3/Kimi MLA (1.0e-4) and GLM-5 DSA (8.2e-4) passed.
- **Why:** transformers' own two ways of running the model disagree. Reading a prompt in one pass
  and adding one word at a time differ by up to 0.234, even in 64-bit maths. Its Nemotron-H code
  clamps Mamba's step size at `time_step_min` (0.001) in the prompt path only. NVIDIA's original code,
  transformers' own Mamba-2 model and llama.cpp all use `time_step_limit`, which is (0, infinity): no
  clamp. An independent 64-bit numpy version of the Mamba-2 layer matched llama.cpp's layer to 2e-6.
- **Decided:** the tiny-model script uses the original no-clamp setting
  (`use_original_dt_limit` in `tests/tiny_models/make_tiny_models.py`). Then both reference paths
  agree, and llama.cpp matches to 5.3e-5 and 5.7e-5. llama.cpp needed no fix. Real Nemotron models
  should be compared against llama.cpp or NVIDIA's original code, not transformers' prompt path.
  This is worth reporting to transformers (an outside project, so it waits for the owner).
- **Real models too:** Nemotron 3 Nano's own `config.json` (NVIDIA's repo, checked 14 Oct 2026) sets
  `time_step_min` to 0.001 and has no `time_step_limit`, so transformers clamps the real model the same
  way when it reads a prompt.
