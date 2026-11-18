# Kera: build spec for the coding agent

You are building **Kera**, a local AI engine that runs models as fast as the hardware allows: dense and mixture-of-experts (MoE), both those that fit in memory and those too big for it. It also supports fine-tuning.

**How to use this file:** read the ground rules (section 0), then the **build playbook** right after them. The playbook gives the milestone order (M0–M12), the shared interfaces, and how to work. The numbered sections and phases after it hold the detail for each milestone. **All code is written in the GitHub repo the owner links for you** (playbook P8): every change is committed to the working branch (the owner chose no pull requests, see below), GitHub builds and tests it, and downloads are published as GitHub Releases. Never skip a pass/fail gate.

The person you work for is the owner. He builds on an **M4 MacBook Pro with 16GB** and runs a **Minisforum mini PC with an AMD Ryzen AI Max+ 395 and 64GB (Windows)**. He has no NVIDIA machine yet. Build order: **AMD (the Minisforum) first, then Mac, then NVIDIA** (the owner changed this on 14 Oct 2026; it was Mac first). He wants explanations in plain words, without jargon. When you report to him, say what changed, what you measured, and what it means.

**Kera is a downloadable app.** Someone downloads it, opens it, and the app does the rest:
- it measures the machine by itself: SSD speed, RAM speed, graphics chip, free disk;
- it recommends which models fit and how fast each will run;
- it tunes its own settings for that machine and that model.

Nobody should ever have to measure anything by hand or edit a settings file. The first download is for Windows, so the owner can test on his Minisforum from the start. Section 4a describes the self-setup.

## Owner decisions that change this spec (14 Oct 2026)

The owner made these changes after writing the spec. Where they disagree with the text below, they win. `docs/DECISIONS.md` (D-001 to D-007) has the details.

1. **Windows first.** The Minisforum comes first and the Mac second. The first download is the Windows app, and the F0 baselines and the streaming work are built and measured on the Minisforum first. For the Mac, do the parts GitHub's macOS machines can build and test; real-Mac measurements wait for the owner.
2. **Where the work happens.** The agent works in the cloud. All code lives in the GitHub repo, and GitHub's own machines build and test it. The Minisforum only runs finished packages, which the agent reaches over Tailscale. Nothing is built or developed on it.
3. **No pull requests.** Commit and push straight to the working branch, with the gate result in each commit message. Releases are tagged from that branch, and a milestone is merged into `main` once its gate passes. The P7 checklist still applies to every commit.
4. **Don't stop.** If a gate fails twice, or a milestone takes more than twice its size, log it in `docs/NEGATIVE_RESULTS.md`, write the options in `docs/STATUS.md`, and move on to milestones that don't depend on it. This replaces the "stop and report" rules in P1.
5. **Permissions granted** (exceptions to ground rule 3):
    - download any model needed, as long as at least 100 GB stays free on the Minisforum's disk; keep downloads inside the Kera folder and delete them when no longer needed; use the owner's existing models where they are, and never move or delete them;
    - install the GitHub runner on the Minisforum as a Windows service labelled `amd`, while the repo is private (it is public now, so this permission is withdrawn);
    - run benchmarks whenever needed, closing other AI apps on the Minisforum meanwhile and reopening them afterwards;
    - merge the agent's own work once the checks are green and the gate passed.
6. **Never, and don't ask either:** anything needing admin rights after the first setup; buying or renting anything; payment methods or plan changes; adding people to the repo; BIOS, driver or Windows changes; deleting anything outside the Kera folder; anything on hardware that isn't the Minisforum (the Mac, NVIDIA, a DGX Spark). Skip these and list them in `docs/STATUS.md` under "Waiting for the owner".
7. **Serving several people at once is a goal (17 Oct 2026).** The owner asked for vLLM's benefits for serving several people at once, so the non-goal "Serving many users at once" and the first item of 12.4 no longer hold. Each piece is still a setting that stays off until its gate passes (`docs/DECISIONS.md` D-023).
8. **The MacBook for a test app (29 Oct 2026).** The owner asked for their 128 GB MacBook Pro to be used, with a small Mac app that races Kera against Ollama live (`docs/DECISIONS.md` D-025). For that MacBook this replaces "anything on hardware that isn't the Minisforum" in item 6.
9. **Published on Hugging Face, open source (4 Nov 2026).** The owner chose to publish Kera publicly on Hugging Face, downloads and source, under the Apache License 2.0 by Kera Labs (`docs/DECISIONS.md` D-026). The GitHub repo was private then; the owner made the fork public on 18 Nov 2026 and cancelled D-004.
10. **Image models (7 Nov 2026).** The owner asked for version 0.1.1 to run image models, both those that see images (vision chat) and those that make them (text to image), with faster model loading and a full bug sweep (`docs/DECISIONS.md` D-027). Making images adds a second engine, stable-diffusion.cpp, beside llama.cpp.
11. **Speed only, never quality (9 Nov 2026).** Rule number one: Kera never changes how a model responds; it only makes things faster (`docs/DECISIONS.md` D-033). A model runs with its own template and its own card's settings, thinks as it does in a chat, and answers in full; seeds stay random where the engine leaves them random. Kera has no mode, option or setting that trades quality for speed, not even an opt-in one (D-034). This drops Phase 11's lossy modes, M12's lossy work (S7 "lossy turbo" in the M9 table, the 8-bit chat memory, cached-expert routing, passage reuse, QuantSpec) and every "lossy mode" mentioned below; where this spec says a trick may run as a lossy mode, it is not built. The engines' own command-line options still pass through as in the stock engines, but Kera neither adds nor suggests any that lower quality.

---

## 0. Ground rules

1. **Measure, don't claim.** No speed-up counts until the benchmark in section 7 shows it. Change one thing at a time. Log every result, good or bad, in `docs/BENCHMARKS.md`. Log failures in `docs/NEGATIVE_RESULTS.md`.
2. **Placement never changes the answer.** Where a weight lives (GPU memory, RAM, SSD) and whether you guess ahead may change speed, never the output. A trick that changes the output (beyond the rounding section 6 allows) is not built, not even as an opt-in mode (owner decision 11; D-034).
3. **Ask the owner first** before you:
    - download more than 50 GB;
    - install system-wide packages or run `sudo`;
    - rent or buy anything;
    - delete files outside the repo;
    - change BIOS, driver or OS settings;
    - add people to the repo, add a payment method or change its plan;
    - install a GitHub runner on one of his machines (P8).
4. **Don't reinvent what exists.** Writing fast GPU kernels for Apple, AMD and NVIDIA from scratch takes years. Build on an existing engine (section 3) and spend your effort on the parts nobody has built.
5. **Verify platform facts.** Anything marked *(verify)* comes from research that may be out of date. Check the current docs or source before relying on it, and note what you found in `docs/DECISIONS.md`.
6. **Push every passed step to GitHub** (P8; no pull requests, per the owner), with the gate result in the commit message (P7).
7. **If a Kera codebase already exists in the owner's projects or on his GitHub, read it first.** Keep what works and say what you are replacing and why.
8. **Never put secrets or model weights in the repo.** No passwords, tokens, keys or signing certificates in any file or commit (they go in GitHub's encrypted secrets). No model files in git (they are downloaded from Hugging Face when needed).
9. **Keep the owner's name and personal details out of the repo.**
    - No real names, emails, usernames or computer names in code, comments, file names, labels, branch names, runner names, commit messages or docs.
    - In repo files, call him "the owner".
    - Commits use GitHub's private no-reply email address, never a real one.

---

## Build playbook (read this before writing any code)

This is the order and method for building Kera with the least wasted work. It is designed to:
- measure before optimising;
- test risky ideas cheaply before spending months on them;
- give the owner something useful early;
- define the parts so they can be built and tested separately, including in parallel.

### P1. How to work

1. **Profile first, then optimise.** Every optimisation starts with a profile showing where the time goes, and ends with a before/after benchmark (section 7). If the profile says the time is elsewhere, work there instead.
2. **Everything behind a setting.** Each new technique is registered as a setting with a safe default, off until its gate passes, so the auto-tuner (4a) can switch it. Nothing is hard-wired on.
3. **Small steps that always work.** Keep the main branch green on all three systems. Merge small changes often. Never leave the app unable to start.
4. **Correct before fast.** A new kernel or format lands only after it matches the reference output (section 6) on the tiny test models (P2).
5. **Stop rules** (changed by the owner: don't stop):
    - If a milestone takes more than twice its size estimate (P4), or a gate fails twice, write the options in `docs/STATUS.md` and move on to milestones that don't depend on it.
    - Log the failure in `docs/NEGATIVE_RESULTS.md`.
6. **Use the right profiler per platform** *(verify current tools)*: Xcode Instruments (Metal System Trace) on Mac, Radeon GPU Profiler (Vulkan) or `rocprofv3` (ROCm) on AMD, Nsight Systems and Nsight Compute on NVIDIA, including GB10.
7. **Report in plain English** after every milestone (section 9), including the one thing the owner can now try himself.

### P2. Set up once, in M0

- **Build:** CMake, the same as llama.cpp. The engine is C/C++. Python is only for converters, tools and tests. The app's interface is a local web page (HTML/TypeScript) served by the engine.
- **Continuous integration (GitHub Actions):** automatic builds and tests on GitHub, set up as in P8.
    - Platforms covered: macOS arm64 (Metal), Windows x64 (Vulkan), and Linux x64 (CPU tests, a Vulkan build, and a CUDA build-only check).
    - P8 says which run on GitHub's machines and which on the owner's, so the free allowance lasts.

    Every platform builds from day one, so ports never rot. The platform-specific speed work still follows the owner's order: Mac, then AMD, then NVIDIA.
- **Tiny test models:**
    - For each target design (Nemotron-H hybrid Mamba/MoE, GLM sparse attention with MLA, Kimi/DeepSeek MLA, Nemotron LatentMoE), generate a tiny random-weight model from its config with Hugging Face transformers.
    - Store "golden" outputs from the reference code.
    - CI checks Kera against them token for token on the CPU (Colibri does the same).
- **Speed tests on real hardware:**
    - With the owner's agreement, install GitHub runners on his Mac and Minisforum (and Spark later), as in P8.
    - A nightly benchmark fails, and opens an issue, if any bench model gets more than 3% slower.
- **Keep the llama.cpp copy easy to update:**
    - All Kera code lives in `src/`, `app/` and `tools/`.
    - Changes inside `third_party/llama.cpp` are small hooks, each listed in `docs/UPSTREAM_PATCHES.md`.
    - Pull in new llama.cpp at least monthly: a scheduled workflow (P8) opens a pull request, and full CI decides.
    - Offer generic fixes upstream, such as AMD Mamba prompt reading or few-row checking kernels. It cuts long-term maintenance and the licence allows it (keep all notices).

### P3. Shared interfaces (frozen in M0, so parts can be built separately)

C-style sketches; adjust names, but keep the shapes. Each part gets unit tests against fakes of the others.

```c
/* Machine profile: written by auto-setup (4a), read by planner and tuner. JSON file kera_machine.json:
   os, chip, ram_gb, usable_gb, gpu{name, max_alloc_gb}, ram_gbps{gpu, cpu, both},
   drives[{path, read_gbps, best_read_mib, best_inflight}], measured_at, kera_version */

/* Weight store (src/store): where every weight lives and how it gets to memory */
const void *kr_expert_get(kr_store *s, int layer, int expert,
                          kr_ready_cb on_ready, void *user);   /* returns the pointer if in memory, else NULL and calls on_ready */
void kr_expert_prefetch(kr_store *s, int layer, const int *experts, int n,
                        double deadline_ms, float confidence);  /* deadline-ordered queue (10.3) */
void kr_store_stats(kr_store *s, kr_store_stats_t *out);       /* hits, misses, GB read, I/O wait */

/* I/O back-end (src/io): one per OS, big aligned direct reads, completion callbacks */
int  kr_io_read(kr_io *io, int file_id, uint64_t offset, size_t len, void *dst,
                kr_io_done cb, void *user);

/* KV store (src/kvstore, section 4b) */
int  kr_kv_lookup(kr_kvstore *k, const kr_cache_id *id, const int32_t *tok, int n); /* tokens already saved */
int  kr_kv_load(kr_kvstore *k, const kr_cache_id *id, const int32_t *tok, int n, kr_kv_dst *dst);
void kr_kv_save_async(kr_kvstore *k, const kr_cache_id *id, const int32_t *tok, int n, const kr_kv_src *src);
void kr_kv_pin(kr_kvstore *k, const kr_cache_id *id, const int32_t *tok, int n, int pin);

/* Guessers (src/spec, 11.1): MTP head, DFlash, EAGLE-3, small draft model, SB4 base, prompt copy, suffix, lookahead */
typedef struct {
    const char *name;
    int  (*propose)(void *self, const kr_ctx *ctx, int k, int32_t *out_tok, kr_routes *out_routes);
    void (*accepted)(void *self, int n_accepted);
} kr_guesser;

/* Guess-ahead controller (Appendix B) */
int kr_choose_k(const kr_ctrl *c, double t_draft, double t_check, kr_bytes_fn new_bytes, double ssd_gbps);

/* Settings registry: every knob, so the auto-tuner can see and try it */
void kr_setting_register(const char *name, double def, double min, double max, int tunable, const char *help);
```

**Bench results:** JSON with these fields:
- machine profile hash, Kera commit, model hash, every setting, prompt id, cache state;
- `ttft_ms`, `prefill_tok_s`, `decode_tok_s` (median of 3);
- `ssd_gb_per_tok`, `expert_hit`, `prefetch_acc`, `kv_hit`, `accept_rate`, `k_histogram`;
- `peak_ram_gb`, `ceiling_tok_s`, and the gate results.

The app's dashboard and `docs/BENCHMARKS.md` both read these files.

### P4. Milestones in order

Sizes are rough, for one agent plus the owner:
- **S:** up to 3 days;
- **M:** 1–2 weeks;
- **L:** 3–6 weeks;
- **XL:** 2–3 months.

| # | Milestone | Covers | Needs | What the owner can try afterwards | Size |
|---|---|---|---|---|---|
| M0 | Foundations | The GitHub repo and workflows (P8), Phase 0 (Mac download + self-measuring), P2 (CI, test models, nightly bench), P3 interfaces, section 11 step F0 baselines on Mac and Minisforum | — | Download the Mac app and see his machine measured in plain words; baseline speeds for his models | M |
| M1 | Risk tests (P5), run in parallel | Five short experiments that decide later milestones | M0 | A one-page "go / no-go" summary | S each |
| M2 | Quick wins for chats | 4c (keep the model loaded, thinking switch, wait breakdown, tool calling), Phase 5b (KV reuse, Mamba snapshots, saved chats), the OpenAI/Anthropic-compatible server, plus the AMD prompt-reading fix if M1 finds one | M0 | Nemotron 3 Nano on the Minisforum with his 8-second wait broken down and cut | M |
| M3 | Speed planner | Phase 1 | M0, M1 routing traces | Accurate "how fast will this run" predictions in the app | S |
| M4 | Guess-ahead for models that fit | 11.1, F1–F2, the core of Phase 6 (controller) | M0 | Dense models about 2–3x faster than without guessing; the switch picks the best guesser | M–L |
| M5 | Streaming giant models (Minisforum first, then the Mac) | Phases 2–5, plus the overlap schedule (10.5) where M1 says it pays | M3 | A 120B-class MoE running from the SSD on his Minisforum, then his MacBook | L |
| M6 | Guess-ahead for streamed models | Rest of Phase 6 (union growth, expert-aware guessing) | M4, M5 | Faster streamed dense models; MoE guessing only where it pays | M |
| M7 | Shadow Bits (only if M1 says go) | Phase 7 | M6 | Dense models bigger than memory about 3x faster than plain streaming | L |
| M8 | Minisforum (AMD) | Phase 8, plus F1–F2 and F4–F5 on AMD (ROCm for prompt reading, Vulkan for writing, Mamba fix). Fused kernels come in M10 | M2–M6 | The Windows download; GLM-5.3 and Kimi streamed; Nano and Lightning faster than llama.cpp | L |
| M9 | NVIDIA and DGX Spark | Phase 9, or the Spark track (10.11, steps S0–S8) if he has a Spark | M8 | The section 10.2 targets | L–XL |
| M10 | Fused whole-word kernels | 11.2 (F3) per platform, hardest first where it pays most (small-active MoE) | M4, the platform's port | Another 1.2–1.6x on Nano-class models | XL |
| M11 | Fine-tuning | Phase 10 | M2 | Train a LoRA on the Mac and load it in Kera | M |
| M12 | Lossy modes and experiments | Phases 11–12, 10.9, 11.5 | the relevant milestones | Opt-in turbo modes with quality labels | M |
| — | Downloads | Phase 13: a new build after every passed milestone | every milestone | Always a working download | continuous |

**Section 12** (speed tricks from other engines) slots into M0, M2, M4, M5, M8–M10 and M12, as its step table (12.5) says.

**Windows first, per the owner (changed 14 Oct 2026; it was Mac first).** The Minisforum gets the first app, the F0 baselines and the streaming work, so Phase 8's `io_windows.c` and Windows auto-setup move forward into M0 and M5; M8 keeps the AMD back-end work (ROCm or Vulkan per phase, memory split, CPU and NPU use). CI builds the Mac app from M0, and the Mac's own speed work and measurements follow.

### P5. Risk tests to run first (M1)

Each one is cheap and decides whether a big milestone is worth building:

1. **Shadow Bits go/no-go, and which version (decides M7).**
    - BitNest (arXiv 2610.02800) published the same nesting idea: a low-precision draft nested inside a higher-precision model, built base-first (the brief mentions a 4-bit draft inside an 8-bit model; check the paper's exact bit widths). Compare four versions on a real dense model in PyTorch (3–4B on the Mac, or 8B on the Minisforum):
        - **SB4 as written:** 2-bit base and 2-bit correction, the base picked by Appendix A's grid search (alpha 0.2);
        - **SB4 with BitNest's base:** the same 2+2 layout, but the base optimised and the correction refined the way the paper does it (both versions build the base first; they differ in how);
        - **4+4:** 4-bit base in RAM, 4-bit correction on the SSD, 8-bit full model. Test this only where the 4-bit base fits in memory;
        - **No nesting, the real rival:** ordinary 4-bit weights streamed from the SSD plus a small draft model from the same family (Phase 6). For a dense 32B on the Mac it reads about the same SSD bytes per check, and its draft runs much faster than a 2-bit base.
    - For each, on 200 prompts: the per-token acceptance at k = 4 (accepted ÷ (accepted + rejections), as Appendix B measures it); the full model's KL divergence from the original 16-bit model, compared with plain 4-bit's (8-bit's for 4+4); and the planner's predicted tokens per second.
    - **Keep** whichever gives the most tokens per second at equal or better quality. If the no-nesting version wins, skip M7.
    - **Go:** the kept nested version has acceptance at least 0.70, and its KL divergence is at most 3% higher than its plain counterpart's.
2. **Routing traces (calibrates M3, decides M6).**
    - Log which experts each word uses in each layer, for Nemotron 3 Nano and one bigger MoE, on real chats and an agent session.
    - From the logs, compute:
        - hit rate against cache size;
        - how predictable the next layer's experts are;
        - how many new experts each extra guessed word adds (union growth);
        - how many experts a prompt of a given length needs.
    - Replace my simulation's guesses (Zipf popularity) with these numbers.
3. **Overlap test (decides 10.5 per machine).**
    - Run a 4-bit matrix-vector loop on the graphics chip at full speed while reading the SSD at full speed with direct reads.
    - Measure how much each slows the other on the Mac, the Minisforum and, later, the Spark.
    - On Macs this was slower in Flash-MoE, so don't assume.
4. **Why Nemotron re-reads or reads prompts slowly on AMD (feeds M2).**
    - **Check first that llama.cpp isn't re-reading the whole chat every message.** Nemotron mixes Mamba and attention layers, so llama.cpp can only reuse a chat from saved checkpoints. Run `llama-server` with `-lv 4` and look for "forcing full prompt re-processing due to lack of cache data" and "erasing context checkpoint too close to an earlier one".
        - `--checkpoint-min-step` defaults to 8,192 tokens, so a short turn's checkpoint can be thrown away. Also check `--ctx-checkpoints` (default 32) and `--cache-ram` (default 8,192 MiB) *(verify the flags on the current build)*.
        - If lowering `--checkpoint-min-step` fixes the owner's 8-second wait, ship that setting in M2 straight away, before building 4b item 9.
    - Profile its prompt reading on Vulkan and ROCm (about 112 tok/s against 1,142 for a similar Qwen model).
    - Check llama.cpp's issue tracker.
    - If it's a missing kernel, it may be the owner's single biggest early win.
5. **ROCm on Windows for gfx1151 (decides 11.4 on the Minisforum).** Can llama.cpp's HIP back-end run on Windows on Strix Halo? If not, plan Vulkan-only, or ask the owner about dual-booting Linux.

### P6. Parallel work (if the owner runs several agent sessions)

After M0 freezes the P3 interfaces, these can run as separate branches at the same time (git worktrees if the sessions share one machine). Each has its own tests against fakes and opens its own pull requests:
- KV store (4b);
- the I/O back-ends per system;
- the guesser interface and controller;
- the repacker and SB4 converter;
- the app interface and dashboard;
- the M1 risk tests.

Each pull request merges only when CI is green. A coordinating session owns `docs/DECISIONS.md`, `docs/STATUS.md` and the milestone order.

### P7. Definition of done (every change)

- **Settings:** registered with a default, off until its gate passes, and visible to the auto-tuner.
- **Tests:** unit tests plus the tiny-model exactness test pass on the CPU; GPU paths meet section 6's near-tie rule.
- **Speed:** a bench JSON before and after on at least one real machine, with a row in `docs/BENCHMARKS.md`.
- **Records:**
    - the gate result goes in the commit message;
    - a failure goes in `docs/NEGATIVE_RESULTS.md`;
    - any choice made goes in `docs/DECISIONS.md`.
- **Build:** CI green on Linux and Windows for every pull request, and the Mac nightly green before a release tag; no new compiler warnings.
- **GitHub:** committed to the working branch with the gate result in the message, with `docs/STATUS.md` updated (no pull requests, per the owner).
- **Explanation:** a plain-English line for the owner's milestone report.

### P8. Working on GitHub

All code lives in one GitHub repo. GitHub is also where it is built, tested, tracked and downloaded from. The owner can follow everything from the GitHub app on his phone.

#### Set it up (first thing in M0)

1. **The owner will link a GitHub repo for you.**
    - Work only in that repo. Don't create, rename or move repos.
    - Never store the access token in a file.
    - If no repo is linked yet, ask for one and do the local parts of M0 meanwhile.
2. **Read what's already there first** (ground rule 7). If the repo is empty, start the layout from section 4.
3. **Don't use GitHub's "Fork" button for llama.cpp.**
    - A fork of a public repo is always public and can't be made private.
    - Instead, copy llama.cpp into `third_party/llama.cpp` with `git subtree add --squash`, so only the code comes in, not its history.
    - Record the upstream commit in `docs/UPSTREAM_PATCHES.md`.
4. **Put this spec in the repo** as `docs/KERA_BUILD_SPEC.md`.
    - Changes to it are committed directly (no pull requests, per the owner).
    - The owner is told in plain words what changed.
5. **Add `AGENTS.md`** at the top of the repo, at most one page:
    - read the spec;
    - the ground rules (section 0);
    - the build and test commands;
    - "start every session by reading `docs/STATUS.md`".

    Also add **`CLAUDE.md`** containing the line `@AGENTS.md`, so Claude Code loads the same file automatically.
6. **Add `docs/STATUS.md`.** It holds the current milestone, what's done, what's in progress, and what is waiting for the owner.
    - Update it in every pull request.
    - A new agent session has no memory of the last one, so this file is how it picks up where the last one stopped.
7. **Add `.gitignore`** for model and cache files: `*.gguf`, `*.safetensors`, `*.bin`, `models/`, `cache/`, `kvstore/`.
    - Tiny test models are generated by a script during the tests.
    - Any test model committed to the repo must be under 5 MB.
    - Don't use Git LFS.
8. **Logo kit:** the owner gives you `kera-brand.zip` with this spec. Unzip it into `assets/brand/` and follow its `BRAND.md`.
    - Ask the owner to upload `assets/brand/social/social-preview-1280x640.png` under the repo's Settings, Social preview. GitHub has no way for you to set it *(verify)*.
    - Never redraw or retype the logo; always use the files.
9. **Labels:** `needs-owner`, `gate-pass`, `gate-fail`, `risk-test`, `mac`, `amd`, `nvidia`, `speed`, `bug`.
10. **Milestones:** one GitHub milestone per row of P4 (M0–M12).
11. **Issues:** one issue per step inside a milestone and one per risk test (P5).

#### Day-to-day work

- **No pull requests (the owner's choice, 14 Oct 2026).** Work on the working branch and push small commits, each with the P7 checklist done and its gate result in the message.
    - Merge a milestone into `main` once its checks are green and its gate passed.
    - Anything in ground rule 3 that the owner hasn't granted, licence changes, and anything that changes output or default settings waits for the owner: list it in `docs/STATUS.md` under "Waiting for the owner".
- **Questions for the owner** go in a new issue labelled `needs-owner`, written in plain words, with a suggested answer. Carry on with other work while you wait. Don't sit idle.
- **Commit messages:** one short summary line, then the gate result, for example:

  `Gate F2: PASS, Qwen3-32B Q4 on Minisforum 2.3x stock llama.cpp (14.0 vs 6.1 tok/s)` (an example of the format, not a real result)

#### Automatic builds and tests (`.github/workflows/`)

**The free allowance:** if the linked repo is private, GitHub Free gives it 2,000 minutes a month on GitHub's own machines. Public repos get GitHub's standard machines free.
- Mac minutes cost about 10 times as much as Linux ones, so they use the allowance up fast *(verify on GitHub's billing page)*.
- Runners on the owner's own machines are free *(verify; GitHub has announced and changed charges here before)*.
- Without a payment method, GitHub stops jobs at the limit instead of charging. Never add one; that's the owner's call.

| Workflow | When | Where it runs | What it does |
|---|---|---|---|
| `ci.yml` | Every push and pull request | GitHub's Linux machines | CPU build, unit tests, tiny-model exactness tests (P2) |
| `ci.yml` | Every push | GitHub's Windows machines (the Minisforum only runs finished packages, per the owner) | Windows Vulkan build and CPU tests |
| `mac.yml` | Nightly, on demand, and before a release tag | GitHub's Mac machines (a `mac` runner if the owner adds one) | Metal build and tiny-model tests. Must be green before a release tag |
| `hardware.yml` | On demand | Every `mac`, `amd` and `nvidia` runner that is online | Section 7 benchmark on real hardware. Posts the before/after table as the job summary and keeps the bench JSON |
| `nightly.yml` | 03:00 UK time | Every `mac`, `amd` and `nvidia` runner; GitHub's Linux machines for the CUDA build | Full bench and the 3% slowdown check; a CUDA build-only check for one or two GPU generations. Opens a `bug` issue if something got slower or broke |
| `release.yml` | When a version tag (`v0.N.x`) is pushed after a passed milestone | GitHub's Mac, Windows and Linux machines | Clean builds of the downloads. Attaches them to a GitHub Release, with the section 9 report as the release notes |
| `upstream.yml` | Monthly | GitHub's Linux machines | Pulls new llama.cpp into the branch `upstream-llama-cpp` with `git subtree pull --squash`; CI on that branch decides (no pull request) |

**Keeping the minutes down:**
- Cache compiled files between runs (ccache through `actions/cache`).
- Cancel a run when a newer push to the same branch replaces it.
- Keep build files for 3 days only, since free storage is 500 MB. Release downloads don't count toward this.
- Check the minutes used each week. If they are heading over the allowance, move more jobs to the owner's runners and tell him.

#### Runners on the owner's machines

GitHub's runner app lets a pull request run on the owner's real hardware. That is the only way to test Metal and AMD speed, especially if the agent itself works in the cloud. It pushes code, the runner tests it on his machine, and the agent reads the results with `gh run view --log`.

- **Labels: just `mac`, `amd` and `nvidia`.**
    - Workflows ask for a kind of machine, never a particular one, for example `runs-on: [self-hosted, amd]`. Any machine with that label can take the job: the owner's MacBook and Minisforum today, and a Spark, a second Mac or a friend's PC later, with no workflow changes.
    - GitHub adds the operating system and chip labels itself (`macOS`, `Windows`, `Linux`, `ARM64`, `X64`).
    - Bench results record the machine profile (P3), so numbers from different machines are never mixed up.
- **The Minisforum (labelled `amd`):**
    - It is always on, so it can take every pull request's AMD build.
    - Benchmarks run only at night or on demand, so they don't slow down his own models.
    - Install the runner as a Windows service.
- **The MacBook (labelled `mac`):**
    - It is a 16 GB laptop he uses every day, so it runs only nightly jobs and jobs he starts.
    - Its benches run only on mains power, with no other heavy app open, as section 7 requires.
    - A job waits while the Mac is asleep. GitHub cancels it after about a day *(verify)*.
- **Safety:**
    - Only connect runners if the linked repo is **private**. On a public repo, a stranger's pull request could run code on the owner's computers.
    - If the repo is public, or ever becomes public, don't install the runners (or remove them first), and tell the owner.

#### Downloads

- Each passed milestone gets a version tag, and `release.yml` attaches the Mac, Windows and Linux downloads to a GitHub Release.
- The owner downloads from the repo's Releases page while signed in.
- **Before anyone else gets it** (friends, testers, buyers), ask the owner where public downloads should live. One option is a separate public repo that holds only the release files. The code now lives in the public GitHub fork (18 Nov 2026).
- **Mac signing:** when the owner has an Apple Developer account, add the certificate and notarisation password as GitHub encrypted secrets, used only by `release.yml`.

---

## 1. What Kera is for, and what it isn't

**Goal:** the fastest local, single-user inference for dense and MoE models, both those that fit in memory (section 11) and those bigger than memory (sections 4–10). Targets: machines where the CPU and GPU share memory (Apple Silicon, AMD Strix Halo, NVIDIA GB10/DGX Spark), plus a discrete-NVIDIA version.

**It must work on any machine of each kind, not just the owner's:**
- **Mac:** any Apple Silicon Mac (M1 or later, any memory size).
- **AMD:** any Ryzen AI Max machine, plus Radeon graphics cards that llama.cpp's Vulkan or ROCm back-ends support.
- **NVIDIA:** any GB10 machine (DGX Spark and the Dell, HP and ASUS versions), plus RTX graphics cards that llama.cpp's CUDA back-end supports.

The owner's machines are the test machines, not the only targets. Auto-setup (4a) measures whatever it runs on and tunes for it, so nothing is hard-coded to one model of computer. Where a feature only pays on some machines (for example the overlap schedule), the tuner switches it per machine. Exact means the same answers as the plain quantized model. After that: fine-tuning that produces adapters Kera can load.

**Not goals:**
- Training models from scratch.
- Clusters of many machines. The exceptions are linking 2–4 DGX Sparks (section 10.7) and one Mac + Minisforum experiment (Phase 12); both are optional.
- Beating physics. No engine reads weights faster than the hardware moves bytes (section 2).
- Serving many users at once on datacenter GPUs. vLLM, SGLang and TensorRT-LLM own that, and it's a different problem.
- Claiming "fastest" without proof. Every claim needs the section 7 benchmark against the best other engine on the same machine and model file.

**What makes Kera different.** Almost every piece below was published or built by someone else first. Kera's difference is putting them together in one self-tuning app built on llama.cpp:
- full GPU speed on Mac, AMD Strix Halo and the DGX Spark, from llama.cpp's kernels;
- llama.cpp's wide model support, including Nemotron and Kimi K2.6;
- fast speeds for models that fit in memory (section 11);
- fine-tuning (Phase 10).

The pieces, and the earlier work behind each:
1. **Streaming experts from the SSD with a learned cache, next-layer prefetch, several drives and KV prefix reuse:** Colibri. Kera ports this code (section 3).
2. **Saving chat memory (KV) to the SSD and reloading it instead of re-reading:** LMCache (section 4b, Phase 5b). This matters more when streaming than anywhere else. Re-reading a prompt makes a MoE model pull almost every expert off the SSD, so reloading saved memory instead of re-reading the earlier chat can take about a second where re-reading takes a minute or more. The new message itself still has to be read.
3. **A guess-ahead controller that measures whether guessing pays, and switches it off when it doesn't:** Cascade (arXiv 2506.20675), with the Appendix B formula only as the starting guess. Guessing can make MoE models slower.
4. **Guessing while waiting for the SSD, and using the guesses to start the next reads early:** MoE-SpeQ (arXiv 2511.14102), section 10.5.
5. **Shadow Bits:** BitNest-style nesting (arXiv 2610.02800) applied to SSD streaming. A low-bit draft is nested inside the full model; the base stays in RAM and drafts, and the correction streams from the SSD only to check (Appendix A).
6. **Using a cheap copy of the streamed model as its own guesser:** SubSpec (arXiv 2509.18344).
7. **SSD reads that overlap the maths, with the CPU as traffic controller** (section 10), building on prefetch work such as ProMoE, Fate and SP-MoE (Appendix E). Doing both at once instead of one after the other is worth about 1.4–1.8× where reads and maths don't slow each other down (the planner's GLM example gives 1.44×). On Macs it was slower in Flash-MoE, so it's measured per machine (P5 item 3). A planner, building on Colibri's `coli plan`, predicts the speed ceiling for any model on any machine, and the benchmark reports how close Kera gets.
8. **Models that fit in memory, beating llama.cpp, MLX and vLLM** (section 11): every kind of guess-ahead behind one controller with fast checking kernels, plus fused whole-word kernels. Expected 2–3x over writing without guessing for dense models that fit, and 1.2–1.6x for small-active MoE over stock llama.cpp. llama.cpp now has its own guessing (MTP), so the real bar is beating each engine's best setup, including its own guessing.

---

## 2. The speed rule and the target machines

```
seconds per token  =  GB read from RAM / RAM speed  +  GB read from SSD / SSD speed
```

Everything in this spec either reads fewer bytes, reads them faster, or makes one read serve several tokens. Appendix C has the planner code.

| Machine | Usable for model | RAM speed | SSD speed | Notes |
|---|---|---|---|---|
| MacBook Pro M4, 16GB | about 10 GB | 120 GB/s on paper | about 2 GB/s (512GB model), 2.8–3.3 GB/s (1TB) *(the app measures it)* | Metal GPU. macOS limits how much RAM the GPU may wire *(verify the `iogpu.wired_limit_mb` setting)* |
| Minisforum MS-S1 MAX, Ryzen AI Max+ 395, 64GB, 2TB | about 50 GB | 256 GB/s on paper, about 210 measured by others | included Kingston 2TB measured 3.0 GB/s in a simple test, roughly 4–5 GB/s with parallel reads *(the app measures it)* | Windows. M.2 slot 1 is PCIe 4.0 x4, slot 2 only x1; the x16 slot is wired as x4 (see Phase 8). The CPU/GPU memory split is set in BIOS or AMD software *(verify)* |
| NVIDIA DGX Spark Founders Edition (GB10, 128GB shared) | about 110 GB (headless) | 273 GB/s on paper; the graphics chip reaches about 231 | about 11 GB/s with 16 MiB parallel reads (Gen5 x4); 4.2–6.6 GB/s in simple one-at-a-time tests | Section 10 is its full plan. Other brands' GB10 boxes measured about 5–5.5 GB/s SSD |
| NVIDIA, rented by the hour | depends | depends | depends | Discrete GPU = VRAM + RAM + SSD tiers |

These are starting estimates for planning the work. The shipped app never relies on them: it measures each machine itself on first launch (section 4a).

Expected ceilings (estimates from the planner; Phase 1 checks them against reality):

| Model | Mac 16GB | Minisforum 64GB |
|---|---|---|
| MoE 35B total, 3B per token | 5–7.6 tok/s | fits in RAM |
| MoE 120B total, 5B per token | 1.5–2.3 tok/s | about 22 tok/s |
| GLM-5.2-size MoE, 744B | will not fit | about 0.8–0.9 tok/s |
| Dense 32B | 0.24–0.38 tok/s plain, about 0.7–1.0 with Shadow Bits if P5 risk test 1 reaches about 75% acceptance | about 12 tok/s (fits) |
| Dense 70B | under 0.12 tok/s, not worth it | about 5 tok/s (fits) |

---

### 2a. Target models and what each one needs

Support these first; the bench uses them. The first job for each model is to check whether the pinned llama.cpp fork already supports its design. If it doesn't, port the design from the official reference code and test it against that code on a tiny random copy of the model (the way Colibri checks every engine in CI).

| Model | Download | Design | Runs where | Engine must support |
|---|---|---|---|---|
| Nemotron 3 Nano 30B-A3B | 63 GB BF16; about 18 GB at 4-bit | 52 layers: 23 Mamba-2, 23 MoE (128 experts, 6 used + 1 shared), 6 attention. No built-in guesser. Thinking on by default | In memory on the Minisforum and Spark; streamed on the 16 GB Mac (4-bit doesn't fit in about 10 GB) | Mamba-2 state, with snapshots for reuse (4b item 9); fast Mamba prompt reading on AMD (11.4); thinking switch (4c) |
| Nemotron 3.5 Lightning 30B-A3B | 66 GB BF16; NVFP4 and GGUF versions exist | Same layout as Nano, plus 1 built-in multi-word guesser (MTP) | Same as Nano. BF16 doesn't fit in 64 GB | Same as Nano, plus MTP guessing |
| GLM-5.3 (744B, about 40B used per word) | about 419 GB (Colibri int4 g64 file) | 78 layers (3 dense); MLA attention plus sparse attention with a top-2048 "indexer"; 256 experts, 8 used + 1 shared; model card lists an MTP head (Colibri's file leaves it out) | Streamed everywhere | MLA, the sparse-attention indexer, expert streaming |
| Kimi K2.6 (1T, 32B used per word) | about 595 GB | DeepSeek-V3-style: 61 layers (1 dense), MLA, 384 experts, 8 used + 1 shared, no MTP. Experts are native INT4 (group 32); attention, shared expert, dense layer and output layer are BF16. Optional MoonViT image encoder | Streamed everywhere; needs the 4TB Spark or a big second drive | INT4 group-32 conversion (compressed-tensors format), MLA, streaming; images optional |
| Nemotron 3 Ultra 550B-A55B | about 352 GB NVFP4 | 108 layers: 48 Mamba-2, 48 MoE, 12 attention. LatentMoE: 512 experts, 22 used, working in a 2048-wide latent space; FP8 shared expert; BF16 projections and attention; 1 MTP layer (BF16) | Streamed everywhere; worst fit on 64 GB (39 GB always-used) | Mamba-2, LatentMoE, NVFP4 weights (unpacked to a supported format on Metal and Vulkan), MTP. Nothing streams it today |

## 3. Foundation: build on an existing engine

**Default choice: build on llama.cpp (MIT licence)** as the compute core. Copy it into the repo with `git subtree` rather than a GitHub fork (P8 explains why). It already has:
- kernels for Metal, CUDA, HIP (AMD) and Vulkan;
- the GGUF format and quantization types;
- coverage of nearly every dense and MoE model;
- speculative decoding, LoRA adapter loading, and an OpenAI-compatible server.

Kera adds its own layers on top: the tiered weight store, the cache and prefetcher, the guess-ahead controller, the SB4 format and kernels, the planner, and the benchmark.

**Second code source: Colibri** (Apache-2.0, https://github.com/JustVugg/colibri). The base stays llama.cpp (MIT). Colibri is not just a reference: port these parts of its code into Kera, one at a time:
- the expert store and cache (`c/expert_store.h`);
- learned pins from routing history (`c/route_trace.h` and the `.coli_usage` log);
- next-layer router lookahead prefetch;
- multi-disk mirror reads (`COLI_MODEL_MIRROR`, `docs/multidisk.md`);
- KV prefix reuse (`c/kv_prefix.h`), KV kept across restarts (Colibri does this for GLM-5.2 only *(verify)*), and several KV slots (`--kv-slots`);
- the one-step setup flow (`c/setup_*.py`) and the plan and tune logic (`c/resource_plan.py`, `coli plan`, `coli tune`).

Where a phase below describes building one of these parts, start from the ported Colibri code. Then:
- that phase's gate compares Kera with the part switched on against switched off;
- the comparison with Colibri covers the whole engine (gates 2 and 8, the 10.2 final gate and section 7).

Colibri's setup and plan code is Python; port its logic into Kera's C/C++ engine, since Python is only for tools (P2). In the Phase 0 decision, choose which file layout streamed experts use: Colibri's own format, or the repacked GGUF from Phase 2.

**Rules for ported code:**
- Keep Colibri's licence and NOTICE, keep its copyright lines, and mark every file you changed with a prominent note saying so, as Apache-2.0 requires. List each ported file in `docs/UPSTREAM_PATCHES.md`.
- Never use the Colibri name or logo for Kera: not in the app, the website or any marketing. Apache-2.0 gives no trademark rights. Crediting it in `NOTICE` and the README is fine.
- Each ported part goes behind a setting, passes the section 6 exactness test, and stays off until it is at least as fast as Colibri itself on the same machine and model.

**Phase 0 decision:** spend at most one day reading both codebases, then write a one-page `docs/DECISIONS.md` entry:
- the fork point;
- where the streaming store hooks into weight loading (llama.cpp loads via mmap, and MoE expert tensors can be routed with `--override-tensor`);
- the risks.

If llama.cpp's graph makes per-expert streaming too awkward, the fallback is Colibri's engine plus ggml kernels. Say which you chose and why.

**Licences:** the owner may sell Kera cheaply. MIT and Apache-2.0 both allow that, but:
- keep every upstream copyright notice;
- ship a `NOTICE` file;
- state Apache-2.0 changes as that licence requires;
- never bundle model weights, since each model has its own licence.

---

## 4. Architecture

```
            ┌───────────── kera CLI / OpenAI-compatible server ─────────────┐
            │  plan · bench · convert · run · serve · finetune              │
            └───────────────────────────────┬───────────────────────────────┘
                                            │
   ┌──────────────────┐   ┌─────────────────▼──────────────────┐   ┌──────────────────┐
   │ Planner          │   │ Decode loop (llama.cpp graph)      │   │ Guess-ahead      │
   │ ceilings,        │──▶│ attention / dense layers / experts │◀──│ controller       │
   │ placement plan   │   │ draft sources: SB4 base, MTP head, │   │ k = 0..8 per step│
   └──────────────────┘   │ small draft model                  │   └──────────────────┘
                          └─────────────────┬──────────────────┘
                                            │ "need expert E of layer L" / "need tensor T"
                          ┌─────────────────▼──────────────────┐        ┌─────────────────────────────┐
                          │ Tiered weight store                │        │ KV cache manager + KV store │
                          │ VRAM (discrete GPU) · RAM · SSD(s) │        │ paged KV · prefix reuse ·   │
                          │ cache policy · prefetcher · I/O    │        │ RAM → SSD → network tiers   │
                          └──┬──────────────┬───────────────┬──┘        └─────────────────────────────┘
                    io_macos.c        io_windows.c      io_linux.c       (section 4b; shares the RAM
                                                                          budget with the expert cache)
```

**Components:**

- **Planner** (`src/planner`): reads the model's tensor sizes and the machine's measured speeds. Decides what goes where and prints the ceiling. After a run, prints measured ÷ ceiling.
- **Repacker** (`tools/repack`): rewrites a GGUF so each expert's weight matrices sit in one contiguous block, aligned to 16 KiB. Writes an index of (layer, expert) → (file, offset, bytes). Supports a mirror copy on a second drive.
- **Tiered weight store** (`src/store`):
    - always-used weights (embeddings, attention, router, shared experts) pinned in RAM, or VRAM on discrete GPUs;
    - experts in a cache with a learned policy, falling back to the SSD.
    - One read per expert, served by a reader thread pool.
- **Cache policy:** start with the OS file cache (on Mac it beat a hand-written cache in Flash-MoE). Then try LRU plus "learned pins" from a usage log that persists across runs. Rank by expected bytes saved per GB of RAM.
- **Prefetcher:** run the next layer's router on the current hidden state to guess its experts, and start those reads early. Feed it draft routes from the guess-ahead controller too.
- **Guess-ahead controller** (`src/spec`): Appendix B. Tracks the acceptance rate, predicts how many new bytes each extra guess costs, and picks k (0 = off).
- **SB4** (`src/sb4`): format, converter, and kernels for CPU, Metal, CUDA and HIP/Vulkan (Appendix A). Full mode merges the two planes into an ordinary 4-bit layout, so it can reuse existing 4-bit kernels.
- **I/O back-ends:**
    - macOS: `pread` thread pool; compare `F_NOCACHE` on/off.
    - Linux: `O_DIRECT` with `io_uring` optional.
    - Windows: unbuffered overlapped `ReadFile`, or the IoRing API *(verify availability)*.
- **Auto-setup and auto-tuner** (`src/setup`, `src/tune`): section 4a. Measures the machine, recommends models, and picks the fastest settings by itself.
- **App shell** (`app/`): the downloadable app. Start with a local web page that the app serves and opens in the browser, like Colibri's dashboard. It is quicker to build than a native window and works the same on Mac, Windows and Linux. A native window can come later. The CLI stays for power users and for you.
- **Benchmark** (`tools/bench`, also built into the app): section 7.
- **Fine-tuning** (`tools/finetune`): wraps existing trainers and exports adapters Kera can load (Phase 10).
- **KV reuse and saving** (`src/kv`, `src/kvstore`): section 4b. Its only job is faster chat replies.

### 4a. Auto-setup: the app measures, chooses and tunes itself

This runs on first launch, from a "Re-check this machine" button, or via `kera setup`. The user does nothing but wait about a minute.

1. **Detect:**
    - chip and core counts;
    - RAM size;
    - graphics chip, and how much memory it is allowed to use;
    - OS version;
    - every drive and its free space.
2. **Measure, without admin rights:**
    - **SSD speed, on every drive:**
        - write a 2–4 GB test file;
        - read it back with parallel reads that bypass the file cache (`F_NOCACHE` on macOS, `O_DIRECT` on Linux, `FILE_FLAG_NO_BUFFERING` on Windows);
        - try several thread counts and read sizes, and keep the best combination;
        - delete the test file.
        - *(Verify that each flag really bypasses the cache on each OS. On macOS, write the test file with `F_NOCACHE` too, or the read may come from memory.)*
    - **RAM speed:** a large memory-copy test, from the CPU and from the graphics chip.
    - **Graphics chip:** a short 4-bit matrix-times-vector test.
3. **Explain** the results in plain words. For example: "Your SSD reads 2.9 GB/s. A 120B MoE model will run at about 2 tokens a second (about 90 words a minute)."
4. **Save** a machine profile (JSON) that the planner reads.
5. **Choose:** an in-app model list. For each model it shows:
    - whether it fits;
    - how it will run (all in memory, or streamed from the SSD);
    - the predicted speed on this machine.

    **Honest limits:** if a model would technically run but slower than 0.5 tokens a second on this machine (about 20 words a minute), say so in plain words. Kimi K2.6 on a 16 GB Mac is an example. Don't list such models like a normal choice: show them separately, and suggest using the model online or picking a smaller one instead.

    Downloads have progress and resume. Before downloading, the app checks free disk space and shows the model's licence.
6. **Tune each model automatically** the first time it loads. This takes about 2–5 minutes, shows progress, and can be skipped. It runs short real tests of candidate settings and keeps the fastest one that passes the exactness check:
    - reader threads and read size;
    - cache mode (OS file cache or Kera's own);
    - next-layer prefetch on or off;
    - SSD reads overlapping graphics work, on or off;
    - guess-ahead source and limits;
    - Shadow Bits on or off, and its group size;
    - CPU or GPU for each part;
    - how RAM is split between conversation memory (KV cache) and the expert cache;
    - KV store chunk size, disk budget, and save/load threads (section 4b).

    The result is saved per model and per machine.
7. **Re-tune by itself** when:
    - the hardware, OS or Kera version changes;
    - measured speed drifts more than 15% below the tuned speed.
8. **Keep learning** from use: the usage log updates which experts stay in memory (Phase 4).

This isn't a new idea: Colibri has `coli setup` and `coli tune`, so read those first. It is still required. **Every setting added in any phase must be added to the auto-tuner with a safe default**, so the user never has to set it by hand.

### 4b. Faster chat replies: KV cache and LMCache-style saving

**The only goal of this section is faster chat replies.** Every feature here must make one of two numbers better, and the bench measures both:
- **Wait before the reply starts** (time to first word).
- **Writing speed** once it starts (tokens per second).

Anything that improves neither is out of scope.

**What these are, in plain words:**
- The **KV cache** is the model's memory of the chat so far. For every token it has read, each layer keeps a small record, so it doesn't have to re-read the whole chat for every new word. Building it ("reading the prompt") is the wait before the first word.
- **LMCache** (https://github.com/LMCache/LMCache) saves those records (RAM, disk) and reloads them instead of re-reading the text.

**What it speeds up, and what it doesn't** (tell the owner this plainly):
- **Big win: the wait before the reply starts**, for follow-up messages in the same chat, going back to an old chat, and coding agents that resend the whole history every turn.
    - On a streamed MoE model, reading new text pulls in most experts: 200 new tokens already need about 99.8% of the 256 experts per layer with even routing (1 − (1 − 8/256)^200), somewhat fewer with real routing.
    - For GLM-5.3 on a 128 GB machine, re-reading a long chat means about 300 GB off the SSD, before any maths. That's roughly 28 seconds on a Founders Edition Spark's ~11 GB/s SSD, and about 45 seconds at 6.6 GB/s.
    - Reloading saved KV for the same text takes well under a second per 10,000 tokens.
- **No help for the first message of a brand-new chat**, unless its start (a system prompt, an agent's tool list) was saved before (item 6).
- **No direct help with writing speed.** Very long chats can make writing slightly *slower*, because their KV takes RAM the expert cache would use. Item 4 balances that.

**KV size** (my arithmetic, from the model configs):

| Model type | KV per token (16-bit) | 32k tokens | 200k tokens |
|---|---|---|---|
| GLM-5.x / DeepSeek-style (MLA + sparse attention): 78 layers × (576 latent + 128 indexer values) | about 100 KB | about 3.2 GB | about 20 GB |
| Llama-3.1-70B-style (no MLA): 80 layers × 8 KV heads × 128 dims × K and V | about 330 KB | about 10.5 GB | about 66 GB |

#### Features, in order of payoff for chat speed

Build them in this order. All are exact (same answers) unless marked lossy.

1. **Reuse what was already read** (`src/kv`).
    - A radix tree of token pages finds the longest already-read start of each new prompt, and only the new part is read (vLLM automatic prefix caching, SGLang RadixAttention).
    - Shared starts (system prompt, tool definitions) are stored once and shared by every chat that uses them.
    - Store KV in fixed-size pages, so many chats share RAM without gaps; the auto-tuner picks the page size.
2. **Recognise what chat apps and agents resend.** OpenAI-style clients resend the whole history every turn, so match by tokens and reuse it.
    - Chat templates must render earlier turns byte for byte the same.
    - Clients that drop reasoning ("thinking") text break the match; Colibri added a `preserve_thinking` option for this (its docs mention issue #1759).
    - The bench reports the KV hit rate so a broken match is caught.
3. **Read new text layer by layer on streamed MoE models.** Run all the new text through layer 1, then layer 2, and so on, so each expert comes off the SSD at most once per message, not once per chunk (the MoE-Gen idea, arXiv 2503.09716). Split into chunks only when working memory runs out.
4. **One RAM budget for KV and experts.**
    - The auto-tuner splits RAM between chat memory and the expert cache by measured writing speed.
    - KV of chats you aren't using right now moves to the SSD (item 5) to free RAM for experts.
5. **Save KV to the SSD and reload it** (`src/kvstore`, LMCache-style).
    - **Chunks with fingerprints:**
        - split each chat's tokens into chunks (start at 256 tokens, as LMCache's examples do; the tuner tests 128–1024);
        - the key = a hash of every token up to the end of the chunk, plus a **cache identity**: model file hash, adapter (LoRA) hash, weight mode (SB4 full, SB4 base, Q4…), KV precision, tokenizer and chat-template version, position (RoPE) settings;
        - a chunk is only valid with exactly the same text before it.
    - **Tiers:** look in RAM first, then the SSD, as LMCache does.
    - **Saving and loading:**
        - save in the background so writing never pauses (LMCache's async save);
        - when a known chat is opened, pre-load its KV from SSD into RAM before the user hits send (LMCache's `prefetch()`);
        - start computing layer 1 while later layers' KV is still loading (LMCache's layerwise loading).
    - **Survives restarts and crashes:** a small index on disk, a checksum on every chunk. Colibri already keeps KV across restarts, so read how it does it.
    - **Eviction:** least-recently-used, within a disk budget that auto-setup picks from free space.
6. **"Remember this" (pinning).**
    - Process an agent's system prompt and tool list, or a document you keep asking about, once, and keep its KV pinned so it is never evicted.
    - Then even the first message of a new chat that starts with it skips that reading.
7. **Several chats at once.** Keep several chats' KV ready ("slots"); Colibri keeps up to 16 with `--kv-slots`. Switching between chats then needs no re-reading.
8. **Guess-ahead support** (needed by Phase 6):
    - roll back the KV of rejected guesses;
    - the SB4 draft reads the full model's KV for already-checked tokens (as SubSpec shares its KV cache) and keeps its own KV only for the guessed tokens. If that lowers acceptance by more than 5 points, give it a separate KV.
9. **Model-aware layouts** (needed for the above to work on each model):
    - MLA models (GLM, DeepSeek, Kimi): the 576-value latent per layer, plus sparse-attention indexer keys;
    - grouped-query attention (Llama, Qwen): plain K/V;
    - sliding-window layers (gpt-oss, Gemma): keep only the window;
    - linear-attention layers (Qwen3-Next-style DeltaNet): store their fixed-size state and snapshot it at page boundaries so reuse still works.
10. **Discrete NVIDIA:** KV in VRAM first, then pinned RAM, then the SSD (Phase 9).
11. **Lossy speed options, off by default** (Phase 11):
    - 8-bit KV (leaves more RAM for experts);
    - CacheBlend: reuse a saved file or document even when it appears at a different place in a new prompt, recomputing a small share of its tokens (the paper says roughly 15%; check).

**Kept minimal on purpose:**
- A disk budget setting and a "clear saved chats" button. Saved KV fills the disk otherwise, and it is effectively a copy of the chats.
- When a chat passes the model's context limit, stop and say so.

**Using LMCache itself:**
- LMCache plugs into vLLM, SGLang and NVIDIA Dynamo, not llama.cpp or Colibri *(verify the current list)*.
- Those engines can't stream models bigger than memory from an SSD, so build these features natively in Kera and don't depend on LMCache.


### 4c. Chat-app basics that decide how fast it feels

1. **Keep the model loaded.**
    - Never unload a model just because it's idle (Ollama's default unload after 5 minutes is the classic cause of a slow first reply).
    - Load the last-used model in the background when the app starts, and show progress.
    - At about 4–5 GB/s, loading an 18 GB model takes about 4–6 s; streamed models only need their always-used part, about 12–39 GB.
2. **Thinking switch, per model:**
    - On, off, or a thinking budget (Nemotron: `enable_thinking`; vLLM-style `chat_template_kwargs`).
    - Default to the model card's setting and its recommended sampling settings. Nemotron 3 Nano: temperature 1.0 and top_p 1.0 with thinking on; greedy with thinking off; temperature 0.6 and top_p 0.95 for tool calling.
    - When thinking is on, keep the thinking text in the history the app resends, so the start of the chat still matches and KV reuse works (4b item 2).
3. **Show where the wait went:** loading, reading your message, thinking, writing. Users then know whether to switch thinking off or keep the chat open.
4. **Tool calling for coding agents:**
    - Parse each model family's tool-call format in the OpenAI- and Anthropic-compatible server. NVIDIA documents the `qwen3_coder` parser for Nemotron; GLM and Kimi have their own formats *(verify each)*.
    - Covered by the JSON/tool-call test in section 6.

**Repo layout:**

```
(root of the linked repo)
  README.md  LICENSE  NOTICE  AGENTS.md  CLAUDE.md  .gitignore
  .github/workflows/               (ci, mac, hardware, nightly, release, upstream; P8)
  .github/pull_request_template.md
  third_party/llama.cpp/          (git subtree of upstream llama.cpp + small hooks, pinned commit)
  app/                             (downloadable app shell + local web-page interface)
  assets/brand/                    (logo kit: K-bolt logo, app icons, favicons; read assets/brand/BRAND.md)
  src/setup/  src/tune/            (auto-setup and auto-tuner, section 4a)
  src/planner/  src/store/  src/cache/  src/prefetch/  src/spec/  src/sb4/
  src/kv/  src/kvstore/            (KV reuse and saving for faster chat replies, section 4b)
  src/io/io_macos.c  src/io/io_windows.c  src/io/io_linux.c
  tools/repack/  tools/convert_sb4/  tools/bench/  tools/finetune/
  reference/shadow_bits.py  reference/controller.py  reference/planner.py   (Appendices A–C, verbatim)
  bench/prompts/  bench/results/
  docs/KERA_BUILD_SPEC.md  docs/STATUS.md  docs/UPSTREAM_PATCHES.md
  docs/BENCHMARKS.md  docs/DECISIONS.md  docs/NEGATIVE_RESULTS.md
```

---

## 5. Phases and gates

Each phase ends with a gate. **Pass:** commit and move on. **Fail:** log it in `docs/NEGATIVE_RESULTS.md`, tell the owner in plain words, and either fix it or drop the feature, as the phase says. Speeds in gates are measured with the protocol in section 7.

### Phase 0: a downloadable app that measures itself (Windows first)

1. Copy llama.cpp into the repo (git subtree, P8), build with Metal, and record the commit.
2. Read Colibri as described in section 3, then write the foundation decision.
3. **App shell:** a downloadable Windows build (a zip, with `kera.ico`) first, and a Mac build (a `.dmg` or zip), each holding the engine, the CLI and the local web-page interface (section 4).
    - Use the logo kit (`assets/brand/`): `kera.icns` as the app icon, the favicon files for the web page, and the K-bolt in the page header.
4. **Signing:** macOS warns about or blocks unsigned apps downloaded from the internet. Signing and notarising needs a paid Apple Developer account, so ask the owner. Until then, ship it unsigned with one plain line on how to open it anyway *(verify current macOS behaviour)*.
5. **Auto-setup steps 1–4** from section 4a: detect, measure, explain, save the profile.
6. **Cross-check the built-in measurements** once by hand (for example `fio` or `dd`), so you know the app's numbers are right.
7. **Pick bench models.** The section 2a target models come first. Ask before each download, and use current equivalents if these are outdated:
    - dense ~8B at 4-bit (fits in RAM);
    - MoE 30–35B with about 3B used per token (Qwen3-30B-A3B class, about 18–20 GB);
    - dense 32B at 4-bit (about 19 GB);
    - later: gpt-oss-120b class (about 61–65 GB).
8. Record stock llama.cpp speeds for each with the app's built-in bench (section 7).

**Gate 0:**
- The owner downloads the app on his Minisforum (the Mac follows) and opens it. With nothing else from him, it shows his SSD, RAM and graphics speeds in plain words.
- Those numbers are within 10% of your manual cross-check.
- `docs/BENCHMARKS.md` has the machine profile and baseline speeds, and `docs/DECISIONS.md` has the foundation choice.

### Phase 1: planner

1. Port Appendix C.
2. Read real tensor sizes from GGUF headers.
3. Read the machine profile that auto-setup saved.
4. `kera plan --model X` prints what goes in RAM, VRAM and SSD, the GB read per token, and the ceiling.
5. The in-app model list (section 4a step 5) shows the same prediction in plain words for every model.

**Gate 1:** for each baseline model, measured speed is below the ceiling. For models that fit in RAM, the ceiling is within 30% of measured. If not, find out why before continuing; a wrong planner misleads every later phase.

### Phase 2: expert streaming store (MoE), Minisforum first

1. **Repacker:** write each expert as one contiguous block aligned to 16 KiB, plus an index of (layer, expert) → (file, offset, bytes).
2. **Store:**
    - always-used weights stay resident;
    - routed experts are read with one `pread` each, by a thread pool, into slot buffers;
    - the MoE matmul reads from those buffers instead of the memory-mapped tensor.
3. **Cache, two versions; the auto-tuner picks the faster one on each machine:**
    - v1: plain `pread` and let macOS's file cache do the caching;
    - v2: Kera's own LRU with `F_NOCACHE` on the reads.
4. **Report per token:** GB read from SSD, cache hit rate, time waiting on I/O.

**Gate 2:**
- **(a) Exactness:** section 6 passes against the same GGUF run in RAM by stock llama.cpp. On the Mac, use a model that fits; on the Minisforum, the big ones.
- **(b) Not slower:** the 30–35B MoE is at least as fast as stock llama.cpp with mmap.
- **(c) Close to the ceiling:** the 120B-class MoE reaches at least 60% of the planner ceiling on the Mac.
- **(d) Against Colibri:** Kera is at least as fast as Colibri on every model both can run (section 7). If it isn't, profile it, find out why, and log the result in `docs/NEGATIVE_RESULTS.md`.

### Phase 3: read the next layer early

1. Run layer L+1's router on layer L's output to guess its experts, and start those reads.
2. Log how often the guess was right; others report 79–97%.
3. Test overlapped against serial reads. On a Mac, reading during GPU work was slower in Flash-MoE, because both compete for the same memory. Also try issuing reads only during CPU-side work.

**Gate 3:** keep it only if the 120B MoE gets at least 10% faster. Otherwise log a negative result and drop it.

### Phase 4: learned cache pins

1. Keep a per-model usage log that persists across runs.
2. Pin the most-used experts.
3. During prompt processing, pin the experts the prompt uses.
4. **Test on held-out prompts:** learn pins on one prompt set, measure on a different one. Colibri warns that pins can over-fit.

**Gate 4:** hit rate at least 10 points better than plain LRU on held-out prompts, or drop it.

### Phase 5: second SSD

Mirror or stripe the expert file across the internal SSD and an external Thunderbolt NVMe. The reader sends each read to the less busy drive. Ask the owner before he buys an enclosure or drive. Colibri measured +37.5% from two drives on independent controllers.

**Gate 5:** at least 15% faster on the 120B MoE, or drop it.

### Phase 5b: faster chat replies (KV reuse and saving)

1. **Audit the fork first.** Check what the llama.cpp fork already has and reuse what works *(verify the current flags)*:
    - slot save/restore (`--slot-save-path`);
    - prompt cache;
    - `--cache-reuse`.

    Read how Colibri does prefix reuse and keeps KV across restarts.
2. **Build section 4b items 1–7, in that order.** Measure after each one and drop any that doesn't improve wait-before-reply or writing speed.
3. **Show the speed-up:** the server reports "cached tokens" for every request (OpenAI-style `prompt_tokens_details.cached_tokens`, *verify*), and the app shows "reply started in X s (Y tokens reused)".
4. **Two replay tests**, recorded once and replayed against the server:
    - **Chat replay:** a normal 30-message chat, then closing the app and continuing it the next day.
    - **Agent replay:** a real coding-agent session (about 15k tokens of system prompt and tool definitions, then 20 turns of tool output and replies).

**Gate 5b:**
- **(a) Exact:** output after reloading saved KV, from RAM or SSD, is identical to re-reading the text (section 6).
- **(b) Follow-up messages:** in the chat replay, the median wait before the reply starts is at least 5x shorter than with reuse switched off, on the streamed MoE.
- **(c) Reopening an old chat:** after a restart, a 32k-token chat gets to its first word at least 20x faster than re-reading it.
- **(d) Agent replay:** total waiting time is at least 3x lower than with reuse switched off.
- **(e) Writing speed holds:** on a 32k-token chat, writing speed is within 5% of a short chat's, or the auto-tuner's RAM split (4b item 4) explains the gap.
- **(f) No wrong reuse:** changing the model file, adapter, weight mode or KV precision never reuses old KV. Test each one; wrong reuse gives wrong answers silently.

### Phase 6: guess-ahead controller

1. **Draft sources, in order of preference:**
    - the model's own multi-token-prediction head, if it has one and llama.cpp supports it *(verify)*;
    - a small draft model from the same family;
    - the SB4 base, after Phase 7.
2. **Predict `new_bytes(k)`:**
    - MoE: the union of experts the draft predicts for the k guessed tokens, minus what is cached;
    - dense: the non-resident bytes, the same for any k.
3. **Each step:**
    - pick k with Appendix B;
    - track the acceptance rate as a running average;
    - feed the draft's expert choices to the prefetcher.
4. **Log "union growth"**: how many new experts each extra guessed token adds. Nobody has published this for shared-memory machines, so the owner can.
5. **KV for guessing:** use section 4b item 8 (roll back rejected tokens, draft KV). After every rejection, check that the KV length matches the accepted text.
6. **Guess while waiting for the SSD** (MoE-SpeQ, arXiv 2511.14102; section 10.5 item 5): run the guesser while a layer waits for experts to arrive, and use the guessed words' predicted experts to start the next SSD reads early. **Gate:** keep it only if a streamed MoE model writes at least 10% faster, with exact output.

**Gate 6:**
- **(a) Exact:** greedy output is identical with guessing on and off.
- **(b) Never meaningfully slower:** on every bench model, controller-on is no more than 3% slower than guessing-off.
- **(c) Dense gain:** the dense 32B streamed on the Mac with a small draft model is at least 2x Phase 2's plain-streaming speed.

### Phase 7: Shadow Bits (SB4)

Build the version that won M1 risk test 1: SB4 as written, SB4 built base-first the BitNest way (arXiv 2610.02800), or 4+4. Appendix A describes 2+2; the other two change how the base is built and the width of each plane.

**7a. Converter.** Implement Appendix A for GGUF tensors.
- Write the base plane plus scales to a RAM-side file, and the correction plane to an SSD-side file, each expert contiguous.
- The merge check (planes → ordinary 4-bit) must pass for every tensor.

**7b. Quality.** Compare SB4 full mode with the 4-bit format the owner normally uses (for example Q4_K_M).
- Measure perplexity (how well it predicts real text; lower is better) on a fixed 50k-token text sample, plus the JSON/tool-call test.
- Target: within 3%. Tune `alpha` (start 0.2, then try 0.1 and 0.3).
- If the gap stays above 3%, add importance weighting to the scale search, using activation statistics as llama.cpp's imatrix does.
- My synthetic tests (Appendix A) compared against plain round-to-nearest 4-bit, which is weaker than K-quants, so expect a bigger gap on real models.

**7c. Kernels.**
- Base-only 2-bit matrix-vector product: CPU NEON first, then Metal.
- Full mode: merge the planes into ordinary 4-bit when an expert or tensor enters the RAM cache, and reuse the existing 4-bit kernels.

**7d. Go / no-go.** Measure base-only draft acceptance on the bench prompts at k = 4.

| Acceptance | Action |
|---|---|
| 0.70 or more | Continue |
| 0.60–0.70 | Try the draft calibration in Phase 10, then re-measure |
| below 0.60 | Stop SB4, keep separate draft models, log the negative result |

**7e. Store integration.**
- Base planes always in RAM. With groups of 64 the base costs 2.5 bits per weight: about 32B total fits in the Mac's ~10 GB, and about 160B in the Minisforum's ~50 GB.
    - A dense 32B is right at the Mac's limit.
    - If it doesn't fit alongside the conversation memory, use groups of 128 for the base (2.25 bits, so about 35B and 175B fit), or test on a 27B-class model.
- Correction planes stream from the SSD.
- The cache holds merged 4-bit copies of hot experts or tensors.
- Bigger models: keep base planes for popular experts only, and skip the missing ones while drafting.

**Gate 7:**
- Dense 32B on the Mac in exact mode is at least 2.5x Phase 2's plain streaming, **and** at least 10% faster than Phase 6's best separate-draft result.
- Base-only "fast mode" is labelled lossy.

### Phase 7b: models that fit in memory

Run section 11's steps on each machine in the order P4 gives: F0 in M0, F1–F2 in M4, F3 in M10, and F4–F6 as they land. On the Mac, review the results here; on other machines, with Phases 8 and 9 and the Spark track.

### Phase 8: AMD port (Minisforum, Windows)

1. **Back-end:**
    - start with llama.cpp's Vulkan back-end, which runs on Windows;
    - test HIP/ROCm for this chip (gfx1151) on Windows or Linux if it is supported *(verify)*;
    - keep the faster one for decoding and log both.
2. **`io_windows.c`:** unbuffered overlapped reads. Measure against the drive's sequential speed.
3. **Memory split:** find and record the CPU/GPU memory split setting, and how much the GPU can really use.
4. **Re-run benches:** re-run the Phase 0–7 benches that apply. With 64GB, the focus is GLM-5.2-class MoE streaming and dense 70B–123B.
5. **Use the Minisforum's other parts** (measure each; keep only what's faster):
    - **The CPU can help here, unlike on the Spark.** Its 16 Zen 5 cores have full AVX-512 and are about 3× stronger at this than the Spark's CPU. gpt-oss-120b on the CPU alone measured about 28 tok/s on Strix Halo against about 9.5 on a Spark, and Colibri runs GLM-5.2 CPU-only at 1.83 tok/s on a 128 GB Strix Halo.
        - Test splitting each layer's experts between the CPU and the integrated graphics, in case the graphics alone doesn't use all of the 256 GB/s.
        - Both share that memory, so expect a small gain, if any.
    - **NPU (XDNA 2):** so far it runs only small models (about 1–8B, through Lemonade/FastFlowLM). Qwen3-8B decoded at 8.2 tok/s on it, slower than the graphics chip.
        - Use it for side jobs: the small draft model for guess-ahead, or a second background chat.
        - Running NPU and graphics together slowed the graphics chip's writing by 14% in one test, so measure.
    - **DirectStorage / IoRing:** test Windows' fast-loading API against plain unbuffered reads. Its graphics-chip decompression only pays if the experts compress, which 4-bit weights barely do (about 5–15%).
6. **Windows auto-setup:**
    - auto-setup and auto-tune must work on Windows;
    - the app detects the CPU/GPU memory split and tells the user in plain words if changing it would help, and how.
7. **Downloads:** a Windows download (a zip or installer).

**Targets on the owner's Minisforum MS-S1 MAX** (64 GB, 2 TB, estimates with the full plan). First number: one fast PCIe 4.0 drive in M.2 slot 1, about 6 GB/s. Second number: plus a second fast drive on a PCIe adapter card in the x16 slot, about 13 GB/s total:

| | GLM-5.3 | Kimi K2.6 | Nemotron 3 Ultra | Lightning 30B (4-bit) |
|---|---|---|---|---|
| Writing speed | 1.0–1.4 / 2.2–2.8 tok/s | 1.0–1.2 / 1.9–2.3 | 0.7–0.8 / 1.3–1.5 | 60–80 tok/s |
| Wait: short follow-up or reopened chat | 20–50 s / 10–26 s | 23–61 s / 12–31 s | 19–43 s / 10–22 s | about 0.1 s |
| Wait: coding-agent turn (2,000 new tokens) | about 66 s / 34 s | about 97 s / 52–69 s | about 51 s / 27–34 s | 1.7–3.4 s |
| Wait: agent setup first time (15,000 new) | 2–3.4 min | 2.5–4.2 min | 2.2–3.7 min | 11–21 s |
| Wait: long document (32,000 new) | 4–7 min | 8–14 min | 4.5–8 min | 26–51 s |

- **The MS-S1 MAX's storage, as confirmed:**
    - M.2 slot 1 is PCIe 4.0 x4. M.2 slot 2 is only PCIe 4.0 x1 (about 1.8 GB/s at most).
    - The full-length x16 slot is wired as PCIe 4.0 x4.
    - The two rear USB4 v2 ports run at 80 Gbps; the two front USB4 ports at 40 Gbps.
- **The included 2 TB Kingston (OM8TAP42048K1-A00) measured 3.0 GB/s** in a simple one-at-a-time read test (Notebookcheck). With parallel reads expect roughly 4–5 GB/s, so about 25% slower than the first numbers: GLM-5.3 writes at about 0.8–1.0 tok/s.
- **Best storage upgrade, in order:**
    1. A fast PCIe 4.0 drive (about 7 GB/s rated) in slot 1.
    2. A second fast drive on a PCIe adapter card in the x16 slot. Together that's about 13 GB/s: GLM-5.3 at about 2.2–2.7 tok/s and agent-turn waits of about 32 s.
    3. A USB4 v2 enclosure plus slot 2 add a few GB/s more, about 17 GB/s in total: GLM-5.3 at about 2.8–3.5 tok/s.

    Auto-setup must measure each drive; the reader spreads reads by measured speed.
- **Most of the waits are SSD time,** except first-time setups and long documents, which are limited by maths.
- **Use Performance mode (130 W).** On the 64 GB model, check how much memory Windows lets the graphics chip use; the 128 GB model allows up to 96 GB, the 64 GB model less.
- **Lightning needs 4-bit or 8-bit here,** because the 66 GB BF16 file doesn't fit in 64 GB.
- **Lightning's 11–21 s setup target needs fast Mamba kernels on AMD.** Today's llama.cpp measured only about 112 tok/s prompt reading for Nemotron 3 Nano on Strix Halo, against 1,142 for a similar-sized Qwen model.

**Gate 8:**
- the Windows download installs and sets itself up on the Minisforum with no manual steps;
- every bench model runs;
- the 120B MoE reaches at least 60% of the ceiling;
- expert reads reach at least 80% of the drive's sequential speed;
- Gate 5b (a)–(e) pass on Windows. The KV store uses the same unbuffered file reads.
- Kera is at least as fast as Colibri on every model both can run (section 7). If it isn't, profile it, find out why, and log the result in `docs/NEGATIVE_RESULTS.md`.

### Phase 9: NVIDIA port

1. **Hardware:** rent an NVIDIA machine by the hour. Ask the owner first and agree a budget.
2. **Discrete GPU tiers:** VRAM for always-used weights and the hottest experts, then RAM, then SSD.
3. **Transfers:** use pinned host buffers and asynchronous copies.
4. **Misses:** for each missing expert, choose between computing it on the CPU and copying it to the GPU, by measured time (the KTransformers and Fiddler idea).
    - **On a normal PC with a separate graphics card, the CPU genuinely helps,** because it has its own memory bandwidth that the graphics card doesn't use.
    - Use the CPU's matrix units: AVX-512, or AMX on Intel Xeons. KTransformers ran DeepSeek-R1 671B at 13.7 tok/s on one RTX 4090 plus two Xeons, with fast prompt reading from AMX.
    - The more memory channels the CPU has, the faster. A server-class CPU with 8–12 channels can hold a whole 600 GB model in RAM with no SSD at all.
5. **Optional:** GPUDirect Storage on Linux *(verify)*.
6. **If the owner buys a DGX Spark:** follow section 10 (steps S0–S8) instead of the discrete-GPU steps above.
7. **Linux/NVIDIA download:** auto-setup also detects VRAM size and how fast data moves between the graphics card and RAM, and the auto-tuner picks what goes in VRAM.
8. **KV on a discrete GPU:**
    - KV pages live in VRAM, spill to pinned RAM, then go to the SSD store;
    - the auto-tuner splits VRAM between KV and hot experts;

**Gate 9:** on the same machine and model, for a MoE that doesn't fit in VRAM, at least 20% faster than stock llama.cpp with `--n-cpu-moe` / `-ot` offload, with exact output.

### Phase 10: fine-tuning

**Don't write a trainer. Wrap the existing ones:**

| Machine | Trainer | Notes |
|---|---|---|
| Mac | MLX-LM LoRA / QLoRA *(verify the current commands)* | 16GB should handle about 7–8B models with QLoRA (estimate; measure) |
| NVIDIA | Unsloth or Hugging Face PEFT *(verify)* | |
| AMD | PyTorch with ROCm, if gfx1151 is supported *(verify)* | Otherwise train on rented NVIDIA |

**MoE models: expert-specialised fine-tuning (ESFT, DeepSeek 2024).**
1. Run the training data through the router.
2. Pick the experts it uses most.
3. Train only those, plus LoRA on attention.

This suits streaming, because only the trained experts must stay resident. KTransformers also has MoE fine-tuning *(verify)*.

**Export:** convert adapters to GGUF LoRA (llama.cpp's converter, *verify*), then `kera run --lora adapter.gguf`.

**KV and adapters:** an adapter changes the KV, so the adapter's hash is part of the KV cache identity (section 4b item 5). Switching adapters must never reuse KV saved under another one.

**Training for speed:**
- **Calibrate the SB4 base:** a short run that tunes the base scales and offsets to agree more often with the full model, then recomputes the corrections. Re-run 7b afterwards.
- **Draft heads:** train an EAGLE-style or MTP draft head for models that ship without one *(verify the EAGLE-3 training code)*.

**Gate 10:**
- **(a)** A LoRA on a small model, using a small dataset the owner provides, trains on the Mac, exports, loads in Kera and changes its answers as expected.
- **(b)** If draft training is tried, acceptance rises by at least 5 points.

### Phase 11: lossy speed modes (opt-in only)

- **Fast:** SB4 base only.
- **Balanced:** check guesses only when the draft is unsure (high entropy).
- **Precision by importance:** for dense models in RAM, read only the top bits of rows whose input is tiny (the GRINQH idea, arXiv 2606.23419). Target: dense 70B on the Minisforum.
- **Cache-aware routing (MoE):** when two experts score almost the same, prefer the cached one. Never swap the top two.
- **Smaller KV, for faster replies in long chats:**
    - 8-bit KV (llama.cpp `q8_0` / FP8) leaves more RAM for experts, so writing stays fast in long chats;
    - 4- or 2-bit KV (KIVI, arXiv 2402.02750) only if 8-bit isn't enough, because quality drops faster.
- **CacheBlend** (section 4b item 11): reuse a saved file or document that appears in a new place, so pasting it again doesn't mean re-reading it.

Every lossy mode gets a quality table next to its speed: perplexity, a small accuracy set, JSON/tool-call pass rate, and a long-answer check.

**Gate 11:** each mode documented with its measured quality cost. Any mode that breaks JSON or tool calls says so in its help text.

### Phase 12: optional experiments (only after Phase 11)

- **NPU** (Apple Neural Engine via Core ML; AMD XDNA via Ryzen AI software; the DGX Spark has none):
    - writing is limited by memory speed, not maths, so expect little there;
    - the M4's Neural Engine reaches about 19 TFLOPS (16-bit) at about 2.8 W, but has only about 32 MB of on-chip memory and high per-call overhead, and no LLM speed has been measured on it;
    - possible uses: the small draft model, or prompt reading for small models.
- **Apple M4 matrix unit (SME):** test it for the CPU-side draft model on the Mac. It doesn't help the 16 GB Mac's real limit, the SSD.
- **Two machines:**
    - use the Mac's spare RAM and SSD as extra expert storage for the Minisforum over wired Ethernet (Colibri has a cluster mode);
    - each layer then costs a network round trip, so measure the delay.
- **Cross-layer co-use disk layout:** a test on Apple Silicon found no gain, so treat it as low priority.

### Phase 13: polish the downloads

Downloads exist from the start: Mac from Phase 0, Windows from Phase 8, Linux/NVIDIA from Phase 9. Every phase that passes its gate ships a new build, as a GitHub Release (P8), that the owner can download and test. This phase finishes them:

1. **Commands** (still there for power users): `kera setup | plan | bench | repack | convert | run | serve | finetune`.
2. **Builds:**
    - Mac: signed and notarised app, if the owner has the developer account;
    - Windows: installer with the faster of Vulkan or HIP, using `kera.ico` as its icon;
    - Linux: CUDA build.
3. **Updates:**
    - the app checks for a new version and offers it, and never updates without asking;
    - the check needs a public download place (P8, Downloads). Until the owner picks one, there is no update check, and GitHub notifies him of new releases instead;
    - an update that changes the KV layout clears or converts saved chats, and says so first.
4. **Docs:**
    - README in plain English, with measured numbers only, and the logo at the top using the light/dark `<picture>` snippet in `assets/brand/BRAND.md`;
    - `NOTICE` file and licences.

**Gate 13:** on a clean machine of each kind, someone who has never used a terminal can download, open, let it set itself up, pick a model, and chat with it.

---

## 6. Exactness and quality tests

**Exactness:** the same model file must give the same answer whatever Kera does with placement or guessing.

1. **Reference run:** stock llama.cpp, the same GGUF fully in memory, CPU back-end, greedy decoding (temperature 0).
2. **Test:** 10 bench prompts, 256 generated tokens each.
    - **CPU path:** must match the reference token for token.
    - **GPU paths:** maths runs in a different order, so tiny rounding differences can flip a near-tie. At the first difference, look at the reference's word probabilities at that step: accept only if its top two were within 0.1 percentage points of each other. Then keep comparing by feeding in the reference's own tokens from there on (teacher forcing): every later step must pick the same top token, or differ only at another near-tie. Any other difference is a bug.
3. **Guessing on vs off:**
    - greedy output must be identical, or differ only at near-ties as above (checking several words at once changes the rounding order);
    - with sampling (temperature > 0), use the standard speculative-sampling acceptance rule so the output distribution is unchanged. Check it by sampling the first token 2,000 times both ways and comparing frequencies.
4. **KV reuse vs recompute:**
    - for each bench prompt, generate once from scratch;
    - then generate again after reloading the prompt's KV from RAM, from the SSD store, and after an app restart;
    - all four must match token for token on the CPU path, and pass the near-tie rule above on GPU paths.
5. **Wrong-identity test:** save KV under one model file, adapter, weight mode or KV precision, then switch. The store must report a miss, never reuse it.

**Quality**, for SB4 full mode vs normal 4-bit, and for every lossy mode:

| Test | Contents |
|---|---|
| Perplexity | Fixed 50k-token text sample; lower is better |
| Accuracy set | About 200 questions: a maths word-problem subset plus multiple choice |
| JSON/tool-call test | 50 prompts that must return valid JSON matching a schema; report the pass rate |
| Long-answer check | A 2,000-token generation; flag loops and repetition |

---

## 7. Benchmark protocol

**Prompts** (in `bench/prompts/`):
- short chat;
- code;
- long context (about 4,000 tokens in);
- JSON tool call.

**Record for every run:**
- **Setup:** machine, Kera commit, model file and its hash, all settings, cache state (cold or warm).
- **Speed:**
    - prompt speed (tok/s);
    - decode speed (tok/s, median of 3 runs of 256 tokens);
    - time to first token.
- **Traffic and memory:**
    - GB read from SSD per token;
    - cache hit %;
    - prefetch accuracy;
    - peak RAM.
- **Guessing:** acceptance rate, and how often each k was chosen.
- **KV:**
    - time to first word three ways: cold (nothing saved), warm from RAM, warm from the SSD store;
    - KV hit rate (tokens reused ÷ prompt tokens);
    - KV GB in RAM;
    - total waiting time in the agent replay (Phase 5b).
- **Ceiling:** the planner ceiling, and measured ÷ ceiling.

**Cold cache:** the built-in bench reads model files with the same cache-bypassing reads as auto-setup, so it needs no admin rights. For a true cold start, also test once after a reboot.

**Warm cache:** run the prompt once first.

**Laptop heat:** plug the charger in, let it cool between runs, and note it if speeds drift down run after run.

**Baseline:** every result is compared with stock llama.cpp on the same machine and model file, run the same day.

**Colibri baseline for streamed MoE models:** every benchmark of a streamed MoE model also runs Colibri's latest release on the same machine, the same day, with the same model where both support it, after running its own `coli tune`. Record Colibri's version and settings with the result. Colibri uses its own file format, so note both model files.

**Output:** JSON in `bench/results/`, plus one row in `docs/BENCHMARKS.md`.

---

## 8. Things not to do (each one already failed for someone)

- **Don't memory-map the experts on the Mac and hope.** Flash-MoE measured mmap 5x slower than parallel `pread`. Measure it once, then use `pread`.
- **Don't write a custom cache before trying the OS file cache.** In Flash-MoE, removing their own cache made it 38% faster.
- **Don't compress 4-bit weights with LZ4 or similar.** It was 13% slower in Flash-MoE; 4-bit data has little left to squeeze. Lossless compression only pays on 16-bit weights (DFloat11 saves about 30%).
- **Don't overlap SSD reads with GPU work on the Mac without measuring.** It was slower in Flash-MoE.
- **Don't switch guessing on for MoE models without the controller.**
    - Colibri measured its draft head 32% slower at about an 85% expert hit rate.
    - DraftExpert measured naive self-drafting at 1.19 tok/s against 2.19 without it.
- **Don't make 2-bit the default.** It broke JSON and tool calls in Flash-MoE.
- **Don't delete layers to save memory.** Layer pruning breaks reasoning. It is only acceptable inside a draft model.
- **Don't trust a paper's speed-up on different hardware.** Reproduce it here first.
- **Don't benchmark with the cache in an unknown state.** That includes the KV store: clear it, or record what was in it.
- **Don't reuse KV unless the whole cache identity matches** (section 4b item 5). KV from another model file, adapter, weight mode or KV precision gives wrong answers silently.
- **Don't make smaller-than-16-bit KV the default.** It changes answers; it is a lossy mode.
- **Don't process long prompts in small chunks on streamed MoE models.** Each chunk pulls in nearly every expert again. Go layer by layer across the whole prompt (section 4b item 3).
- **Don't run expert maths on the DGX Spark's CPU.** Measured on a Spark: DeepSeek V4 Flash fell from 16.2 to 4.7 tok/s as experts moved to the CPU. The CPU's job is the I/O and scheduling (section 10.3).
- **Don't read experts in small pieces on a Spark.** 1 MiB reads got 4.9 GB/s; 16 MiB reads got 11.1 GB/s.
- **Don't rely on GPUDirect Storage on GB10.** NVIDIA staff say it isn't supported.
- **Don't use the video decoder to unpack compressed weights.** It works in research (VcLLM: about 2.9 bits per weight), but NVIDIA's video decoder only manages about 1.1 GB/s, far slower than the SSD, and quality drops.
- **Don't count on CPU caches.** They hold megabytes and the models are hundreds of gigabytes; the CPU already uses them automatically for small things.
- **Don't use USB drives for experts on a Spark.** The ports are 20 Gbps (about 2 GB/s at best) and links have dropped to USB 2.0 speed.
- **Don't grow the pinned cache until the machine freezes.** Find the stable maximum per machine (section 10.6).
- **Don't assume clients resend identical text.** Templates that drop or reformat earlier turns silently kill prefix reuse. Measure the KV hit rate in the agent replay.

---

## 9. Reporting to the owner

After each phase, send him a short update in plain English. Post it as a comment on the milestone's GitHub issue, and use it as the release notes when that milestone ships:
- what you built;
- whether the gate passed;
- the key number next to stock llama.cpp's (for example "1.9 tok/s vs llama.cpp 1.2 tok/s on gpt-oss-120b; 1 tok/s is about 45 words a minute");
- what comes next;
- any decision he needs to make (a download, a purchase, dropping a feature).

Explain any measurement you mention.

---

## 10. DGX Spark: the maximum-speed plan

**Goal:** the fastest way to run models bigger than memory on one NVIDIA DGX Spark Founders Edition (GB10, 128 GB shared memory). Target models: GLM-5.3, Kimi K2.6 and Nemotron 3 Ultra 550B, plus models that fit, such as Nemotron 3.5 Lightning 30B.

Nobody can promise "fastest on earth", so the rule is measurable instead:
- **Beat the best published numbers.** Kera must beat the best numbers anyone has published for one Spark.
- **Get close to the ceiling.** It must also get within 80% of the planner's overlapped ceiling (Appendix C, `overlap=True`).
- **Adopt anything faster.** If anyone publishes something faster, measure it and adopt it.

### 10.1 Where the time goes, and the one big fix

Per word, GLM-5.3 on a Spark:
- **SSD wait:** about 12.7 GB of experts are needed. With about 82% already in memory, about 2.3 GB comes from the SSD, which takes 0.21 s at 11 GB/s.
- **Maths:** about 25 GB has to be read from memory and computed. That's about 0.11 s at the 231 GB/s the graphics chip actually reaches, so 0.14–0.21 s in practice.
- **Today's engines do these one after the other:** about 0.42 s, or 2.4 tok/s. That matches the 2.39 tok/s a GB10 owner measured for GLM-5.2 in Colibri (issue #161).
- **The fix is doing them at the same time,** so a word costs max(SSD, maths) instead of SSD + maths. The CPU is what makes that possible (10.3).

### 10.2 Targets

My estimates, using the same model as the earlier simulations, warmed up, on the NVIDIA Founders Edition:

| Model | Best published on one Spark today | Kera, exact | + network storage box (10.7) | + lossy turbo (10.9) |
|---|---|---|---|---|
| GLM-5.3 (744B) | GLM-5.2: 2.4 tok/s (Colibri); GLM-5.3 NVFP4: 0.81 (SparkLab) | 4.3–5.5 if hits reach 86–88%; about 3.5–4.3 at 82% | about 6.7 | about 7 |
| Kimi K2.6 (1T) | none published; Kimi K3 IQ1_S: 0.70 (Pulsar) | 3.6–4.4 if hits reach 75–80%; about 1.8 at 50% | about 4.9 | about 5.2 |
| Nemotron 3 Ultra (550B) | none published | about 3.1 | about 3.3, up to about 5 with guess-ahead (10.8) | about 3.4 |
| Nemotron 3.5 Lightning (30B, fits) | same layout as Nemotron 3 Nano: 64–74 tok/s at 4-bit (llama.cpp) | 75–115, mostly from fused kernels (11.2); guessing adds little on small-active MoE (11.1 item 4) (estimate) | — | — |

**These targets depend on expert hit rates.**
- With 10.1's numbers (82% hits, SSD 0.21 s, maths 0.14–0.21 s), the planner's overlap ceiling for GLM-5.3 is about 4.1–4.3 tok/s. The 4.3–5.5 range assumes the cache brain (10.3) lifts hits to about 86–88%.
- Kimi K2.6 keeps only about 15% of its expert bytes in memory. Its range assumes GLM-like routing, where that still gives 75–80% hits.
- Reset both from the P5 routing traces before chasing them.

**Why each model is limited differently:**
- **GLM-5.3 and Kimi K2.6** are limited by SSD reads, so storage and caching matter most.
- **Ultra** reads about 37.5 GB of always-used weights every word, so it is limited by memory speed. Guess-ahead helps it most, because all the guessed words share that read.

**Final gate for this section** (exact output, warmed, Founders Edition, median of 3 × 256 tokens):
- GLM-5.3: at least 3.5 tok/s, and at least 80% of the planner ceiling at the measured hit rate;
- Kimi K2.6: at least 80% of the planner ceiling at the measured hit rate;
- Nemotron 3 Ultra: at least 2.8;
- Lightning: at least 1.2× stock llama.cpp with the same model file.
- Kera is at least as fast as Colibri on every model both can run (section 7). If it isn't, profile it, find out why, and log the result in `docs/NEGATIVE_RESULTS.md`.

### 10.3 The CPU's job: run the traffic, not the maths

**Measured on Sparks: CPU expert maths makes it slower.**
- DeepSeek V4 Flash in llama.cpp dropped from 16.2 tok/s (all experts on the graphics chip) to 9.9, 6.6 and 4.7 tok/s as more layers' experts moved to the CPU.
- gpt-oss-120b on the CPU alone ran at about 9.5 tok/s, against 56 on the graphics chip.

**Why:**
- The CPU and graphics chip share the same 273 GB/s of memory, and the graphics chip alone already pulls about 231 GB/s.
- When both read at once, CPU memory delay rises from about 113 ns to over 350 ns.

So the 20 CPU cores must never touch the big data. Their jobs:

1. **I/O engine.**
    - Use `io_uring`, or a thread pool of `pread`s, with `O_DIRECT`.
    - Keep at least 16 MiB of aligned reads in flight in total: for example 16 reads of 1 MiB, or 4 of 4 MiB. Small experts (about 1–4 MB in Nano, Qwen3-30B or Ultra) can't each be read in 16 MiB pieces, so batch neighbouring experts into bigger reads where the file layout allows. Tests on a Spark: 1 MiB reads got 4.9 GB/s, 4 MiB got 6.8, 16 MiB got 11.1, and 16 threads with 1 MiB reads got 11.4 GiB/s.
    - Reads land in pinned buffers that the graphics chip reads directly (zero-copy). Pulsar measured that 6.8% faster than copying.
    - The CPU never reads or copies an expert itself.
2. **Prefetch brain.**
    - Run the next layers' routers ahead of time to guess their experts (GLM-5.2's next layer is 71.6% predictable; Fate reports 79–97%).
    - Look several layers ahead (SeqMoE).
    - Schedule reads by deadline: the expert needed soonest goes first.
    - Wrong guesses waste SSD time, so guessed reads only use bandwidth the certain reads leave free.
3. **Cache manager.**
    - Decide what stays in memory using predicted future use plus learned pins (Phase 4).
    - Evict the expert expected to be needed furthest in the future (SeqMoE and FlashMoE approximate this).
4. **Everything else:** tokenizer, sampling, server, KV store saves (section 4b), the guess-ahead controller.
5. **Experiments only (keep only if measured faster):**
    - **A small draft model on 4–6 CPU cores** while the graphics chip checks guesses.
    - **Lossless decompression of experts.** Expect at most 10–15% smaller for int4 and about 5% for FP4, and the GB10 has no hardware decompressor.

**Thread placement:** pin threads to cores, I/O submission on the fast X925 cores and background work on the A725 cores, and keep total CPU memory traffic tiny.

### 10.4 Graphics chip side

- **One fused kernel per layer for the experts:** a grouped 4-bit matrix-times-vector (W4A16 int4 or NVFP4) covering all 8–22 experts of the layer.
- **Fast attention:** a FlashMLA-class kernel for MLA models (GLM, Kimi), and a Mamba-2 decode kernel for Nemotron. Once experts were in memory, Colibri on a GB10 spent about 80% of decode time in attention, so this matters.
- **Compute-on-arrival:** start the cached experts immediately, and compute each missing expert the moment its read completes (one completion event per read). The layer then finishes about when its last read lands.
- **Whole decode on the graphics chip:** keep the model's running state there between layers; no round trips to the CPU.
- **CUDA graphs:** only if the memory cost is small. vLLM's cost 13 GB for about 3% on a Spark, so measure.
- **Gate:** at 100% expert hits, maths time is within 1.4× of the memory-read floor.

### 10.5 The overlap schedule (the biggest single gain)

For each layer of each word:
1. The router runs and the misses become known. The CPU sends those reads at top priority; they are on the critical path.
2. Guessed experts for the next 1–3 layers are queued behind them.
3. The graphics chip computes cached experts while misses load, then each missing expert on arrival (10.4).
4. **Across words:** while the last layers of a word compute, start reading the first layers' likely experts for the next word. Use the guesser's predicted words if guess-ahead is on (the SP-MoE idea); otherwise use "the experts the last word used", and measure how often that is right.
5. **Guess while waiting for the SSD** (MoE-SpeQ, arXiv 2511.14102): when a layer is waiting for experts to arrive from the SSD, use that idle time to run the guesser (MTP head, Shadow Bits base or prompt copy). Use the guessed words' predicted experts to start the next SSD reads early. Keep it only if a streamed MoE model writes at least 10% faster, with exact output.

**Gate:** time per word ≤ 1.2 × max(SSD time, maths time), measured on GLM-5.3. Expected gain over doing them one after the other: about 1.4–1.8×.

**Mac warning, check on the Spark:** on Macs, reading the SSD during graphics work was slower (Flash-MoE). On the Spark the SSD uses only about 4% of memory bandwidth, so it should be fine, but measure it.

### 10.6 Spark setup (auto-setup does this; ask the owner before changing system settings)

- **Free memory:**
    - Headless mode saves about 2–3 GB.
    - DGX OS 7.5.0 cut usable memory to about 112 GiB because of a kernel feature (KHO). The `kho=off` boot option or NVIDIA's fix restores about 119.5 GiB. Check which this machine has.
    - Set `vm.swappiness=1` and drop the file cache before loading.
    - Never memory-map the model: `--no-mmap` measured 56.3 vs 44.7 tok/s in llama.cpp on a Spark.
- **Expert cache size:**
    - Use one big pinned allocation.
    - Grow it in steps and stop before the system gets unresponsive. Pulsar saw that above about 80 GB of host cache on a Spark.
    - Record the stable maximum for this machine.
- **Heat:**
    - `nvidia-smi -lgc 300,2200` (capping the graphics chip's clock) cut peak temperature by 5–20 °C with no measured cost on memory-bound MoE work.
    - Log clocks during long runs, because throttling silently costs speed.
- **Storage:**
    - Keep models on the internal Gen5 SSD (ext4 or XFS).
    - GPUDirect Storage isn't supported on GB10 (NVIDIA staff), so don't build on it.
    - Other brands' GB10 boxes measured about 5–5.5 GB/s (Dell, HP, ASUS) against about 11 GB/s for NVIDIA, Acer and Gigabyte, so auto-setup must measure, not assume.

### 10.7 More storage speed (hardware; ask the owner before buying)

1. **Network storage box on a ConnectX-7 port.**
    - NVMe over the network measured about 9.5–12 GB/s on a Spark.
    - Split the experts across the internal SSD and the box, about 21 GB/s together.
    - Gains: GLM-5.3 from about 5.5 to 6.7 tok/s; little gain for Ultra (memory-limited).
2. **A second Spark** (or more):
    - **Measured:** GLM-5.2 at 1-bit on 2 Sparks reached 15.9 tok/s over RDMA. GLM-5.2 int4/int8 on 4 Sparks reached 32–36 tok/s; 3 Sparks managed 21–26.
    - In community tests, splitting each layer across Sparks (tensor parallel) beat giving each Spark different experts (expert parallel).
    - This is the only path to 10+ tok/s for these giant models at good quality.
    - An untested alternative: use the second Spark's memory as a remote expert cache over RDMA. The link measured about 24.5 GB/s.
3. **USB drives: not worth it.** The ports are 20 Gbps USB 3.2 (about 2 GB/s per drive at best), and links have dropped to USB 2.0 speed after reboots.

### 10.8 Guess-ahead on the Spark

Use the Appendix B controller with measured union growth (Phase 6). Expectations:
- **GLM-5.3 and Kimi K2.6:** 0–15% gain, unless storage or caching pushes expert hits above about 92%. GB10 owners found the break-even somewhere between 92% and 99.9% hits.
- **Nemotron 3 Ultra:** the most gain, because every guessed word shares the 37.5 GB always-read part. With the network storage box, roughly 3.3 → 5 tok/s (estimate). Use its built-in guesser (NVIDIA's configs guess 3–5 words).
- **Models that fit (Lightning):** a large gain. On a GB10 with nearly every expert in memory, guessing by copying matching text from earlier in the chat measured +66% for GLM.
- **Guess sources:**
    - the built-in multi-word guessers (Ultra and Lightning have one; GLM-5.3's model card lists one, but Colibri's GLM-5.3 file leaves it out);
    - copying matching text from earlier in the chat, which is free and strong for code edits;
    - a small draft model.
- **Shadow Bits doesn't apply to these giant models on a Spark:** their 2-bit base wouldn't fit (GLM-5.3 about 232 GB, Ultra about 172 GB). It does apply to dense models of about 70–300B on the Spark.

### 10.9 Lossy turbo (opt-in, labelled, quality-tested per section 6)

- **Prefer experts already in memory (cache-aware routing):** on a GB10, GLM-5.2 went from 2.39 to 3.33 tok/s with about 14% of experts swapped. Its quality hasn't been benchmarked.
- **Store rarely used experts at 3-bit:** fewer bytes read on misses. They are rarely picked, so quality should suffer less than shrinking everything.
- **Expert-pruned versions (REAP):** remove rarely used experts entirely, if a pruned file of the model exists.
- **8-bit attention for Kimi K2.6:** halves its 21 GB per word of always-read 16-bit weights.

### 10.10 Wait before the first word (prompt reading) on the Spark

- **Read layer by layer** (section 4b item 3), and load the next layer's missing experts while the current layer computes. A long prompt needs nearly all of them anyway, so this is exact.
- **Use the graphics chip's FP4/FP8 maths for prompt reading.** Unlike writing, reading a prompt is limited by maths.
- **Expected for a long new prompt on GLM-5.3:** about (missing experts) ÷ (SSD speed) ≈ 300 GB ÷ 11 GB/s ≈ 28 s, against about 45 s at 6.6 GB/s.
- **Follow-ups:** KV reuse (section 4b) means only the new text has to be read.

**Wait-before-first-word targets** (estimates, Founders Edition, full plan; ranges run from uneven to even expert use):

| Situation | GLM-5.3 | Kimi K2.6 | Nemotron 3 Ultra | Lightning 30B |
|---|---|---|---|---|
| Short follow-up (50 new tokens), or reopening a saved chat | 7–23 s | 8–30 s | 5–18 s | under 0.1 s |
| Paste a page (1,000 new) | about 28 s | 41–46 s | about 21 s | 0.2–0.5 s |
| Coding-agent turn with 2,000 tokens of tool output | about 30 s | about 46 s | about 21 s | 0.6–1.2 s |
| Agent setup already saved, + 200 new | 18–29 s | 22–45 s | 13–21 s | about 0.1 s |
| Agent setup first time (15,000 new) | 40–66 s | 51–84 s | 41–71 s | 4–8 s |
| Long document (32,000 new) | 1.3–2 min | 2–4 min | 1.4–3 min | 9–18 s |

- **Why anything over a few hundred tokens costs about 28 s on GLM-5.3:** that much new text needs nearly every expert, so the whole uncached part of the model (about 300 GB) comes off the SSD once.
- **Why Kimi's long documents are slower:** it has no sparse attention, so its attention maths grows with document length.

**Gate:** within 1.5× of these numbers. For comparison, today's engines on GLM-5.3 (no saved KV, small chunks, no overlap) model out at about 40 s, 1.7 min, 3 min, 26 min and 55 min for the same rows.

### 10.11 Spark track: steps and gates

If the owner has a Spark, do these after Phase 7, in place of Phase 9's discrete-GPU steps. If he gets the Spark earlier and asks, do them first.

| Step | What | Gate |
|---|---|---|
| S0 | Measure: SSD speed at 1/4/16/32 MiB reads and 1–32 in flight; memory speed for the graphics chip alone, the CPU alone, and both; the stable pinned cache size | Numbers in `docs/BENCHMARKS.md` |
| S1 | I/O engine (10.3 item 1) | ≥ 90% of the best measured raw read speed; ≥ 10 GB/s on a Founders Edition |
| S2 | Graphics chip kernels (10.4) | Maths within 1.4× of the memory floor at 100% hits |
| S3 | Overlap schedule (10.5) | Time per word ≤ 1.2 × max(SSD, maths) |
| S4 | Cache brain (10.3 item 3) | ≥ 3 points more hits than LRU + pins, on held-out prompts |
| S5 | Guess-ahead with measured union growth (10.8) | Never > 3% slower than off; Ultra and Lightning gain |
| S6 | Network storage, if bought (10.7) | Combined read speed ≥ 85% of the sum of both sources |
| S7 | Lossy turbo (10.9) | Each mode has a quality table |
| S8 | Several Sparks, if bought (10.7) | Beats the published 2-, 3- or 4-Spark numbers for the same model |

The final gate is in 10.2.

---

## 11. Models that fit in memory: beating llama.cpp, MLX and vLLM

**Goal:** for every model that fits in memory, dense or MoE, Kera must beat the fastest existing engine on the same machine with the same model file: llama.cpp, MLX, vLLM, SGLang, and on the Spark DGPP. Together with sections 4–10 that covers all four cases:

| | Fits in memory | Bigger than memory |
|---|---|---|
| **Dense** | Guess-ahead + fused kernels + the best format per byte (this section) | Shadow Bits + guess-ahead + streaming (Phases 6–7) |
| **MoE** | Fused "whole-word" kernels + expert-aware guess-ahead (this section) | Expert streaming + overlap + cache brain + KV reuse (Phases 2–6, sections 4b and 10) |

**Why this is hard:** llama.cpp already uses most of the memory speed on these machines.
- **Where it stands:**
    - M4: about 77–84% of 120 GB/s on a 7B.
    - Strix Halo: about 83% of 256 GB/s (almost all of the ~215 achievable) on a 70B.
    - Spark: about 87% of 273 GB/s on a 7B.
- **Better code alone adds only about 1.05–1.2x for models of 7B and up.** The big wins come from producing several words per memory read (11.1) and from cutting fixed per-word overhead on small or small-active models (11.2).

### 11.1 Guess-ahead for models that fit (the biggest lever)

When the model fits, checking 3–8 guessed words costs about the same memory reading as writing one. Measured:
- **Built-in multi-word heads (MTP)** in llama.cpp (`--spec-type draft-mtp`, merged around May 2026, *verify*), Qwen3.6-27B:
    - Spark: about 13 → 28 tok/s.
    - Strix Halo (Vulkan, same quant): 3.0–3.3x.
    - On MLX (MTPLX, M4 Pro): about 7 → 18.3 tok/s.
- **DFlash** (a guesser that drafts a whole block of words at once) is the strongest in its own paper, but not everywhere:
    - up to 6.1x in its paper;
    - 1.98x on an RTX 3090 for a 27B;
    - on an M3 Ultra, 32 → 110 tok/s for code at temperature 0 with llama.cpp's new few-row Metal kernel (b11404, the PR author's numbers);
    - on Strix Halo (Vulkan), llama.cpp's MTP beat DFlash by 23–67% (Qwen3.6-27B: 20.9 vs 16.1 tok/s). Try a model's own MTP head first when it has one, and let the controller measure.
- **Guessers that need no model:**
    - copying from the prompt: 2–4x when answering about given text;
    - SuffixDecoding: up to 5.3x for agent workloads (in vLLM);
    - lookahead decoding: 1.8x.
- **EAGLE-3 is weak on Apple** (1.05x, sometimes slower), because checking each extra word cost 0.5–1 ms there. The checking kernel decides the gain: on Jetson Thor, EAGLE-3 gave 1.59x with INT4 weights but 3.45x with NVFP4.

What Kera builds:
1. **Every guesser behind one interface:** MTP heads, DFlash, EAGLE-3, a small draft model, the SB4 base, copying from the prompt, SuffixDecoding and lookahead. The Appendix B controller picks the guesser and how far to guess per model, machine and task, from measured speed. Its formula only gives the starting k; the measured test-and-set loop (Cascade, arXiv 2506.20675; `KTuner` in Appendix B) has the final say.
2. **Fast checking kernels for 2–16 rows on every platform:** Metal (like llama.cpp's few-row matrix kernel), CUDA (including GB10 with NVFP4) and Vulkan/HIP. This is the gate for everything else.
3. **Plain chains of guesses on 4-bit models,** and trees only where the checking kernel makes them cheap. Checking a tree cost 1.8x a normal step on 4-bit weights.
4. **MoE models that fit:** guessing ahead often barely helps or hurts. A Spark measured Qwen3.6-35B-A3B at 64 tok/s with MTP against 68 without, because checking loads every expert any guess uses. So:
    - use expert-aware guessing that prefers guesses sharing experts (the ECOSPEC idea: gpt-oss-120b improved from 1.14x to 1.31x);
    - let the controller switch it off.
5. **Train guessers** for models that ship without one (DFlash or EAGLE-style heads, Phase 10).
6. **Expected gains:**
    - dense models that fit: 2–3x over writing without guessing, and more on code. llama.cpp's own best guessing (MTP or a draft model) is the real bar, and the gain over that will be smaller;
    - small-active MoE: 1.0–1.3x from guessing (most of their gain comes from 11.2).

### 11.2 Fused whole-word kernels (the second lever, biggest for small and small-active models)

At 60–150 tok/s, each word takes only 7–17 ms, and hundreds of separate kernel launches plus gaps between them waste a real share of that.
- **Measured:**
    - **Hazy Research's one-kernel decode** (a whole Llama-1B word in one GPU kernel): 1.5x faster than SGLang on H100, using 78% of memory speed against at most 50% for vLLM and SGLang.
    - **Cohere's MoE megakernel:** a 30B MoE with 3.3B active ran at 292 tok/s against 185 for vLLM (1.58x).
    - **BaseRT's fused Metal kernels on M4 Pro:** 1.56x faster than llama.cpp on a 0.6B model, but only 1.01–1.15x on 3B–30B-A3B.
- **Build:**
    - a persistent "megakernel" decode path for CUDA (including GB10);
    - fused per-layer kernels for Metal and Vulkan/HIP: attention, routing, all experts of the layer, and the Mamba step for Nemotron, with no CPU round trips.
- **Who benefits most:** small-active MoE models (Nemotron 3 Nano, Lightning, Qwen3-30B-A3B class), whose per-word maths is tiny, so overhead is a big share of each word.
- **Target:** at least 80% of memory speed at batch 1 (the best published is 78–87%), and 1.2–1.6x over stock llama.cpp on small-active MoE.

### 11.3 Fewer bytes at the same quality

Writing speed scales with bytes read, so pick the smallest format that meets a quality target, per model and machine.
- **Measure quality as KL divergence** (how different the answers' word probabilities are from the original model; lower is closer). On Qwen3.5-35B-A3B: Q4_K_M 0.019, MXFP4 0.027, Q3_K_M 0.032, IQ3_XXS 0.050.
- **Spark:** NVFP4 was 1.59x faster than FP8 on Qwen3-8B because it's smaller. llama.cpp's NVFP4 files aren't smaller than Q4_K_M, so check file size, not the name.
- **EXL3 (trellis quants):** better quality per bit (KL 0.025 at 4.0 bits vs 0.040 for Q4_K_M) but slower writing in its current CUDA kernels. Only worth it if Kera's kernel is as fast as a K-quant's.
- **The auto-tuner shows a "quality vs speed" slider** labelled with measured KL divergence. The default is the 4-bit format closest to the original model.

### 11.4 The best code path per platform and per phase

- **Strix Halo:** prompt reading on ROCm was 2.6–3.6x faster than on Vulkan (70B: 94.7 vs 26.4 tok/s), while writing was 5–10% faster on Vulkan.
    - A March 2026 test found a smaller gap: ROCm 7 nightlies read prompts at about 502 tok/s against 358 on Vulkan (AMDVLK), while Vulkan wrote about 16% faster (Qwen3-Coder-Next).
    - Kera uses ROCm/HIP for prompt reading and Vulkan for writing in the same run only if both work on Windows for gfx1151 *(verify)* **and** both can share one copy of the weights in memory. Two copies would halve what fits in 64 GB. Otherwise the tuner picks one back-end per model.
    - Fix Nemotron's slow Mamba prompt reading on AMD: about 112 tok/s against 1,142 for a similar-size Qwen model.
- **Mac:** MLX beats llama.cpp on prompt reading for prompts of 512+ tokens and on MoE writing. llama.cpp with flash attention beats MLX on long chats (30K+).
    - Borrow the faster kernel for each case (MLX's are MIT-licensed; keep the notice), or let the auto-tuner choose per model and phase.
- **Spark:** SGLang (52 tok/s) and vLLM beat llama.cpp (46) on gpt-oss-120b, and DGPP, a GB10-specific engine, claims 1.3–1.5x over vLLM.
    - Kera's CUDA path must beat all of these on the bench models.
- **KV precision:** on Metal, 8-bit and smaller KV made writing slower in long chats (44.4 → 40.7 tok/s). Keep 16-bit KV when it fits; use smaller KV only to fit longer chats.
- **Trap:** with llama.cpp on CUDA, quantized KV without `GGML_CUDA_FA_ALL_QUANTS` silently fell back to the CPU, making prompt reading 25x slower.

### 11.5 Lossy extras (opt-in, section 6 quality tests)

- **Activation sparsity:** TEAL, R-Sparse, Prox. 1.4–1.8x with 16-bit weights, but unproven on top of 4-bit weights and never measured on Apple or AMD, and MMLU dropped about 5 points at 50% sparsity. Experiment only.
- **QuantSpec:** the model guesses with a 4-bit copy of its own conversation memory and checks with 8-bit; 1.35–2.49x in long chats, near-lossless.

### 11.6 Steps and gates (run on each machine as it is ported)

| Step | What | Gate |
|---|---|---|
| F0 | Baselines: stock llama.cpp, MLX (Mac), vLLM/SGLang where they run, DGPP (Spark), and Colibri (latest release, after `coli tune`) where it supports the model, on the bench models that fit | Table in `docs/BENCHMARKS.md` |
| F1 | Few-row checking kernels (11.1 item 2) | Checking 8 words costs ≤ 1.3× one word |
| F2 | Guesser interface + controller (11.1) | Dense 27B-class model ≥ 2× llama.cpp without guessing, and faster than llama.cpp's own best guessing setup on the same model; never slower than off |
| F3 | Fused whole-word kernels (11.2) | ≥ 80% of memory speed; small-active MoE ≥ 1.2× stock llama.cpp |
| F4 | Format choice by quality (11.3) | Slider labels match measured KL divergence |
| F5 | Per-phase code paths (11.4) | Prompt reading ≥ the best engine on each machine |
| F6 | Final | For every bench model that fits, Kera is at least as fast as the fastest engine measured in F0, in both writing and wait-before-first-word. On the owner's MS-S1 MAX, his 8-second wait for Nemotron 3 Nano drops below 1.5 s before the first visible word, for the same prompt, measured with thinking off (with thinking on, report thinking time separately) |

**Exactness note:** checking several guessed words at once can change the rounding order. The output keeps the same probabilities but may not be bit-identical; section 6's near-tie rule applies.

---

## 12. Speed tricks borrowed from other engines

Checked on 14 Oct 2026 across llama.cpp, MLX, Ollama, mistral.rs, vLLM, SGLang, TensorRT-LLM, LMDeploy, KTransformers and recent papers. Only tricks that make **one person's** chat faster are kept. Most of the famous vLLM, SGLang and TensorRT-LLM numbers come from serving many users at once, which doesn't make a single chat faster, so those are left out (12.4).

Every item goes behind a setting, is added to the auto-tuner (P1, 4a) and has to pass its gate before it is switched on.

### 12.1 Build these (output stays exact)

1. **Skip the forced parts of JSON and tool calls (grammar fast-forward).**
    - **From:** SGLang ("jump-forward"), XGrammar, llguidance.
    - **What it does:** when a JSON schema or tool-call format leaves only one possible next piece of text, for example `"name": "`, add all those tokens in one step instead of writing them one at a time.
    - **Measured:** Llama-3.1-8B on an RTX 4090, one user: 6.8 → 5.4 ms per token, about 21% less time (XGrammar paper). On function-call schemas, about 38% of output tokens were forced (JSONSchemaBench).
    - **The gap:** llama.cpp uses llguidance only to block illegal tokens; it doesn't fast-forward.
    - **Build:**
        - use llguidance (MIT; already an option in llama.cpp behind `LLAMA_LLGUIDANCE`);
        - feed the forced tokens through as a small prompt-reading batch;
        - re-tokenise the normal way (back up 1–2 tokens, as llguidance does) so the output matches;
        - also use llguidance in place of llama.cpp's own grammar engine (GBNF), which roughly doubled the time per token in a January 2025 test *(re-measure)*.
    - **When:** with tool calling (M2).
    - **Gate:** the section 6 JSON and tool-call test gives identical output with fast-forward on and off, and the JSON bench prompt finishes at least 15% sooner.

2. **Fused kernels on every backend, and the right backend on the Spark.**
    - **What it does:** llama.cpp's CUDA fusions put norm + multiply + add, matrix-vector + gate, and MoE softmax + top-k each into one kernel, and run Q, K and V at the same time.
    - **Measured:** Qwen3-30B-A3B 198 → 271 tok/s on an RTX 4090 and 247 → 352 on a 5090 (fusions plus `GGML_CUDA_GRAPH_OPT=1`).
    - **Build:** Kera inherits the CUDA fusions. Port the missing ones to Metal and Vulkan, but measure each: the Vulkan ports so far gave only 1–3%.
    - **On a GB10:** `GGML_CUDA_GRAPH_OPT=1` made Nemotron 3 Nano slower (61.8 → 54.8 tok/s), and Vulkan was faster than CUDA (64.5). The auto-tuner therefore tries CUDA, CUDA with graph optimisation, and Vulkan on the Spark, per model.
    - **When:** with section 11.2's fused kernels (M10), and in the Spark track (M9).
    - **Gate:** keep each fusion only where it is at least 2% faster on that backend, with exact output.

3. **A smaller vocabulary for the guesser (FR-Spec).**
    - **What it does:** the draft only chooses among the most common tokens (about 32,000), which cuts the draft's output-layer work by 75%. The full model still checks every guess, so the output is unchanged.
    - **Measured:** 1.12× on average over EAGLE-2.
    - **Who gains most:** models with huge vocabularies, such as Qwen (about 150,000 tokens) and Kimi (about 160,000).
    - **Build:** add it to every draft head in 11.1. Code: https://github.com/thunlp/FR-Spec *(check its licence before copying)*.
    - **When:** M4.
    - **Gate:** at least 5% faster than the same guesser with the full vocabulary, exact output.

4. **Token Recycling: a guesser that needs no model.**
    - **What it does:** keeps a small table (under 2 MB) of each token's 8 most likely next tokens, filled from earlier steps, and builds guesses from it.
    - **Measured:** 2.03× against plain Hugging Face, vs 1.53× for prompt lookup on the same test. That baseline is weak, so expect much less against a tuned llama.cpp.
    - **Build:** add it to the 11.1 guesser list, next to prompt lookup and SuffixDecoding; the controller decides when to use it. No code has been published, so implement it from the paper.
    - **When:** M4.
    - **Gate:** the controller picks it on at least one bench task, and it beats prompt lookup there.

5. **Self-guessing by skipping layers, for dense models with no guesser (SWIFT).**
    - **What it does:** the model drafts by skipping some of its own layers, chosen on the fly. No training needed.
    - **Measured:** 1.3–1.6× on Llama-2 (Apache-2.0). It was below 1.0× at temperature 1 in another paper's tests, so the controller must be able to switch it off.
    - **When:** M4.
    - **Gate:** a dense model with no MTP head or draft model is at least 1.2× faster at temperature 0, with exact output.

6. **Sample on the graphics chip, but only inside the guessing loop.**
    - **Measured:** for normal one-user writing it gained nothing (RTX 4090 test run: 486 s vs 478 s with CPU sampling). Inside the MTP draft loop it gave +7% (Qwen3.6-35B-A3B, 2× RTX 5090) and +4% on Vulkan, because the drafts never leave the graphics chip.
    - **When:** M4.
    - **Gate:** guess-ahead at least 3% faster, exact output.

7. **Prepare the next step while the graphics chip works.**
    - **What it does:** SGLang and vLLM do the CPU's work for the next step while the graphics chip runs the current one.
    - **Measured:** NVIDIA estimates up to about 10% of llama.cpp's time per token is gaps where the CPU is busy and the graphics chip waits. Nobody has published a one-user measurement, so treat this as an experiment.
    - **When:** M10.
    - **Gate:** at least 3% faster writing, with exact output; otherwise drop it.

8. **Pick the fastest way to load a model, per machine.**
    - **Measured:** direct reads that skip the OS file cache loaded gpt-oss-120B on a Spark in 10.5 s, against 110 s for the first normal load (67 s for repeats). On an MS-S1 MAX, direct reads were the slowest option in the same test thread. The llama.cpp change (PR #18012) had not been merged when this was checked.
    - **Build:** auto-setup measures both ways and keeps the faster. This matters for the first reply after starting the app.

9. **On discrete NVIDIA cards, reuse llama.cpp's graphics-memory expert cache if it has been merged** (`--moe-cache-mib`, PR #29887; one secondary source says it was merged on 7 Oct 2026, so check).
    - **Measured:** 25.0 → 40.7 tok/s (RTX 4090) and 30.8 → 67.8 (5090) against llama.cpp's automatic `--fit`, but only 1–5% better than a hand-tuned `--n-cpu-moe`.
    - **Build:** if it is at least as fast as Kera's own, use it as Phase 9's graphics-memory tier instead of writing one.

10. **Machine checks in auto-setup:**
    - **macOS Low Power Mode** cut writing from 81.6 to 50.0 tok/s (−39%) in one test. Detect it and tell the user in plain words.
    - **ROCm version on Strix Halo:** prompt reading was 1,132 tok/s on ROCm 6.4.4 against 326 on 7.0.1 (Qwen3-8B BF16, Framework forum). Pin the version that measures fastest, and re-test each new release before switching.
    - **Macs with M5 or later** have matrix units in their graphics cores ("Neural Accelerators"). With MLX 0.30 or later on macOS 26.2 or later, they read prompts 3.3–4.1× faster than an M4. Use that path for prompt reading on M5 and later (MLX, or llama.cpp from build b10734 *(verify)*). An M4, like the owner's, doesn't have them.

11. **Measure prompt reading against the best kernels on the Spark.** One forum test (short prompts, unverified) got vLLM's first word in 131 ms against Ollama's 449 ms on gpt-oss-120b. Benchmark llama.cpp's prompt-reading kernels against CUTLASS and FlashInfer there, and port the faster path if the gap is real.

12. **A trained next-layer expert predictor, used only for prefetch (Edge0, arXiv 2609.18063).** A small add-on predicts the next layer's experts better than running its router early.
    - **Measured:** a 35B MoE at about 20 tok/s in 2.9 GiB on an M4 Pro, but that's its lossy full mode (12.2). One summary claims 4.8 → 8.6 tok/s on a 16 GB M2 *(unverified; another reviewer found its baseline implausible)*.
    - **Build:** used only to choose what to read early, it keeps output exact.
    - **When:** with Phase 3.
    - **Gate:** prefetch accuracy at least 5 points above the plain next-layer lookahead, and at least 5% faster, exact.

### 12.2 Opt-in speed modes that change the output slightly (Phase 11 rules)

Each needs a quality table (section 6) and a plain label, and stays off by default:
- **Fewer experts per word:**
    - Qwen3-30B-A3B at 4-bit on an M4 Max, 8 → 4 experts: 126.5 → 147.3 tok/s (+16.5%).
    - It gains more when experts stream from the SSD, because fewer are read.
    - ASET reported +10% on DeepSeek-V3 with no measurable loss.
- **A low-bit copy when an expert isn't in memory (HOBBIT):**
    - For less important experts that miss the cache, use a 2–4-bit copy already in RAM instead of waiting for the SSD.
    - Measured: 1.31–1.42× against llama.cpp, within 1 point on GSM8K.
    - If Shadow Bits is built, its base plane is that copy at no extra cost.
- **Read only the relevant parts of a long chat's memory:**
    - Quest (MIT): 1.74× (16-bit) to 2.23× (4-bit) at 32,000 tokens; passkey recall stayed at 100% with its 2,048-token budget.
    - DuoAttention: about 1.5× faster writing, after a one-off search for which attention heads need full memory.
- **Skip tiny values in dense models (TEAL, MIT; SpQt on Metal):**
    - At 50% skipped: 1.46–1.55× on an M2 with llama.cpp's Q4_K in a layout that allows skipping. Perplexity rose from 5.93 to 6.67 on Llama-3-8B, a noticeable loss.
    - At 25% skipped it gained almost nothing.
    - Dense models only; nobody has tested it on MoE experts.
- **Expert Deferral, discrete NVIDIA path only (KTransformers):**
    - Uses the lower-ranked experts' output one layer late, so the CPU and graphics chip work at the same time.
    - Measured: +33% on DeepSeek-V3, average quality drop at most 0.5%.
    - It needs separate graphics memory, so it doesn't apply to Mac, Strix Halo or the Spark.
- **Edge0's full mode:** the predicted experts replace the real routing, with a small repair adapter. Average score 79.2 vs 83.2, and AIME −6.1, so label the loss clearly.
- **FP4 maths on Blackwell for prompt reading:**
    - llama.cpp's native FP4 tensor-core path read prompts 19–26% faster on a Spark, with no gain in writing.
    - It rounds the in-between values to FP4, so it's lossy.
    - One forum report says GB10 lacks the FP4 instruction *(verify which is right before building)*.

### 12.3 Already in llama.cpp: inherit, don't rebuild

- **Splitting long-chat attention across the whole graphics chip** (flash-decoding / split-K): llama.cpp's CUDA, Metal and Vulkan code already does it. At most, tune the split for Strix Halo and the Spark.
- **Graph reuse, CUDA graphs, the CUDA fusions in 12.1 item 2, and chunked Gated-DeltaNet prompt reading** (Qwen3.5 and 3.6).
- **Upstream keeps getting faster on its own:** Metal writing on an M2 Ultra rose about 33% (94 → 125 tok/s on a 7B) between 2023 and August 2026. Keep the fork close to upstream (P2).

### 12.4 Checked and left out

- **Server tricks that only pay with many users:** chunked prefill, the XQA kernel, Marlin and Machete kernels, cache-aware load balancing.
- **Graphics-chip sampling for normal writing:** no gain with one user (12.1 item 6 uses it only inside the guessing loop).
- **Neural engines:**
    - Apple's Neural Engine ran an 8B model at about 9 tok/s against over 90 on the graphics chip (secondary source).
    - AMD's XDNA 2 NPU in hybrid mode wrote at 38 tok/s on Llama-3.1-8B, against about 46 for llama.cpp on Vulkan with a 7B on the same chip.
    - Neural engines are only useful for running a second model in the background.
- **CPU and graphics-chip splitting tricks (Fiddler, Expert Deferral) on shared-memory machines:** there is only one memory pool, so there is little to gain.
- **Medusa, Kangaroo, Pre-gated MoE and CATS:** beaten by things already in this spec, or they need retraining.
- **Attention sinks on their own:** passkey recall dropped to 0–10% in tests.
- **Code you can't reuse:**
    - LayerSkip has a non-commercial licence and also needs specially trained checkpoints.
    - WINA, R-Sparse and AdapMoE have no licence, so all rights are reserved.
    - Their ideas are fine to use; don't copy their code.
- **Weak baselines:** many papers compare against plain Hugging Face or Ollama. Expect half or less of their claimed gain against a tuned llama.cpp.

**Calibrate expectations:**
- A learned model for choosing which expert to drop from the cache (FlashMoE) beat plain LRU by only 7% on Qwen3-30B-A3B, and 22% on OLMoE. Phase 4's learned pins may gain as little on some models, which is why its gate is measured.
- SGLang's SSD Expert Pack team found that speculative prefetch could waste SSD bandwidth. Phase 3's gate must compare against no prefetch.

### 12.5 Steps and gates

| Step | What | Milestone | Gate |
|---|---|---|---|
| T0 | Trained expert predictor for prefetch (12.1 item 12) | M5 (Phase 3) | Prefetch accuracy ≥ 5 points better, ≥ 5% faster, exact |
| T1 | Grammar fast-forward with llguidance (12.1 item 1) | M2 | Identical output; JSON prompt at least 15% faster |
| T2 | FR-Spec, Token Recycling, SWIFT, graphics-chip sampling in the draft loop (items 3–6) | M4 | Each at or above its own threshold, exact; the controller is never slower than guessing off |
| T3 | Fusions per backend; CUDA vs CUDA graph-opt vs Vulkan on the Spark (item 2) | M10 (Spark: M9) | At least 2% per fusion, exact |
| T4 | Next-step preparation overlap (item 7) | M10 | At least 3%, exact |
| T5 | Load path, Low Power Mode warning, ROCm pin, M5 path, graphics-memory expert cache, Spark prompt kernels (items 8–11) | M0 (auto-setup checks); M8 and M9 (platform items) | The tuner keeps the faster option; warnings shown |
| T6 | Opt-in lossy modes (12.2) | M12 | Quality table, plain label, off by default |

### 12.6 Sources

- XGrammar (jump-forward at batch 1): https://arxiv.org/abs/2411.15100
- JSONSchemaBench (forced-token share, GBNF timing): https://arxiv.org/abs/2501.10868
- llguidance fast-forward: https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md
- llama.cpp CUDA fusions and GB10 results: https://github.com/ggml-org/llama.cpp/discussions/17621
- Vulkan fusion port: https://github.com/ggml-org/llama.cpp/pull/16977
- FR-Spec: https://arxiv.org/abs/2502.14856
- Token Recycling: https://arxiv.org/abs/2408.08696
- SWIFT: https://arxiv.org/abs/2410.06916
- Graphics-chip sampling: https://github.com/ggml-org/llama.cpp/pull/17004 and https://github.com/ggml-org/llama.cpp/pull/23287
- NVIDIA on llama.cpp CPU gaps and CUDA graphs: https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs
- Direct-read model loading: https://github.com/ggml-org/llama.cpp/pull/18012
- Graphics-memory expert cache: https://github.com/ggml-org/llama.cpp/pull/29887
- Metal speed over time and Low Power Mode: https://github.com/ggml-org/llama.cpp/discussions/4167
- M5 Neural Accelerators with MLX: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Spark engine comparison (vLLM vs Ollama first-word time): https://forums.developer.nvidia.com/t/measured-inference-benchmarks-on-a-single-dgx-spark-same-harness-across-ollama-llama-cpp-and-vllm-notes-data-published/379766
- Fewer experts on MLX: https://vllm-mlx.is-a.dev/guides/moe-top-k/
- HOBBIT: https://arxiv.org/abs/2411.01433
- Quest: https://arxiv.org/abs/2406.10774
- DuoAttention: https://arxiv.org/abs/2410.10819
- TEAL: https://arxiv.org/abs/2408.14690
- SpQt: https://arxiv.org/abs/2511.04477
- KTransformers Expert Deferral (SOSP 2025): https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
- Native FP4 in llama.cpp: https://github.com/ggml-org/llama.cpp/pull/17906
- Edge0 expert predictor: https://arxiv.org/abs/2609.18063
- llama.cpp checkpoints for hybrid models: https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache
- MTP vs DFlash on Strix Halo: https://sleepingrobots.com/dreams/beellama-dflash-strix-halo
- ROCm vs Vulkan on Strix Halo, March 2026: https://github.com/visorcraft/strix-halo-llm-perf
- FlashMoE: https://arxiv.org/abs/2601.17063
- SGLang SSD Expert Pack: https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/

---

## Appendix A: Shadow Bits (SB4) format and reference code

**Idea:** pick the 2-bit grid first, then add a 2-bit correction on a step four times finer. This is BitNest-style nesting (arXiv 2610.02800), which published the same idea, applied to SSD streaming: the base stays in RAM and the correction lives on the SSD.
- The base alone is a good 2-bit model.
- Base plus correction is exactly an ordinary uniform 4-bit model, so full mode can reuse normal 4-bit kernels.

**Per group of 64 weights**, grouped along the input dimension:

| Block | Lives on | Contents | Size |
|---|---|---|---|
| Base | RAM-side file | fp16 `s`, fp16 `m`, then 64 two-bit `q_hi` codes, 4 per byte (code i in bits 2·(i mod 4)) | 20 bytes (2.5 bits/weight) |
| Correction | SSD-side file | 64 two-bit `q_lo` codes, same packing | 16 bytes (2 bits/weight) |
| Total | | | 36 bytes = 4.5 bits/weight, the same as typical 4-bit formats |

Group size is a header field:
- **64:** base 2.5 bits, total 4.5;
- **128:** base 2.25 bits, total 4.25, slightly less accurate.

Test both in Phase 7b.

**Decoding:**

```
base = m + s * q_hi
full = (m - 0.375*s) + (s/4) * (4*q_hi + q_lo)      # q4 = 4*q_hi + q_lo is in 0..15
```

**File layout:**
- For each (layer, expert), and for each dense tensor, the base blocks are contiguous in the RAM-side file.
- The correction blocks are contiguous in the SSD-side file, aligned to 16 KiB.
- Header: magic `SB4\0`, version, group size, the `alpha` used, and a tensor table with offsets.

**`alpha`** picks what the scale is tuned for: 1.0 = best 2-bit base, 0.0 = best 4-bit full model. Start at **0.2**.

**Reference implementation** (tested; copy to `reference/shadow_bits.py` and port to C):

```python
"""Shadow Bits reference quantizer: SB4 = a 2-bit base plane + a 2-bit correction plane.

Per group of G weights (one shared fp16 scale s and fp16 offset m):
    base  = m + s * q_hi                              q_hi in {0,1,2,3}   -> a usable 2-bit model
    full  = base + (s/4) * (q_lo - 1.5)               q_lo in {0,1,2,3}
          = (m - 0.375*s) + (s/4) * (4*q_hi + q_lo)                       -> an ordinary uniform 4-bit grid
So the base plane alone is a 2-bit model (the draft), and base + correction is a plain
4-bit model with scale s/4 and offset m - 0.375*s. Full mode can therefore reuse normal
4-bit kernels: merge the planes into q4 = 4*q_hi + q_lo once, when an expert enters RAM.

Storage per group of 64: base plane 16 B + s,m 4 B = 20 B (2.5 bits/weight, lives in RAM)
                         correction plane 16 B   (2 bits/weight, lives on the SSD)
                         total 36 B = 4.5 bits/weight, the same as a typical 4-bit format.

alpha chooses what the scale is tuned for: 1.0 = best possible 2-bit base,
0.0 = best possible 4-bit full model, in between = a compromise.
"""
import numpy as np


def _f16(x):
    return x.astype(np.float16).astype(np.float32)


def _shrunk_range(W, c):
    lo, hi = W.min(1, keepdims=True), W.max(1, keepdims=True)
    centre, half = (lo + hi) / 2, (hi - lo) / 2 * c
    return centre - half, centre + half


def encode_sb4(w, group=64, alpha=0.5, shrinks=np.linspace(0.5, 1.0, 26)):
    W = np.asarray(w, dtype=np.float32).reshape(-1, group)
    var = W.var(1, keepdims=True) + 1e-12
    best = None
    for c in shrinks:
        lo, hi = _shrunk_range(W, c)
        m = _f16(lo)
        s = _f16(np.maximum((hi - lo) / 3, 1e-8))
        q_hi = np.clip(np.rint((W - m) / s), 0, 3)
        base = m + s * q_hi
        q_lo = np.clip(np.rint((W - base) / (s / 4) + 1.5), 0, 3)
        full = base + (s / 4) * (q_lo - 1.5)
        score = alpha * ((W - base) ** 2).mean(1, keepdims=True) / var \
            + (1 - alpha) * ((W - full) ** 2).mean(1, keepdims=True) / var
        cand = [score, m, s, q_hi, q_lo]
        if best is None:
            best = cand
        else:
            better = score < best[0]
            best = [np.where(better, a, b) for a, b in zip(cand, best)]
    _, m, s, q_hi, q_lo = best
    return {"m": m.astype(np.float16), "s": s.astype(np.float16),
            "q_hi": q_hi.astype(np.uint8), "q_lo": q_lo.astype(np.uint8),
            "shape": np.shape(w), "group": group}


def decode_base(sb):
    m, s = sb["m"].astype(np.float32), sb["s"].astype(np.float32)
    return (m + s * sb["q_hi"]).reshape(sb["shape"])


def decode_full(sb):
    m, s = sb["m"].astype(np.float32), sb["s"].astype(np.float32)
    return (m + s * sb["q_hi"] + (s / 4) * (sb["q_lo"] - 1.5)).reshape(sb["shape"])


def as_plain_q4(sb):
    """Merge both planes into an ordinary 4-bit layout: (codes 0..15, scale, offset)."""
    s = sb["s"].astype(np.float32)
    q4 = (4 * sb["q_hi"] + sb["q_lo"]).astype(np.uint8)
    return q4, s / 4, sb["m"].astype(np.float32) - 0.375 * s


def pack2(q):
    """Pack 2-bit codes, 4 per byte (the on-disk / in-RAM plane layout)."""
    q = q.reshape(-1, 4).astype(np.uint8)
    return (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(np.uint8)


def unpack2(b):
    b = b.astype(np.uint8)
    return np.stack([(b >> k) & 3 for k in (0, 2, 4, 6)], axis=1).reshape(-1)


def rtn(w, bits, group=64, shrinks=np.linspace(0.5, 1.0, 26)):
    """Baseline: plain round-to-nearest uniform quantizer with the best clipping per group."""
    W = np.asarray(w, dtype=np.float32).reshape(-1, group)
    levels = 2 ** bits - 1
    best_err, best_q = None, None
    for c in shrinks:
        lo, hi = _shrunk_range(W, c)
        m = _f16(lo)
        s = _f16(np.maximum((hi - lo) / levels, 1e-8))
        q = m + s * np.clip(np.rint((W - m) / s), 0, levels)
        err = ((W - q) ** 2).mean(1, keepdims=True)
        if best_err is None:
            best_err, best_q = err, q
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_q = np.where(better, q, best_q)
    return best_q.reshape(np.shape(w))
```

**Test results** on synthetic weights: 1.28 million values per distribution, group 64.

Error is shown relative to the weights' spread: 0 = perfect, 1.0 = useless. "Plain" means ordinary round-to-nearest with the best clipping per group.

| Weights | Plain 2-bit | Plain 4-bit | SB4 base (α=0.2) | SB4 full (α=0.2) | Full vs plain 4-bit |
|---|---|---|---|---|---|
| Gaussian | 0.1108 | 0.0073 | 0.1170 | 0.0083 | 1.13x |
| Laplace | 0.1628 | 0.0114 | 0.1760 | 0.0132 | 1.16x |
| Heavy-tailed (Student-t) | 0.1760 | 0.0126 | 0.1910 | 0.0149 | 1.18x |

**Other `alpha` values on Gaussian weights:**

| α | Base | Full |
|---|---|---|
| 1.0 | 0.1108 | 0.0141 |
| 0.5 | 0.1122 | 0.0106 |
| 0.0 | 0.1274 | 0.0073 |

**Checks that passed:**
- **Merge check:** merging the planes reproduces an ordinary 4-bit layout exactly.
- **Packing:** the 2-bit plane packing round-trips.
- **Layer-sized product** (1024×4096 matrix times a vector), relative error at α=0.2:
    - base 0.333;
    - full 0.092;
    - plain 4-bit 0.086.

**What this means:** at α = 0.2 the draft is only 5–9% worse than the best possible 2-bit, and full mode is only 13–18% worse than plain 4-bit.

**What it doesn't tell you:**
- These are synthetic weights. Real models have outliers, and K-quants beat plain 4-bit.
- Phase 7b on real models is the test that counts.

---

## Appendix B: guess-ahead controller (reference code)

Copy to `reference/controller.py` and port it into `src/spec`. In the engine:
- `new_bytes(k)` comes from the draft's predicted experts minus what is cached (MoE), or from the non-resident bytes (dense);
- `t_draft` and `t_check` come from timings measured over the last few steps;
- `new_ram_bytes(k)` (MoE models) comes from the extra experts the guesses use that are already in memory: they still have to be read, which is why guessing barely helped or hurt MoE models that fit (11.1 item 4);
- acceptance is per token: accepted ÷ (accepted + rejections), as `AcceptanceTracker` computes it. Every "acceptance" in this spec means this.

**Measure, don't only predict.** The `choose_k` formula is only the starting guess for k. After that, use the test-and-set method from Cascade (arXiv 2506.20675), the `KTuner` class at the end of the code:
- every so often, briefly try the k values next to the current one, and k = 0 unless guessing is clearly winning (then only every few tests, so probing costs about 3%);
- measure the real tokens per second and keep the best;
- test less often while guessing keeps losing.

The measured result always wins over the formula.

```python
"""Guess-ahead (speculative decoding) controller: pick how many tokens to guess, 0 = off.

The rule: choose the draft length k that gives the most tokens per second, where
    tokens per check = (1 - a^(k+1)) / (1 - a)          a = chance each guess is right
    seconds per check = k * t_draft + t_check + new_bytes(k) / ssd_speed
new_bytes(k) is what the check must read from the SSD for k guesses. For a dense model
it barely grows with k (every check reads the whole model anyway). For a MoE model it
grows with every extra guess, because each guess can pull in experts nobody else used.
That growth is what makes guessing backfire on MoE models that mostly fit in RAM.
"""


def tokens_per_check(a, k):
    return k + 1.0 if a >= 1.0 else (1 - a ** (k + 1)) / (1 - a)


def choose_k(a, t_draft, t_check, new_bytes, ssd_gb_s, k_max=8, new_ram_bytes=None, ram_gb_s=None):
    """Return (k, tokens_per_second). k = 0 means: don't guess, run a normal step.

    new_ram_bytes(k): for MoE models, the extra bytes a check reads from RAM because the
    k guesses use experts the next word alone wouldn't (experts already in memory still
    cost a read). Leave it None for dense models, where a check reads the same weights.
    This is only the starting guess; KTuner below measures and has the final say."""
    def seconds(k):
        s = k * t_draft + t_check + new_bytes(k) / (ssd_gb_s * 1e9)
        if new_ram_bytes is not None:
            s += new_ram_bytes(k) / (ram_gb_s * 1e9)
        return s

    best_k, best_rate = 0, 1.0 / seconds(0)
    for k in range(1, k_max + 1):
        rate = tokens_per_check(a, k) / seconds(k)
        if rate > best_rate:
            best_k, best_rate = k, rate
    return best_k, best_rate


class AcceptanceTracker:
    """Running estimate of a, the chance that each guessed token is right.

    A check stops at the first wrong guess, so the guesses after it are never tested.
    Count only tested guesses: a = accepted / (accepted + rejections), where a check has
    one rejection if it stopped early and none if every guess was accepted. Dividing by
    all guesses instead would under-count a (0.8 reads as about 0.59 at k = 4).
    Both counts are exponential moving averages."""

    def __init__(self, start=0.6, weight=0.1):
        self.accepted, self.tested, self.w = start, 1.0, weight

    @property
    def a(self):
        return self.accepted / self.tested if self.tested else 0.0

    def update(self, guessed, accepted):
        if guessed:
            rejected = 1 if accepted < guessed else 0
            self.accepted = (1 - self.w) * self.accepted + self.w * accepted
            self.tested = (1 - self.w) * self.tested + self.w * (accepted + rejected)
        return self.a


class KTuner:
    """Measure, don't only predict (the Cascade method, arXiv 2506.20675).

    choose_k() gives the starting k. After that, real speed decides: every `period`
    tokens, briefly run k - 1, k and k + 1, measure real tokens per second for each,
    and keep the fastest. k = 0 (no guessing) is tried too, except while guessing is
    clearly winning (at least `skip_zero_gain` times the last measured k = 0 speed);
    then it is re-checked only every `zero_every` tests, so probing costs little.
    While guessing keeps losing (k = 0 wins), test half as often each time, up to
    `max_period`; as soon as guessing wins, test at the normal rate.

    Use: k = tuner.next_k(); run one step with k guesses; tuner.record(tokens, seconds).
    """

    def __init__(self, k_start, k_max=8, trial_tokens=64, period=512, max_period=8192,
                 skip_zero_gain=1.2, zero_every=8):
        self.k, self.k_max = k_start, k_max
        self.trial_tokens = trial_tokens
        self.period = self.base_period = period
        self.max_period = max_period
        self.skip_zero_gain, self.zero_every = skip_zero_gain, zero_every
        self.since = 0          # tokens written since the last test
        self.tests = 0          # tests finished so far
        self.queue = []         # candidates still to try in this test
        self.results = {}       # k -> measured tokens per second in this test
        self.last = {}          # results of the previous test
        self.k0_rate = None     # last measured speed without guessing
        self.trial = None       # [k, tokens, seconds] for the candidate being tried

    def candidates(self):
        c = {max(self.k - 1, 0), self.k, min(self.k + 1, self.k_max)}
        best = max(self.last.values()) if self.last else None
        if (self.k0_rate is None or best is None or best < self.skip_zero_gain * self.k0_rate
                or self.tests % self.zero_every == 0):
            c.add(0)
        return sorted(c)

    def next_k(self):
        """The k to use for the next step."""
        if self.trial is None and self.since >= self.period:
            self.queue, self.results = self.candidates(), {}
            self.trial = [self.queue.pop(0), 0, 0.0]
        return self.trial[0] if self.trial else self.k

    def record(self, tokens, seconds):
        """Report how many tokens the last step produced and how long it took."""
        if self.trial is None:
            self.since += tokens
            return
        self.trial[1] += tokens
        self.trial[2] += seconds
        if self.trial[1] >= self.trial_tokens:
            k, t, s = self.trial
            self.results[k] = t / s
            if self.queue:
                self.trial = [self.queue.pop(0), 0, 0.0]
            else:
                self._finish()

    def _finish(self):
        best = max(self.results, key=self.results.get)
        if 0 in self.results:
            self.k0_rate = self.results[0]
        self.last, self.tests = dict(self.results), self.tests + 1
        self.k, self.trial, self.since = best, None, 0
        if best == 0:   # guessing lost: test less often
            self.period = min(self.period * 2, self.max_period)
        else:
            self.period = self.base_period
```

**Example decisions** (from the reference code, Mac with a 2 GB/s SSD):

| Situation | Chosen k | With controller | Guessing off |
|---|---|---|---|
| Dense 32B, Shadow Bits draft (75% accepted) | 8 | 0.70 tok/s | 0.22 tok/s |
| MoE mostly in RAM (35B), each guess adds 80% more expert reads | 1 | 4.93 tok/s | 4.88 tok/s |
| MoE much bigger than RAM (120B), each guess adds 50% | 2 | 1.81 tok/s | 1.48 tok/s |
| Same 120B, but each guess adds 90% | 0 (off) | 1.48 tok/s | 1.48 tok/s |

The last two rows show why Phase 6 must measure union growth. Guessing helps or not depending on that one number.

---

## Appendix C: speed-ceiling planner (reference code)

Copy to `reference/planner.py`. Example output:
- dense 32B on the Mac: 0.24 tok/s;
- dense 70B on the Minisforum: 5.33 tok/s;
- MoE 120B on the Mac at 50% hits: 1.48 tok/s;
- MoE 35B on the Mac at 75% hits: 4.97 tok/s;
- GLM-5.3-like on a DGX Spark at 82% hits: 3.67 tok/s with SSD reads and maths one after the other, 5.28 when they overlap (section 10).

```python
"""Speed ceiling: the best tokens/second physics allows for a model on a machine.

    without overlap: seconds per token = RAM time + SSD time
    with overlap:    seconds per token = max(RAM time, SSD time) + 0.15 * min(...)
      RAM time = GB read from RAM / RAM speed,  SSD time = GB read from SSD / SSD speed
Overlap means SSD reads happen while the maths runs (section 10). The 0.15 is the part that
never hides in practice; measure it and update it. Real speed always lands below the ceiling.
"""


def ceiling_tok_s(total_b, active_b, always_b, bits, usable_ram_gb, ram_gb_s, ssd_gb_s, hit_rate=None, overlap=False):
    """total_b: all weights (billions). active_b: weights used per token (= total_b for dense).
    always_b: weights used by every token (attention, shared experts, embeddings; = total_b for dense).
    bits: bits per weight including scales (4-bit formats are ~4.5). hit_rate: share of expert
    reads already in RAM (MoE only; None = estimate from how much of the model fits).
    overlap: True if SSD reads run while the maths runs."""
    gb = bits / 8
    always_gb = always_b * gb
    per_token_gb = active_b * gb
    if total_b == active_b:  # dense: every weight every token
        from_ssd = max(total_b * gb - usable_ram_gb, 0)
    else:
        routed_gb = (active_b - always_b) * gb
        pool_gb = (total_b - always_b) * gb
        cache_gb = max(usable_ram_gb - always_gb, 0)
        h = hit_rate if hit_rate is not None else min(cache_gb / pool_gb, 1.0)
        from_ssd = routed_gb * (1 - h)
    ram_t, ssd_t = per_token_gb / ram_gb_s, from_ssd / ssd_gb_s
    if overlap:
        return 1.0 / (max(ram_t, ssd_t) + 0.15 * min(ram_t, ssd_t))
    return 1.0 / (ram_t + ssd_t)


if __name__ == "__main__":
    mac = dict(usable_ram_gb=10, ram_gb_s=100, ssd_gb_s=2.0)
    mini = dict(usable_ram_gb=50, ram_gb_s=210, ssd_gb_s=5.0)
    print("dense 32B, Mac:", round(ceiling_tok_s(32, 32, 32, 4.5, **mac), 2))
    print("dense 70B, Minisforum:", round(ceiling_tok_s(70, 70, 70, 4.5, **mini), 2))
    print("MoE 120B/5B active, Mac, 50% hits:", round(ceiling_tok_s(120, 7.5, 3, 4.5, hit_rate=0.5, **mac), 2))
    print("MoE 35B/3B active, Mac, 75% hits:", round(ceiling_tok_s(35, 4.5, 2, 4.5, hit_rate=0.75, **mac), 2))
    spark = dict(usable_ram_gb=110, ram_gb_s=231, ssd_gb_s=11.0)
    # GLM-5.3-like: 744B total, ~40B used per word, ~21B always used
    print("GLM-5.3-like on a DGX Spark, 82% hits, no overlap:", round(ceiling_tok_s(744, 40, 21, 4.5, hit_rate=0.82, **spark), 2))
    print("GLM-5.3-like on a DGX Spark, 82% hits, overlap:   ", round(ceiling_tok_s(744, 40, 21, 4.5, hit_rate=0.82, overlap=True, **spark), 2))
```

---

## Appendix D: published numbers to sanity-check against

| What | Number | Source |
|---|---|---|
| GLM-5.2 (744B MoE) on a 128GB Ryzen AI Max+ 395, CPU only | 1.83 tok/s | Colibri README |
| GLM-5.2 on a 25GB laptop, cold | 0.05–0.1 tok/s | Colibri README |
| Kimi K3 (2.8T MoE), experts read at 6.3 GB/s | about 9.4 s per token | Colibri README |
| Two NVMe drives on separate controllers | +37.5% decode speed | Colibri `docs/multidisk.md` |
| GLM-5.2 expert choice predictable one layer ahead | 71.6% | Colibri README |
| Qwen3.5-397B MoE at 4-bit on an M3 Max with 48GB | 4.36 tok/s, about 71% file-cache hits | Flash-MoE |
| Dense 70B at 4-bit on a 128GB Ryzen AI Max+ 395 | about 4.5–5 tok/s | secondary report |
| M4 RAM speed | 120 GB/s | Apple / Wikipedia |
| M4 MacBook Pro 1TB SSD read | 2.8–3.3 GB/s | MacRumors thread |
| GLM-5.2 on a GB10 (DGX Spark), stock Colibri, exact | 2.39 tok/s timed, about 2.1 in chat, about 82% expert hits | Colibri issue #161 |
| NVIDIA DGX Spark Founders Edition internal SSD (Samsung 4TB, Gen5 x4) | 4.2–6.6 GB/s one-at-a-time; 11.1 GB/s with 16 MiB `O_DIRECT` reads; 11.4 GiB/s with 16 threads | NVIDIA forum; classmethod; StorageReview |
| GB10 graphics chip memory read | about 231 GB/s of 273 | Chips and Cheese |
| DeepSeek V4 Flash on a Spark, llama.cpp, experts moved to the CPU | 16.2 → 9.9 → 6.6 → 4.7 tok/s as more layers move | classmethod |
| ConnectX-7 RDMA between two Sparks | about 24.5 GB/s using both halves; 1.7 µs latency | Petronella Tech |
| Network NVMe storage box on a Spark (100GbE) | about 9.5 GiB/s read | StorageReview |
| GLM-5.2 on several Sparks | 2 Sparks, 1-bit: 15.9 tok/s; 3 Sparks: 21–26; 4 Sparks int4/int8: 32–36 | NVIDIA forum; noze.it; Context Studios |
| Best published single-Spark giant-model speeds | GLM-5.2 Q4_K 1.13 tok/s and IQ2_XXS 2.12 (Pulsar); GLM-5.3 NVFP4 0.81 (SparkLab) | classmethod; Hugging Face |
| LMCache, saved KV on local disk: Llama-3.1-8B on an L4, 15,376-token prompt | first word 6.314 s cold → 0.148 s warm (42.6x) | LMCache docs, local storage |
| GLM-5.x KV per token (MLA + sparse-attention indexer, 16-bit) | about 100 KB | my arithmetic from the model config |
| Colibri README, more models (read 14 Oct 2026; re-check against its latest README) | Qwen3.6-35B-A3B: 6.0 tok/s CPU, 9.9 Vulkan on a Radeon 780M (Ryzen 7 PRO 8700GE); 30.0 CUDA on an RTX 3070 8 GB. Qwen3-Coder-30B-A3B: 8.5–9.6 CPU, all experts in RAM. OLMoE: 22–23 CPU. MiMo-V2.6 Flash: 2.34–3.37 CPU. Qwen3.8-Flash-Next: 1.91–2.56 CPU. DeepSeek V4 Flash: 0.93 CPU (Ryzen 7 5800X, 32 GB); 1.5–1.6 CUDA (RTX 5080, 32 GB, two NVMe). GLM-5.2: 9.0–9.2 on 6× RTX 5090, all experts resident. Inkling: 0.25 (Ryzen 9 7900, 187 GB, RTX A6000). Qwen3.8-27B int4: 3.45 on a 16-thread CPU server | Colibri README |
| **The owner's own:** Nemotron 3 Nano 30B on his MS-S1 MAX (64 GB) | about 8 s before writing starts, on every message | The owner (record which app, thinking on or off, and prompt length before comparing) |

---

## Appendix E: reading list

**Engines:**
- Colibri: https://github.com/JustVugg/colibri
- llama.cpp: https://github.com/ggml-org/llama.cpp
- KTransformers: https://github.com/kvcache-ai/ktransformers
- Flash-MoE (397B on a Mac): https://github.com/danveloper/flash-moe

**Expert streaming:**
- Fate: https://arxiv.org/abs/2502.12224
- ProMoE: https://arxiv.org/abs/2410.22134
- SeqMoE: https://arxiv.org/abs/2609.12978
- SGLang SSD Expert Pack: https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/

**Guessing ahead with streaming:**
- SubSpec (dense): https://arxiv.org/abs/2509.18344
- SP-MoE: https://arxiv.org/abs/2510.10302
- S²-MoE: https://arxiv.org/abs/2608.15018
- DraftExpert: https://arxiv.org/abs/2607.24434
- SpecMoE: https://arxiv.org/abs/2604.10152

**Nested bits:**
- ELMoE-3D: https://arxiv.org/abs/2604.14626
- BitNest: https://arxiv.org/abs/2610.02800
- AnyBCQ: https://arxiv.org/abs/2510.10467
- Matryoshka Quantization: https://arxiv.org/abs/2502.06786

**Dense models:**
- LLM in a Flash: https://arxiv.org/abs/2312.11514
- TEAL: https://arxiv.org/abs/2408.14690
- GRINQH: https://arxiv.org/abs/2606.23419

**Draft heads:** EAGLE-3: https://arxiv.org/abs/2503.01840

**Models that fit (section 11):**
- DFlash block-diffusion drafter: https://arxiv.org/html/2602.06036v1
- llama.cpp MTP (PR #22673): https://github.com/ggml-org/llama.cpp/pull/22673
- MTP on a Spark (Qwen3.6-27B): https://forums.developer.nvidia.com/t/mtp-llama-cpp-a-look-at-qwen3-6-27b/370298
- MTP on Strix Halo: https://sleepingrobots.com/dreams/mtp-qwen36-strix-halo
- llama.cpp few-row Metal kernel (b11404): https://rohitai.com/blog/llama-cpp-b11404-metal-few-row-mma-speculative-decoding
- SuffixDecoding: https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/
- Lookahead decoding: https://arxiv.org/abs/2402.02057
- MoE speculation limits (MoESD): https://arxiv.org/html/2505.19645v4
- ECOSPEC (expert-sharing drafts): https://www.alphaxiv.org/abs/2607.12696.md
- Hazy Research megakernel: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Mirage Persistent Kernel: https://arxiv.org/abs/2512.22219v1
- Cohere MoE megakernel: https://cohere.com/blog/megakernels
- BaseRT (fused Metal kernels): https://arxiv.org/html/2607.00501v1
- Quant quality (KL divergence): https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks
- NVFP4 on Spark: https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute
- Strix Halo ROCm vs Vulkan: https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521
- Spark engine comparison: https://forums.developer.nvidia.com/t/measured-inference-benchmarks-on-a-single-dgx-spark-same-harness-across-ollama-llama-cpp-and-vllm-notes-data-published/379766
- DGPP (GB10 engine): https://forums.developer.nvidia.com/t/dgpp-a-gb10-optimized-c-cuda-inference-engine/383406
- QuantSpec: https://arxiv.org/html/2502.10424v1

**MoE fine-tuning:** ESFT (DeepSeek): https://github.com/deepseek-ai/ESFT *(verify)*

**DGX Spark:**
- SSD expert streaming on a Spark (Pulsar, read sizes, CPU-expert test): https://dev.classmethod.jp/en/articles/dgx-spark-pulsar-ssd-expert-streaming/
- Pulsar engine: https://github.com/giannisanni/pulsar
- GB10 memory and CPU/GPU contention: https://old.chipsandcheese.com/?p=37687
- StorageReview DGX Spark review (SSD read): https://storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops
- Network storage box: https://www.storagereview.com/review/luisuantech-gp-spark-review-nearly-10gb-s-of-plug-and-play-storage-for-the-dgx-spark
- Two-Spark RDMA bandwidth: https://petronellatech.com/blog/dgx-spark-cluster-bandwidth-what-400g-really-means/
- GPUDirect Storage not supported: https://forums.developer.nvidia.com/t/gds-support-for-dgx-spark/350328
- Memory loss on DGX OS 7.5.0 (KHO): https://forums.developer.nvidia.com/t/dgx-os-7-5-0-ota-uefi-5-36-0acum027-7-2-gib-less-ram-available-at-boot/383222
- Where Spark memory goes: https://forums.developer.nvidia.com/t/memory-creep-on-dgx-spark-where-your-128-gb-actually-goes-and-how-to-stop-it/364886
- Spark heat and clock capping: https://www.spheron.network/blog/dgx-spark-thermal-throttling-and-power-draw-under-load-2026/
- GLM-5.2 on 2 Sparks: https://forums.developer.nvidia.com/t/academic-glm-5-2-on-2x-dgx-spark-gb10-nodes-crazy-1-bit-ud-iq1-s-rpc-llama-cpp-256k-context-8-tok-s/374523
- GLM-5.2 on 4 Sparks: https://forums.developer.nvidia.com/t/fitting-a-high-quality-reap-less-glm-5-2-onto-4x-dgx-spark/374832
- SeqMoE (multi-layer prefetch, Belady-style eviction): https://arxiv.org/abs/2609.12978
- FlashMoE (learned eviction): https://arxiv.org/abs/2601.17063

**KV cache and KV reuse:**
- LMCache: https://github.com/LMCache/LMCache, docs https://docs.lmcache.ai (local storage: https://docs.lmcache.ai/kv_cache/local_storage.html)
- vLLM PagedAttention: https://arxiv.org/abs/2309.06180
- SGLang RadixAttention: https://arxiv.org/abs/2312.07104
- CacheBlend (non-prefix reuse): https://arxiv.org/abs/2405.16444
- MoE-Gen (layer-by-layer batching for offloaded MoE): https://arxiv.org/abs/2503.09716
- KIVI: https://arxiv.org/abs/2402.02750
