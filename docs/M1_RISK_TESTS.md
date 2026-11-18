# M1 risk tests (spec P5)

Five short experiments that decide later milestones. This page is the one-page "go / no-go" summary;
each test's details follow.

| # | Test | Decides | Result | Decision |
| --- | --- | --- | --- | --- |
| RT1 | Shadow Bits go/no-go, and which version | M7 | **No-go** (Qwen3-1.7B). Acceptance: SB4 as written 0.00, SB4 with BitNest's base 0.42, 4+4 0.85, a separate 0.6B draft 0.66. The full models are 10%, 4.5 times and 38 times further from 16-bit than plain weights of the same size | **M7 skipped** (N-004): no nested version meets the go rule, and plain weights plus a small draft predict about 1.8 times the speed (0.68 against 0.38 tok/s for a dense 32B on the Mac) |
| RT2 | Routing traces | M3 calibration, M6 | **Done for Nemotron 3 Nano** (38,558 words). Words reuse recent experts: an LRU cache of a quarter of the experts finds 63% of them. Learned pins lose to plain LRU. The next layer's router guesses 73% of its experts (90% with 12 guesses). The bigger MoE waits for the Minisforum | The planner uses the measured LRU curve (M3). Kera's expert cache is LRU with next-layer guesses (M5); learned pins fail Gate 4 and are off (N-002). Union growth (2.9x the experts for 4 words) sets M6's cost |
| RT3 | Overlap test (graphics chip work while reading the SSD) | 10.5 per machine | Waiting for hardware | Needs the Minisforum, the Mac and later the Spark |
| RT4 | Why Nemotron re-reads or reads prompts slowly on AMD | M2 | **Re-reading: no**, in this llama.cpp, unless the RAM prompt cache is off or the client edits old messages. Slow prompt reading: to profile on the Minisforum | M2 keeps the prompt cache on and sized, saves Mamba snapshots per chat (4b item 9), and measures the wait breakdown on the owner's own app. Lowering `--checkpoint-min-step` is not the fix |
| RT5 | ROCm on Windows for gfx1151 | 11.4 on the Minisforum | **Yes, with two caveats** | Build an optional HIP back-end for Windows; Vulkan stays the default until HIP passes the exactness tests and wins on speed on the Minisforum |

---

## RT1: Shadow Bits go/no-go, and which version

**Question (spec P5 item 1):** is a low-bit draft nested inside the full model (the base in memory guesses, the
correction streams from the SSD only to check) worth building as M7, and in which version: SB4 as written (2-bit
base + 2-bit correction, Appendix A's grid search), SB4 with BitNest's base (the same 2+2 layout, the base built
with GPTQ on calibration text and the correction a rounding of what is left), 4+4 (BitNest's own layout: a 4-bit
base inside an 8-bit model), or no nesting at all (ordinary 4-bit weights streamed, plus a small draft model from
the same family).

**How it was measured (14-15 Oct 2026, cloud, CPU, PyTorch in 32-bit floats):** `tools/risk_tests/rt1_shadow_bits.py`
on Qwen3-1.7B with Qwen3-0.6B as the separate draft. The spec asks for a 3-4B model (or 8B on the Minisforum); this
machine has 15 GB of memory, and 1.7B in 32-bit floats was the largest dense model that fit with room to quantize
it. The text: the 16-bit model's own greedy answers (up to 64 words) to 149 English opening messages from
OpenAssistant's validation split (all the suitable ones; the spec says 200), 9,467 answer words. Calibration text
for GPTQ: 200 assistant answers from other conversations. Each version's weights are quantized in place (every
linear layer of the decoder blocks; embeddings and norms stay as released), then measured on that same text:

- **acceptance:** how often the draft's top word is the full model's top word, word by word (what a greedy check
  accepts; Appendix B's accepted ÷ (accepted + rejections) per guess);
- **KL divergence** of the full model's next-word probabilities from the 16-bit model's (over the top 64 words
  and the rest), next to its plain counterpart's: round-to-nearest 4-bit with the best clipping per group of 64
  (for 4+4: 8-bit, groups of 128);
- **the planner's predicted speed** (`tools/risk_tests/rt1_predict.py`): Appendix C's machines and Appendix B's
  `choose_k` (k from 0 to 8), for dense models bigger than memory. Both a nested model and a plain one read about
  the same bytes per check (base + correction = the full model); what differs is the guesser: the base is big and
  reads all its bytes from memory for every guess, the small draft is cheap but may guess worse.

**Results:**

| Version | Guesser (in memory) | Full model | Acceptance | KL of the full model | KL of plain, same size | Ratio | Full model's top word = 16-bit's |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SB4 as written | 2-bit base | 4-bit (2+2) | 0.000 | 0.668 | 0.609 (4-bit) | 1.10 | 78.1% |
| SB4 with BitNest's base | 2-bit GPTQ base | 4-bit (2+2) | 0.424 | 2.757 | 0.609 (4-bit) | 4.53 | 50.0% |
| 4+4 (BitNest) | 4-bit GPTQ base | 8-bit (4+4) | 0.846 | 0.086 | 0.0023 (8-bit) | 38.2 | 91.1% |
| No nesting | 0.6B model, 4-bit | plain 4-bit | 0.663 | 0.609 | 0.609 | 1.00 | 78.5% |
| No nesting, 8-bit | 0.6B model, 4-bit | plain 8-bit | 0.674 | 0.0023 | 0.0023 | 1.00 | 98.1% |

The guessers alone: SB4's 2-bit base writes nothing right (top word right 0% of the time, KL 13.7); BitNest's 2-bit
base 29.5% (KL 3.7); the 4-bit GPTQ base 83.6% (KL 0.26, much better than plain round-to-nearest 4-bit's 0.61).

**The planner's predicted tokens per second** (Appendix C's machines, k chosen by Appendix B's `choose_k`; ceilings,
not measurements):

| Dense model bigger than memory | SB4 as written | SB4 with BitNest's base | 4+4 | No nesting (4-bit) | No guessing (4-bit) |
| --- | --- | --- | --- | --- | --- |
| 32B on the Mac (10 GB usable, 2 GB/s SSD) | 0.24 (guessing doesn't pay) | 0.38 (k = 3) | the 4-bit base doesn't fit | **0.68** (k = 8) | 0.24 |
| 123B on the Minisforum (50 GB, 5 GB/s) | 0.24 (guessing doesn't pay) | 0.36 (k = 3) | the 4-bit base doesn't fit | **0.70** (k = 8) | 0.24 |
| 70B on the Minisforum (8-bit for 4+4) | | | 0.87 (k = 8) | 0.63 with the 8-bit model (k = 8) | 0.21 (8-bit) |

On the 70B, 4+4 would be faster than plain 8-bit plus the draft, but its full model is 38 times further from the
16-bit model, so it isn't "equal or better quality"; a 70B in plain 4-bit fits in the Minisforum's memory and runs
much faster than either.

**Decision: no-go; M7 is skipped** (N-004). No nested version reaches acceptance 0.70 with a full model within 3% of
plain weights of the same size, and the no-nesting version (plain weights plus a small draft from the same family)
gives the most tokens per second, about 1.8-1.9 times the best nested version. So Phase 7, and Phase 11's Fast and
Balanced modes (SB4's base alone), are not built; guessing for dense models bigger than memory uses separate draft
models (M4, M6).

**Caveat:** the spec asked for 3-4B (or 8B on the Minisforum). Bigger models lose less at 2 bits, so the 2-bit
bases would guess better there, but every version misses its rule by a wide margin, and the speed gap comes from
how many bytes each guess reads, not from the model's size. Rerunning both scripts on an 8B model on the Minisforum
would confirm it.

---

## RT2: which experts each word uses (routing traces)

**Question (spec P5 item 2):** how often a word's experts are already in a cache of a given size, how
predictable the next layer's experts are, how many new experts each extra guessed word adds, and how
many experts a prompt touches. These replace the spec simulation's guesses (Zipf popularity).

**How it was measured (14 Oct 2026, cloud, CPU):** `kera trace` ran Nemotron 3 Nano 30B-A3B (Q4_0,
18 GB, 23 MoE layers of 128 experts, 6 used per word) over 64 real chats (OpenAssistant oasst1,
English, with the model's own chat template) and one agent session (Kera's own code, as a coding agent
would see it): 38,558 words in all. llama.cpp's evaluation callback logged each MoE layer's chosen
experts for every word, and the residual stream after each MoE layer for every 16th word.
`tools/risk_tests/rt2_analyze.py` computed the numbers below. Routing per word is the same whether the
words come in as a prompt or are written one by one, so this holds for writing too. The second, bigger
MoE the spec asks for (gpt-oss-120b or GLM-4.5-Air class) doesn't fit this machine's disk; it waits for
the Minisforum.

**Hit rate against cache size.** Each MoE layer keeps a share of its 128 experts in memory:

| Cache (share of each layer's experts) | Uniform guess | LRU | Pins learned on the first half, tested on the second | Best possible fixed pins | Pins (30% of the cache) plus LRU | Prompt pins (30%) plus LRU |
| --- | --- | --- | --- | --- | --- | --- |
| 5% | 5% | 21% | 8% | 12% | 10% | 17% |
| 10% | 10% | 41% | 16% | 22% | 36% | 41% |
| 25% | 25% | 63% | 34% | 42% | 59% | 63% |
| 50% | 50% | 83% | 60% | 69% | 81% | 83% |
| 75% | 75% | 95% | 82% | 88% | 94% | 94% |

Words keep reusing the experts of the words just before them, so plain LRU beats any fixed choice,
even the best possible one picked with hindsight. Pinning part of the cache only takes room from LRU.
"Prompt pins" keep, while a reply is written, the experts its prompt used most (each chat's first half
played the prompt and its second half the reply): level with LRU, never 10 points better.

**Next-layer lookahead.** The next MoE layer's own router (its norm, router and selection bias) applied
to the residual stream after the previous MoE layer finds **73%** of the experts it then chooses with 6
guesses, and **90%** with 12. Baselines: the layer's most used experts find 12%, the previous word's
experts 32%. Others report 79-97% (spec Phase 3).

**Union growth** (distinct experts per layer for k consecutive words, Phase 6):

| Words | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Experts | 6.0 | 10.2 | 13.8 | 17.1 | 20.2 | 23.1 | 25.7 | 28.2 |
| Times one word's | 1.00 | 1.70 | 2.31 | 2.86 | 3.37 | 3.84 | 4.28 | 4.70 |

Each extra guessed word adds about half a word's experts. Checking 4 guessed words at once reads 2.9
times the experts of one word, not 4.

**Experts a prompt touches:** 16 words use 34% of each layer's experts, 64 words 65%, 256 words 88%,
1,024 words 98%. Reading any real prompt means reading nearly every expert once.

**Decisions**

- **M3:** the planner's hit rate now comes from the measured LRU curve, not the share of experts that
  fit (half the experts in memory: 83% hits, not 50%).
- **M5:** Kera's expert cache is LRU. It reads the next layer's likely experts early from that layer's
  own router (Phase 3). Learned pins and prompt pins fail Gate 4 (10 points better than LRU on held-out
  prompts) on this model, so they are off by default (`docs/NEGATIVE_RESULTS.md` N-002). The code stays,
  so the gate can be re-run on the 120B-class model on the Minisforum.
- **M5, prompts:** a prompt reads nearly all experts, so prompt reading of a streamed model goes layer
  by layer over the whole prompt in large batches (D-008 risk 2), never small chunks.
- **M6:** guessing ahead on a streamed MoE reads 1.7x the experts for 2 words and 2.9x for 4. It pays
  only when the guesser's hit rate makes up for that; the planner works it out per model.

Data: the trace (565 MB) stays on the cloud machine; `docs/data/rt2_nemotron3_nano.json` keeps the
numbers above.

---

## RT5: ROCm (HIP) on Windows for the Minisforum's gfx1151

**Question:** can llama.cpp's HIP back-end run on Windows on Strix Halo (Radeon 8060S, gfx1151)? If
not, plan Vulkan only, or ask the owner about dual-booting Linux.

**Answer: yes.** Checked on 14 Oct 2026 in the pinned llama.cpp (b11619) and its release workflow:

- llama.cpp's own releases include a Windows HIP back-end built for gfx1151 (and gfx1150-1153,
  RDNA3/4), with AMD's "TheRock" ROCm 10.0.0 for Windows (`rocm[libraries,devel]` wheels from AMD).
  It ships as `ggml-hip.dll` plus three AMD runtime files (`amdhip64_7.dll`, `rocm_kpack.dll`,
  `amd_comgr.dll`) placed next to the program. With `GGML_BACKEND_DL`, Kera's Windows download can
  add it beside `ggml-vulkan.dll` without changing anything else.
- Lemonade's nightly llama.cpp builds also ship ROCm 7 builds for Windows on gfx1151, and AMD's
  developer playbook shows building llama.cpp with `GGML_HIP` and `GPU_TARGETS=gfx1151` on Windows.

**Caveats, both to check on the Minisforum before HIP is used for anything:**

1. **Correctness.** llama.cpp's own CI runs its gfx1151 tests (on Linux) with `HIP_LAUNCH_BLOCKING=1`,
   because without it batched work on integrated RDNA3.5 can give wrong results (a perplexity of
   about 88 instead of 9.4). That setting serialises the graphics chip's work and costs speed. Kera
   must run its exactness tests (section 6) on the HIP path, with and without it, before using HIP.
2. **The driver's own copy.** AMD's Adrenalin driver puts its own `amdhip64_7.dll` in System32, which
   Windows finds before anything on the PATH. llama.cpp works around this by shipping the matching DLL
   next to the program, and Kera's download must do the same.

Older reports (2025) also found HIP slower than Vulkan on gfx1151 for some matrix shapes, because of
rocBLAS tuning. The spec's plan (11.4) already measures both: ROCm where it is faster (prompt
reading), Vulkan for writing.

**Decision:** no dual boot is needed. M8 builds an optional HIP back-end for the Windows download, the
same way llama.cpp's release does. Vulkan stays the default; HIP is switched on per phase only after
it passes the exactness tests and wins on the Minisforum.

**Sources:** `third_party/llama.cpp/.github/workflows/release.yml` (job `windows-rocm`) and
`ci-self-hosted-cuda.yml` (job `gpu-rocm`); [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm);
[AMD developer playbook](https://developer.amd.com/playbooks/clustering-rpc-server).

---

## RT4: does llama.cpp re-read the whole chat on Nemotron?

**Question (spec P5 item 4):** Nemotron mixes Mamba and attention layers, so llama.cpp can only
reuse a chat from saved checkpoints. Is it re-reading the whole chat every message, and does lowering
`--checkpoint-min-step` fix the owner's 8-second wait?

**How it was tested (14 Oct 2026, cloud, CPU):** `tools/risk_tests/rt4_checkpoints.py` runs
`kera serve` (llama.cpp b11619's server) on the tiny Nemotron-H model with **Nemotron 3 Nano's own
chat template** (from NVIDIA's model repo), and plays a 30-40 turn chat the way a chat app does:
the whole history plus the new message each time, answers sent back, and the default of dropping
earlier answers' thinking (`truncate_history_thinking`). The server's own counters say, per turn, how
many prompt tokens were reused and how many were read again. The tiny model makes speeds meaningless;
the token counts and the server's checkpoint messages are what count, and the checkpoint logic is
the same for every hybrid model.

| Scenario | Prompt tokens sent | Read again | Full re-reads |
| --- | --- | --- | --- |
| Plain chat, default settings (40 turns, up to 16,000 tokens) | 363,010 | 4.5% (only each new message) | 0 |
| Thinking kept in the history | 271,188 | 6.3% | 0 |
| A side request after every turn (as chat apps do for titles and tags) | 138,765 | 7.3% | 0 |
| The same with `--checkpoint-min-step 0` | 138,765 | 7.3% | 0 |
| Side requests with the RAM prompt cache off (`--cache-ram 0`) | 138,765 | **100%** | **29 of 29** |
| An agent that keeps only the last 12 messages | 76,323 | **89%** | n/a: the start of the chat changes |

**What this means**

- In this llama.cpp version the server saves a checkpoint just before each answer and restores it
  next turn, so a normal chat is not re-read, even with Nemotron's template rewriting the previous
  answer. The checkpoint list filling up (32 entries, from about turn 13) didn't cause re-reads.
- Lowering `--checkpoint-min-step` changes nothing here. The spec's suspected quick fix doesn't apply.
- Two things do make it re-read everything: (1) when the RAM prompt cache is off or full and another
  request uses the slot in between (other chats, title generation), and (2) clients that edit or drop
  old messages. For (2), llama.cpp's `--cache-reuse` helps attention-only models but not hybrid ones.
- So the owner's 8-second wait is more likely slow prompt reading on AMD, thinking time before the
  first visible word, or one of the two cases above in his chat app. As a check on scale: 900 tokens at
  the 112 tok/s the spec quotes for Nemotron prompt reading on AMD is 8 seconds.

**Prompt reading on AMD (to profile on the Minisforum):** b11619's Vulkan back-end runs Nemotron
3 Nano's Mamba layers on the graphics chip (its scan supports state size 128 and head size 64), but
the scan steps through the prompt one token at a time, where Mamba-2's chunked form turns prompt reading
into large matrix multiplications. That makes it a strong candidate for the slowness. A search of
llama.cpp's issue tracker (14 Oct 2026) found no report matching it exactly. The profile on Vulkan and
ROCm, and the owner's own wait breakdown, need the Minisforum.

**Decision for M2:** keep the RAM prompt cache on in Kera's server and size it from the machine
profile; save Mamba snapshots per chat (section 4b item 9) so switching chats or restarting doesn't
re-read; show the wait breakdown (section 4c) so the owner's app is measured, not guessed; and
profile Mamba prompt reading on the Minisforum (a chunked scan for Vulkan is the likely fix, worth
offering upstream).
