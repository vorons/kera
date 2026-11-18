# Upstream code and patches

Kera's own code lives in `src/`, `app/` and `tools/`. Upstream code lives in
`third_party/`. Every change Kera makes inside upstream code is listed here,
so updates stay easy (P2).

## llama.cpp

- **Where:** `third_party/llama.cpp`, added with `git subtree add --squash`
  (only the code, not its history; never a GitHub fork, P8 item 3).
- **Upstream:** https://github.com/ggml-org/llama.cpp (MIT licence)
- **Pinned:** release tag `b12500`, commit
  `aaa807c568f2ffdb50d51455658ee91f5cbeedc1` (18 Nov 2026).
- **Kera's hooks inside it:** seven (1-4 and 6-8; hook 5 was removed with Kera's lossy modes, D-034), plus small fixes,
  each change marked `[Kera]` in the code.

### Hook 1: a second tier for the server's prompt cache (M2, spec 4b item 5)

Lets `kera serve` keep chat states on the SSD, so going back to an old chat or
restarting doesn't mean reading the whole chat again. With no store set, the
server behaves exactly as upstream.

| File | Change |
| --- | --- |
| `common/prompt-store.h`, `common/prompt-store.cpp` (new) | The `common_prompt_store` interface (`save`, `load`, and since 9 Nov `set_identity`) and `common_set_prompt_store()`. They live in the shared `common` library so the program and the server library (a DLL on Windows) use the same pointer. At the b12500 update, `common_saved_prompt` took upstream's `common_state_data` and `std::list<std::shared_ptr<const common_prompt_checkpoint>>` types, since b12500 changed `server_prompt::checkpoints` the same way. Kera's disk adapter in `app/serve.cpp` converts at that boundary. |
| `common/CMakeLists.txt` | Adds the two files to `llama-common`. |
| `tools/server/server-task.h` | Includes `prompt-store.h`. |
| `tools/server/server-slot-sched.cpp` | `server_prompt_cache::load()`: when nothing in RAM matches, ask the store for the saved prompt sharing the longest start, and restore it the way a RAM entry is restored. Upstream moved `server_prompt_cache` here from `server-task.cpp` in b12500; the hook moved with it. |
| `tools/server/server-context.cpp` | `prompt_save()` hands a copy of each state put in the RAM cache to the store; a new `prompt_store_save()` saves the chat's state when a reply has finished, so it survives a restart. Prompts with images aren't saved. Since N-005: `release()` only marks the chat (`kera_store_pending`) and the save runs once the server is idle (`on_idle` below), or before the slot moves to another chat or is cleared; the store gets a chat's newest 2 checkpoints only (`kera_newest_checkpoints`; a hybrid model's checkpoint holds its whole recurrent state, and the server keeps up to 32). Since 9 Nov (D-033): `kera_store_identity()`, called once the model and context are loaded, tells the store what besides the model file decides a saved state's numbers: the settings as parsed (rope, YaRN, cache types, flash attention, kv overrides, adapter and control-vector scales, the guessers, and whether a LongRoPE model's context per chat is over its original one, which picks its long rope factors) and the files they name; `kera_store_check_lora()` turns the store off for the run when a request or `POST /lora-adapters` changes the adapter scales the server started with (states computed with others would otherwise be saved under its identity). |
| `tools/server/server-queue.h`, `tools/server/server-queue.cpp` | `on_idle()`: work run on the server thread once no task has come for 250 ms, and once more before the loop ends. The save above uses it, so a chat's next message never waits for it (N-005). |
| `src/llama-context.cpp` | Kera used to batch the pieces of a saved sequence state with its own `llama_io_pieces_transfer()` (one span copy per tensor group, with heuristics to avoid far-apart rows). Upstream b12500 added `ggml_backend_tensor_get_n()`/`set_n()` instead: a native batched gather/scatter on CUDA, Metal, Vulkan and others, with a grouped and 2D-strided fallback elsewhere. Kera's copy was dropped at the update; the two `llama_io_*_host` classes now call upstream's API. |

Since 0.1.1 (the bug sweep, N-014): `prompt_clear(save_pending)` drops a waiting save instead of saving it where
the slot is cleared because its memory no longer matches its tokens (a decode error, a failed slot restore) or on
request (slot erase); and a slot picked by similarity for another chat with the same start saves the waiting chat
first unless the new task holds nearly all of it (90%), so its own end isn't lost.

Kera's side: `app/serve.cpp` (the adapter and the cache identity) and
`src/kvstore/prompt_disk.cpp` (the files). Tested by
`tests/integration/test_kv_store.py` on a hybrid Mamba model and an MLA model.
Worth offering upstream as a generic "prompt cache on disk".

### Hook 2: a controller for the guess length (M4, spec 11.1 and Appendix B)

Lets Kera pick, at every step, how many tokens llama.cpp's speculative decoding
guesses (0 = don't guess this step), from the speed it measures. Without a
controller set, the server keeps its fixed draft length. (Since 0.1.1, `common/speculative.cpp` also tells a
guesser how far its draft was cut only when a controller is set: without one,
ngram-mod judges its guesses as upstream does.)

| File | Change |
| --- | --- |
| `common/spec-controller.h`, `common/spec-controller.cpp` (new) | The `common_spec_controller` interface (`draft_len`, `accepted`, `reset`) and `common_set_spec_controller()`, in the shared `common` library for the same reason as hook 1. |
| `common/CMakeLists.txt` | Adds the two files to `llama-common`. |
| `tools/server/server-context.cpp` | Before a new draft, the draft length becomes `min(upstream's limit, controller's choice)`; after the check, the controller hears how many guesses were accepted; `release()` resets the slot's measurement. |
| `common/spec-controller.h` | `guesser_len()`: the controller is asked about each guesser at every step (a model guesser before it runs: how far, 0 = skip it; an n-gram guesser after it guessed: how many words to keep). The default keeps upstream's behaviour. |
| `common/speculative.cpp` | `common_speculative_draft()` asks `guesser_len()` per guesser and sequence (a skipped sequence goes on to the next guesser); MTP and EAGLE-3 stop at the step's own limit, as draft-simple already did, instead of guessing words that are then cut; `set_draft_len()` tells a guesser its draft was cut, so `ngram-mod` judges its acceptance on the words actually checked (it used to reset itself after five "low acceptance" drafts that were only cut short). |

Kera's side: `guess_controller` in `app/serve.cpp` and the guesser switch
(`src/spec/kr_guess_switch.h`, one Appendix B KTuner per guesser and chat;
decision D-014). Tested by `tests/cpp/test_guess_switch.cpp` and
`tests/integration/test_guess_controller.py` (the same words with and without
guessing: Kera's own choice of guessers, two guessers at once, fixed lengths,
and with the experts streamed).

### Hook 3: streamed weights (M5, spec Phases 2-3, D-008 and D-012)

Lets Kera keep a MoE model's routed experts out of memory at load time and bring
each layer's chosen experts in just before the layer runs. With no hook set (or
the hook declining a model), loading and running are exactly upstream's.

| File | Change |
| --- | --- |
| `include/llama-stream.h` (new) | The `llama_stream_hook` struct (`begin`, `mode`, `mapped`, `end`, `cb_eval`) and `llama_set_stream_hook()`. For each weight the hook answers: load as usual, map without reading (`LLAMA_STREAM_MAP`), or place in the hook's own buffer type and don't read (`LLAMA_STREAM_BUFT`). `llama_stream_tensor_get()`: from the hook's `cb_eval`, reads a tensor on the queue of the backend that computed it, not on the separate queue `ggml_backend_tensor_get` uses (`docs/NEGATIVE_RESULTS.md` N-012). |
| `src/llama-stream.cpp`, `src/llama-stream-impl.h` (new) | The global hook, and the chained eval callback that calls the hook's callback before the context's own (each only for the tensors it asked for). The chain keeps the context's scheduler, which `llama_stream_tensor_get()` asks for the tensor's backend. |
| `src/CMakeLists.txt` | Adds `llama-stream.cpp`. At b12500, `ggml_backend_buffer_i` gained `set_tensor_n`, `get_tensor_n` and `cpy_tensor_n`; Kera's stream buffer type in `src/stream/session.cpp` sets them to `nullptr` (its own per-piece access stays correct). |
| `src/llama.cpp` | `llama_model_load()` asks the hook's `begin` (with the model path, and whether this is the memory-fit pass), gives the loader the hook, and calls `end` with the loaded model (or NULL). A model that streams anything keeps a copy of the hook. |
| `src/llama-model-loader.h/.cpp` | Records each file's path. `create_tensor()` asks the hook for each weight: MAP weights become lazy weights (mapped, never read at load, never prefetched or locked); BUFT weights go to the hook's buffer type. `init_mappings()` leaves the BUFT weights' data out of the mapping's prefetch. `load_all_data()` skips BUFT weights and tells the hook where each MAP weight was mapped. `lazy_read::add()` takes a `force` flag for the hook's weights. |
| `src/llama-model.h` | `llama_model::stream_hook`, and `stream_buft`, the buffer type its streamed weights are in (the loader's `streamed_buft`). |
| `src/llama-context.h/.cpp` | A context of a model that streams installs the chained eval callback, and gives it the scheduler (`sched_new()`). `sched_copy_experts()` pads a streamed weight's run of experts with zeros instead of reading the next expert, which may not be in memory (on Windows that read faulted: N-013), and warns once if a streamed weight would be copied whole. With `KERA_STREAM_CHECK` set, it prints which experts it read were chosen (N-012). For N-012's tests: `KERA_STREAM_COPY=stage` copies a streamed weight's experts through the chip's own pinned host memory (kept in `copy_experts_info::staging`, at most 64 MB at a time, waiting for the chip before it is written again), and `KERA_STREAM_VERIFY=1` reads every copy back from the chip and says where it differs from the mirror, or that a chosen expert's mirror is all zeros or not in memory at all (`llama_stream_readable()` in `src/llama-stream.cpp`), that is, never brought in. |

Kera's side: `src/stream/` (the session, the buffer type, the expert cache and
the mirror of the model's files). Tested by `tests/cpp/test_stream_cache.cpp`
and by the tiny-model exactness check with streaming on, which must give
bit-identical results (CI). Upstream's GPU expert cache (`--moe-cache-mib`)
works on top of it unchanged: it uploads from the streamed weights.

### Hook 4: the model card's sampling per request (M2, spec 4c item 2)

Some model cards recommend different sampling for thinking on, thinking off and
tool calls (Nemotron 3 Nano: one for each). The server applies one set of
defaults to every request, so Kera fills in the card's values per request,
only for the fields the request doesn't set. Without a hook set, the server
keeps its defaults (the command line's, or the GGUF file's
`general.sampling` values).

| File | Change |
| --- | --- |
| `common/request-defaults.h`, `common/request-defaults.cpp` (new) | The `common_request_defaults` interface (`sampling(thinking, tools, defaults)`, and since 9 Nov `leading_messages(request_has_system)`; the defaults also carry repeat, presence and frequency penalties and stop words) and `common_set_request_defaults()`, in the shared `common` library like hooks 1 and 2. |
| `common/CMakeLists.txt` | Adds the two files to `llama-common`. |
| `tools/server/server-common.cpp` | At the end of `oaicompat_chat_params_parse()` (OpenAI and Anthropic chat requests): asks the hook, and sets `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty` and `frequency_penalty` where neither the request nor the user's server settings set them, and adds its stop words where the request gives none. Whether the model thinks is taken from this request's prompt (`thinking_end_tags` not empty), not the request's flag alone: a template that can't think (Qwen3-2507 Instruct) ignores `enable_thinking`, and so does its card (9 Nov). Before the chat is read: the hook's leading messages (an Ollama Modelfile's SYSTEM where the request has no system message, then its MESSAGE pairs) go before the request's own (9 Nov, D-033). |
| `common/common.h`, `common/arg.cpp` | Two more bits in `common_params_sampling_config`, `..._PENALTY_PRESENT` and `..._PENALTY_FREQ`, set by `--presence-penalty` and `--frequency-penalty` (and their `LLAMA_ARG_*` variables and presets), so hook 4 never overrides them (9 Nov). |
| `tools/server/server-common.h`, `tools/server/server-context.cpp` | `server_chat_params::user_sampling_config`: the sampling fields the user set for the server (command line, `LLAMA_ARG_*`, a preset; llama.cpp's own `params.sampling.user_sampling_config`), copied in when the chat parameters are built, so the hook never overrides them (9 Nov, D-033). |

Kera's side: `model_request_defaults` in `app/serve.cpp`, the per-family values
in `src/tune/model_defaults.cpp` (sources in `docs/DECISIONS.md` D-013). Tested
by `tests/integration/test_model_defaults.py`.

### Hook 5: removed (D-034)

Hook 5 copied a passage a chat had already read to a new place in its memory, for the lossy `--lossy-reuse` mode.
It was removed with Kera's lossy modes on 9 Nov 2026: `common/prompt-reuse.{h,cpp}`, `llama_memory_seq_cp_shift()`
(`include/llama-reuse.h`, `src/llama-reuse.cpp`), `llama_kv_cache::seq_cp_shift()` and its use in
`tools/server/server-context.cpp` are gone. `src/llama-kv-cache.{h,cpp}` are back to upstream's; `include/llama-reuse.h`
and `src/llama-reuse.cpp` now hold hook 7 only, and `server-context.cpp` keeps hooks 1, 2, 6, 7 and (since 9 Nov) 8.

### Hook 6: each reply's shape, for the choice of back-end (M8, D-022)

For `kera serve`'s choice between Vulkan and the ROCm add-on: when a reply finishes, the server tells an observer
how many prompt tokens came from the cache, how many it read and wrote, and how long reading and writing took. Only
counts and times, never text. Without an observer set, the server behaves exactly as upstream.

| File | Change |
| --- | --- |
| `common/reply-observer.h`, `common/reply-observer.cpp` (new) | The `common_reply_observer` interface (`reply(held, read, written, t_read_ms, t_write_ms)`) and `common_set_reply_observer()`, in the shared `common` library like hooks 1, 2 and 4. |
| `common/CMakeLists.txt` | Adds the two files to `llama-common`. |
| `tools/server/server-context.cpp` | `release()`: after a completion task, hands the slot's `n_prompt_cached`, `n_prompt_processed`, `n_gen` and its two times to the observer. |

Kera's side: `reply_notes` in `app/backend_auto.cpp` (one line per reply in `replies/` in Kera's folder), the
choice in `src/planner/backend_pick.h`. Tested by `tests/cpp/test_backend_pick.cpp` and by `kera serve --backend
auto` on the tiny models (the CPU against Mesa's software Vulkan).

### Hook 7: several chats sharing the chip (D-023)

For `kera serve --multi-user` (the settings `serve.short_first`, `serve.read_share`, `serve.read_step_s` and
`serve.share_prefix`, off until measured). Upstream reads waiting prompts in slot order, each as far as the batch
allows, so one long document holds up every other chat's new message, and each of its steps holds up their writing.
Without a scheduler set, the server behaves exactly as upstream.

| File | Change |
| --- | --- |
| `common/batch-scheduler.h`, `common/batch-scheduler.cpp` (new) | The `common_batch_scheduler` interface (`plan(prompts, n_writing, n_batch)`: the order to read waiting prompts in and the most prompt tokens the step takes; `step(n_prompt, n_writing, ms)`; `share_min_tokens()`) and `common_set_batch_scheduler()`, in the shared `common` library like hooks 1, 2, 4 and 6. |
| `common/CMakeLists.txt` | Adds the two files to `llama-common`. |
| `tools/server/server-context.cpp` | `pre_decode()`: before batching waiting prompts, asks the scheduler for their order (each with the tokens it has left; for one not started yet, beyond what its slot holds) and for the step's prompt budget while other chats write; a completion's prompt keeps to the budget (prompts that need a whole batch don't). `update_slots()`: times each step for the scheduler. A new prompt whose start another slot is reading right now (or will read first, having come earlier) waits a step at a time until that start is read, and counts what it will share as read when the prompts are ordered. At a prompt's start: a longer start it has in common with another slot's memory (tokens already read, never ones waiting in this batch) is shared rather than read. In plain attention memory that is `seq_cp`, a link to the same cells in the unified memory rather than a copy; in a hybrid memory (attention plus a recurrent state, such as Nemotron-H's) it goes up to a checkpoint the other slot made (at a user message's start, or near its prompt's end), with `llama_memory_seq_share_hybrid` below, and the slot takes that checkpoint as its own. Only where nothing can change the shared cells: one memory for all chats (`kv_unified`), no draft model, no sliding window, no context shift, the same adapters. Cache-reuse shifting (`n_cache_reuse`) moves a shared cell for every chat holding it, so it is off while sharing is on. |
| `tools/server/server-task.h`, `tools/server/server-schema.cpp` | A request field `priority` (lower is sooner, as vLLM's; 0 by default), used only with a scheduler set: it comes first in the order prompts are read in. |
| `tools/server/server-queue.cpp` | `pop_deferred_task()`: with a scheduler set, of the requests waiting for a free slot the first with the lowest priority goes next, not simply the first. |
| `src/CMakeLists.txt` | Adds `llama-reuse.cpp`. |
| `include/llama-reuse.h`, `src/llama-reuse.cpp` | `llama_memory_seq_share_hybrid()`: gives a sequence another's first n positions in a hybrid memory: clears it, links the attention cells for [0, n), and loads the recurrent state from a partial state saved at n (a checkpoint). A plain copy can't do it: `seq_cp` on the recurrent part links the other sequence's current state, not the one at n. Not for the sparse-attention memory (`llama_memory_hybrid_idx`), which copies whole sequences only. |

Kera's side: `share_scheduler` in `app/serve.cpp`, the policy in `src/planner/chat_share.h`. Tested by
`tests/cpp/test_chat_share.cpp` (a simulated chip) and `tests/integration/test_multi_user.py` (two servers, with and
without `--multi-user`: the shared start, the order, and the same answers but for near ties).

### Hook 8: settings Kera fills in once llama.cpp has read them all (9 Nov, D-033)

`kera serve` finds a vision model's image reader next to it (D-027). It used to add `--mmproj` to the command line,
which overrides llama.cpp's other sources of settings: a reader the user named with `LLAMA_ARG_MMPROJ` or in
`config.ini` was replaced by Kera's pick, and one turned off with `--no-mmproj-auto`, `LLAMA_ARG_MMPROJ_AUTO=0` or
`config.ini` was attached anyway (N-018). Now llama.cpp reads every source first, and Kera fills in the reader only
where none names one or turns it off. Without a fill function set, the server behaves exactly as upstream.

| File | Change |
| --- | --- |
| `common/prompt-store.h`, `common/prompt-store.cpp` | `common_set_params_fill()` / `common_get_params_fill()`: a function the server calls with the settings it will use, in the shared `common` library like hook 1's store. |
| `tools/server/server-context.cpp` | `load_model()`: calls the fill function first, before it copies the settings and loads the model (after the command line, `LLAMA_ARG_*` variables, `config.ini` and presets are read, and after `-hf` downloads). |

Kera's side: `kera_fill_params` in `app/serve.cpp` sets `mmproj.path` only when `mmproj.path` and `mmproj.url` are
empty, `no_mmproj` is off and no Hugging Face repo is named, and turns the saved chats off for a model that reads
images whichever source named its reader (as `--mmproj` did before). Tested by `tests/integration/test_vision.py`
(`LLAMA_ARG_MMPROJ_AUTO=0` leaves the reader out; `LLAMA_ARG_MMPROJ`'s reader is the one used).

### Fixes outside the hooks

- **A guessed word's probabilities (9 Nov, D-033; upstream #27196, merged 20 Oct 2026, after b11619; port dropped at the b12500 update).** With guessing
  ahead, `tools/server/server-context.cpp` sent every accepted guessed word with probability 1 and no alternatives
  (`// TODO: set result.probs`), so a request asking for `n_probs`/`logprobs` got fake numbers whenever Kera guessed
  (on by default; stock llama.cpp guesses only when asked). Ported upstream's fix: the draft's batch indices are kept
  (`spec_i_batch`) and each accepted word gets the model's own probabilities from them, as without guessing. Kera adds
  one line upstream doesn't have: a request for post-sampling probabilities (`post_sampling_probs`), which a guessed
  word can't carry, doesn't guess (a replay in progress finishes). Upstream b12500 now carries both fixes, so Kera's port was dropped at the update.

| File | Change |
| --- | --- |
| `tools/server/server-context.cpp` | After a partly right guess on a model that can't drop words from its memory (hybrid and recurrent ones), the server restores the state and reads the accepted words again. With greedy decoding it checked them again, and on a MoE model on a graphics chip the batch's make-up can round a near tie the other way, so it restored and replayed forever (N-009). A replay now accepts its words as chosen, greedy too, as upstream already did with a temperature. Worth offering upstream. |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `ggml-vulkan-types.h` | A pipeline cache file: with `GGML_VK_PIPELINE_CACHE` set (a path without its ending; one file per device), the Vulkan back-end loads it when the device starts, builds every shader through it, saves it at most every 5 s while new shaders appear (checked once per graph) and at the end. Upstream passes no cache, so only the driver's per-program cache kept built shaders, and a Kera in a new folder rebuilt them all on its first requests (N-011). Kera sets the variable to a file in its folder (the setting `gpu.shader_cache`, on by default since it measured 3.2 s to 0.18 s for a new folder's first word, D-021). Worth offering upstream. |
| `common/arg.cpp` | On Windows, `common_params_parse()` swaps the `argv` it is given for the process's own command line (read as UTF-8) when the two are the same length. Kera's commands rewrite their options before passing them on, and `kera serve --no-auto-guess` happened to be as long as its command line, so llama.cpp read `serve` as an option and stopped. Now the swap needs the words in plain ASCII to match too (they read the same in any code page). Kera's `main()` reads its own words as UTF-8, so nothing is lost. Worth offering upstream (N-008). |
| `ggml/src/ggml-backend.cpp` | An eval callback that returns false now cancels the graph compute (`GGML_STATUS_ABORTED`, which `llama_decode()` reports as 2), as `ggml-backend.h` says it does. The scheduler only left the current split and computed the next ones from data that was never written, so when Kera's expert reads failed the model gave garbage instead of an error (N-012). Tested by `tests/cpp/test_sched_cancel.cpp`. Worth offering upstream. |
| `tools/tts/tts.cpp`, `tools/tts/main.cpp`, `tools/tts/CMakeLists.txt` | The speech tool's `main()` is now `llama_tts()`, built as the library `llama-tts-impl` that `kera speak` calls (D-028); `main.cpp` is a two-line `main()` for `llama-tts` itself. The WAV is opened with `ggml_fopen` instead of `fopen`, so a name outside the Windows code page (`C:\Users\José\hi.wav`) works. Kera-only (the first part); the second is worth offering upstream. |
| `tools/server/server-http.cpp` | On Windows the server's socket uses `SO_EXCLUSIVEADDRUSE` instead of `SO_REUSEADDR`, which there lets a second server bind a port one is already listening on (requests then went to either at random: the app's model and a `kera serve` both on 8080). Elsewhere unchanged. Worth offering upstream. |

### How to update it

The upstream history is large, so fetch only the one commit, then merge it as a
squashed subtree:

```sh
git fetch --depth=1 --no-tags https://github.com/ggml-org/llama.cpp refs/tags/<tag>
git subtree merge --prefix=third_party/llama.cpp FETCH_HEAD --squash \
    -m "Update llama.cpp to <tag>"
```

Then run the full CI, and update the pinned tag and commit above.
`upstream.yml` does this monthly on the branch `upstream-llama-cpp`.

## stable-diffusion.cpp (the image engine, D-027)

- **Upstream:** https://github.com/leejet/stable-diffusion.cpp (MIT), commit
  `9234ad822a376a4724638467736118ed61d68ee9` (7 Nov 2026), with its own ggml fork
  https://github.com/leejet/ggml at `135ce1e0e3f2b0693d3de93d4f6b5fa0a1947820`.
- **Copied, not a subtree:** `CMakeLists.txt`, `LICENSE`, `cmake/`, `include/`, `src/`, `examples/` (without the
  web UI's `server/frontend` submodule) and `thirdparty/` (without the libwebp, libwebm and cutlass submodules; WebP
  and WebM output are off), and the ggml fork's `CMakeLists.txt`, `LICENSE`, `cmake/`, `include/`, `src/` and
  `ggml.pc.in` in `ggml/`. Its sample pictures, docs, scripts and the ggml fork's tests and examples are left out.
- **Changes:** `src/core/probe/backend_probe.{h,cpp}`, `flash_attn.cpp`, `conv_direct.cpp`: the engine picks flash
  attention and direct convolutions by timing both on the graphics chip when it starts, and a close race can go
  either way from one start to the next, changing the picture for the same seed. With `SD_PROBE_CACHE` naming a
  file (Kera sets it to `image-engine-choices.txt` in the Kera folder), a timing result is remembered there per
  device and reused. Only the timing: a remembered "faster" still gets the probe's output check against the reference
  at each start (`check_only`, 9 Nov), as upstream's every start does, so a driver or engine update that breaks the
  method turns it off rather than making wrong pictures (N-018). Without the variable, unchanged. Worth offering
  upstream.
- **Built:** Kera builds it as a separate CMake project (`ExternalProject` in the top `CMakeLists.txt`) so its
  ggml never meets llama.cpp's, as shared libraries into `bin/image/`.
- **Updating:** copy the same folders from a newer commit (and the ggml commit its `.gitmodules` pins), then run
  `tests/integration/test_image.py`.

- **Change for LTX 2.5 (11 Nov 2026), marked `[Kera]`:** `src/model/diffusion/ltxv.hpp`: the small tables LTX adds to
  its 32-bit working values (`scale_shift_table`, `audio_scale_shift_table`, the prompt and audio-video ones,
  `keyframes_abs_pos_embedding`) took the type the file stores them in. LTX 2.5 stores them as bf16, for which the
  addition has no implementation on the graphics chip or the CPU: the first step stopped. Stored as bf16 they are
  now read as 32-bit floats (`ltx_added_type`), which holds every bf16 value exactly. Other stored types are as
  before. Found by the owner on a DGX Spark. Worth offering upstream.
  `src/conditioning/conditioner.hpp` (`LTXAVEmbedder`): Gemma 4, LTX 2.5's text reader, runs without flash
  attention when it is on a graphics chip. With it (the engine's choice on CUDA) its output is not numbers, the
  video model's first step is already wrong, and every frame comes out white with the sound at full scale. Found by
  switching it off part by part on a DGX Spark: off for the text reader alone gives a real video, with the video
  model's own flash attention still on. On the CPU it is right and stays. Not looked into further (which layer, and
  whether Gemma 3 with LTX 2.3 does the same: no LTX 2.3 here). Worth reporting upstream.
- **LTX's schedule as Lightricks call it (D-042, 11 Nov 2026), marked `[Kera]`:** `src/runtime/denoiser.hpp`: the
  LTX2 scheduler is built with its default of 4096 tokens, not the clip's own token count. Lightricks' pipelines call
  theirs without the latent, so every clip gets shift 2.05; the engine gave a 121-frame clip at 768x512 shift 2.78
  and a 33-frame one 1.27, and a long clip could come out as something that was not asked for (the owner's bug 3).
  Worth reporting upstream.
- **Lightricks' own guider for LTX (D-042, 11 Nov 2026), marked `[Kera]`:** the engine guided LTX with plain CFG
  only. Added, off unless an extra sample arg sets it (Kera does not set it by itself yet) (`ltx_stg`, `ltx_stg_blocks`, `ltx_modality`, `ltx_rescale`,
  `ltx_audio_cfg`):
  - `src/model/diffusion/model.hpp` (`LTXAVDiffusionExtra`: `stg_blocks`, `skip_cross_modal`) and
    `src/model/diffusion/ltxv.hpp`: a pass can leave out the video and audio self-attention of the named blocks, or
    all audio-to-video and video-to-audio cross-attention (a skipped attention adds nothing to its stream, as
    Lightricks' blocks do with a zero keep-mask).
  - `src/runtime/guidance.{h,cpp}`: `LTXGuiderParams`, its parsing, and `ltx_guider_combine`: `cond + (cfg - 1)(cond -
    uncond) + stg (cond - perturbed) + (modality - 1)(cond - isolated)` on the denoised predictions, then scaled by
    `rescale * std(cond) / std(pred) + (1 - rescale)`, as `MultiModalGuider.calculate` does.
  - `src/pipeline/diffusion_engine.cpp`: the two extra passes each step (outside the step cache), and the
    combination for the video values and the audio values of the packed latent each by themselves, the audio with
    its own CFG scale.
  Worth offering upstream.

## acestep.cpp (the music engine, D-028)

- **Upstream:** https://github.com/ServeurpersoCom/acestep.cpp (MIT), commit
  `07abf07da3a9c1c45bd85aabed401644f8eb7c72` (6 Nov 2026), with its own ggml fork
  https://github.com/ServeurpersoCom/ggml at `210138933b22` (the commit its `.gitmodules` pins).
- **Copied, not a subtree:** `CMakeLists.txt`, `LICENSE`, `src/`, `mp3/`, `vendor/` (cpp-httplib, minimp3, yyjson),
  `tools/` (the programs, `version.cmake`, `xxd.cmake` and the prebuilt page `public/index.html.gz`) and `tests/`
  (its `CMakeLists.txt` adds it unconditionally; the test logs are left out), and the ggml fork's `CMakeLists.txt`,
  `LICENSE`, `cmake/`, `include/`, `src/` and `ggml.pc.in` in `ggml/`. Its docs, examples, scripts and the web
  page's sources are left out.
- **Changes:** `src/backend.h`: the CPU thread count can come from `ACESTEP_THREADS`, which `kera music` and
  `kera serve --music` set to the machine's physical cores. Upstream uses half the logical CPUs, which is half the
  cores on CPUs without SMT (Apple Silicon, many laptops and virtual machines): a 10 s song took 71 s instead of
  40 s on 4 such cores. Worth offering upstream.
- **Changes for ACE-Step's own settings (D-036, 10 Nov 2026), marked `[Kera]` in the code:**
  - `src/pipeline-synth-ops.cpp`, `src/request.cpp`, `src/request.h`: what a request that sets none gets ("auto").
    For the base and SFT models: guidance 7.0 and shift 3.0, as ACE-Step's own interface and server use (upstream:
    1.0 and 1.0, the bare `GenerationParams` values, which ran them without guidance). For turbo models: the DCW
    correction on ("double", 0.05 low / 0.02 high), as ACE-Step's own pipeline resolves `dcw_enabled = None`
    (upstream: off; a request's `dcw_scaler` of 0 still turns it off, and a negative one, now the default, is auto).
    The resolved values are printed (`[Resolve-Params] steps=... guidance=... shift=... dcw=...`).
  - `src/model-registry.h`: `ACESTEP_DEFAULT_LM` and `ACESTEP_DEFAULT_DIT` name the files a request that names no
    model gets (moved to the front of their lists; upstream takes the first by file name). `kera music` and
    `kera serve --music` set them to the ones ACE-Step's guide recommends for the computer's memory.
- **Change for loading (D-040, 10 Nov 2026), marked `[Kera]`:** `src/weight-ctx.h`: `wctx_alloc` copies each
  tensor to a graphics chip through one buffer of ordinary memory (as big as the largest tensor, freed after),
  filled by up to 8 threads, instead of handing the backend a pointer into the memory-mapped file. On a DGX Spark
  the direct copy ran at about 0.5 GB a second (13.5 s for 8 GB), this one in 0.5 s; the bytes are the same. CPU
  buffers are filled as before. The line it prints now gives the two times (`alloc ... ms, copy ... ms`). Worth
  offering upstream.
- **Built** like the image engine: a separate CMake project (`ExternalProject`), shared libraries, its programs
  `ace-lm`, `ace-synth` and `ace-server` and their ggml copied into `bin/music/` by `cmake/kera_copy_engine.cmake`.
- **Both engines, on Windows:** `cmake/kera_engine_utf8.cmake` (passed as `CMAKE_PROJECT_INCLUDE`, so neither
  project's files change) gives their programs a manifest that makes UTF-8 their code page; they read their command
  line and open files in the code page, and kera passes paths in UTF-8. Both are also built with ggml's CPU
  variants when Kera is (`GGML_BACKEND_DL`, `GGML_CPU_ALL_VARIANTS`), like the chat engine.
- **Updating:** copy the same folders from a newer commit (and the ggml commit its `.gitmodules` pins), apply the
  `backend.h` change and the `[Kera]` changes above again, then run `tests/integration/test_media.py`.

## trellis.cpp (the 3D engine, D-031)

- **Upstream:** https://github.com/pwilkin/trellis.cpp (MIT), commit `6851891d706e2e26a8bb9dd99f28de851d68b64d`
  (1 Nov 2026: release v0.8.1 and 20 fixes after it, among them BiRefNet retrying on the CPU when the graphics chip
  runs out of memory, and small quantized weights widened to F16 at load), with its own ggml fork
  https://github.com/pwilkin/ggml at `f8697b33795db2ee6a2569b40853ca6aaa69f9b9` (the commit its `.gitmodules` pins).
- **Copied, not a subtree:** `CMakeLists.txt`, `LICENSE`, `include/`, `src/` (without `quad_matching.cpp` and
  `quad_collision.cpp`, which need CGAL), `thirdparty/` (cpp-httplib, fqms, meshoptimizer, stb, xatlas; without
  `retopo-dependencies.cmake`, which downloads CGAL, Boost, GMP and MPFR) and the nine
  `tools/mesh_quad_matching/*.cpp` programs its `CMakeLists.txt` defines unconditionally (with `quad_uv_repair.h`),
  and the ggml fork's `CMakeLists.txt`, `LICENSE`, `cmake/`, `include/`, `src/` and `ggml.pc.in` in
  `thirdparty/ggml/`. Its desktop app, sample assets (115 MB), docs, Python tools, installers and the GPL
  retopology are left out. 27 MB.
- **Changes** (each marked `Kera:` in `CMakeLists.txt`):
  - With `GGML_BACKEND_DL` (the downloads' CPU variants) its back-ends are loaded at run time and can't be linked:
    `CMakeLists.txt` links none of them then, and the added `kera/ggml_backend_dl_shim.cpp` gives the four back-end
    calls trellis.cpp makes (`ggml_backend_cpu_init`, `ggml_backend_is_cpu`, `ggml_backend_cpu_set_n_threads`,
    `ggml_backend_cuda_init`) through ggml's registry, and loads the back-ends before `main()`. Its sources are
    unchanged.
  - Its programs' library folder is Kera's (`CMAKE_BUILD_RPATH`) when given: upstream's `$ORIGIN` means nothing on
    macOS, and on the Spark CUDA's own folders are added.
  - `TRELLIS_RETOPO_MATCHING=ON` stops with a message instead of downloading CGAL.
- **Built** like the other engines (`ExternalProject` `kera-3d`): shared libraries, only `trellis-cli`, with
  `TRELLIS_WEBP=OFF` (WebP textures download libwebp at build time; the textures are PNG instead), into `bin/3d/`.
- **Updating:** copy the same files from a newer commit (and the ggml commit its `.gitmodules` pins), apply the
  `Kera:` changes again (and the shim, if it calls other back-end functions), check that a new unconditional
  `add_executable` in `CMakeLists.txt` has its source copied, then run `kera 3d` on a picture.

## audio.cpp (the sound-effect engine, D-031)

- **Upstream:** https://github.com/0xShug0/audio.cpp (Apache-2.0), commit `408935bfe6ffc52aba6ecc08fdf31ce017d46a6d`
  (8 Nov 2026). The latest release, v0.9.4 (4 Nov), doesn't have MOSS-SoundEffect v2, the permissively licensed
  model Kera uses by default. Its ggml copy is part of the repository (`external/ggml`, not a submodule).
- **Copied, only what the sound-effect models need:** `CMakeLists.txt`, `LICENSE`; `model_specs/` for
  `moss_soundeffect` and `stable_audio`; `include/audiocpp.h`, `include/engine/framework/` and `src/framework/`;
  the models `moss_soundeffect`, `stable_audio` and the two every build links (`silero_vad`, `marblenet_vad`) in
  `src/models/` and `include/engine/models/`, and `controlfoley/flow_denoiser.*` (part of its core library);
  `app/`; `tests/perf/model_perf.cpp` (a program its `CMakeLists.txt` always defines); `tools/cmake/` (without the
  eSpeak files: eSpeak is GPL); and in `external/`: cJSON, libyaml, llama_tokenizer, utf8proc, ggml
  (`CMakeLists.txt`, `LICENSE`, `cmake/`, `include/`, `src/`) and SentencePiece (`CMakeLists.txt`, `LICENSE`,
  `README.md`, `VERSION.txt`, `cmake/`, `config.h.in`, `sentencepiece.pc.in`, `src/`, `third_party/`). Left out: the
  other 130 models, docs, tests, web UI (with its demo voices), notebooks, scripts, the server's frontends
  submodule, SentencePiece's data and Python, and cpp-httplib and lz4 (used only by its model manager). 42 MB.
- **Text-encoder cache (D-032):** `src/models/moss_soundeffect/session.cpp`: with `AUDIOCPP_TEXT_CACHE` naming a
  folder (Kera sets one per model file in its folder), each prompt's text encoding is kept there; a prompt seen before
  never loads the 2 GB text encoder, and the encoder is freed before the sound model runs. The cached floats are the
  encoder's own output, so the sound is the same. Without the variable, only the early freeing differs.
- **Changes** (each marked `Kera:` in `CMakeLists.txt`):
  - `audiocpp_add_model`: a model whose sources weren't copied gets an empty target (the added
    `kera/absent_model.cpp`) and can't be selected, so the file's other lines that name it keep working.
  - The server's demo voices, embedded at configure time, become one zero byte when absent (Kera builds only
    `audiocpp_cli`).
  - The `soulx_flashhead` test programs are defined only when that model is there.
  - With its CPU variants, its programs' library folder is Kera's (`CMAKE_BUILD_RPATH`) when given, as for the
    other engines (CUDA's own folders on the Spark); upstream's is `$ORIGIN` alone.
- **Built** like the other engines (`ExternalProject` `kera-sfx`): `AUDIOCPP_MODEL_SET=custom` with
  `moss_soundeffect,stable_audio`, `AUDIOCPP_DEPLOYMENT_BUILD=ON` (the models' specifications built in),
  `ENGINE_ENABLE_OPENMP=OFF`, and its own switches for ggml's CPU variants (`ENGINE_ENABLE_CPU_ALL_VARIANTS`, which
  makes the libraries shared) and native code; only `audiocpp_cli`, into `bin/sfx/`. It gives its program a UTF-8
  manifest on Windows itself.
- **Updating:** copy the same folders from a newer commit, apply the `Kera:` changes again, configure once
  (`cmake -S third_party/audio.cpp -B /tmp/x -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=moss_soundeffect,stable_audio`)
  and copy any file it says is missing, then run `tests/integration/test_3d_sfx.py`.

- **Shader cache (all four engines, D-032, work in progress):** `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (and its
  `ggml-vulkan-types.h` where the device struct lives) get the same pipeline cache file as Kera's llama.cpp: with
  `GGML_VK_PIPELINE_CACHE` set, built shaders are loaded at start and saved as they appear. Kera gives each engine a
  file of its own (`vulkan-pipelines-image`, `-music`, `-3d`, `-sfx`). Marked `[Kera]`. Not yet built or run on a
  graphics chip here.

## ggml's CPU variants, in all five copies (D-041)

`ggml/src/CMakeLists.txt` in llama.cpp, stable-diffusion.cpp and acestep.cpp, `thirdparty/ggml/src/CMakeLists.txt` in
trellis.cpp and `external/ggml/src/CMakeLists.txt` in audio.cpp, marked `[Kera]`: with `GGML_CPU_ALL_VARIANTS` on
Linux ARM, the two variants for ARMv9.2 with SME (`armv9.2_1`, `armv9.2_2`) are added only when the compiler knows
`+sme` (`check_c_compiler_flag`). GCC 13, DGX OS's own compiler, stops at that flag, which stopped the whole build on
a DGX Spark. With GCC 14 (GitHub's build) nothing changes. The GB10 has no SME (SVE2 and I8MM: it takes
`armv8.6_2`), so a download built on a Spark runs there as GitHub's does. Worth offering upstream.

## Colibri

- **Upstream:** https://github.com/JustVugg/colibri (Apache-2.0)
- **Read at:** commit `bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850` (6 Oct 2026).
- **Ported files:** none yet. Each ported file keeps Colibri's licence and
  copyright lines, carries a note saying Kera changed it, and is listed here.
  Colibri's NOTICE goes into Kera's `NOTICE` with the first ported file.
