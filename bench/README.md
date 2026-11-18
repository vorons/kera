# Bench

The section 7 benchmark. `kera bench --model FILE.gguf` runs each prompt in `prompts/` once to warm
up, then three times, and saves the median of each measurement as JSON (the P3 bench results shape):

- `ttft_ms`: the wait for the first word, from the start of reading the prompt.
- `prefill_tok_s`: prompt reading speed.
- `decode_tok_s`: writing speed over 256 words (greedy; the end-of-text word doesn't stop it).
- `peak_ram_gb`: the most memory the process held (GiB, as the OS reports it).
- The machine profile's hash, the Kera and llama.cpp versions, the model file's name and XXH3-128
  hash, and every setting (Kera's registry and the engine's).
- Fields for later milestones stay `null` until those parts exist: `ssd_gb_per_tok`, `expert_hit`,
  `prefetch_acc`, `kv_hit`, `accept_rate`, `k_histogram` and `ceiling_tok_s`.

The prompts are built into the program, so the downloads can run the bench with nothing else:

| Prompt | What it is |
| --- | --- |
| `short_chat` | A short question |
| `code` | Rewrite a buggy Python function, with tests |
| `long_context` | About 4,000 tokens of text (Pride and Prejudice, chapters 1-3, public domain) and a question about it |
| `tool_call` | A question that needs a weather tool, with the tool passed to the model's chat template |

Prompts are rendered with the model's own chat template, as llama.cpp's server does. If a template
can't take tools, the tool is described in the system message instead; `rendered_with` in the JSON
says which way was used.

`--markdown` prints rows for `docs/BENCHMARKS.md`. Results chosen for the record go in `results/`
(file names `YYYY-MM-DDTHHMMSSZ_model.json`); `*.local.json` files there are ignored by git.
