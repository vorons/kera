# Kera: read this first

Kera is a local AI engine and downloadable app built on llama.cpp. The full plan is
`docs/KERA_BUILD_SPEC.md`.

## Every session

1. Start by reading `docs/STATUS.md`. It says where the last session stopped.
2. Read the spec's "Owner decisions" section, section 0 (ground rules) and the build playbook (P1–P8)
   before writing code. Follow P4's "Needs" column. Never skip a gate.
3. The GitHub fork is public by the owner's decision (18 Nov 2026; D-004 was cancelled). Push to it when
   asked; no visibility check is needed.

## The owner's rules (details in `docs/DECISIONS.md`, D-001 to D-010, D-033, D-034 and D-038)

- **Rule number one: Kera never changes how a model responds. It only makes things faster (D-033).** Nothing
  that changes a model's thinking, its answer, its length, its wording or its sampling, by default or as an
  option Kera adds for it: no turning thinking off or capping it, no stopping an answer early or cutting it, no
  editing what it wrote, no fixed seed where the engine's own is random, no handing back a remembered answer
  instead of asking the model. A model runs with its own template and its own card's settings (D-013), as in a chat.
  A speed-up must give the same output (or differ by rounding only, checked against the reference, as D-030's
  were). Kera has no modes that trade quality for speed, not even opt-in ones: M12's lossy modes were removed
  (D-034), and none is to be added.
- **Order:** the owner's DGX Spark first, then Mac (D-038: the Minisforum was sold, 10 Nov). Windows keeps being
  built and checked by GitHub.
- **Where:** work in the cloud, keep all code in this repo, and let GitHub's machines build and test.
  The Spark is reached over Tailscale with the agent's own login: one `kera` folder there, no admin rights.
- **No pull requests:** commit and push to the working branch, with the gate result in each commit
  message. Merge a milestone into `main` once its gate passes, and tag a release.
- **Don't stop:** if a gate fails twice, or a milestone takes over twice its size, log it in
  `docs/NEGATIVE_RESULTS.md`, put the options in `docs/STATUS.md`, and move on to milestones that don't
  depend on it. Do M7 only if M1 says go.
- **After each milestone:** merge, tag, update `docs/STATUS.md`, show its milestone table, write the
  section 9 report, and carry straight on.
- **Allowed:**
  - Download models, keeping 100 GB free on the machine. Keep them inside the Kera folder and delete
    them when done. Use the owner's existing models where they are.
  - Run benchmarks any time; other AI apps may be closed meanwhile, and are reopened after.
  - Merge your own work once checks are green and the gate passed.
- **Never** (list it in `docs/STATUS.md` under "Waiting for the owner" instead):
  - admin rights after the first setup;
  - buying or renting anything, payment methods, or plan changes;
  - adding people to the repo;
  - BIOS, driver or Windows changes;
  - deleting anything outside the Kera folder;
  - hardware other than the owner's DGX Spark (and the owner's MacBook, for the Kera-vs-Ollama test app: D-025);
  - adding any mode or setting that trades quality for speed, even off by default (D-034);
  - putting secrets or model weights in the repo;
  - putting the owner's name or personal details in the repo.

## Build and test

```sh
cmake -S . -B build -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release   # GPU: -DGGML_VULKAN=ON (Windows, Linux); Metal is on by default on Mac
cmake --build build -j 4
ctest --test-dir build --output-on-failure                 # C++ unit tests
python -m pytest tests/python                              # reference-code checks (needs numpy, pytest)
./build/bin/kera setup --quick                              # measure this machine in seconds
./build/bin/kera                                            # start the app (a page in the browser)
```

A cloud machine can change processors between sessions while the build folder and its compiler cache (ccache) stay:
`-march=native` then gets CPU kernels compiled for the earlier processor. Before comparing Kera's numbers with
another build, rebuild them with `CCACHE_DISABLE=1` (N-018).

**Layout:**
- `src/`: the engine parts. Each folder's `kr_*.h` is a P3 interface, frozen in M0.
- `app/`: the `kera` program and its web page.
- `tools/`: the converters and the bench.
- `reference/`: the spec's appendices, copied word for word.
- `tests/`: the tests.
- `third_party/llama.cpp`: a git subtree. Every change inside it is listed in `docs/UPSTREAM_PATCHES.md`.

## Done, for every change (P7)

- **Settings:** a new setting is registered, and stays off until its gate passes.
- **Tests:** unit tests and the tiny-model exactness test pass on the CPU.
- **Speed:** a bench JSON before and after, from a real machine, with a row in `docs/BENCHMARKS.md`.
- **Records:** the gate result goes in the commit message, failures in `docs/NEGATIVE_RESULTS.md`, and
  choices in `docs/DECISIONS.md`.
- **Build:** CI is green with no new warnings, and `docs/STATUS.md` is updated.
