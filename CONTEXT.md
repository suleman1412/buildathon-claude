# AI Research Paper Validator — Agent Context & State

Read this before touching this project. It's written so a Claude Code
session (or any other agent) starting cold — on a different machine, a
different Kaggle account, whatever — can pick up exactly where this one
left off without re-deriving decisions or re-hitting already-solved bugs.

Last updated: 2026-09-01, mid-session, LoRA validation run in progress
(blocked, not failed — see §7).

## 1. What this project is

An autonomous agent that reads a research paper PDF, finds its official
code repo, reproduces its headline eval, and reports whether the claimed
metric holds up. Original target architecture (Claude Haiku/Sonnet +
Modal sandboxes) is in `validator_project_summary.md` at the repo root —
read that first for the "real" v1 design intent.

## 2. Why v0 differs from v1 — the two backend swaps

The user has **no Anthropic API key and no Modal account yet**. Rather
than block on that, v0 swaps both:

| v1 (summary doc) | v0 (this codebase) |
|---|---|
| Claude Haiku (extraction) / Sonnet (debugging) | Local quantized **Qwen2.5-3B-Instruct** (GGUF), run via `llama-cpp-python` |
| `modal.Sandbox` for isolated execution | The Kaggle/Colab notebook's **own GPU VM** — subprocess calls, no extra sandboxing |

**Hard constraint, repeatedly confirmed by the user: no model inference or
repo execution ever happens on the developer's laptop.** All of that
happens inside the Kaggle/Colab notebook's runtime. The laptop only holds
source code and runs `mock`-backend wiring tests (no model, no network).

**Also confirmed: do not `git init`/commit/push without explicit
permission.** This directory is not currently a git repo. Don't add one
unless asked.

## 3. Repo layout

```
buildathon-claude/
  validator_project_summary.md   original v1 architecture doc (untouched)
  CONTEXT.md                     this file
  README.md                      user-facing setup/usage docs
  requirements.txt                core deps, safe to pip install locally (no model)
  requirements-kaggle.txt         deps for the Kaggle/Colab runtime only
  validate.py                    CLI entrypoint (real use is via the notebook, not this)
  src/
    config.py                     Config dataclass: retries, timeouts, tolerance, workdir
    llm_client.py                 pluggable LLM backend -- see §4, has had 1 real bug fix (§7)
    pdf_extract.py                PDF -> raw text (PyMuPDF/fitz, text-only, no VLM)
    extraction.py                 raw text -> ExtractedClaim (repo, dataset, claimed metric) via LLM
    provisioning.py               git clone + pip install -r requirements.txt
    execution.py                  LLM proposes {command, metric_regex}; runs it; extracts metric
    debug_loop.py                 on failure: truncate stderr, ask LLM for a patch command, retry (capped)
    verification.py               compare reproduced vs. claimed metric -> VerificationResult -> markdown
    pipeline.py                   orchestrates all of the above; had 1 real bug fix (§7)
  tests/
    test_pipeline_mock.py         9 tests, all passing, MockLLMClient + synthetic script, no model/network
  notebooks/
    kaggle_colab_runner.ipynb     THE REAL ENTRYPOINT. Self-contained: writes out src/ itself via
                                   %%writefile, installs deps, downloads the model, runs the pipeline.
```

Run `python3 -m unittest tests.test_pipeline_mock -v` from `buildathon-claude/`
locally any time to sanity-check pipeline wiring after an edit — it's fast
(~0.02s), needs no model, no GPU, no network.

## 4. Key design decision: `LLMClient` backend abstraction

`src/llm_client.py` defines an abstract `LLMClient` with `.complete()` /
`.complete_json()`. Every pipeline stage (`extraction.py`, `execution.py`,
`debug_loop.py`) only ever talks to this interface, never to a concrete
backend. Three backends:

- `MockLLMClient` — queued canned responses, used only by local tests.
- `LlamaCppLLMClient` — wraps `llama_cpp.Llama`, lazy-imports `llama_cpp`
  so this module is safely importable on a machine with no GPU. **Only
  ever instantiate this inside the Kaggle/Colab notebook.**
- `anthropic` — not implemented. This is the intended v1 upgrade path:
  add a class, wire it into `get_llm_client()`, done. `pipeline.py`
  should need zero changes.

Same story for the execution sandbox: swapping `provisioning.py` /
`execution.py`'s direct `subprocess` calls for `modal.Sandbox` calls is
the v1 upgrade there, also isolated from `pipeline.py`.

## 5. Kaggle setup — exact steps and specs

1. Kaggle account needs phone verification (required for GPU + internet).
2. Import `notebooks/kaggle_colab_runner.ipynb` (Code → New Notebook →
   File → Import Notebook).
3. Upload the target PDF as a Kaggle **dataset** (Add Input → Upload) —
   Kaggle notebooks can't read arbitrary local paths, only
   `/kaggle/input/<dataset-slug>/<filename>.pdf`. Note the exact path.
4. Session settings (right sidebar): **Accelerator = GPU T4 x2**,
   **Internet = On** (off by default — the most commonly-missed step).
5. Edit the `PDF_PATH = "/kaggle/input/your-dataset/paper.pdf"` line in
   the "Provide input PDF" cell to the real path from step 3.
6. Run all cells top to bottom.
7. Model: `Qwen/Qwen2.5-3B-Instruct-GGUF`,
   `qwen2.5-3b-instruct-q4_k_m.gguf` (~2GB). Fits one T4's 16GB trivially;
   the second T4 (of the x2) sits idle, that's fine. Swap to the 7B quant
   in the same repo if 3B's extraction/command-guessing quality proves too
   weak (it has — see §7).
8. Free-tier quota: 30 GPU-hours/week, ~9-12hr/session cap. Nowhere close
   to being hit by this workload.

## 6. Notebook mechanics — non-obvious gotchas already hit

These cost real debugging time; don't rediscover them.

- **`%%writefile` cannot have an empty cell body.** `src/__init__.py` is
  an empty file; a `%%writefile src/__init__.py` cell with nothing after
  it raises `UsageError: cell body is empty`. Fixed by using a plain
  Python cell instead: `open("src/__init__.py", "w").close()`. Already
  fixed in the notebook — if you see this error, some other empty file
  slipped into `src_files` without the same treatment.

- **Module caching**: re-running a `%%writefile` cell only rewrites the
  file on disk. It does **not** reload an already-`import`ed Python
  module in that kernel session — Python caches by `sys.modules`. Twice
  in this session, a real fix was written to disk and then appeared to
  "not work" because the kernel was still executing old bytecode. Fix:
  after re-writing a file that's already been imported, run
  `importlib.reload(src.<module_name>)` — this works correctly even for
  already-instantiated objects, because a function's `__globals__` is a
  *reference* to the module's (mutated-in-place) `__dict__`, not a
  snapshot, so reload propagates without needing to re-instantiate
  anything. If in doubt whether a reload actually took, verify with
  `inspect.getsource(src.<module>.<name>)` before assuming a fix landed.
  When in doubt, Kernel → Restart & Run All is the guaranteed-clean
  fallback (costs the ~2GB model re-download + reinstall).

- **`llama-cpp-python` CUDA wheel install can silently compile from
  source** if the `--extra-index-url .../cu121` wheel doesn't match
  Kaggle's actual CUDA version, especially combined with
  `CMAKE_ARGS="-DGGML_CUDA=on"` — this can take 10-20+ minutes with zero
  visible output if wrapped in `%%capture`. The install cell was rewritten
  to try the CUDA wheel, print any failure, and explicitly fall back to
  the CPU-only prebuilt wheel rather than ever silently compiling. Current
  install cell content is in `notebooks/kaggle_colab_runner.ipynb` (search
  for `cuda_install`).

- PyMuPDF prints `warning: The 'fitz' API is deprecated...` on import.
  Harmless, ignore it (we still `import fitz` since that's the current
  stable API name despite the warning).

## 7. Bugs found & fixed in `src/` (all applied, all tested)

1. **`_parse_json_loose` couldn't handle invalid JSON escapes from the
   local model.** Small quantized models sometimes emit things like
   `"LAN-\GUAGE"` (a stray literal backslash from trying to preserve a
   PDF hyphenation line-break) — `\G` isn't a valid JSON escape, so
   `json.loads` rejected the entire response outright. Fixed in
   `src/llm_client.py` with `_escape_stray_backslashes()`: doubles any
   backslash not already starting a legal escape (`" \ / b f n r t u`),
   tried as a fallback candidate alongside the raw and brace-extracted
   strings. Regression tests: `TestJSONRepair` in
   `tests/test_pipeline_mock.py`.

2. **`pipeline.py` discarded all failure diagnostics.** On eval failure
   it returned only `"Evaluation script failed after retries."` with no
   command, no stdout, no stderr — undebuggable. Fixed: the `reason` now
   includes the exact command tried and a truncated stderr tail. See
   `run_pipeline()` in `src/pipeline.py`.

Both fixes are synced into `notebooks/kaggle_colab_runner.ipynb`'s
`%%writefile` cells already — the local `.py` files and the notebook's
embedded copies are consistent as of this writing.

## 8. Current test case: LoRA paper — detailed state (IN PROGRESS)

Test PDF: the LoRA paper (Hu et al. 2021), uploaded as a Kaggle dataset by
the user. Extraction correctly found `github_repo:
https://github.com/microsoft/LoRA`, dataset "GLUE...", claimed metric
91.5% accuracy (this number is a plausible-but-unverified conflation of
per-task GLUE scores — worth eyeballing against the paper if this becomes
important later).

**Pipeline mechanics are now proven working end-to-end** through
extraction → clone → propose-run-plan → execute → debug-surface → (would
verify). The remaining blocker is specific to this one repo, not to our
code:

- Round 0: model **hallucinated** an eval command (`eval/eval.py` with
  DART/NLG dataset paths) that doesn't exist in the repo at all — it
  pattern-matched on LoRA's NLG (GPT-2) example instead of grounding in
  the actual repo listing / the claimed GLUE dataset. Root cause: a 3B
  quantized model's weak instruction-following/context-grounding — an
  expected v0 limitation, not a bug to fix in code.
- Diagnosed manually: real GLUE/RoBERTa eval lives at
  `examples/NLU/`. `examples/NLU/roberta_large_sst2.sh` is a **full
  training run** (10 epochs, 8-GPU distributed) — infeasible for a quick
  session. README's "Evaluate the checkpoints" section shows the
  eval-only pattern instead: load a pretrained LoRA adapter checkpoint,
  `--do_eval` only, single GPU.
- Adapted that pattern for RoBERTa-large/SST-2, downloading
  `roberta_large_lora_sst2.bin` from
  `https://github.com/msft-edward/LoRA_private/releases/download/RoBERTa-large/roberta_large_lora_sst2.bin`
  (this succeeded — non-obvious since the org name looks like it could be
  a dead private mirror, but it worked).
- **Round 1 failure**: `ImportError: cannot import name 'load_metric'
  from 'datasets'` — `datasets>=3.0` removed `load_metric` (moved to the
  separate `evaluate` package); this 2021 script predates that split.
  **Fixed** with `pip install -q "datasets<3" --upgrade` (landed on
  2.21.0). Confirmed via `from datasets import load_metric` succeeding.
- **Round 2 failure**: `AttributeError: 'TrainingArguments' object has no
  attribute 'use_deterministic_algorithms'`. **Patched** directly in the
  cloned repo's copy of
  `examples/NLU/examples/text-classification/run_glue.py` (⚠️ **this edit
  lives only in the Kaggle session's ephemeral filesystem at
  `/kaggle/working/runs/<pdf-stem>/repo/...` — it is NOT saved anywhere
  in this git-less local repo. If the Kaggle session restarts or the repo
  gets re-cloned, this patch is lost and must be reapplied**):
  ```python
  old = "torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)"
  new = "torch.use_deterministic_algorithms(getattr(training_args, 'use_deterministic_algorithms', False))"
  ```
- **Round 3 failure** (after the round-2 patch): `AttributeError:
  'TrainingArguments' object has no attribute 'cls_dropout'`, at a
  *different* line. The `TrainingArguments` dump in the error output
  showed clearly-modern HuggingFace fields (`optim=OptimizerNames.ADAMW_TORCH_FUSED`,
  `trackio_space_id`, `parallelism_config`) — none of which existed in
  2021. **Actual root cause identified**: the script is importing
  Kaggle's system-installed **modern** `transformers` package, not this
  repo's own vendored fork at `examples/NLU/src/transformers` (which
  defines the custom fields like `apply_lora`, `lora_r`, `cls_dropout`).
  The editable install of the vendored package either never ran or lost
  the import-path race.
- **Round 4 attempt**: prefixed the run command with
  `PYTHONPATH=<repo>/examples/NLU/src:$PYTHONPATH` to force the vendored
  fork to resolve first. **Result: identical failure, byte-for-byte
  identical `TrainingArguments` dump** — meaning either the fix wasn't
  actually applied (command not re-run with the new prefix) or the
  vendored package still isn't taking precedence for a deeper reason
  (possibly the editable install's `.pth`/finder mechanism actively wins
  over `PYTHONPATH` ordering, or `examples/NLU/src/transformers` isn't a
  complete/importable package on its own).

**This is where the session was interrupted** (user asked for this
context doc instead of picking a next step). Three options were on the
table, not yet decided:

1. **Switch to an easier paper/repo** — one with a small, self-contained,
   currently-maintained eval script (no distributed launch, no checkpoint
   hosting, no 2021-era frozen dependency stack) — fastest path to a
   clean end-to-end PASS that actually validates the loop.
2. **One more real attempt on LoRA**, doing it properly per the README's
   own instructions (`conda env create -f examples/NLU/environment.yml`,
   or a clean `pip uninstall transformers && pip install -e examples/NLU`)
   rather than continuing ad hoc `PYTHONPATH`/patch fixes. Likely needs a
   fresh Kaggle session.
3. **Stop here** — treat the mechanics (extraction → clone → execute →
   debug-surface → verify) as sufficiently proven for v0 even without a
   clean PASS on this specific paper, and move on to other v0 work.

**If you're picking this up fresh: ask the user which of these three they
want before doing more LoRA-specific debugging.** Don't assume — this was
an open question when the session paused.

## 9. Known v0 limitations (also in README.md)

- Text-only PDF extraction (no VLM) — figures/complex tables may be missed.
- Eval command + metric regex are **guessed** by the LLM from a repo file
  listing + top-level README only. Confirmed failure mode: it can
  hallucinate commands referencing files that don't exist, especially
  when the real eval entrypoint is nested in a subdirectory the top-level
  README doesn't mention (exactly what happened with LoRA's `examples/NLU/`).
  A concrete improvement worth making later: give `propose_run_plan` a
  recursive/deeper listing, or a two-pass "which subdirectory looks
  relevant" step, rather than a single flat top-level pass.
- Dependency install is a single best-effort `pip install -r
  requirements.txt`; only the eval *run* step goes through the LLM
  debug/patch retry loop, not the install step.
- No sandbox beyond the Kaggle/Colab VM itself.
- 3B local model is meaningfully weaker than Claude at both JSON
  discipline (needed the escape-repair fix) and grounded instruction
  following (hallucinated the first eval command). If v0 continues to
  struggle on more papers, trying the 7B quant is the cheap next lever
  before reaching for a real API-backed model.

## 10. Path to v1

Swap `llm_client.py`'s backend from `llama_cpp` to a new `anthropic`
class (Haiku for extraction, Sonnet for debugging), and swap
`provisioning.py`/`execution.py`'s direct subprocess calls for
`modal.Sandbox` calls. `pipeline.py` should not need to change for
either swap — that separation was the whole point of the `LLMClient`
interface design.
