# AI Research Paper Validator — Agent Context & State

Read this before touching this project. It's written so a Claude Code
session (or any other agent) starting cold — on a different machine, a
different account, whatever — can pick up exactly where this one left off
without re-deriving decisions or re-hitting already-solved bugs.

Last updated: 2026-09-02, mid **Claude build night** (3-hour build, then
present). Session was working on the Mac (8GB RAM); pivoting to an old
EliteBook (32GB RAM) as the actual compute machine — see §5. A static
presentation dashboard has been added (§5.1) but not yet deployed or
tested against a real run. Anthropic
API key is **pending, not yet received** — see §4.

## 0. If you're picking this up cold, read this first

- **Time-boxed.** This is a 3-hour hackathon build, not an open-ended
  project. Bias toward "get one clean end-to-end PASS on a demoable
  paper" over "perfectly reproduce the hardest paper we tried."
- **Budget: assume ONE Anthropic API key, ~$50.** (A teammate has a second
  $50 key, but plan as if you only have one — don't assume the second is
  available.) A hard per-run spend cap is already built into the code
  (§4) — don't remove it or raise it casually.
- **Open decision, not yet made**: whether to keep pushing on the LoRA
  paper (blocked mid-debug, see §8) or switch to an easier paper (§9) to
  bank a guaranteed clean demo first. Ask the user rather than assuming —
  this was already an open question once before and got interrupted by a
  context/infra pivot (Mac→EliteBook, local-model→Anthropic).
- **Compute machine is changing.** Originally Kaggle/Colab GPU (for a
  local model), now pivoting to an old EliteBook (32GB RAM, presumably no
  GPU) reached via `git pull`, using the Anthropic API for inference
  instead of a local model. The Kaggle notebook path (§6) still exists
  and still works, but is no longer the primary plan — see §5.

## 1. What this project is

An autonomous agent that reads a research paper PDF, finds its official
code repo, reproduces its headline eval, and reports whether the claimed
metric holds up. Original target architecture (Claude Haiku/Sonnet +
Modal sandboxes) is in `validator_project_summary.md` at the repo root —
read that first for the "real" v1 design intent. This codebase has
converged close to that v1 design already (see §4) — what's still
missing is Modal (execution is plain `subprocess`, no sandboxing beyond
whatever machine you run it on).

## 2. Git state

- Repo root is `buildathon-claude/` (not the parent `claude-buildathon/`
  directory, which just contains this one project).
- Remote: `origin` → `git@github.com:suleman1412/buildathon-claude.git`
  (a teammate's account — this is a shared team repo, confirmed
  intentional).
- Working branch: **`v0`** (not `main` — `main` has zero commits,
  intentionally left untouched; all work happens on `v0`).
- `v0` has been pushed to `origin` already. **Always `git pull origin v0`
  on a new machine before assuming local state is current** — this
  session pushes as it goes specifically so other machines/sessions can
  pick up mid-build (see §5's EliteBook workflow).
- Standing rule from the user: **don't push or commit without
  permission** — but permission was already given for the Mac→GH→EliteBook
  handoff workflow specifically, so committing/pushing incremental
  progress on `v0` as you go is expected, not something to re-ask for
  every time, *as long as it's still serving that handoff*. Don't push to
  `main`, don't force-push, don't push to any repo other than this one
  without asking again.
- `.gitignore` excludes `__pycache__/`, `*.pyc`, `*.swp`, `.DS_Store`,
  `runs/` (pipeline scratch dir), `report.md` (CLI output).

## 3. Repo layout

```
buildathon-claude/
  validator_project_summary.md   original v1 architecture doc (untouched)
  CONTEXT.md                     this file
  README.md                      user-facing setup/usage docs (may lag behind
                                  this file during the build-night pivot -- if
                                  they conflict, trust this file, it's fresher)
  requirements.txt                core deps: PyMuPDF, anthropic. Safe to pip
                                   install anywhere -- no model weights, no GPU.
  requirements-kaggle.txt         extra deps for the Kaggle/Colab local-model
                                   path only (llama-cpp-python, huggingface_hub)
                                   -- NOT needed for the Anthropic-backend path.
  validate.py                    CLI entrypoint. Defaults to --backend anthropic.
                                   Appends a run record to dashboard/data/runs.json
                                   after every run (see §5.1); disable with --no-dashboard.
  run_demo.sh                    One-shot: run pipeline + serve dashboard. See §5.1.
  src/
    config.py                     Config dataclass: retries, timeouts, tolerance, workdir
    llm_client.py                 pluggable LLM backend -- see §4
    pdf_extract.py                PDF -> raw text (PyMuPDF/fitz, text-only, no VLM)
    extraction.py                 raw text -> ExtractedClaim (repo, dataset, claimed metric) via LLM
    provisioning.py               git clone + pip install -r requirements.txt
    execution.py                  LLM proposes {command, metric_regex}; runs it; extracts metric
    debug_loop.py                 on failure: truncate stderr, ask LLM for a patch command, retry (capped)
    verification.py               compare reproduced vs. claimed metric -> VerificationResult -> markdown
    pipeline.py                   orchestrates all of the above
    dashboard.py                   record_run() -- appends a run summary to dashboard/data/runs.json
  tests/
    test_pipeline_mock.py         13 tests, all passing -- MockLLMClient + synthetic
                                   script (no model/network) for pipeline wiring,
                                   AnthropicLLMClient spend-cap tests with a mocked API
                                   client, and dashboard record_run() tests
  notebooks/
    kaggle_colab_runner.ipynb     Secondary path: local-model-on-Kaggle-GPU runner.
                                   Self-contained (writes out src/ itself via %%writefile).
                                   Still works, but no longer the primary plan -- see §5.
  dashboard/
    index.html, style.css, app.js  static presentation dashboard, see §5.1
    data/runs.json                 run history, written by validate.py (currently [])
    netlify.toml, README.md        deploy config + steps
```

Run `python3 -m unittest tests.test_pipeline_mock -v` from `buildathon-claude/`
any time to sanity-check pipeline wiring — fast (~0.7s), no GPU, no
network calls (the Anthropic tests mock the API client).

## 4. Key design decision: `LLMClient` backend abstraction

`src/llm_client.py` defines an abstract `LLMClient` with `.complete()` /
`.complete_json()`. Every pipeline stage (`extraction.py`, `execution.py`,
`debug_loop.py`) only ever talks to this interface, never to a concrete
backend — this is why every backend swap so far has needed zero changes
to `pipeline.py`. Backends:

- `MockLLMClient` — queued canned responses, used only by local tests.
- `LlamaCppLLMClient` — wraps `llama_cpp.Llama`, lazy-imports `llama_cpp`.
  Only ever instantiate this inside the Kaggle/Colab notebook (§6) — it's
  the local-model path, now secondary.
- **`AnthropicLLMClient` — implemented, this is now the default/primary
  backend.** Key details:
  - Defaults to **Haiku** (`claude-haiku-4-5-20251001`), not Sonnet.
    Deliberate cost-safety choice given the single-key/$50 budget and
    3-hour clock — cost discipline matters more right now than squeezing
    out extra reasoning quality on every call. Pass
    `model="claude-sonnet-5"` explicitly for a specific hard-debugging
    attempt (e.g. a serious LoRA retry), not as the default.
  - **Hard spend cap, enforced in code, not just documented.** Default
    `max_spend_usd=2.0` per `AnthropicLLMClient` instance (≈ one
    `validate.py` run, since one client is created per run). Tracks real
    cumulative cost from the API's actual `response.usage.input_tokens` /
    `output_tokens` (not a prompt-length estimate) against a rate table,
    and raises `SpendCapExceeded` *before* making another call once the
    cap is hit (the call that pushes it over the cap is still allowed to
    finish — can't know the cost until the response comes back — so worst
    case is one call's overshoot, not unbounded). Rate table in
    `AnthropicLLMClient._RATES_USD_PER_MTOK` is **approximate
    placeholder pricing** — check console.anthropic.com/settings/billing
    if exact numbers matter, but it's deliberately set generous (i.e.
    likely to overestimate spend, which is the safe direction for a
    guardrail). `validate.py` exposes this as `--max-spend-usd` and
    prints `Estimated spend this run: $X` after every run, and catches
    `SpendCapExceeded` to fail cleanly (exit code 2) rather than crash
    ugly.
  - For context, the normal expected cost per run is tiny: the original
    summary doc estimated **$0.02–$0.08/run** with a Haiku+Sonnet split.
    Even 50-100 full runs across a 3-hour session should stay well under
    $10. The $2/run cap is a safety net for a runaway retry loop or
    unexpectedly huge context, not a number you should expect to bump
    into in normal operation. If you DO hit it in normal operation,
    something is actually wrong (e.g. a retry loop not terminating) —
    investigate rather than just raising the cap.
  - Reads `ANTHROPIC_API_KEY` from the environment unless `api_key` is
    passed explicitly. `pip install anthropic` (already in
    `requirements.txt`).

Same interface-isolation story applies to the execution sandbox:
swapping `provisioning.py`/`execution.py`'s direct `subprocess` calls for
`modal.Sandbox` calls is the remaining v1 upgrade, not yet done — no
Modal account set up this session.

## 5. Compute plan: Mac (dev) → GitHub → EliteBook (compute)

**Why this exists**: the Mac has only 8GB RAM (not 24GB — that was an
earlier wrong assumption, corrected mid-session). An old EliteBook with
32GB RAM is available and is now the intended compute machine. Since the
primary LLM backend is now Anthropic (inference on Anthropic's servers,
not local), the EliteBook doesn't need a GPU or any model weights — it
only needs to run `git clone`/`pip install`/the target repo's eval
script. This is a **strict improvement over the Kaggle path** — no
notebook-specific friction (no `%%writefile` magic, no module-caching
gotchas, no session timeouts, no GPU-quota concerns, see §6 for what
those cost us).

**EliteBook setup** (plain terminal, no notebook):
```bash
git clone git@github.com:suleman1412/buildathon-claude.git
cd buildathon-claude
git checkout v0
git pull origin v0   # in case work happened on the Mac after this clone
python3 -m pip install -r requirements.txt   # PyMuPDF + anthropic SDK only
export ANTHROPIC_API_KEY=...                  # once the key arrives
python3 validate.py path/to/paper.pdf         # --backend anthropic is the default
```

**Division of labor while the key is pending**: code/architecture work
(editing `src/`, prepping papers, updating this file) can happen on
either machine — it's just text. Anything requiring an actual API call
or a real repo-eval run should happen on the EliteBook once the key
lands, since that's the intended compute machine going forward.

### 5.1 Presentation dashboard (`dashboard/`)

Static, zero-build, zero-dependency HTML/CSS/JS site for showing run
results during the presentation. **Added this session but not yet
exercised against a real run** — `dashboard/data/runs.json` is currently
just `[]`.

- `dashboard/index.html` + `style.css` + `app.js` — fetches
  `data/runs.json`, renders a run-history list: PASS/FAIL badge, claimed
  vs. reproduced metric bars, backend/model/spend, collapsible full
  `reason` text.
- `src/dashboard.py`'s `record_run()` — called automatically by
  `validate.py` after every run (unless `--no-dashboard`), appends a
  record to `dashboard/data/runs.json`. Deliberately swallows all
  exceptions — a dashboard-write failure must never fail the actual
  validation run. Tests: `TestDashboardRecordRun`.
- `run_demo.sh` (repo root, executable) — the one-shot script for the
  EliteBook: `./run_demo.sh path/to/paper.pdf` runs the pipeline then
  serves the dashboard at `localhost:8000` (best-effort auto-opens a
  browser via `open`/`xdg-open`). `./run_demo.sh` with no args (or
  `--dashboard-only`) skips the pipeline and just reopens the dashboard
  on whatever's already in `runs.json` — use this to avoid burning time
  *and* API spend just to look at results again. Smoke-tested locally
  (served correctly, empty-state renders correctly) — not yet tested with
  real run data end-to-end.
- **Deploy**: `dashboard/README.md` has exact Netlify/Vercel steps —
  both point at `buildathon-claude/dashboard` as the base/root directory,
  no build command. `netlify.toml` is already in place for Netlify.
  Updating the live deployed dashboard after a new local run means
  committing + pushing the updated `data/runs.json` (see that README).
- **Not yet done**: an actual DistilBERT (or other) run to populate real
  data, and an actual Netlify/Vercel deploy. Both need either the
  Anthropic key or a manually-constructed `runs.json` record for a dry
  run of the deploy step.

## 6. Kaggle/Colab path (secondary, kept working, not the current plan)

This was the primary plan before the Anthropic key + EliteBook became
available. Still fully functional if needed again (e.g. Anthropic key
delayed past the build window and you want a fallback), but treat it as
secondary now.

**Setup**: import `notebooks/kaggle_colab_runner.ipynb`, GPU T4 x2 +
Internet On in session settings, upload the target PDF as a Kaggle
dataset (`/kaggle/input/<slug>/<file>.pdf`), edit the `PDF_PATH` cell,
run all. Model: `Qwen/Qwen2.5-3B-Instruct-GGUF`
(`qwen2.5-3b-instruct-q4_k_m.gguf`, ~2GB).

**Non-obvious gotchas already paid for, don't rediscover:**
- `%%writefile` cannot have an empty cell body (`src/__init__.py` is
  empty) — fixed with a plain Python cell (`open(...).close()`) instead.
- Re-running a `%%writefile` cell only rewrites the file on disk, it does
  **not** reload an already-`import`ed module in that kernel — need
  `importlib.reload(src.<module>)`, or Kernel → Restart & Run All as the
  guaranteed-clean fallback.
- `llama-cpp-python`'s CUDA wheel install can silently compile from
  source for 10-20+ minutes if the CUDA version doesn't match, especially
  under `%%capture` (hides all progress). Install cell was rewritten to
  try the CUDA wheel, print failures, and explicitly fall back to the
  CPU wheel rather than ever silently compiling.
- PyMuPDF prints a harmless `fitz` deprecation warning on import — ignore it.

The local 3B model was noticeably weaker than Claude at both JSON
discipline (needed `_escape_stray_backslashes`, see §7) and grounded
instruction-following (hallucinated a nonexistent eval command on the
LoRA repo — see §8). This weakness is a big part of why the plan shifted
to Anthropic once a key became available.

## 7. Bugs found & fixed in `src/` (all applied, all tested)

1. **`_parse_json_loose` couldn't handle invalid JSON escapes from the
   local model.** e.g. `"LAN-\GUAGE"` (hyphenation artifact, `\G` isn't a
   valid JSON escape). Fixed with `_escape_stray_backslashes()` in
   `src/llm_client.py`, tried as a fallback parse candidate. Tests:
   `TestJSONRepair`.

2. **`pipeline.py` discarded all failure diagnostics.** On eval failure
   it returned no command/stdout/stderr — undebuggable. Fixed: `reason`
   now includes the exact command tried and a truncated stderr tail.

3. **Added `AnthropicLLMClient` with an enforced spend cap** — see §4.
   Tests: `TestAnthropicSpendCap` (mocked API client, verifies both that
   spend accumulates correctly from `response.usage` and that
   `SpendCapExceeded` actually fires once the cap is crossed).

Fixes 1-2 are synced into `notebooks/kaggle_colab_runner.ipynb`'s
`%%writefile` cells. Fix 3 (`AnthropicLLMClient`) is **not** relevant to
the notebook path and wasn't added there.

## 8. LoRA paper attempt — detailed state (BLOCKED, not abandoned)

Test PDF: the LoRA paper (Hu et al. 2021). Extraction correctly found
`github_repo: https://github.com/microsoft/LoRA`, claimed metric 91.5%
accuracy on GLUE (this number is a plausible-but-unverified conflation of
per-task GLUE scores — worth eyeballing against the paper if it matters
later). **Pipeline mechanics are proven working end-to-end** through
extraction → clone → propose-run-plan → execute → debug-surface — the
remaining blocker is specific to this one repo's 2021-era code, not a
bug in our pipeline. Full blow-by-blow, in order:

- **Round 0**: local model hallucinated an eval command (`eval/eval.py`
  with DART/NLG paths) that doesn't exist — pattern-matched on LoRA's
  NLG/GPT-2 example instead of the claimed GLUE dataset. Expected local-model
  weakness (§6), likely fixed by switching to Claude.
- Diagnosed manually: real GLUE/RoBERTa eval is at `examples/NLU/`.
  `examples/NLU/roberta_large_sst2.sh` is a full 10-epoch, 8-GPU training
  run — infeasible for this session. README's "Evaluate the checkpoints"
  section shows the eval-only pattern instead (load a pretrained LoRA
  adapter checkpoint, `--do_eval` only, single process).
- Downloaded `roberta_large_lora_sst2.bin` from
  `https://github.com/msft-edward/LoRA_private/releases/download/RoBERTa-large/roberta_large_lora_sst2.bin`
  — succeeded (non-obvious, the org name looks like a dead private mirror
  but it worked).
- **Round 1**: `ImportError: cannot import name 'load_metric' from
  'datasets'` — `datasets>=3.0` removed it. Fixed with
  `pip install -q "datasets<3" --upgrade` (landed on 2.21.0).
- **Round 2**: `AttributeError: 'TrainingArguments' object has no
  attribute 'use_deterministic_algorithms'`. Patched directly in the
  cloned repo's `examples/NLU/examples/text-classification/run_glue.py`
  (⚠️ **this patch lives only in the Kaggle session's ephemeral
  filesystem, `/kaggle/working/runs/<pdf-stem>/repo/...` — not saved
  anywhere in this git repo. Lost on session restart or re-clone; not
  relevant if the EliteBook does a fresh clone anyway**):
  ```python
  old = "torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)"
  new = "torch.use_deterministic_algorithms(getattr(training_args, 'use_deterministic_algorithms', False))"
  ```
- **Round 3**: `AttributeError: 'TrainingArguments' object has no
  attribute 'cls_dropout'`, different line. The args dump showed
  clearly-modern HuggingFace fields (`optim=OptimizerNames.ADAMW_TORCH_FUSED`,
  `trackio_space_id`) that didn't exist in 2021. **Root cause**: the
  script imports Kaggle's system-installed modern `transformers`, not
  this repo's vendored fork at `examples/NLU/src/transformers` (which
  defines `apply_lora`, `lora_r`, `cls_dropout`). The editable install of
  the vendored package either never ran or lost the import-path race.
- **Round 4**: tried `PYTHONPATH=<repo>/examples/NLU/src:$PYTHONPATH` to
  force the vendored fork first. **Identical failure, byte-for-byte
  identical args dump** — either the fix wasn't actually re-run with the
  new prefix, or the vendored package isn't a complete standalone
  package, or its editable install actively wins over `PYTHONPATH`. Not
  resolved.
- Also relevant if resuming on the EliteBook: `torch.distributed.launch
  --nproc_per_node=1` (the README's own eval command) initializes a
  distributed process group that typically wants NCCL, which is
  NVIDIA-only — will likely fail differently on the EliteBook (no GPU at
  all) than it did on Kaggle. **For a single-process eval, the launcher
  isn't actually needed** — calling `python
  examples/NLU/examples/text-classification/run_glue.py <same args>`
  directly (no `torch.distributed.launch` wrapper) should sidestep this
  category of problem entirely. Worth doing regardless of platform.

**Three options were on the table when this was last actively discussed,
not yet decided — ask the user rather than assuming:**
1. Switch to an easier paper (§9) for a guaranteed clean PASS first.
2. One more real attempt on LoRA, done properly (clean `pip uninstall
   transformers && pip install -e examples/NLU`, or the README's own
   `conda env create -f environment.yml`) rather than more ad hoc
   patches — and now with Claude (once the key lands) instead of the 3B
   local model doing the debugging, which changes the odds meaningfully
   (see the LLM-fixes-what table in the "can this be reproduced" plan
   discussed this session — Claude plausibly auto-fixes rounds 1-3 via
   `debug_loop` if it's also given `stdout` visibility, not just
   `stderr` — **currently `debug_loop.py` only feeds the LLM `stderr`,
   never `stdout`, and the round-3 root cause was only diagnosable from
   the `stdout` args dump — this is a concrete, worthwhile pipeline
   improvement regardless of which paper you end up using**).
3. Stop here — treat the proven mechanics as sufficient for the v0 demo
   even without a clean PASS on this specific paper.

## 9. Easier paper candidates (discussed, not yet attempted)

Criteria that actually matter for a fast, clean demo (not RAM — anything
up to RoBERTa-large fits in 32GB trivially): **no CUDA-only deps**
(`bitsandbytes`, `flash-attn`, `apex`, `DeepSpeed` — none work without
an NVIDIA GPU, which the EliteBook likely doesn't have), **no
multi-GPU/distributed launch**, **actively-maintained eval code** (not a
frozen research fork like LoRA's, which is what actually caused all of
§8's pain).

Ranked recommendation:
1. **DistilBERT** (Sanh et al. 2019) — top pick. HF Hub already has a
   fine-tuned checkpoint (`distilbert-base-uncased-finetuned-sst-2-english`),
   eval-only via current, actively-maintained
   `transformers/examples/pytorch/text-classification/run_glue.py`.
   SST-2 validation set is 872 examples (~7MB) — minutes on CPU. Cleanest
   test of whether the loop can go extraction → clone → eval →
   verify → **PASS** without any repo archaeology.
2. **Sentence-BERT/SBERT** (Reimers & Gurevych 2019) —
   `UKPLab/sentence-transformers`, actively maintained, STS-benchmark
   eval, no GPU needed.
3. **ALBERT-base** on SST-2/MRPC — same profile as DistilBERT.
4. **RoBERTa-base** (vanilla, not LoRA-adapted) via a public fine-tuned
   checkpoint (e.g. `textattack/roberta-base-SST-2`) + current
   `run_glue.py` — closer to LoRA's domain without the adapter/vendored-fork
   complexity, if that framing matters for the demo narrative.
5. **SetFit** (Tunstall et al. 2022) — few-shot, small models, actively
   maintained `setfit` package, fast even for training (not just eval).

None of these have been attempted yet this session — all pending the
Anthropic key.

## 10. Path to v1 (partially done)

- ~~Swap `llm_client.py`'s backend to Anthropic~~ **Done** (§4), pending
  only an actual key to test live.
- Swap `provisioning.py`/`execution.py`'s direct subprocess calls for
  `modal.Sandbox` — **not done**, no Modal account this session.
- Worth doing regardless of Modal: give `debug_loop` visibility into
  `stdout`, not just `stderr` (see §8, round 3) — this was the concrete,
  general pipeline gap the LoRA attempt surfaced, independent of which
  paper you're validating.
- Also worth doing: `propose_run_plan` only sees a flat top-level repo
  listing (4000-char truncation) + top-level README — this is why it
  never found `examples/NLU/`. A recursive/targeted listing or a
  two-pass "which subdirectory looks relevant" step would generalize
  better across papers with nested eval code.
