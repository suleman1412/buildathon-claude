# AI Research Paper Validator -- v0

Reads a research paper PDF, extracts its headline claim and code repo, clones
and runs the repo's evaluation, and reports whether the claimed metric
reproduces.

This is `v0` of the architecture in `validator_project_summary.md`, adapted
so no paid API keys are needed yet:

| Summary doc (v1) | v0 (this) |
|---|---|
| Claude Haiku/Sonnet | Local quantized Qwen2.5-3B-Instruct (GGUF, via `llama-cpp-python`) |
| Modal sandbox | Kaggle/Colab notebook's own GPU VM |

All model inference and repo execution happens **inside the Kaggle/Colab
notebook's GPU runtime** -- never on your laptop.

## Layout

```
src/               pipeline package (backend-agnostic; import errors are
                    fine locally for the llama_cpp backend, only "mock" runs here)
  llm_client.py     pluggable LLM backend: mock | llama_cpp | (anthropic, later)
  pdf_extract.py    PDF -> raw text (PyMuPDF)
  extraction.py     raw text -> structured claim (repo, dataset, claimed metric)
  provisioning.py   git clone + pip install
  execution.py      LLM proposes a run command + metric regex; runs it
  debug_loop.py     on failure, LLM proposes a patch command, retries (capped)
  verification.py   compare reproduced vs. claimed metric -> markdown report
  pipeline.py       orchestrates the above end to end
validate.py         CLI entrypoint (python validate.py paper.pdf --backend ...)
notebooks/
  kaggle_colab_runner.ipynb   self-contained notebook: installs deps,
                              downloads the model, writes out src/, runs it
tests/
  test_pipeline_mock.py       local wiring tests -- MockLLMClient + a trivial
                              synthetic script, no model, no network, no GPU
```

## Running it

**On Kaggle or Colab (the real thing):**

Upload `notebooks/kaggle_colab_runner.ipynb`, enable a GPU + internet, run
all cells, provide a PDF when prompted. It's self-contained -- it writes out
`src/` itself via `%%writefile`, so it doesn't need this repo pushed anywhere.

**Locally (wiring tests only, no model):**

```bash
cd buildathon-claude
python3 -m unittest tests.test_pipeline_mock -v
```

This exercises extraction parsing, metric-regex extraction, verification
logic, and full pipeline orchestration against a `MockLLMClient` and a
two-line synthetic script -- it proves the plumbing works without touching a
GPU or downloading anything.

`python3 validate.py paper.pdf --backend mock` also runs, but will hit real
`git clone` / `pip install` against whatever `github_repo` the mock LLM
"extracts" -- only useful if you queue mock responses yourself; it's not a
realistic end-to-end run. The notebook is the real v0 entrypoint.

## Known v0 limitations

- **Text-only PDF extraction** (no vision model) -- figures and complex
  tables may be missed.
- **Eval command is guessed**: the LLM proposes both the shell command to run
  evaluation and a regex to pull the metric out of stdout, from the repo's
  file listing + README. For repos with unusual entrypoints this guess can
  be wrong.
- **Dependency install isn't in the debug loop**: `pip install -r
  requirements.txt` is a single best-effort attempt; only the *eval run*
  step gets LLM-driven retries.
- **No sandbox beyond the Kaggle/Colab VM itself** -- acceptable for
  personal testing against trusted repos, not for untrusted code.
- **Small local model**: Qwen2.5-3B is far weaker than Claude at JSON
  extraction and debugging; expect more wrong guesses than the v1 architecture.

## Path to v1

Swap `llm_client.py`'s backend from `llama_cpp` to `anthropic` (Haiku for
extraction, Sonnet for debugging) and swap `provisioning.py`/`execution.py`'s
direct subprocess calls for `modal.Sandbox` -- `pipeline.py` shouldn't need
to change either time, since every stage only talks to the `LLMClient`
interface.
