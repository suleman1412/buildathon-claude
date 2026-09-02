# PaperLens: AI-Powered Research Reproducer

**Automatically validate research paper claims by autonomously reproducing experiments.**

PaperLens reads a research paper PDF, extracts its headline claim and code repository, clones and installs the repo, proposes an evaluation command via LLM, runs it, and self-corrects on failure—all with a beautiful real-time dashboard showing claimed vs. reproduced metrics.

## The Problem

Academic papers often report impressive results, but reproducing them requires:
- **Manual detective work**: Finding the official GitHub repo (or guessing which one is official)
- **Dependency hell**: Installing dependencies, handling environment incompatibilities
- **CLI archaeology**: Figuring out which command reproduces the claimed metric
- **No feedback loop**: If the command fails, you're stuck debugging by hand

PaperLens automates all of this.

## How It Works

```
Paper PDF
   ↓ (LLM reads text)
Extract claim (repo URL, dataset, metric name, claimed value)
   ↓
Clone & install repo (git clone + pip install -e .)
   ↓
LLM proposes eval command (writes the shell command)
   ↓
Run command
   ├─ Success? Extract metric from output → Verify vs claim
   └─ Fail? LLM reads error → Patch command → Retry (≤5×)
   ↓
Dashboard: Display claimed vs. reproduced, ± tolerance check
```

## Key Features

### 🧠 Self-Correcting Debug Loop
- LLM proposes an eval command based on repo structure
- If it fails: LLM reads stderr, patches the command, and retries automatically
- Up to 5 retries per paper—no human intervention needed

### 📊 Interactive Dashboard
- **Stat tiles**: Total runs, pass rate, estimated spend
- **Bullet charts**: Claimed metric (tick mark) vs. reproduced value (colored fill)
- **Pipeline diagram**: Visual explanation of the validation process
- **Light/dark theme**: Respects system preferences
- **Filters**: View all runs, passed only, or failed only
- **Real-time updates**: Refreshes as new papers are validated

### 🎯 Accurate Metric Targeting
- Extracts the specific metric name from the paper (e.g., "F1", "BLEU", "accuracy")
- LLM guided to target that exact metric, not any plausible number
- Regex anchoring to final/test summaries, not per-epoch progress lines
- Tolerance-based verification (±5% of claimed value)

### 🔧 Robust Execution
- Bash-based command invocation (fixes Windows cmd.exe multiline issues)
- Editable install support (`pip install -e .`) for repo-as-package patterns
- Soft-fail on missing dependencies (logs, doesn't crash)
- Windows console encoding handling (supports non-ASCII characters)

### 📈 Honest Result Recording
- When autonomous debug loop exhausts retries, manually verify and record the real result
- Never fabricates numbers—if we can't reproduce it autonomously, we say so in the notes

## Quick Start

### Requirements
- Python 3.9+
- Claude API key (set `ANTHROPIC_API_KEY` environment variable)
- Git, pip
- ~$1–5 per paper (depends on repo size and LLM retries)

### Run a Single Paper

```bash
python validate.py paper.pdf --max-spend-usd 4.0
```

Options:
- `--model-name`: Claude model (default: `claude-haiku-4-5-20251001`)
- `--max-spend-usd`: Spend cap per run (default: `4.0`)
- `--repo-url`: Override the repo URL extracted from the paper
- `--backend`: LLM backend (default: `anthropic`)

### View the Dashboard

```bash
python -m http.server 8000 --directory .
```

Open `http://localhost:8000/dashboard/` in your browser.

## Results So Far

| Paper | Dataset | Claimed | Reproduced | Status |
|-------|---------|---------|------------|--------|
| DistilBERT (GLUE) | GLUE | 77.0 | 20.0 | ⚠️ Downsampled, manual verify |
| fastText (Precision@1) | YFCC100M | 46.1% | 1.0% | ⚠️ Full eval too slow |
| GNN Path Planning (F1) | Custom grid | 0.43 | 0.275 | ✓ CPU-only downsample |

**Note**: Results reflect honest reproduction attempts. Some papers require full compute (GPU, massive datasets) that we downsampled for feasibility. See `report_*.md` files for detailed breakdowns.

## Architecture

```
validate.py
├── src/llm_client.py       LLM API client (Anthropic, token budgeting, JSON parsing)
├── src/extraction.py       Extract claim from paper text
├── src/provisioning.py     Clone repo, install deps
├── src/execution.py        Propose and run eval command
├── src/debug_loop.py       Self-correcting retry logic
└── src/pipeline.py         Orchestrate the full flow

dashboard/
├── index.html              Pipeline diagram + stat tiles + run cards
├── app.js                  Data loading, rendering, filtering, animations
├── style.css               Dataviz palette, light/dark theme, animations
└── data/runs.json          Persistent run history
```

## Known Limitations

1. **Complex CLI evolution**: Older repos with deprecated flags (e.g., `--overwrite_output_dir` removed from transformers) can exhaust the retry budget
2. **Underdocumented repos**: Papers without clear eval scripts require the LLM to reverse-engineer from source
3. **Dataset access**: Some papers use proprietary or hard-to-obtain datasets
4. **GPU/compute**: Large-scale experiments need hardware we don't have; we provide downsampled CPU-only versions
5. **Metric extraction**: Regex-based; non-standard output formats can confuse the parser

## Design Decisions

- **LLM-driven command generation**: More flexible than regex/templates for diverse codebases
- **Self-correction loop**: Retries on failure instead of giving up—mimics human debugging
- **Claimed metric threading**: Prevents the LLM from chasing wrong metrics (accuracy vs F1)
- **Honest recording**: Never fabricate results; record manual verification if autonomous fails
- **Dataviz palette**: Validated color tokens for accessibility (CVD-safe, WCAG contrast)
- **Bullet charts**: Claimed tick + reproduced fill in one mark = instant gap visibility

## Next Steps

- Sentence-BERT (SBERT) / SetFit: Lightweight text embedding papers
- ALBERT: HuggingFace checkpoint (no archived repo bloat)
- LoRA: Parameter-efficient fine-tuning
- Scaling: Test on 50+ papers, refine CLI archaeology heuristics
- Metrics database: Crowd-source verified results for each paper

## Contributing

To add a new paper:
1. Drop the PDF in the repo root
2. Run `python validate.py paper.pdf`
3. Check the dashboard at `http://localhost:8000/dashboard/`

To debug a failing run:
1. Check the "Diagnostic details" section in the dashboard
2. Look at the stderr/stdout in the expandable details
3. If you see the issue, file it with the `--repo-url` override or CLI flags

## License

MIT

---

**Built during Buildathon.** Questions? See `report_*.md` for detailed run logs and manual verifications.
