# Dashboard

Static, no-build, no-dependency HTML/CSS/JS. Reads `data/runs.json`
(a JSON array appended to by `validate.py` after each pipeline run) and
renders it as a run history with pass/fail badges and claimed-vs-reproduced
metric bars.

## Local preview

From the repo root: `./run_demo.sh --dashboard-only` (or with no args) —
serves this directory at http://localhost:8000 without re-running the
pipeline. Passing a PDF path instead runs the pipeline first, then serves.

## Deploy (Netlify)

1. New site from Git → pick this repo, branch `v0`.
2. **Base directory**: `buildathon-claude/dashboard`
3. **Build command**: (leave empty)
4. **Publish directory**: `.` (relative to base directory — already set in `netlify.toml`)
5. Deploy. Every `git push` to `v0` with an updated `data/runs.json` redeploys automatically.

## Deploy (Vercel)

1. New Project → import this repo.
2. **Root Directory**: `buildathon-claude/dashboard`
3. Framework preset: **Other** (no build step).
4. Deploy.

## Updating the data after a new run

`validate.py` appends to `data/runs.json` automatically (disable with
`--no-dashboard`). For the deployed site to show a new run, commit and
push the updated `data/runs.json`:

```bash
git add dashboard/data/runs.json
git commit -m "Add <paper> validation run"
git push origin v0
```
