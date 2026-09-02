#!/usr/bin/env bash
# One-shot demo runner: validate a paper, then serve the dashboard locally.
#
# Usage:
#   ./run_demo.sh path/to/paper.pdf [extra validate.py flags]   # run pipeline, then open dashboard
#   ./run_demo.sh                                                # just (re)open dashboard on existing runs
#   ./run_demo.sh --dashboard-only                                # same as above, explicit
#
# Re-running with a PDF re-runs the full pipeline (costs API spend + time)
# -- use no-args / --dashboard-only to just reopen results you already have.
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" != "" ] && [ "${1:-}" != "--dashboard-only" ]; then
  PDF="$1"
  shift
  if [ "${ANTHROPIC_API_KEY:-}" = "" ]; then
    echo "WARNING: ANTHROPIC_API_KEY is not set -- --backend anthropic (the default) will fail." >&2
  fi
  echo "==> Running validator on $PDF"
  python3 validate.py "$PDF" "$@"
else
  echo "==> No PDF given -- skipping pipeline, serving existing dashboard/data/runs.json"
fi

echo "==> Starting dashboard at http://localhost:8000"
cd dashboard
python3 -m http.server 8000 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

sleep 1
if command -v open >/dev/null 2>&1; then
  open "http://localhost:8000" 2>/dev/null || true
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:8000" 2>/dev/null || true
else
  echo "Open http://localhost:8000 in your browser."
fi

echo "Press Ctrl+C to stop the dashboard server."
wait $SERVER_PID
