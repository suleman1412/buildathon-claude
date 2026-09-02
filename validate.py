#!/usr/bin/env python3
"""CLI entrypoint. Runs anywhere with Python -- e.g. on a compute machine
(EliteBook, a server, etc.) after `git pull`. Real compute (repo clone,
pip install, eval run) always happens wherever this script runs; model
inference happens wherever the chosen --backend points:
  - --backend anthropic (recommended once you have an API key): inference
    runs on Anthropic's servers, this machine only needs Python + git + pip.
  - --backend llama_cpp: local GGUF model on THIS machine (needs
    llama-cpp-python + a model file -- see notebooks/kaggle_colab_runner.ipynb
    for the Kaggle/Colab GPU path, or run it directly on a beefy local box).
  - --backend mock: no model, no network -- wiring tests only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import Config
from src.dashboard import record_run
from src.llm_client import AnthropicLLMClient, SpendCapExceeded, get_llm_client
from src.pipeline import run_pipeline

DEFAULT_DASHBOARD_DATA = Path(__file__).parent / "dashboard" / "data" / "runs.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a research paper's claimed metric.")
    parser.add_argument("pdf_path", help="Path to the paper PDF")
    parser.add_argument("--backend", choices=["anthropic", "llama_cpp", "mock"], default="anthropic")
    parser.add_argument("--model-path", help="Path to GGUF model file (required for --backend llama_cpp)")
    parser.add_argument("--model", help="Model name (--backend anthropic only; default claude-haiku-4-5-20251001)")
    parser.add_argument(
        "--max-spend-usd", type=float, default=4.0,
        help="Hard spend cap for this run, --backend anthropic only (default $4)",
    )
    parser.add_argument("--out", default="report.md", help="Where to write the markdown report")
    parser.add_argument(
        "--repo-url",
        help="Override the paper's code repo URL (use when the PDF doesn't link its own repo, e.g. a slide deck)",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Skip appending this run to dashboard/data/runs.json",
    )
    args = parser.parse_args()

    if args.backend == "llama_cpp" and not args.model_path:
        parser.error("--model-path is required for --backend llama_cpp")

    if args.backend == "llama_cpp":
        kwargs = {"model_path": args.model_path}
    elif args.backend == "anthropic":
        kwargs = {"max_spend_usd": args.max_spend_usd}
        if args.model:
            kwargs["model"] = args.model
    else:
        kwargs = {}
    llm = get_llm_client(args.backend, **kwargs)
    config = Config()

    try:
        result = run_pipeline(args.pdf_path, llm, config, repo_url_override=args.repo_url)
    except SpendCapExceeded as e:
        print(f"STOPPED: {e}")
        return 2
    finally:
        if isinstance(llm, AnthropicLLMClient):
            print(f"Estimated spend this run: ${llm.spend_usd:.4f}")

    report = result.to_markdown()
    # write the file before printing -- paper text can contain non-ASCII
    # (accents, math symbols) that crashes print() on a non-UTF-8 console
    # (e.g. Windows' default codepage); the report must not be lost either way
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))

    if not args.no_dashboard:
        record_run(
            result,
            backend=args.backend,
            model=getattr(llm, "_model", None),
            spend_usd=getattr(llm, "spend_usd", None),
            pdf_path=args.pdf_path,
            dashboard_data_path=DEFAULT_DASHBOARD_DATA,
        )

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
