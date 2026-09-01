#!/usr/bin/env python3
"""CLI entrypoint. Intended to run inside the Kaggle/Colab notebook with a
real (llama_cpp) LLM backend -- see notebooks/kaggle_colab_runner.ipynb.

Can also run locally with --backend mock for pipeline wiring tests only
(no model, no real repo execution)."""
from __future__ import annotations

import argparse
import sys

from src.config import Config
from src.llm_client import get_llm_client
from src.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a research paper's claimed metric.")
    parser.add_argument("pdf_path", help="Path to the paper PDF")
    parser.add_argument("--backend", choices=["llama_cpp", "mock"], default="llama_cpp")
    parser.add_argument("--model-path", help="Path to GGUF model file (required for --backend llama_cpp)")
    parser.add_argument("--out", default="report.md", help="Where to write the markdown report")
    args = parser.parse_args()

    if args.backend == "llama_cpp" and not args.model_path:
        parser.error("--model-path is required for --backend llama_cpp")

    kwargs = {"model_path": args.model_path} if args.backend == "llama_cpp" else {}
    llm = get_llm_client(args.backend, **kwargs)
    config = Config()

    result = run_pipeline(args.pdf_path, llm, config)
    report = result.to_markdown()
    print(report)
    with open(args.out, "w") as f:
        f.write(report)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
