from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient

SYSTEM_PROMPT = """You are helping reproduce a research paper's result inside \
a Python environment where the paper's official code repository has already \
been cloned and its requirements.txt installed. Given a partial file listing \
of the repo and its README, propose how to run its evaluation.

Respond with ONLY a single JSON object, no prose:
{
  "command": string,       // a single shell command to run from the repo root that runs evaluation and prints a final metric to stdout
  "metric_regex": string   // a Python regex with exactly one capture group that extracts the final numeric metric from that command's stdout
}"""


@dataclass
class RunPlan:
    command: str
    metric_regex: str


def propose_run_plan(llm: LLMClient, repo_dir: Path, dataset: Optional[str]) -> RunPlan:
    listing = "\n".join(
        str(p.relative_to(repo_dir))
        for p in sorted(repo_dir.rglob("*"))
        if p.is_file() and ".git" not in p.parts
    )[:4000]
    readme = ""
    for name in ("README.md", "README.rst", "README.txt"):
        f = repo_dir / name
        if f.exists():
            readme = f.read_text(errors="ignore")[:4000]
            break
    user_prompt = (
        f"Dataset used for the headline result: {dataset or 'unknown'}\n\n"
        f"Repo file listing:\n{listing}\n\nREADME:\n{readme}"
    )
    data = llm.complete_json(SYSTEM_PROMPT, user_prompt, max_tokens=500)
    return RunPlan(command=data["command"], metric_regex=data["metric_regex"])


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str


def run_command(command: str, cwd: Path, timeout_sec: int = 900) -> RunResult:
    try:
        result = subprocess.run(
            command, shell=True, cwd=str(cwd),
            capture_output=True, text=True, timeout=timeout_sec,
        )
        return RunResult(result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="ignore")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode(errors="ignore")
        return RunResult(-1, stdout, stderr + "\n[TIMED OUT]")


def extract_metric(stdout: str, metric_regex: str) -> Optional[float]:
    match = re.search(metric_regex, stdout)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (ValueError, IndexError):
        return None


def truncate_log(text: str, max_lines: int = 30) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-max_lines:])
