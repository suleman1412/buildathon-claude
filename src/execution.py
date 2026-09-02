from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient

SYSTEM_PROMPT = """You are helping reproduce a research paper's result inside \
a Python environment where the paper's official code repository has already \
been cloned and its requirements.txt installed. Given a partial file listing \
of the repo and its README, propose how to run its evaluation.

This machine is CPU-only with a tight time budget (a few minutes). Prefer a \
FAST, DOWNSIZED smoke-test version of the real evaluation over the full-scale \
one: cap epochs/iterations at a small number (e.g. 1-3), and subsample the \
dataset if the eval would otherwise use the full thing -- via an existing \
flag (--max_samples, --limit, --subset, -n, etc.) if the script has one, or \
by slicing/truncating the input data yourself (e.g. `head -n 500`) if not. \
Avoid GPU-only flags. It is fine and expected that this produces a much worse \
metric than the paper's full-scale claim -- the goal is a real numeric result \
in a few minutes, not matching the paper's number.

The metric_regex MUST target the SAME named metric the paper claims (given \
below as "Claimed metric name"), not just any plausible-looking number. If \
the eval prints several metrics (loss, accuracy, F1, precision...), pick the \
regex that captures the one matching that name -- a wrong-metric match (e.g. \
capturing Accuracy when the claim is F1) is worse than no match at all.

If training runs multiple epochs, the metric name likely appears once per \
epoch (progress) AND once more in a final summary -- re.search returns the \
FIRST match, so an unanchored regex will silently grab an early, weaker \
epoch's value instead of the final one. Anchor the regex to the final/test \
summary line specifically (e.g. a line starting "Test" or "Final", not "Val" \
or "Epoch"), or make the command itself print only the final line (e.g. \
pipe through `tail`).

Do not explain your reasoning and do not offer alternatives. Output ONLY the \
JSON object below, nothing before it and nothing after it:
{
  "command": string,       // a single shell command to run from the repo root that runs evaluation and prints a final metric to stdout
  "metric_regex": string   // a Python regex with exactly one capture group that extracts the final numeric metric from that command's stdout
}"""


@dataclass
class RunPlan:
    command: str
    metric_regex: str


_EVAL_KEYWORDS = ("eval", "test", "benchmark", "glue", "run_", "score", "metric")


def _relevance_key(rel_path: str, dataset: Optional[str]) -> tuple:
    lower = rel_path.lower()
    dataset_hit = 1 if dataset and dataset.lower() in lower else 0
    keyword_hit = 1 if any(k in lower for k in _EVAL_KEYWORDS) else 0
    depth = lower.count("/")
    # deeper framework dirs (docs/, tests/ fixtures, i18n/) sort worse by default;
    # dataset/keyword hits and shallower paths sort first
    return (-dataset_hit, -keyword_hit, depth, lower)


def propose_run_plan(
    llm: LLMClient, repo_dir: Path, dataset: Optional[str], claimed_metric_name: Optional[str] = None
) -> RunPlan:
    files = [
        str(p.relative_to(repo_dir)).replace("\\", "/")
        for p in repo_dir.rglob("*")
        if p.is_file() and ".git" not in p.parts
    ]
    files.sort(key=lambda rel: _relevance_key(rel, dataset))
    listing = "\n".join(files)[:8000]

    readme = ""
    for name in ("README.md", "README.rst", "README.txt"):
        f = repo_dir / name
        if f.exists():
            readme = f.read_text(errors="ignore")[:4000]
            break
    # a nested README closer to the eval/dataset code is often more specific
    # than the repo's top-level (framework-wide) README
    for rel in files:
        lower = rel.lower()
        if lower.endswith(("readme.md", "readme.rst", "readme.txt")) and any(
            k in lower for k in _EVAL_KEYWORDS
        ) or (dataset and dataset.lower() in lower and lower.endswith("readme.md")):
            nested = (repo_dir / rel).read_text(errors="ignore")[:4000]
            readme += f"\n\n--- {rel} ---\n{nested}"
            break

    user_prompt = (
        f"Dataset used for the headline result: {dataset or 'unknown'}\n\n"
        f"Claimed metric name (metric_regex must target THIS metric): {claimed_metric_name or 'unknown'}\n\n"
        f"Repo file listing (most relevant first):\n{listing}\n\nREADME:\n{readme}"
    )
    data = llm.complete_json(SYSTEM_PROMPT, user_prompt, max_tokens=3000)
    return RunPlan(command=data["command"], metric_regex=data["metric_regex"])


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str


_BASH = shutil.which("bash")


def run_command(command: str, cwd: Path, timeout_sec: int = 900) -> RunResult:
    # invoke bash directly as argv (["bash", "-c", command]) instead of
    # subprocess.run(command, shell=True) -- on Windows, shell=True goes
    # through cmd.exe, which silently mangles multi-line quoted commands
    # (a common LLM-generated pattern, e.g. `python -c "<multi-line script>"`):
    # it returns exit code 0 with empty stdout/stderr instead of erroring,
    # which is worse than a normal crash since nothing looks wrong. bash -c
    # handles these correctly and works the same way cross-platform.
    args = [_BASH, "-c", command] if _BASH else command
    try:
        result = subprocess.run(
            args, shell=(_BASH is None), cwd=str(cwd),
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
