from __future__ import annotations

from pathlib import Path

from .execution import RunResult, run_command, truncate_log
from .llm_client import LLMClient

SYSTEM_PROMPT = """You are debugging a failed attempt to run a research \
paper's evaluation script. You will see the failing command and the tail of \
its error output. Respond with ONLY a single JSON object, no prose:

{
  "patch_command": string,  // a single shell command that fixes the problem (e.g. "pip install foo==1.2.3"), to run before retrying
  "reasoning": string       // one sentence on what went wrong
}

If you cannot tell what would fix it, set "patch_command" to an empty string."""


def attempt_patch(llm: LLMClient, failing_command: str, stderr_tail: str, repo_dir: Path) -> str:
    user_prompt = f"Failing command:\n{failing_command}\n\nError output (tail):\n{stderr_tail}"
    data = llm.complete_json(SYSTEM_PROMPT, user_prompt, max_tokens=300)
    patch_command = data.get("patch_command") or ""
    if patch_command:
        run_command(patch_command, cwd=repo_dir, timeout_sec=300)
    return patch_command


def run_with_debug_loop(
    llm: LLMClient,
    command: str,
    repo_dir: Path,
    max_retries: int,
    truncate_lines: int,
    timeout_sec: int,
) -> RunResult:
    result = run_command(command, cwd=repo_dir, timeout_sec=timeout_sec)
    attempts = 0
    while result.returncode != 0 and attempts < max_retries:
        stderr_tail = truncate_log(result.stderr, truncate_lines)
        patch_command = attempt_patch(llm, command, stderr_tail, repo_dir)
        if not patch_command:
            break
        result = run_command(command, cwd=repo_dir, timeout_sec=timeout_sec)
        attempts += 1
    return result
