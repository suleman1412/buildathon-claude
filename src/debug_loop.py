from __future__ import annotations

from pathlib import Path
from typing import Optional

from .execution import RunResult, extract_metric, run_command, truncate_log
from .llm_client import LLMClient

SYSTEM_PROMPT = """You are debugging a failed attempt to reproduce a research \
paper's evaluation on a CPU-only machine with a tight time budget. You will \
see the command that was tried, why it's considered failed, and a tail of \
its output.

Two distinct failure kinds are possible, shown in "failure_reason":
- The command crashed or timed out: fix the setup (patch_command) and/or \
  replace the command (replacement_command). If it timed out, it's still too \
  large/slow -- shrink it further (fewer epochs/iterations, a smaller data \
  subsample), don't just retry the same scale.
- The command exited successfully but metric_regex found no match in its \
  stdout: the output shown is stdout, not an error. Find the metric named in \
  "claimed_metric_name" in it and propose a corrected "metric_regex" that \
  targets THAT metric specifically (and a "replacement_command" too, if the \
  command itself should print it more clearly instead) -- capturing a \
  different, wrong metric that just happens to also be a number is worse \
  than no match at all.

Do not explain your reasoning outside the JSON and do not offer alternatives. \
Output ONLY the JSON object below, nothing before it and nothing after it:
{
  "patch_command": string,       // a single shell command to run before retrying (e.g. "pip install foo==1.2.3"). Empty string if nothing to run first.
  "replacement_command": string, // a full replacement command to use instead of the original. Empty string to keep retrying the original command unchanged.
  "metric_regex": string,        // a corrected regex (exactly one capture group) if the old one was wrong. Empty string to keep the original regex unchanged.
  "reasoning": string             // one short sentence on what went wrong
}
If you cannot tell what would fix it, set all three non-reasoning fields to empty strings."""


def attempt_patch(
    llm: LLMClient,
    failing_command: str,
    metric_regex: str,
    log_tail: str,
    failure_reason: str,
    repo_dir: Path,
    claimed_metric_name: Optional[str] = None,
) -> tuple[str, str, str]:
    user_prompt = (
        f"Command tried:\n{failing_command}\n\n"
        f"metric_regex used:\n{metric_regex}\n\n"
        f"claimed_metric_name: {claimed_metric_name or 'unknown'}\n\n"
        f"failure_reason: {failure_reason}\n\n"
        f"Output (tail):\n{log_tail}"
    )
    data = llm.complete_json(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
    patch_command = data.get("patch_command") or ""
    replacement_command = data.get("replacement_command") or ""
    new_metric_regex = data.get("metric_regex") or ""
    if patch_command:
        run_command(patch_command, cwd=repo_dir, timeout_sec=300)
    return patch_command, replacement_command, new_metric_regex


def run_with_debug_loop(
    llm: LLMClient,
    command: str,
    metric_regex: str,
    repo_dir: Path,
    max_retries: int,
    truncate_lines: int,
    timeout_sec: int,
    claimed_metric_name: Optional[str] = None,
) -> tuple[RunResult, str, str, Optional[float]]:
    """Returns (result, final_command, final_metric_regex, reproduced_value).

    Retries on two distinct failure modes: the command crashing/timing out,
    or the command succeeding but metric_regex matching nothing in its
    stdout -- the latter was previously a silent dead end (pipeline.py just
    reported "no metric found" with no retry), even though it's the more
    common failure in practice: the LLM's one-shot command guess runs fine,
    but its regex guess for the metric doesn't."""
    result = run_command(command, cwd=repo_dir, timeout_sec=timeout_sec)
    reproduced = extract_metric(result.stdout, metric_regex) if result.returncode == 0 else None
    attempts = 0
    while (result.returncode != 0 or reproduced is None) and attempts < max_retries:
        if result.returncode != 0:
            # stderr can be empty if the command redirected it into stdout itself
            # (e.g. "cmd 2>&1") -- fall back to stdout so the LLM isn't debugging blind
            stderr_tail = truncate_log(result.stderr, truncate_lines)
            if stderr_tail.strip():
                log_tail = stderr_tail
                failure_reason = "the command crashed or timed out (this is its stderr)"
            else:
                log_tail = truncate_log(result.stdout, truncate_lines)
                failure_reason = "the command crashed or timed out with empty stderr (this is its stdout instead)"
        else:
            log_tail = truncate_log(result.stdout, truncate_lines)
            failure_reason = "the command succeeded but metric_regex matched nothing (this is its stdout)"
        patch_command, replacement_command, new_metric_regex = attempt_patch(
            llm, command, metric_regex, log_tail, failure_reason, repo_dir, claimed_metric_name
        )
        if not patch_command and not replacement_command and not new_metric_regex:
            break
        if replacement_command:
            command = replacement_command
        if new_metric_regex:
            metric_regex = new_metric_regex
        result = run_command(command, cwd=repo_dir, timeout_sec=timeout_sec)
        reproduced = extract_metric(result.stdout, metric_regex) if result.returncode == 0 else None
        attempts += 1
    return result, command, metric_regex, reproduced
