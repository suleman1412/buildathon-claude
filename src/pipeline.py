from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import Config
from .debug_loop import run_with_debug_loop
from .execution import propose_run_plan, truncate_log
from .extraction import extract_claim
from .llm_client import LLMClient
from .pdf_extract import extract_text
from .provisioning import ProvisioningError, clone_repo, install_requirements
from .verification import VerificationResult, verify


def run_pipeline(
    pdf_path: str, llm: LLMClient, config: Config, repo_url_override: Optional[str] = None
) -> VerificationResult:
    paper_text = extract_text(pdf_path)
    claim = extract_claim(llm, paper_text)

    if repo_url_override:
        # e.g. a slide deck or paper that describes its own results without
        # linking its own repo in the text -- extraction has nothing to find
        claim.github_repo = repo_url_override

    if not claim.github_repo:
        return VerificationResult(claim, None, False, "No GitHub repo found in the paper; cannot reproduce.")

    workdir = Path(config.workdir) / Path(pdf_path).stem
    repo_dir = workdir / "repo"

    try:
        clone_repo(claim.github_repo, repo_dir)
        install_requirements(repo_dir)
    except ProvisioningError as e:
        return VerificationResult(claim, None, False, f"Provisioning failed: {e}")

    plan = propose_run_plan(llm, repo_dir, claim.dataset, claim.claimed_metric_name)
    result, final_command, final_metric_regex, reproduced_value = run_with_debug_loop(
        llm, plan.command, plan.metric_regex, repo_dir,
        max_retries=config.max_runtime_retries,
        truncate_lines=config.log_truncate_lines,
        timeout_sec=config.subprocess_timeout_sec,
        claimed_metric_name=claim.claimed_metric_name,
    )

    if result.returncode != 0:
        # commands sometimes redirect their own stderr into stdout (e.g. "cmd
        # 2>&1"), so show both -- stderr alone can be silently empty then
        stderr_tail = truncate_log(result.stderr, config.log_truncate_lines)
        stdout_tail = truncate_log(result.stdout, config.log_truncate_lines)
        reason = (
            f"Evaluation script failed after retries (exit code {result.returncode}).\n\n"
            f"Command tried: `{final_command}`\n\n"
            f"Stderr (tail):\n```\n{stderr_tail}\n```\n\n"
            f"Stdout (tail):\n```\n{stdout_tail}\n```"
        )
        return VerificationResult(claim, None, False, reason)

    if reproduced_value is None:
        stdout_tail = truncate_log(result.stdout, config.log_truncate_lines)
        reason = (
            f"Command ran successfully but no metric matched after retries.\n\n"
            f"Command tried: `{final_command}`\n\n"
            f"metric_regex tried: `{final_metric_regex}`\n\n"
            f"Stdout (tail):\n```\n{stdout_tail}\n```"
        )
        return VerificationResult(claim, None, False, reason)

    return verify(claim, reproduced_value, config.metric_tolerance_relative)
