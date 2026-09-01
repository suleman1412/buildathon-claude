from __future__ import annotations

from pathlib import Path

from .config import Config
from .debug_loop import run_with_debug_loop
from .execution import extract_metric, propose_run_plan, truncate_log
from .extraction import extract_claim
from .llm_client import LLMClient
from .pdf_extract import extract_text
from .provisioning import ProvisioningError, clone_repo, install_requirements
from .verification import VerificationResult, verify


def run_pipeline(pdf_path: str, llm: LLMClient, config: Config) -> VerificationResult:
    paper_text = extract_text(pdf_path)
    claim = extract_claim(llm, paper_text)

    if not claim.github_repo:
        return VerificationResult(claim, None, False, "No GitHub repo found in the paper; cannot reproduce.")

    workdir = Path(config.workdir) / Path(pdf_path).stem
    repo_dir = workdir / "repo"

    try:
        clone_repo(claim.github_repo, repo_dir)
        install_requirements(repo_dir)
    except ProvisioningError as e:
        return VerificationResult(claim, None, False, f"Provisioning failed: {e}")

    plan = propose_run_plan(llm, repo_dir, claim.dataset)
    result = run_with_debug_loop(
        llm, plan.command, repo_dir,
        max_retries=config.max_runtime_retries,
        truncate_lines=config.log_truncate_lines,
        timeout_sec=config.subprocess_timeout_sec,
    )

    if result.returncode != 0:
        stderr_tail = truncate_log(result.stderr, config.log_truncate_lines)
        reason = (
            f"Evaluation script failed after retries.\n\n"
            f"Command tried: `{plan.command}`\n\n"
            f"Stderr (tail):\n```\n{stderr_tail}\n```"
        )
        return VerificationResult(claim, None, False, reason)

    reproduced_value = extract_metric(result.stdout, plan.metric_regex)
    return verify(claim, reproduced_value, config.metric_tolerance_relative)
