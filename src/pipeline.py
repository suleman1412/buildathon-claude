from __future__ import annotations

from .config import Config
from .extraction import extract_claim
from .hf_eval import HFEvalError, run_hf_eval
from .llm_client import LLMClient
from .pdf_extract import extract_text
from .verification import VerificationResult, verify


def run_pipeline(pdf_path: str, llm: LLMClient, config: Config) -> VerificationResult:
    """v0 scope: Hugging Face Hub-native reproduction only. Cloning
    arbitrary GitHub repos and guessing a shell command to run them
    (src/provisioning.py, src/execution.py, src/debug_loop.py -- kept in
    the codebase, not deleted) turned out unreliable in practice across
    every paper tried (LoRA, DistilBERT, fastText all hit real
    environment/dependency archaeology). This path instead uses a fixed,
    tested eval harness (src/hf_eval.py) against a HF Hub model + dataset
    -- no cloning, no guessed commands, no third-party 2021-era code."""
    paper_text = extract_text(pdf_path)
    claim = extract_claim(llm, paper_text)

    if not claim.hf_model_id or not claim.hf_dataset_id:
        return VerificationResult(
            claim, None, False,
            "No Hugging Face Hub model/dataset identified for this claim -- "
            "v0 only reproduces claims backed by a HF Hub model + `datasets` "
            "entry (not arbitrary GitHub repos). "
            f"hf_model_id={claim.hf_model_id!r}, hf_dataset_id={claim.hf_dataset_id!r}",
        )

    try:
        eval_result = run_hf_eval(
            model_id=claim.hf_model_id,
            dataset_id=claim.hf_dataset_id,
            dataset_config=claim.hf_dataset_config,
            dataset_split=claim.hf_dataset_split,
            text_field=claim.text_field,
            label_field=claim.label_field,
            metric_name=_normalize_metric_name(claim.claimed_metric_name),
            max_examples=config.hf_eval_max_examples,
        )
    except HFEvalError as e:
        return VerificationResult(claim, None, False, f"HF eval failed: {e}")

    reason_suffix = f" (evaluated on {eval_result.n_examples} examples)"
    result = verify(claim, eval_result.metric_value, config.metric_tolerance_relative)
    return VerificationResult(result.claim, result.reproduced_value, result.passed, result.reason + reason_suffix)


def _normalize_metric_name(name: str) -> str:
    """Map a claimed metric name to a HF `evaluate` metric id."""
    key = name.strip().lower()
    return {
        "accuracy": "accuracy",
        "acc": "accuracy",
        "f1": "f1",
        "f1-score": "f1",
    }.get(key, "accuracy")
