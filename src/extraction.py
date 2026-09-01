from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .llm_client import LLMClient

SYSTEM_PROMPT = """You are an expert research-paper analyst. Given raw text \
extracted from an academic paper, extract the information needed to \
reproduce its headline result. Respond with ONLY a single JSON object, no \
prose, no markdown fences, matching exactly this schema:

{
  "title": string,
  "github_repo": string or null,   // full https URL to the official code repo, if mentioned
  "dataset": string or null,       // dataset name or HuggingFace dataset id used for the headline result
  "claimed_metric_name": string,   // e.g. "accuracy", "F1", "BLEU"
  "claimed_metric_value": number,  // the headline number reported, as a plain float (e.g. 95.1 not "95.1%")
  "claimed_metric_unit": string,   // e.g. "%", "points"
  "eval_notes": string             // 1-3 sentences on which model/dataset/split produced this number, for someone about to reproduce it
}

If a field cannot be determined from the text, use null (or 0 for the metric value, but only if truly absent)."""


@dataclass
class ExtractedClaim:
    title: str
    github_repo: Optional[str]
    dataset: Optional[str]
    claimed_metric_name: str
    claimed_metric_value: float
    claimed_metric_unit: str
    eval_notes: str

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractedClaim":
        return cls(
            title=d.get("title") or "Unknown",
            github_repo=d.get("github_repo") or None,
            dataset=d.get("dataset") or None,
            claimed_metric_name=d.get("claimed_metric_name") or "unknown metric",
            claimed_metric_value=float(d.get("claimed_metric_value") or 0.0),
            claimed_metric_unit=d.get("claimed_metric_unit") or "",
            eval_notes=d.get("eval_notes") or "",
        )


def extract_claim(llm: LLMClient, paper_text: str) -> ExtractedClaim:
    user_prompt = f"Paper text:\n\n{paper_text}"
    data = llm.complete_json(SYSTEM_PROMPT, user_prompt, max_tokens=800)
    return ExtractedClaim.from_dict(data)
