from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .llm_client import LLMClient

SYSTEM_PROMPT = """You are an expert research-paper analyst. Given raw text \
extracted from an academic paper, extract the information needed to \
reproduce its headline result using a Hugging Face Hub model + dataset \
(NOT by cloning an arbitrary GitHub repo -- that path is unreliable and \
out of scope here). Respond with ONLY a single JSON object, no prose, no \
markdown fences, matching exactly this schema:

{
  "title": string,
  "github_repo": string or null,        // official code repo URL if mentioned, informational only
  "hf_model_id": string or null,        // Hugging Face Hub model id that reproduces the headline result,
                                         // e.g. "distilbert-base-uncased-finetuned-sst-2-english". null if
                                         // you don't know of one that reproduces this specific paper's claim.
  "hf_dataset_id": string or null,      // fully-qualified "namespace/name" Hugging Face Hub dataset id --
                                         // the modern `datasets` library needs the namespace, not the old
                                         // bare legacy id (e.g. "nyu-mll/glue", NOT "glue")
  "hf_dataset_config": string or null,  // dataset config/subset if needed, e.g. "sst2" (null if the dataset has no configs)
  "hf_dataset_split": string,           // e.g. "validation" -- default "validation" if unsure
  "text_field": string,                 // name of the input text column in that dataset split, e.g. "sentence"
  "label_field": string,                // name of the label column, e.g. "label"
  "dataset": string or null,            // human-readable dataset name/description, informational
  "claimed_metric_name": string,        // e.g. "accuracy", "F1", "BLEU"
  "claimed_metric_value": number,       // the headline number reported, as a plain float (e.g. 95.1 not "95.1%")
  "claimed_metric_unit": string,        // e.g. "%", "points"
  "eval_notes": string                  // 1-3 sentences on which model/dataset/split produced this number
}

If a field cannot be determined from the text, use null (or 0 for the metric value, but only if truly absent).
Leave hf_model_id null rather than guessing a model id that doesn't exist."""


@dataclass
class ExtractedClaim:
    title: str
    github_repo: Optional[str]
    dataset: Optional[str]
    claimed_metric_name: str
    claimed_metric_value: float
    claimed_metric_unit: str
    eval_notes: str
    hf_model_id: Optional[str] = None
    hf_dataset_id: Optional[str] = None
    hf_dataset_config: Optional[str] = None
    hf_dataset_split: str = "validation"
    text_field: str = "text"
    label_field: str = "label"

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractedClaim":
        return cls(
            title=d.get("title") or "Unknown",
            github_repo=d.get("github_repo") or None,
            hf_model_id=d.get("hf_model_id") or None,
            hf_dataset_id=d.get("hf_dataset_id") or None,
            hf_dataset_config=d.get("hf_dataset_config") or None,
            hf_dataset_split=d.get("hf_dataset_split") or "validation",
            text_field=d.get("text_field") or "text",
            label_field=d.get("label_field") or "label",
            dataset=d.get("dataset") or None,
            claimed_metric_name=d.get("claimed_metric_name") or "unknown metric",
            claimed_metric_value=float(d.get("claimed_metric_value") or 0.0),
            claimed_metric_unit=d.get("claimed_metric_unit") or "",
            eval_notes=d.get("eval_notes") or "",
        )


def extract_claim(llm: LLMClient, paper_text: str) -> ExtractedClaim:
    user_prompt = f"Paper text:\n\n{paper_text}"
    data = llm.complete_json(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
    return ExtractedClaim.from_dict(data)
