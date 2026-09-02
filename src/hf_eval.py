from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class HFEvalError(RuntimeError):
    pass


@dataclass
class HFEvalResult:
    metric_value: float
    n_examples: int


def run_hf_eval(
    model_id: str,
    dataset_id: str,
    dataset_config: Optional[str],
    dataset_split: str,
    text_field: str,
    label_field: str,
    metric_name: str,
    max_examples: int = 200,
) -> HFEvalResult:
    """Reproduce a claim scoped to: single-sentence text classification, a
    model on Hugging Face Hub, a dataset in the `datasets` library. This is
    deliberately narrow -- it trades generality (won't handle seq2seq,
    NER, custom repos, etc.) for reliability: no git clone, no guessed
    shell commands, no third-party dependency archaeology. Everything here
    is a fixed, tested code path we control.
    """
    try:
        from datasets import load_dataset
        from transformers import pipeline
        import evaluate
    except ImportError as e:
        raise HFEvalError(f"Missing dependency for HF eval: {e}. pip install transformers datasets evaluate torch") from e

    try:
        clf = pipeline("text-classification", model=model_id)
    except Exception as e:
        raise HFEvalError(f"Could not load model {model_id!r} via transformers pipeline: {e}") from e

    # A few well-known legacy bare ids the modern `datasets` library no
    # longer resolves without a namespace -- safety net in case the LLM
    # emits the old-style id despite the prompt asking for "namespace/name".
    _KNOWN_NAMESPACES = {"glue": "nyu-mll/glue", "super_glue": "aps/super_glue"}
    resolved_dataset_id = _KNOWN_NAMESPACES.get(dataset_id, dataset_id)

    try:
        ds = load_dataset(resolved_dataset_id, dataset_config, split=dataset_split)
    except Exception as e:
        raise HFEvalError(
            f"Could not load dataset {resolved_dataset_id!r} (config={dataset_config!r}, split={dataset_split!r}): {e}"
        ) from e

    if text_field not in ds.column_names or label_field not in ds.column_names:
        raise HFEvalError(
            f"Dataset columns {ds.column_names} don't contain text_field={text_field!r} / label_field={label_field!r}"
        )

    n = min(max_examples, len(ds))
    ds = ds.select(range(n))

    label2id = getattr(clf.model.config, "label2id", None)

    predictions = []
    references = []
    for example in ds:
        try:
            output = clf(example[text_field], truncation=True)[0]
        except Exception as e:
            raise HFEvalError(f"Inference failed on an example: {e}") from e

        pred_label = output["label"]
        if label2id and pred_label in label2id:
            pred_id = label2id[pred_label]
        else:
            # fall back: label string looks like "LABEL_0" / "LABEL_1"
            try:
                pred_id = int(pred_label.rsplit("_", 1)[-1])
            except (ValueError, AttributeError):
                raise HFEvalError(f"Could not map predicted label {pred_label!r} to a class id")

        predictions.append(pred_id)
        references.append(example[label_field])

    try:
        metric = evaluate.load(metric_name)
        result = metric.compute(predictions=predictions, references=references)
    except Exception as e:
        raise HFEvalError(f"Could not compute metric {metric_name!r}: {e}") from e

    value = result.get(metric_name)
    if value is None:
        value = next(iter(result.values()))

    return HFEvalResult(metric_value=float(value) * 100, n_examples=n)
