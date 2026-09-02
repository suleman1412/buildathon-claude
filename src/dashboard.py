from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .verification import VerificationResult


def record_run(
    result: VerificationResult,
    *,
    backend: str,
    model: str | None,
    spend_usd: float | None,
    pdf_path: str,
    dashboard_data_path: Path,
) -> None:
    """Append this run's summary to the dashboard's runs.json so the
    static dashboard (dashboard/index.html) has something to render.
    Best-effort: never raises -- a dashboard write failure shouldn't fail
    the actual validation run."""
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pdf": Path(pdf_path).name,
            "paper_title": result.claim.title,
            "github_repo": result.claim.github_repo,
            "dataset": result.claim.dataset,
            "claimed_metric_name": result.claim.claimed_metric_name,
            "claimed_metric_value": result.claim.claimed_metric_value,
            "claimed_metric_unit": result.claim.claimed_metric_unit,
            "reproduced_value": result.reproduced_value,
            "passed": result.passed,
            "reason": result.reason,
            "backend": backend,
            "model": model,
            "spend_usd": spend_usd,
        }
        dashboard_data_path.parent.mkdir(parents=True, exist_ok=True)
        runs = []
        if dashboard_data_path.exists():
            try:
                runs = json.loads(dashboard_data_path.read_text())
            except json.JSONDecodeError:
                runs = []
        runs.append(record)
        dashboard_data_path.write_text(json.dumps(runs, indent=2))
    except Exception:
        pass
