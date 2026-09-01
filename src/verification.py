from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .extraction import ExtractedClaim


@dataclass
class VerificationResult:
    claim: ExtractedClaim
    reproduced_value: Optional[float]
    passed: bool
    reason: str

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        repro = f"{self.reproduced_value}" if self.reproduced_value is not None else "N/A"
        return f"""# Validation Report: {self.claim.title}

| Field | Value |
|---|---|
| Repo | {self.claim.github_repo or 'unknown'} |
| Dataset | {self.claim.dataset or 'unknown'} |
| Claimed metric | {self.claim.claimed_metric_name} = {self.claim.claimed_metric_value}{self.claim.claimed_metric_unit} |
| Reproduced metric | {repro}{self.claim.claimed_metric_unit} |
| **Result** | **{status}** |

**Notes:** {self.reason}

_{self.claim.eval_notes}_
"""


def verify(claim: ExtractedClaim, reproduced_value: Optional[float], tolerance_relative: float) -> VerificationResult:
    if reproduced_value is None:
        return VerificationResult(claim, None, False, "Could not extract a reproduced metric from the eval output.")
    claimed = claim.claimed_metric_value
    if claimed == 0:
        return VerificationResult(claim, reproduced_value, False, "Claimed metric value unknown; cannot compare.")
    relative_diff = abs(reproduced_value - claimed) / abs(claimed)
    passed = relative_diff <= tolerance_relative
    reason = f"Reproduced value differs from claimed by {relative_diff:.1%} (tolerance: {tolerance_relative:.0%})."
    return VerificationResult(claim, reproduced_value, passed, reason)
