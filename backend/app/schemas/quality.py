"""
Data quality schemas.

Mirrors `QualityCheck` in lib/types.ts:

    export interface QualityCheck {
      label: string;
      passed: boolean;
      detail: string;
    }

These are produced by Stage 3 (stats/quality_checks.py, stats/srm.py).
Defined here first so the graph state and API contracts can reference
them before any math is implemented.
"""

from app.schemas.base import CamelModel


class QualityCheck(CamelModel):
    """One row in the frontend's quality-checks list (SRM, outliers, nulls, normality)."""

    label: str
    passed: bool
    detail: str
    # Critical failures invalidate a ship/no-ship decision; non-critical
    # failures remain visible as warnings (e.g. failed normality check).
    critical: bool = False
    # Shown to the user like any other row, but excluded from both
    # `experiment_validity()`'s CAUTION rule and `_assess_confidence()`'s
    # "data quality issue detected" downgrade. Exists specifically for
    # the large-sample normality check: at n>=30/arm, hypothesis_tests.py
    # already justifies Welch's t-test via the Central Limit Theorem
    # WITHOUT relying on normality — so a large-sample Shapiro-Wilk
    # rejection (which is close to guaranteed at real-world sample
    # sizes, e.g. p~1e-28 on this project's own 3,000-user/arm CUPED
    # demo dataset) is expected, harmless, and must not be treated as a
    # genuine data-quality problem that caps confidence at MEDIUM.
    informational: bool = False


class SRMResult(CamelModel):
    """
    Internal, richer result of the Sample Ratio Mismatch check.

    This is NOT sent to the frontend directly — the decision_node /
    report assembly step will fold this into a `QualityCheck` entry
    (label="Sample Ratio Mismatch (SRM)"). Kept separate because the
    graph and the decision node need the raw numbers (chi2, p_value)
    for the `confidenceReason` / `srmWarning` logic, not just the
    boolean pass/fail the UI displays.
    """

    observed_control: int
    observed_variant: int
    expected_ratio: float  # e.g. 0.5 for 50/50
    chi2_statistic: float
    p_value: float
    passed: bool  # p_value >= alpha (no SRM detected)
    arm_counts: list[int] | None = None  # optional multi-arm counts
    # The decision threshold actually used for `passed` above (defaults
    # to stats_thresholds.srm_alpha when the caller doesn't override
    # it — see check_srm/check_srm_multi_arm). Carried alongside the
    # result so the report/evidence can state the threshold explicitly
    # rather than only implying it via pass/fail.
    alpha: float = 0.0
