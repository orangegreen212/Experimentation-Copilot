"""
Covers:

  1. SRM PASS -> normal inference.
  2. SRM FAIL -> causal inference blocked (decision=INVALID, no stats,
     no ship/no-ship recommendation, no causal effect inference).
  3. SRM FAIL + explicit "Stratified Analysis by <column>" request ->
     non-causal DIAGNOSTIC stratification runs, but still no causal
     stratified inference (no eligibility/estimate) and no change to
     the overall INVALID decision.

Uses the real compiled graph end-to-end (classifier -> planner ->
validation -> [experiment] -> decision), same pattern as
tests/graph/test_stratification_integration.py, so this proves the
routing/gating itself, not just the individual functions in isolation.
"""

import numpy as np
import pandas as pd

from app.core.dataset_store import store_dataset
from app.graph.graph_builder import experiment_review_graph
from app.schemas.report import Decision, ExperimentValidity
from app.schemas.settings import AnalysisSettings
from app.schemas.stratification import StratificationStatus


def _ab_dataset(n_control: int, n_treatment: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for arm, n, rate in [("control", n_control, 0.10), ("treatment", n_treatment, 0.14)]:
        conv = rng.binomial(1, rate, n)
        device = rng.choice(["mobile", "desktop"], n)
        for c, d in zip(conv, device):
            rows.append({"user_id": uid, "group": arm, "device_type": d, "converted": int(c)})
            uid += 1
    return pd.DataFrame(rows)


def test_srm_pass_runs_normal_causal_inference():
    """Requirement 1 — a balanced dataset (SRM passes) must still run the
    full hypothesis test and produce a real ship/no-ship decision."""
    df = _ab_dataset(n_control=1000, n_treatment=1000)
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Evaluate this experiment",
            "settings": AnalysisSettings(),
        }
    )

    report = final_state["report"]
    assert report.srm_warning is False
    assert report.experiment_validity == ExperimentValidity.VALID
    assert report.stats != []
    assert report.decision in (
        Decision.GO,
        Decision.GO_WITH_CAUTION,
        Decision.NO_GO,
        Decision.INCONCLUSIVE,
    )
    assert report.decision != Decision.INVALID


def test_srm_fail_blocks_all_causal_inference():
    """Requirement 2 — a severely imbalanced dataset (SRM fails) must
    skip Experiment entirely: INVALID decision, no stats, no
    ship/no-ship recommendation, no causal effect inference."""
    df = _ab_dataset(n_control=950, n_treatment=50)
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Evaluate this experiment",
            "settings": AnalysisSettings(),
        }
    )

    report = final_state["report"]
    assert report.srm_warning is True
    assert report.experiment_validity == ExperimentValidity.INVALID
    assert report.decision == Decision.INVALID
    assert report.stats == []  # no hypothesis test was run
    assert report.practical_significance is None
    # No causal ship/no-ship recommendation — the primary recommendation
    # must be the INVALID template, never a GO/NO-GO framing (a
    # secondary, clearly-attributed "Methodology guidance" line may
    # also be present when knowledge_base ran alongside validation —
    # that line is never a ship/no-ship recommendation either).
    assert report.recommendations[0].startswith("INVALID —")
    assert not any(rec.strip().upper().startswith(("GO", "NO-GO")) for rec in report.recommendations)


def test_srm_fail_with_explicit_stratification_request_runs_diagnostic_only():
    """Requirement 3 — SRM FAIL + an explicit stratified-analysis
    request must still produce a DIAGNOSTIC (non-causal) breakdown,
    while causal inference (both the ordinary hypothesis test AND
    causal stratified inference) remains fully blocked."""
    df = _ab_dataset(n_control=950, n_treatment=50)
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Stratified Analysis by device_type",
            "settings": AnalysisSettings(),
        }
    )

    report = final_state["report"]

    # The overall gate is completely unweakened.
    assert report.srm_warning is True
    assert report.experiment_validity == ExperimentValidity.INVALID
    assert report.decision == Decision.INVALID
    assert report.stats == []

    # But a diagnostic stratification breakdown IS available.
    strat = report.stratification
    assert strat is not None
    assert strat.status == StratificationStatus.DIAGNOSTIC
    assert strat.stratification_column == "device_type"

    # Causal stratified inference itself stays blocked — no eligibility
    # check (that's the causal-eligibility gate) and no combined
    # treatment-effect estimate.
    assert strat.eligibility is None
    assert strat.estimate is None
    assert "BLOCKED" in strat.not_run_reason
    assert "Sample Ratio Mismatch" in strat.not_run_reason

    # The diagnostic block itself is purely descriptive.
    diag = strat.diagnostic
    assert diag is not None
    assert diag.stratification_column == "device_type"
    assert diag.total_n == 1000
    assert sum(diag.allocation_by_variant.values()) == 1000
    assert len(diag.strata) >= 1
    assert diag.label == "Descriptive / Diagnostic only — not causal inference."
    # Per-stratum descriptive facts are present (composition + rates).
    for stratum in diag.strata:
        assert stratum.total_n == stratum.control_n + stratum.variant_n
    # Diagnostic never carries any causal-inference fields — the
    # DiagnosticStratification schema itself has none, but assert
    # explicitly that this object type never leaks a treatment-effect
    # p-value/CI into the report via `estimate`.
    assert strat.estimate is None


def test_srm_fail_without_explicit_stratification_request_stays_not_run():
    """Non-regression: SRM FAIL with NO stratification request at all
    must still return no `stratification` block worth reading (None),
    completely unaffected by the diagnostic feature."""
    df = _ab_dataset(n_control=950, n_treatment=50)
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Evaluate this experiment",
            "settings": AnalysisSettings(),
        }
    )

    report = final_state["report"]
    assert report.stratification is None
