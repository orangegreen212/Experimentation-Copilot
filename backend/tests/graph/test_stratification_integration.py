"""
End-to-end integration test for TRUE stratified analysis — proves it's
genuinely wired through the real graph (classifier -> planner ->
validation -> experiment -> decision -> report), not just correct in
isolation (see tests/stats/test_stratification.py for the isolated
unit tests).

Covers the two headline scenarios from the bug report:
  1. A raw user-level dataset (user_id, timestamp, group, landing_page,
     converted) is classified RAW_USER_LEVEL (Problem 1's fix) and,
     when "Stratified Analysis by landing_page" is explicitly
     selected, the report shows Stratified Analysis (never "Full
     Experiment Review") with stratification correctly rejected as
     INELIGIBLE for the exact stated reason.
  2. A genuinely stratifiable variable (device_type, not associated
     with assignment) produces a real combined estimate.
"""

import numpy as np
import pandas as pd

from app.core.dataset_store import store_dataset
from app.graph.graph_builder import experiment_review_graph
from app.schemas.dataset import DatasetType
from app.schemas.settings import AnalysisSettings
from app.schemas.stratification import StratificationIneligibilityReason, StratificationStatus


def _raw_user_level_bug_dataset(n_per_arm=300, seed=1):
    """Exact schema from the bug report: landing_page is fully determined by group."""
    rng = np.random.default_rng(seed)
    control_conv = rng.binomial(1, 0.10, n_per_arm)
    treatment_conv = rng.binomial(1, 0.12, n_per_arm)
    return pd.DataFrame({
        "user_id": list(range(2 * n_per_arm)),
        "timestamp": pd.date_range("2026-01-01", periods=2 * n_per_arm, freq="min"),
        "group": ["control"] * n_per_arm + ["treatment"] * n_per_arm,
        "landing_page": ["old_page"] * n_per_arm + ["new_page"] * n_per_arm,
        "converted": list(control_conv) + list(treatment_conv),
    })


def _stratifiable_dataset(n_per_stratum_arm=150, seed=2):
    """device_type is NOT determined by group — genuinely stratifiable."""
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for device in ["mobile", "desktop"]:
        base_rate = 0.10 if device == "mobile" else 0.20
        for arm, lift in [("control", 0.0), ("treatment", 0.04)]:
            conv = rng.binomial(1, base_rate + lift, n_per_stratum_arm)
            for c in conv:
                rows.append({"user_id": uid, "group": arm, "device_type": device, "converted": int(c)})
                uid += 1
    return pd.DataFrame(rows)


def test_raw_dataset_classified_correctly_and_stratification_rejected_for_landing_page():
    df = _raw_user_level_bug_dataset()
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Should we ship variant B?",  # phrased like a full-review request on purpose
            "settings": AnalysisSettings(analysis_mode="stratified", stratification_column="landing_page"),
        }
    )

    # Problem 1 — classification fix.
    assert final_state["dataset"].type == DatasetType.RAW_USER_LEVEL

    # Problem 2 — explicit mode survives the whole pipeline, never
    # reclassified as Full Experiment Review despite the ship/evaluate
    # phrasing in user_prompt.
    assert final_state["plan"]["intent_label"] == "Stratified Analysis"

    report = final_state["report"]
    assert report.stratification is not None
    assert report.stratification.status == StratificationStatus.RAN
    assert report.stratification.eligibility.eligible is False
    assert (
        report.stratification.eligibility.ineligibility_reason
        == StratificationIneligibilityReason.PERFECTLY_ASSOCIATED_WITH_ASSIGNMENT
    )
    assert "perfectly associated with experiment assignment" in report.stratification.eligibility.reason
    assert report.stratification.estimate is None

    # The ordinary (non-stratified) hypothesis test must still have run
    # normally — stratification ineligibility must not block or corrupt
    # the standard experiment path.
    assert report.stats != []


def test_valid_stratification_variable_produces_real_combined_estimate():
    df = _stratifiable_dataset()
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Evaluate this experiment",
            "settings": AnalysisSettings(analysis_mode="stratified", stratification_column="device_type"),
        }
    )

    assert final_state["plan"]["intent_label"] == "Stratified Analysis"

    report = final_state["report"]
    assert report.stratification is not None
    assert report.stratification.status == StratificationStatus.RAN
    assert report.stratification.eligibility.eligible is True
    assert report.stratification.estimate is not None
    assert report.stratification.estimate.strata_used == 2
    assert report.stratification.estimate.effect_estimate > 0
    assert report.stats != []


def test_free_text_stratified_analysis_request_survives_full_graph():
    """
    Requirement #11 — end-to-end test using the EXACT free-text user
    request ("Stratified Analysis by landing_page") rather than
    constructing AnalysisSettings directly, proving the fix works
    through the real graph (classifier -> planner -> validation ->
    experiment -> decision -> report), not just at the planner-unit
    level (see tests/graph/test_explicit_analysis_mode.py for those).
    """
    df = _raw_user_level_bug_dataset()
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Stratified Analysis by landing_page",
            "settings": AnalysisSettings(),  # no structured analysis_mode — pure free text
        }
    )

    assert final_state["dataset"].type == DatasetType.RAW_USER_LEVEL
    assert final_state["plan"]["intent_label"] == "Stratified Analysis"
    assert final_state["plan"]["intent_label"] != "Full Experiment Review"

    report = final_state["report"]
    assert report.stratification is not None
    assert report.stratification.eligibility.eligible is False
    assert (
        report.stratification.eligibility.ineligibility_reason
        == StratificationIneligibilityReason.PERFECTLY_ASSOCIATED_WITH_ASSIGNMENT
    )
    assert report.stats != []


def test_free_text_stratified_analysis_request_with_valid_column_end_to_end():
    """Same free-text entry point, but against a genuinely stratifiable variable."""
    df = _stratifiable_dataset()
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "run a stratified analysis using device_type",
            "settings": AnalysisSettings(),
        }
    )

    assert final_state["plan"]["intent_label"] == "Stratified Analysis"
    report = final_state["report"]
    assert report.stratification is not None
    assert report.stratification.eligibility.eligible is True
    assert report.stratification.estimate is not None
    assert report.stratification.estimate.strata_used == 2


def _three_stratum_valid_dataset(n_per_stratum_arm=200, seed=13):
    """
    Fully valid raw user-level dataset: two variants, no conflicting
    assignments, SRM passes, and `device` contains BOTH variants within
    every one of its three strata (mobile/desktop/tablet) — genuinely
    stratifiable, unlike `landing_page` in the bug-report dataset.
    """
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for device, base_rate in [("mobile", 0.10), ("desktop", 0.20), ("tablet", 0.15)]:
        for arm, lift in [("control", 0.0), ("treatment", 0.05)]:
            conv = rng.binomial(1, base_rate + lift, n_per_stratum_arm)
            for c in conv:
                rows.append({"user_id": uid, "group": arm, "device": device, "converted": int(c)})
                uid += 1
    return pd.DataFrame(rows)


def test_valid_three_stratum_device_variable_runs_full_estimator_end_to_end():
    """
    Full requested verification: a valid dataset (two variants, no
    conflicting assignments, SRM passes, device has both variants in
    every stratum) run through the free-text entry point "Stratified
    Analysis by device" must produce a REAL running estimator with:
      - stratum-level descriptive estimates (per-stratum control/variant
        outcome rates, counts, proportions),
      - a combined stratified effect,
      - its standard error,
      - a confidence interval,
      - a p-value,
      - number of observations/units per stratum.
    """
    df = _three_stratum_valid_dataset()
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Stratified Analysis by device",
            "settings": AnalysisSettings(),  # pure free text
        }
    )

    # Validity gates passed normally — this is NOT an invalid-dataset case.
    assert final_state["plan"]["intent_label"] == "Stratified Analysis"
    assert final_state["settings"].stratification_column == "device"

    report = final_state["report"]
    assert report.experiment_validity.value == "VALID"
    assert report.stats != []  # the ordinary hypothesis test ran too

    strat = report.stratification
    assert strat is not None
    assert strat.status == StratificationStatus.RAN
    assert strat.stratification_column == "device"

    eligibility = strat.eligibility
    assert eligibility is not None
    assert eligibility.eligible is True

    # Stratum-level descriptive estimates: 3 strata, each with both
    # variants, real per-arm counts, and a computed outcome rate.
    assert len(eligibility.strata) == 3
    stratum_values = {s.stratum_value for s in eligibility.strata}
    assert stratum_values == {"mobile", "desktop", "tablet"}
    for s in eligibility.strata:
        assert s.has_both_variants is True
        assert s.sufficient is True
        assert s.control_n == 200
        assert s.variant_n == 200
        assert s.total_n == 400
        assert s.proportion_of_total > 0
        assert s.control_outcome_rate is not None
        assert s.variant_outcome_rate is not None

    # Combined stratified effect + full inferential output.
    estimate = strat.estimate
    assert estimate is not None
    assert estimate.strata_used == 3
    assert estimate.effect_estimate > 0  # true lift was positive in every stratum
    assert estimate.standard_error > 0
    assert estimate.ci_lower < estimate.effect_estimate < estimate.ci_upper
    assert 0.0 <= estimate.p_value <= 1.0
    assert isinstance(estimate.significant, bool)


def test_landing_page_still_rejected_as_ineligible_on_original_dataset():
    """
    Re-verification alongside the valid-dataset test above: on the
    ORIGINAL bug-report dataset shape (landing_page perfectly
    determined by group), "Stratified Analysis by landing_page" must
    still be rejected as ineligible for the stated reason — unaffected
    by the device/3-strata fix above.
    """
    df = _raw_user_level_bug_dataset()
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Stratified Analysis by landing_page",
            "settings": AnalysisSettings(),
        }
    )

    report = final_state["report"]
    assert report.experiment_validity.value == "VALID"  # this dataset has no conflicting-assignment issue
    assert report.stratification is not None
    assert report.stratification.status == StratificationStatus.RAN
    assert report.stratification.eligibility.eligible is False
    assert (
        report.stratification.eligibility.ineligibility_reason
        == StratificationIneligibilityReason.PERFECTLY_ASSOCIATED_WITH_ASSIGNMENT
    )
    assert "perfectly associated with experiment assignment" in report.stratification.eligibility.reason
    assert report.stratification.estimate is None
    """
    A dataset with a severe Sample Ratio Mismatch must skip Experiment
    entirely (existing behavior — route_after_validation) — and
    therefore CAUSAL stratification, which only runs inside
    experiment_node, must also never run: no fabricated stratified
    estimate on invalid data. A non-causal DIAGNOSTIC breakdown is,
    however, expected here since a stratification column
    was explicitly requested — see
    test_srm_fail_diagnostic.py for focused coverage of that behavior.
    """
    rng = np.random.default_rng(5)
    n_control, n_treatment = 950, 50  # severe SRM (expected ~50/50)
    rows = []
    uid = 0
    for arm, n, rate in [("control", n_control, 0.10), ("treatment", n_treatment, 0.14)]:
        conv = rng.binomial(1, rate, n)
        device = rng.choice(["mobile", "desktop"], n)
        for c, d in zip(conv, device):
            rows.append({"user_id": uid, "group": arm, "device_type": d, "converted": int(c)})
            uid += 1
    df = pd.DataFrame(rows)
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Evaluate this experiment",
            "settings": AnalysisSettings(analysis_mode="stratified", stratification_column="device_type"),
        }
    )

    report = final_state["report"]
    assert report.srm_warning is True
    assert report.stratification is not None
    # SRM failure + an explicit stratification request surfaces a
    # DIAGNOSTIC (non-causal) breakdown instead of a bare NOT_RUN — see
    # StratificationStatus.DIAGNOSTIC's docstring. Causal inference is
    # still fully blocked: no eligibility (causal check) and no estimate
    # are produced.
    assert report.stratification.status == StratificationStatus.DIAGNOSTIC
    assert report.stratification.eligibility is None
    assert report.stratification.estimate is None
    assert "Sample Ratio Mismatch" in report.stratification.not_run_reason
    assert "BLOCKED" in report.stratification.not_run_reason
    assert report.stratification.diagnostic is not None
    assert report.stratification.diagnostic.stratification_column == "device_type"
    assert report.stratification.diagnostic.total_n == n_control + n_treatment
    assert sum(report.stratification.diagnostic.allocation_by_variant.values()) == n_control + n_treatment
    assert len(report.stratification.diagnostic.strata) > 0
    assert report.stats == []


def test_free_text_stratification_request_not_run_due_to_conflicting_assignments():
    """
    Regression test for the exact reported scenario: a free-text
    request ("Stratified Analysis by landing_page") on a dataset with
    users assigned to MULTIPLE variants (a broken-randomization
    conflict, not a harmless duplicate). The conflicting-assignment
    validity gate must still fire EXACTLY as before (never weakened),
    which means experiment_node — and therefore the stratification
    estimator — never runs. The report must nonetheless show:

      analysis_mode = Stratified Analysis (never Full Experiment Review)
      stratification_column = "landing_page" (reached the report)
      stratification.status = NOT_RUN
      stratification.not_run_reason mentions the conflicting-assignment count
      report.segmentation is untouched by any of this (never mislabeled
        as "Segment Analysis")
    """
    n_per_arm = 300
    n_conflicting = 1895
    rng = np.random.default_rng(9)

    control_conv = rng.binomial(1, 0.10, n_per_arm)
    treatment_conv = rng.binomial(1, 0.12, n_per_arm)
    rows = [
        {"user_id": i, "group": "control", "landing_page": "old_page", "converted": int(control_conv[i])}
        for i in range(n_per_arm)
    ] + [
        {
            "user_id": n_per_arm + i,
            "group": "treatment",
            "landing_page": "new_page",
            "converted": int(treatment_conv[i]),
        }
        for i in range(n_per_arm)
    ]
    # Users assigned to BOTH control and treatment — the exact
    # "conflicting variant assignment" broken-randomization case, never
    # to be confused with a harmless same-variant duplicate. Alternates
    # which arm appears FIRST per conflicting user so that, after
    # validation_node's own dedup-by-first-occurrence step, the overall
    # control/treatment counts stay balanced — this isolates the
    # conflicting-assignment gate specifically, rather than also
    # tripping SRM (which fires first in route_after_validation and
    # would otherwise mask which gate this regression test is for).
    conflicting_rows = []
    for i in range(n_conflicting):
        uid = f"conflict_{i}"
        first, second = ("control", "treatment") if i % 2 == 0 else ("treatment", "control")
        conflicting_rows.append({"user_id": uid, "group": first, "landing_page": "old_page" if first == "control" else "new_page", "converted": 0})
        conflicting_rows.append({"user_id": uid, "group": second, "landing_page": "old_page" if second == "control" else "new_page", "converted": 1})
    df = pd.DataFrame(rows + conflicting_rows)
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Stratified Analysis by landing_page",
            "settings": AnalysisSettings(),  # pure free text — no structured setting at all
        }
    )

    # 1) Free-text request produced both the intent and the column.
    assert final_state["plan"]["intent_label"] == "Stratified Analysis"
    assert final_state["plan"]["intent_label"] != "Full Experiment Review"
    assert final_state["settings"].analysis_mode == "stratified"
    assert final_state["settings"].stratification_column == "landing_page"

    report = final_state["report"]

    # Existing gate fired exactly as before — never weakened.
    assert report.experiment_validity.value == "INVALID"
    assert report.stats == []

    # 2) stratification_column reached the final report.
    assert report.stratification is not None
    assert report.stratification.stratification_column == "landing_page"

    # 3) experiment_node (and therefore the eligibility/estimator logic
    # inside it) never ran on this invalid dataset.
    assert report.stratification.status == StratificationStatus.NOT_RUN
    assert report.stratification.eligibility is None
    assert report.stratification.estimate is None

    # 4) The exact reason, including the real count.
    assert f"{n_conflicting:,}" in report.stratification.not_run_reason
    assert "conflicting variant assignments" in report.stratification.not_run_reason

    # Never mislabeled as Segment Analysis — segmentation is a
    # completely separate, untouched field.
    assert report.segmentation is None
