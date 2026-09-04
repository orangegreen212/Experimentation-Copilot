"""
Phase 3 integration tests — DecisionSupport reaching ExperimentReport
through decision_node, numbers preserved end to end, and backward
compatibility for hypothesis-less reports (Phase 3 spec §11).
"""

import pytest

from app.graph.nodes.decision_node import decision_node
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.hypothesis import ExpectedDirection, Hypothesis
from app.schemas.quality import QualityCheck
from app.schemas.statistics import HypothesisTestType, StatResult


def _stat(control="10.00%", variant="10.80%", delta="+8.0% (rel)", observed_relative_effect=0.08, significant=True):
    return StatResult(
        metric="Conversion Rate",
        test_type=HypothesisTestType.CHI_SQUARE,
        test_name="Chi-square test",
        statistic=12.3,
        selection_reason="Binary metric with sufficient expected cell counts.",
        control=control,
        variant=variant,
        delta=delta,
        observed_relative_effect=observed_relative_effect,
        p_value=0.01 if significant else 0.6,
        significant=significant,
        ci_lower="+0.10pp",
        ci_upper="+0.90pp",
    )


def _dataset(users=100_000):
    return DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=users,
        metric_label="Conversion Rate",
        metric_selection_reason="Selected by the deterministic outcome-column priority.",
    )


def _base_state(**overrides):
    state = {
        "dataset_id": "does-not-matter",
        "user_prompt": "Analyze this experiment",
        "dataset": _dataset(),
        "srm_result": None,
        "quality_checks": [QualityCheck(label="Missing Values", passed=True, detail="0% missing")],
        "stat_results": [_stat()],
        "test_selection": None,
        "power_analysis": None,
        "kb_results": [],
        "funnel_result": None,
        "hypothesis": None,
    }
    state.update(overrides)
    return state


def test_decision_support_reaches_experiment_report_with_hypothesis():
    hypothesis = Hypothesis(
        statement="Increasing CTA visibility will increase conversion.",
        primary_metric="Conversion Rate",
        expected_direction=ExpectedDirection.INCREASE,
        expected_effect_relative=0.05,
    )
    state = decision_node(_base_state(hypothesis=hypothesis))
    report = state["report"]

    assert report.decision_support is not None
    assert report.decision_support.available is True
    # Numbers preserved exactly from the underlying StatResult / evaluation —
    # not re-derived by report generation or the LLM.
    assert report.decision_support.baseline_value == 10.0
    assert report.decision_support.observed_value == 10.8
    assert report.decision_support.observed_effect_relative == 0.08
    assert report.decision_support.expected_value == 10.5
    assert report.decision_support.impact_calculation_method == "population_scaled"
    assert report.decision_support.incremental_count == pytest.approx(800.0)  # 10.8%*100k - 10%*100k


def test_reports_without_hypothesis_remain_backward_compatible():
    """No hypothesis -> decision_support is None; rest of report unaffected."""
    state = decision_node(_base_state(hypothesis=None))
    report = state["report"]
    assert report.hypothesis is None
    assert report.hypothesis_evaluation is None
    assert report.decision_support is None
    # Phase 1/2 fields still populate normally.
    assert report.stats[0].control == "10.00%"


def test_hypothesis_evaluation_unchanged_by_decision_support_addition():
    hypothesis = Hypothesis(
        statement="Increasing CTA visibility will increase conversion.",
        primary_metric="Conversion Rate",
        expected_direction=ExpectedDirection.INCREASE,
        expected_effect_relative=0.05,
    )
    state = decision_node(_base_state(hypothesis=hypothesis))
    report = state["report"]
    # hypothesis_evaluation is the same object computed exactly as in Phase 2 —
    # decision_support consumes it but never mutates or replaces it.
    assert report.hypothesis_evaluation is not None
    assert report.hypothesis_evaluation.verdict == report.decision_support.hypothesis_verdict
    assert report.hypothesis_evaluation.effect_achievement_ratio == report.decision_support.effect_achievement_ratio


def test_decision_support_does_not_alter_existing_decision_field():
    """Decision Support is annotative only — the GO/NO-GO decision is untouched."""
    hypothesis = Hypothesis(
        statement="Increasing CTA visibility will increase conversion.",
        primary_metric="Conversion Rate",
        expected_direction=ExpectedDirection.INCREASE,
        expected_effect_relative=0.05,
    )
    state_with = decision_node(_base_state(hypothesis=hypothesis))
    state_without = decision_node(_base_state(hypothesis=None))
    assert state_with["report"].decision == state_without["report"].decision
    assert state_with["report"].guardrail_status == state_without["report"].guardrail_status
