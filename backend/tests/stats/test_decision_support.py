"""
Phase 3 — deterministic Decision Support tests.

Covers the 14-case list from the Phase 3 spec (§11). Every test
asserts the result is derived purely from the inputs already computed
elsewhere (StatResult / HypothesisEvaluation / DatasetInfo) — no test
here exercises any new statistical computation, since none exists.
"""

import pytest

from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.hypothesis import ExpectedDirection, Hypothesis
from app.schemas.hypothesis_evaluation import HypothesisEvaluation, HypothesisVerdict
from app.schemas.statistics import HypothesisTestType, StatResult
from app.stats.decision_support import build_decision_support
from app.stats.hypothesis_evaluator import evaluate_hypothesis


def _binary_stat_result(
    metric="Conversion Rate",
    control="10.00%",
    variant="10.80%",
    delta="+8.0% (rel)",
    observed_relative_effect=0.08,
    significant=True,
    is_omnibus=False,
    test_type=HypothesisTestType.CHI_SQUARE,
):
    return StatResult(
        metric=metric,
        test_type=test_type,
        test_name="Chi-square test",
        statistic=4.2,
        selection_reason="Binary metric — chi-square test.",
        control=control,
        variant=variant,
        delta=delta,
        observed_relative_effect=observed_relative_effect,
        p_value=0.01 if significant else 0.51,
        significant=significant,
        ci_lower="+0.10pp",
        ci_upper="+0.90pp",
        is_omnibus=is_omnibus,
    )


def _continuous_stat_result(
    metric="AOV",
    control="$50.00",
    variant="$48.00",
    delta="-4.0% (rel)",
    observed_relative_effect=-0.04,
    significant=True,
):
    return StatResult(
        metric=metric,
        test_type=HypothesisTestType.WELCH_T_TEST,
        test_name="Welch's t-test",
        statistic=-2.1,
        selection_reason="Continuous metric, large sample.",
        control=control,
        variant=variant,
        delta=delta,
        observed_relative_effect=observed_relative_effect,
        p_value=0.01 if significant else 0.51,
        significant=significant,
        ci_lower="-2.00",
        ci_upper="-0.50",
    )


def _hypothesis(**overrides):
    kwargs = dict(
        statement="Increasing the checkout CTA visibility will increase checkout conversion.",
        primary_metric="Conversion Rate",
        expected_direction=ExpectedDirection.INCREASE,
        expected_effect_relative=0.05,
    )
    kwargs.update(overrides)
    return Hypothesis(**kwargs)


def _dataset(users=100_000, dtype=DatasetType.AGGREGATED_AB_TEST):
    return DatasetInfo(
        type=dtype,
        variants=2,
        users=users,
        metric_label="Conversion Rate",
        available_metrics=["Conversion Rate", "AOV"],
        metric_selection_reason="Binary metric detected as primary.",
    )


def _build(hypothesis, stat_results, guardrail_results=None, dataset=None):
    dataset = dataset or _dataset()
    hyp_eval = evaluate_hypothesis(hypothesis, stat_results)
    return build_decision_support(
        hypothesis=hypothesis,
        hypothesis_evaluation=hyp_eval,
        stat_results=stat_results,
        guardrail_results=guardrail_results or [],
        dataset=dataset,
    ), hyp_eval


# 1. expected value — increase ------------------------------------------


def test_expected_value_increase():
    h = _hypothesis(expected_direction=ExpectedDirection.INCREASE, expected_effect_relative=0.05)
    ds, _ = _build(h, [_binary_stat_result(control="10.00%", variant="10.80%")])
    assert ds.baseline_value == pytest.approx(10.0)
    assert ds.expected_value == pytest.approx(10.5)


# 2. expected value — decrease -------------------------------------------


def test_expected_value_decrease():
    h = _hypothesis(expected_direction=ExpectedDirection.DECREASE, expected_effect_relative=0.05)
    ds, _ = _build(h, [_binary_stat_result(control="10.00%", variant="9.40%", delta="-6.0% (rel)", observed_relative_effect=-0.06)])
    assert ds.baseline_value == pytest.approx(10.0)
    assert ds.expected_value == pytest.approx(9.5)


# 3. expected vs observed comparison -------------------------------------


def test_expected_vs_observed_comparison_values_present():
    h = _hypothesis(expected_effect_relative=0.05)
    ds, _ = _build(h, [_binary_stat_result(control="10.00%", variant="10.80%", observed_relative_effect=0.08)])
    assert ds.baseline_value == pytest.approx(10.0)
    assert ds.observed_value == pytest.approx(10.8)
    assert ds.observed_effect_relative == pytest.approx(0.08)
    assert ds.expected_effect_relative == pytest.approx(0.05)


# 4. achievement ratio ----------------------------------------------------


def test_achievement_ratio_matches_hypothesis_evaluation():
    h = _hypothesis(expected_effect_relative=0.05)
    ds, hyp_eval = _build(h, [_binary_stat_result(observed_relative_effect=0.08)])
    assert ds.effect_achievement_ratio == pytest.approx(hyp_eval.effect_achievement_ratio)
    assert ds.effect_achievement_ratio == pytest.approx(1.6)


# 5. no expected effect -----------------------------------------------------


def test_no_expected_effect_leaves_expected_value_null():
    h = _hypothesis(expected_direction=ExpectedDirection.NO_CHANGE, expected_effect_relative=None)
    ds, _ = _build(h, [_binary_stat_result()])
    assert ds.expected_value is None
    assert ds.expected_effect_relative is None


# 6. no hypothesis ------------------------------------------------------------


def test_no_hypothesis_returns_none():
    ds = build_decision_support(
        hypothesis=None,
        hypothesis_evaluation=None,
        stat_results=[_binary_stat_result()],
        guardrail_results=[],
        dataset=_dataset(),
    )
    assert ds is None


# 7. missing baseline -----------------------------------------------------


def test_missing_baseline_marks_unavailable_with_warning():
    h = _hypothesis(expected_effect_relative=0.05)
    stat = _binary_stat_result(control="", variant="10.80%")
    ds, _ = _build(h, [stat])
    assert ds.baseline_value is None
    assert ds.expected_value is None
    assert ds.impact_calculation_method == "unavailable"
    assert any("baseline" in w.lower() for w in ds.warnings)


# 8. missing population/exposure -------------------------------------------


def test_missing_population_marks_impact_unavailable():
    h = _hypothesis(expected_effect_relative=0.05)
    ds, _ = _build(h, [_binary_stat_result()], dataset=_dataset(users=0))
    assert ds.impact_calculation_method == "unavailable"
    assert ds.incremental_count is None
    assert any("population" in w.lower() for w in ds.warnings)


# 9. primary metric improvement (absolute impact computed) ------------------


def test_primary_metric_improvement_computes_absolute_impact():
    h = _hypothesis(expected_effect_relative=0.05)
    ds, _ = _build(h, [_binary_stat_result(control="10.00%", variant="11.00%")], dataset=_dataset(users=100_000))
    assert ds.impact_calculation_method == "population_scaled"
    assert ds.baseline_expected_count == pytest.approx(10_000)
    assert ds.observed_count == pytest.approx(11_000)
    assert ds.incremental_count == pytest.approx(1_000)


# 10. additional metric deterioration ----------------------------------------


def test_additional_metric_deterioration_is_flagged():
    h = _hypothesis(primary_metric="Conversion Rate", expected_effect_relative=0.05)
    primary = _binary_stat_result(control="10.00%", variant="10.80%", observed_relative_effect=0.08)
    aov = _continuous_stat_result(metric="AOV", control="$50.00", variant="$48.00", observed_relative_effect=-0.04, significant=True)
    ds, _ = _build(h, [primary, aov])
    assert len(ds.additional_metrics) == 1
    assert ds.additional_metrics[0].metric == "AOV"
    assert ds.additional_metrics[0].direction == "decrease"
    assert ds.additional_metrics[0].statistically_significant is True
    assert ds.business_interpretation is not None
    assert "AOV" in ds.business_interpretation


# 11. guardrail deterioration -----------------------------------------------


def test_guardrail_deterioration_flagged_and_not_hidden_by_positive_primary():
    h = _hypothesis(primary_metric="Conversion Rate", expected_effect_relative=0.05)
    primary = _binary_stat_result(control="10.00%", variant="10.80%", observed_relative_effect=0.08)
    guardrail = _continuous_stat_result(
        metric="Revenue", control="$100,000", variant="$93,000", delta="-7.0% (rel)", observed_relative_effect=-0.07, significant=True
    )
    ds, _ = _build(h, [primary], guardrail_results=[guardrail])
    assert ds.guardrail_violated is True
    assert ds.guardrail_findings[0].violated is True
    assert "Revenue" in ds.business_interpretation


# 12. no significant additional metric -----------------------------------------


def test_no_significant_additional_metric_not_flagged_as_tradeoff():
    h = _hypothesis(primary_metric="Conversion Rate", expected_effect_relative=0.05)
    primary = _binary_stat_result(control="10.00%", variant="10.80%")
    aov = _continuous_stat_result(metric="AOV", observed_relative_effect=-0.01, significant=False)
    ds, _ = _build(h, [primary, aov])
    assert ds.additional_metrics[0].statistically_significant is False
    assert ds.business_interpretation is None or "AOV" not in ds.business_interpretation


# 13. significant wrong-direction additional metric (increase, not flagged as trade-off) --


def test_significant_additional_metric_moving_favorably_not_flagged_as_tradeoff():
    h = _hypothesis(primary_metric="Conversion Rate", expected_effect_relative=0.05)
    primary = _binary_stat_result(control="10.00%", variant="10.80%")
    aov = _continuous_stat_result(metric="AOV", control="$50.00", variant="$52.00", delta="+4.0% (rel)", observed_relative_effect=0.04, significant=True)
    ds, _ = _build(h, [primary, aov])
    assert ds.additional_metrics[0].direction == "increase"
    assert ds.business_interpretation is None or "AOV decreased" not in (ds.business_interpretation or "")


# 14. prevention of fabricated business impact (continuous primary metric) ------


def test_continuous_primary_metric_never_gets_population_scaled_impact():
    h = _hypothesis(primary_metric="AOV", expected_direction=ExpectedDirection.INCREASE, expected_effect_relative=0.05)
    aov = _continuous_stat_result(metric="AOV", control="$50.00", variant="$52.00", delta="+4.0% (rel)", observed_relative_effect=0.04, significant=True)
    ds, _ = _build(h, [aov], dataset=_dataset(users=100_000))
    assert ds.impact_calculation_method == "unavailable"
    assert ds.baseline_expected_count is None
    assert ds.incremental_count is None
    assert any("binary" in w.lower() or "continuous" in w.lower() for w in ds.warnings)


def test_unknown_dataset_type_prevents_population_scaled_impact():
    h = _hypothesis(expected_effect_relative=0.05)
    ds, _ = _build(h, [_binary_stat_result()], dataset=_dataset(dtype=DatasetType.UNKNOWN))
    assert ds.impact_calculation_method == "unavailable"
    assert any("user-level" in w.lower() for w in ds.warnings)


def test_no_hypothesis_field_shapes_stay_null_when_metric_unmatched():
    h = _hypothesis(primary_metric="Nonexistent Metric", expected_effect_relative=0.05)
    ds, _ = _build(h, [_binary_stat_result(metric="Conversion Rate")])
    assert ds.available is False
    assert ds.hypothesis_verdict is None
    assert ds.expected_value is None
    assert any("no statistical result" in w.lower() for w in ds.warnings)
