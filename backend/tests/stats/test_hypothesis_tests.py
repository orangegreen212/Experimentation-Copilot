import numpy as np
import pandas as pd
import pytest

from app.schemas.statistics import HypothesisTestType, MetricType
from app.stats.hypothesis_tests import compute_stat_result, format_p_value, select_test


# --- format_p_value: never display "p=0" -----------------------------------

def test_format_p_value_underflow_never_displays_as_zero():
    """
    Regression test: at very large sample sizes (e.g. the 294,478-user
    chi-square scenario), scipy can return a p-value that underflows to
    exactly 0.0 in floating point. Displaying that raw would print the
    false, meaningless "p = 0" — it must read "< 0.001" instead.
    """
    assert format_p_value(0.0) == "< 0.001"


def test_format_p_value_extremely_small_but_nonzero_floors_at_threshold():
    assert format_p_value(1e-300) == "< 0.001"


def test_format_p_value_just_below_threshold_floors():
    assert format_p_value(0.0009) == "< 0.001"


def test_format_p_value_normal_value_shows_actual_number():
    assert format_p_value(0.0421) == "= 0.0421"


def test_format_p_value_never_mutates_significance_math():
    """Presentation-only: the underlying float used for alpha comparisons is untouched."""
    p = 0.0
    displayed = format_p_value(p)
    assert displayed == "< 0.001"
    assert p == 0.0  # the real float is never rewritten to some floored value
    assert (p < 0.05) is True  # significance comparisons still use the real float


# --- select_test: binary ---------------------------------------------------

def test_binary_large_counts_selects_chi_square():
    control = pd.Series([1] * 500 + [0] * 11400)
    variant = pd.Series([1] * 560 + [0] * 11340)
    result = select_test(control, variant, MetricType.BINARY)
    assert result.test_type == HypothesisTestType.CHI_SQUARE
    assert "chi-square" in result.reason.lower()


def test_binary_small_expected_counts_selects_fishers_exact():
    control = pd.Series([1] * 2 + [0] * 20)
    variant = pd.Series([1] * 1 + [0] * 20)
    result = select_test(control, variant, MetricType.BINARY)
    assert result.test_type == HypothesisTestType.FISHERS_EXACT
    assert "fisher" in result.reason.lower()


# --- select_test: continuous -------------------------------------------------

def test_continuous_large_sample_selects_welch_without_normality_check():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.exponential(scale=10, size=50))  # skewed, but n>=30
    variant = pd.Series(rng.exponential(scale=11, size=50))
    result = select_test(control, variant, MetricType.CONTINUOUS_GENERAL)
    assert result.test_type == HypothesisTestType.WELCH_T_TEST
    assert result.large_sample_rule_applied is True
    assert result.normality is None


def test_continuous_small_normal_sample_selects_welch():
    rng = np.random.default_rng(7)
    control = pd.Series(rng.normal(50, 5, size=20))
    variant = pd.Series(rng.normal(52, 5, size=20))
    result = select_test(control, variant, MetricType.CONTINUOUS_GENERAL)
    assert result.test_type == HypothesisTestType.WELCH_T_TEST
    assert result.large_sample_rule_applied is False
    assert result.normality is not None


def test_continuous_small_nonnormal_sample_selects_mann_whitney():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.exponential(scale=5, size=20))
    variant = pd.Series(rng.exponential(scale=6, size=20))
    result = select_test(control, variant, MetricType.CONTINUOUS_GENERAL)
    assert result.test_type == HypothesisTestType.MANN_WHITNEY_U
    assert result.normality is not None


# --- compute_stat_result: binary --------------------------------------------

def test_compute_binary_significant_uplift():
    # matches the shape of mock-data.ts HIGH_CONFIDENCE_REPORT
    rng = np.random.default_rng(1)
    control = pd.Series(rng.binomial(1, 0.0421, size=6200))
    variant = pd.Series(rng.binomial(1, 0.0456, size=6200))
    result = compute_stat_result(control, variant, MetricType.BINARY, "Conversion Rate")
    assert result.metric == "Conversion Rate"
    assert "%" in result.control
    assert "pp" in result.ci_lower
    # Trust-bug regression guard: the test name shown to the frontend
    # must match the test actually run, never a hardcoded guess.
    assert result.test_type == HypothesisTestType.CHI_SQUARE
    assert result.test_name == "Chi-square test"
    assert result.selection_reason  # non-empty, human-readable
    assert result.statistic >= 0  # chi2 statistic is non-negative


def test_compute_binary_identical_groups_not_significant():
    control = pd.Series([1] * 50 + [0] * 950)
    variant = pd.Series([1] * 50 + [0] * 950)
    result = compute_stat_result(control, variant, MetricType.BINARY, "Conversion Rate")
    assert result.significant is False
    assert result.p_value > 0.05


# --- compute_stat_result: continuous (Welch) --------------------------------

def test_compute_welch_significant_difference():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(48.20, 15, size=200))
    variant = pd.Series(rng.normal(60.0, 15, size=200))  # large, clear shift
    result = compute_stat_result(control, variant, MetricType.CONTINUOUS_MONETARY, "Average Order Value")
    assert result.significant is True
    assert "$" in result.control
    assert result.test_type == HypothesisTestType.WELCH_T_TEST
    assert result.test_name == "Welch's t-test"


def test_compute_welch_no_significant_difference():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(48.20, 15, size=200))
    variant = pd.Series(rng.normal(47.95, 15, size=200))
    result = compute_stat_result(control, variant, MetricType.CONTINUOUS_MONETARY, "Average Order Value")
    assert result.significant is False


# --- compute_stat_result: Mann-Whitney --------------------------------------

def test_compute_mann_whitney_result_shape():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.exponential(scale=5, size=20))
    variant = pd.Series(rng.exponential(scale=15, size=20))  # clear shift, skewed
    selection = select_test(control, variant, MetricType.CONTINUOUS_GENERAL)
    assert selection.test_type == HypothesisTestType.MANN_WHITNEY_U
    result = compute_stat_result(control, variant, MetricType.CONTINUOUS_GENERAL, "Session Duration", selection)
    assert "median" in result.delta
    assert 0.0 <= result.p_value <= 1.0
    assert result.test_type == HypothesisTestType.MANN_WHITNEY_U
    assert result.test_name == "Mann-Whitney U test"


def test_llm_is_never_involved_module_has_no_llm_imports():
    """Structural guardrail: this module must never import anything LLM/LangChain-related."""
    import ast

    import app.stats.hypothesis_tests as module

    tree = ast.parse(open(module.__file__).read())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)

    for name in imported_names:
        assert "langchain" not in name.lower()
        assert "openai" not in name.lower()
        assert not name.lower().startswith("app.llm")

# --- multi-arm analysis ------------------------------------------------------

def test_multi_arm_binary_runs_omnibus_then_holm_corrected_pairwise():
    control = pd.Series([1] * 100 + [0] * 900)
    variant_a = pd.Series([1] * 120 + [0] * 880)
    variant_b = pd.Series([1] * 300 + [0] * 700)

    results = __import__("app.stats.hypothesis_tests", fromlist=["compute_multi_arm_stat_results"]).compute_multi_arm_stat_results(
        {"control": control, "A": variant_a, "B": variant_b},
        MetricType.BINARY,
        "Click Rate",
    )

    assert results[0].is_omnibus is True
    assert results[0].test_name == "Chi-square omnibus test"
    assert results[0].significant is True
    pairwise = results[1:]
    assert len(pairwise) == 2
    assert all(r.multiple_testing_method == "Holm-Bonferroni" for r in pairwise)
    assert all(r.adjusted_p_value is not None for r in pairwise)
    assert min(pairwise, key=lambda r: r.adjusted_p_value).arm == "B"


def test_multi_arm_binary_non_significant_omnibus_stops_pairwise_fishing():
    control = pd.Series([1] * 100 + [0] * 900)
    variant_a = pd.Series([1] * 101 + [0] * 899)
    variant_b = pd.Series([1] * 99 + [0] * 901)

    results = __import__("app.stats.hypothesis_tests", fromlist=["compute_multi_arm_stat_results"]).compute_multi_arm_stat_results(
        {"control": control, "A": variant_a, "B": variant_b},
        MetricType.BINARY,
        "Click Rate",
    )

    assert len(results) == 1
    assert results[0].is_omnibus is True
    assert results[0].significant is False


def test_multi_arm_continuous_uses_anova_for_large_samples():
    rng = np.random.default_rng(42)
    results = __import__("app.stats.hypothesis_tests", fromlist=["compute_multi_arm_stat_results"]).compute_multi_arm_stat_results(
        {
            "control": pd.Series(rng.normal(10, 1, 50)),
            "A": pd.Series(rng.normal(10.1, 1, 50)),
            "B": pd.Series(rng.normal(12, 1, 50)),
        },
        MetricType.CONTINUOUS_GENERAL,
        "Revenue",
    )
    assert results[0].test_type == HypothesisTestType.ONE_WAY_ANOVA
    assert results[0].is_omnibus is True


# --- Phase 2: StatResult.observed_relative_effect ---------------------------
#
# The SAME already-computed relative effect underlying the formatted `delta`
# string, exposed as a raw fraction for deterministic downstream consumers
# (app/stats/hypothesis_evaluator.py). These tests confirm it's derived from
# the identical calculation as `delta` (never parsed from it, never a
# different formula) and can never silently diverge.


def test_observed_relative_effect_matches_delta_for_binary():
    rng = np.random.default_rng(1)
    control = pd.Series(rng.binomial(1, 0.10, size=6000))
    variant = pd.Series(rng.binomial(1, 0.106, size=6000))
    result = compute_stat_result(control, variant, MetricType.BINARY, "Conversion Rate")
    assert result.observed_relative_effect is not None
    # delta is "{relative_delta:+.1f}% (rel)" where relative_delta == observed_relative_effect * 100
    expected_delta_pct = round(result.observed_relative_effect * 100, 1)
    assert result.delta == f"{expected_delta_pct:+.1f}% (rel)"


def test_observed_relative_effect_matches_delta_for_continuous_welch():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(48.20, 15, size=200))
    variant = pd.Series(rng.normal(60.0, 15, size=200))
    result = compute_stat_result(control, variant, MetricType.CONTINUOUS_MONETARY, "Average Order Value")
    assert result.observed_relative_effect is not None
    expected_delta_pct = round(result.observed_relative_effect * 100, 1)
    assert result.delta == f"{expected_delta_pct:+.1f}% (rel)"


def test_observed_relative_effect_matches_delta_for_mann_whitney():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.exponential(scale=5, size=20))
    variant = pd.Series(rng.exponential(scale=15, size=20))
    selection = select_test(control, variant, MetricType.CONTINUOUS_GENERAL)
    result = compute_stat_result(control, variant, MetricType.CONTINUOUS_GENERAL, "Session Duration", selection)
    assert result.observed_relative_effect is not None
    expected_delta_pct = round(result.observed_relative_effect * 100, 1)
    assert result.delta == f"{expected_delta_pct:+.1f}% (rel, median)"


def test_observed_relative_effect_is_none_when_control_baseline_is_zero():
    """Zero control-arm baseline -> relative effect undefined -> None, never inf/NaN serialized onto the schema."""
    control = pd.Series([0] * 1000)
    variant = pd.Series([1] * 20 + [0] * 980)
    result = compute_stat_result(control, variant, MetricType.BINARY, "Conversion Rate")
    assert result.observed_relative_effect is None
    assert "inf" in result.delta.lower()  # the legacy display string keeps its existing (unmodified) behavior


def test_observed_relative_effect_sign_matches_direction_of_change():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(60.0, 5, size=200))
    variant = pd.Series(rng.normal(48.0, 5, size=200))  # variant LOWER than control
    result = compute_stat_result(control, variant, MetricType.CONTINUOUS_MONETARY, "Average Order Value")
    assert result.observed_relative_effect < 0
    assert result.delta.startswith("-")


def test_omnibus_result_has_no_observed_relative_effect():
    """Omnibus (multi-arm, 'does any arm differ') rows have no single well-defined relative effect."""
    from app.stats.hypothesis_tests import compute_multi_arm_stat_results

    rng = np.random.default_rng(42)
    results = compute_multi_arm_stat_results(
        {
            "control": pd.Series(rng.normal(10, 1, 50)),
            "A": pd.Series(rng.normal(10.1, 1, 50)),
            "B": pd.Series(rng.normal(12, 1, 50)),
        },
        MetricType.CONTINUOUS_GENERAL,
        "Revenue",
    )
    assert results[0].is_omnibus is True
    assert results[0].observed_relative_effect is None
    # But the pairwise comparisons that follow DO have it:
    pairwise = [r for r in results if not r.is_omnibus]
    assert len(pairwise) > 0
    assert all(r.observed_relative_effect is not None for r in pairwise)


def test_multi_arm_reference_arm_is_positional_not_semantic_control():
    """
    CHARACTERIZATION TEST — documents a real, currently-untested gap,
    it does not assert this is correct behavior.

    `compute_multi_arm_stat_results` has no `control_label`/
    `reference_label` parameter: `reference = labels[0]` is whichever
    key happens to be FIRST in the `arms` dict. Every other test in
    this file happens to insert "control" first, which is exactly why
    this was never caught. `experiment_node.py` builds `arms` straight
    from `df[variant_col].dropna().unique()` (first-appearance order
    in the raw file) — never from `resolve_control_label` — so on a
    real dataset where the actual control/holdout arm is NOT the
    first-appearing row (e.g. Hillstrom: 'Womens E-Mail' appears
    before 'No E-Mail' in the raw CSV), the pairwise comparisons use
    the wrong reference arm, and only 2 of the 3 possible pairs among
    3 arms are ever computed (no direct treatment-vs-treatment
    comparison when the omitted pair doesn't include the reference).

    This test pins today's actual behavior so a future engine change
    is a deliberate, visible decision instead of a silent one.
    """
    from app.stats.hypothesis_tests import compute_multi_arm_stat_results

    # "No E-Mail" is the real control/holdout, but it is NOT first in
    # insertion order — mirrors real Hillstrom column order exactly.
    arms = {
        "Womens E-Mail": pd.Series([1] * 200 + [0] * 800),
        "No E-Mail": pd.Series([1] * 100 + [0] * 900),
        "Mens E-Mail": pd.Series([1] * 150 + [0] * 850),
    }
    results = compute_multi_arm_stat_results(arms, MetricType.BINARY, "Conversion Rate")

    pairwise = [r for r in results if not r.is_omnibus]
    comparisons = {r.comparison for r in pairwise}

    # Documents CURRENT (order-dependent) behavior: reference is the
    # first dict key ("Womens E-Mail"), not the actual control
    # ("No E-Mail"). If this assertion ever fails, it means the
    # reference-arm selection became semantic (e.g. control-aware) —
    # a deliberate engine change, not a regression, and this test
    # should be updated to match.
    assert comparisons == {"Womens E-Mail vs No E-Mail", "Womens E-Mail vs Mens E-Mail"}
    assert "Mens E-Mail vs No E-Mail" not in comparisons
    assert all(r.reference_arm == "Womens E-Mail" for r in pairwise)


def test_multi_arm_control_label_overrides_positional_reference():
    """
    Regression test for the fix: passing `control_label` makes the
    ACTUAL control arm the reference, regardless of its position in
    the `arms` dict — closing the gap documented by
    test_multi_arm_reference_arm_is_positional_not_semantic_control
    above. Uses the same Hillstrom-shaped ordering (control arm is
    NOT first) to prove this isn't order-dependent anymore.
    """
    from app.stats.hypothesis_tests import compute_multi_arm_stat_results

    arms = {
        "Womens E-Mail": pd.Series([1] * 200 + [0] * 800),
        "No E-Mail": pd.Series([1] * 100 + [0] * 900),
        "Mens E-Mail": pd.Series([1] * 150 + [0] * 850),
    }
    results = compute_multi_arm_stat_results(
        arms, MetricType.BINARY, "Conversion Rate", control_label="No E-Mail"
    )

    pairwise = [r for r in results if not r.is_omnibus]
    comparisons = {r.comparison for r in pairwise}
    assert comparisons == {"No E-Mail vs Womens E-Mail", "No E-Mail vs Mens E-Mail"}
    assert all(r.reference_arm == "No E-Mail" for r in pairwise)


def test_multi_arm_control_label_not_in_arms_falls_back_to_positional():
    """An invalid/absent control_label must not crash — falls back to the original labels[0] behavior."""
    from app.stats.hypothesis_tests import compute_multi_arm_stat_results

    arms = {
        "control": pd.Series([1] * 100 + [0] * 900),
        "A": pd.Series([1] * 120 + [0] * 880),
        "B": pd.Series([1] * 300 + [0] * 700),
    }
    results = compute_multi_arm_stat_results(
        arms, MetricType.BINARY, "Conversion Rate", control_label="does_not_exist"
    )
    pairwise = [r for r in results if not r.is_omnibus]
    assert all(r.reference_arm == "control" for r in pairwise)
