"""
Phase 2 — deterministic Hypothesis Evaluator tests.

Covers the full case list from the Phase 2 spec (§14), plus the
metric-matching and zero-relative-effect edge cases (§6, §11) and the
new StatResult.observed_relative_effect field this all reads from
(§12) — see tests/stats/test_hypothesis_tests.py for that field's own
dedicated tests at the compute_stat_result level.
"""

import pytest

from app.schemas.hypothesis import ExpectedDirection, Hypothesis
from app.schemas.hypothesis_evaluation import HypothesisVerdict
from app.schemas.statistics import HypothesisTestType, StatResult
from app.stats.hypothesis_evaluator import evaluate_hypothesis


def _stat_result(
    metric="Conversion Rate",
    observed_relative_effect=0.06,
    significant=True,
    is_omnibus=False,
):
    return StatResult(
        metric=metric,
        test_type=HypothesisTestType.CHI_SQUARE,
        test_name="Chi-square test",
        statistic=4.2,
        selection_reason="Binary metric — chi-square test.",
        control="10.00%",
        variant="10.60%",
        delta="+6.0% (rel)",
        observed_relative_effect=observed_relative_effect,
        p_value=0.01 if significant else 0.51,
        significant=significant,
        ci_lower="+0.10pp",
        ci_upper="+0.90pp",
        is_omnibus=is_omnibus,
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


# --- Case A: expected effect provided ---------------------------------


def test_significant_correct_direction_effect_achieved_is_supported():
    """Expected +5%, observed +6%, significant=True -> SUPPORTED."""
    h = _hypothesis(expected_effect_relative=0.05)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.06, significant=True)])
    assert ev.verdict == HypothesisVerdict.SUPPORTED
    assert ev.direction_supported is True
    assert ev.statistically_significant is True
    assert ev.effect_achievement_ratio == pytest.approx(1.2)


def test_significant_correct_direction_effect_below_target_is_partially_supported():
    """Expected +5%, observed +3.2%, significant=True -> PARTIALLY_SUPPORTED."""
    h = _hypothesis(expected_effect_relative=0.05)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.032, significant=True)])
    assert ev.verdict == HypothesisVerdict.PARTIALLY_SUPPORTED
    assert ev.direction_supported is True
    assert ev.statistically_significant is True
    assert ev.effect_achievement_ratio == pytest.approx(0.64)


def test_exact_expected_effect_is_supported():
    """Observed == expected exactly (ratio == 1.0) -> SUPPORTED, not PARTIALLY_SUPPORTED."""
    h = _hypothesis(expected_effect_relative=0.05)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.05, significant=True)])
    assert ev.verdict == HypothesisVerdict.SUPPORTED
    assert ev.effect_achievement_ratio == pytest.approx(1.0)


def test_slightly_below_expected_effect_is_partially_supported():
    h = _hypothesis(expected_effect_relative=0.05)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.0499, significant=True)])
    assert ev.verdict == HypothesisVerdict.PARTIALLY_SUPPORTED
    assert ev.effect_achievement_ratio < 1.0


def test_wrong_direction_is_not_supported_regardless_of_significance():
    """Expected +5%, observed -2%, significant=True -> NOT_SUPPORTED."""
    h = _hypothesis(expected_effect_relative=0.05)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=-0.02, significant=True)])
    assert ev.verdict == HypothesisVerdict.NOT_SUPPORTED
    assert ev.direction_supported is False
    assert ev.statistically_significant is True  # significance is reported honestly either way


def test_not_significant_with_expected_direction_is_not_supported():
    """
    Expected +5%, observed +2%, significant=False -> NOT_SUPPORTED. This is
    the specific case the spec calls out: moving in the expected direction
    is NOT enough for PARTIALLY_SUPPORTED without significance.
    """
    h = _hypothesis(expected_effect_relative=0.05)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.02, significant=False)])
    assert ev.verdict == HypothesisVerdict.NOT_SUPPORTED
    assert ev.direction_supported is True  # direction alone is still honestly reported...
    assert ev.statistically_significant is False  # ...but this is what sinks the verdict


# --- Case B: no expected effect provided -------------------------------


def test_no_expected_effect_significant_correct_direction_is_supported():
    h = _hypothesis(expected_effect_relative=None)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.02, significant=True)])
    assert ev.verdict == HypothesisVerdict.SUPPORTED
    assert ev.effect_achievement_ratio is None  # never calculated per spec §4


def test_no_expected_effect_non_significant_is_not_supported():
    h = _hypothesis(expected_effect_relative=None)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.02, significant=False)])
    assert ev.verdict == HypothesisVerdict.NOT_SUPPORTED
    assert ev.effect_achievement_ratio is None


def test_no_expected_effect_wrong_direction_is_not_supported():
    h = _hypothesis(expected_effect_relative=None)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=-0.02, significant=True)])
    assert ev.verdict == HypothesisVerdict.NOT_SUPPORTED
    assert ev.direction_supported is False


# --- direction_supported must be independent of significance -----------


def test_direction_supported_does_not_itself_require_significance():
    """direction_supported != statistically_significant — verified as two independently-set fields."""
    h = _hypothesis(expected_effect_relative=0.05)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.02, significant=False)])
    assert ev.direction_supported is True
    assert ev.statistically_significant is False
    # And the converse — wrong direction but significant:
    ev2 = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=-0.02, significant=True)])
    assert ev2.direction_supported is False
    assert ev2.statistically_significant is True


@pytest.mark.parametrize(
    "direction,observed,expected_supported",
    [
        (ExpectedDirection.INCREASE, 0.01, True),
        (ExpectedDirection.INCREASE, -0.01, False),
        (ExpectedDirection.INCREASE, 0.0, False),
        (ExpectedDirection.DECREASE, -0.01, True),
        (ExpectedDirection.DECREASE, 0.01, False),
        (ExpectedDirection.DECREASE, 0.0, False),
    ],
)
def test_direction_supported_sign_mapping(direction, observed, expected_supported):
    h = _hypothesis(expected_direction=direction, expected_effect_relative=None)
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=observed, significant=True)])
    assert ev.direction_supported is expected_supported


# --- no_change direction: explicitly out of scope this phase -------------


def test_no_change_direction_never_produces_a_fabricated_verdict():
    """
    Phase 2 does not implement 'no_change' evaluation — a non-significant
    result does not itself prove the absence of an effect, and that would
    require a dedicated equivalence-testing methodology this phase doesn't
    provide. Regardless of what the matched StatResult shows (significant
    or not, any observed effect), 'no_change' must yield an unavailable
    evaluation with no verdict — never a guessed/fabricated one.
    """
    h = _hypothesis(expected_direction=ExpectedDirection.NO_CHANGE, expected_effect_relative=None)

    ev_significant = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.08, significant=True)])
    assert ev_significant.verdict is None
    assert ev_significant.direction_supported is None
    assert ev_significant.statistically_significant is None
    assert ev_significant.effect_achievement_ratio is None
    assert ev_significant.evaluation_note is not None

    ev_not_significant = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.001, significant=False)])
    assert ev_not_significant.verdict is None
    assert ev_not_significant.direction_supported is None
    assert ev_not_significant.statistically_significant is None
    assert ev_not_significant.evaluation_note is not None


# --- no hypothesis -------------------------------------------------------


def test_no_hypothesis_yields_no_evaluation():
    """No hypothesis -> evaluate_hypothesis returns None, never a fabricated verdict."""
    assert evaluate_hypothesis(None, [_stat_result()]) is None


# --- zero expected effect -------------------------------------------------


def test_zero_expected_effect_relative_is_rejected_at_schema_level():
    """Phase 2 §6 — zero has no meaningful interpretation for increase/decrease; rejected deterministically."""
    with pytest.raises(Exception):  # ValidationError
        Hypothesis(
            statement="x",
            primary_metric="Conversion Rate",
            expected_direction=ExpectedDirection.INCREASE,
            expected_effect_relative=0.0,
        )


def test_effect_achievement_ratio_defensive_guard_never_divides_by_zero():
    """
    Defense in depth (Phase 2 §6): even if a Hypothesis with
    expected_effect_relative=0 existed (bypassing normal validation,
    e.g. via model_construct), the evaluator itself must never divide
    by zero — ratio must be None, not inf/NaN, and no exception raised.
    """
    h = Hypothesis.model_construct(
        statement="x",
        primary_metric="Conversion Rate",
        expected_direction=ExpectedDirection.INCREASE,
        expected_effect_relative=0.0,
        rationale=None,
    )
    ev = evaluate_hypothesis(h, [_stat_result(observed_relative_effect=0.02, significant=True)])
    assert ev.effect_achievement_ratio is None
    assert ev.verdict is not None  # still resolves a verdict (Case B path), doesn't crash


# --- primary metric matching ----------------------------------------------


def test_primary_metric_matching_uses_exact_metric_string():
    h = _hypothesis(primary_metric="Conversion Rate")
    ev = evaluate_hypothesis(
        h,
        [
            _stat_result(metric="Revenue", observed_relative_effect=0.5, significant=True),
            _stat_result(metric="Conversion Rate", observed_relative_effect=0.06, significant=True),
        ],
    )
    assert ev.metric_matched is True
    assert ev.observed_effect_relative == pytest.approx(0.06)  # NOT the Revenue row


def test_primary_metric_not_found_yields_no_fabricated_verdict():
    """Spec §11 — metric not found must be an explicit unavailable state, never a silent fallback to another metric."""
    h = _hypothesis(primary_metric="Conversion Rate")
    ev = evaluate_hypothesis(h, [_stat_result(metric="Revenue", observed_relative_effect=0.5, significant=True)])
    assert ev.metric_matched is False
    assert ev.verdict is None
    assert ev.direction_supported is None
    assert ev.statistically_significant is None
    assert ev.effect_achievement_ratio is None
    assert ev.evaluation_note is not None
    assert "Conversion Rate" in ev.evaluation_note


def test_omnibus_row_is_never_matched_even_with_same_metric_name():
    """A multi-arm omnibus row must never be picked over a real pairwise result with the same metric label."""
    h = _hypothesis(primary_metric="Conversion Rate")
    ev = evaluate_hypothesis(
        h,
        [
            _stat_result(metric="Conversion Rate", observed_relative_effect=None, significant=True, is_omnibus=True),
            _stat_result(metric="Conversion Rate", observed_relative_effect=0.06, significant=True, is_omnibus=False),
        ],
    )
    assert ev.metric_matched is True
    assert ev.observed_effect_relative == pytest.approx(0.06)


def test_undefined_observed_relative_effect_yields_no_fabricated_verdict():
    """Matched metric found, but its relative effect is undefined (e.g. zero control baseline) -> unavailable, not fabricated."""
    h = _hypothesis(primary_metric="Conversion Rate")
    ev = evaluate_hypothesis(h, [_stat_result(metric="Conversion Rate", observed_relative_effect=None, significant=True)])
    assert ev.metric_matched is True
    assert ev.verdict is None
    assert ev.evaluation_note is not None
