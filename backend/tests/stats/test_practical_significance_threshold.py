"""
Bug 6 regression tests — practical-significance threshold precedence.

Root cause: `_practical_significance()` in `app/graph/report_generator.py`
always compared the observed effect against the post-hoc relative MDE
(`PowerAnalysisResult.minimum_detectable_effect_relative`), even when the
run's `Hypothesis` carried a user-specified `expected_effect_relative` for
the same primary metric. That silently discarded a pre-registered business
threshold in favor of a post-hoc statistical artifact, and mislabeled the
comparison as "MDE" in report text regardless of which value was actually
used.

These tests cover the four behaviors required by the fix:
  1. an explicitly supplied expected effect is propagated and used as the
     threshold (not the MDE);
  2. a missing expected effect (or a hypothesis for a different metric)
     still falls back to the existing MDE-based behavior, unchanged;
  3. the post-hoc MDE remains separately available/displayed even when the
     user-specified expected effect is what's actually used for the
     comparison;
  4. a relative expected effect (a fraction, e.g. 0.05 == "+5% relative")
     is never confused with a percentage-point effect — the threshold used
     in the percent-based comparison is `expected_effect_relative * 100`,
     not the raw fraction and not a `pp` value.
"""

from app.graph.report_generator import (
    ReportFacts,
    TemplateReportGenerator,
    _practical_significance,
    _practical_significance_threshold,
)
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.hypothesis import ExpectedDirection, Hypothesis
from app.schemas.statistics import HypothesisTestType, PowerAnalysisResult, StatResult


def _stat(metric="Conversion Rate", control="2.95%", variant="3.56%", mde_matching=True):
    return StatResult(
        metric=metric,
        test_type=HypothesisTestType.CHI_SQUARE,
        test_name="Chi-square test",
        statistic=13.8,
        selection_reason="Binary metric with sufficient expected cell counts.",
        control=control,
        variant=variant,
        delta="+20.6% (rel)",
        p_value=0.0002,
        significant=True,
        ci_lower="+0.29pp",
        ci_upper="+0.93pp",
    )


def _power(mde=15.1):
    return PowerAnalysisResult(
        minimum_detectable_effect_relative=mde,
        required_sample_size=10000,
        observed_sample_size=48312,
        achieved_power=0.965,
        alpha=0.05,
        is_sufficiently_powered=True,
    )


def _dataset():
    return DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=48312,
        metric_label="Conversion Rate",
        metric_selection_reason="Selected by the deterministic outcome-column priority.",
    )


def _hypothesis(**overrides):
    kwargs = dict(
        statement=(
            "If we introduce the new version of the checkout experience, the conversion "
            "rate will increase compared with the control group."
        ),
        primary_metric="Conversion Rate",
        expected_direction=ExpectedDirection.INCREASE,
        expected_effect_relative=0.05,
        rationale="Minimum practically meaningful improvement: +5% relative",
    )
    kwargs.update(overrides)
    return Hypothesis(**kwargs)


def _facts(*, hypothesis=None, stat=None, mde=15.1):
    stat = stat or _stat()
    return ReportFacts(
        user_prompt="Should we ship the new checkout experience?",
        dataset=_dataset(),
        quality_checks=[],
        srm_passed=True,
        stat_results=[stat],
        test_selections=[],
        power_analysis=_power(mde=mde),
        mde_display=f"{mde}% (relative)",
        sample_size_note="48,312 users observed — achieved power 96.5% at α=0.05",
        hypothesis=hypothesis,
    )


# --- 1. explicitly supplied expected effect is propagated and used --------


def test_user_specified_expected_effect_is_used_as_threshold():
    """The +5% relative expected effect wins over the 15.1% post-hoc MDE."""
    facts = _facts(hypothesis=_hypothesis(expected_effect_relative=0.05), mde=15.1)
    stat = facts.stat_results[0]

    threshold, source, _note = _practical_significance_threshold(facts, stat)
    assert source == "user_specified"
    assert threshold == 5.0  # 0.05 relative -> 5.0 percent, never 15.1 (the MDE)

    practical, reason, _source, _exceeds = _practical_significance(facts, stat)
    assert practical is True
    assert "user-specified expected effect of 5.0% relative" in reason
    assert "15.1" not in reason  # the MDE must not be silently substituted in


def test_user_specified_expected_effect_flows_into_decision_and_report_text():
    """End-to-end: TemplateReportGenerator's decision/report text uses the
    user-specified expected effect, not the MDE, once a hypothesis is
    present for the matched metric."""
    facts = _facts(hypothesis=_hypothesis(expected_effect_relative=0.05), mde=15.1)
    report = TemplateReportGenerator().generate(facts)

    assert report.practical_significance is True
    assert "user-specified expected effect of 5.0% relative" in report.decision_reason
    assert "user-specified expected effect of 5.0% relative" in report.executive_summary
    # the observed +20.6% clears +5% comfortably -> GO_WITH_CAUTION (no guardrails run)
    assert report.decision == "GO_WITH_CAUTION"


# --- 2. missing expected effect still uses the existing fallback ----------


def test_missing_expected_effect_falls_back_to_post_hoc_mde():
    """
    No hypothesis at all -> threshold still falls back to the post-hoc
    MDE (unchanged), but per the Phase 1 fix `practical` is reported as
    NOT ASSESSED (`None`), never a fabricated `True` business
    endorsement, purely because no pre-registered business threshold
    exists to compare against. `exceeds_threshold` still carries the
    raw magnitude comparison for `determine_decision`'s safety-net
    checks (see `_practical_significance`'s docstring) — that part of
    the pre-existing MDE-based behavior is unchanged.
    """
    facts = _facts(hypothesis=None, mde=15.1)
    stat = facts.stat_results[0]

    threshold, source, _note = _practical_significance_threshold(facts, stat)
    assert source == "post_hoc_mde"
    assert threshold == 15.1

    practical, reason, _source, exceeds_threshold = _practical_significance(facts, stat)
    assert practical is None
    assert exceeds_threshold is True
    assert "15.1%" in reason
    assert "not assessed" in reason.lower()


def test_hypothesis_without_expected_effect_falls_back_to_post_hoc_mde():
    """A hypothesis is present but didn't specify a magnitude -> still MDE."""
    facts = _facts(hypothesis=_hypothesis(expected_effect_relative=None), mde=15.1)
    stat = facts.stat_results[0]

    threshold, source, _note = _practical_significance_threshold(facts, stat)
    assert source == "post_hoc_mde"
    assert threshold == 15.1


def test_hypothesis_for_a_different_metric_falls_back_to_post_hoc_mde():
    """A hypothesis exists but targets a different primary_metric than the
    matched StatResult -> never borrow its expected effect for this metric."""
    facts = _facts(
        hypothesis=_hypothesis(primary_metric="Revenue", expected_effect_relative=0.05),
        mde=15.1,
    )
    stat = facts.stat_results[0]  # metric="Conversion Rate" — does not match "Revenue"

    threshold, source, _note = _practical_significance_threshold(facts, stat)
    assert source == "post_hoc_mde"
    assert threshold == 15.1


# --- 3. post-hoc MDE remains separately available/labeled -----------------


def test_post_hoc_mde_still_displayed_separately_when_expected_effect_used():
    """Even when the expected effect wins the comparison, the post-hoc MDE
    must remain visible elsewhere in the report, clearly labeled as such —
    never silently dropped."""
    facts = _facts(hypothesis=_hypothesis(expected_effect_relative=0.05), mde=15.1)
    report = TemplateReportGenerator().generate(facts)

    assert report.mde == "15.1% (relative)"
    assert "post-hoc MDE" in report.decision_reason
    assert "not the post-hoc MDE" not in report.mde  # mde field itself is just the value


# --- 4. relative effects are never confused with percentage-point effects -


def test_relative_expected_effect_not_confused_with_percentage_points():
    """expected_effect_relative=0.05 means +5% RELATIVE (2.95% -> ~3.10%),
    not +5 percentage points (2.95% -> 7.95%). The threshold used for the
    percent-based comparison must be 5.0, not 5.0 divided/multiplied
    incorrectly, and not treated as already being in percentage points."""
    control = 2.95
    hypothesis = _hypothesis(expected_effect_relative=0.05)

    # observed effect just barely clears +5% relative but would NOT clear
    # +5 percentage points (which would require variant >= 7.95%)
    stat = _stat(control="2.95%", variant="3.10%")  # +0.15pp, ~+5.08% relative
    facts = _facts(hypothesis=hypothesis, stat=stat, mde=15.1)

    threshold, source, _note = _practical_significance_threshold(facts, stat)
    assert source == "user_specified"
    assert threshold == 5.0  # exactly the relative percent, not a pp value

    observed_relative_pct = (3.10 - control) / control * 100
    assert observed_relative_pct > threshold  # clears the *relative* bar...

    observed_pp = 3.10 - control
    assert observed_pp < threshold  # ...but would NOT clear a 5-point bar

    practical, _reason, _source, _exceeds = _practical_significance(facts, stat)
    assert practical is True  # correctly evaluated against the relative threshold
