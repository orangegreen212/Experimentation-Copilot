"""
Deterministic Decision Support — Phase 3.

Turns already-computed facts (`StatResult`, `HypothesisEvaluation`,
`DatasetInfo`) into a structured `DecisionSupport` object that answers
"what does this experiment result mean for the business?" This is
explicitly NOT a second statistical analysis (Phase 3 spec, intro):

  - No p-values, deltas, effects, or significance are (re)computed
    here. Every number either comes straight off an existing
    `StatResult`/`HypothesisEvaluation` field, or off `_display_number`
    (already used by `report_generator._practical_significance` to
    turn a display string like "10.87%" back into a float) — reused
    here rather than reimplemented, per Phase 3 spec §14 ("reuse
    existing deterministic values wherever possible").
  - No LLM is involved anywhere in this module.
  - `evaluate_hypothesis()` (app/stats/hypothesis_evaluator.py) is
    never re-run or duplicated here — its output
    (`HypothesisEvaluation`) is consumed as-is.
  - `determine_decision()` / the final `Decision` (GO/NO-GO-equivalent)
    is never read, overridden, or reproduced here. Decision Support is
    explanatory/annotative only (Phase 3 spec §5, §8).

ABSOLUTE BUSINESS IMPACT (Phase 3 spec §3, refined by the approved
plan): `dataset.users` is used as the exposure denominator ONLY when
ALL of the following hold —

  1. the primary metric's `StatResult` is binary/conversion-style,
     identified deterministically by `test_type` being CHI_SQUARE or
     FISHERS_EXACT (the only tests `select_test()` ever chooses for a
     binary metric — see hypothesis_tests.py's module docstring; this
     is reading an existing deterministic fact, not a new
     classification);
  2. `dataset.type` is not `UNKNOWN` (i.e. the dataset was confidently
     classified as user-level/aggregated-by-user, not an
     unrecognized shape);
  3. `dataset.users` is a positive int;
  4. the baseline value (parsed from `StatResult.control`, a
     percentage display string for binary metrics) is available and
     non-null.

If any condition fails, `impact_calculation_method="unavailable"` and
a `warnings` entry explains why — an absolute impact number is NEVER
fabricated. `dataset.users` is NEVER used this way for continuous
metrics (Welch's t-test / Mann-Whitney U — includes revenue, AOV, and
any other continuous metric) — Phase 3 spec §10 and the approved plan
both forbid this explicitly, since "average metric * user count" is
not a valid incremental-impact calculation without a real exposure
denominator, which this phase does not add.
"""

from __future__ import annotations

from app.graph.report_generator import _display_number
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.decision_support import (
    AdditionalMetricComparison,
    DecisionSupport,
    GuardrailFinding,
)
from app.schemas.hypothesis import ExpectedDirection, Hypothesis
from app.schemas.hypothesis_evaluation import HypothesisEvaluation, HypothesisVerdict
from app.schemas.statistics import HypothesisTestType, StatResult

_BINARY_TEST_TYPES = (HypothesisTestType.CHI_SQUARE, HypothesisTestType.FISHERS_EXACT)

_VERDICT_INTERPRETATION: dict[HypothesisVerdict, str] = {
    HypothesisVerdict.SUPPORTED: (
        "The observed effect met or exceeded the expected effect and was statistically significant."
    ),
    HypothesisVerdict.PARTIALLY_SUPPORTED: (
        "The metric moved in the expected direction and was statistically significant, but the "
        "observed effect was smaller than the expected effect."
    ),
    HypothesisVerdict.NOT_SUPPORTED: (
        "The experiment did not provide sufficient evidence that the hypothesis was achieved."
    ),
}


def _find_matching_stat_result(metric: str, stat_results: list[StatResult]) -> StatResult | None:
    """Same exact-match rule as `hypothesis_evaluator._find_matching_stat_result` — never fuzzy."""
    for result in stat_results:
        if result.is_omnibus:
            continue
        if result.metric == metric:
            return result
    return None


def _direction_of(relative_change: float | None) -> str:
    if relative_change is None or relative_change == 0:
        return "no_change"
    return "increase" if relative_change > 0 else "decrease"


def _additional_metric_comparisons(
    primary_metric: str, stat_results: list[StatResult]
) -> list[AdditionalMetricComparison]:
    """
    Every non-primary, non-omnibus metric already present in
    `stat_results` (Phase 3 spec §4). No new test is run — this only
    reads fields already on each `StatResult`.
    """
    comparisons: list[AdditionalMetricComparison] = []
    for result in stat_results:
        if result.is_omnibus or result.metric == primary_metric:
            continue
        baseline = _display_number(result.control)
        observed = _display_number(result.variant)
        absolute_change = (observed - baseline) if (baseline is not None and observed is not None) else None
        relative_change = result.observed_relative_effect
        comparisons.append(
            AdditionalMetricComparison(
                metric=result.metric,
                baseline_value=baseline,
                observed_value=observed,
                absolute_change=absolute_change,
                relative_change=relative_change,
                statistically_significant=result.significant,
                direction=_direction_of(relative_change if relative_change is not None else absolute_change),
            )
        )
    return comparisons


def _guardrail_findings(guardrail_results: list[StatResult]) -> tuple[list[GuardrailFinding], bool]:
    """
    Deterministic guardrail status (Phase 3 spec §5). A guardrail is
    "violated" only when it is BOTH statistically significant AND
    moved in the negative/harmful direction (same rule
    `determine_decision()` already applies to `guardrail_results` in
    report_generator.py — read here, not reimplemented differently).
    """
    findings: list[GuardrailFinding] = []
    any_violated = False
    for result in guardrail_results:
        observed = _display_number(result.variant)
        relative_change = result.observed_relative_effect
        # Direction-aware (doc3 §6/§7): "harmful" depends on the metric —
        # an increase in Bounce Rate is bad, an increase in Revenue is
        # not. See StatResult.higher_is_better
        # (app.stats.dataset_classifier.infer_guardrail_direction), set
        # by guardrail_node.py. Falls back to the delta string's sign
        # only when observed_relative_effect is None (zero-baseline).
        effect = relative_change if relative_change is not None else (_display_number(result.delta) or 0.0)
        is_harmful = (effect < 0) if result.higher_is_better else (effect > 0)
        violated = bool(result.significant and is_harmful)
        any_violated = any_violated or violated
        findings.append(
            GuardrailFinding(
                metric=result.metric,
                observed_value=observed,
                relative_change=relative_change,
                statistically_significant=result.significant,
                violated=violated,
            )
        )
    return findings, any_violated


def _business_interpretation(
    verdict: HypothesisVerdict | None,
    additional_metrics: list[AdditionalMetricComparison],
    guardrail_findings_: list[GuardrailFinding],
    guardrail_violated: bool,
) -> str | None:
    """
    Plain-Python sentence assembly from already-decided facts — not a
    template the LLM fills in, and not a new judgment call: every
    clause below is a direct restatement of a boolean/enum already
    computed elsewhere (verdict, `violated`, `statistically_significant`).
    """
    parts: list[str] = []
    if verdict is not None:
        parts.append(_VERDICT_INTERPRETATION[verdict])

    trade_offs = [
        m for m in additional_metrics if m.statistically_significant and m.direction == "decrease"
    ]
    if trade_offs:
        names = ", ".join(m.metric for m in trade_offs)
        parts.append(f"Primary metric improved, but {names} decreased.")

    if guardrail_violated:
        violated_names = ", ".join(g.metric for g in guardrail_findings_ if g.violated)
        parts.append(
            f"The {violated_names} guardrail deteriorated. The experiment should not be "
            "considered a straightforward positive outcome."
        )

    return " ".join(parts) if parts else None


def build_decision_support(
    *,
    hypothesis: Hypothesis | None,
    hypothesis_evaluation: HypothesisEvaluation | None,
    stat_results: list[StatResult],
    guardrail_results: list[StatResult],
    dataset: DatasetInfo,
) -> DecisionSupport | None:
    """
    The single deterministic entry point (Phase 3 spec §1). Returns
    `None` when there is no hypothesis at all — Decision Support never
    fabricates hypothesis-shaped facts for a run that didn't provide
    one (Phase 3 spec §7), matching `evaluate_hypothesis()`'s own
    None-when-no-hypothesis rule.
    """
    if hypothesis is None:
        return None

    warnings: list[str] = []

    additional_metrics = _additional_metric_comparisons(hypothesis.primary_metric, stat_results)
    guardrail_findings_, guardrail_violated = _guardrail_findings(guardrail_results)

    matched = _find_matching_stat_result(hypothesis.primary_metric, stat_results)

    if hypothesis_evaluation is None or matched is None:
        warnings.append(
            f'No statistical result was found for the hypothesis\'s primary metric '
            f'"{hypothesis.primary_metric}" — expected-vs-observed impact cannot be calculated.'
        )
        return DecisionSupport(
            available=False,
            primary_metric=hypothesis.primary_metric,
            expected_effect_relative=hypothesis.expected_effect_relative,
            impact_calculation_method="unavailable",
            additional_metrics=additional_metrics,
            guardrail_findings=guardrail_findings_,
            guardrail_violated=guardrail_violated,
            warnings=warnings,
        )

    baseline_value = _display_number(matched.control)
    observed_value = _display_number(matched.variant)
    observed_effect_relative = hypothesis_evaluation.observed_effect_relative

    if baseline_value is None:
        warnings.append(
            f'The baseline value for "{hypothesis.primary_metric}" is unavailable — impact cannot '
            "be calculated. A baseline is never inferred."
        )

    observed_effect_absolute = (
        (observed_value - baseline_value) if (baseline_value is not None and observed_value is not None) else None
    )

    # --- expected value (spec §2) ---------------------------------------
    expected_value: float | None = None
    if (
        baseline_value is not None
        and hypothesis.expected_effect_relative is not None
        and hypothesis.expected_direction in (ExpectedDirection.INCREASE, ExpectedDirection.DECREASE)
    ):
        if hypothesis.expected_direction == ExpectedDirection.INCREASE:
            expected_value = baseline_value * (1 + hypothesis.expected_effect_relative)
        else:
            expected_value = baseline_value * (1 - hypothesis.expected_effect_relative)

    # --- absolute business impact (spec §3, approved-plan rule) --------
    impact_calculation_method = "unavailable"
    baseline_expected_count: float | None = None
    observed_count: float | None = None
    incremental_count: float | None = None

    is_binary_metric = matched.test_type in _BINARY_TEST_TYPES
    dataset_is_user_level = dataset.type != DatasetType.UNKNOWN
    population = dataset.users

    if not is_binary_metric:
        warnings.append(
            f'"{hypothesis.primary_metric}" is not a binary/conversion-style metric — absolute '
            "business impact is not calculated from user count for continuous metrics (e.g. "
            "revenue, AOV) in this phase."
        )
    elif not dataset_is_user_level:
        warnings.append(
            "The dataset's structure could not be confidently classified as user-level — "
            "absolute business impact cannot be reliably calculated."
        )
    elif not (isinstance(population, int) and population > 0):
        warnings.append("No valid user population is available — absolute business impact cannot be calculated.")
    elif baseline_value is None or observed_value is None:
        warnings.append(
            "Baseline or observed value is unavailable — absolute business impact cannot be calculated."
        )
    else:
        # baseline_value / observed_value are percentage-point display
        # numbers (e.g. 10.0 for "10.00%") for binary metrics — convert
        # to a fraction before multiplying by the user population.
        baseline_expected_count = (baseline_value / 100.0) * population
        observed_count = (observed_value / 100.0) * population
        incremental_count = observed_count - baseline_expected_count
        impact_calculation_method = "population_scaled"
        warnings.append(
            "This is the observed experimental impact scaled to the reported dataset population, "
            "not a projected/scaled forecast for a different or future population."
        )

    business_interpretation = _business_interpretation(
        hypothesis_evaluation.verdict, additional_metrics, guardrail_findings_, guardrail_violated
    )

    return DecisionSupport(
        available=True,
        primary_metric=hypothesis.primary_metric,
        baseline_value=baseline_value,
        observed_value=observed_value,
        observed_effect_absolute=observed_effect_absolute,
        observed_effect_relative=observed_effect_relative,
        expected_effect_relative=hypothesis.expected_effect_relative,
        expected_value=expected_value,
        effect_achievement_ratio=hypothesis_evaluation.effect_achievement_ratio,
        statistical_significance=hypothesis_evaluation.statistically_significant,
        hypothesis_verdict=hypothesis_evaluation.verdict,
        business_interpretation=business_interpretation,
        impact_calculation_method=impact_calculation_method,
        baseline_expected_count=baseline_expected_count,
        observed_count=observed_count,
        incremental_count=incremental_count,
        additional_metrics=additional_metrics,
        guardrail_findings=guardrail_findings_,
        guardrail_violated=guardrail_violated,
        warnings=warnings,
    )
