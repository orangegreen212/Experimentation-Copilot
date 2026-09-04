"""
Deterministic Hypothesis Evaluator — Phase 2.

Compares a Phase 1 `Hypothesis` against the already-computed
`StatResult` for its `primary_metric` and produces a structured
`HypothesisEvaluation`. Pure Python, zero LLM involvement anywhere in
this module — no LLM ever calculates the effect, significance,
direction, achievement ratio, or verdict (Phase 2 spec §15).

This module does NOT recompute or duplicate any statistical formula.
It only reads `StatResult.observed_relative_effect` and
`StatResult.significant` — both already computed in
`hypothesis_tests.py` — and applies the deterministic comparison
rules below.

VERDICT RULES (Phase 2 spec §2-4), summarized:

  Case A — hypothesis.expected_effect_relative IS provided (only
  possible when expected_direction is 'increase' or 'decrease' —
  Hypothesis's own schema rejects a zero/negative magnitude for those
  directions):
    - wrong direction                              -> NOT_SUPPORTED
    - not statistically significant                -> NOT_SUPPORTED
    - right direction, significant, ratio >= 1.0    -> SUPPORTED
    - right direction, significant, ratio <  1.0    -> PARTIALLY_SUPPORTED

  Case B — no expected_effect_relative provided (direction 'increase'
  or 'decrease'):
    - direction_supported AND significant           -> SUPPORTED
    - otherwise (wrong direction OR not significant) -> NOT_SUPPORTED
    - PARTIALLY_SUPPORTED is never reachable here — there is no
      magnitude target to "fall short of" (spec §4).

  expected_direction == 'no_change' — OUT OF SCOPE for Phase 2. The
  spec only defines direction_supported/verdict rules for 'increase'
  and 'decrease'. A 'no_change' hypothesis is deliberately NOT
  evaluated: a non-significant result does not prove the absence of an
  effect — that would require a dedicated equivalence-testing
  methodology (e.g. TOST), which this phase does not implement. So
  `direction_supported := not significant` would encode an incorrect
  statistical interpretation and is NOT used. Any `no_change`
  hypothesis always yields an "unavailable" evaluation (no verdict,
  see `_unavailable()`) regardless of what the matched StatResult
  shows — never a fabricated verdict.
"""

from __future__ import annotations

from app.schemas.hypothesis import ExpectedDirection, Hypothesis
from app.schemas.hypothesis_evaluation import HypothesisEvaluation, HypothesisVerdict
from app.schemas.statistics import StatResult


def _find_matching_stat_result(primary_metric: str, stat_results: list[StatResult]) -> StatResult | None:
    """
    Match against the canonical metric identifier already used by the
    rest of the system (`StatResult.metric`, the same string as
    `DatasetInfo.metric_label` / `available_metrics`, which is what
    the frontend's Hypothesis form's metric dropdown is populated
    from — see hypothesis-form.tsx). Exact string match, never fuzzy —
    a silent near-match would risk evaluating the wrong metric.

    Omnibus rows (`is_omnibus=True`, multi-arm 3+ case) are always
    skipped: they represent "does any arm differ" across all arms, not
    a single well-defined relative effect for one comparison, and
    `observed_relative_effect` is never set on them (see
    hypothesis_tests.py's `compute_multi_arm_stat_results`). Skipping
    them here also means a caller never accidentally matches the
    always-present omnibus row before reaching the real pairwise
    result later in the list.
    """
    for result in stat_results:
        if result.is_omnibus:
            continue
        if result.metric == primary_metric:
            return result
    return None


def _direction_supported(expected_direction: ExpectedDirection, observed_effect: float) -> bool:
    """
    Purely directional — Phase 2 spec §3. Deliberately independent of
    significance. Only ever called for 'increase'/'decrease' —
    'no_change' is short-circuited to an unavailable evaluation before
    this is reached (see module docstring).
    """
    if expected_direction == ExpectedDirection.INCREASE:
        return observed_effect > 0
    return observed_effect < 0  # ExpectedDirection.DECREASE


def _unavailable(hypothesis: Hypothesis, *, metric_matched: bool, evaluation_note: str) -> HypothesisEvaluation:
    return HypothesisEvaluation(
        hypothesis_present=True,
        expected_direction=hypothesis.expected_direction,
        expected_effect_relative=hypothesis.expected_effect_relative,
        observed_effect_relative=None,
        direction_supported=None,
        statistically_significant=None,
        effect_achievement_ratio=None,
        verdict=None,
        metric_matched=metric_matched,
        evaluation_note=evaluation_note,
    )


def evaluate_hypothesis(
    hypothesis: Hypothesis | None,
    stat_results: list[StatResult],
) -> HypothesisEvaluation | None:
    """
    The single deterministic entry point. Returns `None` when there is
    no hypothesis at all (Phase 2 spec §1 — never a fabricated
    verdict; the whole object is simply absent). Otherwise always
    returns a `HypothesisEvaluation`, which itself may represent an
    "unavailable" evaluation (see `_unavailable()` above) when the
    primary metric can't be matched, its relative effect is undefined,
    or `expected_direction` is `no_change` (out of scope this phase —
    see module docstring).
    """
    if hypothesis is None:
        return None

    if hypothesis.expected_direction == ExpectedDirection.NO_CHANGE:
        return _unavailable(
            hypothesis,
            metric_matched=True,
            evaluation_note=(
                "Hypothesis evaluation for expected_direction='no_change' is not yet "
                "implemented — a non-significant result does not itself prove the absence of "
                "an effect, and correctly testing that requires a dedicated equivalence-testing "
                "methodology this phase does not provide. No verdict is produced."
            ),
        )

    matched = _find_matching_stat_result(hypothesis.primary_metric, stat_results)
    if matched is None:
        return _unavailable(
            hypothesis,
            metric_matched=False,
            evaluation_note=(
                f'No statistical result was found for the hypothesis\'s primary metric '
                f'"{hypothesis.primary_metric}" — evaluation unavailable. This does not '
                f"necessarily mean the metric doesn't exist in the dataset; it may not have "
                f"been part of this run's statistical analysis."
            ),
        )

    observed_effect = matched.observed_relative_effect
    if observed_effect is None:
        return _unavailable(
            hypothesis,
            metric_matched=True,
            evaluation_note=(
                f'The statistical result for "{hypothesis.primary_metric}" has no defined '
                f"relative effect (the control-arm baseline was zero) — evaluation unavailable."
            ),
        )

    significant = matched.significant
    direction_supported = _direction_supported(hypothesis.expected_direction, observed_effect)

    effect_achievement_ratio: float | None = None
    if hypothesis.expected_effect_relative is not None and hypothesis.expected_effect_relative != 0:
        # Never divide by zero (Phase 2 spec §6) — Hypothesis's own
        # validator already rejects exactly 0 for increase/decrease,
        # but this guard is kept regardless as defense in depth.
        effect_achievement_ratio = observed_effect / hypothesis.expected_effect_relative

    verdict = _compute_verdict(
        direction_supported=direction_supported,
        significant=significant,
        effect_achievement_ratio=effect_achievement_ratio,
    )

    return HypothesisEvaluation(
        hypothesis_present=True,
        expected_direction=hypothesis.expected_direction,
        expected_effect_relative=hypothesis.expected_effect_relative,
        observed_effect_relative=observed_effect,
        direction_supported=direction_supported,
        statistically_significant=significant,
        effect_achievement_ratio=effect_achievement_ratio,
        verdict=verdict,
        metric_matched=True,
        evaluation_note=None,
    )


def _compute_verdict(
    *,
    direction_supported: bool,
    significant: bool,
    effect_achievement_ratio: float | None,
) -> HypothesisVerdict:
    if not direction_supported:
        return HypothesisVerdict.NOT_SUPPORTED
    if not significant:
        return HypothesisVerdict.NOT_SUPPORTED

    if effect_achievement_ratio is None:
        # Case B (Phase 2 spec §4) — no expected magnitude to compare
        # against. PARTIALLY_SUPPORTED is not reachable here.
        return HypothesisVerdict.SUPPORTED

    # Case A (Phase 2 spec §2) — ratio >= 1.0 means the observed
    # magnitude met or exceeded the expected magnitude (both are
    # guaranteed the same sign here, since direction_supported is
    # True, so this magnitude comparison via the signed ratio is
    # equivalent to comparing absolute magnitudes).
    if effect_achievement_ratio >= 1.0:
        return HypothesisVerdict.SUPPORTED
    return HypothesisVerdict.PARTIALLY_SUPPORTED
