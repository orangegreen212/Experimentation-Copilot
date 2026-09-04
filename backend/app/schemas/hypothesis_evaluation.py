"""
Hypothesis Evaluation schema — Phase 2.

Deterministic comparison of the Phase 1 `Hypothesis` (what the analyst
expected) against the already-computed `StatResult` (what actually
happened). See `app/stats/hypothesis_evaluator.py` for the pure-Python
logic that produces this object — no LLM is involved anywhere in this
computation (see that module's docstring for the full rule set this
implements).

SCOPE (Phase 2 only): this is strictly "compare and produce a
verdict." It does NOT implement metric metadata/registry, guardrails,
segmentation, business impact, decision support, or recommendations —
those remain out of scope.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.hypothesis import ExpectedDirection


class HypothesisVerdict(str, Enum):
    """
    Closed set, exactly three values per the Phase 2 spec — no
    `NO_HYPOTHESIS`/`UNAVAILABLE` member. When there is no hypothesis,
    or evaluation isn't possible (metric not matched, or the matched
    result's `observed_relative_effect` is undefined), the caller
    represents that by NOT constructing a verdict at all — see
    `HypothesisEvaluation.verdict: HypothesisVerdict | None` and its
    `metric_matched`/`evaluation_note` fields, not by adding a value
    here. A verdict is never fabricated.
    """

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class HypothesisEvaluation(CamelModel):
    """
    Deterministic comparison of one `Hypothesis` against the
    `StatResult` row matching its `primary_metric`.

    `hypothesis_present` is always `True` on any constructed instance
    of this object — when there is NO hypothesis, the caller (see
    `evaluate_hypothesis()`) returns `None` instead of an instance
    with this flag `False`, per the Phase 2 spec ("hypothesis_evaluation
    = None" for the no-hypothesis case). The flag is kept anyway
    for API-shape completeness/explicitness, matching the originally
    specified field list.

    UNAVAILABLE EVALUATION (metric not found, or the matched result's
    relative effect is undefined — e.g. a zero control-arm baseline):
    this is represented EXPLICITLY, not silently, by `metric_matched`
    and/or `evaluation_note` being set while `direction_supported`,
    `statistically_significant`, `effect_achievement_ratio`, and
    `verdict` all stay `None` — no verdict is ever fabricated from
    incomplete data. This follows the project's existing pattern of
    pairing a boolean/optional result with an explanatory string (e.g.
    `QualityCheck.detail`, `TestSelectionResult.reason`) rather than
    adding a new enum member for "N/A".
    """

    hypothesis_present: bool = True
    expected_direction: ExpectedDirection
    expected_effect_relative: float | None
    # The exact number from the matched StatResult.observed_relative_effect
    # — never recomputed, never parsed from a display string. None when
    # no metric could be matched, or the matched result's own relative
    # effect is undefined (see StatResult.observed_relative_effect).
    observed_effect_relative: float | None = None
    # Purely directional — "does the observed effect have the expected
    # sign?" — deliberately independent of statistical_significant (see
    # evaluate_hypothesis()'s docstring, Phase 2 spec §3). None only
    # when evaluation itself is unavailable (see class docstring).
    direction_supported: bool | None = None
    # Copied verbatim from the matched StatResult.significant. None
    # only when evaluation is unavailable.
    statistically_significant: bool | None = None
    # observed_effect_relative / expected_effect_relative. Only ever
    # computed when expected_effect_relative was provided and is
    # nonzero (Hypothesis's own schema now rejects exactly 0 for
    # increase/decrease directions — see hypothesis.py — but this is
    # still guarded defensively here). Never NaN/inf — None instead.
    effect_achievement_ratio: float | None = None
    verdict: HypothesisVerdict | None = None
    # False when hypothesis.primary_metric had no matching StatResult
    # row (Phase 2 spec §11) — evaluation is unavailable in that case,
    # never silently substituted with a different metric's result.
    metric_matched: bool = True
    # Human-readable reason whenever evaluation is degraded/unavailable
    # (metric not matched, or observed_effect_relative undefined).
    # None when evaluation completed normally.
    evaluation_note: str | None = Field(
        default=None,
        description=(
            "Set only when evaluation is unavailable — explains why "
            "(e.g. metric not found, or the matched result's relative "
            "effect is undefined). None when evaluation completed."
        ),
    )
