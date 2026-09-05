"""
Experiment Hypothesis schema — Phase 1.

Captures what the analyst expected BEFORE looking at the result, as
structured, validated context — never as free text stuffed into an
LLM prompt or the LangGraph state. See module docstrings on
`app/graph/state.py` and `app/graph/report_generator.py` for how this
flows: request -> GraphState -> ReportFacts.

SCOPE (Phase 1 only): this file only defines "capture, validate,
store, carry through the pipeline." It deliberately does NOT compute
or represent:
  - a verdict (SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED)
  - an expected-vs-observed comparison
  - any decision-support/recommendation logic

Those are explicitly out of scope for this phase and belong to a
later phase that consumes this schema, not this file.

ARCHITECTURAL BOUNDARY: a `Hypothesis` is input context, structurally
identical in spirit to `AnalysisSettings` — it flows through
deterministic Python code only. No LLM is ever given write access to
these fields, and no LLM call is part of validating, normalizing, or
constructing a `Hypothesis` (see `report_generator.py`'s `ReportFacts`
docstring addition for the "LLM never modifies input context" rule
this also follows).
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import Field, field_validator, model_validator

from app.schemas.base import CamelModel
from app.schemas.statistics import MetricType

# Reasonable maximum length for a free-text hypothesis statement —
# long enough for a real, specific hypothesis sentence or two, short
# enough to keep this a structured field rather than an essay.
_MAX_STATEMENT_LENGTH = 500
_MAX_RATIONALE_LENGTH = 2000


class ExpectedDirection(str, Enum):
    """
    The analyst's predicted direction of the effect on the primary
    metric. Deliberately a closed set (not a free string) — mirrors
    the project's existing enum convention (see `MetricType`,
    `HypothesisTestType` in schemas/statistics.py).
    """

    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"


class Hypothesis(CamelModel):
    """
    A structured, pre-registered experiment hypothesis.

    `expected_effect_relative` is explicitly a RELATIVE effect, never
    percentage points — e.g. `0.05` means "+5% relative": a baseline
    of 10% is expected to become 10.5% (10% * 1.05), NOT 15%
    (10% + 5 percentage points). This is deliberately a plain float
    (not a formatted display string like `StatResult`'s fields),
    because it is an INPUT the analyst provides, not a computed
    display value — see this field's `description` below, which is
    the API-visible documentation of the distinction the ticket asked
    for.

    No LLM is involved in constructing, validating, or normalizing
    this object — every rule below is a plain deterministic Pydantic
    validator. In particular, a value here is NEVER silently
    reinterpreted (e.g. "5" is never treated as "5 percentage
    points" or auto-divided by 100) — the caller must supply the
    relative fraction directly (0.05, not 5).
    """

    statement: str = Field(
        description=(
            "The analyst's hypothesis in plain language, e.g. "
            '"Increasing the checkout CTA visibility will increase checkout conversion."'
        )
    )
    primary_metric: str = Field(
        description=(
            "The metric this hypothesis is about. Should match one of the dataset's "
            "already-detected metrics (DatasetInfo.available_metrics / metric_label) — "
            "this is NOT a second, independent metric-selection mechanism."
        )
    )
    expected_direction: ExpectedDirection = Field(
        description="The predicted direction of the effect on primary_metric."
    )
    expected_effect_relative: float | None = Field(
        default=None,
        description=(
            "Optional. A RELATIVE effect, expressed as a fraction — e.g. 0.05 means "
            '"+5% relative" (baseline 10% -> expected 10.5%), NOT +5 percentage points '
            "(which would be 10% -> 15%). Never percentage points."
        ),
    )
    rationale: str | None = Field(
        default=None,
        description="Optional free-text business rationale for why this effect is expected.",
    )

    @field_validator("statement")
    @classmethod
    def _validate_statement(cls, value: str) -> str:
        stripped = (value or "").strip()
        if not stripped:
            raise ValueError("statement is required and must not be empty.")
        if len(stripped) > _MAX_STATEMENT_LENGTH:
            raise ValueError(
                f"statement must be at most {_MAX_STATEMENT_LENGTH} characters "
                f"(got {len(stripped)})."
            )
        return stripped

    @field_validator("primary_metric")
    @classmethod
    def _validate_primary_metric(cls, value: str) -> str:
        stripped = (value or "").strip()
        if not stripped:
            raise ValueError("primary_metric is required and must not be empty.")
        return stripped

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            # Optional field submitted as blank/whitespace — treat as omitted
            # rather than as an empty string sitting in stored context.
            return None
        if len(stripped) > _MAX_RATIONALE_LENGTH:
            raise ValueError(
                f"rationale must be at most {_MAX_RATIONALE_LENGTH} characters "
                f"(got {len(stripped)})."
            )
        return stripped

    @field_validator("expected_effect_relative")
    @classmethod
    def _validate_effect_is_finite(cls, value: float | None) -> float | None:
        if value is None:
            return value
        # Pydantic already rejects non-numeric input at the type-coercion
        # level (str, None handled above); this guards the remaining
        # numeric edge cases (NaN, +/-inf) that a float type alone allows.
        if not math.isfinite(value):
            raise ValueError("expected_effect_relative must be a finite number (not NaN or infinite).")
        return value

    @model_validator(mode="after")
    def _validate_effect_sign_matches_direction(self) -> "Hypothesis":
        if self.expected_effect_relative is None:
            return self
        if self.expected_direction in (ExpectedDirection.INCREASE, ExpectedDirection.DECREASE):
            if self.expected_effect_relative < 0:
                raise ValueError(
                    "expected_effect_relative must be non-negative when expected_direction is "
                    "'increase' or 'decrease' — the sign of the change is carried by "
                    "expected_direction, not by this field. Provide the expected MAGNITUDE "
                    "here (e.g. 0.05 for a 5% relative change in either direction)."
                )
            if self.expected_effect_relative == 0:
                # Phase 2 (§6 — Zero expected effect): zero has no meaningful
                # interpretation for an 'increase'/'decrease' hypothesis — an
                # expected magnitude of exactly 0% is really a 'no_change'
                # hypothesis. Rejecting it here (rather than special-casing a
                # zero-magnitude "increase" downstream) keeps the effect-
                # achievement-ratio division in HypothesisEvaluator free of
                # a zero-denominator case entirely — see
                # app/stats/hypothesis_evaluator.py for the corresponding
                # defensive (never-divide-by-zero) handling kept there
                # anyway, in case this object is ever constructed via a path
                # that bypasses this validator (e.g. model_construct in a
                # future migration of old persisted data).
                raise ValueError(
                    "expected_effect_relative must not be exactly 0 when expected_direction is "
                    "'increase' or 'decrease' — an expected magnitude of 0% is a 'no_change' "
                    "hypothesis; set expected_direction to 'no_change' instead (and typically "
                    "omit expected_effect_relative entirely, since 'no change' has no target "
                    "magnitude to compare against)."
                )
        return self


# ---------------------------------------------------------------------------
# Sample size planning — pre-experiment "Create Experiment" screen.
#
# Deliberately separate from the Hypothesis class above: a Hypothesis is
# captured and stored alongside a dataset (post-hoc evaluation context).
# A sample-size PLAN happens before any dataset exists — it is never
# persisted against an experiment run, only computed on demand by
# POST /experiments/plan-sample-size and shown to the analyst.
# ---------------------------------------------------------------------------


class SampleSizePlanRequest(CamelModel):
    """
    What the analyst can reasonably guess BEFORE running anything: an
    assumed baseline for the primary metric, and the smallest relative
    change worth being able to detect. See
    `app/stats/power_analysis.py::plan_required_sample_size` for the
    actual calculation this feeds.
    """

    metric_type: MetricType = Field(
        description="BINARY for a conversion-style metric (e.g. checkout rate); "
        "CONTINUOUS for a metric like order value or session duration."
    )
    baseline_rate: float = Field(
        gt=0,
        description="Assumed current value of the primary metric under control — "
        "a proportion between 0 and 1 for BINARY (e.g. 0.12 for a 12% conversion "
        "rate), or the assumed mean for CONTINUOUS (e.g. 45.0 for average order value).",
    )
    baseline_std: float | None = Field(
        default=None,
        gt=0,
        description="Required for CONTINUOUS metrics only — the assumed standard "
        "deviation of the metric. Ignored for BINARY metrics, whose variance is "
        "fully determined by baseline_rate.",
    )
    mde_relative_pct: float = Field(
        gt=0,
        description="Smallest RELATIVE change (in percent) worth being able to "
        "detect, e.g. 5.0 for 'a 5% relative lift'.",
    )
    num_variants: int = Field(
        default=2, ge=2, le=10,
        description="Total number of arms including control (2 for classic A/B, "
        "3+ for A/B/n) — required sample size is computed per arm, then "
        "multiplied by this to get the total.",
    )
    daily_traffic_per_arm: int | None = Field(
        default=None, gt=0,
        description="Optional — expected number of eligible users per arm per "
        "day. If provided, the response includes an estimated number of days "
        "to reach the required sample size.",
    )

    @model_validator(mode="after")
    def _baseline_rate_valid_for_binary(self) -> "SampleSizePlanRequest":
        if self.metric_type == MetricType.BINARY and not (0 < self.baseline_rate < 1):
            raise ValueError("baseline_rate must be between 0 and 1 (exclusive) for a BINARY metric.")
        return self

    @model_validator(mode="after")
    def _baseline_std_required_for_continuous(self) -> "SampleSizePlanRequest":
        if self.metric_type != MetricType.BINARY and not self.baseline_std:
            raise ValueError("baseline_std is required when metric_type is not BINARY.")
        return self


class SampleSizePlan(CamelModel):
    """Pure calculation result from plan_required_sample_size — no
    formatting, no display strings (unlike PowerAnalysisResult's
    siblings in statistics.py), since the frontend needs the raw
    numbers to build its own summary sentence."""

    required_n_per_arm: int
    required_n_total: int
    alpha: float
    target_power: float
    effect_size: float


class SampleSizePlanResponse(CamelModel):
    plan: SampleSizePlan
    estimated_days: float | None = Field(
        default=None,
        description="required_n_per_arm / daily_traffic_per_arm, rounded up — "
        "omitted if daily_traffic_per_arm wasn't provided in the request.",
    )
