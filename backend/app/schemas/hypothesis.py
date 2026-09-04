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
