"""
Segmentation — Phase 5 schemas.

Pure data shapes for `app/stats/segmentation.py`'s deterministic,
no-LLM segmentation analysis. See that module's docstring for the
guardrails these shapes exist to represent (minimum segment size,
cardinality limits, missing-value limits, identifier/metric exclusion,
multiple-comparisons correction).

These are facts, not statistics: everything here is produced by
`app/stats/segmentation.py` and consumed downstream by the report
layer — this file defines no behavior of its own.
"""

from __future__ import annotations

from enum import Enum

from app.schemas.base import CamelModel
from app.schemas.statistics import StatResult


class SegmentSkipReason(str, Enum):
    """Why a candidate column was not used as a segmentation dimension."""

    IS_IDENTIFIER_OR_METRIC_COLUMN = "is_identifier_or_metric_column"
    EXCESSIVE_MISSING_VALUES = "excessive_missing_values"
    TOO_FEW_DISTINCT_VALUES = "too_few_distinct_values"
    HIGH_CARDINALITY = "high_cardinality"
    POST_TREATMENT_VARIABLE = "post_treatment_variable"


class SegmentSampleSizeStatus(str, Enum):
    """Whether a single segment value had enough per-arm observations to test."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class SkippedDimension(CamelModel):
    """A candidate column that was excluded from segmentation, and why."""

    column: str
    reason: SegmentSkipReason
    detail: str


class SegmentEffect(CamelModel):
    """Control-vs-variant comparison for one value of one dimension."""

    segment_value: str
    control_n: int
    variant_n: int
    sample_size_status: SegmentSampleSizeStatus
    stat_result: StatResult | None = None
    skip_detail: str | None = None


class SegmentDimensionResult(CamelModel):
    """
    Full within-dimension analysis: every segment value's effect, plus
    correction.

    Phase 2 fix (heterogeneity-logic audit): `has_reliable_segment_effect`
    and `has_heterogeneous_effect` are DELIBERATELY separate, independent
    facts — conflating them was the root bug this schema now prevents by
    construction:

      - `has_reliable_segment_effect` — "is the treatment effect
        statistically significant in at least one segment, after
        Holm-Bonferroni correction across the segments tested within
        this dimension?" This is WITHIN-segment significance. It says
        nothing about whether segments differ from each other.
      - `has_heterogeneous_effect` — "do the treatment effects
        themselves statistically differ ACROSS segments?" This is
        answered by a real interaction test (see
        `app/stats/segmentation.py::_test_effect_heterogeneity`), never
        derived from `reliable_segment_values`. A metric can be
        significant in every single segment (`has_reliable_segment_effect
        =True` everywhere) while the effect sizes are statistically
        indistinguishable across segments (`has_heterogeneous_effect
        =False`) — and vice versa: two segments can each be individually
        non-significant (underpowered) while their point estimates
        still differ significantly from each other is NOT something
        this module claims either — see `heterogeneity_test_method`'s
        docstring for when the interaction test itself could not be run.

    `heterogeneity_test_method`/`heterogeneity_p_value` are optional,
    additive fields (never previously on this schema) that expose HOW
    `has_heterogeneous_effect` was computed, for transparency in the
    report/audit layers — `None` whenever fewer than 2 segments had
    sufficient sample to test for an interaction at all (heterogeneity
    is then reported as `False`, meaning "not detected", which is
    distinct from — but deliberately not further distinguished from —
    "not assessed"; the human-readable `heterogeneity_test_method`
    string always states which case occurred).
    """

    dimension: str
    segment_effects: list[SegmentEffect]
    multiple_testing_method: str
    reliable_segment_values: list[str]
    has_reliable_segment_effect: bool
    has_heterogeneous_effect: bool
    heterogeneity_test_method: str | None = None
    heterogeneity_p_value: float | None = None


class SegmentationResult(CamelModel):
    """
    Top-level segmentation output, always returned (never None) by
    `run_segmentation_analysis`. `ran=False` with a `reason` when no
    usable dimension exists, so the report layer can always say
    something truthful about segmentation instead of omitting it.
    """

    ran: bool
    reason: str
    usable_dimensions: list[str]
    skipped_dimensions: list[SkippedDimension]
    dimension_results: list[SegmentDimensionResult]
    min_segment_size: int
