"""
Dataset classification schemas.

Mirrors `DatasetInfo` in lib/types.ts:

    export interface DatasetInfo {
      type: string;
      variants: number;
      users: number;
      metricLabel: string;
    }

`DatasetInfo` is the OUTPUT of the classifier (Stage 2). This file only
defines the contract — no classification logic lives here.
"""

from enum import Enum
from pydantic import Field

from app.schemas.base import CamelModel


class DatasetType(str, Enum):
    """
    Closed set of dataset shapes the classifier can detect.

    Kept as an enum (not a free string) even though the frontend types
    it as `string` — this gives us type safety on the backend while
    still serializing to a plain string the frontend already expects.
    """

    AGGREGATED_AB_TEST = "Aggregated A/B Test Data"
    RAW_EVENT_LEVEL = "Raw Event-Level Data"
    # A dataset with a genuine per-unit identifier column
    # (user_id/customer_id/visitor_id/...) can never be a pre-aggregated
    # summary table — aggregation, by definition, discards the
    # individual unit and leaves one row per arm/stratum, not one row
    # (or a few rows) per user. This type captures that shape
    # explicitly: raw, individual-level experiment data with a rows-per-
    # user ratio near 1 (one observation per user), distinct from
    # AGGREGATED_AB_TEST even when row count happens to be close to
    # user count.
    RAW_USER_LEVEL = "Raw User-Level Experiment Data"
    UNKNOWN = "Unknown / Unsupported Format"


class ExperimentUnitLevel(str, Enum):
    """
    The GRAIN of one row in the dataset, relative to the randomized
    experimental unit. Orthogonal to `DatasetType` above (which votes
    on raw/aggregated/unknown using the legacy two-heuristic scheme):
    this is the explicit, evidence-backed answer to "does one row
    equal one randomized unit, many rows, or a pre-aggregated group?"
    that CRM/marketing datasets need — most CRM exports have no
    explicit customer id at all, and DatasetType.UNKNOWN alone doesn't
    say whether that's a fatal problem (event-level, no id) or a
    perfectly usable dataset (unit-level, no id needed).

    See `app.stats.dataset_classifier.classify_experiment_unit_level`.
    """

    EVENT_LEVEL = "event_level"
    UNIT_LEVEL = "unit_level"
    AGGREGATE_LEVEL = "aggregate_level"
    UNKNOWN = "unknown"


class UnitIdentifierType(str, Enum):
    """How the experimental unit is identified in this dataset."""

    EXPLICIT_COLUMN = "explicit_column"
    IMPLICIT_ROW = "implicit_row"
    MISSING = "missing"


class RatioMetricCandidate(CamelModel):
    """
    A plausible ratio metric (numerator column / denominator column)
    detected by naming/shape heuristics — NOT a new statistical method.
    Downstream code decides which existing engine path applies (e.g. a
    'conversions/users' ratio is usually better analyzed via the
    existing binary/proportion path using the raw numerator+denominator
    counts, rather than treated as a single continuous observation per
    row — see app.stats.dataset_classifier.detect_ratio_metric_candidates
    for the exact heuristic). Naming/shape match only, same "_candidates"
    convention as stratification_candidates/guardrail_candidates below —
    this field never asserts eligibility, only that a plausible pair
    exists.
    """

    metric_name: str = Field(description="Human/business name for the ratio, e.g. 'conversion_rate', 'revenue_per_user'.")
    numerator: str = Field(description="Column name used as the ratio's numerator, e.g. 'conversions'.")
    denominator: str = Field(description="Column name used as the ratio's denominator, e.g. 'users'.")
    ratio_definition: str = Field(description="Human-readable definition, e.g. 'conversions / users'.")


class DatasetInfo(CamelModel):
    """Result of dataset classification — what the UI's green banner shows."""

    type: DatasetType
    variants: int
    users: int
    metric_label: str
    available_metrics: list[str] = Field(default_factory=list)
    metric_selection_reason: str = Field(
        description=(
            "Deterministic explanation of why metric_label was chosen as the "
            "primary metric over any other available metric — never assumed, "
            "never LLM-generated. See app.stats.dataset_classifier._select_metric_column."
        )
    )

    # --- Full classification visibility (additive; all optional so any
    # response/fixture produced before these fields existed still
    # validates unchanged) ---------------------------------------------
    #
    # Every value below comes exclusively from the deterministic
    # classifier in app.stats.dataset_classifier — never from the LLM —
    # same guarantee as the fields above. The last three are
    # deliberately named "*_candidates": the classifier is only
    # reporting a plausible naming/shape match, never asserting that a
    # column IS eligible for that statistical role. Actual eligibility
    # (e.g. stratification) is decided by a separate, dedicated gate —
    # see app.stats.stratification.check_stratification_eligibility —
    # which this field does not replace or preempt.

    user_id_column: str | None = Field(
        default=None,
        description="The column recognized as the experiment-unit identifier (e.g. 'user_id'). None if no such column was found.",
    )
    variant_column: str | None = Field(
        default=None,
        description="The column recognized as the variant/arm assignment. None if no such column was found.",
    )
    variant_values: list[str] = Field(
        default_factory=list,
        description="Distinct values observed in variant_column (e.g. ['control', 'treatment']). Empty if variant_column is None.",
    )
    primary_metric: str = Field(
        default="",
        description="Same value as metric_label, exposed under this name for the full Dataset Classification view.",
    )
    additional_metrics: list[str] = Field(
        default_factory=list,
        description="Other recognized numeric outcome columns besides the primary metric (available_metrics minus primary_metric).",
    )
    stratification_candidates: list[str] = Field(
        default_factory=list,
        description=(
            "Low-cardinality, non-structural columns that could plausibly serve as a "
            "stratification dimension (e.g. 'landing_page'). Naming/shape heuristic only — "
            "not a statement of stratification eligibility."
        ),
    )
    guardrail_candidates: list[str] = Field(
        default_factory=list,
        description=(
            "Numeric columns whose name matches common guardrail-metric vocabulary "
            "(latency, error_rate, churn, ...). Naming heuristic only — no guardrail "
            "statistical check has been run against these."
        ),
    )
    covariate_candidates: list[str] = Field(
        default_factory=list,
        description=(
            "Numeric columns whose name matches CUPED/pre-experiment covariate vocabulary "
            "(e.g. 'pre_experiment_metric'). Naming heuristic only."
        ),
    )

    # --- Experiment-unit-level classification (CRM/marketing datasets
    # without an explicit customer id) — additive, all with safe
    # defaults so any response/fixture produced before these fields
    # existed still validates unchanged. See
    # app.stats.dataset_classifier.classify_experiment_unit_level for
    # the deterministic logic that produces every value below. ---------

    experiment_unit_level: ExperimentUnitLevel = Field(
        default=ExperimentUnitLevel.UNKNOWN,
        description=(
            "The row grain relative to the randomized experimental unit: "
            "event_level (multiple rows can belong to one unit — an explicit "
            "identifier is required), unit_level (one row IS one unit — an "
            "explicit identifier is optional), aggregate_level (rows are "
            "group-level counts, not individual units), or unknown."
        ),
    )
    unit_identifier: UnitIdentifierType = Field(
        default=UnitIdentifierType.MISSING,
        description=(
            "How the experimental unit is identified: explicit_column (a real "
            "id column, e.g. user_id), implicit_row (no id column, but each "
            "row was determined to be one randomized unit), or missing."
        ),
    )
    unit_level_confidence: float = Field(
        default=0.0,
        description="Confidence (0-1) in experiment_unit_level/unit_identifier, from the multi-signal grain classifier.",
    )
    unit_level_evidence: list[str] = Field(
        default_factory=list,
        description="Human-readable evidence statements supporting experiment_unit_level, e.g. 'No event column was detected'.",
    )
    unit_level_blocking_reason: str | None = Field(
        default=None,
        description=(
            "Populated only when the grain is recognized but analysis is blocked "
            "(e.g. event-level data with no unit identifier) — explains what's "
            "missing, never a generic/unexplained failure."
        ),
    )
    pre_treatment_segmentation_candidates: list[str] = Field(
        default_factory=list,
        description=(
            "Columns classified as pre-treatment (measured before/independent of "
            "randomization, e.g. 'recency', 'channel') — safe to use as "
            "segmentation dimensions for treatment-effect analysis."
        ),
    )
    excluded_post_treatment_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Columns classified as post-treatment (measured after assignment, "
            "e.g. 'conversion', 'click', 'campaign_revenue') and therefore "
            "excluded from segmentation candidates to prevent post-treatment bias."
        ),
    )
    funnel_metrics: list[str] = Field(
        default_factory=list,
        description=(
            "CRM funnel-stage columns detected in this (unit-level) dataset, e.g. "
            "['sent', 'delivered', 'opened', 'clicked', 'converted'] — naming "
            "heuristic only, populated only when 2+ stage columns are present."
        ),
    )
    ratio_metric_candidates: list[RatioMetricCandidate] = Field(
        default_factory=list,
        description=(
            "Plausible ratio metrics (numerator/denominator column pairs) detected "
            "by naming heuristics, e.g. conversion_rate = conversions/users, "
            "revenue_per_user = revenue/users. Naming/shape match only — downstream "
            "code, not this field, decides whether the existing binary/proportion "
            "path or a continuous path is the statistically appropriate way to "
            "analyze a given ratio."
        ),
    )


class ClassifyDatasetRequest(CamelModel):
    """
    Request to classify an uploaded/demo dataset.

    NOTE: for CSV upload we accept multipart/form-data (UploadFile) in
    the route directly, not through this schema. This schema covers the
    "Load Demo" path and any future non-file classification trigger.
    """

    use_demo: bool = False
    simulate_low_quality: bool = False


class ClassifyDatasetResponse(CamelModel):
    """Response envelope for POST /datasets/classify."""

    dataset: DatasetInfo
    dataset_id: str  # server-side handle used later in /experiments/analyze
    file_name: str
