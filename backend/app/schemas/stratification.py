"""
Stratified-analysis schemas.

TRUE stratification (not to be confused with segmentation — see
`app/schemas/segmentation.py`, which is exploratory/descriptive):

  - a baseline categorical variable divides observations into strata
  - the treatment effect is estimated WITHIN each stratum
  - stratum-level estimates are COMBINED into a single overall
    treatment effect using a pre-specified weighting scheme (never a
    naive average of stratum rates/means)
  - the goal is precision / accounting for baseline heterogeneity, not
    hunting for where the effect differs (that's segmentation)

`StratificationEligibility` is ALWAYS produced whenever a
stratification column is requested — even when the variable turns out
to be unusable — so the report can state plainly WHY, exactly like
`SegmentationResult.reason` always being populated. `StratifiedEstimate`
is None whenever `eligibility.eligible` is False: an ineligible
variable never gets a fabricated/misleading combined estimate.

None of this is allowed to override SRM / validity gates. Stratified
inference only ever runs downstream of the SAME validity gate the
ordinary hypothesis test already goes through (see
`app/graph/graph_builder.py::route_after_validation` and
`app/graph/nodes/experiment_node.py`) — this module has no gating
logic of its own for SRM/conflicting-assignment, only for the
stratification variable itself.
"""

from __future__ import annotations

from enum import Enum

from app.schemas.base import CamelModel


class StratificationIneligibilityReason(str, Enum):
    """Closed set of reasons a requested stratification variable cannot be used."""

    COLUMN_NOT_FOUND = "column_not_found"
    IS_TREATMENT_ASSIGNMENT = "is_treatment_assignment"
    PERFECTLY_ASSOCIATED_WITH_ASSIGNMENT = "perfectly_associated_with_assignment"
    NO_STRATA_WITH_BOTH_VARIANTS = "no_strata_with_both_variants"
    MULTI_ARM_NOT_SUPPORTED = "multi_arm_not_supported"
    ALL_STRATA_TOO_SPARSE = "all_strata_too_sparse"


class StratumSummary(CamelModel):
    """
    Descriptive summary of one stratum value — reported regardless of
    whether the overall variable ends up eligible, so the report can
    always show "N observations, X% of total" for transparency (per
    the eligibility-checks requirement to report count/proportion per
    stratum even when inference doesn't run).
    """

    stratum_value: str
    control_n: int
    variant_n: int
    total_n: int
    proportion_of_total: float
    has_both_variants: bool
    sufficient: bool  # both arms meet the minimum-per-arm threshold
    control_outcome_rate: float | None = None  # descriptive only — mean/rate within stratum
    variant_outcome_rate: float | None = None


class StratificationEligibility(CamelModel):
    """
    Whether the requested stratification variable can be used for TRUE
    stratified inference on THIS dataset, and why. Always populated —
    `reason` is a plain-English explanation regardless of outcome.
    """

    stratification_column: str
    eligible: bool
    reason: str
    ineligibility_reason: StratificationIneligibilityReason | None = None
    strata: list[StratumSummary] = []
    total_n: int = 0
    missing_count: int = 0
    missing_proportion: float = 0.0
    min_arm_size: int = 0
    sparse_stratum_values: list[str] = []


class StratifiedEstimate(CamelModel):
    """
    The combined (across-strata) treatment-effect estimate — never a
    naive average of per-stratum rates/means. Produced by an
    inverse-variance-weighted fixed-effect combination of each
    stratum's own control-vs-variant effect (see
    `app/stats/stratification.py::run_stratified_analysis`).
    """

    method: str
    metric_label: str
    strata_used: int
    effect_estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool


class StratumSRMConcentration(CamelModel):
    """
    One stratum's OWN allocation balance — a diagnostic (not causal)
    chi-square check of whether THAT stratum's control/variant split
    matches the expected ratio, exactly like the overall SRM check but
    scoped to a single stratum. Lets the report show whether the
    overall SRM failure is concentrated in a particular stratum value
    (e.g. one landing page/browser/region) rather than spread evenly.
    This p-value is never a treatment-effect p-value and never feeds
    a ship/no-ship decision — it describes allocation balance only.
    """

    stratum_value: str
    observed_control: int
    observed_variant: int
    p_value: float
    srm_flagged: bool  # True if this stratum's own allocation also fails the SRM check


class DiagnosticStratification(CamelModel):
    """
    Descriptive/exploratory-only breakdown, produced when TRUE causal
    stratified inference is BLOCKED by an overall experiment validity
    failure (currently: SRM). Every field here is a plain descriptive
    fact — counts, proportions, rates, allocation splits — computed
    the same simple way regardless of experiment validity. NOTHING in
    this object is, or is derived from, a causal treatment-effect
    estimate: there is no combined effect, standard error, confidence
    interval, or p-value for a treatment effect anywhere here (contrast
    with `StratifiedEstimate`, which is exactly that and must never be
    populated alongside this object — see `StratificationResult`).
    Purpose: help the analyst investigate the SOURCE and STRUCTURE of
    the allocation problem (e.g. "is the SRM concentrated in one
    landing page?"), not answer "did the treatment work?".
    """

    stratification_column: str
    label: str = "Descriptive / Diagnostic only — not causal inference."
    allocation_by_variant: dict[str, int]
    total_n: int
    missing_count: int
    missing_proportion: float
    strata: list[StratumSummary]
    srm_by_stratum: list[StratumSRMConcentration] = []


class StratificationStatus(str, Enum):
    """
    Whether — and how — stratified analysis executed.

      - RAN: experiment_node ran the full causal stratified estimate
        (see `run_stratified_analysis`). `eligibility`/`estimate` are
        populated (`estimate` is None only if the variable itself
        turned out ineligible).
      - NOT_RUN: experiment_node never ran AT ALL because the
        experiment failed a validity gate (conflicting variant
        assignment / critical quality failure) OR stratification
        wasn't reached for some other non-SRM reason — see
        route_after_validation. `eligibility`/`estimate`/`diagnostic`
        are all None (never fabricated) and `not_run_reason` explains
        why.
      - DIAGNOSTIC: the experiment failed the SRM validity gate
        specifically, so CAUSAL stratified inference is BLOCKED (same
        as NOT_RUN would report), but a non-causal, purely descriptive
        breakdown was still computed and is available in `diagnostic`
        — see `DiagnosticStratification`. `eligibility`/`estimate`
        remain None in this state; only `diagnostic` is populated,
        and `not_run_reason` still explains why causal inference is
        blocked.
    """

    RAN = "ran"
    NOT_RUN = "not_run"
    DIAGNOSTIC = "diagnostic"


class StratificationResult(CamelModel):
    """
    Top-level fact attached to the report under its OWN "Stratified
    Analysis" section — deliberately separate from `SegmentationResult`
    (see module docstring). `eligibility`/`estimate` are None whenever
    `status != RAN`. `estimate` is also None whenever
    `eligibility.eligible` is False even though `status == RAN`.
    `diagnostic` is populated ONLY when `status == DIAGNOSTIC` — a
    non-causal, descriptive-only breakdown; it is never a substitute
    for, and never contains, a causal treatment-effect estimate.
    """

    status: StratificationStatus
    stratification_column: str
    eligibility: StratificationEligibility | None = None
    estimate: StratifiedEstimate | None = None
    # Populated ONLY when status == DIAGNOSTIC — see
    # DiagnosticStratification's docstring. Never populated alongside
    # `estimate`.
    diagnostic: DiagnosticStratification | None = None
    # Populated when status == NOT_RUN or status == DIAGNOSTIC —
    # explains which validity gate blocked (causal) stratification
    # from running, e.g. "Experiment validity failed because 1,895
    # users have conflicting variant assignments." or "Causal
    # stratified inference is BLOCKED because the Sample Ratio
    # Mismatch check failed (p=0.0003)."
    not_run_reason: str | None = None
