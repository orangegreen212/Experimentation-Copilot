"""
Segmentation — Phase 5.

PURE, DETERMINISTIC, NO LLM — mirrors every other module in this
package. Answers one question: "does the experiment result differ
meaningfully across relevant user segments?", using the SAME
test-selection/execution machinery as the primary analysis
(`app/stats/hypothesis_tests.py`) — there is no separate, looser
statistical engine for segments.

ARCHITECTURAL PRINCIPLE (per project decision): Data -> segmentation/
statistics -> structured facts -> decision support -> LLM explanation.
This module only ever produces `SegmentationResult` facts. It never
overrides SRM/experiment-validity/guardrail gating — the caller
(experiment_node) runs this only as additional evidence alongside,
never instead of, the existing pipeline.

GUARDRAILS (all enforced here, not left to the LLM or the caller):

  - minimum segment size: a segment value's control/variant arm must
    each have >= `MIN_SEGMENT_ARM_SIZE` observations, or that segment
    value is marked INSUFFICIENT and no test is run on it at all.
  - cardinality: a candidate column with more than `MAX_CARDINALITY`
    distinct values is skipped entirely — it's not a meaningful
    segmentation dimension (e.g. a near-unique free-text field), and
    running dozens of segment tests on it would just be multiple-
    comparisons fishing.
  - too few distinct values: a column with < 2 distinct values (after
    dropna) carries no segmentation information and is skipped.
  - missing values: a column that is missing for more than
    `MAX_MISSING_FRACTION` of rows is skipped — segmenting mostly-null
    data produces misleading small-n comparisons.
  - identifier/metric columns are never candidates (user_col,
    variant_col, metric_col, or any column matching known event/
    timestamp/CUPED-covariate vocabularies from dataset_classifier).
  - multiple comparisons: WITHIN a dimension, per-segment p-values are
    Holm-corrected (reusing `_holm_adjust`, the same correction already
    used for multi-arm pairwise comparisons) before a segment is
    called "reliable" — descriptive segment_effects are always
    returned, but `reliable_segment_values` only lists segments that
    survive correction.
  - no blind fishing across arbitrary columns: only columns that pass
    every guardrail above are ever tested; every other column is
    recorded in `skipped_dimensions` with a reason, never silently
    dropped.
"""

from __future__ import annotations

import pandas as pd

from app.core.config import stats_thresholds
from app.schemas.segmentation import (
    SegmentDimensionResult,
    SegmentEffect,
    SegmentSampleSizeStatus,
    SegmentSkipReason,
    SegmentationResult,
    SkippedDimension,
)
from app.schemas.statistics import MetricType
from app.stats.dataset_classifier import (
    _CUPED_COVARIATE_CANDIDATES,
    _EVENT_COLUMN_CANDIDATES,
    _TIMESTAMP_COLUMN_CANDIDATES,
    classify_column_treatment_timing,
    humanize_metric_label,
)
from app.stats.hypothesis_tests import _holm_adjust, compute_stat_result, select_test

# Per-arm minimum observations within a single segment value before a
# hypothesis test is trusted at all. Matches the large-sample threshold
# already used elsewhere in this package (`_LARGE_SAMPLE_THRESHOLD`) —
# below this, small-sample noise dominates and a segment "finding"
# would be more likely to mislead than inform.
MIN_SEGMENT_ARM_SIZE = 30

# A column with more distinct values than this is not treated as a
# categorical segmentation dimension — either it's a near-identifier
# (e.g. free-text, an ID-like field) or splitting into this many
# segments would mean dozens of near-empty comparisons.
MAX_CARDINALITY = 10

# A column missing for more than this fraction of rows is skipped —
# segmenting mostly-null data produces misleading small-n comparisons.
MAX_MISSING_FRACTION = 0.30

_NON_CANDIDATE_VOCAB = set(
    c.lower() for c in _EVENT_COLUMN_CANDIDATES + _TIMESTAMP_COLUMN_CANDIDATES + _CUPED_COVARIATE_CANDIDATES
)


def discover_segmentation_dimensions(
    df: pd.DataFrame,
    user_col: str,
    variant_col: str,
    metric_col: str,
) -> tuple[list[str], list[SkippedDimension]]:
    """
    Identify usable categorical segmentation dimensions and explain,
    per-column, why every other column was excluded. Deterministic —
    no guessing beyond the fixed rules documented in the module
    docstring.
    """
    reserved = {user_col, variant_col, metric_col}
    usable: list[str] = []
    skipped: list[SkippedDimension] = []

    for col in df.columns:
        if col in reserved:
            continue
        if col.lower() in _NON_CANDIDATE_VOCAB:
            skipped.append(
                SkippedDimension(
                    column=col,
                    reason=SegmentSkipReason.IS_IDENTIFIER_OR_METRIC_COLUMN,
                    detail=f"'{col}' is a known event/timestamp/covariate column, not a segmentation dimension.",
                )
            )
            continue

        # Post-treatment leakage guard: an outcome measured AFTER
        # randomization (e.g. 'visit', 'spend', 'click') must never be
        # a segmentation dimension for treatment-effect analysis, even
        # when it isn't the chosen primary metric_col — conditioning on
        # a descendant of the treatment itself can fabricate or mask a
        # real effect. Checked BEFORE the cardinality/missing-value
        # checks below so a post-treatment column is always excluded
        # for the true reason, never coincidentally for an unrelated one.
        if classify_column_treatment_timing(col) == "post_treatment":
            skipped.append(
                SkippedDimension(
                    column=col,
                    reason=SegmentSkipReason.POST_TREATMENT_VARIABLE,
                    detail=(
                        f"'{col}' is an outcome measured after treatment assignment. "
                        "Segmenting on a post-treatment variable can fabricate or mask "
                        "a treatment effect, so it is excluded as a segmentation dimension."
                    ),
                )
            )
            continue

        series = df[col]
        missing_fraction = float(series.isna().mean()) if len(series) else 1.0
        if missing_fraction > MAX_MISSING_FRACTION:
            skipped.append(
                SkippedDimension(
                    column=col,
                    reason=SegmentSkipReason.EXCESSIVE_MISSING_VALUES,
                    detail=f"{missing_fraction:.0%} of values are missing (max allowed {MAX_MISSING_FRACTION:.0%}).",
                )
            )
            continue

        # Numeric columns with many distinct values (e.g. a raw score)
        # aren't categorical segments; numeric columns with few distinct
        # values (e.g. a 1-5 rating) are still usable as categories.
        n_distinct = int(series.dropna().nunique())
        if n_distinct < 2:
            skipped.append(
                SkippedDimension(
                    column=col,
                    reason=SegmentSkipReason.TOO_FEW_DISTINCT_VALUES,
                    detail=f"Only {n_distinct} distinct non-null value(s) — no segmentation information.",
                )
            )
            continue
        if n_distinct > MAX_CARDINALITY:
            skipped.append(
                SkippedDimension(
                    column=col,
                    reason=SegmentSkipReason.HIGH_CARDINALITY,
                    detail=f"{n_distinct} distinct values exceeds the cardinality limit ({MAX_CARDINALITY}) for a segmentation dimension.",
                )
            )
            continue

        if pd.api.types.is_numeric_dtype(series) and n_distinct > MAX_CARDINALITY:
            # unreachable given the check above, kept for clarity of intent
            continue

        usable.append(col)

    return usable, skipped


def _segment_effect_for_value(
    df: pd.DataFrame,
    dimension: str,
    value,
    variant_col: str,
    control_label,
    metric_col: str,
    metric_type: MetricType,
    metric_label: str,
) -> SegmentEffect:
    segment_df = df[df[dimension] == value]
    control_mask = segment_df[variant_col] == control_label
    variant_mask = ~control_mask

    control_series = segment_df.loc[control_mask, metric_col].dropna()
    variant_series = segment_df.loc[variant_mask, metric_col].dropna()
    control_n = int(len(control_series))
    variant_n = int(len(variant_series))

    if control_n < MIN_SEGMENT_ARM_SIZE or variant_n < MIN_SEGMENT_ARM_SIZE:
        return SegmentEffect(
            segment_value=str(value),
            control_n=control_n,
            variant_n=variant_n,
            sample_size_status=SegmentSampleSizeStatus.INSUFFICIENT,
            stat_result=None,
            skip_detail=(
                f"Segment '{value}' has control n={control_n}, variant n={variant_n} — "
                f"below the minimum of {MIN_SEGMENT_ARM_SIZE} per arm required to test this "
                f"segment reliably. No test was run; treat this segment as inconclusive, "
                f"not as 'no effect'."
            ),
        )

    test_selection = select_test(control_series, variant_series, metric_type)
    stat_result = compute_stat_result(
        control_series, variant_series, metric_type, metric_label, test_selection
    )
    # Display-only safeguard: compute_stat_result formats an
    # undefined relative delta (zero control baseline) as "+inf%"/
    # "-inf%" in `delta`, while `observed_relative_effect` is already
    # None in that exact case — that's the shared, unchanged signal
    # from the underlying statistical logic. Here in segmentation we
    # never surface a raw "inf%" to the report; we just relabel it as
    # a safe "N/A" for display. No statistic, p-value, or decision
    # logic is touched — only this string.
    if stat_result.observed_relative_effect is None and "inf%" in stat_result.delta:
        stat_result = stat_result.model_copy(update={"delta": "N/A (zero baseline)"})
    return SegmentEffect(
        segment_value=str(value),
        control_n=control_n,
        variant_n=variant_n,
        sample_size_status=SegmentSampleSizeStatus.SUFFICIENT,
        stat_result=stat_result,
    )


_NOT_ENOUGH_SEGMENTS_FOR_INTERACTION = (
    "N/A — fewer than 2 segments had sufficient sample size to test for an interaction "
    "(heterogeneity across segments is separate from within-segment significance; see "
    "SegmentDimensionResult's docstring)"
)


def _test_effect_heterogeneity(
    df: pd.DataFrame,
    dimension: str,
    values: list,
    tested_segment_values: list[str],
    variant_col: str,
    metric_col: str,
    metric_type: MetricType,
) -> tuple[bool, str, float | None]:
    """
    Does the TREATMENT EFFECT ITSELF statistically differ across
    segments? This is a genuinely different question from "is the
    effect significant within at least one segment" (that's
    `reliable_segment_values`/`has_reliable_segment_effect`, computed
    separately from Holm-corrected per-segment p-values above) — see
    the architectural bug this replaces: `has_heterogeneous_effect`
    used to be `len(reliable) > 0`, which is simply wrong (within-
    segment significance says nothing about whether segments differ
    from EACH OTHER).

    Answered here with a real interaction test: fit
    `metric ~ variant + segment` (no interaction) and
    `metric ~ variant + segment + variant:segment` (with interaction)
    on the SAME rows, and test whether the interaction term(s)
    significantly improve the fit — a joint test across however many
    segments are compared, so it stays correct for >2 segments, not
    just pairwise:

      - BINARY metric: logistic regression, likelihood-ratio test
        (chi-square) comparing the two nested models. This is the
        standard interaction test for a binary/proportion outcome
        across categorical strata (a joint generalization of the
        classic Breslow-Day homogeneity-of-odds-ratios test) and
        matches the primary metric type this project's guardrail
        dataset actually uses.
      - CONTINUOUS metric (monetary or general): OLS, nested F-test
        (`compare_f_test`) comparing the two models — the standard
        interaction test for a continuous outcome across strata
        (equivalent to the interaction term of a two-way ANOVA).

    Only ever run on segments that were themselves individually
    testable (`tested_segment_values` — i.e. NOT the ones marked
    INSUFFICIENT for per-arm sample size). An insufficient segment is
    excluded from this comparison entirely, never silently treated as
    "no difference" or folded into evidence for heterogeneity — see
    module requirement: "insufficient sample in a segment must not be
    used to claim heterogeneity."

    Returns `(has_heterogeneous_effect, method_description, p_value)`.
    `p_value` is `None` exactly when fewer than 2 segments could be
    compared, or when model fitting failed (e.g. perfect separation,
    singular design after filtering) — in both cases
    `has_heterogeneous_effect` is `False` ("not detected", not
    fabricated as True), and `method_description` states plainly why.
    """
    if len(tested_segment_values) < 2:
        return False, _NOT_ENOUGH_SEGMENTS_FOR_INTERACTION, None

    value_by_str = {str(v): v for v in values}
    comparable_raw_values = [value_by_str[s] for s in tested_segment_values if s in value_by_str]
    subset = df[df[dimension].isin(comparable_raw_values)][[variant_col, dimension, metric_col]].dropna(
        subset=[metric_col]
    )

    work = pd.DataFrame(
        {
            "y": pd.to_numeric(subset[metric_col], errors="coerce"),
            "variant": subset[variant_col].astype(str),
            "segment": subset[dimension].astype(str),
        }
    ).dropna(subset=["y"])

    alpha = stats_thresholds.significance_alpha
    n_segments = len(comparable_raw_values)

    try:
        import statsmodels.formula.api as smf
        from scipy import stats as scipy_stats

        if metric_type == MetricType.BINARY:
            full = smf.logit("y ~ C(variant) * C(segment)", data=work).fit(disp=0)
            reduced = smf.logit("y ~ C(variant) + C(segment)", data=work).fit(disp=0)
            df_diff = full.df_model - reduced.df_model
            if df_diff <= 0:
                return False, _NOT_ENOUGH_SEGMENTS_FOR_INTERACTION, None
            lr_stat = 2 * (full.llf - reduced.llf)
            p_value = float(scipy_stats.chi2.sf(lr_stat, df_diff))
            method = (
                f"Logistic regression interaction test (likelihood-ratio, df={int(df_diff)}) "
                f"across {n_segments} segments — tests whether the treatment effect on "
                f"{humanize_metric_label(metric_col)} differs across segments, not merely "
                f"whether it is significant within any one of them."
            )
        else:
            full = smf.ols("y ~ C(variant) * C(segment)", data=work).fit()
            reduced = smf.ols("y ~ C(variant) + C(segment)", data=work).fit()
            _f_stat, p_value, df_diff = full.compare_f_test(reduced)
            p_value = float(p_value)
            if df_diff <= 0:
                return False, _NOT_ENOUGH_SEGMENTS_FOR_INTERACTION, None
            method = (
                f"OLS interaction F-test (df={int(df_diff)}) across {n_segments} segments — "
                f"tests whether the treatment effect on {humanize_metric_label(metric_col)} "
                f"differs across segments, not merely whether it is significant within any "
                f"one of them."
            )
    except Exception as exc:  # noqa: BLE001 — defensive: separation/singular-design/convergence
        return (
            False,
            (
                f"Interaction test could not be computed ({type(exc).__name__}) — likely "
                f"perfect separation or a singular design after restricting to the "
                f"{n_segments} segment(s) with sufficient sample. Heterogeneity across "
                f"segments is NOT ASSESSED, not ruled out."
            ),
            None,
        )

    return bool(p_value < alpha), method, p_value


def analyze_dimension(
    df: pd.DataFrame,
    dimension: str,
    variant_col: str,
    control_label,
    metric_col: str,
    metric_type: MetricType,
    metric_label: str,
) -> SegmentDimensionResult:
    """
    Full within-dimension analysis for one already-validated usable
    dimension: per-segment-value comparisons, Holm-corrected across
    the segments that actually had a test run (segments skipped for
    insufficient size are excluded from the correction — correcting
    across tests that were never run would only dilute power further
    for no honesty benefit).
    """
    values = [v for v in df[dimension].dropna().unique().tolist()]
    effects = [
        _segment_effect_for_value(
            df, dimension, v, variant_col, control_label, metric_col, metric_type, metric_label
        )
        for v in values
    ]

    tested = [e for e in effects if e.stat_result is not None]
    reliable: list[str] = []
    method = "N/A — fewer than 2 segments had sufficient sample size to compare"

    if len(tested) >= 2:
        raw_p_values = [e.stat_result.p_value for e in tested]
        adjusted = _holm_adjust(raw_p_values)
        method = f"Holm-Bonferroni across {len(tested)} segment-level tests"
        updated_tested = []
        for effect, adj_p in zip(tested, adjusted):
            is_reliable = bool(adj_p < 0.05)
            updated_result = effect.stat_result.model_copy(
                update={
                    "adjusted_p_value": float(adj_p),
                    "significant": is_reliable,
                    "multiple_testing_method": "Holm-Bonferroni",
                }
            )
            updated_tested.append(effect.model_copy(update={"stat_result": updated_result}))
            if is_reliable:
                reliable.append(effect.segment_value)
        tested_by_value = {e.segment_value: e for e in updated_tested}
        effects = [tested_by_value.get(e.segment_value, e) for e in effects]
    elif len(tested) == 1:
        # A single testable segment can't be compared against anything
        # else within its own dimension — report it descriptively only,
        # never call a lone result "reliable" via correction of one.
        method = "N/A — only one segment had sufficient sample size; no multiple-comparisons correction to apply"

    # Phase 2 fix: heterogeneity is a SEPARATE, independently-computed
    # fact from `reliable`/Holm-corrected within-segment significance
    # above — never derived from it. See `_test_effect_heterogeneity`'s
    # docstring for the actual interaction test and why within-segment
    # significance cannot substitute for it. Only segments that were
    # themselves individually testable are ever compared here —
    # segments marked INSUFFICIENT never contribute "evidence" for
    # heterogeneity.
    heterogeneous_effect, heterogeneity_method, heterogeneity_p_value = _test_effect_heterogeneity(
        df, dimension, values, [e.segment_value for e in tested], variant_col, metric_col, metric_type
    )

    return SegmentDimensionResult(
        dimension=dimension,
        segment_effects=effects,
        multiple_testing_method=method,
        reliable_segment_values=reliable,
        has_reliable_segment_effect=len(reliable) > 0,
        has_heterogeneous_effect=heterogeneous_effect,
        heterogeneity_test_method=heterogeneity_method,
        heterogeneity_p_value=heterogeneity_p_value,
    )


def run_segmentation_analysis(
    df: pd.DataFrame,
    user_col: str,
    variant_col: str,
    control_label,
    metric_col: str,
    metric_type: MetricType,
) -> SegmentationResult:
    """
    Top-level entry point, called by experiment_node once per analysis
    run. Always returns a `SegmentationResult` — `ran=False` with a
    clear `reason` when no usable dimension exists, rather than
    raising or returning None, so the report can always say something
    truthful about segmentation.
    """
    metric_label = humanize_metric_label(metric_col)
    usable, skipped = discover_segmentation_dimensions(df, user_col, variant_col, metric_col)

    if not usable:
        return SegmentationResult(
            ran=False,
            reason=(
                "No usable segmentation dimensions were found in this dataset "
                f"({len(skipped)} candidate column(s) considered and excluded — "
                "see skipped_dimensions for why)."
            ),
            usable_dimensions=[],
            skipped_dimensions=skipped,
            dimension_results=[],
            min_segment_size=MIN_SEGMENT_ARM_SIZE,
        )

    dimension_results = [
        analyze_dimension(df, dim, variant_col, control_label, metric_col, metric_type, metric_label)
        for dim in usable
    ]

    # Phase 2 fix: `reliable_segment_effect` (within-segment significance)
    # and `heterogeneous` (across-segment interaction test) are reported
    # as the two separate facts they are — this summary previously used
    # `has_heterogeneous_effect` (then itself just `len(reliable) > 0`)
    # to describe "reliable segment-level differences", which conflated
    # the two concepts at the very point a human reads this. Both are
    # independently computed in `analyze_dimension`/
    # `_test_effect_heterogeneity`; this text only narrates them.
    reliable_dims = [d.dimension for d in dimension_results if d.has_reliable_segment_effect]
    heterogeneous_dims = [d.dimension for d in dimension_results if d.has_heterogeneous_effect]

    reason_parts = [f"Segmentation ran across {len(usable)} dimension(s) ({', '.join(usable)})."]
    if reliable_dims:
        reason_parts.append(
            "Statistically reliable within-segment effects (after multiple-comparisons "
            f"correction) were found in: {', '.join(reliable_dims)}."
        )
    else:
        reason_parts.append("No segment showed a statistically reliable effect after multiple-comparisons correction.")
    if heterogeneous_dims:
        reason_parts.append(
            f"A statistically significant interaction (the treatment effect itself differs "
            f"across segments) was detected in: {', '.join(heterogeneous_dims)}."
        )
    else:
        reason_parts.append(
            "No dimension showed a statistically significant interaction — the treatment "
            "effect's magnitude is not shown to differ across segments (this is a separate "
            "question from whether the effect is significant within any one segment)."
        )
    reason = " ".join(reason_parts)

    return SegmentationResult(
        ran=True,
        reason=reason,
        usable_dimensions=usable,
        skipped_dimensions=skipped,
        dimension_results=dimension_results,
        min_segment_size=MIN_SEGMENT_ARM_SIZE,
    )
