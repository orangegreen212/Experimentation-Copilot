"""
Data quality checks — Stage 3: nulls, outliers, normality.

Each check is a pure function on pandas Series/arrays, returning either
a frontend-facing `QualityCheck` (nulls, outliers — simple pass/fail +
detail string, matching mock-data.ts's style) or a richer internal
schema (`NormalityCheckResult` — needed by test_selection_node in
Stage 4 to decide Welch vs Student vs Mann-Whitney, not just to show
a checkmark).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.config import stats_thresholds
from app.schemas.quality import QualityCheck
from app.schemas.statistics import MetricType, NormalityCheckResult



def check_nulls(df: pd.DataFrame, columns: list[str], threshold_pct: float | None = None) -> QualityCheck:
    """
    Fraction of missing values across the given columns. Fails if the
    WORST column's missing fraction exceeds `threshold_pct` (default
    from config, 1%).

    Matches mock-data.ts's two report styles:
      passed: "0% missing across all metric columns"
      failed: "3.2% missing in variant arm — exceeds 1% threshold"
    """
    threshold_pct = stats_thresholds.null_threshold_pct if threshold_pct is None else threshold_pct

    if not columns:
        raise ValueError("check_nulls requires at least one column to check.")

    missing_fractions = {col: df[col].isna().mean() for col in columns}
    worst_col = max(missing_fractions, key=missing_fractions.get)
    worst_pct = missing_fractions[worst_col]

    passed = worst_pct <= threshold_pct

    if worst_pct == 0:
        detail = f"0% missing across all metric columns"
    elif passed:
        detail = f"{worst_pct * 100:.1f}% missing in '{worst_col}' — within {threshold_pct * 100:.0f}% threshold"
    else:
        detail = f"{worst_pct * 100:.1f}% missing in '{worst_col}' — exceeds {threshold_pct * 100:.0f}% threshold"

    return QualityCheck(label="Null / Missing Values", passed=passed, detail=detail)


def check_outliers(series: pd.Series, metric_type: MetricType, iqr_multiplier: float = 1.5) -> QualityCheck:
    """
    Outlier detection strategy depends on metric type (Stage 3 decision):

      - BINARY: skipped entirely — a 0/1 conversion flag has no
        meaningful notion of an "outlier".
      - CONTINUOUS_MONETARY: zeros are excluded before detection (they
        represent "did not convert," not a true revenue observation —
        see ZEROS EXCLUDED SEMANTICS below for what this does and does
        NOT affect), then a LOG-IQR method is used on the remaining
        positive values: the Tukey fences are computed in log-space
        (i.e. IQR of log(values), bounds translated back), not on the
        raw linear values. See METHOD SELECTION below for why.
      - CONTINUOUS_GENERAL: plain linear IQR on all values, no
        exclusion, unchanged from the original behavior — this metric
        type has no guarantee of positivity/right-skew, so log-space
        is not applicable here.

    METHOD SELECTION (why monetary gets log-IQR, not linear IQR):
    real order-value/revenue data is right-skewed (roughly lognormal),
    and plain linear IQR is too aggressive on it — a handful of
    perfectly ordinary large orders routinely sit outside the 1.5x
    Tukey fence purely from right-skew, not from any data-quality
    problem (verified against this project's own
    data/demo/demo_ab_aov_cuped.csv: a real, non-corrupted e-commerce
    AOV dataset). Taking the IQR of log(values) instead (valid here
    since zeros were already excluded, guaranteeing strict positivity)
    normalizes that skew away first, so the fence reflects genuine
    spread rather than the shape of the distribution — while a
    genuinely corrupted value (e.g. a stray $1,000,000 row among ~$50
    orders) still clears it easily. This is a change of WHICH SPACE
    the existing IQR test runs in, not a change to the 1.5x multiplier
    itself — the multiplier is never touched by metric type.

    Symmetric continuous metrics (CONTINUOUS_GENERAL) are left on
    plain linear IQR exactly as before — this fix only changes the
    monetary branch.

    IMPORTANT — outliers are NEVER dropped from the statistical
    analysis here or anywhere downstream: this function only produces
    a `QualityCheck` (a warning/flag row in the report). The series
    passed to hypothesis testing and power analysis
    (experiment_node.py's `control_metric`/`variant_metric`) is
    entirely independent of this function and is never filtered by
    its result.

    ZEROS EXCLUDED SEMANTICS: "zeros excluded" here means ONLY that
    structural zeros (no-purchase rows) are left out of the pool this
    function runs outlier detection on — it does NOT shrink the
    experiment's total n, and does NOT touch the series used by the
    hypothesis test / power analysis (those run on the full,
    non-zero-excluded arm — see experiment_node.py). The returned
    `detail` states all three counts explicitly so this is auditable
    from the report alone: the arm's total n, how many of those were
    structural zeros (monetary only), and how many values outlier
    detection actually ran on.

    CRITICAL FLAG: an IQR-flagged outlier is additionally marked
    `critical=True` on the returned `QualityCheck` when at least one
    of the flagged outlier values lies beyond `stats_thresholds.
    outlier_sigma` (4.0 by default) standard deviations from the
    arm's OWN mean, computed in the SAME space (log, for monetary; 
    linear, for general) that the IQR fence itself was computed in —
    so this is now just "how many sigma is this flagged point," no
    separate transform needed. This is the documented rule from
    `StatsThresholds.outlier_sigma`'s docstring ("values beyond this
    many standard deviations from the arm's mean are flagged"), wired
    up here so a routine mild outlier (e.g. just outside the Tukey
    fence) doesn't halt the pipeline, while a genuinely extreme/
    corrupted value does — see `route_after_validation` in
    graph_builder.py for the resulting skip-to-Decision behavior, and
    `experiment_validity()`/`_assess_confidence()` in
    report_generator.py for how a critical failure caps confidence at
    LOW and marks the experiment INVALID.

    DEGENERATE-IQR FIX (Q1 == Q3): sparse/zero-inflated count metrics
    (e.g. `clicks`, where the vast majority of users have 0) routinely
    push both quartiles to the same value, collapsing the Tukey fence
    to a single point. The old behavior in that case flagged every
    value that merely differed from one arbitrary observation as an
    "outlier" — for a metric like this, that means EVERY positive
    count gets flagged, which can spuriously trigger the critical-
    outlier path above and mark an otherwise healthy experiment
    INVALID. When Q1 == Q3, this function now falls back to a MAD
    (median absolute deviation) based fence instead: modified z-score
    (Iglewicz & Hoaglin) > 3.5 from the median. This fence doesn't
    require the quartiles to be distinct, so it degrades gracefully
    exactly where the plain Tukey fence can't. If the MAD itself is
    also 0 (e.g. the modal value is the majority of the sample, as
    with `clicks` above), no value has a robust basis to be called an
    outlier and none are flagged. This fallback applies uniformly
    whenever Q1 == Q3 regardless of metric type (linear space for
    CONTINUOUS_GENERAL, log space for CONTINUOUS_MONETARY) — it only
    replaces the degenerate zero-width-fence path, and never changes
    the ordinary IQR/Log-IQR fence computation used whenever Q1 != Q3.
    """
    if metric_type == MetricType.BINARY:
        return QualityCheck(
            label="Outlier Detection",
            passed=True,
            detail="Skipped — binary metric (outlier detection not applicable)",
        )

    zeros_excluded = metric_type == MetricType.CONTINUOUS_MONETARY
    method_name = "Log-IQR" if zeros_excluded else "IQR"

    non_null = series.dropna()
    total_n = len(series)  # (a) total experiment sample size for this arm
    zeros_count = 0  # (b) structural-zero observations (monetary only)
    working = non_null
    if zeros_excluded:
        zeros_count = int((non_null == 0).sum())
        working = non_null[non_null > 0]
    # len(working) is (c): observations actually used for outlier detection.

    zero_note = f" ({zeros_count:,} zeros excluded of {total_n:,} total in arm)" if zeros_excluded else ""

    if len(working) < 4:
        # IQR is not meaningful below a handful of points.
        return QualityCheck(
            label="Outlier Detection",
            passed=True,
            detail=f"{method_name} method on {len(working)} values{zero_note} — insufficient data to assess",
        )

    # Space the Tukey fence (and, below, the critical-sigma check) is
    # computed in: log-space for monetary (guaranteed strictly
    # positive — zeros already excluded above), linear otherwise.
    transformed = np.log(working) if zeros_excluded else working

    q1, q3 = transformed.quantile(0.25), transformed.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    if iqr == 0:
        # Degenerate Tukey fence: Q1 == Q3, so the naive fence collapses
        # to a single point [Q1, Q1]. This happens routinely for
        # sparse/zero-inflated count metrics (e.g. `clicks`, where the
        # majority of users have 0) — the modal value dominates both
        # quartiles even though the metric has genuine, ordinary
        # spread among its non-modal values. Flagging "!= the first
        # value" (the old behavior) would misclassify EVERY positive
        # observation as an outlier purely because the fence has zero
        # width, not because those values are actually unusual.
        #
        # Fall back to a MAD-based fence (median absolute deviation,
        # modified z-score per Iglewicz & Hoaglin) instead: it doesn't
        # depend on the quartiles being distinct, so it stays
        # meaningful exactly when the plain IQR fence can't be. Values
        # are flagged only when they are genuinely far (modified
        # z-score > 3.5, the standard Iglewicz & Hoaglin threshold)
        # from the median — not merely different from one arbitrary
        # observation.
        median = transformed.median()
        mad = (transformed - median).abs().median()
        if mad == 0:
            # Even MAD is degenerate (e.g. the series is constant, or
            # the modal value makes up more than half the sample —
            # true for `clicks` in the motivating example, where
            # 84.6% of users have 0) — there is no robust basis left
            # to call anything an outlier.
            outlier_mask = pd.Series(False, index=transformed.index)
        else:
            modified_z = 0.6745 * (transformed - median) / mad
            outlier_mask = modified_z.abs() > 3.5
        degenerate_iqr_note = " (degenerate IQR fence [Q1=Q3] — used MAD fallback)"
    else:
        outlier_mask = (transformed < lower_bound) | (transformed > upper_bound)
        degenerate_iqr_note = ""
    n_outliers = int(outlier_mask.sum())

    passed = n_outliers == 0

    critical = False
    if n_outliers > 0:
        # Sigma computed on the arm's own mean/std (ddof=1, standard
        # sample std) over the same (already-transformed) values the
        # fence itself was computed on — NOT just over the flagged
        # outliers — so the threshold reflects the arm's actual spread
        # in whichever space is meaningful for this metric type.
        arm_mean = transformed.mean()
        arm_std = transformed.std(ddof=1)
        if arm_std and arm_std > 0:
            outlier_values = transformed[outlier_mask]
            max_abs_sigma = ((outlier_values - arm_mean).abs() / arm_std).max()
            critical = bool(max_abs_sigma > stats_thresholds.outlier_sigma)

    if passed:
        detail = f"{method_name} method on {len(working)} values{zero_note}{degenerate_iqr_note} — no outliers detected"
    else:
        detail = (
            f"{method_name} method on {len(working)} values{zero_note}{degenerate_iqr_note} "
            f"— {n_outliers} outlier(s) detected"
        )

    return QualityCheck(label="Outlier Detection", passed=passed, detail=detail, critical=critical)


def combine_outlier_checks(control_check: QualityCheck, variant_check: QualityCheck) -> QualityCheck:
    """
    Merge per-arm outlier checks into the single row the frontend
    displays.

    Both arms' details are always included, labeled, so the report never
    drops one arm's outlier-detection outcome — each arm's total n,
    structural-zero count, and n used for outlier detection must stay
    visible in the report regardless of whether that arm passed.
    """
    passed = control_check.passed and variant_check.passed
    critical = control_check.critical or variant_check.critical
    detail = f"control: {control_check.detail}; variant: {variant_check.detail}"
    return QualityCheck(label="Outlier Detection", passed=passed, detail=detail, critical=critical)


def check_normality(control: pd.Series, variant: pd.Series, alpha: float | None = None) -> NormalityCheckResult:
    """
    Shapiro-Wilk normality test on each arm independently.

    Matches mock-data.ts:
      passed: "Both arms p > 0.05 — normality assumption holds"

    Shapiro-Wilk requires n >= 3; for larger samples (n > 5000) it
    becomes overly sensitive to tiny deviations, so we subsample to
    5000 for the test statistic while still reporting on the full arm
    conceptually. This is a standard, well-documented mitigation, not
    an ad-hoc hack.
    """
    alpha = stats_thresholds.normality_alpha if alpha is None else alpha

    control_clean = control.dropna()
    variant_clean = variant.dropna()

    if len(control_clean) < 3 or len(variant_clean) < 3:
        raise ValueError("Shapiro-Wilk requires at least 3 non-null observations per arm.")

    control_sample = _subsample_for_shapiro(control_clean)
    variant_sample = _subsample_for_shapiro(variant_clean)

    control_stat, control_p = scipy_stats.shapiro(control_sample)
    variant_stat, variant_p = scipy_stats.shapiro(variant_sample)

    return NormalityCheckResult(
        control_statistic=float(control_stat),
        control_p_value=float(control_p),
        control_normal=bool(control_p >= alpha),
        variant_statistic=float(variant_stat),
        variant_p_value=float(variant_p),
        variant_normal=bool(variant_p >= alpha),
    )


def normality_result_to_quality_check(
    result: NormalityCheckResult, alpha: float | None = None, large_sample_rule_applied: bool = False
) -> QualityCheck:
    """
    Fold NormalityCheckResult into the frontend-facing QualityCheck row.

    `large_sample_rule_applied` must be True exactly when
    `hypothesis_tests.select_test()` used the n>=30-per-arm CLT branch
    (i.e. never actually relied on this normality result to pick the
    test). In that case a failure is marked `informational=True`: still
    shown, still explained, but excluded from the confidence/validity
    downgrades a genuine data-quality issue triggers — see
    `QualityCheck.informational`'s docstring for why. When the small-
    sample branch was used instead, this check is exactly what decided
    Welch vs Mann-Whitney, so a failure here is real, decision-relevant
    information and stays non-informational (unchanged behavior).
    """
    alpha = stats_thresholds.normality_alpha if alpha is None else alpha
    both_normal = result.control_normal and result.variant_normal

    if both_normal:
        detail = f"Both arms p > {alpha:g} — normality assumption holds"
    elif large_sample_rule_applied:
        detail = (
            f"control p={result.control_p_value:.3f}, "
            f"variant p={result.variant_p_value:.3f} "
            f"— normality assumption does not hold for both arms, but both arms are large "
            f"enough (n>=30) that Welch's t-test was already selected via the Central Limit "
            f"Theorem without relying on this result; shown for transparency only, does not "
            f"lower confidence"
        )
    else:
        detail = (
            f"control p={result.control_p_value:.3f}, "
            f"variant p={result.variant_p_value:.3f} "
            f"— normality assumption does not hold for both arms"
        )

    return QualityCheck(
        label="Normality (Shapiro-Wilk)",
        passed=both_normal,
        detail=detail,
        informational=large_sample_rule_applied,
    )


def _subsample_for_shapiro(series: pd.Series, max_n: int = 5000, seed: int = 42) -> np.ndarray:
    """Shapiro-Wilk is unreliable/overly sensitive above ~5000 samples; subsample deterministically."""
    values = series.to_numpy()
    if len(values) <= max_n:
        return values
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=max_n, replace=False)
