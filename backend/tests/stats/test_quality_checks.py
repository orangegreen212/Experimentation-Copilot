import numpy as np
import pandas as pd
import pytest

from app.schemas.statistics import MetricType
from app.stats.quality_checks import (
    check_normality,
    check_nulls,
    check_outliers,
    combine_outlier_checks,
    normality_result_to_quality_check,
)
from app.graph.report_generator import ReportFacts, experiment_validity
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.quality import QualityCheck
from app.schemas.report import ExperimentValidity


# --- nulls ---------------------------------------------------------------

def test_no_missing_values_passes():
    df = pd.DataFrame({"order_value": [1.0, 2.0, 3.0, 4.0]})
    qc = check_nulls(df, ["order_value"])
    assert qc.passed is True
    assert "0%" in qc.detail


def test_missing_above_threshold_fails():
    df = pd.DataFrame({"order_value": [1.0, np.nan, np.nan, 4.0, 5.0]})  # 40% missing
    qc = check_nulls(df, ["order_value"])
    assert qc.passed is False
    assert "exceeds" in qc.detail


def test_missing_at_low_pct_within_threshold_passes():
    # 1 missing out of 200 = 0.5%, under 1% default threshold
    values = [1.0] * 199 + [np.nan]
    df = pd.DataFrame({"order_value": values})
    qc = check_nulls(df, ["order_value"])
    assert qc.passed is True


def test_check_nulls_requires_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        check_nulls(df, [])


# --- outliers --------------------------------------------------------------

def test_binary_metric_skips_outlier_detection():
    series = pd.Series([0, 1, 0, 1, 1, 0])
    qc = check_outliers(series, metric_type=MetricType.BINARY)
    assert qc.passed is True
    assert "Skipped" in qc.detail


def test_no_outliers_passes_for_continuous_general():
    # NOTE: standard Tukey 1.5x-IQR fences flag ~0.7% of points even on
    # truly normal data at large N — that's expected IQR behavior, not
    # a bug. Use a small, tight, outlier-free sample instead of relying
    # on a large N staying at exactly zero flagged points.
    series = pd.Series([48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 50.8])
    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_GENERAL)
    assert qc.passed is True
    assert "IQR" in qc.detail


def test_extreme_outlier_detected_continuous_general():
    values = list(np.random.default_rng(0).normal(50, 5, size=200)) + [10000.0]
    series = pd.Series(values)
    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_GENERAL)
    assert qc.passed is False
    assert "outlier" in qc.detail


def test_insufficient_data_passes_gracefully():
    series = pd.Series([5.0])
    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_GENERAL)
    assert qc.passed is True


def test_monetary_metric_excludes_zeros():
    # Zeros represent non-converted users, not true revenue observations.
    positive_values = [48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 50.8]
    zeros = [0.0] * 700  # majority non-converted
    series = pd.Series(positive_values + zeros)
    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY)
    assert "zeros excluded" in qc.detail
    assert "10 values" in qc.detail
    assert qc.passed is True  # tight positive cluster, no real outliers


def test_monetary_metric_detects_real_outlier_among_positives():
    rng = np.random.default_rng(0)
    positive_values = list(rng.normal(50, 5, size=300)) + [5000.0]
    zeros = [0.0] * 700
    series = pd.Series(positive_values + zeros)
    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY)
    assert qc.passed is False
    assert "zeros excluded" in qc.detail


def test_combine_outlier_checks_both_pass():
    tight_a = pd.Series([48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 50.8])
    tight_b = pd.Series([47.5, 48.5, 49.5, 50.0, 50.5, 51.0, 51.5, 49.0, 50.1, 50.6])
    a = check_outliers(tight_a, metric_type=MetricType.CONTINUOUS_GENERAL)
    b = check_outliers(tight_b, metric_type=MetricType.CONTINUOUS_GENERAL)
    combined = combine_outlier_checks(a, b)
    assert combined.passed is True


def test_combine_outlier_checks_one_fails():
    good = check_outliers(
        pd.Series([48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 50.8]),
        metric_type=MetricType.CONTINUOUS_GENERAL,
    )
    bad_values = [48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 999999.0]
    bad = check_outliers(pd.Series(bad_values), metric_type=MetricType.CONTINUOUS_GENERAL)
    combined = combine_outlier_checks(good, bad)
    assert combined.passed is False
    assert "variant:" in combined.detail


# --- normality ---------------------------------------------------------------

def test_normal_data_holds():
    # NOTE: Shapiro-Wilk has an inherent ~5% false-positive rate even on
    # truly normal data — seed=7 verified stable for both arms; this
    # tests the check's logic, not statistical luck.
    rng = np.random.default_rng(7)
    control = pd.Series(rng.normal(50, 5, size=200))
    variant = pd.Series(rng.normal(51, 5, size=200))
    result = check_normality(control, variant)
    assert result.control_normal is True
    assert result.variant_normal is True

    qc = normality_result_to_quality_check(result)
    assert qc.passed is True
    assert "holds" in qc.detail


def test_skewed_data_fails_normality():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.exponential(scale=2.0, size=300))
    variant = pd.Series(rng.normal(50, 5, size=300))
    result = check_normality(control, variant)
    assert result.control_normal is False

    qc = normality_result_to_quality_check(result)
    assert qc.passed is False


def test_normality_requires_min_samples():
    with pytest.raises(ValueError):
        check_normality(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0, 3.0, 4.0]))


def test_normality_subsamples_large_input_without_error():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(50, 5, size=8000))
    variant = pd.Series(rng.normal(51, 5, size=8000))
    result = check_normality(control, variant)
    assert 0.0 <= result.control_p_value <= 1.0
    assert 0.0 <= result.variant_p_value <= 1.0


# --- Regression: large-sample CLT branch must not let a Shapiro-Wilk
# rejection downgrade validity/confidence (informational-only). ---------

def test_large_sample_normality_failure_is_marked_informational():
    """A skewed-but-large sample fails Shapiro-Wilk, but since the CLT
    branch (n>=30/arm) was used for test selection, the failure must be
    informational only — never a real quality/confidence downgrade."""
    rng = np.random.default_rng(0)
    control = pd.Series(rng.exponential(scale=2.0, size=5000))
    variant = pd.Series(rng.exponential(scale=2.1, size=5000))
    result = check_normality(control, variant)
    assert result.control_normal is False  # genuinely non-normal, large p-rejection expected

    qc = normality_result_to_quality_check(result, large_sample_rule_applied=True)
    assert qc.passed is False
    assert qc.informational is True
    assert "does not lower confidence" in qc.detail


def test_small_sample_normality_failure_is_not_informational():
    """Unchanged behavior: when the small-sample branch is used, a
    Shapiro-Wilk failure is real, decision-relevant information."""
    rng = np.random.default_rng(0)
    control = pd.Series(rng.exponential(scale=2.0, size=20))
    variant = pd.Series(rng.normal(50, 5, size=20))
    result = check_normality(control, variant)

    qc = normality_result_to_quality_check(result, large_sample_rule_applied=False)
    assert qc.informational is False


def test_informational_quality_failure_does_not_downgrade_validity():
    """End-to-end: an informational-only failed QualityCheck must not push
    ExperimentValidity below VALID, while a real (non-informational)
    failure still downgrades to CAUTION."""
    dataset = DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=10000,
        metric_label="Conversion Rate",
        metric_selection_reason="test",
    )
    base_kwargs = dict(
        user_prompt="Analyze this experiment end-to-end",
        dataset=dataset,
        srm_passed=True,
        stat_results=[],
        test_selections=[],
        power_analysis=None,
        mde_display=None,
        sample_size_note=None,
    )

    informational_failure = QualityCheck(
        label="Normality (Shapiro-Wilk)",
        passed=False,
        detail="large-sample CLT branch — informational only",
        informational=True,
    )
    facts_informational = ReportFacts(quality_checks=[informational_failure], **base_kwargs)
    assert experiment_validity(facts_informational) == ExperimentValidity.VALID

    real_failure = QualityCheck(
        label="Missing Values",
        passed=False,
        detail="12% missing",
        informational=False,
    )
    facts_real = ReportFacts(quality_checks=[real_failure], **base_kwargs)
    assert experiment_validity(facts_real) == ExperimentValidity.CAUTION


# --- critical outlier detection (Task 1 — sigma-from-mean wired to StatsThresholds.outlier_sigma) ---
#
# `StatsThresholds.outlier_sigma` (app/core/config.py) was defined but never
# consumed anywhere before this: an IQR-flagged outlier is additionally
# marked `critical=True` when it lies beyond `outlier_sigma` (4.0) standard
# deviations from the arm's own mean. These regression tests cover: an
# ordinary (non-critical) outlier, a genuinely critical one, no outlier at
# all, a large sample with many mild (non-critical) outliers, and the
# sigma=4.0 boundary itself in both directions.

def test_ordinary_outlier_is_not_critical():
    """An IQR-flagged outlier that stays under 4 sigma from the arm's mean must NOT be marked critical."""
    values = [48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 65.0]
    qc = check_outliers(pd.Series(values), metric_type=MetricType.CONTINUOUS_GENERAL)
    assert qc.passed is False  # IQR-flagged
    assert qc.critical is False  # but nowhere near 4 sigma from the mean


def test_critical_outlier_beyond_sigma_threshold():
    """A genuinely extreme value (a corrupted/anomalous row) beyond 4 sigma must be marked critical."""
    values = [50.0 + (i % 5) for i in range(99)] + [999999.0]
    qc = check_outliers(pd.Series(values), metric_type=MetricType.CONTINUOUS_GENERAL)
    assert qc.passed is False
    assert qc.critical is True


def test_no_outlier_means_no_critical_flag():
    """A clean, tight sample with no IQR-flagged outliers is never critical, regardless of sigma."""
    values = [48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 50.8]
    qc = check_outliers(pd.Series(values), metric_type=MetricType.CONTINUOUS_GENERAL)
    assert qc.passed is True
    assert qc.critical is False


def test_large_sample_many_mild_outliers_not_critical():
    """
    A large sample can have MANY IQR-flagged points (heavy-tailed/right-skewed
    data naturally trips the 1.5x-IQR Tukey fence more often than the ~0.7%
    baseline rate for normal data — see test_no_outliers_passes_for_continuous_general's
    note) without any single one of them being a genuinely extreme/corrupted
    value. None of these mild tail points should be marked critical.
    """
    rng = np.random.default_rng(3)
    tight_cluster = rng.normal(50, 2, size=1700)
    # A batch of moderately-elevated (but not corrupted) values, well
    # clear of the 4-sigma critical threshold on this exact sample.
    mild_tail = np.full(50, 60.0)
    values = np.concatenate([tight_cluster, mild_tail])
    qc = check_outliers(pd.Series(values), metric_type=MetricType.CONTINUOUS_GENERAL)
    assert qc.passed is False  # IQR flags a meaningful number of tail points
    assert qc.critical is False  # but none of them are >4 sigma from the mean


def test_monetary_outlier_critical_check_uses_log_space():
    """
    Regression for the right-skew false-positive this project's own
    data/demo/demo_ab_aov_cuped.csv originally tripped: a real, healthy
    monetary (order-value) distribution can have its largest legitimate
    orders land ~4-5 LINEAR standard deviations from the mean purely from
    right-skew, which must NOT be flagged critical. Working in log-space
    (valid here since zeros are already excluded, guaranteeing positivity)
    correctly normalizes that skew away, while a genuinely corrupted value
    (e.g. a stray $1,000,000 order) still scores far beyond 4 sigma even
    in log-space and IS correctly flagged critical.
    """
    rng = np.random.default_rng(11)
    healthy_orders = list(rng.lognormal(mean=3.8, sigma=0.5, size=1500))  # realistic right-skewed AOV data
    zeros = [0.0] * 500
    healthy_series = pd.Series(healthy_orders + zeros)
    qc_healthy = check_outliers(healthy_series, metric_type=MetricType.CONTINUOUS_MONETARY)
    assert qc_healthy.critical is False

    corrupted_series = pd.Series(healthy_orders + [1_000_000.0] + zeros)
    qc_corrupted = check_outliers(corrupted_series, metric_type=MetricType.CONTINUOUS_MONETARY)
    assert qc_corrupted.passed is False
    assert qc_corrupted.critical is True


def test_critical_sigma_boundary_just_above_threshold_is_critical():
    """Edge case: a value scoring just OVER 4.0 sigma from the arm's mean must be critical."""
    rng = np.random.default_rng(0)
    base = list(rng.normal(50, 5, size=40))
    values = base + [72.0]  # verified ~4.14 sigma from this exact sample's mean/std
    qc = check_outliers(pd.Series(values), metric_type=MetricType.CONTINUOUS_GENERAL)
    assert qc.passed is False
    assert qc.critical is True


def test_critical_sigma_boundary_just_below_threshold_is_not_critical():
    """Edge case: a value scoring just UNDER 4.0 sigma from the arm's mean must NOT be critical."""
    rng = np.random.default_rng(0)
    base = list(rng.normal(50, 5, size=40))
    values = base + [69.0]  # verified ~3.80 sigma from this exact sample's mean/std
    qc = check_outliers(pd.Series(values), metric_type=MetricType.CONTINUOUS_GENERAL)
    assert qc.passed is False  # still IQR-flagged
    assert qc.critical is False  # but under the 4.0 sigma critical threshold


# --- Regression: metric-aware outlier method selection ---------------------
# Plain linear IQR is too aggressive on right-skewed monetary data — a
# handful of ordinary large orders routinely clear the 1.5x Tukey fence
# purely from right-skew, not from any real data-quality problem. Monetary
# metrics must use Log-IQR (Tukey fences computed on log(values)); other
# continuous metrics must be completely unaffected (still plain linear IQR).

def test_monetary_metric_uses_log_iqr_method_name():
    series = pd.Series([48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 50.8])
    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY)
    assert "Log-IQR" in qc.detail


def test_general_metric_still_uses_plain_iqr_method_name():
    """Symmetric continuous metrics keep the original linear IQR — unaffected by the monetary fix."""
    series = pd.Series([48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 50.8])
    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_GENERAL)
    assert "Log-IQR" not in qc.detail
    assert "IQR method" in qc.detail


def test_log_iqr_avoids_false_positives_on_healthy_skewed_monetary_data():
    """
    A real, non-corrupted right-skewed AOV distribution must NOT be
    flagged as having outliers en masse under Log-IQR, even though plain
    linear IQR would flag a large fraction of it purely from right-skew.
    """
    rng = np.random.default_rng(11)
    healthy_orders = list(rng.lognormal(mean=3.8, sigma=0.5, size=2000))
    zeros = [0.0] * 500
    series = pd.Series(healthy_orders + zeros)

    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY)

    # Log-space IQR should flag close to the ~0.7% Tukey baseline rate for
    # genuinely normal (here: log-normal) data — nowhere near the ~10%+ a
    # naive linear IQR flags on lognormal(sigma=0.5) data.
    import re
    match = re.search(r"(\d+) outlier", qc.detail)
    n_flagged = int(match.group(1)) if match else 0
    assert n_flagged < 0.03 * len(healthy_orders)


def test_linear_iqr_would_have_overflagged_the_same_healthy_data():
    """
    Sanity check proving the log-space fix actually matters on this data:
    plain linear IQR on the SAME healthy right-skewed values flags a much
    larger fraction than Log-IQR does — demonstrating the pre-fix method
    really was too aggressive for this metric shape.
    """
    rng = np.random.default_rng(11)
    healthy_orders = pd.Series(list(rng.lognormal(mean=3.8, sigma=0.5, size=2000)))

    q1, q3 = healthy_orders.quantile(0.25), healthy_orders.quantile(0.75)
    iqr = q3 - q1
    linear_flagged = int(((healthy_orders < q1 - 1.5 * iqr) | (healthy_orders > q3 + 1.5 * iqr)).sum())

    log_vals = np.log(healthy_orders)
    q1l, q3l = log_vals.quantile(0.25), log_vals.quantile(0.75)
    iqrl = q3l - q1l
    log_flagged = int(((log_vals < q1l - 1.5 * iqrl) | (log_vals > q3l + 1.5 * iqrl)).sum())

    assert linear_flagged > log_flagged


def test_log_iqr_still_detects_genuine_monetary_outlier():
    """A truly corrupted value must still be caught under Log-IQR — the fix must not make detection blind."""
    rng = np.random.default_rng(11)
    healthy_orders = list(rng.lognormal(mean=3.8, sigma=0.5, size=1500))
    zeros = [0.0] * 500
    series = pd.Series(healthy_orders + [1_000_000.0] + zeros)

    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY)
    assert qc.passed is False
    assert qc.critical is True


def test_log_iqr_multiplier_still_respects_custom_multiplier():
    """
    Regression guard: the fix changes WHICH SPACE the fence is computed in
    for monetary metrics, never the 1.5x multiplier itself — `iqr_multiplier`
    must still be the single lever: a wider multiplier must flag fewer or
    equal outliers than the default, confirming no separate/inflated
    multiplier was silently hardcoded for the monetary branch.
    """
    rng = np.random.default_rng(2)
    series = pd.Series(list(rng.lognormal(3.8, 0.5, size=300)) + [5000.0])
    default_mult = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY, iqr_multiplier=1.5)
    wider_mult = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY, iqr_multiplier=5.0)

    def _n_flagged(qc):
        import re
        m = re.search(r"(\d+) outlier", qc.detail)
        return int(m.group(1)) if m else 0

    assert _n_flagged(wider_mult) <= _n_flagged(default_mult)


def test_outlier_detection_is_deterministic_for_monetary():
    """Same input, called twice, must give identical output — no hidden randomness in the log-space path."""
    rng = np.random.default_rng(5)
    series = pd.Series(list(rng.lognormal(3.8, 0.5, size=500)) + [0.0] * 200)
    a = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY)
    b = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY)
    assert a == b


# --- Regression: zeros-excluded semantics -----------------------------------
# "zeros excluded" must only ever affect the pool check_outliers itself runs
# on. It must never be mistaken for (or leak into) the experiment's total n,
# and the report text must clearly separate the three counts.

def test_outlier_detail_reports_total_n_zero_count_and_used_n_separately():
    positive_values = [48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 50.8]
    zeros = [0.0] * 700
    series = pd.Series(positive_values + zeros)  # total arm n = 710

    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY)

    assert "710" in qc.detail  # (a) total experiment sample size for this arm
    assert "700" in qc.detail  # (b) structural-zero observations
    assert "10 values" in qc.detail  # (c) observations actually used for outlier detection


def test_zeros_excluded_does_not_reduce_statistical_sample_size():
    """
    Regression: excluding structural zeros from
    OUTLIER DETECTION must never shrink the n used by the main hypothesis
    test / power analysis. This mirrors the real pipeline wiring
    (validation_node/experiment_node both build control_metric/variant_metric
    from the SAME full, non-zero-excluded series) at the unit level: running
    check_outliers (which internally drops zeros) must not mutate or affect
    a separately-computed power analysis on the original series.
    """
    from app.stats.power_analysis import compute_power_analysis

    rng = np.random.default_rng(3)
    positive_values = rng.lognormal(3.8, 0.5, size=300)
    zeros = np.zeros(700)
    control = pd.Series(np.concatenate([positive_values, zeros]))
    variant = pd.Series(np.concatenate([positive_values * 1.05, zeros]))

    total_n_control, total_n_variant = len(control), len(variant)

    outlier_qc = check_outliers(control, metric_type=MetricType.CONTINUOUS_MONETARY)
    assert "700" in outlier_qc.detail  # confirms zeros really were excluded from outlier detection

    power_result = compute_power_analysis(control, variant, MetricType.CONTINUOUS_MONETARY)

    # The statistical sample size must equal the FULL arms (zeros included),
    # never the outlier-detection subset (300/arm).
    assert power_result.observed_sample_size == total_n_control + total_n_variant
    assert power_result.observed_sample_size == 2000


def test_general_metric_has_no_zero_exclusion_note():
    """CONTINUOUS_GENERAL has no structural-zero concept — detail must not claim any zeros were excluded."""
    series = pd.Series([48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 50.8, 0.0])
    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_GENERAL)
    assert "zeros excluded" not in qc.detail


def test_combine_outlier_checks_propagates_critical_from_either_arm():
    """`critical` on the combined row must be True if EITHER arm's check was critical."""
    clean = check_outliers(
        pd.Series([48.0, 49.0, 50.0, 50.5, 51.0, 51.5, 52.0, 49.5, 50.2, 50.8]),
        metric_type=MetricType.CONTINUOUS_GENERAL,
    )
    assert clean.critical is False

    critical_arm_values = [50.0 + (i % 5) for i in range(99)] + [999999.0]
    critical_arm = check_outliers(pd.Series(critical_arm_values), metric_type=MetricType.CONTINUOUS_GENERAL)
    assert critical_arm.critical is True

    combined_variant_critical = combine_outlier_checks(clean, critical_arm)
    assert combined_variant_critical.critical is True

    combined_control_critical = combine_outlier_checks(critical_arm, clean)
    assert combined_control_critical.critical is True

    combined_neither_critical = combine_outlier_checks(clean, clean)
    assert combined_neither_critical.critical is False


# --- BUG 1 regression: degenerate IQR fence (Q1 == Q3) ----------------------
# Sparse/zero-inflated count metrics (e.g. `clicks`) routinely push both
# quartiles to the same value (often 0), collapsing the Tukey fence to a
# single point. The pre-fix behavior flagged every value that merely
# differed from one arbitrary observation as an "outlier" — for a metric
# where the majority of values equal the modal value, that misclassifies
# EVERY other observation, which can spuriously trip the critical-outlier
# path and mark an otherwise healthy experiment INVALID.
#
# Fixed behavior: when Q1 == Q3, fall back to a MAD (median absolute
# deviation) based fence — modified z-score (Iglewicz & Hoaglin) > 3.5 —
# which doesn't require the quartiles to be distinct. If MAD is also 0
# (as with real-world `clicks`, where the modal value is the majority of
# the sample), there is no robust basis to flag anything, so nothing is
# flagged.

def test_zero_inflated_count_metric_with_degenerate_iqr_is_not_all_flagged():
    """
    Direct regression for the reported bug: a sparse/zero-inflated count
    metric (Q1 == Q3 == 0, most values are 0) must NOT have every positive
    observation classified as an outlier.
    """
    rng = np.random.default_rng(7)
    n = 6000
    zeros = np.zeros(int(n * 0.846))
    positive_clicks = rng.integers(1, 4, size=n - len(zeros)).astype(float)
    series = pd.Series(np.concatenate([zeros, positive_clicks]))

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    assert q1 == q3 == 0  # confirms this sample actually reproduces the degenerate case

    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_GENERAL)

    # The bug flagged ~all positive values (i.e. n_outliers ~= len(positive_clicks)).
    # The fix must flag dramatically fewer than that — here, none, since
    # ordinary small positive click counts have no robust (MAD) basis to be
    # called outliers when the modal value dominates the sample.
    import re

    match = re.search(r"(\d+) outlier", qc.detail)
    n_flagged = int(match.group(1)) if match else 0
    assert n_flagged < len(positive_clicks) * 0.05
    assert qc.passed is True
    assert qc.critical is False


def test_real_clicks_dataset_shape_no_longer_invalidates():
    """
    Reproduces the exact motivating scenario: ~84.6% zeros, Q1 = Q3 = 0,
    60,000 users per arm. Pre-fix this produced 8,820/9,703 "outliers" and
    could trigger a critical data-quality failure. Post-fix it must pass
    cleanly with zero outliers, since a zero-width Tukey fence provides no
    genuine evidence that any positive click count is anomalous.
    """
    rng = np.random.default_rng(42)
    n_per_arm = 60000
    n_zero = int(n_per_arm * 0.846)
    n_positive = n_per_arm - n_zero
    positive_values = rng.integers(1, 10, size=n_positive).astype(float)
    control = pd.Series(np.concatenate([np.zeros(n_zero), positive_values]))
    variant = pd.Series(np.concatenate([np.zeros(n_zero), positive_values]))

    control_qc = check_outliers(control, metric_type=MetricType.CONTINUOUS_GENERAL)
    variant_qc = check_outliers(variant, metric_type=MetricType.CONTINUOUS_GENERAL)
    combined = combine_outlier_checks(control_qc, variant_qc)

    assert control_qc.passed is True
    assert variant_qc.passed is True
    assert combined.passed is True
    assert combined.critical is False


def test_degenerate_iqr_detail_notes_mad_fallback_was_used():
    """The report detail must be auditable: it should be clear a fallback method was used, not silent."""
    series = pd.Series([0.0] * 90 + [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 4.0, 1.0, 2.0])
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    assert q1 == q3  # degenerate

    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_GENERAL)
    assert "degenerate" in qc.detail.lower()


def test_normal_continuous_metric_without_degenerate_iqr_is_unaffected():
    """
    Regression guard: a normal continuous metric with a healthy, non-zero
    IQR must be completely unaffected by the MAD fallback — it should never
    even reach that code path, so behavior/method must stay identical to
    the existing plain linear IQR (Q1 != Q3 in this sample).
    """
    rng = np.random.default_rng(1)
    values = list(rng.normal(50, 5, size=200))
    series = pd.Series(values)
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    assert q1 != q3  # confirms this is the ordinary, non-degenerate path

    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_GENERAL)
    assert "degenerate" not in qc.detail.lower()
    assert "IQR method" in qc.detail


def test_existing_monetary_log_iqr_behavior_is_unaffected_by_degenerate_fix():
    """
    Regression guard: healthy right-skewed monetary data (Q1 != Q3
    after log-transform) must still use ordinary Log-IQR — the
    degenerate-IQR fallback must not fire and must not change this
    metric type's existing behavior at all.
    """
    rng = np.random.default_rng(11)
    healthy_orders = list(rng.lognormal(mean=3.8, sigma=0.5, size=2000))
    zeros = [0.0] * 500
    series = pd.Series(healthy_orders + zeros)

    positive = series[series > 0]
    log_vals = np.log(positive)
    q1, q3 = log_vals.quantile(0.25), log_vals.quantile(0.75)
    assert q1 != q3  # confirms this is the ordinary, non-degenerate Log-IQR path

    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_MONETARY)
    assert "Log-IQR" in qc.detail
    assert "degenerate" not in qc.detail.lower()


def test_fully_constant_series_has_no_outliers_via_either_method():
    """Edge case: a perfectly constant series (MAD also 0) must report zero outliers, not error out."""
    series = pd.Series([5.0] * 50)
    qc = check_outliers(series, metric_type=MetricType.CONTINUOUS_GENERAL)
    assert qc.passed is True
    assert "0 outlier" not in qc.detail  # "no outliers detected" phrasing, not "0 outlier(s)"
    assert "no outliers detected" in qc.detail
