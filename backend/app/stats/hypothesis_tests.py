"""
Hypothesis testing — Stage 4.

TWO responsibilities, deliberately kept separate:

  1. `select_test()` — the SINGLE public decision-tree function. All
     branching logic for "which test to use" lives inside this one
     function (mirrors how analytics libraries like statsmodels /
     scipy structure a top-level dispatcher). The LLM NEVER makes this
     decision — this function is pure Python, fully deterministic, and
     unit-testable in isolation from any LLM call.

  2. `compute_stat_result()` — given a (possibly pre-selected) test,
     actually runs it and returns a `StatResult` with real numbers.
     Kept separate from selection because "which test" and "what did
     the test say" are different concerns with different callers in
     the graph (the ExecutionStepper wants to show *which* test was
     picked and why, before/independent of showing the numeric result).

Also home to `format_p_value()` — the one place that turns a raw float
p-value into display text. At very large sample sizes (e.g. a
294,478-user chi-square test) scipy can return a p-value that
underflows to exactly 0.0 in floating point; formatting that directly
(e.g. `f"{p:.4g}"`) would print the false, meaningless "p=0". This
mirrors the threshold `stats/srm.py` already used for its own p-value
display — centralized here so every report/chat surface goes through
the same rule instead of re-implementing the threshold check.

DECISION TREE (confirmed):

  Binary metrics:
    -> Chi-square test of independence on the 2x2 contingency table.
    -> If any expected cell count < 5 (standard rule of thumb for
       chi-square validity), use Fisher's exact test instead.

  Continuous metrics:
    -> If both arms have n >= 30: Welch's t-test by default. No
       normality check performed — justified by the Central Limit
       Theorem (sampling distribution of the mean is approximately
       normal regardless of the underlying distribution at this N).
    -> Otherwise: run Shapiro-Wilk on both arms.
         -> Both normal -> Welch's t-test.
         -> Either non-normal -> Mann-Whitney U test (non-parametric).

  Welch's t-test is used in both the large-sample and normal-small-
  sample branches (never Student's t) because Welch's does not assume
  equal variances and is strictly safer with no meaningful downside
  when variances happen to be equal — this removes the need for a
  separate Levene's-test branch entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.config import stats_thresholds
from app.schemas.statistics import (
    TEST_TYPE_DISPLAY_NAMES,
    HypothesisTestType,
    MetricType,
    StatResult,
    TestSelectionResult,
)
from app.stats.quality_checks import check_normality

_LARGE_SAMPLE_THRESHOLD = 30
_MIN_EXPECTED_CELL_COUNT = 5  # standard chi-square validity rule of thumb

# Same threshold app/stats/srm.py already uses for its own p-value display.
_P_VALUE_DISPLAY_FLOOR = 0.001


def format_p_value(p_value: float) -> str:
    """
    Presentation-only — never changes the underlying numeric p-value
    used for `significant` / alpha comparisons or any calculation.
    Returns a full comparison ready to follow "p ", e.g. "= 0.0421" or
    "< 0.001" — and specifically NEVER "= 0", which floating-point
    underflow can otherwise produce at very large sample sizes (a
    294,478-user chi-square test can return p_value == 0.0 exactly).
    Callers write f"p {format_p_value(p)}".
    """
    if p_value < _P_VALUE_DISPLAY_FLOOR:
        return f"< {_P_VALUE_DISPLAY_FLOOR:g}"
    return f"= {p_value:.4g}"


def select_test(
    control: pd.Series,
    variant: pd.Series,
    metric_type: MetricType,
) -> TestSelectionResult:
    """
    The single deterministic decision-tree entry point. See module
    docstring for the full rule set. Returns not just the chosen test
    but a human-readable `reason` string — this is what feeds both the
    ExecutionStepper detail text and the decision LLM's prompt context
    (the LLM explains the choice in prose; it never makes the choice).
    """
    if metric_type == MetricType.BINARY:
        return _select_test_binary(control, variant)
    return _select_test_continuous(control, variant)


def _select_test_binary(control: pd.Series, variant: pd.Series) -> TestSelectionResult:
    control_clean = control.dropna()
    variant_clean = variant.dropna()

    table = _build_contingency_table(control_clean, variant_clean)
    expected = scipy_stats.contingency.expected_freq(table)
    min_expected = expected.min()

    if min_expected < _MIN_EXPECTED_CELL_COUNT:
        reason = (
            f"Binary metric with a small expected cell count "
            f"(min expected = {min_expected:.1f} < {_MIN_EXPECTED_CELL_COUNT}) — "
            f"chi-square approximation is unreliable here, so Fisher's exact "
            f"test is used instead."
        )
        return TestSelectionResult(test_type=HypothesisTestType.FISHERS_EXACT, reason=reason)

    reason = (
        f"Binary metric with sufficient expected cell counts "
        f"(min expected = {min_expected:.1f} >= {_MIN_EXPECTED_CELL_COUNT}) — "
        f"chi-square test of independence is valid and used."
    )
    return TestSelectionResult(test_type=HypothesisTestType.CHI_SQUARE, reason=reason)


def _select_test_continuous(control: pd.Series, variant: pd.Series) -> TestSelectionResult:
    control_clean = control.dropna()
    variant_clean = variant.dropna()
    n_control, n_variant = len(control_clean), len(variant_clean)

    if n_control >= _LARGE_SAMPLE_THRESHOLD and n_variant >= _LARGE_SAMPLE_THRESHOLD:
        reason = (
            f"Both arms have n >= {_LARGE_SAMPLE_THRESHOLD} "
            f"(control={n_control}, variant={n_variant}) — by the Central "
            f"Limit Theorem the sampling distribution of the mean is "
            f"approximately normal regardless of the underlying "
            f"distribution, so Welch's t-test is used by default without "
            f"a normality check."
        )
        return TestSelectionResult(
            test_type=HypothesisTestType.WELCH_T_TEST,
            reason=reason,
            large_sample_rule_applied=True,
        )

    normality = check_normality(control_clean, variant_clean)
    both_normal = normality.control_normal and normality.variant_normal

    if both_normal:
        reason = (
            f"Small sample (control={n_control}, variant={n_variant}, "
            f"below the n>={_LARGE_SAMPLE_THRESHOLD} large-sample threshold) — "
            f"Shapiro-Wilk indicates both arms are approximately normal "
            f"(control p={normality.control_p_value:.3f}, "
            f"variant p={normality.variant_p_value:.3f}), so Welch's t-test "
            f"is used."
        )
        return TestSelectionResult(
            test_type=HypothesisTestType.WELCH_T_TEST,
            reason=reason,
            normality=normality,
        )

    reason = (
        f"Small sample (control={n_control}, variant={n_variant}, "
        f"below the n>={_LARGE_SAMPLE_THRESHOLD} large-sample threshold) — "
        f"Shapiro-Wilk indicates at least one arm is not normally "
        f"distributed (control p={normality.control_p_value:.3f}, "
        f"variant p={normality.variant_p_value:.3f}), so the non-parametric "
        f"Mann-Whitney U test is used instead of a t-test."
    )
    return TestSelectionResult(
        test_type=HypothesisTestType.MANN_WHITNEY_U,
        reason=reason,
        normality=normality,
    )


def _build_contingency_table(control: pd.Series, variant: pd.Series) -> np.ndarray:
    """2x2 table: rows = [control, variant], columns = [success, failure]."""
    control_success = int(control.sum())
    control_failure = len(control) - control_success
    variant_success = int(variant.sum())
    variant_failure = len(variant) - variant_success
    return np.array([[control_success, control_failure], [variant_success, variant_failure]])


def compute_stat_result(
    control: pd.Series,
    variant: pd.Series,
    metric_type: MetricType,
    metric_label: str,
    test_selection: TestSelectionResult | None = None,
    alpha: float | None = None,
) -> StatResult:
    """
    Run the (possibly already-selected) hypothesis test and return the
    frontend-facing `StatResult`. If `test_selection` is not provided,
    `select_test()` is called internally first.
    """
    alpha = stats_thresholds.significance_alpha if alpha is None else alpha
    if test_selection is None:
        test_selection = select_test(control, variant, metric_type)

    control_clean = control.dropna()
    variant_clean = variant.dropna()

    if test_selection.test_type in (HypothesisTestType.CHI_SQUARE, HypothesisTestType.FISHERS_EXACT):
        return _compute_binary_result(control_clean, variant_clean, metric_label, test_selection, alpha)
    if test_selection.test_type == HypothesisTestType.WELCH_T_TEST:
        return _compute_welch_result(control_clean, variant_clean, metric_type, metric_label, test_selection, alpha)
    return _compute_mann_whitney_result(control_clean, variant_clean, metric_type, metric_label, test_selection, alpha)


def _compute_binary_result(
    control: pd.Series,
    variant: pd.Series,
    metric_label: str,
    test_selection: TestSelectionResult,
    alpha: float,
) -> StatResult:
    n_c, n_v = len(control), len(variant)
    p_c, p_v = control.mean(), variant.mean()

    if test_selection.test_type == HypothesisTestType.FISHERS_EXACT:
        table = _build_contingency_table(control, variant)
        odds_ratio, p_value = scipy_stats.fisher_exact(table)
        statistic = odds_ratio  # Fisher's has no chi2-like statistic; odds ratio is the standard reported number
    else:
        table = _build_contingency_table(control, variant)
        chi2_stat, p_value, _, _ = scipy_stats.chi2_contingency(table, correction=True)
        statistic = chi2_stat

    # Wald CI for the difference in proportions (p_v - p_c), standard
    # industry approximation used for both chi-square and Fisher's
    # exact contexts since Fisher's exact has no universally-agreed
    # native CI equivalent.
    diff = p_v - p_c
    se = ((p_c * (1 - p_c) / n_c) + (p_v * (1 - p_v) / n_v)) ** 0.5
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    ci_lower_pp = (diff - z * se) * 100
    ci_upper_pp = (diff + z * se) * 100

    relative_delta = (diff / p_c * 100) if p_c != 0 else float("inf")
    # Phase 2 — same underlying ratio as `relative_delta` above, kept as
    # a raw fraction (not *100, not a formatted string) for
    # StatResult.observed_relative_effect. None when the baseline is
    # zero — a relative effect is undefined there, and we deliberately
    # never serialize inf/NaN onto the schema.
    observed_relative_effect = (diff / p_c) if p_c != 0 else None

    return StatResult(
        metric=metric_label,
        test_type=test_selection.test_type,
        test_name=TEST_TYPE_DISPLAY_NAMES[test_selection.test_type],
        statistic=float(statistic),
        selection_reason=test_selection.reason,
        control=f"{p_c * 100:.2f}%",
        variant=f"{p_v * 100:.2f}%",
        delta=f"{relative_delta:+.1f}% (rel)",
        observed_relative_effect=observed_relative_effect,
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        ci_lower=f"{ci_lower_pp:+.2f}pp",
        ci_upper=f"{ci_upper_pp:+.2f}pp",
    )


def _compute_welch_result(
    control: pd.Series,
    variant: pd.Series,
    metric_type: MetricType,
    metric_label: str,
    test_selection: TestSelectionResult,
    alpha: float,
) -> StatResult:
    t_stat, p_value = scipy_stats.ttest_ind(variant, control, equal_var=False)

    mean_c, mean_v = control.mean(), variant.mean()
    diff = mean_v - mean_c

    # Welch-Satterthwaite CI for the difference in means.
    se_c = control.var(ddof=1) / len(control)
    se_v = variant.var(ddof=1) / len(variant)
    se_diff = (se_c + se_v) ** 0.5
    df = (se_c + se_v) ** 2 / ((se_c**2 / (len(control) - 1)) + (se_v**2 / (len(variant) - 1)))
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
    ci_lower = diff - t_crit * se_diff
    ci_upper = diff + t_crit * se_diff

    relative_delta = (diff / mean_c * 100) if mean_c != 0 else float("inf")
    observed_relative_effect = (diff / mean_c) if mean_c != 0 else None
    fmt = _formatter_for(metric_type)

    return StatResult(
        metric=metric_label,
        test_type=test_selection.test_type,
        test_name=TEST_TYPE_DISPLAY_NAMES[test_selection.test_type],
        statistic=float(t_stat),
        selection_reason=test_selection.reason,
        control=fmt(mean_c),
        variant=fmt(mean_v),
        delta=f"{relative_delta:+.1f}% (rel)",
        observed_relative_effect=observed_relative_effect,
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        ci_lower=f"{'+' if ci_lower >= 0 else ''}{fmt(ci_lower)}",
        ci_upper=f"{'+' if ci_upper >= 0 else ''}{fmt(ci_upper)}",
    )


def _compute_mann_whitney_result(
    control: pd.Series,
    variant: pd.Series,
    metric_type: MetricType,
    metric_label: str,
    test_selection: TestSelectionResult,
    alpha: float,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> StatResult:
    """
    Mann-Whitney U test for the p-value/significance call. Mann-Whitney
    has no closed-form CI for a location shift, so the CI on the median
    difference is estimated via percentile bootstrap — still fully
    deterministic given the fixed seed, still zero LLM involvement.

    PERF (segmentation can call this dozens of times per run — each
    small segment that fails the normality check lands here): the
    original implementation did `bootstrap_iterations` (2000) resamples
    in a pure-Python `for` loop, each iteration calling `rng.choice`
    and `np.median` separately. That's the same statistical procedure
    as below (percentile bootstrap over resampled medians), just
    executed one draw at a time instead of as batched numpy ops — on a
    dataset with several small segments this dominated the
    `segmentation` pipeline stage's wall time. Below draws all
    `bootstrap_iterations` resamples' worth of indices in two vectorized
    `rng.integers` calls and takes the median along axis=1 in one shot,
    which is the same math, just not re-entering the Python interpreter
    2000 times. Uses a fresh `default_rng(seed)` exactly as before, so
    this is still fully deterministic given the fixed seed — only the
    RNG call pattern (batched vs. one-at-a-time) changed, so the exact
    stream of numbers drawn (and therefore the exact CI bounds) differs
    slightly from the old loop, but it's the same estimator with the
    same iteration count on the same data.
    """
    u_stat, p_value = scipy_stats.mannwhitneyu(variant, control, alternative="two-sided")

    median_c, median_v = control.median(), variant.median()
    diff = median_v - median_c

    rng = np.random.default_rng(seed)
    control_arr, variant_arr = control.to_numpy(), variant.to_numpy()
    n_control, n_variant = len(control_arr), len(variant_arr)

    control_idx = rng.integers(0, n_control, size=(bootstrap_iterations, n_control))
    variant_idx = rng.integers(0, n_variant, size=(bootstrap_iterations, n_variant))
    boot_c_medians = np.median(control_arr[control_idx], axis=1)
    boot_v_medians = np.median(variant_arr[variant_idx], axis=1)
    boot_diffs = boot_v_medians - boot_c_medians

    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    relative_delta = (diff / median_c * 100) if median_c != 0 else float("inf")
    observed_relative_effect = (diff / median_c) if median_c != 0 else None
    fmt = _formatter_for(metric_type)

    return StatResult(
        metric=metric_label,
        test_type=test_selection.test_type,
        test_name=TEST_TYPE_DISPLAY_NAMES[test_selection.test_type],
        statistic=float(u_stat),
        selection_reason=test_selection.reason,
        control=fmt(median_c),
        variant=fmt(median_v),
        delta=f"{relative_delta:+.1f}% (rel, median)",
        observed_relative_effect=observed_relative_effect,
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        ci_lower=f"{'+' if ci_lower >= 0 else ''}{fmt(ci_lower)}",
        ci_upper=f"{'+' if ci_upper >= 0 else ''}{fmt(ci_upper)}",
    )


def _formatter_for(metric_type: MetricType):
    """Display formatting matches mock-data.ts conventions per metric type."""
    if metric_type == MetricType.CONTINUOUS_MONETARY:
        return lambda x: f"${abs(x):.2f}" if x >= 0 else f"-${abs(x):.2f}"
    return lambda x: f"{x:.2f}"


def test_selection_to_quality_check_detail(test_selection: TestSelectionResult) -> str:
    """
    Short label for the ExecutionStepper detail text, e.g. the
    frontend's mock: "Normality verified, equal variance rejected,
    Welch selected". This produces the equivalent short-form summary
    from the real `reason` text.
    """
    label_by_type = {
        HypothesisTestType.WELCH_T_TEST: "Welch's t-test",
        HypothesisTestType.MANN_WHITNEY_U: "Mann-Whitney U test",
        HypothesisTestType.CHI_SQUARE: "Chi-square test",
        HypothesisTestType.FISHERS_EXACT: "Fisher's exact test",
    }
    return f"Automated Test Selection ({label_by_type[test_selection.test_type]})"



def compute_multi_arm_stat_results(
    arms: dict[str, pd.Series],
    metric_type: MetricType,
    metric_label: str,
    alpha: float | None = None,
    control_label: str | None = None,
) -> list[StatResult]:
    """Run a deterministic omnibus + corrected pairwise analysis for 3+ arms.

    The first result is the omnibus test (does any arm differ?). If it is
    significant, control-vs-each-treatment pairwise tests are returned with
    Holm correction. The LLM never selects the test or performs correction.

    `control_label` (optional): which key in `arms` is the actual
    control/reference arm. Without this, the reference arm defaults to
    `labels[0]` — whichever key happens to be first in dict insertion
    order — which is only guaranteed correct if the caller inserted
    "control" first. A real dataset's variant column has no guaranteed
    insertion order (e.g. a CRM dataset where the actual control arm is
    not the first-appearing row), so pairwise comparisons could
    otherwise use the wrong arm as the baseline. When omitted or not
    present in `arms`, falls back to the `labels[0]` behavior.
    """
    alpha = stats_thresholds.significance_alpha if alpha is None else alpha
    clean_arms = {label: series.dropna() for label, series in arms.items()}
    clean_arms = {label: series for label, series in clean_arms.items() if len(series) > 0}
    if len(clean_arms) < 3:
        raise ValueError("Multi-arm analysis requires at least three non-empty arms.")

    labels = list(clean_arms)
    reference = control_label if control_label in clean_arms else labels[0]

    if metric_type == MetricType.BINARY:
        table = np.array([
            [int(series.sum()), int(len(series) - series.sum())]
            for series in clean_arms.values()
        ], dtype=float)
        statistic, p_value, _, _ = scipy_stats.chi2_contingency(table, correction=False)
        omnibus_name = "Chi-square omnibus test"
        omnibus_type = HypothesisTestType.CHI_SQUARE
    else:
        samples = list(clean_arms.values())
        # ANOVA is used when every arm has a reasonably large sample. For
        # small/non-normal arms, Kruskal-Wallis avoids pretending that the
        # normality assumption holds.
        normality_ok = all(
            len(sample) >= _LARGE_SAMPLE_THRESHOLD
            or check_normality(sample, sample).control_normal
            for sample in samples
        )
        if normality_ok and all(len(sample) >= _LARGE_SAMPLE_THRESHOLD for sample in samples):
            statistic, p_value = scipy_stats.f_oneway(*samples)
            omnibus_name = "One-way ANOVA"
            omnibus_type = HypothesisTestType.ONE_WAY_ANOVA
        else:
            statistic, p_value = scipy_stats.kruskal(*samples)
            omnibus_name = "Kruskal-Wallis test"
            omnibus_type = HypothesisTestType.KRUSKAL_WALLIS

    omnibus = StatResult(
        metric=metric_label,
        test_type=omnibus_type,
        test_name=omnibus_name,
        statistic=float(statistic),
        selection_reason=(
            f"{len(clean_arms)}-arm experiment — omnibus test checks whether any arm differs "
            f"before pairwise comparisons."
        ),
        control="All arms",
        variant=f"{len(clean_arms)} arms",
        delta="Omnibus",
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        ci_lower="N/A",
        ci_upper="N/A",
        comparison="All arms",
        is_omnibus=True,
        adjusted_p_value=float(p_value),
    )

    # No pairwise fishing when the omnibus test is null.
    if not omnibus.significant:
        return [omnibus]

    pairwise: list[StatResult] = []
    raw_p_values: list[float] = []
    treatment_labels = [label for label in labels if label != reference]
    for label in treatment_labels:
        result = compute_stat_result(
            clean_arms[reference], clean_arms[label], metric_type, metric_label
        )
        pairwise.append(
            result.model_copy(update={
                "comparison": f"{reference} vs {label}",
                "reference_arm": str(reference),
                "arm": str(label),
            })
        )
        raw_p_values.append(result.p_value)

    adjusted = _holm_adjust(raw_p_values)
    corrected: list[StatResult] = []
    for result, adjusted_p in zip(pairwise, adjusted):
        corrected.append(result.model_copy(update={
            "p_value": float(adjusted_p),
            "adjusted_p_value": float(adjusted_p),
            "significant": bool(adjusted_p < alpha),
            "multiple_testing_method": "Holm-Bonferroni",
        }))

    return [omnibus, *corrected]


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down correction, implemented locally to keep the rule explicit."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted
