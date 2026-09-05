"""
Stratified analysis — eligibility checks + estimator.

PURE, DETERMINISTIC, NO LLM (same discipline as dataset_classifier.py
and hypothesis_tests.py). Two responsibilities:

  1. `check_stratification_eligibility()` — decide whether a requested
     baseline categorical variable can be used to stratify the
     control-vs-variant comparison AT ALL on this dataset. A variable
     that is perfectly (or near-perfectly) associated with the
     treatment assignment cannot be used — every stratum would contain
     only one arm, so no within-stratum treatment effect exists to
     estimate. This is exactly the `landing_page` case in the bug
     report: `control -> old_page`, `treatment -> new_page`.

  2. `run_stratified_analysis()` — given an ELIGIBLE variable, compute
     the stratum-specific treatment effects and COMBINE them into one
     overall effect using inverse-variance weighting (a standard fixed-
     effect meta-analytic combination — Cochran, 1954). This is
     deliberately NOT a naive average of per-stratum conversion
     rates/means: strata with more precise (larger-N / lower-variance)
     estimates get proportionally more weight, and the combined
     standard error/CI/p-value are derived from the same weights,
     never just "spread across strata and eyeballed."

Stratification is orthogonal to (not a replacement for) CUPED and
Bootstrap — see module docstrings in variance_reduction.py. This
module never touches those; `experiment_node.py` runs stratification
against the SAME raw metric column CUPED/segmentation use, not a
CUPED-adjusted array (mirrors segmentation.py's own choice).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from app.core.lazy_import import LazyModule

# Deferred: only actually needed once run_stratified_analysis() computes
# a z-based p-value, not at import time — see app/core/lazy_import.py.
scipy_stats = LazyModule("scipy.stats")

from app.schemas.statistics import MetricType
from app.schemas.stratification import (
    DiagnosticStratification,
    StratificationEligibility,
    StratificationIneligibilityReason,
    StratifiedEstimate,
    StratumSRMConcentration,
    StratumSummary,
)
from app.stats.srm import check_srm

# Minimum observations required in EACH arm within a stratum for that
# stratum to be usable in the combined estimate. Below this, the
# within-stratum effect/variance estimate is too noisy to trust —
# flagged as sparse rather than silently included or silently dropped.
_MIN_ARM_SIZE = 10

_Z_95 = 1.959963984540054  # scipy_stats.norm.ppf(0.975), inlined for clarity in the formula below


def check_stratification_eligibility(
    df: pd.DataFrame,
    variant_col: str,
    control_label,
    stratification_col: str,
    min_arm_size: int = _MIN_ARM_SIZE,
) -> StratificationEligibility:
    """
    Validate a requested stratification variable against the checks
    the audit specifies:

      1. It must not BE the treatment assignment column itself.
      2. It must not be perfectly/deterministically associated with
         treatment assignment (every stratum must contain BOTH arms).
      3. Each usable stratum must have sufficient observations in each
         arm (flagged individually as `sufficient`; strata below the
         threshold are excluded from the combined estimate but still
         reported, per the "report count/proportion per stratum"
         requirement).
      4. Missing stratification values are handled explicitly (counted
         and reported, never silently dropped without disclosure).
    """
    if stratification_col not in df.columns:
        return StratificationEligibility(
            stratification_column=stratification_col,
            eligible=False,
            reason=f"Column '{stratification_col}' was not found in this dataset.",
            ineligibility_reason=StratificationIneligibilityReason.COLUMN_NOT_FOUND,
            min_arm_size=min_arm_size,
        )

    if stratification_col == variant_col:
        return StratificationEligibility(
            stratification_column=stratification_col,
            eligible=False,
            reason=(
                f"'{stratification_col}' is the treatment-assignment column itself — "
                "stratifying by the treatment variable is not a meaningful baseline stratum."
            ),
            ineligibility_reason=StratificationIneligibilityReason.IS_TREATMENT_ASSIGNMENT,
            min_arm_size=min_arm_size,
        )

    total_n = len(df)
    missing_mask = df[stratification_col].isna()
    missing_count = int(missing_mask.sum())
    working = df.loc[~missing_mask]

    strata: list[StratumSummary] = []
    sparse_values: list[str] = []
    strata_with_both_variants = 0

    for value, group in working.groupby(stratification_col, dropna=True):
        control_n = int((group[variant_col] == control_label).sum())
        variant_n = int(len(group) - control_n)
        has_both = control_n > 0 and variant_n > 0
        sufficient = has_both and control_n >= min_arm_size and variant_n >= min_arm_size
        if has_both:
            strata_with_both_variants += 1
        if has_both and not sufficient:
            sparse_values.append(str(value))

        strata.append(
            StratumSummary(
                stratum_value=str(value),
                control_n=control_n,
                variant_n=variant_n,
                total_n=len(group),
                proportion_of_total=(len(group) / total_n) if total_n else 0.0,
                has_both_variants=has_both,
                sufficient=sufficient,
            )
        )

    missing_proportion = (missing_count / total_n) if total_n else 0.0

    if len(strata) == 0:
        return StratificationEligibility(
            stratification_column=stratification_col,
            eligible=False,
            reason=f"'{stratification_col}' has no non-missing values to stratify on.",
            ineligibility_reason=StratificationIneligibilityReason.NO_STRATA_WITH_BOTH_VARIANTS,
            strata=strata,
            total_n=total_n,
            missing_count=missing_count,
            missing_proportion=missing_proportion,
            min_arm_size=min_arm_size,
        )

    if strata_with_both_variants == 0:
        return StratificationEligibility(
            stratification_column=stratification_col,
            eligible=False,
            reason=(
                f"Stratified analysis cannot be performed using {stratification_col} because "
                "the selected variable is perfectly associated with experiment assignment. "
                "Each stratum contains only one variant, so a within-stratum treatment effect "
                "cannot be estimated."
            ),
            ineligibility_reason=StratificationIneligibilityReason.PERFECTLY_ASSOCIATED_WITH_ASSIGNMENT,
            strata=strata,
            total_n=total_n,
            missing_count=missing_count,
            missing_proportion=missing_proportion,
            min_arm_size=min_arm_size,
            sparse_stratum_values=sparse_values,
        )

    sufficient_strata = [s for s in strata if s.sufficient]
    if len(sufficient_strata) == 0:
        return StratificationEligibility(
            stratification_column=stratification_col,
            eligible=False,
            reason=(
                f"'{stratification_col}' has strata containing both variants, but every one of "
                f"them has fewer than {min_arm_size} observations in at least one arm — none are "
                "large enough to produce a trustworthy within-stratum estimate."
            ),
            ineligibility_reason=StratificationIneligibilityReason.ALL_STRATA_TOO_SPARSE,
            strata=strata,
            total_n=total_n,
            missing_count=missing_count,
            missing_proportion=missing_proportion,
            min_arm_size=min_arm_size,
            sparse_stratum_values=sparse_values,
        )

    reason = (
        f"'{stratification_col}' is eligible for stratified analysis: "
        f"{len(sufficient_strata)} of {len(strata)} stratum value(s) contain both variants "
        f"with at least {min_arm_size} observations per arm."
    )
    if sparse_values:
        reason += (
            f" {len(sparse_values)} stratum value(s) — {', '.join(sparse_values)} — were "
            "excluded from the combined estimate for having too few observations in one arm."
        )
    if missing_count:
        reason += (
            f" {missing_count} row(s) ({missing_proportion:.1%}) had a missing "
            f"{stratification_col} value and were excluded from stratification."
        )

    return StratificationEligibility(
        stratification_column=stratification_col,
        eligible=True,
        reason=reason,
        strata=strata,
        total_n=total_n,
        missing_count=missing_count,
        missing_proportion=missing_proportion,
        min_arm_size=min_arm_size,
        sparse_stratum_values=sparse_values,
    )


def _stratum_effect_and_variance(
    control_values: pd.Series, variant_values: pd.Series, metric_type: MetricType
) -> tuple[float, float]:
    """
    One stratum's (effect, variance) pair.

    Binary: effect = risk difference (variant rate - control rate);
    variance = the usual two-proportion normal-approximation variance
    p(1-p)/n summed across arms.

    Continuous: effect = mean difference; variance = the Welch-style
    sum of each arm's own sample-variance/n (no equal-variance
    assumption, matching hypothesis_tests.py's own Welch default).
    """
    n1, n2 = len(control_values), len(variant_values)
    if metric_type == MetricType.BINARY:
        p1 = float(control_values.mean())
        p2 = float(variant_values.mean())
        var = (p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2)
        return p2 - p1, var

    m1, m2 = float(control_values.mean()), float(variant_values.mean())
    v1 = float(control_values.var(ddof=1)) if n1 > 1 else 0.0
    v2 = float(variant_values.var(ddof=1)) if n2 > 1 else 0.0
    var = (v1 / n1 if n1 else 0.0) + (v2 / n2 if n2 else 0.0)
    return m2 - m1, var


def run_stratified_analysis(
    df: pd.DataFrame,
    variant_col: str,
    control_label,
    metric_col: str,
    metric_type: MetricType,
    stratification_col: str,
    metric_label: str,
    min_arm_size: int = _MIN_ARM_SIZE,
) -> tuple[StratificationEligibility, StratifiedEstimate | None]:
    """
    Run the eligibility check, then — only if eligible — compute the
    inverse-variance-weighted combined treatment effect across the
    sufficiently-sized strata. Returns (eligibility, estimate);
    `estimate` is None whenever `eligibility.eligible` is False.
    """
    eligibility = check_stratification_eligibility(
        df, variant_col, control_label, stratification_col, min_arm_size=min_arm_size
    )

    # Enrich each stratum with its own descriptive per-arm outcome rate
    # ("stratum-level estimates" for the report/UI) — purely additive
    # display information computed the same simple way regardless of
    # whether the stratum ends up used in the combined estimate below.
    # This does NOT touch the weighting/estimator math further down;
    # it only fills in `StratumSummary.control_outcome_rate` /
    # `variant_outcome_rate`, which existed in the schema but were
    # never populated.
    missing_mask = df[stratification_col].isna()
    working = df.loc[~missing_mask]
    enriched_strata = []
    for summary in eligibility.strata:
        if summary.has_both_variants:
            stratum_df = working[working[stratification_col].astype(str) == summary.stratum_value]
            control_vals = stratum_df.loc[stratum_df[variant_col] == control_label, metric_col]
            variant_vals = stratum_df.loc[stratum_df[variant_col] != control_label, metric_col]
            summary = summary.model_copy(
                update={
                    "control_outcome_rate": float(control_vals.mean()) if len(control_vals) else None,
                    "variant_outcome_rate": float(variant_vals.mean()) if len(variant_vals) else None,
                }
            )
        enriched_strata.append(summary)
    eligibility = eligibility.model_copy(update={"strata": enriched_strata})

    if not eligibility.eligible:
        return eligibility, None

    missing_mask = df[stratification_col].isna()
    working = df.loc[~missing_mask]

    weights: list[float] = []
    effects: list[float] = []
    strata_used = 0

    for summary in eligibility.strata:
        if not summary.sufficient:
            continue
        stratum_df = working[working[stratification_col].astype(str) == summary.stratum_value]
        control_values = stratum_df.loc[stratum_df[variant_col] == control_label, metric_col]
        variant_values = stratum_df.loc[stratum_df[variant_col] != control_label, metric_col]
        effect, var = _stratum_effect_and_variance(control_values, variant_values, metric_type)
        if var <= 0:
            # A stratum with exactly zero within-arm variance (e.g. every
            # observation identical) cannot contribute a finite weight —
            # exclude it rather than dividing by zero or fabricating an
            # infinite-precision vote for that stratum.
            continue
        weights.append(1.0 / var)
        effects.append(effect)
        strata_used += 1

    if strata_used == 0:
        eligibility = eligibility.model_copy(
            update={
                "eligible": False,
                "reason": (
                    eligibility.reason
                    + " No stratum produced a usable (non-zero-variance) within-stratum estimate."
                ),
                "ineligibility_reason": StratificationIneligibilityReason.ALL_STRATA_TOO_SPARSE,
            }
        )
        return eligibility, None

    weights_arr = np.array(weights)
    effects_arr = np.array(effects)
    combined_effect = float(np.sum(weights_arr * effects_arr) / np.sum(weights_arr))
    combined_var = float(1.0 / np.sum(weights_arr))
    se = float(np.sqrt(combined_var))

    z = combined_effect / se if se > 0 else 0.0
    p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
    ci_lower = combined_effect - _Z_95 * se
    ci_upper = combined_effect + _Z_95 * se

    estimate = StratifiedEstimate(
        method=(
            "Inverse-variance-weighted fixed-effect combination of within-stratum "
            f"{'risk differences' if metric_type == MetricType.BINARY else 'mean differences'} "
            f"across {strata_used} stratum/strata (Cochran 1954-style stratified analysis)."
        ),
        metric_label=metric_label,
        strata_used=strata_used,
        effect_estimate=combined_effect,
        standard_error=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        significant=bool(p_value < 0.05),
    )
    return eligibility, estimate


def build_diagnostic_stratification(
    df: pd.DataFrame,
    variant_col: str,
    control_label,
    metric_col: str,
    stratification_col: str,
    srm_alpha: float | None = None,
) -> DiagnosticStratification:
    """
    Descriptive-ONLY diagnostic breakdown for a stratification column,
    safe to compute even when the overall experiment FAILED the SRM
    validity gate — unlike `run_stratified_analysis` above (which
    estimates a CAUSAL combined treatment effect and must never run on
    an invalid experiment), this function never combines strata into a
    treatment-effect estimate, never computes a standard error/CI for a
    treatment effect, and never emits a significance verdict about
    "did the treatment work". Every number it returns is a plain
    descriptive fact: allocation counts, stratum sizes/proportions,
    per-arm outcome rates (means — reported for descriptive color, not
    as a comparison claim), missingness, and each stratum's OWN
    allocation-balance check (a diagnostic, not causal, p-value — see
    `StratumSRMConcentration`).

    Callers should resolve/validate `stratification_col` against the
    dataset's real columns BEFORE calling this (see
    `planner_strategy.py`'s natural-language column resolution) — this
    function has no eligibility gate of its own by design (diagnostics
    are shown "where statistically appropriate", not gated the same
    way causal `check_stratification_eligibility` is). If the column
    genuinely isn't in `df`, this returns a `DiagnosticStratification`
    with `strata=[]` so the caller can report that plainly rather than
    this function raising.
    """
    allocation_by_variant: dict[str, int] = {
        str(k): int(v) for k, v in df[variant_col].value_counts(dropna=False).items()
    }
    total_n = len(df)

    if stratification_col not in df.columns:
        return DiagnosticStratification(
            stratification_column=stratification_col,
            allocation_by_variant=allocation_by_variant,
            total_n=total_n,
            missing_count=0,
            missing_proportion=0.0,
            strata=[],
            srm_by_stratum=[],
        )

    missing_mask = df[stratification_col].isna()
    missing_count = int(missing_mask.sum())
    missing_proportion = (missing_count / total_n) if total_n else 0.0
    working = df.loc[~missing_mask]

    strata: list[StratumSummary] = []
    srm_by_stratum: list[StratumSRMConcentration] = []

    for value, group in working.groupby(stratification_col, dropna=True):
        control_mask = group[variant_col] == control_label
        control_n = int(control_mask.sum())
        variant_n = int(len(group) - control_n)
        has_both = control_n > 0 and variant_n > 0

        control_vals = group.loc[control_mask, metric_col]
        variant_vals = group.loc[~control_mask, metric_col]

        strata.append(
            StratumSummary(
                stratum_value=str(value),
                control_n=control_n,
                variant_n=variant_n,
                total_n=len(group),
                proportion_of_total=(len(group) / total_n) if total_n else 0.0,
                has_both_variants=has_both,
                # Diagnostic display only — this does NOT apply the
                # causal `_MIN_ARM_SIZE` threshold used by
                # `check_stratification_eligibility`; it simply mirrors
                # `has_both_variants` since no causal estimate is ever
                # produced from this object.
                sufficient=has_both,
                control_outcome_rate=float(control_vals.mean()) if len(control_vals) else None,
                variant_outcome_rate=float(variant_vals.mean()) if len(variant_vals) else None,
            )
        )

        if control_n + variant_n > 0:
            stratum_srm = check_srm(observed_control=control_n, observed_variant=variant_n, alpha=srm_alpha)
            srm_by_stratum.append(
                StratumSRMConcentration(
                    stratum_value=str(value),
                    observed_control=control_n,
                    observed_variant=variant_n,
                    p_value=stratum_srm.p_value,
                    srm_flagged=not stratum_srm.passed,
                )
            )

    return DiagnosticStratification(
        stratification_column=stratification_col,
        allocation_by_variant=allocation_by_variant,
        total_n=total_n,
        missing_count=missing_count,
        missing_proportion=missing_proportion,
        strata=strata,
        srm_by_stratum=srm_by_stratum,
    )
