"""
Variance reduction — Stage 4: CUPED and Bootstrap.

Both are OPTIONAL, controlled by `Settings.cuped` / `Settings.bootstrap`
(schemas/settings.py) — neither is ever required for the pipeline to
produce a valid report. Pure deterministic math, no LLM.

CUPED (Controlled-experiment Using Pre-Experiment Data):
  Requires a pre-experiment covariate correlated with the metric
  (e.g. the same metric measured before the experiment started).
  DECISION (confirmed): CUPED auto-detects whether a usable covariate
  exists in the dataset. If not, it is skipped gracefully — the report
  explains why — rather than failing or silently ignoring the toggle.

Bootstrap:
  An alternative, distribution-free method for estimating the
  confidence interval on the difference between arms, via percentile
  resampling. Useful as a cross-check alongside the parametric CI from
  compute_stat_result(), particularly for skewed metrics. Also
  deterministic given a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Column name patterns that plausibly represent a pre-experiment value
# of the SAME metric. Matched against `{metric_col}_{suffix}` and
# `{prefix}_{metric_col}` combinations, case-insensitive.
_COVARIATE_SUFFIXES = ["pre", "prev", "before", "baseline", "historical"]

# Explicit pre-experiment column names used by common user-level experiment
# datasets. These are checked in addition to the existing metric_{suffix} /
# {suffix}_metric convention, preserving all legacy detection behavior.
_EXPLICIT_COVARIATE_NAMES = [
    "pre_metric",
    "pre_experiment_metric",
    "pre_experiment_value",
    "pre_metric_value",
    "baseline",
    "baseline_metric",
    "baseline_value",
    "covariate",
    "cuped_covariate",
]

_MIN_CORRELATION_FOR_USEFUL_COVARIATE = 0.05  # below this, CUPED isn't worth applying


class CupedSkippedReason:
    NO_COVARIATE_COLUMN = "no pre-experiment covariate column found in the dataset"
    COVARIATE_TOO_WEAK = "pre-experiment covariate is too weakly correlated with the metric to help"
    INSUFFICIENT_DATA = "insufficient non-null paired data to compute CUPED adjustment"


def detect_cuped_covariate(df: pd.DataFrame, metric_column: str) -> str | None:
    """
    Heuristically find a pre-experiment covariate column for the given
    metric, by explicit pre-experiment names (e.g. `pre_metric`) or by the
metric-relative pattern (e.g. metric_column='order_value' matches
'order_value_pre', 'pre_order_value', 'order_value_baseline', ...).

    Returns None if nothing plausible is found — the caller must treat
    this as "CUPED not available," not as an error.
    """
    columns_lower = {c.lower(): c for c in df.columns}
    metric_lower = metric_column.lower()

    # First support explicit generic CUPED names such as `pre_metric`.
    # These are intentionally checked by exact name only, so unrelated
    # numeric columns are never guessed as a covariate.
    for candidate in _EXPLICIT_COVARIATE_NAMES:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]

    # Preserve the original metric-relative naming convention unchanged.
    for suffix in _COVARIATE_SUFFIXES:
        for candidate in (f"{metric_lower}_{suffix}", f"{suffix}_{metric_lower}"):
            if candidate in columns_lower:
                return columns_lower[candidate]
    return None


def estimate_pooled_cuped_theta(
    pooled_metric: pd.Series,
    pooled_covariate: pd.Series,
) -> tuple[float | None, float | None, str | None]:
    """
    Canonical CUPED theta estimation: theta = Cov(X, Y) / Var(X),
    estimated ONCE from the pooled (control + variant) sample — never
    per-arm. The pooled mean of X is returned alongside it, since the
    CUPED adjustment `Y - theta * (X - mean(X))` must also center on
    the pooled mean for both arms to remain on the same footing.

    Returns (theta, x_mean, skip_reason). If skip_reason is not None,
    theta and x_mean are both None and the caller must fall back to
    the raw (unadjusted) metric for both arms.
    """
    paired = pd.DataFrame({"y": pooled_metric, "x": pooled_covariate}).dropna()
    if len(paired) < 10:
        return None, None, CupedSkippedReason.INSUFFICIENT_DATA

    covariance = paired["y"].cov(paired["x"])
    variance_x = paired["x"].var(ddof=1)
    if variance_x == 0:
        return None, None, CupedSkippedReason.COVARIATE_TOO_WEAK

    correlation = paired["y"].corr(paired["x"])
    if pd.isna(correlation) or abs(correlation) < _MIN_CORRELATION_FOR_USEFUL_COVARIATE:
        return None, None, CupedSkippedReason.COVARIATE_TOO_WEAK

    theta = float(covariance / variance_x)
    x_mean = float(paired["x"].mean())
    return theta, x_mean, None


def apply_cuped(
    metric: pd.Series,
    covariate: pd.Series,
    theta: float | None = None,
    x_mean: float | None = None,
) -> tuple[pd.Series, float, str | None]:
    """
    Apply the CUPED adjustment to a single arm's metric values, given
    the matching pre-experiment covariate values (same index/order).

    Returns (adjusted_metric, theta, skip_reason). If skip_reason is
    not None, `adjusted_metric` is simply the original `metric`
    unchanged and `theta` is 0.0 — callers should treat this as "CUPED
    did not apply" and fall back to the unadjusted metric.

    Formula: Y_cuped = Y - theta * (X - mean(X))

    Canonical CUPED requires theta = Cov(Y, X) / Var(X) to be
    estimated ONCE from the POOLED (control + variant) sample and then
    applied identically to both arms — never estimated independently
    per arm, which would let each arm's own noise leak back into its
    own adjustment. Callers orchestrating a full experiment (see
    `apply_cuped_to_experiment`) MUST supply a pooled `theta`/`x_mean`
    via `estimate_pooled_cuped_theta`. This function falls back to
    estimating theta/x_mean from `metric`/`covariate` alone ONLY when
    neither is supplied, purely to preserve this function's direct
    single-arm callability (e.g. ad hoc scripts, existing unit tests
    of this function in isolation) — the experiment-level orchestrator
    never takes that path.
    """
    paired = pd.DataFrame({"y": metric, "x": covariate}).dropna()
    if len(paired) < 10:
        return metric, 0.0, CupedSkippedReason.INSUFFICIENT_DATA

    if theta is None or x_mean is None:
        covariance = paired["y"].cov(paired["x"])
        variance_x = paired["x"].var(ddof=1)
        if variance_x == 0:
            return metric, 0.0, CupedSkippedReason.COVARIATE_TOO_WEAK

        correlation = paired["y"].corr(paired["x"])
        if pd.isna(correlation) or abs(correlation) < _MIN_CORRELATION_FOR_USEFUL_COVARIATE:
            return metric, 0.0, CupedSkippedReason.COVARIATE_TOO_WEAK

        theta = float(covariance / variance_x)
        x_mean = float(paired["x"].mean())

    adjusted = metric.copy()
    valid_idx = paired.index
    adjusted.loc[valid_idx] = paired["y"] - theta * (paired["x"] - x_mean)

    return adjusted, float(theta), None


def apply_cuped_to_experiment(
    df: pd.DataFrame,
    metric_column: str,
    control_mask: pd.Series,
    variant_mask: pd.Series,
) -> tuple[pd.Series | None, pd.Series | None, "VarianceReductionResult | None"]:
    """
    Orchestrates CUPED for a full control/variant pair: detects the
    covariate, applies the adjustment to each arm, and reports the
    variance reduction achieved (or the reason it was skipped).

    Returns (adjusted_control, adjusted_variant, result_or_none). When
    skipped, both adjusted series are None and the result carries the
    skip reason in its `method` field.
    """
    from app.schemas.statistics import VarianceReductionResult  # local import avoids a cycle at module load

    covariate_column = detect_cuped_covariate(df, metric_column)
    if covariate_column is None:
        return None, None, _skip_result("cuped", CupedSkippedReason.NO_COVARIATE_COLUMN)

    control_metric = df.loc[control_mask, metric_column]
    control_covariate = df.loc[control_mask, covariate_column]
    variant_metric = df.loc[variant_mask, metric_column]
    variant_covariate = df.loc[variant_mask, covariate_column]

    original_variance = pd.concat([control_metric, variant_metric]).var(ddof=1)

    # Canonical CUPED: theta = Cov(X, Y) / Var(X) is estimated ONCE from
    # the pooled (control + variant) sample, then applied identically
    # to both arms — never estimated independently per arm.
    pooled_metric = pd.concat([control_metric, variant_metric])
    pooled_covariate = pd.concat([control_covariate, variant_covariate])
    theta, x_mean, pooled_skip_reason = estimate_pooled_cuped_theta(pooled_metric, pooled_covariate)

    if pooled_skip_reason is not None:
        return None, None, _skip_result("cuped", pooled_skip_reason)

    adj_control, _theta_c, skip_c = apply_cuped(control_metric, control_covariate, theta=theta, x_mean=x_mean)
    adj_variant, _theta_v, skip_v = apply_cuped(variant_metric, variant_covariate, theta=theta, x_mean=x_mean)

    if skip_c is not None or skip_v is not None:
        # A per-arm skip here means one arm alone had too few paired
        # (non-null) observations to apply the shared theta, even
        # though the pooled sample cleared the threshold.
        reason = skip_c or skip_v
        return None, None, _skip_result("cuped", reason)

    adjusted_variance = pd.concat([adj_control, adj_variant]).var(ddof=1)
    reduction_pct = (
        100 * (original_variance - adjusted_variance) / original_variance if original_variance > 0 else 0.0
    )

    result = VarianceReductionResult(
        method="cuped",
        variance_before=float(original_variance),
        variance_after=float(adjusted_variance),
        variance_reduction_pct=float(reduction_pct),
    )
    return adj_control, adj_variant, result


def _skip_result(method: str, reason: str) -> "VarianceReductionResult | None":
    from app.schemas.statistics import VarianceReductionResult

    return VarianceReductionResult(
        method=f"{method}_skipped: {reason}",
        variance_before=0.0,
        variance_after=0.0,
        variance_reduction_pct=0.0,
    )


def bootstrap_ci_for_difference(
    control: pd.Series,
    variant: pd.Series,
    alpha: float = 0.05,
    statistic: str = "mean",
    iterations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for the difference (variant - control) in
    the chosen statistic ('mean' or 'median'). Distribution-free —
    doesn't assume normality — used as the CI method when
    `Settings.bootstrap` is enabled, as an alternative/cross-check to
    the parametric CI from compute_stat_result().

    Deterministic given a fixed seed, per project requirement that all
    statistics be reproducible.
    """
    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be 'mean' or 'median'")

    control_arr = control.dropna().to_numpy()
    variant_arr = variant.dropna().to_numpy()
    stat_fn = np.mean if statistic == "mean" else np.median

    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations)
    for i in range(iterations):
        boot_c = rng.choice(control_arr, size=len(control_arr), replace=True)
        boot_v = rng.choice(variant_arr, size=len(variant_arr), replace=True)
        diffs[i] = stat_fn(boot_v) - stat_fn(boot_c)

    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper
