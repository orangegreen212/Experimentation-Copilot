"""
Power analysis — computes the Minimum Detectable Effect (MDE) at the
observed sample size, and whether the observed effect had adequate
power. Pure statsmodels/scipy math, no LLM.

Two branches by metric type, since the effect-size definition differs:
  - BINARY: proportions -> statsmodels NormalIndPower on Cohen's h
  - CONTINUOUS: means -> statsmodels TTestIndPower on Cohen's d
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

from app.core.config import stats_thresholds
from app.schemas.statistics import MetricType, PowerAnalysisResult


def compute_power_analysis(
    control: pd.Series,
    variant: pd.Series,
    metric_type: MetricType,
    alpha: float | None = None,
    target_power: float | None = None,
) -> PowerAnalysisResult:
    alpha = stats_thresholds.significance_alpha if alpha is None else alpha
    target_power = stats_thresholds.target_power if target_power is None else target_power

    control_clean = control.dropna()
    variant_clean = variant.dropna()
    n_control, n_variant = len(control_clean), len(variant_clean)
    ratio = n_variant / n_control if n_control > 0 else 1.0

    if metric_type == MetricType.BINARY:
        return _binary_power(control_clean, variant_clean, n_control, ratio, alpha, target_power)
    return _continuous_power(control_clean, variant_clean, n_control, ratio, alpha, target_power)


def _binary_power(
    control: pd.Series, variant: pd.Series, n_control: int, ratio: float, alpha: float, target_power: float
) -> PowerAnalysisResult:
    p_c, p_v = control.mean(), variant.mean()
    observed_effect_size = abs(proportion_effectsize(p_v, p_c))

    analysis = NormalIndPower()
    achieved_power = float(
        analysis.power(effect_size=observed_effect_size, nobs1=n_control, alpha=alpha, ratio=ratio)
    ) if observed_effect_size > 0 else 0.0

    mde_effect_size = _solve_mde(analysis, n_control, ratio, alpha, target_power)
    mde_relative = _cohens_h_to_relative_proportion_change(mde_effect_size, p_c)
    required_n = _solve_required_n(analysis, observed_effect_size, ratio, alpha, target_power, fallback=n_control)

    return PowerAnalysisResult(
        minimum_detectable_effect_relative=mde_relative,
        required_sample_size=required_n,
        observed_sample_size=n_control + len(variant),
        achieved_power=min(max(achieved_power, 0.0), 1.0),
        alpha=alpha,
        is_sufficiently_powered=achieved_power >= target_power,
    )


def _continuous_power(
    control: pd.Series, variant: pd.Series, n_control: int, ratio: float, alpha: float, target_power: float
) -> PowerAnalysisResult:
    mean_c, mean_v = control.mean(), variant.mean()
    pooled_std = _pooled_std(control, variant)
    observed_effect_size = abs(mean_v - mean_c) / pooled_std if pooled_std > 0 else 0.0

    analysis = TTestIndPower()
    achieved_power = float(
        analysis.power(effect_size=observed_effect_size, nobs1=n_control, alpha=alpha, ratio=ratio)
    ) if observed_effect_size > 0 else 0.0
    # `TTestIndPower.power()` is numerically unstable (returns NaN) for
    # certain large-effect-size/large-df combinations — an overflow in
    # statsmodels' underlying noncentral-t evaluation, not a real "power
    # is undefined" case. This combination is reachable once CUPED
    # uses canonical pooled-theta variance reduction,
    # which can push the observed effect size into that region on
    # highly-correlated covariates. A NaN here always occurs at effect
    # sizes/sample sizes where power is unambiguously ~1.0 (see the
    # monotonic power curve just below the unstable region), so this is
    # a numerical-stability clamp, not a change to the power calculation
    # itself.
    if np.isnan(achieved_power):
        achieved_power = 1.0

    mde_effect_size = _solve_mde(analysis, n_control, ratio, alpha, target_power)
    mde_relative = (mde_effect_size * pooled_std / mean_c * 100) if mean_c != 0 else float("inf")
    required_n = _solve_required_n(analysis, observed_effect_size, ratio, alpha, target_power, fallback=n_control)

    return PowerAnalysisResult(
        minimum_detectable_effect_relative=mde_relative,
        required_sample_size=required_n,
        observed_sample_size=n_control + len(variant),
        achieved_power=min(max(achieved_power, 0.0), 1.0),
        alpha=alpha,
        is_sufficiently_powered=achieved_power >= target_power,
    )


def _solve_mde(analysis, n_control: int, ratio: float, alpha: float, target_power: float) -> float:
    """
    The smallest effect size this sample size (`n_control`, at the
    observed control/variant ratio) is powered to detect at
    `target_power`. Independent of the effect actually observed —
    this answers "what could we have detected," not "what did we
    detect."
    """
    try:
        return float(
            analysis.solve_power(nobs1=n_control, alpha=alpha, power=target_power, ratio=ratio, effect_size=None)
        )
    except Exception:
        return float("nan")


def _solve_required_n(analysis, observed_effect_size: float, ratio: float, alpha: float, target_power: float, fallback: int) -> int:
    """
    The sample size (per control arm) that WOULD be required to reach
    `target_power` for the effect size actually observed in this
    experiment. If the observed effect is ~0, this is undefined/huge —
    fall back to the current sample size rather than returning
    infinity, since the report just needs a sane number to display.
    """
    if observed_effect_size <= 1e-9:
        return fallback
    try:
        return int(
            np.ceil(analysis.solve_power(effect_size=observed_effect_size, alpha=alpha, power=target_power, ratio=ratio, nobs1=None))
        )
    except Exception:
        return fallback


def _pooled_std(control: pd.Series, variant: pd.Series) -> float:
    n_c, n_v = len(control), len(variant)
    var_c, var_v = control.var(ddof=1), variant.var(ddof=1)
    pooled_var = ((n_c - 1) * var_c + (n_v - 1) * var_v) / (n_c + n_v - 2) if (n_c + n_v) > 2 else 0.0
    return float(pooled_var**0.5)


def _cohens_h_to_relative_proportion_change(effect_size: float, baseline_p: float) -> float:
    """
    Approximate inverse of Cohen's h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1)).
    Solves for p2 given h and p1, returns (p2-p1)/p1 * 100.
    """
    if pd.isna(effect_size) or baseline_p <= 0 or baseline_p >= 1:
        return float("nan")
    phi1 = 2 * np.arcsin(np.sqrt(baseline_p))
    phi2 = phi1 + effect_size
    p2 = np.sin(phi2 / 2) ** 2
    p2 = min(max(p2, 0.0), 1.0)
    return float((p2 - baseline_p) / baseline_p * 100)


def format_mde(mde_relative: float) -> str:
    """
    `mde_relative` is always a RELATIVE percentage change (see
    `_cohens_h_to_relative_proportion_change`/the continuous-metric
    branch in `compute_power_analysis` — both compute a relative %,
    never percentage points). The word "(relative)" is included
    explicitly in the display string so it can never be misread as
    percentage points, especially next to a StatResult table that
    shows BOTH relative deltas ("-1.2% (rel)") and absolute
    percentage-point confidence intervals ("[-0.38pp, +0.09pp]") —
    without this label, "MDE: 2.8%" is genuinely ambiguous between the
    two.
    """
    if pd.isna(mde_relative) or np.isinf(mde_relative):
        return "N/A"
    return f"{abs(mde_relative):.1f}% (relative)"



def _format_power(power: float) -> str:
    if power >= 0.999:
        return ">99.9%"
    return f"{power * 100:.1f}%"

def format_sample_size_note(result: PowerAnalysisResult) -> str:
    """
    `result.required_sample_size` is solved from the OBSERVED effect
    size (see `_solve_required_n`'s call site in `_binary_power`/
    `_continuous_power`, passed `observed_effect_size` — never the MDE
    effect size) — i.e. "how many users would this specific observed
    effect need to reach target power," not "how many users would it
    take to reliably detect the (target) MDE." Those are two different
    numbers computed from two different effect sizes
    (`result.minimum_detectable_effect_relative` is the independent,
    sample-size-only MDE from `_solve_mde`, which never touches the
    observed effect). The wording below must say "observed effect,"
    never "MDE," to match what was actually calculated — conflating
    the two here was the original bug this docstring guards against.
    """
    if result.is_sufficiently_powered:
        return (
            f"{result.observed_sample_size:,} users observed — "
            f"achieved power {_format_power(result.achieved_power)} at \u03b1={result.alpha:g}"
        )
    return (
        f"{result.observed_sample_size:,} users observed — "
        f"UNDERPOWERED (achieved {_format_power(result.achieved_power)} power, "
        f"target was {stats_thresholds.target_power * 100:.0f}%); "
        f"~{result.required_sample_size:,} users/arm recommended to reliably detect "
        f"the observed effect (not the {abs(result.minimum_detectable_effect_relative):.1f}% relative MDE)"
    )


def plan_required_sample_size(
    baseline_rate: float,
    mde_relative_pct: float,
    metric_type: MetricType,
    baseline_std: float | None = None,
    alpha: float | None = None,
    target_power: float | None = None,
    ratio: float = 1.0,
    num_variants: int = 2,
) -> "SampleSizePlan":
    """
    A-PRIORI sample size planning — the inverse problem from
    `compute_power_analysis` above. That function asks "given the data
    I already collected, was it enough?" This function asks "before
    collecting anything, how much would I need?"

    Inputs are hypothetical (an assumed baseline + the smallest
    relative effect worth detecting), not observed data — this is the
    only function in this module that takes no pandas Series. Used by
    the pre-experiment "Create Experiment" planning screen, never by
    the post-hoc analysis pipeline.

    For CONTINUOUS metrics, `baseline_std` (the assumed standard
    deviation of the metric) is required — unlike the binary case,
    there is no way to derive a variance from a single baseline
    number for a continuous metric.
    """
    from app.schemas.hypothesis import SampleSizePlan  # local import: avoids a cycle (hypothesis.py -> statistics.py is the other direction)

    alpha = stats_thresholds.significance_alpha if alpha is None else alpha
    target_power = stats_thresholds.target_power if target_power is None else target_power

    if metric_type == MetricType.BINARY:
        p1 = baseline_rate
        p2 = min(max(p1 * (1 + mde_relative_pct / 100), 0.0), 1.0)
        effect_size = abs(proportion_effectsize(p2, p1))
        analysis = NormalIndPower()
    else:
        if not baseline_std or baseline_std <= 0:
            raise ValueError("baseline_std must be a positive number for continuous metrics (metric_type != BINARY).")
        delta = baseline_rate * (mde_relative_pct / 100)
        effect_size = abs(delta) / baseline_std
        analysis = TTestIndPower()

    if effect_size <= 1e-9:
        raise ValueError("The minimum detectable effect must be non-zero.")

    n_per_arm = int(np.ceil(
        analysis.solve_power(effect_size=effect_size, alpha=alpha, power=target_power, ratio=ratio, nobs1=None)
    ))
    total_n = n_per_arm * num_variants

    return SampleSizePlan(
        required_n_per_arm=n_per_arm,
        required_n_total=total_n,
        alpha=alpha,
        target_power=target_power,
        effect_size=float(effect_size),
    )
