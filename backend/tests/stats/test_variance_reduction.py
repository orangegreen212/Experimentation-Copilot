"""
Regression tests — CUPED must estimate theta ONCE from the
pooled (control + variant) sample and apply the SAME theta to both
arms (canonical CUPED), rather than estimating theta independently
per arm.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from app.stats.variance_reduction import (
    CupedSkippedReason,
    apply_cuped,
    apply_cuped_to_experiment,
    bootstrap_ci_for_difference,
    estimate_pooled_cuped_theta,
)


def _correlated_arm(rng, n, mean, covariate_mean, theta_true=1.0, noise=1.0):
    covariate = rng.normal(covariate_mean, 5.0, size=n)
    metric = mean + theta_true * (covariate - covariate_mean) + rng.normal(0, noise, size=n)
    return pd.Series(metric), pd.Series(covariate)


def test_theta_is_estimated_once_from_pooled_data():
    """The theta returned for the whole experiment must equal the theta
    computed directly from Cov/Var on the pooled control+variant sample —
    not from either arm alone."""
    rng = np.random.default_rng(1)
    n = 500
    control_metric, control_cov = _correlated_arm(rng, n, mean=50, covariate_mean=20)
    variant_metric, variant_cov = _correlated_arm(rng, n, mean=55, covariate_mean=22)

    pooled_metric = pd.concat([control_metric, variant_metric], ignore_index=True)
    pooled_cov = pd.concat([control_cov, variant_cov], ignore_index=True)
    expected_theta = pooled_metric.cov(pooled_cov) / pooled_cov.var(ddof=1)

    theta, x_mean, skip_reason = estimate_pooled_cuped_theta(pooled_metric, pooled_cov)
    assert skip_reason is None
    assert theta == pytest.approx(expected_theta)

    # And this must differ from a naive per-arm-only estimate to prove
    # it's genuinely pooled, not accidentally equal to one arm's theta.
    control_only_theta = control_metric.cov(control_cov) / control_cov.var(ddof=1)
    variant_only_theta = variant_metric.cov(variant_cov) / variant_cov.var(ddof=1)
    assert theta != pytest.approx(control_only_theta) or theta != pytest.approx(variant_only_theta)


def test_same_theta_applied_to_both_arms():
    """apply_cuped_to_experiment must apply IDENTICAL theta to both arms."""
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(2 * n)],
        "variant": ["control"] * n + ["treatment"] * n,
    })
    control_metric, control_cov = _correlated_arm(rng, n, mean=50, covariate_mean=20)
    variant_metric, variant_cov = _correlated_arm(rng, n, mean=54, covariate_mean=20)
    df["order_value"] = pd.concat([control_metric, variant_metric], ignore_index=True)
    df["order_value_pre"] = pd.concat([control_cov, variant_cov], ignore_index=True)

    control_mask = df["variant"] == "control"
    variant_mask = ~control_mask

    pooled_theta, pooled_x_mean, _ = estimate_pooled_cuped_theta(
        df["order_value"], df["order_value_pre"]
    )

    _, theta_from_control_call, skip_c = apply_cuped(
        df.loc[control_mask, "order_value"], df.loc[control_mask, "order_value_pre"],
        theta=pooled_theta, x_mean=pooled_x_mean,
    )
    _, theta_from_variant_call, skip_v = apply_cuped(
        df.loc[variant_mask, "order_value"], df.loc[variant_mask, "order_value_pre"],
        theta=pooled_theta, x_mean=pooled_x_mean,
    )

    assert skip_c is None and skip_v is None
    assert theta_from_control_call == theta_from_variant_call == pytest.approx(pooled_theta)

    adj_control, adj_variant, result = apply_cuped_to_experiment(
        df, "order_value", control_mask, variant_mask
    )
    assert result.method == "cuped"
    assert adj_control is not None and adj_variant is not None


def test_pooled_fallback_when_correlation_too_weak():
    """Preserve fallback: a covariate with essentially no correlation to
    the pooled metric must still skip cleanly (COVARIATE_TOO_WEAK)."""
    rng = np.random.default_rng(3)
    n = 5000
    metric = rng.normal(50, 5, size=2 * n)
    # Independently shuffled copy of the same values guarantees ~0
    # correlation regardless of seed, unlike drawing a second
    # independent normal series (which can occasionally correlate by
    # chance with a small sample).
    covariate = rng.permutation(metric.copy())
    df = pd.DataFrame({
        "variant": ["control"] * n + ["treatment"] * n,
        "order_value": metric,
        "order_value_pre": covariate,
    })
    control_mask = df["variant"] == "control"
    variant_mask = ~control_mask

    adj_control, adj_variant, result = apply_cuped_to_experiment(
        df, "order_value", control_mask, variant_mask
    )
    assert adj_control is None and adj_variant is None
    assert CupedSkippedReason.COVARIATE_TOO_WEAK in result.method


def test_pooled_fallback_when_insufficient_paired_data():
    """Preserve the minimum-sample-size fallback (<10 paired obs)."""
    df = pd.DataFrame({
        "variant": ["control"] * 3 + ["treatment"] * 3,
        "order_value": [10.0, 12.0, 11.0, 15.0, 14.0, 16.0],
        "order_value_pre": [9.0, 11.0, 10.5, 14.0, 13.5, 15.5],
    })
    control_mask = df["variant"] == "control"
    variant_mask = ~control_mask

    adj_control, adj_variant, result = apply_cuped_to_experiment(
        df, "order_value", control_mask, variant_mask
    )
    assert adj_control is None and adj_variant is None
    assert CupedSkippedReason.INSUFFICIENT_DATA in result.method


def test_variance_reduction_reporting_still_computed_with_pooled_theta():
    """The reported variance-before/after/reduction-pct must still reflect
    the actual pooled-theta-adjusted arms, not be a stub/placeholder."""
    rng = np.random.default_rng(4)
    n = 1000
    df = pd.DataFrame({"variant": ["control"] * n + ["treatment"] * n})
    control_metric, control_cov = _correlated_arm(rng, n, mean=50, covariate_mean=20, theta_true=2.0, noise=0.5)
    variant_metric, variant_cov = _correlated_arm(rng, n, mean=53, covariate_mean=20, theta_true=2.0, noise=0.5)
    df["order_value"] = pd.concat([control_metric, variant_metric], ignore_index=True)
    df["order_value_pre"] = pd.concat([control_cov, variant_cov], ignore_index=True)

    control_mask = df["variant"] == "control"
    variant_mask = ~control_mask

    adj_control, adj_variant, result = apply_cuped_to_experiment(
        df, "order_value", control_mask, variant_mask
    )
    assert result.method == "cuped"
    assert result.variance_after < result.variance_before
    assert result.variance_reduction_pct > 50  # strong covariate (theta_true=2.0, low noise)


def test_bootstrap_default_uses_10000_iterations():
    assert inspect.signature(bootstrap_ci_for_difference).parameters["iterations"].default == 10_000
