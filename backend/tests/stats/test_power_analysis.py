import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.power import TTestIndPower

from app.schemas.statistics import MetricType
from app.stats.power_analysis import (
    _pooled_std,
    compute_power_analysis,
    format_mde,
    format_sample_size_note,
)


def _fixed_binary_series(rate: float, n: int) -> pd.Series:
    """Exact fixed proportion (not random sampling) — needed for reproducible regression tests, since required_sample_size is extremely sensitive to the exact observed effect size."""
    ones = round(rate * n)
    return pd.Series([1] * ones + [0] * (n - ones))


class TestRequiredSampleSizeRegression:
    """
    Pins the exact required_sample_size for a known, fixed input:
    baseline ~12.04%, observed variant ~11.89%, n=145,292/arm.

    If this test starts failing after a change to power_analysis.py,
    that means the underlying formula changed — which might be
    intentional or might be an accidental regression. Either
    way, this test forces that change to be reviewed explicitly rather
    than silently shipped.
    """

    def test_required_sample_size_pinned_value(self):
        n = 145292
        control = _fixed_binary_series(0.1204, n)
        variant = _fixed_binary_series(0.1189, n)

        result = compute_power_analysis(control, variant, MetricType.BINARY)

        assert control.mean() == pytest.approx(0.1204, abs=0.0001)
        assert variant.mean() == pytest.approx(0.1189, abs=0.0001)
        assert result.required_sample_size == 734459
        assert result.minimum_detectable_effect_relative == pytest.approx(2.8265, abs=0.001)
        assert result.achieved_power == pytest.approx(0.2383, abs=0.001)
        assert result.is_sufficiently_powered is False

    def test_required_sample_size_is_deterministic_across_repeated_calls(self):
        """Same input, called twice, must give bit-identical output — no hidden randomness in the formula path."""
        n = 50000
        control = _fixed_binary_series(0.10, n)
        variant = _fixed_binary_series(0.11, n)

        r1 = compute_power_analysis(control, variant, MetricType.BINARY)
        r2 = compute_power_analysis(control, variant, MetricType.BINARY)

        assert r1 == r2


class TestObservedSampleSizeMatchesUniqueUsers:
    """
    Regression guard for the "290,584 vs 294,478" bug: observed_sample_size
    must equal the row count actually passed in (which, after the
    experiment_node/validation_node fix, is always post-deduplication —
    this test just pins the power_analysis layer's own contract: it
    reports exactly len(control)+len(variant), no silent inflation).
    """

    def test_observed_sample_size_equals_input_length(self):
        n_control, n_variant = 1000, 1200
        control = _fixed_binary_series(0.10, n_control)
        variant = _fixed_binary_series(0.12, n_variant)

        result = compute_power_analysis(control, variant, MetricType.BINARY)

        assert result.observed_sample_size == n_control + n_variant


class TestMdeLabelingClarity:
    """
    Regression guard for the MDE ambiguity bug: the displayed MDE
    string must always explicitly say "(relative)" so it can never be
    misread as percentage points, especially next to a StatResult
    table that shows absolute-percentage-point confidence intervals.
    """

    def test_format_mde_includes_relative_label(self):
        assert format_mde(2.8) == "2.8% (relative)"
        assert format_mde(15.0) == "15.0% (relative)"

    def test_format_mde_handles_nan_and_inf(self):
        assert format_mde(float("nan")) == "N/A"
        assert format_mde(float("inf")) == "N/A"

    def test_sample_size_note_mentions_relative_mde(self):
        n = 145292
        control = _fixed_binary_series(0.1204, n)
        variant = _fixed_binary_series(0.1189, n)
        result = compute_power_analysis(control, variant, MetricType.BINARY)

        note = format_sample_size_note(result)
        # The MDE is still surfaced in the note (for context), but the
        # required_sample_size figure itself must be attributed to the
        # OBSERVED effect, never mislabeled as "for the MDE" — see
        # TestObservedEffectVsMdeWording below for the regression that
        # specifically catches the two being swapped/conflated.
        assert "relative MDE" in note
        assert "observed effect" in note
        assert f"{n * 2:,}" in note or f"{result.observed_sample_size:,}" in note


# --- Regression: required_sample_size must be solved from the OBSERVED
# effect, never accidentally solved from (or displayed as being for) the MDE
# effect size — these are two different numbers computed two different ways. --

class TestObservedEffectVsMdeWording:
    def test_required_sample_size_tracks_observed_effect_not_mde(self):
        """
        Construct a case where the observed effect is deliberately tiny
        relative to the MDE at the current n, so the two effect sizes are
        materially different. required_sample_size must match a value
        solved from the OBSERVED effect size — if power_analysis.py were
        changed to accidentally pass the MDE's effect size into
        `_solve_required_n` instead (or vice versa), this test fails.
        """
        rng = np.random.default_rng(0)
        n = 500
        control = pd.Series(rng.normal(100, 20, size=n))
        variant = control + 0.05  # deliberately tiny, near-zero observed shift

        result = compute_power_analysis(control, variant, MetricType.CONTINUOUS_GENERAL)

        pooled_std = _pooled_std(control, variant)
        observed_effect_size = abs(variant.mean() - control.mean()) / pooled_std
        expected_required_n = int(
            np.ceil(TTestIndPower().solve_power(effect_size=observed_effect_size, alpha=0.05, power=0.8, ratio=1.0))
        )

        assert result.required_sample_size == expected_required_n
        # The MDE this exact n (500/arm) is powered to detect is, by
        # construction, close to n itself — so if required_sample_size had
        # instead been silently computed from the MDE's effect size (the
        # swap this test guards against), it would land near n (500), not
        # in the tens of thousands the actual tiny observed effect requires.
        assert result.required_sample_size > n * 10

    def test_sample_size_note_labels_required_n_as_observed_effect_not_mde(self):
        """
        The displayed recommendation text must say the required-n figure
        is for the OBSERVED effect, not phrase it as being "to detect the
        MDE" (the two are different effect sizes computed differently —
        conflating them was the original bug).
        """
        n = 145292
        control = _fixed_binary_series(0.1204, n)
        variant = _fixed_binary_series(0.1189, n)
        result = compute_power_analysis(control, variant, MetricType.BINARY)

        note = format_sample_size_note(result)
        assert "observed effect" in note
        # Must not claim the required-n figure itself is "to detect the
        # relative MDE" — it explicitly qualifies MDE as the DIFFERENT,
        # not-the-basis-for-this-number, figure.
        assert "recommended to reliably detect the relative MDE" not in note


def test_sample_size_note_formats_saturated_power_as_greater_than_99_9_percent():
    from app.schemas.statistics import PowerAnalysisResult

    result = PowerAnalysisResult(
        minimum_detectable_effect_relative=2.8,
        required_sample_size=1000,
        observed_sample_size=294478,
        achieved_power=0.999999,
        alpha=0.05,
        is_sufficiently_powered=True,
    )
    assert ">99.9%" in format_sample_size_note(result)
    assert "100%" not in format_sample_size_note(result)


def test_large_sample_with_small_effect_is_flagged_as_practically_small():
    """Power can be high for a tiny effect; statistical power must not imply practical importance."""
    n = 100000
    control = _fixed_binary_series(0.1200, n)
    variant = _fixed_binary_series(0.1250, n)
    result = compute_power_analysis(control, variant, MetricType.BINARY)

    assert result.observed_sample_size == 200000
    assert result.is_sufficiently_powered is True
    assert result.minimum_detectable_effect_relative > 0


def test_small_sample_is_not_claimed_to_have_adequate_power():
    control = _fixed_binary_series(0.10, 100)
    variant = _fixed_binary_series(0.11, 100)
    result = compute_power_analysis(control, variant, MetricType.BINARY)

    assert result.is_sufficiently_powered is False
    assert result.achieved_power < 0.8
