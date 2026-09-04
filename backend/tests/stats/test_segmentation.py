import numpy as np
import pandas as pd
import pytest

from app.schemas.segmentation import SegmentSampleSizeStatus, SegmentSkipReason
from app.schemas.statistics import MetricType
from app.stats.dataset_classifier import classify_column_treatment_timing
from app.stats.segmentation import (
    MAX_CARDINALITY,
    MIN_SEGMENT_ARM_SIZE,
    discover_segmentation_dimensions,
    run_segmentation_analysis,
)


def _make_binary_dataset(n_per_segment=50, seed=42):
    """
    control/variant binary metric, with a `device` dimension where
    'mobile' has a real effect and 'desktop' does not, plus a
    high-cardinality `session_id` column and a mostly-null `notes`
    column, to exercise every guardrail in one dataset.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for variant, base_rate in [("control", 0.10), ("treatment", 0.10)]:
        for device, lift in [("mobile", 0.25 if variant == "treatment" else 0.0), ("desktop", 0.0)]:
            rate = base_rate + lift
            metric = rng.binomial(1, rate, n_per_segment)
            for i in range(n_per_segment):
                rows.append(
                    {
                        "user_id": f"{variant}_{device}_{i}",
                        "variant": variant,
                        "device": device,
                        "converted": int(metric[i]),
                        "session_id": f"sess_{variant}_{device}_{i}",  # near-unique -> high cardinality
                        "notes": None,  # entirely missing -> excessive missing
                    }
                )
    df = pd.DataFrame(rows)
    return df


def _build_binary_df(segments):
    """
    Deterministic (no RNG) binary dataset builder: `segments` is a list
    of `(control_n, control_successes, variant_n, variant_successes,
    segment_label)` tuples. Using exact counts rather than random draws
    keeps these heterogeneity tests fully reproducible — the point is
    to exercise the interaction-test CONTRACT, not to fit a particular
    random seed.
    """
    rows = []
    for control_n, control_succ, variant_n, variant_succ, label in segments:
        for i in range(control_n):
            rows.append({"user_id": f"c_{label}_{i}", "variant": "control", "device": label, "converted": 1 if i < control_succ else 0})
        for i in range(variant_n):
            rows.append({"user_id": f"t_{label}_{i}", "variant": "treatment", "device": label, "converted": 1 if i < variant_succ else 0})
    return pd.DataFrame(rows)


# --- Phase 2: heterogeneity is a SEPARATE fact from within-segment significance ---

def test_same_effect_in_every_segment_is_not_heterogeneous():
    """
    Both segments have the IDENTICAL true effect (10% -> 25%) and both
    are individually significant. There is no reason to believe the
    treatment effect differs across segments, so the interaction test
    must find no heterogeneity — even though `has_reliable_segment_effect`
    is True for both.
    """
    df = _build_binary_df([
        (800, 80, 800, 200, "mobile"),
        (800, 80, 800, 200, "desktop"),
    ])
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    device_dim = next(d for d in result.dimension_results if d.dimension == "device")
    assert set(device_dim.reliable_segment_values) == {"mobile", "desktop"}
    assert device_dim.has_reliable_segment_effect is True
    assert device_dim.has_heterogeneous_effect is False
    assert device_dim.heterogeneity_p_value > 0.5


def test_genuinely_different_treatment_effects_are_flagged_heterogeneous():
    """
    Segment A has a huge real lift (10% -> 40%); segment B has none
    (10% -> 10%). The treatment effect itself differs across segments
    — this is exactly what the interaction test exists to detect.
    """
    df = _build_binary_df([
        (500, 50, 500, 200, "A"),
        (500, 50, 500, 50, "B"),
    ])
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    device_dim = next(d for d in result.dimension_results if d.dimension == "device")
    assert device_dim.has_heterogeneous_effect is True
    assert device_dim.heterogeneity_p_value < 0.05


def test_one_significant_one_not_does_not_automatically_imply_heterogeneity():
    """
    The architectural bug this Phase fixes: `has_heterogeneous_effect`
    used to be `len(reliable) > 0`, so ANY mix of significant/
    non-significant segments looked "heterogeneous". Here both segments
    share the SAME true effect (10% -> 25%) — one is well-powered
    (n=800/arm, significant) and one is underpowered (n=32/arm, not
    individually significant) purely because of sample size, not
    because the effect actually differs. The interaction test must
    NOT report heterogeneity just because one segment happened to
    reach significance and the other didn't.
    """
    df = _build_binary_df([
        (800, 80, 800, 200, "big"),
        (32, 3, 32, 8, "small"),
    ])
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    device_dim = next(d for d in result.dimension_results if d.dimension == "device")
    # Mixed within-segment reliability — exactly the case the old logic got wrong.
    assert device_dim.reliable_segment_values == ["big"]
    assert device_dim.has_reliable_segment_effect is True
    # But NOT heterogeneous: the true effect is the same in both segments.
    assert device_dim.has_heterogeneous_effect is False
    assert device_dim.heterogeneity_p_value > 0.5


def test_insufficient_sample_segment_excluded_from_heterogeneity_test():
    """
    Segments 'A' and 'B' have the identical real effect (homogeneous).
    Segment 'C' has only 5 users per arm — below MIN_SEGMENT_ARM_SIZE —
    with an extreme, wild observed swing (0% -> 100%) that would look
    like strong "heterogeneity" if it were ever included in the
    interaction test. It must be excluded entirely: the test is run
    only across A and B, and stays non-significant.
    """
    df = _build_binary_df([
        (800, 80, 800, 200, "A"),
        (800, 80, 800, 200, "B"),
        (5, 0, 5, 5, "C"),  # far below MIN_SEGMENT_ARM_SIZE — must be excluded
    ])
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    device_dim = next(d for d in result.dimension_results if d.dimension == "device")
    c_effect = next(e for e in device_dim.segment_effects if e.segment_value == "C")
    assert c_effect.sample_size_status == SegmentSampleSizeStatus.INSUFFICIENT
    assert c_effect.stat_result is None
    # Heterogeneity test ran only across the 2 sufficiently-sampled segments.
    assert device_dim.has_heterogeneous_effect is False
    assert "2 segments" in device_dim.heterogeneity_test_method


def test_fewer_than_two_testable_segments_reports_heterogeneity_not_assessed():
    """
    When fewer than 2 segments have sufficient sample, there is nothing
    to compare — heterogeneity must be reported as False (not detected)
    with no p-value, never fabricated.
    """
    df = _make_binary_dataset(n_per_segment=5)  # below MIN_SEGMENT_ARM_SIZE for both segments
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    device_dim = next(d for d in result.dimension_results if d.dimension == "device")
    assert device_dim.has_heterogeneous_effect is False
    assert device_dim.heterogeneity_p_value is None




def test_discovers_usable_low_cardinality_column():
    df = _make_binary_dataset()
    usable, skipped = discover_segmentation_dimensions(df, "user_id", "variant", "converted")
    assert "device" in usable


def test_skips_high_cardinality_column():
    df = _make_binary_dataset()
    usable, skipped = discover_segmentation_dimensions(df, "user_id", "variant", "converted")
    assert "session_id" not in usable
    reasons = {s.column: s.reason for s in skipped}
    assert reasons["session_id"] == SegmentSkipReason.HIGH_CARDINALITY


def test_skips_excessively_missing_column():
    df = _make_binary_dataset()
    usable, skipped = discover_segmentation_dimensions(df, "user_id", "variant", "converted")
    assert "notes" not in usable
    reasons = {s.column: s.reason for s in skipped}
    assert reasons["notes"] == SegmentSkipReason.EXCESSIVE_MISSING_VALUES


def test_skips_single_valued_column():
    df = _make_binary_dataset()
    df["constant_col"] = "same_value"
    usable, skipped = discover_segmentation_dimensions(df, "user_id", "variant", "converted")
    assert "constant_col" not in usable
    reasons = {s.column: s.reason for s in skipped}
    assert reasons["constant_col"] == SegmentSkipReason.TOO_FEW_DISTINCT_VALUES


def test_excludes_reserved_columns():
    df = _make_binary_dataset()
    usable, _ = discover_segmentation_dimensions(df, "user_id", "variant", "converted")
    assert "user_id" not in usable
    assert "variant" not in usable
    assert "converted" not in usable


def test_cardinality_boundary_is_inclusive():
    """A column with exactly MAX_CARDINALITY distinct values is still usable."""
    df = _make_binary_dataset()
    df["bucket"] = (np.arange(len(df)) % MAX_CARDINALITY).astype(str)
    usable, _ = discover_segmentation_dimensions(df, "user_id", "variant", "converted")
    assert "bucket" in usable


# --- run_segmentation_analysis: end-to-end guardrails ------------------------

def test_valid_segmentation_runs_and_flags_reliable_segment_effect():
    df = _make_binary_dataset(n_per_segment=200)  # large enough to detect the mobile lift
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    assert result.ran is True
    assert "device" in result.usable_dimensions
    device_dim = next(d for d in result.dimension_results if d.dimension == "device")
    mobile_effect = next(e for e in device_dim.segment_effects if e.segment_value == "mobile")
    assert mobile_effect.sample_size_status == SegmentSampleSizeStatus.SUFFICIENT
    assert mobile_effect.stat_result is not None


def test_insufficient_segment_size_produces_no_test():
    df = _make_binary_dataset(n_per_segment=5)  # below MIN_SEGMENT_ARM_SIZE
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    device_dim = next(d for d in result.dimension_results if d.dimension == "device")
    for effect in device_dim.segment_effects:
        assert effect.sample_size_status == SegmentSampleSizeStatus.INSUFFICIENT
        assert effect.stat_result is None
        assert effect.skip_detail is not None


def test_high_cardinality_dimension_excluded_from_results():
    df = _make_binary_dataset()
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    dims_analyzed = {d.dimension for d in result.dimension_results}
    assert "session_id" not in dims_analyzed
    skipped_cols = {s.column for s in result.skipped_dimensions}
    assert "session_id" in skipped_cols


def test_missing_values_column_excluded_from_results():
    df = _make_binary_dataset()
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    dims_analyzed = {d.dimension for d in result.dimension_results}
    assert "notes" not in dims_analyzed


def test_segment_level_control_treatment_comparison_uses_correct_arms():
    df = _make_binary_dataset(n_per_segment=200)
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    device_dim = next(d for d in result.dimension_results if d.dimension == "device")
    for effect in device_dim.segment_effects:
        assert effect.control_n == 200
        assert effect.variant_n == 200


def test_statistically_significant_segment_effect_is_flagged_reliable():
    df = _make_binary_dataset(n_per_segment=300, seed=7)
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    device_dim = next(d for d in result.dimension_results if d.dimension == "device")
    # mobile has a real +25pp lift baked into the fixture — with n=300/arm
    # this should survive Holm correction across the 2 segment tests.
    assert "mobile" in device_dim.reliable_segment_values
    assert device_dim.has_reliable_segment_effect is True


def test_non_significant_segment_difference_not_flagged_reliable():
    rng = np.random.default_rng(1)
    rows = []
    for variant in ["control", "treatment"]:
        for device in ["mobile", "desktop"]:
            metric = rng.binomial(1, 0.10, 150)  # identical rate everywhere, no real effect
            for i in range(150):
                rows.append({"user_id": f"{variant}_{device}_{i}", "variant": variant, "device": device, "converted": int(metric[i])})
    df = pd.DataFrame(rows)
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    device_dim = next(d for d in result.dimension_results if d.dimension == "device")
    assert device_dim.reliable_segment_values == []
    assert device_dim.has_reliable_segment_effect is False


def test_multiple_segments_all_analyzed():
    df = _make_binary_dataset(n_per_segment=100)
    df["country"] = np.where(np.arange(len(df)) % 2 == 0, "US", "UK")
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    dims_analyzed = {d.dimension for d in result.dimension_results}
    assert {"device", "country"} <= dims_analyzed


def test_no_usable_dimensions_reports_ran_false():
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(60)],
            "variant": ["control"] * 30 + ["treatment"] * 30,
            "converted": [0, 1] * 30,
        }
    )
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    assert result.ran is False
    assert result.reason  # always a human-readable explanation
    assert result.dimension_results == []


def test_min_segment_size_guardrail_value_is_reported():
    df = _make_binary_dataset()
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="converted", metric_type=MetricType.BINARY,
    )
    assert result.min_segment_size == MIN_SEGMENT_ARM_SIZE


def test_post_treatment_columns_excluded_from_segmentation_dimensions():
    """
    Regression test: `discover_segmentation_dimensions` previously never
    called `classify_column_treatment_timing` at all (despite the
    module docstring claiming this guardrail exists), so a real
    post-treatment outcome like 'visit' — present but not chosen as the
    primary metric_col — silently leaked through as a usable
    segmentation dimension. Segmenting a treatment-effect analysis on a
    post-treatment variable can fabricate or mask a real effect, so it
    must be excluded, with the specific POST_TREATMENT_VARIABLE reason
    — not coincidentally excluded (or not excluded at all) for an
    unrelated reason like cardinality.
    """
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n)],
        "variant": ["control"] * (n // 2) + ["treatment"] * (n // 2),
        "conversion": rng.integers(0, 2, size=n),
        "visit": rng.integers(0, 2, size=n),  # post-treatment, binary -> would pass cardinality
        "recency": rng.integers(1, 30, size=n),  # pre-treatment, safe
    })
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="conversion", metric_type=MetricType.BINARY,
    )
    assert "visit" not in result.usable_dimensions
    visit_skip = next(s for s in result.skipped_dimensions if s.column == "visit")
    assert visit_skip.reason == SegmentSkipReason.POST_TREATMENT_VARIABLE
    assert "recency" in result.usable_dimensions or any(
        s.column == "recency" for s in result.skipped_dimensions
    )


def test_add_to_cart_classified_as_post_treatment():
    """
    Regression test: 'add_to_cart' is a per-session behavioral outcome
    (recorded during/after the treated session), exactly like 'visit'
    or 'bounced' above, and must be excluded from segmentation with
    the specific POST_TREATMENT_VARIABLE reason.
    """
    assert classify_column_treatment_timing("add_to_cart") == "post_treatment"

    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n)],
        "variant": ["control"] * (n // 2) + ["treatment"] * (n // 2),
        "conversion": rng.integers(0, 2, size=n),
        "add_to_cart": rng.integers(0, 2, size=n),  # post-treatment, binary -> would pass cardinality
        "device": rng.choice(["mobile", "desktop"], size=n),  # pre-treatment, safe
    })
    result = run_segmentation_analysis(
        df, user_col="user_id", variant_col="variant", control_label="control",
        metric_col="conversion", metric_type=MetricType.BINARY,
    )
    assert "add_to_cart" not in result.usable_dimensions
    cart_skip = next(s for s in result.skipped_dimensions if s.column == "add_to_cart")
    assert cart_skip.reason == SegmentSkipReason.POST_TREATMENT_VARIABLE
    assert "device" in result.usable_dimensions
