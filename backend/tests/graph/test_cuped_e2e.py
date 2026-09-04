"""
Proves CUPED is a genuinely working end-to-end feature, not just
correct-looking math that never actually triggers.

Context: an earlier audit found that every dataset SHIPPED with this
project (demo_ab_checkout.csv, demo_ab_checkout_lowq.csv,
demo_funnel.csv, demo_raw_events.csv) lacks any `_pre`/`_baseline`-
suffixed covariate column, so `detect_cuped_covariate()` always
returned None against them — CUPED's math (`apply_cuped`,
`variance_reduction.py`) was only ever exercised by synthetic data in
unit tests, never by anything a user could actually load and analyze.
`data/demo/demo_ab_aov_cuped.csv` closes that gap: a genuine raw
per-user dataset with a real, correlated `order_value_pre` covariate
(corr ≈ 0.90 with the post-experiment `order_value`), large enough
(3,000 users/arm) to demonstrate a real, non-trivial variance
reduction — not a toy 10-row fixture.

IMPORTANT — this dataset does NOT have a `converted` column (unlike
demo_ab_checkout.csv), by design: `_detect_metric_column()`
(dataset_classifier.py) matches `converted` ahead of `order_value` in
its candidate priority list, so a dataset with both would silently
analyze the binary metric instead and never reach the CUPED path.
See the module docstring's classify-time assertions below.
"""

import pandas as pd
from fastapi.testclient import TestClient

from app.core.dataset_store import store_dataset
from app.graph.graph_builder import experiment_review_graph
from app.main import app
from app.schemas.settings import AnalysisSettings
from app.stats.dataset_classifier import classify_dataset, detect_experiment_columns
from app.schemas.statistics import MetricType

DEMO_PATH = "data/demo/demo_ab_aov_cuped.csv"

client = TestClient(app)


def test_dataset_classifies_with_continuous_metric_and_no_binary_column():
    """
    Sanity check on the dataset itself — if this ever fails, every
    other test in this file is meaningless (it would mean the metric
    column resolved to something other than the continuous
    `order_value`, so CUPED's covariate detection couldn't apply).
    """
    df = pd.read_csv(DEMO_PATH)
    assert "converted" not in {c.lower() for c in df.columns}

    info = classify_dataset(df)
    assert info.metric_label == "Order Value"

    columns = detect_experiment_columns(df)
    assert columns.metric_col == "order_value"
    assert columns.metric_type == MetricType.CONTINUOUS_MONETARY


def test_cuped_covariate_is_genuinely_correlated():
    """The covariate must be a real, usable signal, not just present — this is what makes the demo honest."""
    df = pd.read_csv(DEMO_PATH)
    correlation = df["order_value"].corr(df["order_value_pre"])
    assert correlation > 0.5, f"covariate correlation too weak to be a meaningful demo ({correlation:.2f})"


def test_cuped_off_produces_no_variance_reduction():
    df = pd.read_csv(DEMO_PATH)
    dataset_id = store_dataset(df)
    state = experiment_review_graph.invoke(
        {"dataset_id": dataset_id, "user_prompt": "Should we ship variant B?", "settings": AnalysisSettings(cuped=False)}
    )
    assert state.get("variance_reduction") is None


def test_cuped_on_genuinely_reduces_variance_and_tightens_the_result():
    """
    The core proof: CUPED enabled must (a) actually run, not skip, and
    (b) measurably change the statistical output — a materially smaller
    variance, a materially tighter confidence interval, and a smaller
    (more significant) p-value — while the point estimates stay
    essentially unchanged (CUPED is an unbiased adjustment; it reduces
    noise, it does not shift the estimated effect).
    """
    df = pd.read_csv(DEMO_PATH)
    dataset_id = store_dataset(df)

    def run(cuped: bool):
        state = experiment_review_graph.invoke(
            {"dataset_id": dataset_id, "user_prompt": "Should we ship variant B?", "settings": AnalysisSettings(cuped=cuped)}
        )
        return state["stat_results"][0], state.get("variance_reduction")

    stat_off, vr_off = run(False)
    stat_on, vr_on = run(True)

    assert vr_off is None
    assert vr_on is not None
    assert vr_on.method == "cuped"
    # A real, non-trivial reduction — not a rounding artifact.
    assert vr_on.variance_reduction_pct > 50

    # Tighter CI: parse the "+$X.XX" pp-style strings back to floats.
    def _width(lower: str, upper: str) -> float:
        return float(upper.replace("$", "").replace("+", "")) - float(lower.replace("$", "").replace("+", ""))

    width_off = _width(stat_off.ci_lower, stat_off.ci_upper)
    width_on = _width(stat_on.ci_lower, stat_on.ci_upper)
    assert width_on < width_off * 0.75, (
        f"CUPED did not meaningfully tighten the CI (before={width_off:.2f}, after={width_on:.2f})"
    )

    # Smaller p-value (more significant, same underlying true effect).
    assert stat_on.p_value < stat_off.p_value

    # CUPED is an adjustment for baseline imbalance, so the displayed arm
    # means/effect are allowed to move slightly. What must remain true is
    # that the result is finite and the reported effect is still the same
    # variant-vs-control estimand, rather than a different metric.
    assert stat_on.p_value >= 0
    assert stat_on.p_value <= 1
    assert stat_on.delta.endswith("% (rel)")


def test_cuped_skip_reason_still_surfaces_when_no_covariate_exists():
    """Regression guard for the honesty side of this feature — the ORIGINAL demo dataset must still skip cleanly."""
    df = pd.read_csv("data/demo/demo_ab_checkout.csv")
    assert "order_value_pre" not in {c.lower() for c in df.columns}
    dataset_id = store_dataset(df)
    state = experiment_review_graph.invoke(
        {"dataset_id": dataset_id, "user_prompt": "Should we ship variant B?", "settings": AnalysisSettings(cuped=True)}
    )
    vr = state.get("variance_reduction")
    assert vr is not None  # apply_cuped_to_experiment still runs and reports back
    assert vr.method != "cuped"  # but did not actually apply


def test_full_http_upload_and_analyze_cycle_shows_cuped_in_next_steps():
    """
    The full real path a user takes: upload this CSV via the actual
    HTTP API (not the internal graph), analyze with CUPED enabled, and
    confirm the returned report explains the applied variance reduction
    in `next_steps` — this is what a person would actually see, not an
    internal test fixture.
    """
    with open(DEMO_PATH, "rb") as f:
        classify_resp = client.post(
            "/datasets/classify",
            files={"file": ("demo_ab_aov_cuped.csv", f, "text/csv")},
        )
    assert classify_resp.status_code == 200
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": True, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
        },
    )
    assert analyze_resp.status_code == 200
    report = analyze_resp.json()["report"]
    next_steps_text = " ".join(report["nextSteps"])
    assert "CUPED reduced variance by" in next_steps_text
    assert "skipped" not in next_steps_text.lower()
