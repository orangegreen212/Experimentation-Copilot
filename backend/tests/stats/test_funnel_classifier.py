import pandas as pd
import pytest

from app.stats.funnel_classifier import detect_funnel_columns, infer_step_order


class TestDetectFunnelColumns:
    def test_detects_real_demo_funnel_csv(self):
        df = pd.read_csv("data/demo/demo_funnel.csv")
        cols = detect_funnel_columns(df)
        assert cols is not None
        assert cols.user_col == "user_id"
        assert cols.event_col == "event"
        assert cols.timestamp_col == "timestamp"
        assert cols.group_col == "experiment_group"

    def test_returns_none_for_aggregated_ab_dataset(self):
        """A normal one-row-per-user A/B dataset has no event column — must not be misdetected as a funnel."""
        df = pd.read_csv("data/demo/demo_ab_checkout.csv")
        cols = detect_funnel_columns(df)
        assert cols is None

    def test_returns_none_without_user_column(self):
        df = pd.DataFrame({"event": ["Visit", "Signup"], "timestamp": ["2026-01-01", "2026-01-02"]})
        assert detect_funnel_columns(df) is None

    def test_returns_none_with_only_one_distinct_event(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "event": ["Visit", "Visit"],
            "timestamp": ["2026-01-01", "2026-01-02"],
        })
        assert detect_funnel_columns(df) is None

    def test_returns_none_without_timestamp_column(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "event": ["Visit", "Signup"],
        })
        assert detect_funnel_columns(df) is None

    def test_group_col_is_none_when_no_variant_like_column_present(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "event": ["Visit", "Signup"],
            "timestamp": ["2026-01-01", "2026-01-02"],
        })
        cols = detect_funnel_columns(df)
        assert cols is not None
        assert cols.group_col is None


class TestInferStepOrder:
    def test_orders_events_by_median_timestamp(self):
        df = pd.DataFrame({
            "event": ["Purchase", "Visit", "Signup", "Visit", "Purchase", "Signup"],
            "timestamp": [
                "2026-01-10", "2026-01-01", "2026-01-05",
                "2026-01-02", "2026-01-11", "2026-01-06",
            ],
        })
        order = infer_step_order(df, "event", "timestamp")
        assert order == ["Visit", "Signup", "Purchase"]

    def test_matches_known_step_order_on_real_demo_data(self):
        df = pd.read_csv("data/demo/demo_funnel.csv")
        order = infer_step_order(df, "event", "timestamp")
        assert order == ["Visit", "Signup", "Trial", "Purchase"]
