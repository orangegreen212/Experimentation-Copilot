"""
Guardrail resolution — unit tests (guardrail root-cause fix).

Covers the deterministic, exact-match-only resolution of a user's
explicitly requested guardrail metrics against a dataset's actual
columns — the piece that was completely missing before this fix (see
the design audit: there was no field, no parser, and no matching step
anywhere in the pipeline for a user-specified guardrail).
"""

import pandas as pd
import pytest

from app.stats.dataset_classifier import (
    build_metric_column_map,
    detect_available_metrics,
    humanize_metric_label,
    infer_guardrail_direction,
    resolve_guardrail_metrics,
)


def _ab_testing_data_frame() -> pd.DataFrame:
    """
    Mirrors the shape of `AB Testing Data.csv` used throughout the
    design audit: user_id, timestamp, group, landing_page, converted,
    age, gender, location, session_duration, pages_visited,
    device_type, purchase_amount. No Revenue or Bounce Rate column.
    """
    return pd.DataFrame(
        {
            "user_id": [f"U{i}" for i in range(10)],
            "group": ["control"] * 5 + ["treatment"] * 5,
            "converted": [0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            "age": [25, 31, 40, 22, 35, 29, 41, 26, 38, 33],
            "session_duration": [3.2, 1.1, 4.5, 2.2, 3.9, 2.1, 5.0, 1.8, 3.3, 2.7],
            "pages_visited": [4, 2, 6, 3, 5, 3, 7, 2, 4, 3],
            "purchase_amount": [0.0, 12.5, 0.0, 9.99, 0.0, 20.0, 15.0, 0.0, 0.0, 30.0],
        }
    )


def test_available_metrics_match_the_design_audit_example():
    df = _ab_testing_data_frame()
    exclude = {"user_id", "group"}
    metrics = detect_available_metrics(df, exclude=exclude)
    # Order-independent — the design audit's Example 1 lists these five.
    assert set(metrics) >= {"Session Duration", "Pages Visited", "Purchase Amount"}


def test_requested_guardrails_never_silently_substitute_for_a_different_column():
    """
    Design audit Example 1: Revenue and Bounce Rate are requested but
    this dataset has no such columns — they must resolve to False, and
    critically must NOT silently resolve to Purchase Amount just
    because it's semantically related.
    """
    df = _ab_testing_data_frame()
    exclude = {"user_id", "group"}
    col_map = build_metric_column_map(df, exclude=exclude)

    resolutions = resolve_guardrail_metrics(
        ["Revenue", "Bounce Rate"],
        available_metrics=list(col_map.keys()),
        primary_metric_label="Conversion Rate",
    )

    by_name = {r.requested_name: r for r in resolutions}
    assert by_name["Revenue"].resolved is False
    assert by_name["Revenue"].resolved_metric_label is None
    assert by_name["Bounce Rate"].resolved is False
    assert by_name["Bounce Rate"].resolved_metric_label is None


def test_requested_guardrail_resolves_when_column_actually_exists():
    df = _ab_testing_data_frame()
    exclude = {"user_id", "group"}
    col_map = build_metric_column_map(df, exclude=exclude)

    resolutions = resolve_guardrail_metrics(
        ["Purchase Amount", "Session Duration"],
        available_metrics=list(col_map.keys()),
        primary_metric_label="Conversion Rate",
    )

    by_name = {r.requested_name: r for r in resolutions}
    assert by_name["Purchase Amount"].resolved is True
    assert by_name["Purchase Amount"].resolved_metric_label == "Purchase Amount"
    assert col_map[by_name["Purchase Amount"].resolved_metric_label] == "purchase_amount"
    assert by_name["Session Duration"].resolved is True


def test_matching_is_case_and_whitespace_insensitive_but_never_fuzzy():
    df = _ab_testing_data_frame()
    exclude = {"user_id", "group"}
    col_map = build_metric_column_map(df, exclude=exclude)

    resolutions = resolve_guardrail_metrics(
        ["  purchase amount  ", "PURCHASE AMOUNT"],
        available_metrics=list(col_map.keys()),
        primary_metric_label="Conversion Rate",
    )
    assert all(r.resolved for r in resolutions)

    # "Purchase" alone must NOT fuzzy/substring-match "Purchase Amount".
    partial = resolve_guardrail_metrics(
        ["Purchase"],
        available_metrics=list(col_map.keys()),
        primary_metric_label="Conversion Rate",
    )
    assert partial[0].resolved is False


def test_primary_metric_is_never_eligible_as_its_own_guardrail():
    df = _ab_testing_data_frame()
    exclude = {"user_id", "group"}
    col_map = build_metric_column_map(df, exclude=exclude)

    resolutions = resolve_guardrail_metrics(
        ["Conversion Rate"],
        available_metrics=list(col_map.keys()) + ["Conversion Rate"],
        primary_metric_label="Conversion Rate",
    )
    assert resolutions[0].resolved is False


def test_duplicate_requests_are_deduplicated_once():
    df = _ab_testing_data_frame()
    exclude = {"user_id", "group"}
    col_map = build_metric_column_map(df, exclude=exclude)

    resolutions = resolve_guardrail_metrics(
        ["Purchase Amount", "purchase amount", "Purchase Amount"],
        available_metrics=list(col_map.keys()),
        primary_metric_label="Conversion Rate",
    )
    assert len(resolutions) == 1


@pytest.mark.parametrize(
    "column,expected_higher_is_better",
    [
        ("bounce_rate", False),
        ("bounced", False),
        ("churn_rate", False),
        ("early_cancel_14d", False),  # doc4: "cancel" keyword coverage fix
        ("error_rate", False),
        ("purchase_amount", True),
        ("revenue", True),
        ("revenue_gbp", True),
        ("session_duration", True),
        ("conversion_rate", True),
    ],
)
def test_guardrail_direction_heuristic(column, expected_higher_is_better):
    assert infer_guardrail_direction(column) is expected_higher_is_better


def test_humanize_metric_label_matches_design_audit_example_purchase_amount_is_not_revenue():
    # This is the crux of "never invent equivalence" — Purchase Amount's
    # own humanized label must never collide with "Revenue".
    assert humanize_metric_label("purchase_amount") == "Purchase Amount"
    assert humanize_metric_label("purchase_amount") != "Revenue"
