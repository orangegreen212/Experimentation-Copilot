import pandas as pd
import pytest

from app.schemas.dataset import DatasetType
from app.stats.dataset_classifier import (
    DatasetClassificationError,
    classify_dataset,
)

DEMO_DIR = "data/demo"


def test_high_quality_demo_classified_as_raw_user_level():
    """
    This dataset is `user_id,variant,converted,order_value` with one
    row per user_id — that is raw, individual-level experiment data,
    not a pre-aggregated summary table; see `_quantitative_vote` /
    `_structural_vote` docstrings.
    """
    df = pd.read_csv(f"{DEMO_DIR}/demo_ab_checkout.csv")
    info = classify_dataset(df)

    assert info.type == DatasetType.RAW_USER_LEVEL
    assert info.variants == 2
    assert info.users == 12400
    assert info.metric_label == "Conversion Rate"


def test_low_quality_demo_classified_as_raw_user_level():
    df = pd.read_csv(f"{DEMO_DIR}/demo_ab_checkout_lowq.csv")
    info = classify_dataset(df)

    assert info.type == DatasetType.RAW_USER_LEVEL
    assert info.variants == 2
    assert info.users == 980
    assert info.metric_label == "Conversion Rate"


def test_raw_event_level_demo_classified_as_raw():
    df = pd.read_csv(f"{DEMO_DIR}/demo_raw_events.csv")
    info = classify_dataset(df)

    assert info.type == DatasetType.RAW_EVENT_LEVEL
    assert info.variants == 2
    assert info.users == 500
    # metric_label falls back to "Unknown Metric" — raw event logs have
    # no per-user outcome column by design; this is expected, not a bug.
    assert info.metric_label == "Unknown Metric"


def test_reported_raw_user_level_bug_regression():
    """
    Exact regression test for the reported live bug: a raw user-level
    A/B test export (user_id, timestamp, group, landing_page,
    converted — one row per user) was being classified as "Aggregated
    A/B Test Data — 290,585 users, 2 variants". It must classify as
    RAW_USER_LEVEL instead.
    """
    n_users = 290585
    user_ids = list(range(n_users))
    groups = ["control" if i % 2 == 0 else "treatment" for i in range(n_users)]
    pages = ["old_page" if g == "control" else "new_page" for g in groups]
    converted = [1 if i % 10 == 0 else 0 for i in range(n_users)]
    df = pd.DataFrame({
        "user_id": user_ids,
        "timestamp": pd.date_range("2026-01-01", periods=n_users, freq="s"),
        "group": groups,
        "landing_page": pages,
        "converted": converted,
    })
    info = classify_dataset(df)
    assert info.type == DatasetType.RAW_USER_LEVEL
    assert info.variants == 2
    assert info.users == n_users
    assert info.metric_label == "Conversion Rate"


def test_empty_dataframe_raises():
    df = pd.DataFrame()
    with pytest.raises(DatasetClassificationError):
        classify_dataset(df)


def test_no_user_column_classified_unknown():
    df = pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})
    info = classify_dataset(df)
    assert info.type == DatasetType.UNKNOWN
    assert info.users == 3  # falls back to row count


def test_ambiguous_structural_signal_falls_back_to_quantitative():
    # Has BOTH event-log columns AND outcome columns (structural vote = None),
    # and rows-per-user ratio is 1 (one row per user) — since a real
    # user_col is present, that ratio means RAW_USER_LEVEL, never
    # aggregated (aggregation always discards the per-unit identifier).
    # structural is ambiguous (None) so quantitative alone decides.
    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "variant": ["control", "variant", "control"],
        "event_name": ["purchase", "purchase", "purchase"],
        "converted": [1, 0, 1],
    })
    info = classify_dataset(df)
    # structural: has both event_cols and outcome_cols -> None (ambiguous)
    # quantitative: 1 row per user, user_col present -> RAW_USER_LEVEL
    # only one vote present -> use it
    assert info.type == DatasetType.RAW_USER_LEVEL


def test_conflicting_votes_yield_unknown():
    # Structural says RAW (has timestamp, no outcome col),
    # but there's exactly 1 row per user -> quantitative says AGGREGATED.
    # This is a genuine conflict -> UNKNOWN.
    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "variant": ["control", "variant", "control", "variant"],
        "timestamp": pd.to_datetime(["2026-01-01"] * 4),
    })
    info = classify_dataset(df)
    assert info.type == DatasetType.UNKNOWN


def test_variant_column_case_insensitive_and_alt_names():
    df = pd.DataFrame({
        "User_ID": ["u1", "u2", "u3", "u4"],
        "Group": ["control", "control", "test", "test"],
        "converted": [1, 0, 1, 0],
    })
    info = classify_dataset(df)
    assert info.variants == 2
    assert info.users == 4


def test_count_duplicate_user_rows_no_duplicates():
    from app.stats.dataset_classifier import count_duplicate_user_rows

    df = pd.DataFrame({"user_id": ["u1", "u2", "u3"], "converted": [1, 0, 1]})
    assert count_duplicate_user_rows(df, "user_id") == 0


def test_count_duplicate_user_rows_counts_extra_rows_not_affected_users():
    from app.stats.dataset_classifier import count_duplicate_user_rows

    # u1 appears 3 times total -> 2 EXTRA rows beyond the first
    df = pd.DataFrame({"user_id": ["u1", "u1", "u1", "u2", "u3"], "converted": [1, 1, 0, 0, 1]})
    assert count_duplicate_user_rows(df, "user_id") == 2


def test_deduplicate_by_user_keeps_first_occurrence():
    from app.stats.dataset_classifier import deduplicate_by_user

    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"],
        "converted": [1, 0, 1],  # u1's first row says converted=1
    })
    result = deduplicate_by_user(df, "user_id")
    assert len(result) == 2
    assert result[result["user_id"] == "u1"]["converted"].iloc[0] == 1


def test_deduplicate_by_user_reproduces_the_reported_discrepancy():
    """
    Regression test for the exact live bug report: a dataset with
    294,478 total rows but only 290,584 unique users (3,894 duplicate
    rows) must deduplicate down to exactly 290,584 rows — matching
    DatasetInfo.users, which was already nunique()-based.
    """
    from app.stats.dataset_classifier import count_duplicate_user_rows, deduplicate_by_user

    n_unique = 290584
    n_duplicate_extra = 3894
    user_ids = [f"u{i}" for i in range(n_unique)] + [f"u{i}" for i in range(n_duplicate_extra)]
    df = pd.DataFrame({"user_id": user_ids, "converted": [0] * len(user_ids)})

    assert len(df) == n_unique + n_duplicate_extra  # 294,478
    assert count_duplicate_user_rows(df, "user_id") == n_duplicate_extra

    deduped = deduplicate_by_user(df, "user_id")
    assert len(deduped) == n_unique  # 290,584 — matches the classifier's user count


class TestAnalyzeDuplicateUsers:
    """
    Regression coverage for the duplicate-conflict severity rule:
    same-variant duplicates are harmless (dedup and move on), but a
    user assigned to MULTIPLE variants is a severe randomization-
    integrity issue, and same-variant-different-metric is a lesser
    but still explicitly disclosed issue.
    """

    def test_harmless_duplicate_same_variant_same_metric(self):
        from app.stats.dataset_classifier import analyze_duplicate_users

        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "variant": ["control", "control", "variant"],
            "converted": [1, 1, 0],
        })
        result = analyze_duplicate_users(df, "user_id", "variant", "converted")
        assert result.duplicate_row_count == 1
        assert result.conflicting_variant_users == 0
        assert result.conflicting_metric_users == 0
        assert result.has_severe_conflict is False

    def test_severe_conflict_user_in_two_variants(self):
        from app.stats.dataset_classifier import analyze_duplicate_users

        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "variant": ["control", "variant", "variant"],  # u1 in BOTH arms
            "converted": [1, 0, 0],
        })
        result = analyze_duplicate_users(df, "user_id", "variant", "converted")
        assert result.conflicting_variant_users == 1
        assert result.has_severe_conflict is True

    def test_same_variant_conflicting_metric_is_disclosed_but_not_severe(self):
        from app.stats.dataset_classifier import analyze_duplicate_users

        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "variant": ["control", "control", "variant"],
            "converted": [1, 0, 0],  # u1: same variant, conflicting metric value
        })
        result = analyze_duplicate_users(df, "user_id", "variant", "converted")
        assert result.conflicting_variant_users == 0
        assert result.conflicting_metric_users == 1
        assert result.has_severe_conflict is False

    def test_no_duplicates_at_all(self):
        from app.stats.dataset_classifier import analyze_duplicate_users

        df = pd.DataFrame({
            "user_id": ["u1", "u2", "u3"],
            "variant": ["control", "control", "variant"],
            "converted": [1, 0, 1],
        })
        result = analyze_duplicate_users(df, "user_id", "variant", "converted")
        assert result.duplicate_row_count == 0
        assert result.conflicting_variant_users == 0
        assert result.conflicting_metric_users == 0


def test_preferred_metric_selects_revenue_when_dataset_has_multiple_metrics():
    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "variant": ["control", "variant", "control", "variant"],
        "converted": [0, 1, 0, 1],
        "revenue": [0.0, 20.0, 0.0, 30.0],
    })
    info = classify_dataset(df, preferred_metric="Please analyze revenue")
    assert info.metric_label == "Revenue"

    from app.stats.dataset_classifier import detect_experiment_columns
    columns = detect_experiment_columns(df, preferred_metric="Please analyze revenue")
    assert columns.metric_col == "revenue"


def test_multiple_metrics_keep_existing_default_when_prompt_does_not_choose_one():
    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "variant": ["control", "variant", "control", "variant"],
        "converted": [0, 1, 0, 1],
        "revenue": [0.0, 20.0, 0.0, 30.0],
    })
    info = classify_dataset(df)
    assert info.metric_label == "Conversion Rate"


def test_aggregated_ab_without_user_id_is_recognized_but_inference_is_honestly_rejected():
    """Capability honesty. The dataset SHAPE (variant | users |
    outcome, no user_id) must be correctly recognized as
    AGGREGATED_AB_TEST with the correct summed denominator, but the
    current experiment-analysis pipeline requires individual-level
    observations, so downstream inference must be explicitly and
    honestly rejected — never a misleading "no user_id" error implying
    the dataset is malformed, and the project must NOT invent a new
    aggregated statistical engine to work around this."""
    from app.stats.dataset_classifier import detect_experiment_columns

    df = pd.DataFrame({
        "variant": ["control", "A", "B", "C", "D", "E"],
        "users": [33340, 33330, 33330, 33330, 33330, 33340],
        "click_rate": [0.10, 0.11, 0.105, 0.12, 0.09, 0.115],
    })
    info = classify_dataset(df)
    assert info.type == DatasetType.AGGREGATED_AB_TEST
    assert info.variants == 6
    assert info.users == 200000  # sum of the 'users' column, not row count (6)

    with pytest.raises(DatasetClassificationError) as exc_info:
        detect_experiment_columns(df)

    message = str(exc_info.value)
    assert "recognized" in message.lower()
    assert "individual-level" in message.lower()
    assert "6 arms" in message
    assert "200,000" in message
    # Must not read as a malformed-data error.
    assert "no user_id" not in message.lower()


def test_aggregated_ab_with_conversions_column_matches_audit_example():
    """The exact `variant | users | conversions` shape from the audit."""
    df = pd.DataFrame({
        "variant": ["A", "B"],
        "users": [50000, 50000],
        "conversions": [4250, 4500],
    })
    info = classify_dataset(df)
    assert info.type == DatasetType.AGGREGATED_AB_TEST
    assert info.users == 100000
    assert info.variants == 2


def test_aggregated_ab_requires_denominator_without_user_id():
    from app.stats.dataset_classifier import detect_experiment_columns, DatasetClassificationError

    df = pd.DataFrame({
        "variant": ["control", "A"],
        "click_rate": [0.10, 0.12],
    })
    with pytest.raises(DatasetClassificationError, match="sample-size column"):
        detect_experiment_columns(df)


# --- Case A / Case B failure-mode regression tests (targeted fix) --------


def test_case_a_unit_level_without_id_is_accepted_but_blocked_on_single_arm():
    """
    Dataset 3 shape: click/group/session_time/... with no user_id-like
    column, but one row per experimental unit. Per the unit-level
    classification rules, this must NOT be rejected merely for lacking
    an explicit id — a synthetic row-level unit identifier is assigned
    instead. It's still correctly blocked here, but for an unrelated
    reason: only one treatment arm is present, so no comparison is
    possible. The failure reason must reflect that distinction, not a
    generic/unexplained failure, and must never claim a real user_id
    was found.
    """
    from app.stats.dataset_classifier import detect_experiment_columns

    df = pd.DataFrame({
        "click": [1, 0, 0, 1],
        "group": ["exp", "exp", "exp", "exp"],
        "session_time": [0.04036, 1.63957, 2.96171, 2.78454],
        "device_type": ["mobile", "mobile", "mobile", "desktop"],
    })
    with pytest.raises(DatasetClassificationError) as exc_info:
        detect_experiment_columns(df)

    message = str(exc_info.value)
    assert "one treatment arm" in message.lower()
    assert "user_id" not in df.columns
    # The synthetic row-level unit id must be clearly synthetic, never
    # presented as if a real customer/user identifier had been found.
    assert "__experiment_unit_row_id__" in message


def test_case_b_single_arm_gives_explicit_reason_distinct_from_case_a():
    """
    A dataset WITH a real experiment-unit id but only one distinct
    group value (no control/variant comparison possible) must raise a
    DIFFERENT, more specific reason than the no-unit case (Case A),
    and must never invent a second/control arm.
    """
    from app.stats.dataset_classifier import detect_experiment_columns

    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "group": ["exp", "exp", "exp", "exp"],
        "click": [1, 0, 0, 1],
    })
    with pytest.raises(DatasetClassificationError) as exc_info:
        detect_experiment_columns(df)

    message = str(exc_info.value)
    assert "one treatment arm" in message.lower()
    assert "user_id" in message  # confirms the unit WAS identified, unlike Case A
    assert "control" not in df["group"].unique().tolist()  # no invented control label


def test_case_a_and_case_b_reasons_are_distinct():
    from app.stats.dataset_classifier import detect_experiment_columns

    no_unit_df = pd.DataFrame({"click": [1, 0], "group": ["exp", "exp"]})
    one_arm_df = pd.DataFrame({"user_id": ["u1", "u2"], "group": ["exp", "exp"], "click": [1, 0]})

    with pytest.raises(DatasetClassificationError) as exc_a:
        detect_experiment_columns(no_unit_df)
    with pytest.raises(DatasetClassificationError) as exc_b:
        detect_experiment_columns(one_arm_df)

    assert str(exc_a.value) != str(exc_b.value)


def test_two_arms_still_resolve_successfully_unchanged():
    """Regression guard: a normal two-arm dataset must still resolve columns exactly as before."""
    from app.stats.dataset_classifier import detect_experiment_columns

    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "variant": ["control", "control", "treatment", "treatment"],
        "converted": [1, 0, 1, 0],
    })
    columns = detect_experiment_columns(df)
    assert columns.user_col == "user_id"
    assert columns.variant_col == "variant"
    assert columns.metric_col == "converted"


# --- describe_dataset_structure (Dataset 2 wording fix) -------------------


def test_describe_dataset_structure_unchanged_when_type_is_known():
    """Any dataset that already classifies cleanly must show the exact same label as before."""
    from app.stats.dataset_classifier import describe_dataset_structure

    df = pd.read_csv(f"{DEMO_DIR}/demo_ab_checkout.csv")
    info = classify_dataset(df)
    assert info.type == DatasetType.RAW_USER_LEVEL
    assert describe_dataset_structure(info, columns_resolved=True) == info.type.value
    assert describe_dataset_structure(info, columns_resolved=False) == info.type.value


def test_describe_dataset_structure_unchanged_when_columns_not_resolved():
    """Dataset 3-style case: UNKNOWN type AND no resolvable columns keeps the plain UNKNOWN label."""
    from app.stats.dataset_classifier import describe_dataset_structure

    df = pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})
    info = classify_dataset(df)
    assert info.type == DatasetType.UNKNOWN
    assert describe_dataset_structure(info, columns_resolved=False) == "Unknown / Unsupported Format"


def test_describe_dataset_structure_explains_unknown_when_columns_did_resolve():
    """
    Dataset 2-style case: the structural/quantitative vote lands on
    UNKNOWN, but detect_experiment_columns actually resolved a real
    user/variant/metric structure. The description must no longer say
    plain "Unknown / Unsupported Format" — but `info.type` itself must
    stay UNKNOWN (no change to the classifier's own semantics/contract).
    """
    from app.stats.dataset_classifier import describe_dataset_structure

    # Structural vote: has an outcome col (revenue), no event/timestamp
    # col, and a real user_col -> votes RAW_USER_LEVEL. Quantitative
    # vote: >=1.5 rows per user -> votes RAW_EVENT_LEVEL. The two votes
    # genuinely disagree -> UNKNOWN (same mechanism as
    # test_conflicting_votes_yield_unknown), even though
    # user_id/variant/revenue are all real, resolvable columns.
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
        "variant": ["control", "control", "variant", "variant", "control", "control"],
        "revenue": [10.0, 1.0, 12.0, 2.0, 9.0, 1.5],
    })
    info = classify_dataset(df)
    assert info.type == DatasetType.UNKNOWN  # classifier semantics unchanged

    description = describe_dataset_structure(info, columns_resolved=True)
    assert description != "Unknown / Unsupported Format"
    assert "revenue" in description.lower()


def test_hillstrom_shaped_segment_column_is_recognized_as_variant():
    """
    Regression test: `_VARIANT_CANDIDATES` previously omitted "segment"
    entirely, so a real Hillstrom-shaped CRM dataset (segment/visit/
    conversion/spend, no customer id column) was rejected outright by
    detect_experiment_columns with "no experiment-unit identifier...",
    even though it is exactly the unit-level-without-id case the
    classifier is supposed to accept. Must now resolve variant_col to
    'segment' and succeed as an implicit-row unit-level dataset.
    """
    from app.stats.dataset_classifier import MetricType, detect_experiment_columns

    df = pd.DataFrame({
        "recency": [10, 6, 7, 9],
        "history": [142.44, 329.08, 180.0, 675.1],
        "mens": [1, 1, 0, 0],
        "womens": [0, 1, 1, 1],
        "zip_code": ["Surburban", "Rural", "Urban", "Rural"],
        "newbie": [0, 1, 0, 1],
        "channel": ["Phone", "Web", "Web", "Multichannel"],
        "segment": ["Womens E-Mail", "No E-Mail", "Mens E-Mail", "Womens E-Mail"],
        "visit": [0, 0, 1, 0],
        "conversion": [0, 0, 1, 0],
        "spend": [0.0, 0.0, 29.99, 0.0],
    })
    columns = detect_experiment_columns(df)
    assert columns.variant_col == "segment"
    assert columns.metric_col == "conversion"
    assert columns.metric_type == MetricType.BINARY
    assert columns.is_implicit_unit is True
    assert columns.user_col == "__experiment_unit_row_id__"


def test_attach_implicit_unit_id_matches_detect_experiment_columns_contract():
    """
    Regression test: `detect_experiment_columns` builds its synthetic
    unit-id column on an internal `working_df` copy that it never
    returns — only the column NAME (`ExperimentColumns.user_col`).
    Any caller that persists/reuses the dataset under that name (e.g.
    `classifier_node` storing it for every downstream node to
    re-fetch) needs `attach_implicit_unit_id` to build the identical
    column, or every downstream `df[columns.user_col]` lookup raises
    KeyError. This must hold end-to-end for a real unit-level dataset.
    """
    from app.stats.dataset_classifier import attach_implicit_unit_id, detect_experiment_columns

    df = pd.DataFrame({
        "segment": ["control", "treatment", "control", "treatment"],
        "conversion": [0, 1, 0, 1],
        "recency": [1, 2, 3, 4],
    })
    columns = detect_experiment_columns(df)
    assert columns.is_implicit_unit is True

    materialized_df, implicit_col = attach_implicit_unit_id(df)
    assert implicit_col == columns.user_col
    assert implicit_col in materialized_df.columns
    assert materialized_df[implicit_col].nunique() == len(materialized_df)
    assert materialized_df[implicit_col].is_unique


def test_describe_unit_count_never_says_users_for_implicit_row():
    """
    Regression test: report_generator previously always rendered
    "{dataset.users:,} users" regardless of unit_identifier, so a
    genuinely unit-level CRM dataset with no customer_id column (e.g.
    Hillstrom's `segment` column) would be reported as "64,000 users"
    even though no user/customer identifier exists. describe_unit_count
    is the single source of truth report_generator now calls for every
    such phrase — must never say "users" for IMPLICIT_ROW.
    """
    from app.schemas.dataset import DatasetInfo, DatasetType, UnitIdentifierType
    from app.stats.dataset_classifier import describe_unit_count

    implicit = DatasetInfo(
        type=DatasetType.UNKNOWN,
        variants=3,
        users=64000,
        metric_label="Conversion Rate",
        metric_selection_reason="test",
        unit_identifier=UnitIdentifierType.IMPLICIT_ROW,
    )
    phrase = describe_unit_count(implicit)
    assert "users" not in phrase
    assert "64,000" in phrase
    assert "experimental units" in phrase

    explicit = DatasetInfo(
        type=DatasetType.UNKNOWN,
        variants=2,
        users=500,
        metric_label="Conversion Rate",
        metric_selection_reason="test",
        unit_identifier=UnitIdentifierType.EXPLICIT_COLUMN,
    )
    assert describe_unit_count(explicit) == "500 users"


# --- resolve_control_label: holdout/no-treatment label recognition ---------
#
# Regression coverage for the fix: a CRM/marketing "No E-Mail"-style
# holdout arm must resolve as control regardless of where it happens
# to sit in the variant column's value order — a fix that only worked
# for ONE ordering would just be re-testing insertion order again,
# not the actual semantics.

@pytest.mark.parametrize(
    "order",
    [
        ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
        ["Womens E-Mail", "No E-Mail", "Mens E-Mail"],
        ["Mens E-Mail", "Womens E-Mail", "No E-Mail"],
    ],
)
def test_resolve_control_label_recognizes_no_email_holdout_regardless_of_order(order):
    """
    Regression test: previously, with no exact 'control' match, this
    fell straight to the alphabetical fallback, which picks
    'Mens E-Mail' — a real treatment arm — as "control" for Hillstrom.
    Every pairwise comparison downstream then used the wrong baseline.
    'No E-Mail' must now resolve as control no matter which arm
    happens to appear first in the raw data.
    """
    from app.stats.dataset_classifier import resolve_control_label

    df = pd.DataFrame({"segment": order * 4})
    assert resolve_control_label(df, "segment") == "No E-Mail"


@pytest.mark.parametrize(
    "label",
    [
        "control", "Control", "CTRL", "ctrl", "Holdout", "holdout",
        "No E-Mail", "No Email", "no_email", "no-email",
        "No Campaign", "No Treatment", "Untreated",
    ],
)
def test_resolve_control_label_recognizes_known_holdout_vocabulary(label):
    """Every documented holdout/no-treatment label variant resolves as control, not just Hillstrom's exact spelling."""
    from app.stats.dataset_classifier import resolve_control_label

    df = pd.DataFrame({"variant": [label, "Treatment A", "Treatment B"] * 3})
    assert resolve_control_label(df, "variant") == label


def test_resolve_control_label_exact_control_match_still_wins_over_holdout_vocabulary():
    """Priority order: an exact 'control' match must still win even if a holdout-style label is also present."""
    from app.stats.dataset_classifier import resolve_control_label

    df = pd.DataFrame({"variant": ["control", "No E-Mail", "Treatment"] * 3})
    assert resolve_control_label(df, "variant") == "control"


def test_resolve_control_label_falls_back_to_alphabetical_when_no_known_label_present():
    """No exact 'control' match and no holdout vocabulary match -> unchanged alphabetical fallback (e.g. 'A'/'B' datasets)."""
    from app.stats.dataset_classifier import resolve_control_label

    df = pd.DataFrame({"variant": ["B", "A", "C"] * 3})
    assert resolve_control_label(df, "variant") == "A"


# --- ratio metric detection: numerator/denominator pairs -------------------
#
# Not a new statistical method — only reports a plausible naming/shape
# match so downstream code can decide which existing engine path fits
# (a rate ratio like conversions/users usually keeps the existing
# binary/proportion path using raw counts; a monetary ratio like
# revenue/users is usually genuinely continuous).

def test_ratio_detection_conversions_over_users():
    """Exact example: variant|users|conversions -> conversion_rate = conversions/users."""
    from app.stats.dataset_classifier import classify_dataset

    df = pd.DataFrame({"variant": ["A", "B"], "users": [1000, 1000], "conversions": [80, 95]})
    info = classify_dataset(df)
    assert len(info.ratio_metric_candidates) == 1
    candidate = info.ratio_metric_candidates[0]
    assert candidate.metric_name == "conversion_rate"
    assert candidate.numerator == "conversions"
    assert candidate.denominator == "users"
    assert candidate.ratio_definition == "conversions / users"


def test_ratio_detection_revenue_over_users():
    """Exact example: variant|revenue|users -> revenue_per_user = revenue/users."""
    from app.stats.dataset_classifier import classify_dataset

    df = pd.DataFrame({"variant": ["A", "B"], "revenue": [50000, 62000], "users": [1000, 1000]})
    info = classify_dataset(df)
    assert len(info.ratio_metric_candidates) == 1
    candidate = info.ratio_metric_candidates[0]
    assert candidate.metric_name == "revenue_per_user"
    assert candidate.numerator == "revenue"
    assert candidate.denominator == "users"
    assert candidate.ratio_definition == "revenue / users"


def test_ratio_detection_does_not_exclude_the_chosen_primary_metric():
    """
    Regression test: role_exclude (used for funnel/stratification/
    guardrail detection) always contains the chosen metric_col. Ratio
    detection deliberately does NOT use role_exclude for this reason —
    the primary metric is very often exactly the ratio's numerator
    (e.g. 'conversions' above). Using role_exclude here would silently
    return an empty list for the most common real case, exactly the
    same shape of bug detect_funnel_metrics already has.
    """
    from app.stats.dataset_classifier import classify_dataset

    df = pd.DataFrame({"variant": ["A", "B"], "users": [1000, 1000], "conversions": [80, 95]})
    info = classify_dataset(df)
    assert info.metric_label == "Conversions"  # confirms 'conversions' WAS chosen as metric_col
    assert any(c.numerator == "conversions" for c in info.ratio_metric_candidates)


def test_ratio_detection_multiple_candidates_when_both_shapes_present():
    df = pd.DataFrame({
        "variant": ["A", "B"], "users": [1000, 1000],
        "conversions": [80, 95], "revenue": [4000, 5200],
    })
    from app.stats.dataset_classifier import classify_dataset

    info = classify_dataset(df)
    names = {c.metric_name for c in info.ratio_metric_candidates}
    assert names == {"conversion_rate", "revenue_per_user"}


def test_ratio_detection_empty_without_a_denominator_column():
    """No known denominator (users/sessions/impressions/...) present -> no ratio candidates, even with a numerator-shaped column."""
    from app.stats.dataset_classifier import detect_ratio_metric_candidates

    df = pd.DataFrame({"variant": ["A", "B"], "conversions": [80, 95]})
    assert detect_ratio_metric_candidates(df, exclude={"variant"}) == []


def test_ratio_detection_empty_for_hillstrom_unit_level_data():
    """
    A genuinely unit-level dataset (one row per customer, binary
    outcome columns) has no aggregate numerator/denominator pair at
    all — must not fire just because column names loosely resemble
    the vocabulary (Hillstrom has no 'users'/'sessions'/'impressions'
    denominator column).
    """
    from app.stats.dataset_classifier import classify_dataset

    df = pd.DataFrame({
        "segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"] * 10,
        "visit": [0, 1, 0] * 10,
        "conversion": [0, 1, 0] * 10,
        "spend": [0.0, 25.0, 0.0] * 10,
    })
    info = classify_dataset(df)
    assert info.ratio_metric_candidates == []


def test_ratio_detection_structural_columns_never_used_as_either_side():
    """user_id/variant columns must never be proposed as a numerator or denominator, even if their name happened to collide with the vocabulary."""
    from app.stats.dataset_classifier import detect_ratio_metric_candidates

    df = pd.DataFrame({"user_id": [1, 2], "users": [10, 20], "conversions": [1, 2]})
    candidates = detect_ratio_metric_candidates(df, exclude={"user_id"})
    for c in candidates:
        assert c.numerator != "user_id"
        assert c.denominator != "user_id"


# --- funnel-stage detection: primary-metric-exclusion fix -------------------

def test_funnel_metrics_includes_hillstrom_shaped_stages():
    """
    Real-data regression: Hillstrom's 'conversion' column is BOTH the
    chosen primary metric AND a genuine funnel stage (visit ->
    conversion). Previously `detect_funnel_metrics` was called with
    `role_exclude` (which always contains metric_col), so it only ever
    saw 'visit' — 1 stage, below _MIN_FUNNEL_STAGES — and reported an
    empty funnel for a dataset that clearly has one. Same bug shape,
    same fix, as detect_ratio_metric_candidates above.
    """
    from app.stats.dataset_classifier import classify_dataset

    df = pd.DataFrame({
        "segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"] * 10,
        "recency": list(range(1, 31)),
        "visit": [0, 1, 0] * 10,
        "conversion": [0, 1, 0] * 10,
        "spend": [0.0, 25.0, 0.0] * 10,
    })
    info = classify_dataset(df)
    assert info.metric_label == "Conversion Rate"  # confirms 'conversion' WAS chosen as metric_col
    assert set(info.funnel_metrics) == {"visit", "conversion"}


def test_funnel_metrics_still_requires_at_least_two_stages():
    """Unchanged behavior: a single funnel-shaped column alone is just the primary metric, not a funnel — still reports empty."""
    from app.stats.dataset_classifier import classify_dataset

    df = pd.DataFrame({
        "variant": ["control", "treatment"] * 10,
        "conversion": [0, 1] * 10,
    })
    info = classify_dataset(df)
    assert info.funnel_metrics == []


def test_funnel_metrics_unaffected_for_non_funnel_datasets():
    """A dataset with no funnel-stage-shaped columns at all is unaffected by the exclude-set change."""
    from app.stats.dataset_classifier import classify_dataset

    df = pd.DataFrame({
        "variant": ["control", "treatment"] * 10,
        "revenue": [10.0, 15.0] * 10,
    })
    info = classify_dataset(df)
    assert info.funnel_metrics == []


# --- Guardrail root-cause audit follow-up: candidate classifier scope ----
#
# Covers the Age-as-guardrail finding from the audited "AB Testing
# Data.csv" report: `Age` was never invented by the decision engine —
# it was selectable in the first place because `detect_available_metrics`
# (which the guardrail-selection UI is built from) treated any numeric
# column as an outcome metric, with no notion of a demographic/identity
# attribute. See `_DEMOGRAPHIC_ATTRIBUTE_KEYWORDS` and its use in
# `detect_available_metrics`.


def _ab_testing_style_df(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [f"U{i}" for i in range(n)],
        "group": (["control"] * (n // 2)) + (["treatment"] * (n // 2)),
        "converted": ([0, 1] * (n // 2)),
        "age": list(range(18, 18 + n)),
        "gender": (["Male", "Female"] * (n // 2)),
        "session_duration": [float(i) for i in range(n)],
        "purchase_amount": [float(i) * 1.1 for i in range(n)],
        "revenue_gbp": [float(i) * 0.9 for i in range(n)],
        "visitor_id": list(range(n)),
    })


def test_age_excluded_from_available_metrics():
    info = classify_dataset(_ab_testing_style_df())
    assert "Age" not in info.available_metrics
    assert "Age" not in info.additional_metrics


def test_gender_excluded_from_available_metrics():
    info = classify_dataset(_ab_testing_style_df())
    assert "Gender" not in info.available_metrics


def test_age_still_available_as_stratification_candidate_path():
    """Excluding Age as a metric must not remove it as a dimension elsewhere."""
    df = _ab_testing_style_df()
    df["age"] = ([25, 45] * (len(df) // 2))  # low-cardinality, within the stratification cap
    # The demographic exclusion must be scoped to `detect_available_metrics`
    # only, never to stratification.
    from app.stats.dataset_classifier import detect_stratification_candidates
    candidates = detect_stratification_candidates(df, exclude={"user_id", "group", "converted"})
    assert "age" in candidates


def test_purchase_amount_detected_as_guardrail_candidate():
    info = classify_dataset(_ab_testing_style_df())
    assert "Purchase Amount" in info.guardrail_candidates


def test_session_duration_detected_as_guardrail_candidate():
    info = classify_dataset(_ab_testing_style_df())
    assert "Session Duration" in info.guardrail_candidates


def test_revenue_gbp_detected_as_guardrail_candidate():
    info = classify_dataset(_ab_testing_style_df())
    assert "Revenue Gbp" in info.guardrail_candidates


def test_early_cancel_14d_detected_as_guardrail_candidate():
    df = pd.DataFrame({
        "user_id": [f"U{i}" for i in range(20)],
        "group": (["control"] * 10) + (["treatment"] * 10),
        "converted": [0, 1] * 10,
        "early_cancel_14d": [0, 1] * 10,
    })
    info = classify_dataset(df)
    assert "Early Cancel 14D" in info.guardrail_candidates


def test_visitor_id_not_a_false_positive_guardrail_candidate():
    """'visit' is a real guardrail keyword now — must not match inside `visitor_id`."""
    info = classify_dataset(_ab_testing_style_df())
    assert "Visitor Id" not in info.guardrail_candidates


def test_stratification_candidates_still_finds_categorical_dimensions():
    """Regression for the frontend contract bug: backend truth, independent of the UI field-name fix."""
    df = pd.DataFrame({
        "user_id": [f"U{i}" for i in range(40)],
        "group": (["control"] * 20) + (["treatment"] * 20),
        "converted": [0, 1] * 20,
        "gender": (["Male", "Female"] * 20),
        "device_type": (["Mobile", "Desktop"] * 20),
    })
    info = classify_dataset(df)
    assert "gender" in info.stratification_candidates
    assert "device_type" in info.stratification_candidates
