"""
Focused tests for `enrich_with_assignment` and its integration into
`classify_dataset` / `detect_experiment_columns`.

SCOPE: the primary dataset (e.g. ``user_id | order_value``) has no
variant column of its own; a separate assignment dataset (``user_id |
variant``) supplies it. Before this fix, `classify_dataset` and
`detect_experiment_columns` only ever looked at the primary dataframe,
so an assignment file existing had zero effect — the UI showed
"0 Variants" and then "missing a recognizable variant/group column"
even with a perfectly valid assignment file provided.
"""

import pandas as pd
import pytest

from app.schemas.dataset import DatasetType
from app.stats.dataset_classifier import (
    DatasetClassificationError,
    classify_dataset,
    detect_experiment_columns,
    enrich_with_assignment,
)


def _primary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "order_value": [120, 95, 140],
        }
    )


def _assignment_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "variant": ["control", "treatment", "treatment"],
        }
    )


class TestEnrichWithAssignment:
    def test_none_assignment_df_returns_original_unchanged(self):
        df = _primary_df()
        result = enrich_with_assignment(df, assignment_df=None)
        pd.testing.assert_frame_equal(result, df)

    def test_merges_only_user_and_variant_columns(self):
        """Only user + variant are imported — no other assignment-file
        column (e.g. a business field) leaks into the merged frame."""
        assignment = _assignment_df()
        assignment["plan"] = ["pro", "pro", "free"]
        assignment["billing_frequency"] = ["monthly", "annual", "monthly"]

        merged = enrich_with_assignment(_primary_df(), assignment_df=assignment)

        assert "variant" in merged.columns
        assert "plan" not in merged.columns
        assert "billing_frequency" not in merged.columns
        assert list(merged["variant"]) == ["control", "treatment", "treatment"]

    def test_never_infers_variant_from_business_field_alone(self):
        """An assignment file with no recognizable variant column (only
        business fields) must raise, never silently pick one of them."""
        assignment = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "plan": ["pro", "pro", "free"],
                "subscription_type": ["monthly", "annual", "monthly"],
            }
        )
        with pytest.raises(DatasetClassificationError, match="variant/group column"):
            enrich_with_assignment(_primary_df(), assignment_df=assignment)

    def test_duplicate_user_assignment_raises(self):
        assignment = pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "variant": ["control", "treatment", "treatment"],
            }
        )
        with pytest.raises(DatasetClassificationError, match="duplicate"):
            enrich_with_assignment(_primary_df(), assignment_df=assignment)

    def test_missing_user_column_in_assignment_raises(self):
        assignment = pd.DataFrame({"variant": ["control", "treatment", "treatment"]})
        with pytest.raises(DatasetClassificationError, match="user identifier"):
            enrich_with_assignment(_primary_df(), assignment_df=assignment)

    def test_missing_user_column_in_primary_raises(self):
        primary = pd.DataFrame({"order_value": [120, 95, 140]})
        with pytest.raises(DatasetClassificationError, match="user identifier"):
            enrich_with_assignment(primary, assignment_df=_assignment_df())

    def test_left_join_preserves_all_primary_rows_even_without_assignment(self):
        """how="left" — a user missing from the assignment file keeps
        their primary-dataset row (with NaN variant), never gets dropped."""
        primary = _primary_df()
        assignment = pd.DataFrame({"user_id": [1, 2], "variant": ["control", "treatment"]})
        merged = enrich_with_assignment(primary, assignment_df=assignment)
        assert len(merged) == 3
        assert merged.loc[merged["user_id"] == 3, "variant"].isna().all()

    def test_different_user_column_names_still_merge(self):
        """Primary uses `user_id`, assignment uses `customer_id` — both
        match the existing user-id vocabulary, so the merge still works
        and the redundant assignment-side key column is dropped."""
        primary = _primary_df()
        assignment = pd.DataFrame({"customer_id": [1, 2, 3], "variant": ["control", "treatment", "treatment"]})
        merged = enrich_with_assignment(primary, assignment_df=assignment)
        assert "customer_id" not in merged.columns
        assert list(merged["variant"]) == ["control", "treatment", "treatment"]


class TestClassifyDatasetWithAssignment:
    def test_reported_scenario_matches_expected_output(self):
        """Exact scenario from the bug report."""
        info = classify_dataset(_primary_df(), assignment_df=_assignment_df())
        assert info.variants == 2
        assert info.users == 3
        assert info.metric_label == "Order Value"

    def test_without_assignment_df_zero_variants_as_before(self):
        """Sanity check: without the assignment file, the primary
        dataset alone (no variant column) legitimately has 0 variants
        — confirms the fix doesn't change single-file behavior."""
        info = classify_dataset(_primary_df())
        assert info.variants == 0

    def test_none_assignment_df_is_a_no_op(self):
        info_with_none = classify_dataset(_primary_df(), assignment_df=None)
        info_without_param = classify_dataset(_primary_df())
        assert info_with_none.variants == info_without_param.variants
        assert info_with_none.users == info_without_param.users


class TestDetectExperimentColumnsWithAssignment:
    def test_reported_scenario_resolves_all_columns(self):
        columns = detect_experiment_columns(_primary_df(), assignment_df=_assignment_df())
        assert columns.user_col == "user_id"
        assert columns.variant_col == "variant"
        assert columns.metric_col == "order_value"

    def test_without_assignment_df_still_raises_missing_variant(self):
        """Confirms the exact previously-reported failure mode still
        occurs (correctly) when no assignment data is provided at all
        — this fix only changes behavior when assignment_df IS given."""
        with pytest.raises(DatasetClassificationError, match="variant/group column"):
            detect_experiment_columns(_primary_df())

    def test_business_field_in_assignment_never_becomes_variant(self):
        assignment = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "product": ["basic", "pro", "pro"],
            }
        )
        with pytest.raises(DatasetClassificationError, match="variant/group column"):
            detect_experiment_columns(_primary_df(), assignment_df=assignment)
