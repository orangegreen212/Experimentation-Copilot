"""
"Stratified Analysis by landing page"
(a natural-language phrase with a SPACE, not the real column's
underscore) must resolve to the dataset's real, exact, un-truncated
column name "landing_page" — not "landing".

Before the fix, `_STRATIFICATION_COLUMN_PATTERN` only captured
`\\w*` after "by/using/on", so it stopped at the space and silently
truncated "landing_page" down to "landing". See
app/graph/planner_strategy.py.
"""

import pandas as pd

from app.core.dataset_store import store_dataset
from app.graph.graph_builder import experiment_review_graph
from app.graph.nodes.planner_node import planner_node
from app.graph.planner_strategy import (
    detect_explicit_stratification_request,
    plan_from_free_text_stratification_request,
)
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.settings import AnalysisSettings
from app.schemas.stratification import StratificationStatus


def _demo_dataset() -> DatasetInfo:
    return DatasetInfo(
        type=DatasetType.RAW_USER_LEVEL,
        variants=2,
        users=1000,
        metric_label="Conversion Rate",
        available_metrics=["Conversion Rate"],
        metric_selection_reason="test fixture",
    )


class TestDetectExplicitStratificationRequestMultiWord:
    def test_space_separated_column_phrase_is_captured_whole(self):
        is_match, column = detect_explicit_stratification_request(
            "Stratified Analysis by landing page"
        )
        assert is_match is True
        assert column == "landing page"  # captured whole, not truncated to "landing"


class TestResolveAgainstRealDatasetColumns:
    def test_landing_page_space_resolves_to_landing_page_underscore(self):
        result = plan_from_free_text_stratification_request(
            "Stratified Analysis by landing page",
            ["user_id", "group", "landing_page", "converted"],
        )
        assert result is not None
        plan, resolved_column = result
        assert plan["intent_label"] == "Stratified Analysis"
        assert resolved_column == "landing_page"
        assert resolved_column != "landing"  # the exact truncation bug this fixes

    def test_trailing_extra_words_still_resolve_to_the_real_column(self):
        """A phrase with words AFTER the real column name must still resolve
        to the real column, not swallow the extra words into it."""
        result = plan_from_free_text_stratification_request(
            "Stratified Analysis by landing page for signups",
            ["user_id", "group", "landing_page", "converted"],
        )
        assert result is not None
        _, resolved_column = result
        assert resolved_column == "landing_page"


class TestPlannerNodeResolvesLandingPageColumnName:
    def _dataset_id(self):
        df = pd.DataFrame(
            {
                "user_id": range(20),
                "group": (["control"] * 10) + (["treatment"] * 10),
                "landing_page": (["old_page"] * 10) + (["new_page"] * 10),
                "converted": [0, 1] * 10,
            }
        )
        return store_dataset(df)

    def test_planner_node_resolves_space_phrase_to_exact_column_name(self):
        state = {
            "user_prompt": "Stratified Analysis by landing page",
            "dataset_id": self._dataset_id(),
            "dataset": _demo_dataset(),
            "settings": AnalysisSettings(),
        }
        result_state = planner_node(state)
        assert result_state["plan"]["intent_label"] == "Stratified Analysis"
        # Expected: "landing_page" (exact, un-truncated dataset column name).
        assert result_state["settings"].stratification_column == "landing_page"
        assert result_state["settings"].stratification_column != "landing"


def test_report_shows_exact_resolved_column_name_end_to_end():
    """Full graph run: the final report's stratification section must
    say `Variable: landing_page`-equivalent (stratification_column ==
    "landing_page"), never the truncated "landing"."""
    n = 300
    df = pd.DataFrame(
        {
            "user_id": range(2 * n),
            "group": (["control"] * n) + (["treatment"] * n),
            "landing_page": (["old_page", "new_page"] * n),
            "converted": ([0, 1] * n),
        }
    )
    dataset_id = store_dataset(df)

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Stratified Analysis by landing page",
            "settings": AnalysisSettings(),
        }
    )

    report = final_state["report"]
    assert report.stratification is not None
    assert report.stratification.stratification_column == "landing_page"
    assert report.stratification.stratification_column != "landing"
    # Regardless of eligibility outcome, the column name reported must
    # never be truncated.
    if report.stratification.status == StratificationStatus.RAN and report.stratification.eligibility:
        assert report.stratification.eligibility.stratification_column == "landing_page"
