"""
Regression tests for the reported bug: selecting "Stratified Analysis
by <column>" in the UI must survive all the way through the planner —
never silently reclassified as "Full Experiment Review" by keyword or
LLM free-text intent detection.
"""

from app.graph.planner_strategy import (
    Intent,
    KeywordPlanner,
    detect_explicit_stratification_request,
    plan_from_explicit_settings,
    plan_from_free_text_stratification_request,
)
from app.graph.nodes.planner_node import planner_node
from app.core.dataset_store import store_dataset
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.settings import AnalysisSettings
import pandas as pd


def _demo_dataset() -> DatasetInfo:
    return DatasetInfo(
        type=DatasetType.RAW_USER_LEVEL,
        variants=2,
        users=1000,
        metric_label="Conversion Rate",
        available_metrics=["Conversion Rate"],
        metric_selection_reason="test fixture",
    )


class TestPlanFromExplicitSettings:
    def test_stratified_mode_with_column_produces_stratified_plan(self):
        plan = plan_from_explicit_settings("stratified", "landing_page")
        assert plan is not None
        assert plan["intent_label"] == "Stratified Analysis"
        assert plan["run_capability_nodes"] == ["validation", "experiment"]
        assert plan["llm_status"] == "not_used"

    def test_no_analysis_mode_returns_none(self):
        assert plan_from_explicit_settings(None, None) is None

    def test_stratified_mode_without_column_falls_through(self):
        """An incomplete/malformed request (mode selected, no column) must not fabricate a column."""
        assert plan_from_explicit_settings("stratified", None) is None
        assert plan_from_explicit_settings("stratified", "  ") is None

    def test_unrecognized_mode_falls_through(self):
        assert plan_from_explicit_settings("something_else", "landing_page") is None


class TestPlannerNodeHonorsExplicitMode:
    def test_stratified_selection_is_never_reclassified_as_full_review(self):
        """
        Exact bug scenario: user_prompt phrased like a normal full-review
        request ("Should we ship variant B?") — which alone would
        normally route to Intent.FULL_REVIEW via KeywordPlanner — but
        settings explicitly selected Stratified Analysis. The explicit
        selection must win.
        """
        state = {
            "user_prompt": "Should we ship variant B? Evaluate the experiment.",
            "dataset": _demo_dataset(),
            "settings": AnalysisSettings(analysis_mode="stratified", stratification_column="landing_page"),
        }

        # Sanity check: this exact prompt WOULD normally be classified
        # as Full Experiment Review by KeywordPlanner — proving the
        # override is actually doing something, not just vacuously true.
        keyword_only = KeywordPlanner().plan(state["user_prompt"], state["dataset"])
        assert keyword_only["intent_label"] == "Full Experiment Review"

        result_state = planner_node(state)
        plan = result_state["plan"]
        assert plan["intent_label"] == "Stratified Analysis"
        assert plan["intent_label"] != "Full Experiment Review"
        assert "validation" in plan["run_capability_nodes"]
        assert "experiment" in plan["run_capability_nodes"]

    def test_without_explicit_mode_falls_back_to_normal_planner(self):
        state = {
            "user_prompt": "Should we ship variant B?",
            "dataset": _demo_dataset(),
            "settings": AnalysisSettings(),
        }
        result_state = planner_node(state)
        assert result_state["plan"]["intent_label"] == "Full Experiment Review"

    def test_no_settings_key_at_all_does_not_crash(self):
        """Defensive: some callers may not set `settings` at all."""
        state = {"user_prompt": "Should we ship variant B?", "dataset": _demo_dataset()}
        result_state = planner_node(state)
        assert result_state["plan"]["intent_label"] == "Full Experiment Review"


# ---------------------------------------------------------------------------
# Free-text explicit stratification detection (typed into the ordinary
# prompt box, NOT a structured UI setting) — this is the actual reported
# bug: "Stratified Analysis by landing_page" typed as free text was
# becoming "Full Experiment Review" because KeywordPlanner had no concept
# of "stratified"/"stratify" at all (only "analysis", matching the
# generic full-review keyword list).
# ---------------------------------------------------------------------------


class TestDetectExplicitStratificationRequest:
    """Requirement #8/#9 — the exact reported string, plus natural-language
    and capitalization variants."""

    def test_exact_reported_string(self):
        is_match, column = detect_explicit_stratification_request("Stratified Analysis by landing_page")
        assert is_match is True
        assert column == "landing_page"

    def test_stratify_by_variant(self):
        is_match, column = detect_explicit_stratification_request("stratify by landing_page")
        assert is_match is True
        assert column == "landing_page"

    def test_stratified_analysis_by_variant(self):
        is_match, column = detect_explicit_stratification_request("stratified analysis by landing_page")
        assert is_match is True
        assert column == "landing_page"

    def test_run_a_stratified_analysis_using_variant(self):
        is_match, column = detect_explicit_stratification_request("run a stratified analysis using landing_page")
        assert is_match is True
        assert column == "landing_page"

    def test_analyze_using_stratification_by_variant(self):
        is_match, column = detect_explicit_stratification_request("analyze using stratification by landing_page")
        assert is_match is True
        assert column == "landing_page"

    def test_all_uppercase_variant(self):
        is_match, column = detect_explicit_stratification_request("STRATIFIED ANALYSIS BY LANDING_PAGE")
        assert is_match is True
        assert column.lower() == "landing_page"

    def test_mixed_capitalization_variant(self):
        is_match, column = detect_explicit_stratification_request("Stratify By Landing_Page")
        assert is_match is True
        assert column.lower() == "landing_page"

    def test_bare_stratified_analysis_with_no_column_clause(self):
        is_match, column = detect_explicit_stratification_request("stratified analysis")
        assert is_match is True
        assert column is None

    def test_negative_segmentation_phrase_is_not_stratification(self):
        """Requirement #10 — 'analyze segments by X' must remain segmentation, not stratification."""
        is_match, column = detect_explicit_stratification_request("analyze segments by landing_page")
        assert is_match is False
        assert column is None

    def test_ordinary_full_review_prompt_does_not_match(self):
        is_match, column = detect_explicit_stratification_request("Should we ship variant B? Evaluate the experiment.")
        assert is_match is False


class TestPlanFromFreeTextStratificationRequest:
    def test_resolves_column_against_real_dataset_columns_case_insensitively(self):
        result = plan_from_free_text_stratification_request(
            "Stratify By Landing_Page", ["user_id", "group", "landing_page", "converted"]
        )
        assert result is not None
        plan, resolved_column = result
        assert plan["intent_label"] == "Stratified Analysis"
        assert resolved_column == "landing_page"  # canonical dataset casing, not the prompt's casing

    def test_unresolvable_column_passes_through_raw_text_for_downstream_validation(self):
        result = plan_from_free_text_stratification_request(
            "stratify by nonexistent_col", ["user_id", "group", "landing_page", "converted"]
        )
        assert result is not None
        _, resolved_column = result
        assert resolved_column == "nonexistent_col"

    def test_no_stratification_intent_returns_none(self):
        assert plan_from_free_text_stratification_request("Should we ship variant B?", ["group"]) is None

    def test_segmentation_phrase_returns_none(self):
        assert plan_from_free_text_stratification_request("analyze segments by landing_page", ["group"]) is None


class TestPlannerNodeHonorsFreeTextStratificationRequest:
    """Requirement #4/#5/#6/#11 — proves the fix through the actual planner_node,
    with a real dataset_id (so column validation runs against real data),
    using the exact free-text request rather than constructing AnalysisSettings."""

    def _dataset_id(self):
        df = pd.DataFrame({
            "user_id": range(10),
            "group": ["control", "treatment"] * 5,
            "landing_page": ["old_page", "new_page"] * 5,
            "converted": [0, 1] * 5,
        })
        return store_dataset(df)

    def test_exact_reported_free_text_string_is_never_downgraded(self):
        state = {
            "user_prompt": "Stratified Analysis by landing_page",
            "dataset_id": self._dataset_id(),
            "dataset": _demo_dataset(),
            "settings": AnalysisSettings(),  # no structured analysis_mode set — pure free text
        }

        # Sanity check: KeywordPlanner alone WOULD normally call this Full
        # Experiment Review (the word "analysis" alone triggers it) —
        # proving the fix is actually doing something.
        keyword_only = KeywordPlanner().plan(state["user_prompt"], state["dataset"])
        assert keyword_only["intent_label"] == "Full Experiment Review"

        result_state = planner_node(state)
        plan = result_state["plan"]
        assert plan["intent_label"] == "Stratified Analysis"
        assert plan["intent_label"] != "Full Experiment Review"
        assert result_state["settings"].analysis_mode == "stratified"
        assert result_state["settings"].stratification_column == "landing_page"

    def test_natural_language_variant_run_a_stratified_analysis_using(self):
        state = {
            "user_prompt": "run a stratified analysis using landing_page",
            "dataset_id": self._dataset_id(),
            "dataset": _demo_dataset(),
            "settings": AnalysisSettings(),
        }
        result_state = planner_node(state)
        assert result_state["plan"]["intent_label"] == "Stratified Analysis"
        assert result_state["settings"].stratification_column == "landing_page"

    def test_segmentation_phrase_is_not_reclassified_as_stratification(self):
        """Requirement #10, at the planner_node level."""
        state = {
            "user_prompt": "analyze segments by landing_page",
            "dataset_id": self._dataset_id(),
            "dataset": _demo_dataset(),
            "settings": AnalysisSettings(),
        }
        result_state = planner_node(state)
        assert result_state["plan"]["intent_label"] != "Stratified Analysis"
        assert result_state["plan"]["intent_label"] == "Full Experiment Review"
