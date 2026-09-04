"""
LLMPlanner tests — mocked LLM client, no real network call to
OpenRouter (this sandbox can't reach openrouter.ai anyway). Verifies:
  - happy path: valid structured output is used correctly
  - malformed output (invalid capabilities) degrades to a safe default
  - any exception (network, auth, parsing) falls back to KeywordPlanner
  - the LLM is given dataset metadata only, never raw data
  - REGRESSION: arbitrary LLM text can never become the displayed
    intent label (the live-observed bug: OpenRouter echoing the raw
    user prompt as `intent`, e.g. "evaluate this experiment with cuped"
    showing up as the Execution Step label instead of a real
    classification)
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.graph.planner_strategy import (
    INTENT_DISPLAY_NAMES,
    Intent,
    KeywordPlanner,
    LLMPlanner,
    PlannerLLMResponseModel,
)
from app.schemas.dataset import DatasetInfo, DatasetType


def _fake_llm(response):
    """A fake LangChain chat model: .with_structured_output(schema, include_raw=True).invoke(messages) -> {"parsed": response, ...}."""
    structured = SimpleNamespace(invoke=lambda messages: {"parsed": response, "raw": None, "parsing_error": None})
    return SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)


def _demo_dataset() -> DatasetInfo:
    return DatasetInfo(type=DatasetType.AGGREGATED_AB_TEST, variants=2, users=12400, metric_label="Conversion Rate", metric_selection_reason="Selected by the deterministic outcome-column priority — no competing outcome metrics were available in this dataset.")


class TestLLMPlannerHappyPath:
    def test_valid_response_used_as_is(self):
        response = SimpleNamespace(
            intent=Intent.FULL_REVIEW,
            capabilities=["validation", "experiment"],
            reason="User explicitly asked to evaluate and ship.",
        )
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            result = LLMPlanner().plan("Evaluate and ship this experiment", _demo_dataset())

        assert result["intent_label"] == "Full Experiment Review"
        assert result["run_capability_nodes"] == ["validation", "experiment", "knowledge_base"]

    def test_knowledge_base_only_response(self):
        response = SimpleNamespace(
            intent=Intent.KNOWLEDGE_BASE, capabilities=["knowledge_base"], reason="Asked what CUPED is."
        )
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            result = LLMPlanner().plan("What is CUPED?", _demo_dataset())

        assert result["intent_label"] == "Methodology Consultation"
        assert result["run_capability_nodes"] == ["knowledge_base"]

    @pytest.mark.parametrize("intent,expected_label", list(INTENT_DISPLAY_NAMES.items()))
    def test_every_intent_enum_value_maps_to_its_display_name(self, intent, expected_label):
        response = SimpleNamespace(intent=intent, capabilities=["validation", "experiment"], reason="x")
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            result = LLMPlanner().plan("anything", _demo_dataset())

        assert result["intent_label"] == expected_label


class TestIntentLabelCannotBeArbitraryText:
    """
    Regression coverage for the live-observed bug: the OpenRouter model
    echoed the user's raw prompt ("evaluate this experiment with cuped")
    into the `intent` field instead of classifying it. This must now be
    structurally impossible.
    """

    def test_schema_rejects_the_exact_observed_bug_value(self):
        """The literal string observed in production must fail schema validation."""
        with pytest.raises(ValidationError):
            PlannerLLMResponseModel(
                intent="evaluate this experiment with cuped",
                capabilities=["validation", "experiment"],
                reason="x",
            )

    @pytest.mark.parametrize(
        "arbitrary_text",
        [
            "evaluate this experiment with cuped",
            "Full Experiment Review",  # even a previously-valid-looking label string is rejected — only the enum VALUE works
            "",
            "something completely unrelated",
            "FULL_REVIEW",  # wrong case
        ],
    )
    def test_schema_rejects_any_non_enum_string(self, arbitrary_text):
        with pytest.raises(ValidationError):
            PlannerLLMResponseModel(intent=arbitrary_text, capabilities=["validation"], reason="x")

    def test_only_the_five_declared_values_are_accepted(self):
        for intent in Intent:
            # Should NOT raise — every real enum member is valid.
            PlannerLLMResponseModel(intent=intent.value, capabilities=["validation"], reason="x")

    def test_llm_call_raising_validation_error_falls_back_to_keyword_planner(self):
        """
        Simulates what actually happens in production when the live
        model's raw output can't be parsed into the strict schema:
        LangChain's structured-output layer raises during `.invoke()`,
        which LLMPlanner's existing try/except already catches.
        """

        def raise_validation_error(messages):
            PlannerLLMResponseModel(intent="garbage from the model", capabilities=[], reason="x")  # raises

        structured = SimpleNamespace(invoke=raise_validation_error)
        fake_llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)

        with patch("app.llm.client.get_llm", return_value=fake_llm):
            llm_result = LLMPlanner().plan("evaluate this experiment with cuped", _demo_dataset())
            keyword_result = KeywordPlanner().plan("evaluate this experiment with cuped", _demo_dataset())

        # Routing (intent_label, run_capability_nodes) still degrades to
        # exactly what KeywordPlanner produces, but the result is not a
        # bitwise-identical dict — a failed LLM call is visibly marked as
        # a fallback (llm_status), never indistinguishable from a genuine
        # keyword-routed request.
        assert llm_result["intent_label"] == keyword_result["intent_label"]
        assert llm_result["run_capability_nodes"] == keyword_result["run_capability_nodes"]
        assert llm_result["llm_status"] == "fallback"
        assert keyword_result["llm_status"] == "not_used"
        assert llm_result["llm_error"] is not None
        assert llm_result["intent_label"] in INTENT_DISPLAY_NAMES.values()


class TestLLMPlannerRobustness:
    def test_invalid_capabilities_falls_back_to_full_review(self):
        response = SimpleNamespace(intent=Intent.FULL_REVIEW, capabilities=["make_coffee", "delete_files"], reason="???")
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            result = LLMPlanner().plan("something weird", _demo_dataset())

        # Stage 10: FULL_REVIEW's capability floor (validation, experiment,
        # knowledge_base) fills in directly from an empty filtered list —
        # the plain ["validation", "experiment"] fallback line is never
        # reached for this intent anymore, since the floor alone is
        # already non-empty.
        assert result["run_capability_nodes"] == ["validation", "experiment", "knowledge_base"]

    def test_empty_capabilities_falls_back_to_full_review(self):
        response = SimpleNamespace(intent=Intent.FULL_REVIEW, capabilities=[], reason="No capabilities chosen")
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            result = LLMPlanner().plan("something", _demo_dataset())

        assert result["run_capability_nodes"] == ["validation", "experiment", "knowledge_base"]

    def test_llm_exception_falls_back_to_keyword_planner(self):
        def raise_error(**kwargs):
            raise RuntimeError("OPENROUTER_API_KEY not set")

        with patch("app.llm.client.get_llm", side_effect=raise_error):
            llm_result = LLMPlanner().plan("What is CUPED?", _demo_dataset())
            keyword_result = KeywordPlanner().plan("What is CUPED?", _demo_dataset())

        # Falls back to the SAME routing KeywordPlanner would have
        # produced (intent_label, run_capability_nodes), but is now
        # explicitly marked as a fallback rather than looking identical
        # to a real keyword-routed request — see planner_strategy.py.
        assert llm_result["intent_label"] == keyword_result["intent_label"]
        assert llm_result["run_capability_nodes"] == keyword_result["run_capability_nodes"]
        assert llm_result["llm_status"] == "fallback"
        assert "OPENROUTER_API_KEY" in llm_result["llm_error"]

    def test_llm_never_receives_raw_data(self):
        """Structural guarantee: plan() only accepts (user_prompt, DatasetInfo) — no DataFrame parameter exists at all."""
        import inspect

        sig = inspect.signature(LLMPlanner.plan)
        param_names = list(sig.parameters.keys())
        assert "df" not in param_names
        assert "dataframe" not in param_names
        assert "data" not in param_names

class TestLLMPlannerCapabilityContract:
    @pytest.mark.parametrize(
        "intent,returned,expected",
        [
            (Intent.FULL_REVIEW, ["validation"], ["validation", "experiment", "knowledge_base"]),
            (Intent.STATISTICAL_ANALYSIS, ["validation"], ["validation", "experiment"]),
            (Intent.EXPLANATION, ["validation"], ["validation", "experiment"]),
            (Intent.QUALITY_CHECK, ["experiment"], ["validation"]),
            (Intent.KNOWLEDGE_BASE, ["validation", "experiment"], ["knowledge_base"]),
        ],
    )
    def test_intent_cannot_drop_required_capability(self, intent, returned, expected):
        response = SimpleNamespace(intent=intent, capabilities=returned, reason="x")
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            result = LLMPlanner().plan("Evaluate this experiment end-to-end", _demo_dataset())
        assert result["run_capability_nodes"] == expected

    def test_full_review_with_funnel_keeps_all_required_capabilities(self):
        response = SimpleNamespace(
            intent=Intent.FULL_REVIEW,
            capabilities=["funnel"],
            reason="combined request",
        )
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            result = LLMPlanner().plan("Analyze this experiment end-to-end", _demo_dataset())
        # Stage 10: FULL_REVIEW's floor now includes knowledge_base too —
        # appended after validation/experiment, same fill order as
        # _CAPABILITY_FLOOR_BY_INTENT[FULL_REVIEW].
        assert result["run_capability_nodes"] == ["funnel", "validation", "experiment", "knowledge_base"]
