"""
Model selection — regression tests.

Covers:
  1. AppSettings.llm_model (the "Backend default") must be a member of
     AppSettings.available_llm_models (the curated dropdown) — these
     must never silently diverge.
  2. The frontend's selected model (AnalysisSettings.model /
     planner_node's `requested_model`) must actually reach
     app.llm.client.get_llm(model=...) inside LLMPlanner, so a UI
     selection has a real effect on the planner LLM call.
  3. A failed LLM call must be visibly marked (`llm_status="fallback"`,
     with the real error) rather than looking identical to a normal
     keyword-routed / successful request.
  4. The execution-step detail text surfaces that status so the user
     can tell "selected model actually ran" apart from "silently fell
     back".
"""

from types import SimpleNamespace
from unittest.mock import patch

from app.api.routes_experiments import _planner_step_detail
from app.core.config import app_settings
from app.graph.planner_strategy import Intent, KeywordPlanner, LLMPlanner
from app.schemas.dataset import DatasetInfo, DatasetType


def _demo_dataset() -> DatasetInfo:
    return DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=12400,
        metric_label="Conversion Rate",
        metric_selection_reason="Selected by the deterministic outcome-column priority.",
    )


def _fake_llm(response):
    structured = SimpleNamespace(invoke=lambda messages: {"parsed": response, "raw": None, "parsing_error": None})
    return SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)


class TestBackendDefaultIsCurated:
    def test_llm_model_is_a_member_of_available_llm_models(self):
        """The 'Backend default' id must be one of the dropdown's own ids — no drift."""
        curated_ids = {m["id"] for m in app_settings.available_llm_models}
        assert app_settings.llm_model in curated_ids

    def test_no_label_falsely_claims_default_status(self):
        """Only the frontend's derived 'Backend default (<id>)' option should claim default status."""
        for m in app_settings.available_llm_models:
            assert "default" not in m["label"].lower()


class TestSelectedModelReachesLLMLayer:
    def test_requested_model_is_passed_to_get_llm(self):
        """The exact model id the user selected must reach get_llm(model=...)."""
        response = SimpleNamespace(
            intent=Intent.FULL_REVIEW,
            capabilities=["validation", "experiment"],
            reason="ok",
        )
        captured = {}

        def fake_get_llm(model=None):
            captured["model"] = model
            return _fake_llm(response)

        with patch("app.llm.client.get_llm", side_effect=fake_get_llm):
            LLMPlanner().plan("Evaluate and ship this", _demo_dataset(), "google/gemma-4-26b-a4b-it:free")

        assert captured["model"] == "google/gemma-4-26b-a4b-it:free"

    def test_none_model_uses_backend_default_and_is_reported(self):
        response = SimpleNamespace(intent=Intent.FULL_REVIEW, capabilities=["validation", "experiment"], reason="ok")
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            result = LLMPlanner().plan("Evaluate and ship this", _demo_dataset(), None)

        assert result["llm_status"] == "success"
        assert result["llm_requested_model"] == app_settings.llm_model

    def test_keyword_planner_reports_not_used_regardless_of_model_arg(self):
        result = KeywordPlanner().plan("evaluate this", _demo_dataset(), "some/model:free")
        assert result["llm_status"] == "not_used"
        assert result["llm_requested_model"] is None


class TestFailedModelIsNeverSilent:
    def test_fallback_marks_status_and_error(self):
        def raise_error(model=None):
            raise RuntimeError("model unavailable: 429 rate limited")

        with patch("app.llm.client.get_llm", side_effect=raise_error):
            result = LLMPlanner().plan("Evaluate and ship this", _demo_dataset(), "poolside/laguna-s-2.1:free")

        assert result["llm_status"] == "fallback"
        assert result["llm_requested_model"] == "poolside/laguna-s-2.1:free"
        assert "rate limited" in result["llm_error"]

    def test_execution_step_detail_shows_success(self):
        plan = {
            "intent_label": "Full Experiment Review",
            "run_capability_nodes": ["validation", "experiment"],
            "llm_status": "success",
            "llm_requested_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "llm_error": None,
        }
        detail = _planner_step_detail(plan)
        assert "nvidia/nemotron-3-ultra-550b-a55b:free" in detail
        assert "succeeded" in detail

    def test_execution_step_detail_shows_fallback_not_success(self):
        plan = {
            "intent_label": "Full Experiment Review",
            "run_capability_nodes": ["validation", "experiment"],
            "llm_status": "fallback",
            "llm_requested_model": "poolside/laguna-s-2.1:free",
            "llm_error": "429 rate limited",
        }
        detail = _planner_step_detail(plan)
        assert "FAILED" in detail
        assert "poolside/laguna-s-2.1:free" in detail
        assert "429 rate limited" in detail
        assert "succeeded" not in detail

    def test_execution_step_detail_unchanged_when_llm_not_used(self):
        plan = {
            "intent_label": "Full Experiment Review",
            "run_capability_nodes": ["validation", "experiment"],
            "llm_status": "not_used",
            "llm_requested_model": None,
            "llm_error": None,
        }
        detail = _planner_step_detail(plan)
        assert detail == 'Intent identified as "Full Experiment Review"'
