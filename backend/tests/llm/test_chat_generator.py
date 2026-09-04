"""
Unit tests for app/graph/chat_generator.py — the module that replaced
the previously-stubbed `follow_up_chat` (`raise NotImplementedError`).
"""

from types import SimpleNamespace
from unittest.mock import patch

from app.graph.chat_generator import LLMChatResponder, TemplateChatResponder, build_chat_message
from app.schemas.chat import ChatMessage, ChatRole
from app.schemas.quality import QualityCheck
from app.schemas.report import ConfidenceLevel, Decision, ExperimentReport, ExperimentValidity, GuardrailStatus, KnowledgeBaseReference
from app.schemas.statistics import HypothesisTestType, StatResult


def _report(**overrides) -> ExperimentReport:
    defaults = dict(
        confidence=ConfidenceLevel.HIGH,
        confidence_reason="SRM passed, well-powered, significant result.",
        confidence_stars=5,
        srm_warning=False,
        executive_summary="The variant shows a significant +8.4% lift in conversion rate.",
        quality_checks=[QualityCheck(label="Sample Ratio Mismatch (SRM)", passed=True, detail="p=0.62")],
        stats=[
            StatResult(
                metric="Conversion Rate", test_type=HypothesisTestType.CHI_SQUARE, test_name="Chi-square test",
                statistic=12.4, selection_reason="binary metric", control="4.21%", variant="4.56%",
                delta="+8.4% (rel)", delta_relative=8.4, p_value=0.0008, significant=True, ci_lower="+0.18pp", ci_upper="+0.52pp",
            )
        ],
        mde="1.8% relative",
        sample_size_note="12,400 users — exceeds the 8,200 required for 80% power",
        recommendations=["GO WITH CAUTION — significant and practically significant; no guardrail metrics were evaluated for this dataset."],
        next_steps=["Monitor AOV for 14 days."],
        knowledge_base_references=[],
        experiment_validity=ExperimentValidity.VALID,
        guardrail_status=GuardrailStatus.NOT_AVAILABLE,
        practical_significance=True,
        decision=Decision.GO_WITH_CAUTION,
        decision_reason="Conversion Rate is statistically and practically significant; no guardrail metrics were evaluated for this dataset.",
        recommendation_confidence=ConfidenceLevel.MEDIUM,
    )
    defaults.update(overrides)
    return ExperimentReport(**defaults)


class TestTemplateChatResponder:
    def test_srm_question_when_passed(self):
        reply = TemplateChatResponder().respond(_report(), "Did the SRM check pass?")
        assert "passed" in reply.lower()
        assert "p=0.62" in reply  # real detail from the report, not a hardcoded number

    def test_srm_question_when_failed(self):
        report = _report(
            srm_warning=True,
            quality_checks=[QualityCheck(label="Sample Ratio Mismatch (SRM)", passed=False, detail="58/42 split, p<0.001")],
        )
        reply = TemplateChatResponder().respond(report, "was there an srm issue?")
        assert "FAILED" in reply
        assert "58/42" in reply  # real detail, not the old mock's hardcoded "58/42"-that-was-actually-fake

    def test_ship_question_reflects_real_decision(self):
        """
        Grounded in the canonical `decision` field, not the legacy
        `confidence` field — see chat_generator.py / schemas/report.py.
        """
        reply = TemplateChatResponder().respond(_report(), "Should we ship this?")
        assert "GO_WITH_CAUTION" in reply
        assert "no guardrail metrics were evaluated" in reply  # real decision_reason, not a hardcoded script

    def test_sample_power_question_uses_real_numbers(self):
        reply = TemplateChatResponder().respond(_report(), "Is the sample size big enough?")
        assert "12,400 users" in reply
        assert "1.8% relative" in reply

    def test_falls_back_to_first_recommendation_for_unmatched_question(self):
        reply = TemplateChatResponder().respond(_report(), "What color should the button be?")
        assert "GO WITH CAUTION" in reply

    def test_never_fabricates_a_number_not_in_the_report(self):
        """Regression guard for the exact bug class this replaced: the old frontend mock
        hardcoded '12,400 users' / '980 users' regardless of the real dataset."""
        report = _report(sample_size_note="640 users — far below the ~8,200 required")
        reply = TemplateChatResponder().respond(report, "is the sample big enough?")
        assert "640 users" in reply
        assert "12,400" not in reply


    def test_significance_question_uses_reported_p_value_and_ci(self):
        reply = TemplateChatResponder().respond(_report(), "Was the result statistically significant?")
        assert "statistically significant" in reply
        assert "0.0008" in reply
        assert "+0.18pp" in reply
        assert "1.8% relative" in reply

    def test_practical_significance_question_uses_effect_and_mde(self):
        reply = TemplateChatResponder().respond(_report(), "Is the effect practically significant?")
        assert "+8.4% (rel)" in reply
        assert "1.8% relative" in reply
        assert "statistical significance alone is not enough" in reply

    def test_cuped_question_does_not_invent_adjusted_result(self):
        reply = TemplateChatResponder().respond(_report(), "What would happen if CUPED were applied?")
        assert "cannot be inferred" in reply
        assert "adjusted p-value" in reply
        assert "pre-experiment" in reply

    def test_metric_question_uses_authoritative_stat_metric(self):
        report = _report()
        reply = TemplateChatResponder().respond(report, "What is the primary metric?")
        assert "Conversion Rate" in reply

    def test_history_defaults_to_none_unchanged_behavior(self):
        """Existing call shape (no history arg) must keep working exactly as before."""
        reply = TemplateChatResponder().respond(_report(), "Should we ship this?")
        assert "GO_WITH_CAUTION" in reply

    def test_uses_prior_turn_to_disambiguate_an_unmatched_follow_up(self):
        """
        'what about that?' matches no keyword on its own — but combined
        with the prior user turn ('Should we ship this?') it should
        still land on the ship branch instead of the generic fallback.
        """
        history = [
            ChatMessage(id="u1", role=ChatRole.USER, content="Should we ship this?"),
            ChatMessage(id="a1", role=ChatRole.ASSISTANT, content="Decision: GO_WITH_CAUTION (recommendation confidence: MEDIUM). ..."),
        ]
        reply = TemplateChatResponder().respond(_report(), "what about that?", history)
        assert "GO_WITH_CAUTION" in reply

    def test_unrelated_follow_up_with_history_still_falls_back_to_grounded_default(self):
        """History widening must never invent a match that isn't actually there."""
        history = [ChatMessage(id="u1", role=ChatRole.USER, content="What was the conversion rate?")]
        reply = TemplateChatResponder().respond(_report(), "why was it significant?", history)
        # Significance is now a first-class grounded chat topic.
        assert "statistically significant" in reply
        assert "0.0008" in reply


class TestLLMChatResponder:
    def test_falls_back_to_template_on_llm_failure(self):
        with patch("app.llm.client.get_llm", side_effect=RuntimeError("no API key")):
            reply = LLMChatResponder().respond(_report(), "Should we ship this?")
        assert "GO_WITH_CAUTION" in reply  # template path still produced a grounded answer

    def test_uses_llm_output_when_available(self):
        structured = SimpleNamespace(invoke=lambda messages: {"parsed": SimpleNamespace(content="LLM-generated answer."), "raw": None, "parsing_error": None})
        fake_llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)
        with patch("app.llm.client.get_llm", return_value=fake_llm):
            reply = LLMChatResponder().respond(_report(), "Why medium confidence?")
        assert reply == "LLM-generated answer."

    def test_history_threaded_as_real_conversation_turns(self):
        """
        Prior turns must reach the LLM as actual user/assistant messages
        (not just prose), so pronoun references like 'it' resolve the
        way they would in any normal chat completion.
        """
        captured = {}

        def _capture(messages):
            captured["messages"] = messages
            return {"parsed": SimpleNamespace(content="ok"), "raw": None, "parsing_error": None}

        structured = SimpleNamespace(invoke=_capture)
        fake_llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)
        history = [
            ChatMessage(id="u1", role=ChatRole.USER, content="What was the conversion rate?"),
            ChatMessage(id="a1", role=ChatRole.ASSISTANT, content="The variant conversion rate was 4.56%."),
        ]
        with patch("app.llm.client.get_llm", return_value=fake_llm):
            LLMChatResponder().respond(_report(), "Why was it significant?", history)

        messages = captured["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "What was the conversion rate?"}
        assert messages[2] == {"role": "assistant", "content": "The variant conversion rate was 4.56%."}
        assert messages[-1] == {"role": "user", "content": "Why was it significant?"}

    def test_no_history_produces_single_user_turn(self):
        """No history passed (or empty) must behave exactly as before this change."""
        captured = {}

        def _capture(messages):
            captured["messages"] = messages
            return {"parsed": SimpleNamespace(content="ok"), "raw": None, "parsing_error": None}

        structured = SimpleNamespace(invoke=_capture)
        fake_llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)
        with patch("app.llm.client.get_llm", return_value=fake_llm):
            LLMChatResponder().respond(_report(), "Should we ship this?")

        assert len(captured["messages"]) == 2  # system + the one user turn, no history in between

    def test_falls_back_to_template_with_history_on_llm_failure(self):
        history = [
            ChatMessage(id="u1", role=ChatRole.USER, content="Should we ship this?"),
            ChatMessage(id="a1", role=ChatRole.ASSISTANT, content="Decision: GO_WITH_CAUTION (recommendation confidence: MEDIUM). ..."),
        ]
        with patch("app.llm.client.get_llm", side_effect=RuntimeError("no API key")):
            reply = LLMChatResponder().respond(_report(), "what about that?", history)
        assert "GO_WITH_CAUTION" in reply  # fallback still got the history-widened match

    def test_prompt_includes_real_stats_not_placeholders(self):
        captured = {}

        def _capture(messages):
            captured["system"] = messages[0]["content"]
            return {"parsed": SimpleNamespace(content="ok"), "raw": None, "parsing_error": None}

        structured = SimpleNamespace(invoke=_capture)
        fake_llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)
        with patch("app.llm.client.get_llm", return_value=fake_llm):
            LLMChatResponder().respond(_report(), "explain the p-value")

        assert "< 0.001" in captured["system"]  # 0.0008 floors per format_p_value, same rule report_generator uses
        assert "Do NOT invent, recalculate, or override" in captured["system"]

    def test_invalid_experiment_grounding_prompt_preserves_exact_conflict_count(self):
        """
        Regression for the exact reported bug: the report's structured
        QUALITY CHECKS fact (1,541 conflicting users) must reach the LLM
        prompt verbatim, and the prompt must explicitly forbid deriving
        a different number (e.g. 1,967) or inventing a cause.
        """
        captured = {}

        def _capture(messages):
            captured["system"] = messages[0]["content"]
            return {"parsed": SimpleNamespace(content="ok"), "raw": None, "parsing_error": None}

        structured = SimpleNamespace(invoke=_capture)
        fake_llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)

        report = _report(
            experiment_validity=ExperimentValidity.INVALID,
            decision=Decision.INVALID,
            decision_reason=(
                "Statistical testing was skipped because a critical experiment-validity check "
                "failed: 1,541 user(s) found assigned to MORE THAN ONE variant. Comparing the "
                "variants would produce an unreliable result."
            ),
            stats=[],
            quality_checks=[
                QualityCheck(
                    label="Duplicate User Variant Conflicts",
                    passed=False,
                    detail=(
                        "1,541 user(s) found assigned to MORE THAN ONE variant — this is not a "
                        "harmless duplicate, it indicates a broken randomization/assignment "
                        "pipeline. Results below cannot be trusted until this is fixed."
                    ),
                ),
                QualityCheck(
                    label="Duplicate User Rows",
                    passed=True,
                    detail="3,676 duplicate user_id row(s) found — deduplicated (kept first occurrence) before analysis",
                ),
            ],
        )

        with patch("app.llm.client.get_llm", return_value=fake_llm):
            LLMChatResponder().respond(report, "How many users had conflicting variant assignments?")

        system_prompt = captured["system"]
        # The exact structured fact must be present verbatim.
        assert "1,541" in system_prompt
        # A plausible invented/derived number must never appear.
        assert "1,967" not in system_prompt
        # The prompt must explicitly instruct against recomputation and invented causes.
        assert "never derive" in system_prompt.lower() or "never invent" in system_prompt.lower()
        assert "do not invent a cause" in system_prompt.lower() or "not invent a cause" in system_prompt.lower()
        # Validity + skip-reason guardrails must be present.
        assert "INVALID" in system_prompt
        assert "never say it was skipped merely because" in system_prompt.lower()


class TestBuildChatMessage:
    def test_returns_assistant_role_message_with_uuid_id(self):
        msg = build_chat_message(_report(), "Should we ship this?")
        assert msg.role.value == "assistant"
        assert len(msg.id) > 0
        assert "GO_WITH_CAUTION" in msg.content

    def test_passes_history_through_to_the_responder(self):
        history = [
            ChatMessage(id="u1", role=ChatRole.USER, content="Should we ship this?"),
            ChatMessage(id="a1", role=ChatRole.ASSISTANT, content="Decision: GO_WITH_CAUTION (recommendation confidence: MEDIUM). ..."),
        ]
        msg = build_chat_message(_report(), "what about that?", history)
        assert "GO_WITH_CAUTION" in msg.content
