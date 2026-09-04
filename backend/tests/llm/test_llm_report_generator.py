"""
LLMReportGenerator tests — mocked LLM client, no real network call.
Verifies the project's core architectural guarantee: the LLM can
change TEXT (executive_summary, confidence_reason, next_steps) but can
NEVER change NUMBERS (confidence level/stars, stats, mde,
sample_size_note, quality_checks) — those must be byte-identical to
what deterministic Python computed, regardless of what the LLM
returns. `recommendations` is no longer LLM-authored text at all —
see report_generator.deterministic_recommendations_for_decision — so
any LLM-proposed recommendations text is simply never read.
"""

from types import SimpleNamespace
from unittest.mock import patch

from app.graph.report_generator import LLMReportGenerator, ReportFacts, TemplateReportGenerator
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.quality import QualityCheck
from app.schemas.statistics import HypothesisTestType, PowerAnalysisResult, StatResult


def _fake_llm(response):
    structured = SimpleNamespace(invoke=lambda messages: {"parsed": response, "raw": None, "parsing_error": None})
    return SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)


def _high_confidence_facts() -> ReportFacts:
    return ReportFacts(
        user_prompt="Evaluate the checkout redesign",
        dataset=DatasetInfo(type=DatasetType.AGGREGATED_AB_TEST, variants=2, users=12400, metric_label="Conversion Rate", metric_selection_reason="Selected by the deterministic outcome-column priority — no competing outcome metrics were available in this dataset."),
        quality_checks=[QualityCheck(label="Sample Ratio Mismatch (SRM)", passed=True, detail="p=0.83")],
        srm_passed=True,
        stat_results=[
            StatResult(
                metric="Conversion Rate", test_type=HypothesisTestType.CHI_SQUARE, test_name="Chi-square test",
                statistic=12.4, selection_reason="binary metric", control="4.21%", variant="4.56%",
                delta="+8.4% (rel)", delta_relative=8.4, p_value=0.0003, significant=True, ci_lower="+0.18pp", ci_upper="+0.52pp",
            )
        ],
        test_selections=[],
        power_analysis=PowerAnalysisResult(
            minimum_detectable_effect_relative=1.8, required_sample_size=8200, observed_sample_size=12400,
            achieved_power=0.91, alpha=0.05, is_sufficiently_powered=True,
        ),
        mde_display="1.8%",
        sample_size_note="12,400 users — exceeds requirement",
    )


class TestLLMReportGeneratorNumericBoundary:
    """The LLM's response is never allowed to change any numeric/structured field."""

    def test_stats_are_copied_unchanged_from_facts(self):
        facts = _high_confidence_facts()
        response = SimpleNamespace(
            executive_summary="LLM summary", confidence_reason="LLM confidence text",
            recommendations=["LLM rec 1"], next_steps=["LLM next step"],
        )
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            report = LLMReportGenerator().generate(facts)

        assert report.stats == facts.stat_results
        assert report.quality_checks == facts.quality_checks
        assert report.mde == facts.mde_display
        assert report.sample_size_note == facts.sample_size_note

    def test_confidence_matches_deterministic_assessment_not_llm(self):
        """Even if we could smuggle a different confidence into the LLM response, the schema doesn't expose that field — so this tests the schema boundary itself is correct."""
        facts = _high_confidence_facts()
        response = SimpleNamespace(
            executive_summary="x", confidence_reason="y", recommendations=["z"], next_steps=["w"],
        )
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            report = LLMReportGenerator().generate(facts)

        expected_confidence, expected_stars, _ = TemplateReportGenerator()._assess_confidence(facts)
        assert report.confidence == expected_confidence
        assert report.confidence_stars == expected_stars

    def test_text_fields_come_from_llm_response(self):
        facts = _high_confidence_facts()
        response = SimpleNamespace(
            executive_summary="Custom LLM executive summary.",
            confidence_reason="Custom LLM confidence reasoning.",
            recommendations=["Custom rec A", "Custom rec B"],
            next_steps=["Custom step A"],
        )
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            report = LLMReportGenerator().generate(facts)

        assert report.executive_summary == "Custom LLM executive summary."
        assert report.confidence_reason == "Custom LLM confidence reasoning."
        assert report.next_steps == ["Custom step A"]

    def test_recommendations_are_always_deterministic_never_from_llm(self):
        """
        The actual fix for the "LLM can smuggle a SHIP recommendation"
        risk: `recommendations` is never populated from the LLM
        response at all, for any confidence/decision level — not just
        filtered after the fact. This response's ["Custom rec A", ...]
        must never appear anywhere in the final report.
        """
        facts = _high_confidence_facts()
        response = SimpleNamespace(
            executive_summary="x", confidence_reason="y",
            recommendations=["Custom rec A", "Custom rec B"], next_steps=["z"],
        )
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            report = LLMReportGenerator().generate(facts)

        assert "Custom rec A" not in report.recommendations
        assert "Custom rec B" not in report.recommendations
        assert report.recommendations[0].strip().upper().startswith(("GO", "NO-GO", "INCONCLUSIVE", "INVALID"))


class TestLLMReportGeneratorFallback:
    def test_exception_falls_back_to_template_report_exactly(self):
        facts = _high_confidence_facts()

        def raise_error(*args, **kwargs):
            raise RuntimeError("network error")

        with patch("app.llm.client.get_llm", side_effect=raise_error):
            llm_report = LLMReportGenerator().generate(facts)

        template_report = TemplateReportGenerator().generate(facts)
        # PHASE 8 — the fallback is now visible via `report_fallback_reason`
        # (previously this silent fallback made an LLM failure look
        # identical to a normal template report). Every OTHER field
        # must still match the plain template report exactly — the
        # fallback changes observability, never the underlying numbers
        # or text.
        assert llm_report.report_fallback_reason is not None
        assert "network error" in llm_report.report_fallback_reason
        assert llm_report.model_copy(update={"report_fallback_reason": None}) == template_report

    def test_typeerror_none_not_iterable_is_logged_with_full_traceback(self, caplog):
        """
        Regression for the real, previously-unfixable bug report: this
        exact error ("TypeError: 'NoneType' object is not iterable")
        recurred across multiple unrelated real datasets/metric types/
        arm counts, but its traceback was NEVER captured — the catch
        site only logged `str(exc)` via `log.warning("...%s...", exc)`,
        which discards the traceback entirely. That made it impossible
        to tell whether the bug was in our code, LangChain,
        langchain-openai, or OpenRouter's response parsing.

        This test reproduces the exact exception type/message (via a
        mocked `get_llm` raising it, standing in for wherever inside
        `llm.invoke(...)` it actually originates) and asserts the log
        record now carries the traceback (`exc_info`) — the minimal
        diagnostic fix — while proving the fallback's USER-VISIBLE
        behavior is byte-for-byte unchanged from any other exception
        type (see the test above): same report, same
        report_fallback_reason wording, no new None/empty semantics
        introduced anywhere.
        """
        import logging

        facts = _high_confidence_facts()

        def raise_the_real_bug(*args, **kwargs):
            raise TypeError("'NoneType' object is not iterable")

        with caplog.at_level(logging.WARNING, logger="app.graph.report_generator"):
            with patch("app.llm.client.get_llm", side_effect=raise_the_real_bug):
                llm_report = LLMReportGenerator().generate(facts)

        template_report = TemplateReportGenerator().generate(facts)
        assert llm_report.report_fallback_reason is not None
        assert "TypeError" in llm_report.report_fallback_reason
        assert "'NoneType' object is not iterable" in llm_report.report_fallback_reason
        # Fallback behavior itself is unaffected by the diagnostic change.
        assert llm_report.model_copy(update={"report_fallback_reason": None}) == template_report

        # The actual regression check: the traceback must now be
        # present in the log record, not merely the exception's str().
        matching_records = [r for r in caplog.records if "LLM report generation failed" in r.getMessage()]
        assert len(matching_records) == 1
        record = matching_records[0]
        assert record.exc_info is not None, (
            "traceback was not captured — this is exactly the gap that made the "
            "real 'NoneType' object is not iterable bug unfixable from logs alone"
        )
        assert record.exc_info[0] is TypeError

    def test_conceptual_path_never_calls_llm(self):
        """A pure conceptual (RAG-only) question must skip the LLM entirely — no mock needed since get_llm must never be called."""
        facts = ReportFacts(
            user_prompt="What is CUPED?",
            dataset=DatasetInfo(type=DatasetType.AGGREGATED_AB_TEST, variants=2, users=100, metric_label="x", metric_selection_reason="Selected by the deterministic outcome-column priority — no competing outcome metrics were available in this dataset."),
            quality_checks=[], srm_passed=True, stat_results=[], test_selections=[],
            power_analysis=None, mde_display="N/A", sample_size_note="N/A",
            validation_ran=False, kb_results=[],
        )

        with patch("app.llm.client.get_llm") as mock_get_llm:
            report = LLMReportGenerator().generate(facts)
            mock_get_llm.assert_not_called()

        assert report.confidence.value == "MEDIUM"
        assert report.stats == []


class TestLLMReportGeneratorMethodologyContext:
    """
    Stage 10 — `facts.kb_results` must actually reach the LLM prompt on
    the main statistical path (not just the conceptual-only path), and
    the numeric boundary must hold even when methodology is present.
    """

    def _facts_with_kb(self) -> ReportFacts:
        from app.rag.retriever import DocumentChunk, RetrievedChunk

        facts = _high_confidence_facts()
        facts.kb_results = [
            RetrievedChunk(
                chunk=DocumentChunk(
                    source="kohavi.md",
                    heading="Underpowered Experiments",
                    content="A non-significant result from an underpowered experiment is not evidence of no effect.",
                ),
                score=0.42,
            )
        ]
        return facts

    def test_methodology_context_included_in_prompt(self):
        facts = self._facts_with_kb()
        captured = {}

        def _capture_invoke(messages):
            captured["system"] = messages[0]["content"]
            return {
                "parsed": SimpleNamespace(
                    executive_summary="ok", confidence_reason="ok",
                    recommendations=["ok"], next_steps=["ok"],
                ),
                "raw": None,
                "parsing_error": None,
            }

        structured = SimpleNamespace(invoke=_capture_invoke)
        fake_llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)

        with patch("app.llm.client.get_llm", return_value=fake_llm):
            LLMReportGenerator().generate(facts)

        assert "kohavi.md" in captured["system"]
        assert "Underpowered Experiments" in captured["system"]
        assert "Do NOT invent statistics" in captured["system"]
        assert "Do NOT" in captured["system"] and "recalculate" in captured["system"]

    def test_no_kb_results_omits_methodology_block_entirely(self):
        """No retrieved guidance -> no forced/fabricated methodology section."""
        facts = _high_confidence_facts()
        assert facts.kb_results == []
        captured = {}

        def _capture_invoke(messages):
            captured["system"] = messages[0]["content"]
            return {
                "parsed": SimpleNamespace(
                    executive_summary="ok", confidence_reason="ok",
                    recommendations=["ok"], next_steps=["ok"],
                ),
                "raw": None,
                "parsing_error": None,
            }

        structured = SimpleNamespace(invoke=_capture_invoke)
        fake_llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)

        with patch("app.llm.client.get_llm", return_value=fake_llm):
            LLMReportGenerator().generate(facts)

        assert "METHODOLOGY" not in captured["system"]

    def test_numeric_fields_unchanged_when_methodology_present(self):
        """The numeric boundary (core architectural guarantee) holds even with kb_results populated."""
        facts = self._facts_with_kb()
        response = SimpleNamespace(
            executive_summary="LLM summary", confidence_reason="LLM confidence text",
            recommendations=["LLM rec citing Kohavi"], next_steps=["LLM next step"],
        )
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            report = LLMReportGenerator().generate(facts)

        assert report.stats == facts.stat_results
        assert report.mde == facts.mde_display
        assert report.sample_size_note == facts.sample_size_note
        assert report.confidence == TemplateReportGenerator()._assess_confidence(facts)[0]

    def test_knowledge_base_references_populated_on_llm_path(self):
        facts = self._facts_with_kb()
        response = SimpleNamespace(
            executive_summary="ok", confidence_reason="ok", recommendations=["ok"], next_steps=["ok"],
        )
        with patch("app.llm.client.get_llm", return_value=_fake_llm(response)):
            report = LLMReportGenerator().generate(facts)

        assert len(report.knowledge_base_references) == 1
        assert report.knowledge_base_references[0].source == "kohavi.md"


class TestTemplateReportGeneratorMethodologyContext:
    """Same guarantees as above, but for the no-LLM deterministic fallback path (TemplateReportGenerator)."""

    def _facts_with_kb(self) -> ReportFacts:
        from app.rag.retriever import DocumentChunk, RetrievedChunk

        facts = _high_confidence_facts()
        facts.kb_results = [
            RetrievedChunk(
                chunk=DocumentChunk(source="kohavi.md", heading="Underpowered Experiments", content="..."),
                score=0.42,
            )
        ]
        return facts

    def test_retrieved_source_surfaces_in_knowledge_base_references(self):
        """
        Was: asserted "kohavi.md" appeared inside `recommendations` text.
        `recommendations` is now always the deterministic decision
        template (see deterministic_recommendations_for_decision) and
        never cites a source — the retrieved chunk surfaces in the
        structured `knowledge_base_references` field instead, which the
        frontend renders separately.
        """
        facts = self._facts_with_kb()
        report = TemplateReportGenerator().generate(facts)
        assert report.knowledge_base_references[0].source == "kohavi.md"
        assert report.knowledge_base_references[0].heading == "Underpowered Experiments"

    def test_no_kb_results_no_methodology_recommendation(self):
        facts = _high_confidence_facts()
        report = TemplateReportGenerator().generate(facts)
        assert report.knowledge_base_references == []
        assert not any("kohavi.md" in r or "Methodology guidance" in r for r in report.recommendations)

    def test_stats_untouched_by_methodology(self):
        """Template path: methodology only appends a recommendation line, never touches stats/mde/etc."""
        facts = self._facts_with_kb()
        report = TemplateReportGenerator().generate(facts)
        assert report.stats == facts.stat_results
        assert report.mde == facts.mde_display
