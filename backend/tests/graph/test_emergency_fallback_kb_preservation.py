"""
Focused tests for the emergency-fallback evidence-preservation fix in
`build_emergency_fallback_report()` (backend/app/graph/report_generator.py).

SCOPE: `decision_node.py` wraps `generator.generate(facts)` in a
try/except; if report NARRATION fails for any reason, the graph must
still return a report, via `build_emergency_fallback_report()`. Before
this fix, that fallback unconditionally discarded already-retrieved,
already-relevance-gated KB evidence (`knowledge_base_references=[]`,
`knowledge_base_attempted` left at its schema default of `False`) even
though `facts.kb_results`/`facts.kb_attempted` were already fully
populated by the time `generator.generate()` was ever called. These
tests confirm the fallback now preserves that evidence exactly, and
that everything else about the fallback (decision-safety, stats,
quality checks) is unaffected.
"""

from app.graph.report_generator import ReportFacts, build_emergency_fallback_report
from app.rag.retriever import DocumentChunk, RetrievedChunk
from app.schemas.dataset import DatasetInfo, DatasetType


def _chunk(source: str, heading: str, content: str = "content") -> DocumentChunk:
    return DocumentChunk(source=source, heading=heading, content=content)


def _dataset(users: int = 48_312) -> DatasetInfo:
    return DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=users,
        metric_label="Conversion Rate",
        metric_selection_reason="Selected by the deterministic outcome-column priority because no specific metric was requested.",
    )


def _facts(kb_results=None, **overrides) -> ReportFacts:
    defaults = dict(
        user_prompt="Analyze this experiment end-to-end",
        dataset=_dataset(),
        quality_checks=[],
        srm_passed=True,
        stat_results=[],
        test_selections=[],
        power_analysis=None,
        mde_display="N/A — no hypothesis test was run",
        sample_size_note="N/A — no hypothesis test was run",
        kb_results=kb_results,
    )
    defaults.update(overrides)
    return ReportFacts(**defaults)


class TestEmergencyFallbackPreservesKbEvidence:
    def test_three_kb_results_preserved(self):
        """(1) Emergency fallback with 3 KB results: all 3 appear in
        `knowledge_base_references`, and `knowledge_base_attempted` is True."""
        kb_results = [
            RetrievedChunk(chunk=_chunk("kohavi.md", "Minimum Detectable Effect (MDE) and Power", "MDE content"), score=0.24),
            RetrievedChunk(chunk=_chunk("microsoft.md", "Guardrail Metrics", "Guardrail content"), score=0.21),
            RetrievedChunk(chunk=_chunk("netflix.md", "CUPED", "CUPED content"), score=0.20),
        ]
        facts = _facts(kb_results=kb_results)

        report = build_emergency_fallback_report(facts, ValueError("boom"))

        assert len(report.knowledge_base_references) == 3
        assert report.knowledge_base_attempted is True
        sources = {r.source for r in report.knowledge_base_references}
        assert sources == {"kohavi.md", "microsoft.md", "netflix.md"}

    def test_empty_kb_results_but_attempted(self):
        """(2) Emergency fallback with an empty (but attempted) KB
        result list: references stay empty, attempted stays True."""
        facts = _facts(kb_results=[])  # retrieval ran, nothing cleared the threshold

        report = build_emergency_fallback_report(facts, RuntimeError("boom"))

        assert report.knowledge_base_references == []
        assert report.knowledge_base_attempted is True

    def test_kb_never_attempted(self):
        """(3) Emergency fallback when the knowledge_base node never ran
        for this request at all: references empty, attempted False."""
        facts = _facts(kb_results=None)  # knowledge_base node never ran

        report = build_emergency_fallback_report(facts, RuntimeError("boom"))

        assert report.knowledge_base_references == []
        assert report.knowledge_base_attempted is False

    def test_source_and_relevance_score_preserved_exactly(self):
        """(4) `source` is the real retrieved filename and
        `relevance_score` is the original (rounded-for-display, per the
        existing convention shared by every other report path) score —
        nothing invented, nothing renamed."""
        kb_results = [
            RetrievedChunk(chunk=_chunk("kohavi.md", "MDE and Power", "MDE excerpt text"), score=0.247193),
        ]
        facts = _facts(kb_results=kb_results)

        report = build_emergency_fallback_report(facts, ValueError("boom"))

        ref = report.knowledge_base_references[0]
        assert ref.source == "kohavi.md"
        assert ref.heading == "MDE and Power"
        assert ref.excerpt == "MDE excerpt text"
        # Same `round(r.score, 3)` convention as TemplateReportGenerator/
        # LLMReportGenerator — not the raw float, not re-derived.
        assert ref.relevance_score == round(0.247193, 3)

    def test_decision_unaffected_by_kb_evidence(self):
        """(5) The deterministic decision made by the emergency fallback
        depends only on validity/quality-check facts, never on whether
        KB evidence is present — same `facts` produce the same decision
        with or without kb_results."""
        from app.schemas.quality import QualityCheck

        quality_checks = [
            QualityCheck(label="SRM Check", passed=False, critical=True, detail="SRM failed", informational=False),
        ]
        facts_with_kb = _facts(
            kb_results=[RetrievedChunk(chunk=_chunk("kohavi.md", "SRM"), score=0.30)],
            srm_passed=False,
            quality_checks=quality_checks,
        )
        facts_without_kb = _facts(kb_results=None, srm_passed=False, quality_checks=quality_checks)

        report_with_kb = build_emergency_fallback_report(facts_with_kb, ValueError("boom"))
        report_without_kb = build_emergency_fallback_report(facts_without_kb, ValueError("boom"))

        assert report_with_kb.decision == report_without_kb.decision
        assert report_with_kb.experiment_validity == report_without_kb.experiment_validity
        assert report_with_kb.decision_reason == report_without_kb.decision_reason

    def test_regression_48312_user_aggregated_scenario(self):
        """(6) Regression for the reported scenario: a 48,312-user
        aggregated A/B test dataset where report narration raised and
        the emergency fallback fired — the 3 already-retrieved KB
        references (top match: MDE and Power, score 0.24) must survive
        into the final report instead of being silently discarded."""
        kb_results = [
            RetrievedChunk(chunk=_chunk("kohavi.md", "Minimum Detectable Effect (MDE) and Power", "MDE content"), score=0.24),
            RetrievedChunk(chunk=_chunk("microsoft.md", "Guardrail Metrics", "Guardrail content"), score=0.21),
            RetrievedChunk(chunk=_chunk("netflix.md", "CUPED", "CUPED content"), score=0.20),
        ]
        facts = _facts(kb_results=kb_results, dataset=_dataset(users=48_312))

        report = build_emergency_fallback_report(facts, RuntimeError("simulated narration failure"))

        assert report.knowledge_base_attempted is True
        assert len(report.knowledge_base_references) == 3
        top = report.knowledge_base_references[0]
        assert top.source == "kohavi.md"
        assert top.heading == "Minimum Detectable Effect (MDE) and Power"
        assert top.relevance_score == 0.24
        # report_fallback_reason is the observability signal that this
        # WAS the emergency-fallback path, not a normal or LLM-specific one.
        assert report.report_fallback_reason is not None
        assert "RuntimeError" in report.report_fallback_reason
