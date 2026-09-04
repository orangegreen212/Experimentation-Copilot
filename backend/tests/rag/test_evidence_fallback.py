"""
Covers:

  5. Evidence retrieval success — real retrieved chunks are shown as
     ordinary (non-fallback) references.
  6. Evidence retrieval failure — a genuine retriever exception is
     reported (`knowledge_base_retrieval_error` on the report), not
     silently swallowed into an indistinguishable empty result, and
     the report still explains SRM via the built-in fallback.
  7. No fabricated citations when retrieval fails — the only reference
     shown in that case is the clearly-labeled system fallback
     (`is_system_fallback=True`, `source="system-methodology"`); no
     invented source/heading/excerpt is ever produced.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from app.core.dataset_store import store_dataset
from app.graph.graph_builder import experiment_review_graph
from app.graph.report_generator import (
    ReportFacts,
    TemplateReportGenerator,
    _SRM_METHODOLOGY_FALLBACK_REFERENCE,
    apply_evidence_fallback,
)
from app.rag.retriever import DocumentChunk, RetrievedChunk
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.report import KnowledgeBaseReference
from app.schemas.settings import AnalysisSettings


def _dataset_info() -> DatasetInfo:
    return DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=1000,
        metric_label="Conversion Rate",
        metric_selection_reason="test fixture",
    )


def _facts(srm_passed: bool, kb_results=None, kb_error=None) -> ReportFacts:
    return ReportFacts(
        user_prompt="Evaluate this experiment",
        dataset=_dataset_info(),
        quality_checks=[],
        srm_passed=srm_passed,
        stat_results=[],
        test_selections=[],
        power_analysis=None,
        mde_display="N/A",
        sample_size_note="N/A",
        kb_results=kb_results,
        kb_error=kb_error,
    )


def _ab_dataset(n_control: int, n_treatment: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for arm, n, rate in [("control", n_control, 0.10), ("treatment", n_treatment, 0.14)]:
        conv = rng.binomial(1, rate, n)
        for c in conv:
            rows.append({"user_id": uid, "group": arm, "converted": int(c)})
            uid += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Requirement 5 — evidence retrieval SUCCESS
# ---------------------------------------------------------------------------


class TestEvidenceRetrievalSuccess:
    def test_srm_pass_with_real_results_shows_only_real_references(self):
        chunk = RetrievedChunk(
            chunk=DocumentChunk(source="kohavi.md", heading="Some Topic", content="Some real retrieved content."),
            score=0.5,
        )
        facts = _facts(srm_passed=True, kb_results=[chunk])
        report = TemplateReportGenerator().generate(facts)

        assert len(report.knowledge_base_references) == 1
        assert report.knowledge_base_references[0].source == "kohavi.md"
        assert report.knowledge_base_references[0].is_system_fallback is False
        assert report.knowledge_base_retrieval_error is None

    def test_srm_fail_with_an_srm_relevant_result_does_not_add_fallback(self):
        """If retrieval genuinely found SRM-relevant content, the
        built-in fallback must not be piled on top of it."""
        chunk = RetrievedChunk(
            chunk=DocumentChunk(
                source="kohavi.md",
                heading="Sample Ratio Mismatch (SRM)",
                content="SRM detects broken randomization via a chi-square test.",
            ),
            score=0.8,
        )
        facts = _facts(srm_passed=False, kb_results=[chunk])
        report = TemplateReportGenerator().generate(facts)

        assert len(report.knowledge_base_references) == 1
        assert report.knowledge_base_references[0].is_system_fallback is False

    def test_end_to_end_srm_pass_retrieves_real_evidence(self):
        """Full graph run against the real knowledge base."""
        df = _ab_dataset(1000, 1000)
        dataset_id = store_dataset(df)
        final_state = experiment_review_graph.invoke(
            {
                "dataset_id": dataset_id,
                "user_prompt": "Should we ship this experiment? Evaluate it.",
                "settings": AnalysisSettings(),
            }
        )
        report = final_state["report"]
        assert report.knowledge_base_attempted is True
        assert report.knowledge_base_retrieval_error is None
        assert len(report.knowledge_base_references) > 0
        assert all(not r.is_system_fallback for r in report.knowledge_base_references)


# ---------------------------------------------------------------------------
# Requirement 6 — evidence retrieval FAILURE is reported, not hidden
# ---------------------------------------------------------------------------


class TestEvidenceRetrievalFailure:
    def test_retrieval_error_is_surfaced_on_the_report_not_swallowed(self):
        facts = _facts(srm_passed=False, kb_results=[], kb_error="RuntimeError: index not built")
        report = TemplateReportGenerator().generate(facts)

        assert report.knowledge_base_retrieval_error == "RuntimeError: index not built"
        # The report still explains SRM despite the failed retrieval —
        # never just an empty references list with no explanation.
        assert len(report.knowledge_base_references) >= 1

    def test_end_to_end_retriever_exception_is_captured_gracefully(self):
        """A real retriever blow-up must degrade gracefully (kb_results=[])
        and be recorded as an actual failure — not confused with a
        legitimate "nothing relevant" empty result."""
        df = _ab_dataset(950, 50)  # SRM fails, so Evidence & Sources is load-bearing
        dataset_id = store_dataset(df)

        def _boom():
            raise RuntimeError("simulated retriever outage")

        with patch("app.graph.nodes.knowledge_base_node.get_retriever", side_effect=_boom):
            final_state = experiment_review_graph.invoke(
                {
                    "dataset_id": dataset_id,
                    "user_prompt": "Evaluate this experiment",
                    "settings": AnalysisSettings(),
                }
            )

        report = final_state["report"]
        assert report.srm_warning is True
        assert report.knowledge_base_retrieval_error is not None
        assert "simulated retriever outage" in report.knowledge_base_retrieval_error


# ---------------------------------------------------------------------------
# Requirement 7 — no fabricated citations when retrieval fails
# ---------------------------------------------------------------------------


class TestNoFabricatedCitationsOnFailure:
    def test_only_reference_on_failure_is_the_labeled_system_fallback(self):
        facts = _facts(srm_passed=False, kb_results=[], kb_error="ConnectionError: boom")
        report = TemplateReportGenerator().generate(facts)

        assert len(report.knowledge_base_references) == 1
        ref = report.knowledge_base_references[0]
        assert ref.is_system_fallback is True
        assert ref.source == "system-methodology"
        assert ref.relevance_score == 0.0
        # Never presented as if it came from the real KB doc set.
        assert ref.source not in {"kohavi.md", "microsoft.md", "netflix.md", "booking.md", "airbnb.md", "optimizely_google.md"}

    def test_fallback_reference_is_never_added_when_srm_passed(self):
        """The fallback exists specifically to explain SRM on an
        SRM-failed report — it must never appear when SRM passed, even
        if retrieval genuinely found nothing."""
        facts = _facts(srm_passed=True, kb_results=[], kb_error="boom")
        report = TemplateReportGenerator().generate(facts)
        assert report.knowledge_base_references == []

    def test_apply_evidence_fallback_never_mutates_real_references(self):
        real = KnowledgeBaseReference(
            source="netflix.md", heading="Other Topic", excerpt="Unrelated real content.", relevance_score=0.4
        )
        facts = _facts(srm_passed=False)
        result = apply_evidence_fallback(facts, [real])
        assert real in result
        assert _SRM_METHODOLOGY_FALLBACK_REFERENCE in result
        assert len(result) == 2

    def test_apply_evidence_fallback_is_noop_when_srm_passed(self):
        facts = _facts(srm_passed=True)
        assert apply_evidence_fallback(facts, []) == []
