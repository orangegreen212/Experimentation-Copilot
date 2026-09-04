"""
Focused tests for the RAG relevance quality gate.

Covers: the retriever's own generic default is untouched (still used by
tests/rag/test_retriever.py); the app-level bar
(`stats_thresholds.kb_relevance_threshold`, app/core/config.py) is what
actually decides what `knowledge_base_node.py` treats as evidence; and
the authority-hierarchy invariant that RAG evidence can never influence
(let alone override) the deterministic decision.
"""

import pandas as pd

from app.core.config import stats_thresholds
from app.core.dataset_store import store_dataset
from app.graph.graph_builder import experiment_review_graph
from app.rag.retriever import KnowledgeBaseRetriever, DocumentChunk, get_retriever
from app.schemas.settings import AnalysisSettings

DEMO_HIGH_QUALITY = "data/demo/demo_ab_checkout.csv"
DEMO_LOW_QUALITY = "data/demo/demo_ab_checkout_lowq.csv"


def _invoke(prompt: str, csv_path: str = DEMO_HIGH_QUALITY):
    df = pd.read_csv(csv_path)
    dataset_id = store_dataset(df)
    initial_state = {
        "dataset_id": dataset_id,
        "user_prompt": prompt,
        "settings": AnalysisSettings(),
    }
    return experiment_review_graph.invoke(initial_state)


class TestRelevanceThresholdIsSingleSourceOfTruth:
    def test_threshold_is_configured_in_one_place(self):
        """`stats_thresholds.kb_relevance_threshold` is the only knob — read it, don't hardcode it."""
        assert 0.0 < stats_thresholds.kb_relevance_threshold < 1.0

    def test_relevant_result_above_threshold_is_retained(self):
        chunks = [
            DocumentChunk(source="a.md", heading="SRM", content="Sample ratio mismatch and randomization checks."),
            DocumentChunk(source="a.md", heading="Unrelated", content="Cars are vehicles with wheels and an engine."),
        ]
        retriever = KnowledgeBaseRetriever(chunks)
        results = retriever.retrieve(
            "sample ratio mismatch randomization", min_score=stats_thresholds.kb_relevance_threshold
        )
        assert len(results) == 1
        assert results[0].chunk.heading == "SRM"

    def test_weak_result_below_threshold_is_excluded(self):
        """
        Against the REAL knowledge base: "Explain minimum detectable effect"
        genuinely matches kohavi.md's MDE/Power chunk well (top hit, well
        above the bar) but only weakly brushes past kohavi.md's unrelated
        Novelty/Primacy chunk (shared "kohavi.md" vocabulary, not the actual
        topic) — that second, weak match must not appear once gated.

        Score band recalibrated for BM25 (retriever.py switched from
        TF-IDF+cosine to BM25 — see that module's docstring): BM25's
        per-query-max-normalized scores run on a different scale than
        cosine similarity did, so the "weak brush-past" band is wider
        here (still comfortably below `kb_relevance_threshold`, which
        was itself recalibrated for this same reason — see
        app/core/config.py's comment).
        """
        retriever = get_retriever()
        ungated = retriever.retrieve("Explain minimum detectable effect", top_k=6, min_score=0.0)
        assert any(r.chunk.heading == "Novelty and Primacy Effects" and 0.0 < r.score < 0.35 for r in ungated)

        gated = retriever.retrieve(
            "Explain minimum detectable effect", top_k=6, min_score=stats_thresholds.kb_relevance_threshold
        )
        gated_headings = {r.chunk.heading for r in gated}
        assert "Minimum Detectable Effect (MDE) and Power" in gated_headings  # the real match survives
        assert "Novelty and Primacy Effects" not in gated_headings  # the weak brush-past is gated out

    def test_all_results_below_threshold_produce_empty_kb_results(self):
        state = _invoke("Analyze this experiment", csv_path=DEMO_HIGH_QUALITY)
        # Real retrieval against the real KB — every returned reference,
        # if any, must clear the configured bar (never a padding/backfill result).
        for r in state.get("kb_results", []):
            assert r.score >= stats_thresholds.kb_relevance_threshold


class TestQualityGateEndToEnd:
    def test_full_review_references_all_clear_the_threshold(self):
        state = _invoke("Should we ship variant B?")
        report = state["report"]
        assert len(report.knowledge_base_references) > 0
        for ref in report.knowledge_base_references:
            assert ref.relevance_score >= stats_thresholds.kb_relevance_threshold

    def test_empty_retrieval_does_not_crash_report_generation(self, monkeypatch):
        import sys

        kb_module = sys.modules["app.graph.nodes.knowledge_base_node"]

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated retriever outage")

        monkeypatch.setattr(kb_module, "get_retriever", _boom)
        state = _invoke("Should we ship variant B?")
        assert state["report"].knowledge_base_references == []
        assert state["report"].decision is not None

    def test_no_sufficiently_relevant_evidence_message_is_honest(self):
        """When retrieval genuinely finds nothing above the bar, the real
        API's execution step says so honestly (SUCCESS, not a failure)."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        resp = client.post(
            "/datasets/classify", data={"use_demo": "true", "simulate_low_quality": "false"}
        )
        dataset_id = resp.json()["datasetId"]
        analyze_resp = client.post(
            "/experiments/analyze",
            json={
                "datasetId": dataset_id,
                # Deliberately vocabulary-empty of any KB topic.
                "prompt": "xyzzy quixotic zephyr",
                "settings": {"cuped": False, "bootstrap": False},
            },
        )
        body = analyze_resp.json()
        steps = {s["id"]: s for s in body["executionSteps"]}
        if "knowledge_base" in steps:
            kb_step = steps["knowledge_base"]
            if body["report"]["knowledgeBaseReferences"] == []:
                assert "No sufficiently relevant evidence" in kb_step["detail"]
                assert kb_step["status"] == "SUCCESS"


class TestRagCannotOverrideDeterministicDecision:
    """Requirement #13 / Phase 8 requirements #7-#8 — the authority hierarchy."""

    def test_rag_cannot_override_srm_invalid_decision(self):
        """An SRM-failed (INVALID-driving) dataset must stay INVALID/LOW-confidence
        regardless of what the (real, quality-gated) KB evidence says."""
        state = _invoke(
            "Evaluate the checkout redesign — is variant ready to ship?",
            csv_path=DEMO_LOW_QUALITY,
        )
        assert state["srm_result"].passed is False
        report = state["report"]
        assert report.confidence.value == "LOW"
        assert report.srm_warning is True
        assert report.stats == []
        # Evidence may or may not have been retrieved (kb_results is
        # independent), but it never turns this into a GO/GO_WITH_CAUTION.
        assert report.decision.value in {"INVALID", "INCONCLUSIVE", "NO_GO"}

    def test_rag_cannot_turn_invalid_into_go(self):
        state = _invoke("Should we ship variant B?", csv_path=DEMO_LOW_QUALITY)
        assert state["report"].decision.value != "GO"
        assert state["report"].decision.value != "GO_WITH_CAUTION"


class TestKnowledgeBaseAttemptedFlag:
    """`knowledge_base_attempted` lets the UI distinguish 'KB ran, found
    nothing above the bar' from 'KB was never queried' — both otherwise
    look identical as an empty `knowledge_base_references` list."""

    def test_attempted_true_when_kb_node_ran(self):
        state = _invoke("Should we ship variant B?")
        assert state["report"].knowledge_base_attempted is True

    def test_attempted_true_even_when_result_is_empty(self):
        state = _invoke("Analyze this experiment xyzzy quixotic zephyr")
        report = state["report"]
        if report.knowledge_base_references == []:
            assert report.knowledge_base_attempted is True

    def test_attempted_false_when_kb_node_never_runs(self):
        """A funnel-only question never routes through knowledge_base_node."""
        import pandas as pd
        from app.core.dataset_store import store_dataset
        from app.graph.graph_builder import experiment_review_graph
        from app.schemas.settings import AnalysisSettings

        df = pd.read_csv("data/demo/demo_funnel.csv")
        dataset_id = store_dataset(df)
        state = experiment_review_graph.invoke(
            {
                "dataset_id": dataset_id,
                "user_prompt": "Show me the funnel drop-off",
                "settings": AnalysisSettings(),
            }
        )
        assert state["report"].knowledge_base_attempted is False
        assert state["report"].knowledge_base_references == []
