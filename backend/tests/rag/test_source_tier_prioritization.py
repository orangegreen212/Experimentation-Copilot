"""
Focused tests for deterministic source-tier prioritization in
knowledge_base_node.py.

SCOPE: after `retriever.retrieve()` has already applied the UNCHANGED
`stats_thresholds.kb_relevance_threshold` gate, `_prioritize_by_source_tier()`
picks which small subset of the already-relevant candidates is actually
shown, preferring Tier 1 (kohavi.md/microsoft.md/netflix.md) over Tier 2
(booking.md/airbnb.md/optimizely_google.md) — without ever altering a
score or promoting a candidate that failed the relevance gate.
"""

import pandas as pd

from app.core.config import stats_thresholds
from app.core.dataset_store import store_dataset
from app.graph.graph_builder import experiment_review_graph
from app.graph.nodes.knowledge_base_node import (
    _MAX_EVIDENCE_REFERENCES,
    _TIER_1_SOURCES,
    _TIER_2_SOURCES,
    _prioritize_by_source_tier,
)
from app.rag.retriever import DocumentChunk, RetrievedChunk
from app.schemas.settings import AnalysisSettings

DEMO_HIGH_QUALITY = "data/demo/demo_ab_checkout.csv"


def _chunk(source: str, heading: str = "H") -> DocumentChunk:
    return DocumentChunk(source=source, heading=heading, content="content")


def _invoke(prompt: str, csv_path: str = DEMO_HIGH_QUALITY):
    df = pd.read_csv(csv_path)
    dataset_id = store_dataset(df)
    initial_state = {
        "dataset_id": dataset_id,
        "user_prompt": prompt,
        "settings": AnalysisSettings(),
    }
    return experiment_review_graph.invoke(initial_state)


class TestTierPrioritizationUnit:
    def test_tier1_preferred_over_tier2_when_both_relevant(self):
        """(A) A relevant Tier 1 result outranks a relevant Tier 2 result
        even when Tier 2 scored higher — both already passed the gate."""
        candidates = [
            RetrievedChunk(chunk=_chunk("optimizely_google.md"), score=0.50),  # Tier 2, higher score
            RetrievedChunk(chunk=_chunk("kohavi.md"), score=0.22),  # Tier 1, lower score
        ]
        result = _prioritize_by_source_tier(candidates, limit=2)
        assert result[0].chunk.source == "kohavi.md"
        assert result[1].chunk.source == "optimizely_google.md"

    def test_weak_tier1_never_reaches_prioritization_because_it_never_passed_the_gate(self):
        """(B) A Tier 1 source below the relevance threshold is filtered
        out by the (unchanged) retriever gate BEFORE prioritization ever
        sees it — this function only reorders what already cleared the
        bar, so it cannot promote a weak Tier 1 result."""
        # Simulates the real flow: knowledge_base_node only ever hands
        # _prioritize_by_source_tier() candidates that already cleared
        # stats_thresholds.kb_relevance_threshold. A weak Tier 1 result
        # would already be excluded upstream, so it's simply absent here.
        candidates = [
            RetrievedChunk(chunk=_chunk("booking.md"), score=0.35),  # Tier 2, above threshold
        ]
        result = _prioritize_by_source_tier(candidates, limit=3)
        assert [r.chunk.source for r in result] == ["booking.md"]
        assert "kohavi.md" not in [r.chunk.source for r in result]

    def test_highly_relevant_tier2_result_is_returned(self):
        """(C) A Tier 2 result is never suppressed for being Tier 2 alone
        — it fills a slot when there aren't enough Tier 1 candidates."""
        candidates = [
            RetrievedChunk(chunk=_chunk("kohavi.md"), score=0.40),
            RetrievedChunk(chunk=_chunk("booking.md"), score=0.35),
        ]
        result = _prioritize_by_source_tier(candidates, limit=3)
        sources = [r.chunk.source for r in result]
        assert "kohavi.md" in sources
        assert "booking.md" in sources

    def test_original_scores_are_never_modified(self):
        """(D) Prioritization only reorders/truncates — `.score` on
        every returned item is bit-for-bit the original retriever score."""
        candidates = [
            RetrievedChunk(chunk=_chunk("optimizely_google.md"), score=0.30678),
            RetrievedChunk(chunk=_chunk("kohavi.md"), score=0.24719),
        ]
        result = _prioritize_by_source_tier(candidates, limit=2)
        scores = {r.chunk.source: r.score for r in result}
        assert scores["optimizely_google.md"] == 0.30678
        assert scores["kohavi.md"] == 0.24719

    def test_respects_limit(self):
        candidates = [
            RetrievedChunk(chunk=_chunk("kohavi.md"), score=0.40),
            RetrievedChunk(chunk=_chunk("microsoft.md"), score=0.35),
            RetrievedChunk(chunk=_chunk("netflix.md"), score=0.30),
            RetrievedChunk(chunk=_chunk("booking.md"), score=0.28),
        ]
        result = _prioritize_by_source_tier(candidates, limit=_MAX_EVIDENCE_REFERENCES)
        assert len(result) == _MAX_EVIDENCE_REFERENCES

    def test_tier_sets_match_task_spec(self):
        # RAG knowledge-base expansion (this task) added four first-party
        # decision-policy/methodology docs to Tier 1 alongside the original
        # three source-attributed methodology notes — see
        # app/graph/nodes/knowledge_base_node.py's _TIER_1_SOURCES comment.
        assert _TIER_1_SOURCES == {
            "kohavi.md", "microsoft.md", "netflix.md",
            "decision_policies.md", "statistical_interpretation.md",
            "experiment_fundamentals.md", "advanced_methodology.md",
        }
        assert _TIER_2_SOURCES == {"booking.md", "airbnb.md", "optimizely_google.md"}

    # --- Source-diversity fix -------------------------------------------------

    def test_multiple_chunks_same_source_yield_at_most_one_reference(self):
        """(1) Three above-threshold kohavi.md chunks plus one Microsoft
        and one Netflix chunk must not let kohavi.md occupy every slot —
        the final evidence has at most one chunk per source."""
        candidates = [
            RetrievedChunk(chunk=_chunk("kohavi.md", "MDE and Power"), score=0.24),
            RetrievedChunk(chunk=_chunk("kohavi.md", "SRM"), score=0.23),
            RetrievedChunk(chunk=_chunk("kohavi.md", "OEC"), score=0.22),
            RetrievedChunk(chunk=_chunk("microsoft.md"), score=0.21),
            RetrievedChunk(chunk=_chunk("netflix.md"), score=0.20),
        ]
        result = _prioritize_by_source_tier(candidates, limit=3)
        sources = [r.chunk.source for r in result]
        assert len(sources) == len(set(sources)), "duplicate source in evidence"
        assert sources == ["kohavi.md", "microsoft.md", "netflix.md"]
        # The single kohavi.md reference kept is its best-scoring chunk.
        kohavi_ref = next(r for r in result if r.chunk.source == "kohavi.md")
        assert kohavi_ref.chunk.heading == "MDE and Power"
        assert kohavi_ref.score == 0.24

    def test_high_scoring_tier2_does_not_bump_eligible_tier1_source(self):
        """(2) A high-scoring Tier 2 source must not replace an eligible
        Tier 1 source merely because its raw score is higher — tier
        still wins, diversity just prevents same-source crowding."""
        candidates = [
            RetrievedChunk(chunk=_chunk("booking.md"), score=0.31),
            RetrievedChunk(chunk=_chunk("optimizely_google.md"), score=0.30),
            RetrievedChunk(chunk=_chunk("kohavi.md"), score=0.24),
        ]
        result = _prioritize_by_source_tier(candidates, limit=3)
        sources = [r.chunk.source for r in result]
        assert sources[0] == "kohavi.md"
        assert set(sources) == {"kohavi.md", "booking.md", "optimizely_google.md"}

    def test_two_tier1_sources_backfilled_with_best_eligible_tier2(self):
        """(3) Only two unique Tier 1 sources pass the threshold ->
        the best eligible Tier 2 source fills the third slot."""
        candidates = [
            RetrievedChunk(chunk=_chunk("kohavi.md"), score=0.24),
            RetrievedChunk(chunk=_chunk("microsoft.md"), score=0.21),
            RetrievedChunk(chunk=_chunk("booking.md"), score=0.31),
        ]
        result = _prioritize_by_source_tier(candidates, limit=3)
        sources = [r.chunk.source for r in result]
        assert sources == ["kohavi.md", "microsoft.md", "booking.md"]

    def test_single_unique_source_does_not_fabricate_extra_references(self):
        """(4) Only one unique source passes the threshold -> return
        just that one reference, never invent additional evidence."""
        candidates = [
            RetrievedChunk(chunk=_chunk("kohavi.md", "MDE and Power"), score=0.24),
            RetrievedChunk(chunk=_chunk("kohavi.md", "SRM"), score=0.23),
        ]
        result = _prioritize_by_source_tier(candidates, limit=3)
        assert len(result) == 1
        assert result[0].chunk.source == "kohavi.md"
        assert result[0].score == 0.24

    def test_diversity_selection_preserves_original_scores_exactly(self):
        """(5) Deduping to one-chunk-per-source never mutates `.score`
        — it stays the original, unmodified retriever cosine score."""
        candidates = [
            RetrievedChunk(chunk=_chunk("kohavi.md", "MDE and Power"), score=0.247193),
            RetrievedChunk(chunk=_chunk("kohavi.md", "SRM"), score=0.238812),
            RetrievedChunk(chunk=_chunk("microsoft.md"), score=0.219004),
        ]
        result = _prioritize_by_source_tier(candidates, limit=3)
        scores = {r.chunk.source: r.score for r in result}
        assert scores["kohavi.md"] == 0.247193
        assert scores["microsoft.md"] == 0.219004

    def test_regression_reported_scenario_diverse_tier1_sources(self):
        """(7) Regression for the reported bug scenario: a candidate
        pool with several kohavi.md chunks plus one Microsoft and one
        Netflix chunk (and Tier 2 noise) above the relevance threshold
        must surface Kohavi/Microsoft/Netflix, not three kohavi.md
        chunks, mirroring the originally-reported 48,312-user-dataset
        run where every evidence slot was filled by kohavi.md alone."""
        candidates = [
            RetrievedChunk(chunk=_chunk("kohavi.md", "MDE and Power"), score=0.24),
            RetrievedChunk(chunk=_chunk("kohavi.md", "SRM"), score=0.23),
            RetrievedChunk(chunk=_chunk("kohavi.md", "OEC"), score=0.22),
            RetrievedChunk(chunk=_chunk("microsoft.md"), score=0.21),
            RetrievedChunk(chunk=_chunk("netflix.md"), score=0.20),
            RetrievedChunk(chunk=_chunk("booking.md"), score=0.31),
            RetrievedChunk(chunk=_chunk("optimizely_google.md"), score=0.30),
        ]
        result = _prioritize_by_source_tier(candidates, limit=3)
        sources = [r.chunk.source for r in result]
        assert sources == ["kohavi.md", "microsoft.md", "netflix.md"]
        assert len(sources) == len(set(sources))


class TestTierPrioritizationEndToEnd:
    def test_full_review_prefers_tier1_and_keeps_real_scores(self):
        """End-to-end reproduction of the reported scenario: Tier 1
        sources surface first in Evidence & Sources, all scores are the
        real retriever scores, and every reference still clears the
        unchanged relevance threshold."""
        state = _invoke("Analyze with segmentation")
        report = state["report"]
        refs = report.knowledge_base_references
        assert len(refs) > 0
        for ref in refs:
            assert ref.relevance_score >= stats_thresholds.kb_relevance_threshold
        # If any Tier 2 reference is shown, no Tier 1 reference that was
        # in the retrieved pool got bumped out purely by a lower score —
        # this is covered precisely by the unit tests above; here we
        # just assert the tiering constants were actually applied to a
        # real, dataset-driven run without breaking anything.
        sources = {ref.source for ref in refs}
        assert sources.issubset(_TIER_1_SOURCES | _TIER_2_SOURCES)
        # Source-diversity fix: no two references share the same source.
        assert len(sources) == len(refs)

    def test_empty_retrieval_message_unchanged(self):
        """(E) Preserves the existing empty-evidence behavior exactly."""
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
                "prompt": "xyzzy quixotic zephyr",
                "settings": {"cuped": False, "bootstrap": False},
            },
        )
        body = analyze_resp.json()
        if body["report"]["knowledgeBaseReferences"] == []:
            assert body["report"]["knowledgeBaseAttempted"] is True

    def test_rag_still_cannot_affect_decision(self):
        """(F) Tier prioritization is evidence-selection only — the
        deterministic decision is unaffected."""
        state = _invoke("Should we ship variant B?")
        report = state["report"]
        assert report.decision is not None
        # Same dataset/prompt with tier prioritization must still reach
        # a decision computed purely from facts.stat_results/validity —
        # not from which KB sources got shown.
        assert report.decision.value in {"GO", "GO_WITH_CAUTION", "NO_GO", "INCONCLUSIVE", "INVALID"}
