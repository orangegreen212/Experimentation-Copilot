import pytest

from app.rag.retriever import DocumentChunk, KnowledgeBaseRetriever, load_documents


def test_load_documents_finds_all_kb_files():
    chunks = load_documents()
    sources = {c.source for c in chunks}
    # Original six source-attributed methodology docs, plus the four
    # docs added by the RAG knowledge-base expansion (this task):
    # explicit decision policies, statistical interpretation, core
    # experiment fundamentals, and advanced methodology.
    assert sources == {
        "kohavi.md", "booking.md", "netflix.md", "microsoft.md",
        "airbnb.md", "optimizely_google.md",
        "decision_policies.md", "statistical_interpretation.md",
        "experiment_fundamentals.md", "advanced_methodology.md",
    }
    assert len(chunks) > 10  # each doc has several ## sections


def test_load_documents_chunk_has_heading_and_content():
    chunks = load_documents()
    # Two docs now cover SRM by design (kohavi.md's original note, plus
    # experiment_fundamentals.md's fundamentals-level treatment) — filter
    # by source to keep this test unambiguous.
    srm_chunk = next(
        c for c in chunks if c.source == "kohavi.md" and "Sample Ratio Mismatch" in c.heading
    )
    assert srm_chunk.source == "kohavi.md"
    assert "chi-square" in srm_chunk.content.lower()


def test_retrieve_finds_relevant_chunk_for_srm_query():
    chunks = load_documents()
    retriever = KnowledgeBaseRetriever(chunks)
    results = retriever.retrieve("What is Sample Ratio Mismatch?")
    assert len(results) > 0
    assert "SRM" in results[0].chunk.heading or "Sample Ratio" in results[0].chunk.heading


def test_retrieve_finds_relevant_chunk_for_cuped_query():
    chunks = load_documents()
    retriever = KnowledgeBaseRetriever(chunks)
    results = retriever.retrieve("Explain CUPED variance reduction")
    assert len(results) > 0
    assert "CUPED" in results[0].chunk.heading


def test_retrieve_returns_empty_for_unrelated_query():
    chunks = load_documents()
    retriever = KnowledgeBaseRetriever(chunks)
    results = retriever.retrieve("What is the weather like in Lisbon today?")
    assert results == []


def test_retrieve_respects_top_k():
    chunks = load_documents()
    retriever = KnowledgeBaseRetriever(chunks)
    results = retriever.retrieve("statistical test significance p-value", top_k=2)
    assert len(results) <= 2


def test_retrieve_scores_are_descending():
    chunks = load_documents()
    retriever = KnowledgeBaseRetriever(chunks)
    results = retriever.retrieve("Welch t-test variance")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# --- metadata (this task) --------------------------------------------------


def test_decision_policy_chunks_have_metadata():
    chunks = load_documents()
    underpowered = next(
        c for c in chunks if c.source == "decision_policies.md" and "UNDERPOWERED" in c.heading
    )
    assert underpowered.metadata["category"] == "decision_policy"
    assert underpowered.metadata["concept"] == "underpowered"
    assert underpowered.metadata["document_type"] == "decision_rule"
    assert underpowered.metadata["priority"] == "high"
    # the metadata comment line must not leak into the retrievable content
    assert "<!--" not in underpowered.content


def test_legacy_chunks_without_metadata_still_load():
    chunks = load_documents()
    srm_chunk = next(c for c in chunks if c.source == "kohavi.md" and "Sample Ratio Mismatch" in c.heading)
    assert srm_chunk.metadata == {}


def test_retrieve_can_filter_by_category():
    chunks = load_documents()
    retriever = KnowledgeBaseRetriever(chunks)
    results = retriever.retrieve(
        "underpowered insufficient evidence", top_k=5, min_score=0.0, category="decision_policy"
    )
    assert results
    assert all(r.chunk.metadata.get("category") == "decision_policy" for r in results)


def test_empty_chunks_raises():
    with pytest.raises(ValueError):
        KnowledgeBaseRetriever([])


def test_retriever_is_deterministic():
    chunks = load_documents()
    retriever = KnowledgeBaseRetriever(chunks)
    r1 = retriever.retrieve("guardrail metrics")
    r2 = retriever.retrieve("guardrail metrics")
    assert [(r.chunk.heading, r.score) for r in r1] == [(r.chunk.heading, r.score) for r in r2]


def test_synthetic_chunks_no_network_or_files_needed():
    """Retriever works on in-memory chunks too — doesn't require the real docs/ directory."""
    chunks = [
        DocumentChunk(source="a.md", heading="Apples", content="Apples are a fruit that grows on trees."),
        DocumentChunk(source="a.md", heading="Cars", content="Cars are vehicles with wheels and an engine."),
    ]
    retriever = KnowledgeBaseRetriever(chunks)
    results = retriever.retrieve("fruit growing on trees")
    assert len(results) == 1
    assert results[0].chunk.heading == "Apples"
