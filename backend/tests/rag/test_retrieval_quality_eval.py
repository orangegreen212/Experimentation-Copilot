"""
RAG retrieval quality evaluation (task: "Improve the Experimentation RAG
Knowledge Base", item 6).

Uses the retriever's own default `min_score` (0.12) — the same bar the
pre-existing test suite in test_retriever.py already exercises — not
the app's stricter `stats_thresholds.kb_relevance_threshold` (0.20),
since these tests are about the knowledge base's retrievability in
general, not the specific evidence-selection policy in
knowledge_base_node.py (which has its own dedicated tests).
"""

from app.rag.retriever import KnowledgeBaseRetriever, load_documents

_RETRIEVER = KnowledgeBaseRetriever(load_documents())


def _top_headings(query: str, top_k: int = 5) -> list[str]:
    return [r.chunk.heading for r in _RETRIEVER.retrieve(query, top_k=top_k)]


def test_underpowered_query_is_relevant():
    headings = _top_headings("What does an underpowered experiment mean?")
    assert headings
    assert any("underpowered" in h.lower() or "power" in h.lower() for h in headings)


def test_mde_query_is_relevant():
    headings = _top_headings("How should MDE be interpreted?")
    assert headings
    assert any("mde" in h.lower() or "minimum detectable effect" in h.lower() for h in headings)


def test_srm_failure_query_is_relevant():
    headings = _top_headings("What should happen when SRM fails?")
    assert headings
    assert any("srm" in h.lower() or "sample ratio" in h.lower() for h in headings)


def test_multiple_comparisons_segmentation_query_is_relevant():
    headings = _top_headings("How should multiple comparisons affect segmentation analysis?")
    assert headings
    assert any("multiple comparisons" in h.lower() or "segmentation" in h.lower() for h in headings)


def test_significant_but_should_not_ship_query_is_relevant():
    headings = _top_headings("When should a statistically significant result not be shipped?")
    assert headings
    assert any(
        "practical significance" in h.lower() or "no-go" in h.lower() or "ship" in h.lower()
        for h in headings
    )


def test_practical_significance_query_is_relevant():
    headings = _top_headings("What is practical significance?")
    assert headings
    assert any("practical significance" in h.lower() for h in headings)


def test_statistical_vs_practical_significance_query_is_relevant():
    headings = _top_headings("What is the difference between statistical significance and practical significance?")
    assert headings
    assert any("significance" in h.lower() for h in headings)


def test_guardrail_regression_query_is_relevant():
    headings = _top_headings("How should guardrail regression affect the decision?")
    assert headings
    assert any("guardrail" in h.lower() for h in headings)


def test_unrelated_query_retrieves_nothing():
    """Irrelevant queries must not retrieve knowledge-base noise."""
    assert _RETRIEVER.retrieve("weather forecast Lisbon tomorrow") == []
    assert _RETRIEVER.retrieve("aquarium fish tank filter maintenance") == []
    assert _RETRIEVER.retrieve("electric guitar tuning pegs") == []


def test_unrelated_concept_does_not_pollute_srm_results():
    """A CUPED query shouldn't surface SRM content, and vice versa."""
    cuped_headings = _top_headings("Explain CUPED variance reduction", top_k=3)
    assert not any("sample ratio" in h.lower() for h in cuped_headings)

    srm_headings = _top_headings("What should happen when SRM fails?", top_k=3)
    assert not any("cuped" in h.lower() for h in srm_headings)
