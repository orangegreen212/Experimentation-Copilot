"""
Tests for evaluation/evaluators/rag_evaluator.py.

Covers task spec section 11: "RAG retrieval evaluation" — exercised
against the REAL `get_retriever()` (the same deterministic TF-IDF
retriever the production knowledge_base_node uses), not a mock, since
retrieval quality is deterministic and CI-safe to assert on directly
(same pattern as the existing scripts/evaluate_retrieval.py).
"""

from __future__ import annotations

from app.rag.retriever import get_retriever
from evaluation.evaluators.rag_evaluator import RAG_EVAL_SET, RagCase, evaluate_rag, evaluate_rag_case


def test_rag_eval_runs_and_produces_bounded_metrics():
    report = evaluate_rag()
    assert report.n_queries == len(RAG_EVAL_SET)
    for metric in (report.avg_precision_at_k, report.avg_recall_at_k, report.avg_context_relevance, report.avg_answer_faithfulness):
        assert 0.0 <= metric <= 1.0


def test_a_clearly_relevant_query_achieves_full_recall():
    case = RagCase(
        query="How can pre-experiment data reduce variance in my results?",
        relevant_chunks=[("netflix.md", "CUPED")],
        expected_answer_facts=["cuped"],
    )
    result = evaluate_rag_case(case, get_retriever(), k=3)
    assert result.recall_at_k == 1.0
    assert result.answer_faithfulness == 1.0


def test_an_irrelevant_gibberish_query_still_returns_bounded_scores():
    """Retrieval always returns its top-k best-effort matches (TF-IDF
    doesn't have a 'no match' state) — this test locks in that a
    nonsense query still produces valid [0,1]-bounded precision/recall,
    rather than crashing or returning an out-of-range score."""
    case = RagCase(
        query="zzz qqq xyzzy plugh unrelated nonsense",
        relevant_chunks=[("kohavi.md", "Sample Ratio Mismatch")],
        expected_answer_facts=["chi-square"],
    )
    result = evaluate_rag_case(case, get_retriever(), k=3)
    assert 0.0 <= result.precision_at_k <= 1.0
    assert 0.0 <= result.recall_at_k <= 1.0
    assert 0.0 <= result.answer_faithfulness <= 1.0
