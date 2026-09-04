"""
Regression guard for retrieval quality — runs the same 24-query
labeled eval set as scripts/evaluate_retrieval.py and asserts accuracy
stays above a floor. If someone edits docs/knowledge_base/ or tweaks
the retriever's scoring and accuracy drops, this test catches it
instead of it being noticed only by someone manually re-running the
script.
"""

from scripts.evaluate_retrieval import evaluate_retrieval


def test_retrieval_top3_accuracy_above_floor():
    """
    Floor set at 85%, just below the measured 87.5% on the current
    eval set — not at an aspirational 90%+. The 3 known misses are
    documented, deliberate limitations of a pure TF-IDF retriever (no
    semantic understanding): abstract paraphrases with genuinely low
    lexical overlap with the target chunk's wording (e.g. "too good to
    be true" vs. "Twyman's Law"'s actual text) will not be found by
    term-overlap scoring, regardless of tuning. A semantic embedding
    model would likely catch these — that's the real tradeoff of the
    project's "no embeddings API, no network call" design decision
    (see app/rag/retriever.py's module docstring).
    """
    result = evaluate_retrieval()
    assert result.top3_accuracy >= 0.85, (
        f"Top-3 accuracy dropped to {result.top3_accuracy:.1%} "
        f"(floor: 85%). Misses: {result.misses}"
    )


def test_retrieval_top1_accuracy_above_floor():
    result = evaluate_retrieval()
    assert result.top1_accuracy >= 0.70, (
        f"Top-1 accuracy dropped to {result.top1_accuracy:.1%} (floor: 70%)"
    )


def test_retrieval_mrr_above_floor():
    """
    MRR floor at 0.80, just below the measured ~0.83 — catches a
    regression where the correct chunk still appears in the top-3 but
    slips from rank 1 to rank 2/3 more often (a ranking-quality
    regression Hit@3 alone wouldn't flag, since Hit@3 treats every
    top-3 position as equally good).
    """
    result = evaluate_retrieval()
    assert result.mrr >= 0.80, f"MRR dropped to {result.mrr:.3f} (floor: 0.80)"


def test_retrieval_ndcg3_above_floor():
    result = evaluate_retrieval()
    assert result.ndcg_at_3 >= 0.80, f"NDCG@3 dropped to {result.ndcg_at_3:.3f} (floor: 0.80)"


def test_retrieval_latency_is_fast():
    """TF-IDF over a 16-chunk corpus should be sub-millisecond-to-low-millisecond — no network, no embeddings API."""
    result = evaluate_retrieval()
    assert result.avg_latency_ms < 50, f"Average retrieval latency {result.avg_latency_ms:.2f}ms is unexpectedly high"
