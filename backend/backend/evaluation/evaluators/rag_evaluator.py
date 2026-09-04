"""
RAG evaluation — Layer 5 of the evaluation strategy.

The project already has a deterministic TF-IDF retriever
(`app/rag/retriever.py`) and its own regression eval
(`scripts/evaluate_retrieval.py`, Hit@1/Hit@3/MRR/NDCG@3). This module
does NOT replace either — it wraps the SAME retriever
(`get_retriever()`) with:

  1. A small labelled RAG dataset (question, expected relevant
     chunk(s), expected answer facts) — extending `evaluate_retrieval`'s
     one-relevant-chunk-per-query eval set with multi-relevant-chunk
     support (precision@k/recall@k need >=1 relevant item to be
     meaningful, and are typically reported alongside a query that can
     have more than one correct chunk).
  2. `context_relevance` — of the top-k retrieved chunks, what
     fraction are actually relevant (same numerator/denominator as
     precision@k, kept as a separately-named metric because it's the
     conventional RAG-eval term reviewers will look for).
  3. `answer_faithfulness` — a deterministic, non-LLM proxy: does the
     retrieved chunk's content actually CONTAIN the expected answer
     facts (as substrings/keywords)? This checks "is the ground truth
     actually present in what was retrieved", i.e. whether an
     LLM answer grounded in this context COULD be faithful — it does
     not grade any actual generated answer text (there is no RAG
     answer-generation step in this codebase to grade; retrieval feeds
     the report generator's prompt context only — see
     report_generator.py's `kb_results` usage).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import stats_thresholds
from app.rag.retriever import KnowledgeBaseRetriever, RetrievedChunk, get_retriever


@dataclass
class RagCase:
    query: str
    # A chunk is "relevant" if its (source, heading-substring) pair
    # matches ANY entry here — supports queries with more than one
    # correct chunk, unlike scripts/evaluate_retrieval.py's exactly-one
    # convention.
    relevant_chunks: list[tuple[str, str]]  # (source, heading_substring)
    # Facts/keywords that a faithful answer to `query` must be
    # supportable by — checked against retrieved chunk CONTENT, not
    # against any generated text (see module docstring).
    expected_answer_facts: list[str]


# A focused labelled subset (not a duplicate of scripts/evaluate_retrieval.py's
# 24-query set) built specifically to exercise precision/recall@k with
# multi-relevant-chunk cases and answer-faithfulness checking.
RAG_EVAL_SET: list[RagCase] = [
    RagCase(
        query="How do I know if my A/B test randomization is broken?",
        relevant_chunks=[("kohavi.md", "Sample Ratio Mismatch")],
        expected_answer_facts=["chi-square", "expected"],
    ),
    RagCase(
        query="What should I watch besides my primary success metric to avoid shipping something harmful?",
        relevant_chunks=[("booking.md", "Guardrail Metrics")],
        expected_answer_facts=["guardrail"],
    ),
    RagCase(
        query="How can pre-experiment data reduce variance in my results?",
        relevant_chunks=[("netflix.md", "CUPED")],
        expected_answer_facts=["cuped", "variance"],
    ),
    RagCase(
        query="What sample size do I need to detect a given effect size at 80% power?",
        relevant_chunks=[("kohavi.md", "Minimum Detectable Effect")],
        expected_answer_facts=["power", "sample size"],
    ),
    RagCase(
        query="Should I use Welch's t-test or assume equal variance?",
        relevant_chunks=[("microsoft.md", "Welch's t-test")],
        expected_answer_facts=["welch", "variance"],
    ),
]


@dataclass
class RagCaseResult:
    query: str
    precision_at_k: float
    recall_at_k: float
    context_relevance: float
    answer_faithfulness: float  # fraction of expected_answer_facts found in retrieved content
    retrieved_headings: list[str]


def _is_relevant(chunk_ref: tuple[str, str], relevant: list[tuple[str, str]]) -> bool:
    source, heading = chunk_ref
    return any(source == exp_source and exp_heading.lower() in heading.lower() for exp_source, exp_heading in relevant)


def evaluate_rag_case(
    case: RagCase, retriever: KnowledgeBaseRetriever, k: int = 3
) -> RagCaseResult:
    # Gate at the SAME threshold the real product uses
    # (`stats_thresholds.kb_relevance_threshold`, applied at
    # knowledge_base_node.py) rather than retriever.retrieve()'s own
    # generic/permissive default — otherwise this eval measures a
    # looser, hypothetical retrieval pool that users never actually
    # see, rather than production's real precision@k.
    results: list[RetrievedChunk] = retriever.retrieve(
        case.query, top_k=k, min_score=stats_thresholds.kb_relevance_threshold
    )

    relevant_retrieved = [
        r for r in results if _is_relevant((r.chunk.source, r.chunk.heading), case.relevant_chunks)
    ]
    precision = len(relevant_retrieved) / len(results) if results else 0.0
    n_relevant_total = len(case.relevant_chunks)
    recall = len(relevant_retrieved) / n_relevant_total if n_relevant_total else 0.0

    combined_content = " ".join(r.chunk.content.lower() for r in results)
    facts_found = sum(1 for fact in case.expected_answer_facts if fact.lower() in combined_content)
    faithfulness = facts_found / len(case.expected_answer_facts) if case.expected_answer_facts else 1.0

    return RagCaseResult(
        query=case.query,
        precision_at_k=precision,
        recall_at_k=recall,
        context_relevance=precision,  # same computation, conventional alias — see module docstring
        answer_faithfulness=faithfulness,
        retrieved_headings=[r.chunk.heading for r in results],
    )


@dataclass
class RagEvalReport:
    n_queries: int
    avg_precision_at_k: float
    avg_recall_at_k: float
    avg_context_relevance: float
    avg_answer_faithfulness: float
    case_results: list[RagCaseResult] = field(default_factory=list)


def evaluate_rag(cases: list[RagCase] = RAG_EVAL_SET, k: int = 3) -> RagEvalReport:
    retriever = get_retriever()
    results = [evaluate_rag_case(c, retriever, k=k) for c in cases]
    n = len(results)
    return RagEvalReport(
        n_queries=n,
        avg_precision_at_k=sum(r.precision_at_k for r in results) / n if n else 0.0,
        avg_recall_at_k=sum(r.recall_at_k for r in results) / n if n else 0.0,
        avg_context_relevance=sum(r.context_relevance for r in results) / n if n else 0.0,
        avg_answer_faithfulness=sum(r.answer_faithfulness for r in results) / n if n else 0.0,
        case_results=results,
    )
