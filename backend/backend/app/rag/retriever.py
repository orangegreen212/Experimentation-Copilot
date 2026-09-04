"""
Retriever — Stage 9 (Agentic RAG), deliberately dependency-light.

DECISION: no Chroma, no embeddings API. This retriever ranks
`docs/knowledge_base/*.md` chunks with BM25 (Okapi) — pure Python/math,
no numpy, no external ranking library. BM25 replaced an earlier plain
TF-IDF + cosine-similarity implementation (see git history) for two
concrete reasons that matter for THIS corpus specifically (short,
uneven-length markdown sections):

  - Term-frequency SATURATION (the `k1` parameter): a term appearing 5
    times in a chunk isn't "5x as relevant" as one appearing once —
    diminishing returns, closer to how relevance actually behaves.
    Raw TF-IDF has no such cap.
  - Document-length NORMALIZATION (the `b` parameter): chunks here vary
    a lot in length (a one-paragraph section vs. a several-paragraph
    one) — BM25 explicitly corrects for this; plain cosine similarity's
    L2-normalization does so much more crudely.

This is the same ranking function Elasticsearch/Lucene use by default,
for the same reason — it is a strict, well-established improvement
over TF-IDF+cosine for exactly this kind of small, heterogeneous
document set, while remaining:

  - Zero network calls, zero API keys, zero extra heavy dependencies
    (still pure Python — no embeddings model, no torch, no
    sentence-transformers).
  - Fully deterministic — same query always returns the same ranked
    chunks, trivially unit-testable without mocking an embeddings API.
  - Runs identically for anyone who clones the repo, same guarantee
    the project already makes for LangSmith tracing (Stage 8.1).

Still a genuine tradeoff vs. a real embedding model (BM25 is still
term-overlap-based, not semantic — "rental car" won't match "auto
lease"). If semantic retrieval quality becomes a real requirement
later, only this module changes — `KnowledgeBaseRetriever.retrieve()`'s
signature (query in, ranked `RetrievedChunk` list out) is the seam.

Score scale note: raw BM25 scores are unbounded (not naturally in
[0, 1] the way cosine similarity is). `retrieve()` normalizes each
query's scores by dividing by that query's own top raw score, so
`min_score` keeps meaning the same thing it always has here — "at
least this fraction of the best match this query got" — and every
existing caller/threshold (`stats_thresholds.kb_relevance_threshold`,
tests) keeps working unchanged.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "knowledge_base"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# BM25 Okapi standard defaults (same values used by Elasticsearch/Lucene).
# k1 controls term-frequency saturation; b controls document-length
# normalization strength (1.0 = full normalization, 0.0 = none).
_BM25_K1 = 1.5
_BM25_B = 0.75

# Minimal English stopword list — filters out function words that would
# otherwise create spurious term overlap between an unrelated query and
# every document (e.g. "is", "the", "in" appear in nearly all of them).
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "of", "to", "in",
    "on", "for", "with", "as", "at", "by", "from", "and", "or", "but",
    "if", "then", "than", "so", "not", "no", "do", "does", "did",
    "can", "could", "will", "would", "should", "what", "which", "who",
    "how", "when", "where", "why", "i", "you", "he", "she", "we",
    "they", "my", "your", "our", "their", "me", "us", "them",
    "like", "just", "also", "very", "get", "gets", "make", "makes",
})


@dataclass(frozen=True)
class DocumentChunk:
    """One retrievable unit — a single `## Heading` section of a knowledge base doc."""

    source: str  # filename, e.g. "kohavi.md"
    heading: str  # e.g. "Sample Ratio Mismatch (SRM)"
    content: str  # the section's body text


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


def _tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _stem(word: str) -> str:
    """
    Minimal, conservative suffix stripping (Porter-lite) — not a full
    Porter stemmer, just enough to match common morphological variants
    ("rigorous"/"rigor", "testing"/"tests") without a stemming library
    dependency. Never strips below 4 characters, to avoid collapsing
    short unrelated words together.
    """
    for suffix in ("ously", "ing", "tion", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def load_documents(docs_dir: Path = _DOCS_DIR) -> list[DocumentChunk]:
    """
    Splits every `*.md` file in `docs_dir` into chunks on `## ` headers.
    The `# Title` line is skipped (not retrievable on its own); each
    `## Heading` section becomes one chunk.
    """
    chunks: list[DocumentChunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        sections = re.split(r"\n(?=## )", text)
        for section in sections:
            section = section.strip()
            if not section.startswith("## "):
                continue  # skip the `# Title` preamble and any stray text before the first heading
            heading, _, body = section.partition("\n")
            heading = heading.removeprefix("## ").strip()
            body = body.strip()
            if body:
                chunks.append(DocumentChunk(source=path.name, heading=heading, content=body))
    return chunks


class KnowledgeBaseRetriever:
    """
    Scores every chunk against a query with BM25 (Okapi) — see module
    docstring for why this replaced plain TF-IDF+cosine. `retrieve()`
    normalizes each query's raw BM25 scores by that query's own top
    score, so the returned `score` stays comparable to the previous
    cosine-similarity scale (0 to 1, 1 = the best match this query got)
    and every existing `min_score` threshold keeps its meaning.
    """

    def __init__(self, chunks: list[DocumentChunk], k1: float = _BM25_K1, b: float = _BM25_B):
        if not chunks:
            raise ValueError("KnowledgeBaseRetriever requires at least one document chunk.")
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._doc_term_counts = [Counter(_tokenize(f"{c.heading} {c.content}")) for c in chunks]
        self._doc_lengths = [sum(counts.values()) for counts in self._doc_term_counts]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        n_docs = len(self._doc_term_counts)
        doc_freq: Counter = Counter()
        for counts in self._doc_term_counts:
            doc_freq.update(counts.keys())
        # Okapi BM25 IDF (Robertson–Sparck Jones), "+1" variant — keeps
        # IDF non-negative even for a term that appears in most/all
        # documents of a small corpus (the classic unmodified formula
        # can go negative there, which would let a very common term
        # actively PENALIZE a score — wrong for a KB this size).
        return {
            term: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

    def _bm25_raw_score(self, query_terms: list[str], doc_idx: int) -> float:
        counts = self._doc_term_counts[doc_idx]
        doc_len = self._doc_lengths[doc_idx]
        length_norm = 1 - self._b + self._b * (doc_len / self._avg_doc_length if self._avg_doc_length else 0.0)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term)
            if not tf:
                continue
            idf = self._idf.get(term, 0.0)
            score += idf * (tf * (self._k1 + 1)) / (tf + self._k1 * length_norm)
        return score

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.12) -> list[RetrievedChunk]:
        """
        Returns up to `top_k` chunks with (normalized) score >=
        `min_score`, ranked highest-first. `min_score` exists so an
        unrelated query (no term overlap with the knowledge base at
        all) returns an empty list rather than the top-k least-bad
        noise matches — the caller (knowledge_base_node) should treat
        an empty result as "nothing relevant found," not as an error.
        """
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        raw_scores = [self._bm25_raw_score(query_terms, i) for i in range(len(self._chunks))]
        max_raw = max(raw_scores) if raw_scores else 0.0
        if max_raw <= 0:
            # No chunk shares any query term at all — genuinely
            # unrelated query, not just "everything scored low".
            return []

        scored = [
            RetrievedChunk(chunk=chunk, score=raw / max_raw)
            for chunk, raw in zip(self._chunks, raw_scores)
        ]
        scored = [r for r in scored if r.score >= min_score]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


@lru_cache(maxsize=1)
def get_retriever() -> KnowledgeBaseRetriever:
    """Loads and fits the retriever once per process — the knowledge base is static, no need to rebuild per request."""
    chunks = load_documents()
    return KnowledgeBaseRetriever(chunks)
