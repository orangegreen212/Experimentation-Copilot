"""
Retriever — Stage 9 (Agentic RAG), deliberately dependency-light.

DECISION: no Chroma, no embeddings API. This retriever is a from-scratch
TF-IDF + cosine-similarity implementation over `docs/knowledge_base/*.md`,
using only numpy. That's a genuine tradeoff (a real embedding model
would retrieve based on semantic similarity, not just term overlap),
but it means:

  - Zero network calls, zero API keys, zero extra heavy dependencies.
  - Fully deterministic — same query always returns the same ranked
    chunks, trivially unit-testable without mocking an embeddings API.
  - Runs identically for anyone who clones the repo, same guarantee
    the project already makes for LangSmith tracing (Stage 8.1).

If semantic retrieval quality becomes a real requirement later, only
this module changes — `KnowledgeBaseRetriever.retrieve()`'s signature
(query in, ranked `RetrievedChunk` list out) is the seam.
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
    metadata: dict = None  # type: ignore[assignment]  # e.g. {"category": "decision_policy", "concept": "underpowered", "document_type": "decision_rule", "priority": "high"}

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


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


_METADATA_LINE_RE = re.compile(r"^<!--\s*(.*?)\s*-->\s*$")


def _parse_metadata_line(line: str) -> dict[str, str] | None:
    """
    Parses an optional `<!-- key: value; key2: value2 -->` comment line
    immediately after a `## Heading` into a metadata dict (e.g.
    `category`, `concept`, `document_type`, `priority` — see task spec
    section 3). Returns `None` if `line` isn't a metadata comment, so
    the caller can tell "no metadata" apart from "empty metadata" and
    leave pre-existing docs (which have none) working unchanged.
    """
    m = _METADATA_LINE_RE.match(line.strip())
    if not m:
        return None
    meta: dict[str, str] = {}
    for pair in m.group(1).split(";"):
        if ":" not in pair:
            continue
        key, _, value = pair.partition(":")
        key, value = key.strip(), value.strip()
        if key:
            meta[key] = value
    return meta


def load_documents(docs_dir: Path = _DOCS_DIR) -> list[DocumentChunk]:
    """
    Splits every `*.md` file in `docs_dir` into chunks on `## ` headers.
    The `# Title` line is skipped (not retrievable on its own); each
    `## Heading` section becomes one chunk.

    METADATA (task spec section 3): if the first line of a section's
    body is an HTML-comment metadata line (`<!-- category: ...; concept:
    ...; document_type: ...; priority: ... -->`), it's parsed into
    `DocumentChunk.metadata` and stripped out of `content` — it's
    structured metadata, not retrievable prose. This is fully optional
    and backward-compatible: sections without a metadata line (all the
    pre-existing docs) just get `metadata={}`, exactly as before.
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

            metadata: dict[str, str] = {}
            first_line, _, rest = body.partition("\n")
            parsed = _parse_metadata_line(first_line)
            if parsed is not None:
                metadata = parsed
                body = rest.strip()

            if body:
                chunks.append(DocumentChunk(source=path.name, heading=heading, content=body, metadata=metadata))
    return chunks


class KnowledgeBaseRetriever:
    """
    Fits a TF-IDF matrix over the given chunks at construction time;
    `retrieve()` scores a query against every chunk via cosine
    similarity and returns the top-k.
    """

    def __init__(self, chunks: list[DocumentChunk]):
        if not chunks:
            raise ValueError("KnowledgeBaseRetriever requires at least one document chunk.")
        self._chunks = chunks
        self._doc_term_counts = [Counter(_tokenize(f"{c.heading} {c.content}")) for c in chunks]
        self._vocab = sorted({term for counts in self._doc_term_counts for term in counts})
        self._term_index = {term: i for i, term in enumerate(self._vocab)}
        self._idf = self._compute_idf()
        self._doc_vectors = [self._vectorize_counts(counts) for counts in self._doc_term_counts]

    def _compute_idf(self) -> dict[str, float]:
        n_docs = len(self._doc_term_counts)
        doc_freq = Counter()
        for counts in self._doc_term_counts:
            doc_freq.update(counts.keys())
        # Smoothed IDF (add-1 in numerator and denominator) — standard
        # to avoid division by zero and to avoid a zero IDF for terms
        # appearing in every document.
        return {term: math.log((1 + n_docs) / (1 + df)) + 1.0 for term, df in doc_freq.items()}

    def _vectorize_counts(self, counts: Counter) -> dict[int, float]:
        """Sparse TF-IDF vector, stored as {vocab_index: weight} — dense numpy would waste memory on a tiny corpus."""
        total = sum(counts.values())
        if total == 0:
            return {}
        vec = {}
        for term, count in counts.items():
            idx = self._term_index.get(term)
            if idx is None:
                continue
            tf = count / total
            vec[idx] = tf * self._idf[term]
        return vec

    @staticmethod
    def _cosine_similarity(a: dict[int, float], b: dict[int, float]) -> float:
        if not a or not b:
            return 0.0
        shared = set(a) & set(b)
        dot = sum(a[i] * b[i] for i in shared)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.12,
        category: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Returns up to `top_k` chunks with cosine similarity >= `min_score`,
        ranked highest-first. `min_score` exists so an unrelated query
        (no term overlap with the knowledge base at all) returns an
        empty list rather than the top-k least-bad noise matches — the
        caller (knowledge_base_node) should treat an empty result as
        "nothing relevant found," not as an error.

        `category`, if given, restricts candidates to chunks whose
        `metadata["category"]` matches (e.g. `"decision_policy"`) BEFORE
        scoring/ranking — this is metadata filtering (task spec section
        4), a cheap precision improvement for callers that already know
        they want decision policy vs. methodology content, distinct
        from `min_score`'s relevance gate. Chunks with no metadata (or
        no matching category) are simply excluded from that call, never
        scored against a category they don't have.
        """
        query_vec = self._vectorize_counts(Counter(_tokenize(query)))
        if not query_vec:
            return []

        pairs = zip(self._chunks, self._doc_vectors)
        if category is not None:
            pairs = ((c, v) for c, v in pairs if c.metadata.get("category") == category)

        scored = [
            RetrievedChunk(chunk=chunk, score=self._cosine_similarity(query_vec, doc_vec))
            for chunk, doc_vec in pairs
        ]
        scored = [r for r in scored if r.score >= min_score]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


@lru_cache(maxsize=1)
def get_retriever() -> KnowledgeBaseRetriever:
    """Loads and fits the retriever once per process — the knowledge base is static, no need to rebuild per request."""
    chunks = load_documents()
    return KnowledgeBaseRetriever(chunks)
