"""
Knowledge base node — Stage 9 (Agentic RAG). Thin adapter over
app/rag/retriever.py, same pattern as every other capability node: no
retrieval logic lives here, only wiring.

This node only runs when Planner's `run_capability_nodes` routes here
(see graph_builder.py's `route_after_planner`) — a request with no
methodology signal at all never touches this node or the knowledge
base. That conditional routing is what makes this Agentic RAG rather
than "always retrieve and stuff it into every prompt."

STAGE 10 — PARALLEL-SAFE RETURN: as of Stage 10, this node can run
CONCURRENTLY with `validation` (see graph_builder.py's fan-out in
`route_after_planner`). Every other node in this graph returns
`{**state, ...}` (the full state dict), which is safe as long as it's
the only writer for every key it touches in that superstep. Under
fan-out that stops being true: `validation` is, in the same superstep,
also writing `dataset_id`/`user_prompt`/`plan`/etc. back onto the
`GraphState` TypedDict's plain (unannotated, default `LastValue`)
channels. LangGraph raises `InvalidUpdateError` when two nodes in the
same superstep both write to the same channel — so this node
deliberately returns ONLY its own delta (`{"kb_results": ...}`), never
a full-state spread, so it never contends for a channel `validation`
is also writing. If another node needs to run in parallel with this
one in the future, it must follow the same delta-only pattern rather
than copying this node's old `{**state, ...}` style.

GRACEFUL DEGRADATION (requirement #6): retrieval is wrapped in a
try/except — if the retriever blows up for any reason (index not
built, I/O error, etc.), this node logs the failure and returns
`kb_results=[]` rather than raising, so a retrieval outage NEVER
breaks the deterministic validation/experiment analysis running
alongside it.

QUERY CONSTRUCTION (Phase 4 fix — vocabulary-mismatch diagnosis): the
retriever is plain TF-IDF (see app/rag/retriever.py) — it only matches
on shared vocabulary between the query and the knowledge base's
methodology docs. Passing `state["user_prompt"]` verbatim works fine
for an actual methodology QUESTION ("What is CUPED?" shares vocabulary
with the CUPED doc by construction), but a normal Full Experiment
Review prompt ("Analyze this experiment", "Should we ship variant
B?") is about the DATASET, not about methodology terms, so it often
has near-zero term overlap with the knowledge base and legitimately
scores below `min_score` — this is the exact zero-references case
this fix addresses.

`_build_retrieval_query()` below only changes what string is handed to
the SAME retriever with the SAME threshold: for the dataset-driven
path (knowledge_base combined with validation/experiment) it appends a
fixed, deterministic set of core experimentation-methodology concepts
(SRM, guardrails, significance, power/MDE, segmentation, multiple
comparisons — the same universally-relevant topics a "Full Experiment
Review" always touches) plus a metric-type-specific term (binary vs.
continuous) drawn from `experiment_columns`, which is already resolved
by `classifier_node` before this node ever runs. This never adds a
source name or citation — it only adds search terms; the retriever
still decides, by real cosine similarity against the real knowledge
base, what (if anything) comes back, and an irrelevant/empty result
remains possible.

For a pure conceptual question (`run_capability_nodes == ["knowledge_base"]`,
no dataset in play), the query is left exactly as `user_prompt` —
unchanged from before this fix — since the user's own wording IS the
actual content query there and diluting it with generic terms would
risk drowning out or changing what's actually being asked (see
`TestMethodologyRag`'s conceptual-question tests).
QUALITY GATE (this task): `retriever.retrieve()`'s own `min_score`
default (0.12, see app/rag/retriever.py) is that module's generic,
unit-test-covered default for direct/standalone callers. The app's
actual quality bar for "relevant enough to show as evidence" is
`stats_thresholds.kb_relevance_threshold` (app/core/config.py) —
passed explicitly below so there is exactly ONE configurable place
that answers that question, not two. See that field's docstring for
why 0.20 was chosen and how it was measured against this KB's real
score distribution. Raising/lowering the bar is a one-line config
change; the retriever itself is untouched.
"""

from app.core.config import stats_thresholds
from app.core.logging import get_node_logger
from app.graph.state import GraphState
from app.rag.retriever import RetrievedChunk, get_retriever
from app.schemas.statistics import MetricType

log = get_node_logger("KnowledgeBase")

# SOURCE-TIER PRIORITIZATION (this task) — deterministic evidence
# SELECTION only, applied strictly AFTER `retriever.retrieve()`'s own
# `min_score` gate (stats_thresholds.kb_relevance_threshold) has
# already filtered candidates. This never touches `.score` (the real
# TF-IDF cosine similarity) and never promotes a source that failed
# the relevance gate — it only decides which small subset of the
# already-relevant candidates gets shown, preferring Tier 1
# (Kohavi/Microsoft/Netflix) over Tier 2 (Booking/Airbnb/
# Optimizely-Google) when both are relevant, per filename — the same
# `source` metadata the report/UI already display, so there is no
# separate attribution table to keep in sync or get wrong.
#
# SOURCE DIVERSITY FIX: selection also enforces at most one chunk per
# unique `source` filename (see `_prioritize_by_source_tier`), so three
# above-threshold chunks from the same doc can no longer fill every
# evidence slot and crowd out other equally-preferred sources.
_TIER_1_SOURCES = frozenset({
    "kohavi.md", "microsoft.md", "netflix.md",
    # RAG knowledge-base expansion (this task): first-party, in-house
    # methodology and decision-policy docs — authoritative for this
    # app's own decision engine, so treated as Tier 1 alongside the
    # original source-attributed methodology notes.
    "decision_policies.md", "statistical_interpretation.md",
    "experiment_fundamentals.md", "advanced_methodology.md",
})
_TIER_2_SOURCES = frozenset({"booking.md", "airbnb.md", "optimizely_google.md"})

# Target number of references actually shown in Evidence & Sources.
_MAX_EVIDENCE_REFERENCES = 3

# Candidate pool size requested from the retriever — larger than
# _MAX_EVIDENCE_REFERENCES so tier-prioritization has real
# already-relevant candidates to choose between (e.g. a Tier 2 result
# that outscored a Tier 1 one isn't lost before prioritization even
# sees it). Every candidate in this pool has still cleared the
# UNCHANGED `min_score` gate below — only the pool size changed, not
# the threshold or the retriever itself.
_CANDIDATE_POOL_SIZE = 8

# Phase 4 fix — core experimentation-methodology concepts that are
# always relevant context for a dataset-driven experiment review,
# regardless of what this particular dataset's numbers turn out to be
# (mirrors the fixed set of topics KeywordPlanner already always
# floors in "knowledge_base" for full reviews — see
# planner_strategy.py). Concepts only, never a source/author name.
_CORE_REVIEW_CONCEPTS = (
    "A/B testing experiment validity randomization sample ratio mismatch "
    "statistical significance practical significance minimum detectable effect "
    "power analysis guardrail metrics multiple comparisons segmentation "
    "heterogeneous treatment effects interpreting experiment results"
)

_METRIC_TYPE_TERMS = {
    MetricType.BINARY: "binary conversion metric chi-square test",
    MetricType.CONTINUOUS_MONETARY: "continuous revenue metric t-test confidence interval",
    MetricType.CONTINUOUS_GENERAL: "continuous metric t-test confidence interval",
}


def _build_retrieval_query(state: GraphState) -> str:
    """
    Returns the string handed to `retriever.retrieve()`. See module
    docstring's "QUERY CONSTRUCTION" section for the reasoning.
    """
    user_prompt = state["user_prompt"]
    capabilities = state.get("plan", {}).get("run_capability_nodes", [])

    # Pure conceptual question — knowledge_base is the ONLY capability,
    # there's no dataset-driven review in play. Leave the query exactly
    # as the user's own wording (unchanged, pre-existing behavior).
    if capabilities == ["knowledge_base"]:
        return user_prompt

    terms = [user_prompt, _CORE_REVIEW_CONCEPTS]
    experiment_columns = state.get("experiment_columns")
    if experiment_columns is not None:
        metric_terms = _METRIC_TYPE_TERMS.get(experiment_columns.metric_type)
        if metric_terms:
            terms.append(metric_terms)
    return " ".join(terms)


def _source_tier(chunk_source: str) -> int:
    """0 = Tier 1 (preferred/primary), 1 = Tier 2 (supporting), 2 = unranked (unknown filename — still shown, just not prioritized)."""
    if chunk_source in _TIER_1_SOURCES:
        return 0
    if chunk_source in _TIER_2_SOURCES:
        return 1
    return 2


def _prioritize_by_source_tier(
    results: list[RetrievedChunk], limit: int = _MAX_EVIDENCE_REFERENCES
) -> list[RetrievedChunk]:
    """
    Deterministic evidence SELECTION, applied only to candidates that
    already cleared `stats_thresholds.kb_relevance_threshold` (every
    item in `results`). Sorts by (tier, -score): Tier 1 before Tier 2
    before unranked, and by the ORIGINAL, unmodified cosine score
    within each tier — never invents, boosts, or fakes a score, only
    reorders/truncates an already-relevant list.

    UNIQUE SOURCE CONSTRAINT (this fix): the same source filename
    (e.g. "kohavi.md") can contribute at most one chunk to the
    returned evidence. Without this, three separate above-threshold
    chunks from the same Tier 1 doc could fill every slot and crowd
    out other equally-preferred sources (e.g. Microsoft, Netflix),
    defeating the point of tier prioritization — a diverse reference
    list, not just "whichever doc happened to have the most relevant
    sections." Within a source, only its single best-scoring candidate
    is even eligible to be picked; that candidate still competes for a
    slot on the same (tier, -score) ordering as before.

    A Tier 1 source that never passed the relevance gate never reaches
    this function at all (see `knowledge_base_node` below), so this
    can't "promote" an irrelevant Tier 1 result. A strong Tier 2 result
    is never dropped for being Tier 2 alone — it's only excluded if the
    pool has more relevant, unique-source candidates than `limit`, the
    same slot competition that applied before this change existed.

    If fewer than `limit` unique sources are eligible, this returns
    fewer than `limit` references rather than reusing a source for a
    second chunk — `knowledge_base_references` is a plain list with no
    fixed-length contract (see app/schemas/report.py), so there is no
    product requirement forcing exactly `limit` items, and manufacturing
    a duplicate-source reference just to hit a count would be worse
    than an honestly shorter, non-redundant list.
    """
    # Keep only each source's single best-scoring candidate — the
    # original score is untouched, this just picks which one chunk
    # from that source is even eligible to compete for a slot.
    best_per_source: dict[str, RetrievedChunk] = {}
    for r in results:
        current = best_per_source.get(r.chunk.source)
        if current is None or r.score > current.score:
            best_per_source[r.chunk.source] = r

    deduped = list(best_per_source.values())
    return sorted(deduped, key=lambda r: (_source_tier(r.chunk.source), -r.score))[:limit]


def knowledge_base_node(state: GraphState) -> GraphState:
    kb_error: str | None = None
    try:
        retriever = get_retriever()
        query = _build_retrieval_query(state)
        # Retrieve a larger already-relevant candidate pool (pool size
        # only — `min_score` is the SAME unchanged
        # stats_thresholds.kb_relevance_threshold), then apply
        # deterministic tier-based selection to pick the ~3 actually
        # shown. See _prioritize_by_source_tier's docstring.
        candidates = retriever.retrieve(
            query, top_k=_CANDIDATE_POOL_SIZE, min_score=stats_thresholds.kb_relevance_threshold
        )
        results = _prioritize_by_source_tier(candidates)
    except Exception as exc:  # noqa: BLE001 — a retrieval outage must never break the graph
        log.warning("[KnowledgeBase] Retrieval failed (%s) — degrading gracefully with no results.", exc)
        results = []
        # PHASE 8 — this is the one case that is an actual FAILURE, not
        # an empty (but legitimate) retrieval result. The short
        # exception type/message is safe to surface (no secrets live
        # in this code path — the retriever only touches static local
        # markdown files), but the trace never gets a raw stack trace.
        kb_error = f"{type(exc).__name__}: {exc}"

    if results:
        log.info(
            "[KnowledgeBase] Retrieved %d chunk(s) — top match: [%s] %s (score=%.2f)",
            len(results),
            results[0].chunk.source,
            results[0].chunk.heading,
            results[0].score,
        )
    elif kb_error is None:
        log.info("[KnowledgeBase] No relevant chunks found for this query.")

    # Delta-only — see module docstring's "PARALLEL-SAFE RETURN" note.
    return {"kb_results": results, "kb_error": kb_error}
