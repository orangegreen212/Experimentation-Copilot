"""
LangGraph state definition.

ARCHITECTURAL PRINCIPLE: this state is a growing FACTS object, not a
conversation. Each node reads what it needs and writes ONE new field.
Nothing here is optional/ambiguous about who's responsible for what:

  classifier_node    -> writes `dataset`
  planner_node        -> reads `dataset` + `user_prompt`, writes `plan`
  quality_node         -> writes `quality_checks`, `srm_result`
  test_selection_node  -> writes `test_selection`, `stat_results`
  power_node            -> writes `power_analysis` (folded into test_selection stage,
                            see Stage 4 planning notes — may end up being its own
                            node or a function called inside test_selection_node)
  variance_reduction_node (conditional, only if cuped/bootstrap enabled)
                        -> writes `variance_reduction`
  decision_node          -> reads EVERYTHING above, writes `report`
                            (LLM call — but only synthesizes text fields,
                            never touches the numeric ones)

No node ever mutates a field another node owns. This is what makes the
graph auditable: you can log the state after every node and see exactly
which step produced which fact.

Using a TypedDict (not a Pydantic model) because LangGraph's StateGraph
expects a mapping type with reducer-annotated fields; the individual
values stored inside ARE the Pydantic schemas above, so we still get
full validation at the field level — just not at the top-level dict.
"""

from typing import TypedDict

from app.schemas.chat import ChatMessage
from app.schemas.dataset import DatasetInfo
from app.schemas.execution import ExecutionStep
from app.schemas.guardrails import GuardrailRequestState, GuardrailResolution
from app.schemas.hypothesis import Hypothesis
from app.schemas.quality import QualityCheck, SRMResult
from app.schemas.report import ExperimentReport
from app.schemas.settings import AnalysisSettings
from app.schemas.statistics import (
    PowerAnalysisResult,
    StatResult,
    TestSelectionResult,
    VarianceReductionResult,
)
from app.stats.dataset_classifier import ExperimentColumns
from app.stats.funnel import FunnelAnalysisResult
from app.rag.retriever import RetrievedChunk
from app.schemas.segmentation import SegmentationResult
from app.schemas.stratification import StratificationResult


class PlannerOutput(TypedDict, total=False):
    """
    Minimal router output. The planner does NOT do free-form
    branching — it produces a small, fixed decision:

      - intent_label: a human-readable label for the ExecutionStep UI
        (e.g. "Full Experiment Review"), chosen from a small closed set
      - run_capability_nodes: which of the fixed capability nodes to
        run (currently always all of them — this field exists so the
        seam for real branching later doesn't require a state shape
        change, only a planner_node logic change)

    This keeps the LLM's job to genuine "intent classification into a
    known set of labels," not open-ended planning — consistent with
    "LLM never does math, and for now, LLM does not really branch
    the graph either."
    """

    intent_label: str
    run_capability_nodes: list[str]

    # Which LLM model the Planner actually tried to use, and whether it
    # succeeded. "not_used" for
    # KeywordPlanner (no LLM call is made at all — the deterministic,
    # default path). "success"/"fallback" only ever come from
    # LLMPlanner, so a caller can tell "the selected model actually
    # ran" apart from "the LLM call failed and we silently fell back
    # to KeywordPlanner" instead of just seeing a plausible-looking
    # intent label either way. See planner_strategy.py.
    llm_status: str  # "not_used" | "success" | "fallback"
    llm_requested_model: str | None
    llm_error: str | None


class GraphState(TypedDict, total=False):
    """
    Full pipeline state. `total=False` because fields are populated
    progressively as nodes execute — not all fields exist at every
    point in the graph's execution.
    """

    # ---- input (set before graph.invoke) ----
    dataset_id: str
    # Optional separate experiment-assignment dataset (e.g. `user_id |
    # variant`), uploaded through the SAME /datasets/classify mechanism
    # as the primary dataset — see routes_datasets.py. None (the
    # default/absent case) for every existing single-file flow;
    # classifier_node loads it via get_dataset() and merges it onto
    # the primary dataset via enrich_with_assignment (see
    # app/stats/dataset_classifier.py), then persists the merged frame
    # under a NEW dataset_id and repoints this same field at it, so
    # every downstream node's own get_dataset(state["dataset_id"])
    # call sees the merged data through the same mechanism as always.
    assignment_dataset_id: str | None
    user_prompt: str
    settings: AnalysisSettings
    # Phase 1 — Experiment Hypothesis (structured, optional). Set once
    # before graph.invoke(), from AnalyzeExperimentRequest.hypothesis.
    # NEVER inserted as free text into any prompt: it flows through
    # untouched to ReportFacts.hypothesis (see decision_node.py /
    # report_generator.py) as input CONTEXT, not a statistic and not
    # something any node — LLM or deterministic — is allowed to
    # modify. Optional so existing datasets/flows without a hypothesis
    # are completely unaffected (see Hypothesis's own docstring for
    # the full scope of what Phase 1 does and does NOT implement).
    hypothesis: Hypothesis | None
    # NOTE: deliberately NO raw DataFrame field here — see
    # app/core/dataset_store.py's module docstring. Any node needing
    # the actual data calls get_dataset(state["dataset_id"]) fresh;
    # putting a DataFrame in GraphState would make LangGraph re-trace
    # (and re-upload to LangSmith) the full dataset on every node.

    # ---- classifier_node output ----
    dataset: DatasetInfo
    experiment_columns: ExperimentColumns | None  # internal only — None if the dataset has no metric column (e.g. a funnel/event-log dataset)
    experiment_columns_error: str | None  # the specific reason experiment_columns is None (e.g. "no experiment-unit identifier" vs "no metric column") — surfaced verbatim by validation_node instead of a generic message

    # ---- planner_node output ----
    plan: PlannerOutput

    # ---- quality_node output ----
    quality_checks: list[QualityCheck]
    srm_result: SRMResult
    has_conflicting_variant_duplicates: bool
    conflicting_variant_user_count: int

    # ---- test_selection_node output ----
    test_selection: TestSelectionResult
    stat_results: list[StatResult]
    power_analysis: PowerAnalysisResult

    # ---- variance_reduction_node output (conditional) ----
    variance_reduction: VarianceReductionResult | None
    bootstrap_ci_check: tuple[float, float] | None
    bootstrap_iterations: int | None

    # ---- guardrail_node output ----
    # The user's own explicit guardrail request, exactly as sent in
    # `settings.guardrail_metrics` (kept here too, alongside `settings`,
    # so downstream nodes read one obvious key instead of reaching back
    # into `state["settings"]`). Empty list, never absent, once
    # guardrail_node has run.
    requested_guardrails: list[str]
    # Per-requested-guardrail resolution outcome (found/not found +
    # which dataset column it matched) — the "availability" dimension.
    # See app.schemas.guardrails.GuardrailResolution.
    guardrail_resolutions: list[GuardrailResolution]
    # Aggregate of the list above — NOT_SPECIFIED / REQUESTED_NOT_FOUND
    # / PARTIALLY_AVAILABLE / AVAILABLE. See
    # app.schemas.guardrails.GuardrailRequestState.
    guardrail_request_state: GuardrailRequestState
    # The actual StatResult rows for every guardrail that resolved AND
    # was evaluated (two-arm experiments only — see guardrail_node.py).
    # THIS is the field that was always `state.get("guardrail_results", [])`
    # with nothing ever writing it before guardrail_node existed.
    guardrail_results: list[StatResult]

    # ---- knowledge_base_node output (Stage 9 — only runs when Planner
    # routes to "knowledge_base"; a normal experiment-review run never
    # populates this) ----
    kb_results: list[RetrievedChunk]
    # Phase 8 — distinguishes "retrieval ran and legitimately found
    # nothing relevant" (kb_results == [], kb_error is None) from "the
    # retriever itself blew up" (kb_results == [], kb_error is the
    # short, non-sensitive reason). Never set at all when
    # knowledge_base_node never ran. See knowledge_base_node.py.
    kb_error: str | None

    # ---- funnel_node output (only runs when Planner routes to
    # "funnel") ----
    funnel_result: FunnelAnalysisResult | None
    funnel_by_group: dict[str, FunnelAnalysisResult] | None  # populated only when combined with "experiment"
    funnel_skip_reason: str | None

    # ---- experiment_node output (Phase 5 — segmentation, additive) ----
    # Only populated on the two-arm path (multi_arm segmentation is out
    # of scope for Phase 5 — see experiment_node.py). Always a
    # SegmentationResult (never absent) once experiment_node has run;
    # `ran=False` covers the "no usable dimension" case explicitly
    # rather than this key being missing.
    segmentation_result: SegmentationResult | None

    # ---- experiment_node output (TRUE stratified analysis, additive) ----
    # Only populated when settings.analysis_mode == "stratified" AND a
    # stratification_column was provided AND this is a two-arm
    # experiment (mirrors segmentation_result's own two-arm-only scope).
    # Distinct from `segmentation_result` — see
    # app/schemas/stratification.py's module docstring for why
    # stratification and segmentation are never conflated.
    stratification_result: StratificationResult | None

    # ---- internal display strings, produced alongside power_analysis ----
    _mde_display: str
    _sample_size_note: str

    # ---- bookkeeping for the (currently synchronous) ExecutionStepper ----
    execution_steps: list[ExecutionStep]

    # ---- decision_node output (final) ----
    report: ExperimentReport

    # ---- follow-up chat (used by the /chat endpoint, not /analyze) ----
    chat_history: list[ChatMessage]
