"""
Graph builder — Stage 5/7/9/Funnel architecture.

                              +--> Funnel --+
                              |             v
                          (funnel)      (validation?)
                              |             |
    START -> Classifier -> Planner -> Validation --+--> Experiment --> Decision --> END
                              |             ^                              ^
                              +-- Knowledge Base ---------------------------+
                                  (Stage 9, Agentic RAG)

Three independent conditional decisions make this agentic rather than
a fixed pipeline:

  1. `route_after_planner` — decides which of three entry paths a
     request takes:
       a. Planner's ONLY capability is the knowledge base (a
          conceptual question, e.g. "What is CUPED?") -> straight to
          Knowledge Base, skipping everything else (Agentic RAG).
       b. Planner selected "funnel" (a drop-off/conversion-funnel
          question, e.g. "why did conversion decrease?"), whether
          alone or combined with validation/experiment -> Funnel runs
          first.
       c. Otherwise -> Validation, as before.

  2. `route_after_funnel` — after Funnel, if Planner ALSO selected
     "validation" (the combined "why did conversion decrease, and did
     variant B fix it?" case), continue into Validation/Experiment as
     normal; otherwise skip straight to Decision with just the funnel
     numbers.

  3. `route_after_validation` — Validation can stop the pipeline
     before Experiment runs, in two independent cases:
       a. SRM FAILED — running a hypothesis test on data with broken
          randomization would produce a misleading number. The graph
          skips straight to Decision, which (via TemplateReportGenerator)
          already returns LOW confidence and explains why, without any
          fabricated stats.
       b. PLANNER DECIDED EXPERIMENT ISN'T NEEDED — e.g. "check data
          quality only" doesn't need a hypothesis test at all; Planner
          already excluded "experiment" from `run_capability_nodes`.

All three decision functions are logged so the reason a request took
a given path is visible in the execution trace, not just inferred
from the graph shape.
"""

from langgraph.graph import END, START, StateGraph

from app.core.pipeline_events import instrument_node
from app.graph.nodes import (
    classifier_node,
    decision_node,
    experiment_node,
    funnel_node,
    guardrail_node,
    knowledge_base_node,
    planner_node,
    validation_node,
)
from app.graph.state import GraphState


def route_after_planner(state: GraphState) -> str | list[str]:
    """
    Decides which of the three entry paths a request takes. "funnel"
    is checked BEFORE the knowledge-base-only check since a funnel
    request can legitimately be combined with validation/experiment
    (the combined "why did conversion decrease, and did B fix it?"
    case) — a bare conceptual knowledge_base request, by contrast, is
    always exclusive (no dataset to validate at all).

    Stage 10 — Agentic RAG, requirement #2: when "knowledge_base" is
    combined with "validation" (the normal "should we ship?" case,
    now that KeywordPlanner always floors in knowledge_base for full
    reviews — see planner_strategy.py), this FANS OUT to BOTH
    "validation" and "knowledge_base" in parallel — returning a list
    of node names is how LangGraph's conditional edges express
    concurrent branches. Both converge back at "decision" (either
    directly, from knowledge_base, or via validation/experiment's own
    routing) — never sequential, and knowledge_base never blocks or
    is blocked by the deterministic validation/experiment path (see
    knowledge_base_node.py's try/except and its delta-only return for
    why this fan-out doesn't hit a state-channel merge conflict).
    """
    from app.core.logging import get_node_logger

    log = get_node_logger("Planner")

    plan = state["plan"]
    capabilities = plan["run_capability_nodes"]

    if "funnel" in capabilities:
        log.info(
            "[Planner] Routing to Funnel Analysis (intent=%s, capabilities=%s).",
            plan["intent_label"],
            capabilities,
        )
        return "funnel"

    if capabilities == ["knowledge_base"]:
        log.info(
            "[Planner] Routing directly to Knowledge Base (intent=%s) — skipping Validation and Experiment.",
            plan["intent_label"],
        )
        return "knowledge_base"

    if "knowledge_base" in capabilities:
        log.info(
            "[Planner] Fanning out to Validation + Knowledge Base in parallel (intent=%s, capabilities=%s).",
            plan["intent_label"],
            capabilities,
        )
        return ["validation", "knowledge_base"]

    return "validation"


def route_after_funnel(state: GraphState) -> str:
    """
    After Funnel runs, continue into Validation/Experiment only if
    Planner also selected "validation" (the combined use case) —
    otherwise the funnel numbers alone are the answer, skip straight
    to Decision.
    """
    from app.core.logging import get_node_logger

    log = get_node_logger("Funnel")

    plan = state["plan"]
    if "validation" in plan["run_capability_nodes"]:
        log.info("[Funnel] Combined with Validation/Experiment — continuing the pipeline.")
        return "validation"

    log.info("[Funnel] Funnel-only request — routing directly to Decision.")
    return "decision"


def route_after_validation(state: GraphState) -> str:
    """
    The actual routing decision — this function is what LangGraph calls
    after `validation` to decide the next node. All stop conditions are
    logged so the reason is visible in the execution trace, not just
    inferred from the graph shape.
    """
    from app.core.logging import get_node_logger

    log = get_node_logger("Validation")

    srm_result = state["srm_result"]
    if not srm_result.passed:
        log.info("[Validation] SRM FAILED — routing directly to Decision, skipping Experiment.")
        return "decision"

    if state.get("has_conflicting_variant_duplicates"):
        log.info(
            "[Validation] Users found assigned to MULTIPLE variants (broken assignment pipeline) — "
            "routing directly to Decision, skipping Experiment."
        )
        return "decision"

    critical_quality_failures = [
        qc for qc in state.get("quality_checks", []) if not qc.passed and qc.critical
    ]
    if critical_quality_failures:
        labels = ", ".join(qc.label for qc in critical_quality_failures)
        log.info(
            "[Validation] Critical quality failure(s): %s — routing directly to Decision, skipping Experiment.",
            labels,
        )
        return "decision"

    plan = state["plan"]
    intent_label = plan["intent_label"]

    # Full Experiment Review and Statistical Analysis are dataset-specific
    # experiment requests.  They must reach the deterministic experiment
    # node whenever the quality gates above have passed, even if a stale or
    # malformed planner payload omitted the ``experiment`` capability.
    # The planner contract also enforces this, but the graph itself is the
    # final safety boundary: an LLM/planner routing mistake must not silently
    # turn an end-to-end experiment review into a validation-only run.
    experiment_required_intents = {
        "Full Experiment Review",
        "Statistical Analysis",
        "Explanation",
        "Stratified Analysis",
    }
    if intent_label in experiment_required_intents:
        if "experiment" not in plan["run_capability_nodes"]:
            log.warning(
                "[Validation] Planner omitted 'experiment' for required intent=%s; "
                "forcing Experiment routing after quality gates passed.",
                intent_label,
            )
        return "experiment"

    if "experiment" not in plan["run_capability_nodes"]:
        log.info(
            "[Validation] Planner excluded 'experiment' from this run (intent=%s) — skipping to Decision.",
            intent_label,
        )
        return "decision"

    return "experiment"


def build_graph():
    builder = StateGraph(GraphState)

    # Every node is wrapped with `instrument_node(stage)` — see
    # app/core/pipeline_events.py. This is pure side-channel timing +
    # optional streaming-event emission: the wrapper calls the exact
    # same node function with the exact same state, so the graph's
    # execution semantics (routing, conditional edges, defer=True
    # fan-in) are unchanged. It's applied here, at registration, so no
    # individual node module has to import or know about
    # instrumentation at all.
    builder.add_node("classifier", instrument_node("classifier")(classifier_node))
    builder.add_node("planner", instrument_node("planner")(planner_node))
    builder.add_node("validation", instrument_node("validation")(validation_node))
    builder.add_node("experiment", instrument_node("experiment")(experiment_node))
    builder.add_node("funnel", instrument_node("funnel")(funnel_node))
    builder.add_node("guardrail", instrument_node("guardrail")(guardrail_node))
    builder.add_node("knowledge_base", instrument_node("knowledge_base")(knowledge_base_node))
    # defer=True — REQUIRED for the Stage 10 fan-out (knowledge_base
    # running in parallel with validation/experiment, see
    # `route_after_planner`'s docstring). Without this, "decision" has
    # incoming edges of different path lengths (knowledge_base -> decision
    # is 1 hop; validation -> experiment -> decision is 2 hops), so
    # LangGraph would fire "decision" as soon as the SHORTER branch
    # (knowledge_base) completes — running it concurrently with, and
    # before, "experiment" finishes on the other branch. That produced
    # two symptoms: a premature report missing the stats that hadn't
    # been computed yet, AND an `InvalidUpdateError` from "decision" and
    # "experiment" both writing the shared state channels in the same
    # superstep. `defer=True` makes "decision" wait until every other
    # node scheduled to run in this invocation has finished before it
    # executes, regardless of how many hops each branch took — the
    # correct fan-in synchronization for branches of unequal length.
    builder.add_node("decision", instrument_node("decision")(decision_node), defer=True)

    builder.add_edge(START, "classifier")
    builder.add_edge("classifier", "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"funnel": "funnel", "validation": "validation", "knowledge_base": "knowledge_base"},
    )
    builder.add_conditional_edges(
        "funnel",
        route_after_funnel,
        {"validation": "validation", "decision": "decision"},
    )
    builder.add_conditional_edges(
        "validation",
        route_after_validation,
        {"experiment": "experiment", "decision": "decision"},
    )
    # Guardrail analysis (the previously-missing deterministic producer
    # of `guardrail_results` — see guardrail_node.py) runs right after
    # Experiment, using the same resolved experiment_columns/dataframe.
    # It ALWAYS runs on this path (even when settings.guardrail_metrics
    # is empty, in which case it's a cheap no-op that stamps
    # NOT_SPECIFIED) so `decision` never needs to guess whether
    # guardrail resolution happened.
    builder.add_edge("experiment", "guardrail")
    builder.add_edge("guardrail", "decision")
    builder.add_edge("knowledge_base", "decision")
    builder.add_edge("decision", END)

    return builder.compile()


experiment_review_graph = build_graph()


def export_mermaid() -> str:
    """
    Generates the graph's Mermaid diagram straight from the compiled
    LangGraph object — not hand-drawn, so it can never drift from the
    actual graph topology. Used by scripts/export_graph.py to refresh
    docs/graph.mmd and the diagram embedded in README.md.
    """
    return experiment_review_graph.get_graph().draw_mermaid()
