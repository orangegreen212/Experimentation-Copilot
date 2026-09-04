"""
Guardrail analysis node.

THE MISSING PIECE identified by the guardrail root-cause audit: this
is the first and only place in the graph that writes
`state["guardrail_results"]`. Before this node existed, every read of
that key (`state.get("guardrail_results", [])` in decision_node.py)
saw the default `[]` on every single run, because nothing ever wrote
it — see the audit for the full trace.

What this node does, in order:

  1. Reads `state["settings"].guardrail_metrics` — the user's own
     EXPLICIT, structured guardrail request (never parsed from free
     text — see AnalysisSettings.guardrail_metrics).
  2. Resolves each requested name against this dataset's actual
     available metric columns, deterministically and exactly (never
     fuzzy — see app.stats.dataset_classifier.resolve_guardrail_metrics).
     This produces `state["guardrail_resolutions"]` and
     `state["guardrail_request_state"]` — the availability dimension —
     regardless of whether any statistical test can actually run.
  3. For every metric that DID resolve, on a two-arm experiment, runs
     the exact same deterministic hypothesis-test infrastructure
     (app.stats.hypothesis_tests) already used for the primary metric
     in experiment_node.py — no bespoke guardrail math, no LLM.

Scope, matching the rest of the pipeline's existing two-arm-only
carve-outs (segmentation, true stratification): guardrail STATISTICAL
EVALUATION only runs for two-arm experiments. A multi-arm dataset
still gets full, correct resolution (`guardrail_request_state`/
`guardrail_resolutions`), it just won't have `guardrail_results` —
this is a real, honest limitation, not a bug, and is logged.

Runs after `experiment_node` (needs `experiment_columns` — the
resolved user/variant/metric column roles — and the deduplicated
dataframe) and before `decision_node`. Never touches: primary-metric
`stat_results`, power/MDE, SRM, segmentation, stratification, CUPED,
or any existing decision formula.
"""

from app.core.dataset_store import get_dataset
from app.core.logging import get_node_logger
from app.graph.state import GraphState
from app.schemas.guardrails import GuardrailRequestState, GuardrailResolution
from app.stats.dataset_classifier import (
    build_metric_column_map,
    deduplicate_by_user,
    humanize_metric_label,
    infer_guardrail_direction,
    infer_metric_type,
    resolve_control_label,
    resolve_guardrail_metrics,
)
from app.stats.hypothesis_tests import compute_stat_result, select_test

log = get_node_logger("Guardrails")


def derive_guardrail_request_state(resolutions: list[GuardrailResolution]) -> GuardrailRequestState:
    """Pure function of the resolution list — no I/O, easy to unit test and to reuse as a fallback (see decision_node.py)."""
    if not resolutions:
        return GuardrailRequestState.NOT_SPECIFIED
    found = [r for r in resolutions if r.resolved]
    if not found:
        return GuardrailRequestState.REQUESTED_NOT_FOUND
    if len(found) < len(resolutions):
        return GuardrailRequestState.PARTIALLY_AVAILABLE
    return GuardrailRequestState.AVAILABLE


def guardrail_node(state: GraphState) -> GraphState:
    settings = state["settings"]
    requested = list(settings.guardrail_metrics or [])

    empty_result: GraphState = {
        **state,
        "requested_guardrails": [],
        "guardrail_resolutions": [],
        "guardrail_request_state": GuardrailRequestState.NOT_SPECIFIED,
        "guardrail_results": [],
    }

    if not requested:
        return empty_result

    dataset = state.get("dataset")
    columns = state.get("experiment_columns")

    if dataset is None or columns is None:
        # No usable experiment column structure at all (validation_node's
        # own gates, or a dataset with no recognizable metric column, will
        # already have surfaced why) — there's nothing to resolve requested
        # guardrails against, but the request itself must still be
        # preserved and reported honestly, never silently dropped into
        # NOT_SPECIFIED.
        resolutions = [
            GuardrailResolution(requested_name=(n or "").strip(), resolved=False, resolved_metric_label=None)
            for n in requested
            if (n or "").strip()
        ]
        log.info(
            "[Guardrails] %d guardrail(s) requested but no experiment column structure was resolved — reporting as unresolved.",
            len(resolutions),
        )
        return {
            **state,
            "requested_guardrails": requested,
            "guardrail_resolutions": resolutions,
            "guardrail_request_state": derive_guardrail_request_state(resolutions),
            "guardrail_results": [],
        }

    df = get_dataset(state["dataset_id"])
    df = deduplicate_by_user(df, columns.user_col)

    exclude = {columns.user_col, columns.variant_col}
    metric_column_map = build_metric_column_map(df, exclude=exclude)
    resolutions = resolve_guardrail_metrics(
        requested,
        available_metrics=list(metric_column_map.keys()),
        primary_metric_label=humanize_metric_label(columns.metric_col),
    )
    request_state = derive_guardrail_request_state(resolutions)

    guardrail_results = []
    variant_values = df[columns.variant_col].dropna().unique().tolist()
    if len(variant_values) == 2:
        control_label = resolve_control_label(df, columns.variant_col)
        control_mask = df[columns.variant_col] == control_label
        variant_mask = ~control_mask

        for res in resolutions:
            if not res.resolved:
                continue
            guardrail_col = metric_column_map[res.resolved_metric_label]
            guardrail_type = infer_metric_type(df, guardrail_col)
            control_series = df.loc[control_mask, guardrail_col]
            variant_series = df.loc[variant_mask, guardrail_col]
            test_selection = select_test(control_series, variant_series, guardrail_type)
            stat_result = compute_stat_result(
                control_series,
                variant_series,
                guardrail_type,
                humanize_metric_label(guardrail_col),
                test_selection,
            )
            # Directionality (doc3 §6/§7): a guardrail's "harmful" direction
            # depends on the metric — an increase in Bounce Rate is bad, an
            # increase in Revenue is good. See determine_decision()'s
            # guardrail-evaluation branch, the only place this is read.
            higher_is_better = infer_guardrail_direction(guardrail_col)
            stat_result = stat_result.model_copy(update={"higher_is_better": higher_is_better})
            guardrail_results.append(stat_result)
            log.info(
                "[Guardrails] %s evaluated — test=%s p=%.4f significant=%s delta=%s higher_is_better=%s",
                stat_result.metric,
                stat_result.test_name,
                stat_result.p_value,
                stat_result.significant,
                stat_result.delta,
                higher_is_better,
            )
    elif any(r.resolved for r in resolutions):
        log.info(
            "[Guardrails] %d guardrail(s) resolved but NOT evaluated — this experiment has %d arms; "
            "guardrail statistical evaluation is two-arm only in this phase (same scope as segmentation).",
            sum(r.resolved for r in resolutions),
            len(variant_values),
        )

    log.info(
        "[Guardrails] requested=%d resolved=%d evaluated=%d request_state=%s",
        len(resolutions),
        sum(r.resolved for r in resolutions),
        len(guardrail_results),
        request_state.value,
    )

    return {
        **state,
        "requested_guardrails": requested,
        "guardrail_resolutions": resolutions,
        "guardrail_request_state": request_state,
        "guardrail_results": guardrail_results,
    }
