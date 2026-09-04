"""
Funnel node — thin adapter over app/stats/funnel.py and
funnel_classifier.py, same pattern as every other capability node.

Handles three cases:
  - Funnel-only intent: compute one overall funnel.
  - Funnel + experiment combined, dataset ALREADY has a metric column
    (Classifier found one): compute one funnel PER experiment arm (via
    compute_funnel_by_group) for the drop-off comparison; Validation/
    Experiment run on the existing metric column as normal.
  - Funnel + experiment combined, dataset has NO metric column (a pure
    event log, e.g. the demo funnel dataset): derive a "converted"
    metric — did each user complete the whole funnel — via
    derive_conversion_dataframe(), store it as a new dataset, and swap
    state's dataset_id/experiment_columns to point at it. This is what
    makes the "why did conversion decrease, and did variant B fix it?"
    demo scenario possible on a dataset that was never an aggregated
    A/B file to begin with.
"""

from app.core.dataset_store import get_dataset, store_dataset
from app.core.logging import get_node_logger
from app.graph.state import GraphState
from app.schemas.statistics import MetricType
from app.stats.dataset_classifier import ExperimentColumns, humanize_metric_label
from app.stats.funnel import (
    FunnelComputationError,
    compute_funnel,
    compute_funnel_by_group,
    derive_conversion_dataframe,
)
from app.stats.funnel_classifier import detect_funnel_columns, infer_step_order

log = get_node_logger("Funnel")


def funnel_node(state: GraphState) -> GraphState:
    df = get_dataset(state["dataset_id"])
    columns = detect_funnel_columns(df)

    if columns is None:
        log.warning("[Funnel] Dataset does not have funnel/event-log structure — skipping funnel analysis.")
        return {**state, "funnel_result": None, "funnel_by_group": None, "funnel_skip_reason": "not_a_funnel_dataset"}

    step_order = infer_step_order(df, columns.event_col, columns.timestamp_col)

    try:
        overall_result = compute_funnel(df, columns.user_col, columns.event_col, step_order)
    except FunnelComputationError as exc:
        log.warning("[Funnel] Could not compute funnel — %s", exc)
        return {**state, "funnel_result": None, "funnel_by_group": None, "funnel_skip_reason": str(exc)}

    plan = state.get("plan", {})
    wants_experiment_comparison = "experiment" in plan.get("run_capability_nodes", [])

    funnel_by_group = None
    derived_state_updates = {}

    if wants_experiment_comparison and columns.group_col is not None:
        try:
            funnel_by_group = compute_funnel_by_group(
                df, columns.user_col, columns.event_col, step_order, columns.group_col
            )
        except FunnelComputationError as exc:
            log.warning("[Funnel] Could not compute per-group funnel comparison — %s", exc)

        if state.get("experiment_columns") is None:
            # No metric column exists — derive "converted" (reached the
            # final funnel step) so Validation/Experiment have something
            # to run on. See module docstring.
            try:
                conversion_df = derive_conversion_dataframe(
                    df, columns.user_col, columns.event_col, columns.group_col, step_order
                )
                derived_dataset_id = store_dataset(conversion_df)
                derived_columns = ExperimentColumns(
                    user_col=columns.user_col,
                    variant_col=columns.group_col,
                    metric_col="converted",
                    metric_type=MetricType.BINARY,
                )
                derived_state_updates = {
                    "dataset_id": derived_dataset_id,
                    "experiment_columns": derived_columns,
                }
                log.info(
                    "[Funnel] Derived a 'converted' metric (reached %s) for Experiment — "
                    "%d users, %s",
                    step_order[-1],
                    len(conversion_df),
                    humanize_metric_label("converted"),
                )
            except FunnelComputationError as exc:
                log.warning("[Funnel] Could not derive a conversion metric for the combined analysis — %s", exc)

    log.info(
        "[Funnel] Computed %d-step funnel (%s) — largest drop-off %s -> %s (%.1f%%)%s",
        len(step_order),
        " -> ".join(step_order),
        overall_result.largest_dropoff_from,
        overall_result.largest_dropoff_to,
        overall_result.largest_dropoff_rate * 100,
        f" — comparing {len(funnel_by_group)} groups" if funnel_by_group else "",
    )

    return {
        **state,
        **derived_state_updates,
        "funnel_result": overall_result,
        "funnel_by_group": funnel_by_group,
        "funnel_skip_reason": None,
    }
