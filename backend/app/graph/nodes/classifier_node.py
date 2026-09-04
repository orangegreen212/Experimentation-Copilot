"""
Classifier node — wraps app/stats/dataset_classifier.py. No logic
lives here.

ARCHITECTURAL RULE: Classifier describes what's IN the dataset — it
never requires the dataset to be suitable for every capability. A
funnel/event-log dataset (user_id, event, timestamp) legitimately has
no "metric" column at all, and that's fine — `experiment_columns` is
simply None in that case. It's each CAPABILITY's job (validation_node,
experiment_node) to check whether the columns IT needs are present
and fail clearly — with a real error message, never a bare KeyError —
only if that capability actually gets routed to. A funnel-only request
on this same dataset never touches `experiment_columns` at all.
"""

from app.core.dataset_store import get_dataset, store_dataset
from app.core.logging import get_node_logger
from app.graph.state import GraphState
from app.stats.dataset_classifier import (
    DatasetClassificationError,
    attach_implicit_unit_id,
    classify_dataset,
    detect_experiment_columns,
    enrich_with_assignment,
)

log = get_node_logger("Classifier")


def classifier_node(state: GraphState) -> GraphState:
    df = get_dataset(state["dataset_id"])

    # The user's own request text is matched against column names/labels
    # deterministically inside classify_dataset (see
    # dataset_classifier._select_metric_column) — this is plain string
    # matching, not an LLM call, so metric selection stays a classifier
    # fact rather than an LLM interpretation.
    preferred_metric = state.get("user_prompt")

    # Optional separate experiment-assignment dataset (user_id | variant),
    # uploaded through the SAME /datasets/classify mechanism as the
    # primary dataset — see routes_datasets.py / routes_experiments.py.
    # `state.get(...)` returns None for every existing single-file
    # request, so this whole block is a no-op / unchanged behavior
    # unless an analyst actually uploaded a second file.
    assignment_dataset_id = state.get("assignment_dataset_id")
    resolved_dataset_id = state["dataset_id"]
    if assignment_dataset_id:
        assignment_df = get_dataset(assignment_dataset_id)
        # Merge ONCE, here, and persist the merged frame under a NEW
        # dataset_id via the existing dataset_store — never a raw
        # DataFrame in GraphState (see dataset_store.py's module
        # docstring). This is required, not cosmetic: validation_node
        # and experiment_node each independently call
        # get_dataset(state["dataset_id"]) themselves; without
        # persisting the merge here and repointing dataset_id at it,
        # those nodes would re-fetch the ORIGINAL unenriched primary
        # dataset and crash looking up the variant column classify_
        # dataset/detect_experiment_columns resolved only transiently
        # against the merged frame. Every node downstream of this one
        # sees the SAME already-merged dataset via the same dataset_id
        # mechanism as always — no parallel data path, no schema change.
        df = enrich_with_assignment(df, assignment_df)
        resolved_dataset_id = store_dataset(df)

    dataset_info = classify_dataset(df, preferred_metric=preferred_metric)

    try:
        columns = detect_experiment_columns(df, preferred_metric=preferred_metric)
        metric_summary = f"metric_col={columns.metric_col} ({columns.metric_type.value})"
        columns_error = None
        if columns.is_implicit_unit:
            # `detect_experiment_columns` only returns the synthetic
            # unit-id COLUMN NAME — the dataframe it actually built that
            # column on was a transient local copy that's already gone.
            # Every downstream node (experiment_node, segmentation)
            # re-fetches `df` via `get_dataset(state["dataset_id"])` and
            # indexes it by `columns.user_col`; without materializing
            # and persisting the same column here, that lookup raises
            # KeyError for every genuinely unit-level (implicit-row)
            # dataset — e.g. a CRM export with no customer_id column.
            # Mirrors the assignment-merge re-store pattern above.
            df, implicit_col = attach_implicit_unit_id(df)
            assert implicit_col == columns.user_col
            resolved_dataset_id = store_dataset(df)
    except DatasetClassificationError as exc:
        # Not an error at the Classifier stage — just means this dataset
        # doesn't support experiment/validation capabilities. A funnel-
        # only or knowledge-base-only request never needs this field.
        columns = None
        columns_error = str(exc)
        metric_summary = f"no experiment metric column detected ({exc})"

    log.info(
        "[Classifier] Dataset classified — type=%s, users=%d, variants=%d, %s",
        dataset_info.type.value,
        dataset_info.users,
        dataset_info.variants,
        metric_summary,
    )

    return {
        **state,
        "dataset_id": resolved_dataset_id,
        "dataset": dataset_info,
        "experiment_columns": columns,
        "experiment_columns_error": columns_error,
    }
