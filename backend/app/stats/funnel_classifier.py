"""
Funnel dataset classifier — detects whether a dataset has the
event-log structure a funnel analysis needs (user_id + event +
timestamp), separate from `dataset_classifier.py`'s Aggregated-vs-Raw
classification, since a funnel needs specifically an EVENT column
with multiple distinct step values, not just "raw event-level data" in
general.

Step order is INFERRED from the data itself (median timestamp per
event), not hardcoded to any particular funnel's step names — this is
a deliberate, documented heuristic: it assumes each event's typical
position in a user's journey correlates with when it tends to happen
across the whole dataset. This holds for genuine funnels (Visit always
precedes Signup precedes...) and is the honest, general-purpose
alternative to hardcoding "Visit, Signup, Trial, Purchase" as if that
were the only possible funnel.
"""

from __future__ import annotations

import pandas as pd

from app.stats.dataset_classifier import _detect_user_column, _detect_variant_column

_EVENT_COLUMN_CANDIDATES = ["event", "event_name", "event_type", "action", "step"]
_TIMESTAMP_COLUMN_CANDIDATES = ["timestamp", "event_time", "created_at", "occurred_at", "time"]
_MIN_DISTINCT_EVENTS_FOR_FUNNEL = 2


class FunnelClassificationError(ValueError):
    """Raised when a funnel analysis is requested but the dataset doesn't have the required structure."""


class FunnelColumns:
    """Resolved column roles for funnel analysis — mirrors ExperimentColumns' role for A/B analysis."""

    def __init__(self, user_col: str, event_col: str, timestamp_col: str, group_col: str | None):
        self.user_col = user_col
        self.event_col = event_col
        self.timestamp_col = timestamp_col
        self.group_col = group_col  # optional — e.g. an A/B experiment_group column, for compute_funnel_by_group


def detect_funnel_columns(df: pd.DataFrame) -> FunnelColumns | None:
    """
    Returns FunnelColumns if this dataset looks like a funnel/event
    log (has a user id column, an event column with >=2 distinct
    values, and a timestamp column), otherwise None — callers must
    treat None as "not a funnel dataset," not an error, since most
    datasets legitimately aren't.
    """
    user_col = _detect_user_column(df)
    if user_col is None:
        return None

    event_col = _find_first_match(df, _EVENT_COLUMN_CANDIDATES)
    if event_col is None or df[event_col].nunique(dropna=True) < _MIN_DISTINCT_EVENTS_FOR_FUNNEL:
        return None

    timestamp_col = _find_first_match(df, _TIMESTAMP_COLUMN_CANDIDATES)
    if timestamp_col is None:
        return None

    group_col = _detect_variant_column(df)  # optional — reused from dataset_classifier, may be None

    return FunnelColumns(user_col=user_col, event_col=event_col, timestamp_col=timestamp_col, group_col=group_col)


def infer_step_order(df: pd.DataFrame, event_col: str, timestamp_col: str) -> list[str]:
    """
    Orders distinct event values by their median timestamp across the
    whole dataset — earliest-typically-occurring event first. See
    module docstring for why this heuristic (not a hardcoded step
    list) is the honest general-purpose choice.
    """
    working = df[[event_col, timestamp_col]].copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col])
    order = working.groupby(event_col)[timestamp_col].median().sort_values().index.tolist()
    return order


def _find_first_match(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None
