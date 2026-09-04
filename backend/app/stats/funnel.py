"""
Funnel analysis — deterministic tool, Stage: Product Analytics
capability #2. Same rule as every other module in app/stats/: no LLM,
no network, fully unit-testable in isolation.

FUNNEL SEMANTICS (a deliberate choice, documented so it's not a silent
assumption): a user counts as having reached step N only if they ALSO
reached every step before it, in order — i.e. `users_at_step[N]` is
computed as the intersection of `users_at_step[N-1]` with the set of
users who ever performed step N's event, not just "distinct users who
did step N's event" (which would let a user who skipped step 2 but did
step 3 count toward step 3 — not a real funnel). This is what makes
step counts monotonically non-increasing, which is what "funnel" means.
"""

from __future__ import annotations

import pandas as pd

from app.schemas.funnel import FunnelAnalysisResult, FunnelStepResult


class FunnelComputationError(ValueError):
    """Raised when the funnel can't be computed at all (empty step order, no users at step 1, etc.)."""


def compute_funnel(df: pd.DataFrame, user_col: str, event_col: str, step_order: list[str]) -> FunnelAnalysisResult:
    """
    `step_order` is the caller-supplied (or inferred, via
    funnel_classifier.infer_step_order) sequence of event names, first
    step first. Returns FunnelAnalysisResult with strict step-over-step
    user counts — see module docstring for the exact semantics.
    """
    if len(step_order) < 2:
        raise FunnelComputationError("A funnel needs at least 2 steps to compute drop-off.")

    users_per_event: dict[str, set] = {
        step: set(df.loc[df[event_col] == step, user_col].unique()) for step in step_order
    }

    if len(users_per_event[step_order[0]]) == 0:
        raise FunnelComputationError(f"No users found for the first step ({step_order[0]!r}) — cannot compute a funnel.")

    surviving_users = users_per_event[step_order[0]]
    steps: list[FunnelStepResult] = [
        FunnelStepResult(
            name=step_order[0],
            users=len(surviving_users),
            conversion_from_start=1.0,
            conversion_from_previous=1.0,
        )
    ]

    first_step_count = len(surviving_users)
    previous_count = first_step_count

    for step in step_order[1:]:
        surviving_users = surviving_users & users_per_event[step]
        count = len(surviving_users)
        steps.append(
            FunnelStepResult(
                name=step,
                users=count,
                conversion_from_start=count / first_step_count,
                conversion_from_previous=(count / previous_count) if previous_count > 0 else 0.0,
            )
        )
        previous_count = count

    # Largest drop-off = the step transition (i-1 -> i) with the
    # HIGHEST fraction of users lost, i.e. lowest conversion_from_previous
    # among steps after the first.
    worst_idx = min(range(1, len(steps)), key=lambda i: steps[i].conversion_from_previous)
    largest_dropoff_rate = 1.0 - steps[worst_idx].conversion_from_previous

    return FunnelAnalysisResult(
        steps=steps,
        overall_conversion=steps[-1].users / first_step_count,
        largest_dropoff_from=steps[worst_idx - 1].name,
        largest_dropoff_to=steps[worst_idx].name,
        largest_dropoff_rate=largest_dropoff_rate,
    )


def compute_funnel_by_group(
    df: pd.DataFrame, user_col: str, event_col: str, step_order: list[str], group_col: str
) -> dict[str, FunnelAnalysisResult]:
    """
    Same computation, once per distinct value in `group_col` (e.g.
    A/B experiment_group). This is what makes the combined "did B fix
    the drop-off?" use case possible — one funnel per arm, comparable
    side by side. Still zero LLM involvement; comparison/interpretation
    happens in report generation, not here.
    """
    if not group_col:
        raise FunnelComputationError("compute_funnel_by_group requires a non-empty group_col.")

    results = {}
    for group_value in sorted(df[group_col].dropna().unique().tolist(), key=str):
        group_df = df[df[group_col] == group_value]
        results[str(group_value)] = compute_funnel(group_df, user_col, event_col, step_order)
    return results


def derive_conversion_dataframe(
    df: pd.DataFrame, user_col: str, event_col: str, group_col: str, step_order: list[str]
) -> pd.DataFrame:
    """
    Bridges Funnel Analysis and Experiment Review: builds a one-row-
    per-user aggregated DataFrame (columns: `user_col`, `group_col`,
    "converted") from a raw event log, so a funnel/event dataset with
    NO separate outcome column can still be used for a hypothesis test
    — the natural conversion metric for a funnel IS "did this user
    complete the whole funnel" (reach the final step, in order).

    Only used by funnel_node.py for the COMBINED funnel+experiment
    case, when Classifier found no metric column at all. This is a
    genuine, defensible metric definition — not a workaround: "did the
    user reach the last funnel step" is exactly what "converted" means
    for a multi-step conversion funnel. Uses the SAME strict step-over-
    step semantics as `compute_funnel` (a user only counts as
    converted if they went through every step in order), so the
    resulting "converted" column is consistent with the funnel numbers
    shown alongside it in the report.

    One row per user who appears at all in the dataset (not just users
    who reached step 1) — users who never even entered the funnel are
    excluded, since they were never really in the experiment's funnel
    population to begin with.
    """
    if len(step_order) < 2:
        raise FunnelComputationError("A funnel needs at least 2 steps to derive a conversion metric.")

    users_per_event: dict[str, set] = {
        step: set(df.loc[df[event_col] == step, user_col].unique()) for step in step_order
    }
    completed_users = users_per_event[step_order[0]]
    for step in step_order[1:]:
        completed_users = completed_users & users_per_event[step]

    all_funnel_users = set(df[user_col].unique())
    group_by_user = df.drop_duplicates(subset=[user_col]).set_index(user_col)[group_col].to_dict()

    rows = [
        {
            user_col: user_id,
            group_col: group_by_user.get(user_id),
            "converted": 1 if user_id in completed_users else 0,
        }
        for user_id in all_funnel_users
    ]
    return pd.DataFrame(rows)
