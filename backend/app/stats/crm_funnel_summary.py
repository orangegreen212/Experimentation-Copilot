"""
CRM funnel summary — small, additive tool for unit-level CRM/marketing
experiments (e.g. Hillstrom-shaped data: one row per customer, a
sequence of business outcomes like visit -> conversion -> spend,
rather than an event log).

DELIBERATELY DOES NOT:
  - add a new statistical test, correction method, or test-selection
    rule. Every number in `CrmFunnelMetricRow.treatment_results` comes
    straight from the EXISTING `compute_stat_result()` in
    `app.stats.hypothesis_tests` — this module only calls it once per
    (funnel metric, treatment arm) pair and assembles the results.
  - replace or extend `app.stats.funnel` (that module is for genuine
    EVENT-LOG funnels — one row per event, an event_col naming which
    step occurred, step order inferred from timestamps). A CRM
    unit-level dataset like Hillstrom has no event log at all: each
    funnel "stage" is just a separate column on the same row. This
    module exists specifically for that shape.

The narrative is plain, deterministic Python string composition over
already-computed `StatResult.significant` flags — never an LLM
decision and never a new number.
"""

from __future__ import annotations

import pandas as pd

from app.schemas.crm_funnel_summary import (
    CrmFunnelArmValue,
    CrmFunnelMetricRow,
    CrmFunnelSummary,
)
from app.schemas.statistics import MetricType, StatResult
from app.stats.dataset_classifier import humanize_metric_label
from app.stats.hypothesis_tests import compute_stat_result


def compute_crm_funnel_summary(
    df: pd.DataFrame,
    variant_col: str,
    control_label: str,
    funnel_metrics: list[tuple[str, MetricType]],
) -> CrmFunnelSummary:
    """
    `funnel_metrics`: ordered list of (column_name, metric_type) pairs,
    earliest funnel stage first (e.g. [("visit", BINARY),
    ("conversion", BINARY), ("spend", CONTINUOUS_MONETARY)]). Caller
    decides which columns are funnel stages and their order — this
    function does not infer that (see `detect_funnel_metrics` in
    `dataset_classifier.py` for one way to get that list; it currently
    excludes the dataset's chosen primary metric, so for something
    like Hillstrom's `conversion` you may need to pass it in
    explicitly alongside `detect_funnel_metrics`'s output).

    Every treatment arm (every value in `variant_col` except
    `control_label`) is compared against `control_label`, once per
    funnel metric, via the existing two-arm `compute_stat_result`.
    Raises no new error type — an empty/missing metric column, or a
    dataset with fewer than 2 arms, surfaces as whatever
    `compute_stat_result`/pandas already raises for that input.
    """
    treatment_labels = [
        v for v in df[variant_col].dropna().unique().tolist() if v != control_label
    ]

    rows: list[CrmFunnelMetricRow] = []
    for metric_col, metric_type in funnel_metrics:
        metric_label = humanize_metric_label(metric_col)
        control_series = df.loc[df[variant_col] == control_label, metric_col]

        treatment_results: list[StatResult] = []
        arm_values = [
            CrmFunnelArmValue(arm=str(control_label), display_value=_display_value(control_series, metric_type))
        ]
        for label in treatment_labels:
            variant_series = df.loc[df[variant_col] == label, metric_col]
            result = compute_stat_result(control_series, variant_series, metric_type, metric_label)
            result = result.model_copy(update={"comparison": f"{control_label} vs {label}", "arm": str(label)})
            treatment_results.append(result)
            arm_values.append(CrmFunnelArmValue(arm=str(label), display_value=result.variant))

        rows.append(
            CrmFunnelMetricRow(
                metric_column=metric_col,
                metric_label=metric_label,
                control_label=str(control_label),
                arm_values=arm_values,
                treatment_results=treatment_results,
            )
        )

    return CrmFunnelSummary(rows=rows, narrative=_build_narrative(rows))


def _display_value(series: pd.Series, metric_type: MetricType) -> str:
    """Same display convention `compute_stat_result` already uses for `control`/`variant` — recomputed only for the control arm's own row (never re-derives a treatment arm's value; those come straight from the StatResult above)."""
    clean = series.dropna()
    if metric_type == MetricType.BINARY:
        return f"{clean.mean() * 100:.2f}%"
    if metric_type == MetricType.CONTINUOUS_MONETARY:
        return f"${clean.mean():.2f}"
    return f"{clean.mean():.2f}"


def _build_narrative(rows: list[CrmFunnelMetricRow]) -> list[str]:
    """
    Deterministic, plain-language bridge between CONSECUTIVE funnel
    stages, per treatment arm — e.g. "Mens E-Mail increased visit
    rate, but the additional visits did not translate into a
    statistically significant increase in conversion." Compares only
    `StatResult.significant` (already computed above) — no new
    statistic, no threshold not already used elsewhere in the engine.
    """
    if len(rows) < 2:
        return []

    narrative: list[str] = []
    for earlier, later in zip(rows, rows[1:]):
        earlier_by_arm = {r.arm: r for r in earlier.treatment_results}
        later_by_arm = {r.arm: r for r in later.treatment_results}
        for arm in earlier_by_arm:
            earlier_result = earlier_by_arm[arm]
            later_result = later_by_arm.get(arm)
            if later_result is None:
                continue
            if earlier_result.significant and later_result.significant:
                narrative.append(
                    f"{arm} increased {earlier.metric_label.lower()} and this carried through to a "
                    f"statistically significant increase in {later.metric_label.lower()}."
                )
            elif earlier_result.significant and not later_result.significant:
                narrative.append(
                    f"{arm} increased {earlier.metric_label.lower()}, but the gain did not translate into a "
                    f"statistically significant increase in {later.metric_label.lower()}."
                )
            elif not earlier_result.significant and later_result.significant:
                narrative.append(
                    f"{arm} showed no statistically significant change in {earlier.metric_label.lower()}, but "
                    f"{later.metric_label.lower()} was still significantly higher than {earlier.control_label}."
                )
            # Neither significant: no narrative line — nothing notable to report for this arm/stage pair.
    return narrative
