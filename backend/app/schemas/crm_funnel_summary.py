"""
CRM funnel summary schemas — small, additive output shape for
`app.stats.crm_funnel_summary`. See that module's docstring for why
this exists and what it deliberately does NOT do (no new statistical
method, no new correction — it only assembles results already
produced by the existing per-metric hypothesis-test engine).
"""

from __future__ import annotations

from app.schemas.base import CamelModel
from app.schemas.statistics import StatResult


class CrmFunnelArmValue(CamelModel):
    """One arm's observed value for one funnel metric — a display string (e.g. '18.5%', '$1.42')."""

    arm: str
    display_value: str


class CrmFunnelMetricRow(CamelModel):
    """
    One funnel-stage metric (e.g. 'visit', 'conversion', 'spend')
    across every arm, plus each treatment arm's StatResult vs the
    resolved control — the SAME StatResult objects
    `compute_stat_result` already produces for a normal two-arm
    comparison; nothing new is computed here.
    """

    metric_column: str
    metric_label: str
    control_label: str
    arm_values: list[CrmFunnelArmValue]
    treatment_results: list[StatResult]


class CrmFunnelSummary(CamelModel):
    """
    Top-level result: one row per funnel-stage metric, in the order
    they were supplied, plus a plain-language narrative connecting
    consecutive stages (e.g. "X increased visit rate, but the
    additional visits did not translate into a statistically
    significant increase in conversion").
    """

    rows: list[CrmFunnelMetricRow]
    narrative: list[str]
