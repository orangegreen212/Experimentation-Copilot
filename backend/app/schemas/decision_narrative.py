"""
Decision Narrative — deterministic, template-only explanation layer on
top of the existing canonical decision model (`Decision` /
`ExperimentValidity` / `GuardrailStatus` / `determine_decision()`).

This is explicitly NOT a new decision engine and NOT a second
statistical analysis: every field here is assembled from facts that
already exist on `ExperimentReport` / `ReportFacts` by the time
`decision_node.py` builds it — the same pattern as `DecisionSupport`
(`app/stats/decision_support.py`) and `SegmentationResult`. No LLM is
involved anywhere in this module; see `app/stats/decision_narrative.py`
for the pure-Python assembly logic.

Purely additive: existing reports render identically if this field is
absent/None (e.g. a purely conceptual/funnel-only report that never
reached `determine_decision()` at all).
"""

from __future__ import annotations

from app.schemas.base import CamelModel


class MonitoringInfo(CamelModel):
    """
    What to watch after rollout — built ONLY from metrics the
    application already knows about (`ReportFacts.dataset.available_metrics`,
    `ReportFacts.guardrail_results`). Never invents a metric name.
    """

    primary_metric: str | None
    # Metrics that were ACTUALLY evaluated as guardrails (i.e. present
    # in ReportFacts.guardrail_results) — today this is almost always
    # empty, since guardrail computation is a separate, not-yet-built
    # phase (see decision_support.py's module docstring). Never
    # conflated with `potential_monitoring_metrics` below.
    guardrails_evaluated: list[str]
    # Other numeric metrics already present in the dataset
    # (`DatasetInfo.available_metrics`) that were NOT evaluated as
    # guardrails and are not the primary metric — offered as candidates
    # to watch, never labeled as guardrails.
    potential_monitoring_metrics: list[str]


class DecisionNarrative(CamelModel):
    """
    Structured, decision-adaptive explanation of `ExperimentReport.decision`.
    Every bullet is a direct restatement of an already-computed fact
    (decision, decision_reason, experiment_validity, guardrail_status,
    practical_significance, stat_results, power_analysis) — nothing is
    inferred beyond what those fields already say.
    """

    why_this_decision: list[str]
    # Empty for GO/INCONCLUSIVE/NO_GO/INVALID — only ever populated for
    # GO_WITH_CAUTION, where there's a specific, factual reason a full
    # GO wasn't reached (e.g. guardrails unavailable).
    what_prevents_full_go: list[str]
    # Populated ONLY for INCONCLUSIVE when the existing power-analysis
    # result says the experiment is underpowered — reuses
    # PowerAnalysisResult.required_sample_size verbatim, never a new
    # calculation.
    what_would_change_decision: list[str]
    monitoring: MonitoringInfo
    recommended_next_step: str
