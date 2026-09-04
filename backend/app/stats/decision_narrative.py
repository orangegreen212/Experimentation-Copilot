"""
Decision Narrative builder — Product-improvement (small, focused).

Turns the already-computed canonical decision facts (`Decision`,
`decision_reason`, `ExperimentValidity`, `GuardrailStatus`,
`practical_significance`, `stat_results`, `PowerAnalysisResult`,
`DatasetInfo.available_metrics`) into a structured, decision-adaptive
explanation for the report's Decision section.

Deliberately mirrors `app/stats/decision_support.py`'s stated
constraints:
  - No p-values, deltas, effects, or significance are (re)computed
    here — every bullet reads an already-decided fact.
  - `determine_decision()` / the final `Decision` is never read as an
    input to change itself here — this module is explanatory only,
    called AFTER `determine_decision()` has already run.
  - No LLM is involved anywhere in this module.

Called once, from `decision_node.py`, using the exact same
`primary_stat` selection rule `determine_decision()` itself uses
(`_select_primary_stat` below — copied rather than imported from
`TemplateReportGenerator._primary_stat` to avoid depending on a
generator's private method from a stats module; both apply the
identical winners -> significant-pairwise -> first-pairwise -> first
rule).
"""

from __future__ import annotations

from app.schemas.decision_narrative import DecisionNarrative, MonitoringInfo
from app.schemas.guardrails import GuardrailRequestState
from app.schemas.report import Decision, ExperimentValidity, GuardrailStatus
from app.schemas.statistics import PowerAnalysisResult, StatResult


def _guardrail_harmful(g: StatResult) -> bool:
    """Same direction-aware check as report_generator._guardrail_harmful — duplicated (not imported) to avoid a decision_narrative -> report_generator circular import; both read only StatResult fields, nothing decision-specific."""
    effect = g.observed_relative_effect
    if effect is None:
        try:
            effect = float((g.delta or "0").strip().rstrip("%").replace("+", ""))
        except ValueError:
            effect = 0.0
    return effect < 0 if g.higher_is_better else effect > 0


def _select_primary_stat(stat_results: list[StatResult]) -> StatResult | None:
    """Same rule as TemplateReportGenerator._primary_stat / determine_decision()'s caller."""
    winners = [r for r in stat_results if r.is_winner]
    if winners:
        return winners[0]
    pairwise = [r for r in stat_results if not r.is_omnibus]
    if pairwise:
        significant_pairwise = [r for r in pairwise if r.significant]
        return significant_pairwise[0] if significant_pairwise else pairwise[0]
    return stat_results[0] if stat_results else None


def _monitoring_info(
    primary_metric: str | None,
    guardrail_results: list[StatResult],
    available_metrics: list[str],
) -> MonitoringInfo:
    guardrails_evaluated = [g.metric for g in guardrail_results]
    potential = [
        m for m in available_metrics
        if m != primary_metric and m not in guardrails_evaluated
    ]
    return MonitoringInfo(
        primary_metric=primary_metric,
        guardrails_evaluated=guardrails_evaluated,
        potential_monitoring_metrics=potential,
    )


def build_decision_narrative(
    *,
    decision: Decision,
    decision_reason: str,
    experiment_validity: ExperimentValidity,
    guardrail_status: GuardrailStatus,
    practical_significance: bool | None,
    stat_results: list[StatResult],
    guardrail_results: list[StatResult],
    available_metrics: list[str],
    power_analysis: PowerAnalysisResult | None,
    guardrail_request_state: GuardrailRequestState = GuardrailRequestState.NOT_SPECIFIED,
) -> DecisionNarrative:
    primary_stat = _select_primary_stat(stat_results)
    primary_metric = primary_stat.metric if primary_stat is not None else None
    monitoring = _monitoring_info(primary_metric, guardrail_results, available_metrics)

    if decision == Decision.INVALID:
        return DecisionNarrative(
            why_this_decision=[decision_reason],
            what_prevents_full_go=[],
            what_would_change_decision=[
                "Resolve the underlying data-quality/validity issue and rerun the experiment.",
            ],
            monitoring=monitoring,
            recommended_next_step=(
                "Do not roll out. Investigate and fix the validity issue before drawing any "
                "conclusion from this data."
            ),
        )

    if decision == Decision.INCONCLUSIVE:
        underpowered = power_analysis is not None and not power_analysis.is_sufficiently_powered
        if underpowered:
            why = [
                "The result is inconclusive because the experiment does not have sufficient "
                "statistical power to reliably confirm the effect observed in this data."
            ]
            what_would_change = [
                f"Increase the sample size to approximately {power_analysis.required_sample_size:,} "
                "users per arm (the existing power-analysis recommendation) and rerun the experiment."
            ]
            next_step = "Increase the sample size to the recommended level and rerun the experiment."
        else:
            why = [decision_reason]
            what_would_change = [
                "Run a valid hypothesis test on this dataset (e.g. ask for a specific metric to "
                "evaluate) before a decision can be made."
            ]
            next_step = "Address why no conclusive result was produced, then rerun the analysis."
        return DecisionNarrative(
            why_this_decision=why,
            what_prevents_full_go=[],
            what_would_change_decision=what_would_change,
            monitoring=monitoring,
            recommended_next_step=next_step,
        )

    if decision == Decision.NO_GO:
        why = [decision_reason]
        if guardrail_status == GuardrailStatus.FAIL:
            failed = [g.metric for g in guardrail_results if g.significant and _guardrail_harmful(g)]
            if failed:
                why.append(f"Guardrail metric(s) {', '.join(failed)} showed a statistically significant negative change.")
        return DecisionNarrative(
            why_this_decision=why,
            what_prevents_full_go=[],
            what_would_change_decision=[],
            monitoring=monitoring,
            recommended_next_step=(
                "Do not ship this variant. Investigate the underlying cause before considering a "
                "revised experiment."
            ),
        )

    # --- GO / GO_WITH_CAUTION -------------------------------------------
    why: list[str] = []
    if primary_stat is not None and primary_stat.significant:
        why.append(f"{primary_stat.metric} is statistically significant.")
    if practical_significance is True:
        why.append("The observed effect is practically meaningful.")
    elif practical_significance is None:
        why.append("Practical significance of the effect could not be reliably established.")
    if experiment_validity == ExperimentValidity.VALID:
        why.append("Data quality and randomization checks passed.")
    elif experiment_validity == ExperimentValidity.CAUTION:
        why.append("Data quality checks passed with some non-critical warnings.")
    if power_analysis is not None and power_analysis.is_sufficiently_powered:
        why.append("Statistical power is sufficient.")
    if guardrail_status == GuardrailStatus.NOT_AVAILABLE:
        if guardrail_request_state == GuardrailRequestState.REQUESTED_NOT_FOUND:
            why.append("The requested guardrails could not be evaluated because the corresponding metrics were not available in the dataset.")
        elif guardrail_request_state in (
            GuardrailRequestState.AVAILABLE,
            GuardrailRequestState.PARTIALLY_AVAILABLE,
        ):
            # Resolved-but-not-evaluated (e.g. multi-arm — see
            # guardrail_node.py) must never read the same as "not
            # specified"; it's a distinct, more specific fact.
            why.append("The requested guardrail metric(s) were found in this dataset but could not be statistically evaluated for this experiment.")
        else:
            why.append("No guardrail metrics were evaluated.")
    elif guardrail_status == GuardrailStatus.PASS:
        why.append("Guardrail metrics evaluated so far passed.")
    elif guardrail_status == GuardrailStatus.WARNING:
        why.append("A guardrail metric showed a borderline/warning-level change.")

    if decision == Decision.GO:
        return DecisionNarrative(
            why_this_decision=why,
            what_prevents_full_go=[],
            what_would_change_decision=[],
            monitoring=monitoring,
            recommended_next_step="Proceed with rollout and monitor the primary metric and relevant guardrails.",
        )

    # GO_WITH_CAUTION
    what_prevents: list[str] = []
    if guardrail_status == GuardrailStatus.NOT_AVAILABLE:
        if guardrail_request_state == GuardrailRequestState.REQUESTED_NOT_FOUND:
            what_prevents.append("The requested guardrail metric(s) were not found in this dataset.")
            what_prevents.append("Business impact has not been validated against those requested guardrails.")
        elif guardrail_request_state in (
            GuardrailRequestState.AVAILABLE,
            GuardrailRequestState.PARTIALLY_AVAILABLE,
        ):
            what_prevents.append("The requested guardrail metric(s) were found but could not be statistically evaluated for this experiment (e.g. multi-arm guardrail evaluation is not yet supported).")
            what_prevents.append("Business impact has not been validated against those requested guardrails.")
        else:
            what_prevents.append("Guardrail metrics are unavailable.")
            what_prevents.append("Business impact has not been explicitly validated against a guardrail.")
        what_prevents.append("The business decision therefore relies on the primary metric only.")
    elif guardrail_status == GuardrailStatus.WARNING:
        what_prevents.append(
            "A guardrail metric shows a borderline/warning-level change that should be reviewed "
            "before a full rollout."
        )
    if practical_significance is None:
        what_prevents.append("Practical significance of the effect could not be reliably established.")

    return DecisionNarrative(
        why_this_decision=why,
        what_prevents_full_go=what_prevents,
        what_would_change_decision=[],
        monitoring=monitoring,
        recommended_next_step=(
            "Proceed with a controlled rollout and monitor the primary metric together with "
            "relevant guardrails before full rollout."
        ),
    )
