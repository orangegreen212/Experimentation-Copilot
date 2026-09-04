"""
Deterministic Decision Audit Trail — Phase 7.

Turns already-computed facts (`ReportFacts`, the final `ExperimentReport`
— specifically its canonical `decision`/`decisionReason`/
`experimentValidity`/`guardrailStatus`/`practicalSignificance` fields)
into a structured `DecisionAuditTrail` that answers "why did the
system reach this exact decision?" This is explicitly NOT a second
statistical analysis and NOT a second decision engine (Phase 7 spec
§12):

  - `decision` is always copied from `report.decision` (produced by
    `determine_decision()` in app/graph/report_generator.py) — never
    recomputed, never overridden here.
  - No p-values, confidence intervals, MDE, or power are (re)computed
    — every number is read off an existing `StatResult` /
    `PowerAnalysisResult` row via `select_primary_stat()` /
    `format_p_value()` (both reused, not reimplemented).
  - No LLM is involved anywhere in this module.
  - Segmentation findings are surfaced as supporting evidence only —
    exactly like `SegmentationResult` itself — and never change
    `decision`.

Called once, from `decision_node.py`, AFTER report generation (and
after the emergency-fallback path, if that's what ran) — see that
module's docstring for why every field is stamped on in one place.
"""

from __future__ import annotations

from app.graph.report_generator import ReportFacts, select_primary_stat
from app.schemas.decision_audit import (
    AuditCategory,
    AuditFact,
    AuditImpact,
    AuditStatus,
    DecisionAuditTrail,
)
from app.schemas.quality import QualityCheck
from app.schemas.report import Decision, ExperimentReport, GuardrailStatus
from app.schemas.guardrails import GuardrailRequestState
from app.schemas.statistics import StatResult
from app.stats.hypothesis_tests import format_p_value


def _find_quality_check(quality_checks: list[QualityCheck], *label_fragments: str) -> QualityCheck | None:
    """First quality check whose label contains any of the given fragments (case-insensitive)."""
    for qc in quality_checks:
        if any(fragment.lower() in qc.label.lower() for fragment in label_fragments):
            return qc
    return None


def _critical_checks(facts: ReportFacts) -> list[AuditFact]:
    """
    Validity-specific facts (spec §3/INVALID, §12): SRM, conflicting
    variant assignment, and any other critical/non-critical
    data-quality check already present in `facts.quality_checks`.
    Always populated, even when everything passed — so a VALID
    experiment's audit trail can positively state "no critical
    warnings were found" rather than saying nothing.
    """
    checks: list[AuditFact] = []

    srm_check = _find_quality_check(facts.quality_checks, "Sample Ratio Mismatch", "SRM")
    checks.append(
        AuditFact(
            status=AuditStatus.PASS if facts.srm_passed else AuditStatus.FAIL,
            category=AuditCategory.VALIDITY,
            label="Sample Ratio Mismatch (SRM)",
            value="PASS" if facts.srm_passed else "FAIL",
            impact=AuditImpact.SUPPORTS_DECISION if facts.srm_passed else AuditImpact.BLOCKS_DECISION,
            detail=srm_check.detail if srm_check is not None else None,
        )
    )

    dup_check = _find_quality_check(facts.quality_checks, "Duplicate User Variant Conflicts")
    checks.append(
        AuditFact(
            status=AuditStatus.FAIL if facts.has_conflicting_variant_duplicates else AuditStatus.PASS,
            category=AuditCategory.VALIDITY,
            label="Duplicate / conflicting variant assignment",
            value="CONFLICTS FOUND" if facts.has_conflicting_variant_duplicates else "NONE",
            impact=(
                AuditImpact.BLOCKS_DECISION
                if facts.has_conflicting_variant_duplicates
                else AuditImpact.SUPPORTS_DECISION
            ),
            detail=dup_check.detail if dup_check is not None else None,
        )
    )

    critical_quality_failures = [qc for qc in facts.quality_checks if not qc.passed and qc.critical]
    if critical_quality_failures:
        for qc in critical_quality_failures:
            checks.append(
                AuditFact(
                    status=AuditStatus.FAIL,
                    category=AuditCategory.DATA_QUALITY,
                    label=qc.label,
                    value="FAIL (critical)",
                    impact=AuditImpact.BLOCKS_DECISION,
                    detail=qc.detail,
                )
            )
    else:
        checks.append(
            AuditFact(
                status=AuditStatus.PASS,
                category=AuditCategory.DATA_QUALITY,
                label="Critical data-quality checks",
                value="NONE FAILED",
                impact=AuditImpact.SUPPORTS_DECISION,
            )
        )

    non_critical_failures = [
        qc for qc in facts.quality_checks if not qc.passed and not qc.critical and not qc.informational
    ]
    for qc in non_critical_failures:
        checks.append(
            AuditFact(
                status=AuditStatus.WARNING,
                category=AuditCategory.DATA_QUALITY,
                label=qc.label,
                value="WARNING",
                impact=AuditImpact.LIMITS_CONFIDENCE,
                detail=qc.detail,
            )
        )

    return checks


def _statistical_evidence(primary_stat: StatResult | None) -> list[AuditFact]:
    """Primary-metric identity + significance — never re-derives p-values, only reads them."""
    if primary_stat is None:
        return [
            AuditFact(
                status=AuditStatus.NOT_AVAILABLE,
                category=AuditCategory.STATISTICAL_SIGNIFICANCE,
                label="Primary metric",
                value="No hypothesis test was run",
                impact=AuditImpact.NEUTRAL,
            )
        ]
    return [
        AuditFact(
            status=AuditStatus.INFO,
            category=AuditCategory.STATISTICAL_SIGNIFICANCE,
            label="Primary metric",
            value=primary_stat.metric,
            impact=AuditImpact.NEUTRAL,
            detail=primary_stat.selection_reason,
        ),
        AuditFact(
            status=AuditStatus.PASS if primary_stat.significant else AuditStatus.WARNING,
            category=AuditCategory.STATISTICAL_SIGNIFICANCE,
            label="Statistical significance",
            value=(
                f"{'Significant' if primary_stat.significant else 'Not significant'} "
                f"(p {format_p_value(primary_stat.p_value)})"
            ),
            impact=AuditImpact.SUPPORTS_DECISION if primary_stat.significant else AuditImpact.NEUTRAL,
            detail=(
                f"Control {primary_stat.control} vs. variant {primary_stat.variant}, "
                f"delta {primary_stat.delta}, 95% CI [{primary_stat.ci_lower}, {primary_stat.ci_upper}]."
            ),
        ),
    ]


def _practical_significance_evidence(facts: ReportFacts, report: ExperimentReport) -> AuditFact | None:
    if report.practical_significance is None:
        return AuditFact(
            status=AuditStatus.NOT_AVAILABLE,
            category=AuditCategory.PRACTICAL_SIGNIFICANCE,
            label="Practical significance",
            value="Could not be established",
            impact=AuditImpact.LIMITS_CONFIDENCE,
            detail=f"MDE: {facts.mde_display}",
        )
    return AuditFact(
        status=AuditStatus.PASS if report.practical_significance else AuditStatus.WARNING,
        category=AuditCategory.PRACTICAL_SIGNIFICANCE,
        label="Practical significance",
        value="Confirmed" if report.practical_significance else "Below the practical MDE threshold",
        impact=AuditImpact.SUPPORTS_DECISION if report.practical_significance else AuditImpact.NEUTRAL,
        detail=f"MDE: {facts.mde_display} (post-hoc, computed from the observed sample size).",
    )


def _power_evidence(facts: ReportFacts) -> AuditFact | None:
    pa = facts.power_analysis
    if pa is None:
        return AuditFact(
            status=AuditStatus.NOT_AVAILABLE,
            category=AuditCategory.POWER,
            label="Statistical power",
            value="Not calculated",
            impact=AuditImpact.NEUTRAL,
        )
    return AuditFact(
        status=AuditStatus.PASS if pa.is_sufficiently_powered else AuditStatus.WARNING,
        category=AuditCategory.POWER,
        label="Statistical power",
        value=(
            f"{pa.achieved_power * 100:.1f}% achieved"
            + ("" if pa.is_sufficiently_powered else " — underpowered")
        ),
        impact=AuditImpact.SUPPORTS_DECISION if pa.is_sufficiently_powered else AuditImpact.LIMITS_CONFIDENCE,
        detail=facts.sample_size_note,
    )


_GUARDRAIL_STATUS_TO_AUDIT_STATUS = {
    GuardrailStatus.PASS: AuditStatus.PASS,
    GuardrailStatus.WARNING: AuditStatus.WARNING,
    GuardrailStatus.FAIL: AuditStatus.FAIL,
    GuardrailStatus.NOT_AVAILABLE: AuditStatus.NOT_AVAILABLE,
}
_GUARDRAIL_STATUS_TO_IMPACT = {
    GuardrailStatus.PASS: AuditImpact.SUPPORTS_DECISION,
    GuardrailStatus.WARNING: AuditImpact.LIMITS_CONFIDENCE,
    GuardrailStatus.FAIL: AuditImpact.BLOCKS_DECISION,
    GuardrailStatus.NOT_AVAILABLE: AuditImpact.LIMITS_CONFIDENCE,
}


def _guardrail_evidence(report: ExperimentReport) -> AuditFact:
    """
    NOT_AVAILABLE is never rendered as PASS (spec §7) — it gets its
    own status/impact, distinct from an actual guardrail PASS, and an
    explicit explanatory detail that distinguishes "never requested"
    from "requested but not found in this dataset" (guardrail
    root-cause fix) via `report.guardrail_request_state`.
    """
    detail = None
    if report.guardrail_status == GuardrailStatus.NOT_AVAILABLE:
        if report.guardrail_request_state == GuardrailRequestState.REQUESTED_NOT_FOUND:
            missing = ", ".join(r.requested_name for r in report.guardrail_resolutions if not r.resolved)
            detail = (
                f"Requested guardrail(s) could not be evaluated because the corresponding metric(s) "
                f"were not available in this dataset: {missing}. The recommendation is based only on "
                "the primary metric."
            )
        elif report.guardrail_request_state in (
            GuardrailRequestState.AVAILABLE,
            GuardrailRequestState.PARTIALLY_AVAILABLE,
        ):
            # BUG FIX: resolved (matched a real dataset column) but not
            # evaluated (e.g. multi-arm — see guardrail_node.py) is a
            # distinct fact from "not found"; conflating the two is what
            # produced a Decision Strip badge/reason contradiction.
            found = ", ".join(r.requested_name for r in report.guardrail_resolutions if r.resolved)
            detail = (
                f"Requested guardrail(s) were found in this dataset ({found}) but could not be "
                "statistically evaluated for this experiment (e.g. multi-arm guardrail evaluation is "
                "not yet supported). The recommendation is based only on the primary metric."
            )
        else:
            detail = (
                "No guardrail metric was evaluated for this dataset — the recommendation is based "
                "only on the primary metric."
            )
    return AuditFact(
        status=_GUARDRAIL_STATUS_TO_AUDIT_STATUS[report.guardrail_status],
        category=AuditCategory.GUARDRAILS,
        label="Guardrails",
        value=report.guardrail_status.value.replace("_", " "),
        impact=_GUARDRAIL_STATUS_TO_IMPACT[report.guardrail_status],
        detail=detail,
    )


def _segmentation_evidence(facts: ReportFacts) -> AuditFact:
    """
    Segmentation is supporting evidence only (spec §6/§12) — this fact
    always has `impact=NEUTRAL`; it is never allowed to imply it
    changed the decision.
    """
    seg = facts.segmentation_result
    if seg is None:
        return AuditFact(
            status=AuditStatus.NOT_AVAILABLE,
            category=AuditCategory.SEGMENTATION,
            label="Segmentation",
            value="Not run",
            impact=AuditImpact.NEUTRAL,
            detail="Segmentation did not run for this experiment.",
        )
    if not seg.ran:
        return AuditFact(
            status=AuditStatus.NOT_AVAILABLE,
            category=AuditCategory.SEGMENTATION,
            label="Segmentation",
            value="Not informative",
            impact=AuditImpact.NEUTRAL,
            detail=seg.reason,
        )
    # Phase 2 fix (heterogeneity-logic audit): this fact is describing
    # "was there a statistically reliable segment-level finding" — that
    # is `has_reliable_segment_effect` (within-segment significance),
    # NOT `has_heterogeneous_effect` (a separate, real interaction-test
    # fact — see SegmentDimensionResult's docstring). Reading
    # `has_heterogeneous_effect` here used to be correct only because
    # it was (buggily) computed as `len(reliable) > 0` — now that the
    # two are properly independent facts, this evidence line must read
    # the one it actually describes. `impact` stays NEUTRAL either way
    # (segmentation is never allowed to move the decision) — nothing in
    # decision logic changes.
    reliable_dims = [d.dimension for d in seg.dimension_results if d.has_reliable_segment_effect]
    if reliable_dims:
        return AuditFact(
            status=AuditStatus.INFO,
            category=AuditCategory.SEGMENTATION,
            label="Segmentation",
            value=f"Reliable differences found in: {', '.join(reliable_dims)}",
            impact=AuditImpact.NEUTRAL,
            detail=seg.reason,
        )
    return AuditFact(
        status=AuditStatus.INFO,
        category=AuditCategory.SEGMENTATION,
        label="Segmentation",
        value="No statistically reliable segment-level differences",
        impact=AuditImpact.NEUTRAL,
        detail=seg.reason,
    )


def _rationale_and_impact(
    facts: ReportFacts,
    report: ExperimentReport,
    primary_stat: StatResult | None,
) -> tuple[list[str], str]:
    """
    Decision-specific reasoning (Phase 7 spec §3). Every sentence is
    built from facts already on `facts`/`report` — nothing here is
    free-form LLM text. Falls back to `report.decision_reason` for any
    decision value not explicitly handled (defensive, keeps this
    function total).
    """
    if report.decision == Decision.INVALID:
        reasons: list[str] = []
        if facts.has_conflicting_variant_duplicates:
            reasons.append(
                "Users were found assigned to more than one variant — the assignment/randomization "
                "pipeline appears broken."
            )
        if not facts.srm_passed:
            reasons.append(
                "A Sample Ratio Mismatch (SRM) was detected — the observed control/variant split "
                "does not match the expected allocation."
            )
        for qc in facts.quality_checks:
            if not qc.passed and qc.critical:
                reasons.append(f"Critical data-quality issue: {qc.label} — {qc.detail}")
        if not reasons:
            reasons.append(report.decision_reason)
        reasons.append(
            "Statistical significance does not matter here because the experiment is invalid — "
            "no ship/no-ship recommendation can be made from this data, regardless of any p-value."
        )
        return reasons, (
            "No decision can be made from this data. Fix the underlying validity issue and rerun "
            "the experiment before evaluating results again."
        )

    if report.decision == Decision.INCONCLUSIVE:
        reasons = []
        if primary_stat is None:
            reasons.append("No hypothesis test was run for this request, so no decision can be made.")
        else:
            reasons.append(
                f"{primary_stat.metric} showed no statistically significant difference between "
                f"variants (p {format_p_value(primary_stat.p_value)})."
            )
            reasons.append(
                "A non-significant result is not proof of \"no effect\" — it means the data did not "
                "provide enough evidence to detect one either way."
            )
            if facts.power_analysis is not None and not facts.power_analysis.is_sufficiently_powered:
                reasons.append(
                    f"The experiment is also underpowered (achieved "
                    f"{facts.power_analysis.achieved_power * 100:.0f}% power) — insufficient "
                    f"statistical power may explain the non-significant result; it does not rule "
                    f"out a real effect."
                )
        reasons.append(
            "Consider collecting more data, extending the experiment, or redesigning it before "
            "drawing a ship/no-ship conclusion."
        )
        return reasons, (
            "No reliable ship/no-ship decision can be made yet — treat this as \"more evidence "
            "needed,\" not as evidence of no effect."
        )

    if report.decision == Decision.NO_GO:
        reasons = [
            "The experiment passed its validity checks, so this result is trustworthy.",
            report.decision_reason,
        ]
        return reasons, "Do not ship this change based on the current evidence."

    if report.decision == Decision.GO_WITH_CAUTION:
        reasons = [report.decision_reason]
        if report.guardrail_status == GuardrailStatus.NOT_AVAILABLE:
            if report.guardrail_request_state == GuardrailRequestState.REQUESTED_NOT_FOUND:
                reasons.append(
                    "The requested guardrail metrics were not available in this dataset — this "
                    "recommendation is based on the primary metric only, which is why the "
                    "recommendation remains cautious."
                )
            elif report.guardrail_request_state in (
                GuardrailRequestState.AVAILABLE,
                GuardrailRequestState.PARTIALLY_AVAILABLE,
            ):
                reasons.append(
                    "The requested guardrail metrics were found but could not be statistically "
                    "evaluated for this experiment — this recommendation is based on the primary "
                    "metric only, which is why the recommendation remains cautious."
                )
            else:
                reasons.append(
                    "No guardrail metrics were evaluated for this dataset — this recommendation is "
                    "based on the primary metric only, which is why the recommendation remains cautious."
                )
        return reasons, (
            "The primary result supports rollout, but proceed cautiously and monitor closely given "
            "the limitation(s) above."
        )

    if report.decision == Decision.GO:
        reasons = [
            report.decision_reason,
            "The experiment is valid, the primary result is statistically and practically "
            "significant, and no critical warnings were found.",
        ]
        return reasons, "The evidence supports shipping this change."

    return [report.decision_reason], report.decision_reason


def build_decision_audit_trail(facts: ReportFacts, report: ExperimentReport) -> DecisionAuditTrail:
    """
    THE single deterministic builder for `DecisionAuditTrail`.
    Pure function of already-computed `facts`/`report` — same inputs
    always produce the same output (Phase 7 spec §11F).
    """
    significant_results = [s for s in report.stats if s.significant]
    primary_stat = select_primary_stat(report.stats, significant_results) or select_primary_stat(report.stats)

    critical_checks = _critical_checks(facts)
    statistical_evidence = _statistical_evidence(primary_stat)
    practical_evidence = _practical_significance_evidence(facts, report)
    power_evidence = _power_evidence(facts)
    guardrail_evidence = _guardrail_evidence(report)
    segmentation_evidence = _segmentation_evidence(facts)

    all_facts = [
        *critical_checks,
        *statistical_evidence,
        *([practical_evidence] if practical_evidence is not None else []),
        *([power_evidence] if power_evidence is not None else []),
        guardrail_evidence,
        segmentation_evidence,
    ]
    supporting_facts = [f for f in all_facts if f.impact == AuditImpact.SUPPORTS_DECISION]
    warnings = [
        f for f in all_facts if f.impact in (AuditImpact.LIMITS_CONFIDENCE, AuditImpact.BLOCKS_DECISION)
    ]

    rationale, decision_impact = _rationale_and_impact(facts, report, primary_stat)

    return DecisionAuditTrail(
        decision=report.decision,
        headline=report.decision.value.replace("_", " "),
        rationale=rationale,
        supporting_facts=supporting_facts,
        warnings=warnings,
        critical_checks=critical_checks,
        statistical_evidence=statistical_evidence,
        practical_significance_evidence=practical_evidence,
        power_evidence=power_evidence,
        guardrail_evidence=guardrail_evidence,
        segmentation_evidence=segmentation_evidence,
        decision_impact=decision_impact,
    )
