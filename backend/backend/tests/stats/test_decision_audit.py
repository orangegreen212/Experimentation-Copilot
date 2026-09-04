"""
Phase 7 — Decision Audit Trail tests.

Builds real `ReportFacts` -> `TemplateReportGenerator.generate()` ->
`build_decision_audit_trail()`, so every case exercises the SAME
`determine_decision()` the rest of the app uses, rather than
hand-constructing an `ExperimentReport` that might drift from what the
backend actually produces. No test here recomputes or contradicts a
statistical result — every assertion is about the audit trail's
narration/structure, not the underlying numbers.
"""

from app.graph.report_generator import ReportFacts, TemplateReportGenerator
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.quality import QualityCheck
from app.schemas.report import Decision, GuardrailStatus
from app.schemas.segmentation import (
    SegmentDimensionResult,
    SegmentationResult,
    SegmentEffect,
    SegmentSampleSizeStatus,
)
from app.schemas.statistics import HypothesisTestType, PowerAnalysisResult, StatResult
from app.stats.decision_audit import build_decision_audit_trail


def _dataset(**overrides):
    defaults = dict(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=294478,
        metric_label="Conversion Rate",
        metric_selection_reason="Selected by the deterministic outcome-column priority because no specific metric was requested.",
    )
    defaults.update(overrides)
    return DatasetInfo(**defaults)


def _stat(
    control="11.87%",
    variant="17.95%",
    delta="+51.2% (rel)",
    p_value=0.000001,
    significant=True,
    ci_lower="+5.82pp",
    ci_upper="+6.33pp",
):
    return StatResult(
        metric="Conversion Rate",
        test_type=HypothesisTestType.CHI_SQUARE,
        test_name="Chi-square test",
        statistic=123.4,
        selection_reason="Binary metric with sufficient expected cell counts.",
        control=control,
        variant=variant,
        delta=delta,
        p_value=p_value,
        significant=significant,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def _power(mde=2.8, is_sufficiently_powered=True, achieved_power=0.999999):
    return PowerAnalysisResult(
        minimum_detectable_effect_relative=mde,
        required_sample_size=10000,
        observed_sample_size=294478,
        achieved_power=achieved_power,
        alpha=0.05,
        is_sufficiently_powered=is_sufficiently_powered,
    )


def _facts(**overrides):
    defaults = dict(
        user_prompt="Analyze this experiment end-to-end",
        dataset=_dataset(),
        quality_checks=[],
        srm_passed=True,
        stat_results=[_stat()],
        test_selections=[],
        power_analysis=_power(),
        mde_display="2.8% (relative)",
        sample_size_note="294,478 users observed — achieved power >99.9% at α=0.05",
    )
    defaults.update(overrides)
    return ReportFacts(**defaults)


def _build(facts):
    report = TemplateReportGenerator().generate(facts)
    audit = build_decision_audit_trail(facts, report)
    return report, audit


# --- A. INVALID -------------------------------------------------------------


def test_invalid_conflicting_variant_assignment_explains_why():
    facts = _facts(has_conflicting_variant_duplicates=True)
    report, audit = _build(facts)

    assert report.decision == Decision.INVALID
    assert audit.decision == Decision.INVALID
    assert any("more than one variant" in r for r in audit.rationale)
    assert any("does not matter" in r for r in audit.rationale)
    # The specific validity blocker must show up as a blocking critical check.
    dup_fact = next(f for f in audit.critical_checks if "conflicting variant" in f.label.lower())
    assert dup_fact.status == "fail"
    assert dup_fact.impact == "blocks_decision"


def test_invalid_srm_failure_explains_why():
    facts = _facts(srm_passed=False)
    report, audit = _build(facts)

    assert report.decision == Decision.INVALID
    assert any("Sample Ratio Mismatch" in r for r in audit.rationale)
    srm_fact = next(f for f in audit.critical_checks if "SRM" in f.label)
    assert srm_fact.status == "fail"
    assert srm_fact.impact == "blocks_decision"


def test_invalid_never_says_significance_supports_shipping():
    """Even with a wildly significant primary stat, INVALID must not imply a ship signal."""
    facts = _facts(srm_passed=False, stat_results=[_stat(p_value=1e-10)])
    _, audit = _build(facts)

    assert audit.decision == Decision.INVALID
    assert not any(f.impact == "supports_decision" and f.category == "guardrails" for f in audit.supporting_facts)
    assert "does not matter" in audit.rationale[-1]


# --- B. INCONCLUSIVE ----------------------------------------------------------


def test_inconclusive_non_significant_does_not_claim_no_effect():
    facts = _facts(stat_results=[_stat(significant=False, p_value=0.51, delta="+1.1% (rel)")])
    report, audit = _build(facts)

    assert report.decision == Decision.INCONCLUSIVE
    full_text = " ".join(audit.rationale) + audit.decision_impact
    assert "no effect" not in full_text.lower() or "not proof" in full_text.lower()
    assert "not proof" in " ".join(audit.rationale).lower()


def test_inconclusive_underpowered_mentions_power_as_a_reason():
    facts = _facts(
        stat_results=[_stat(significant=False, p_value=0.51, delta="+1.1% (rel)")],
        power_analysis=_power(is_sufficiently_powered=False, achieved_power=0.10),
    )
    report, audit = _build(facts)

    assert report.decision == Decision.INCONCLUSIVE
    assert any("underpowered" in r.lower() for r in audit.rationale)
    assert audit.power_evidence is not None
    assert audit.power_evidence.status == "warning"


# --- C. GO_WITH_CAUTION ---------------------------------------------------------


def test_go_with_caution_mentions_guardrail_limitation():
    facts = _facts()  # significant + practical, guardrails NOT_AVAILABLE by default
    report, audit = _build(facts)

    assert report.decision == Decision.GO_WITH_CAUTION
    assert report.guardrail_status == GuardrailStatus.NOT_AVAILABLE
    assert any("guardrail" in r.lower() for r in audit.rationale)
    assert audit.guardrail_evidence.value.upper() == "NOT AVAILABLE"
    assert audit.guardrail_evidence.detail is not None


# --- D. GO -----------------------------------------------------------------------


def test_go_decision_evidence_is_all_present():
    facts = _facts(guardrail_results=[])
    facts.guardrail_results = [
        StatResult(
            metric="Bounce Rate",
            test_type=HypothesisTestType.CHI_SQUARE,
            test_name="Chi-square test",
            statistic=1.0,
            selection_reason="guardrail",
            control="20.0%",
            variant="19.5%",
            delta="+2.5% (rel)",
            p_value=0.4,
            significant=False,
            ci_lower="-1.0pp",
            ci_upper="+0.5pp",
        )
    ]
    report, audit = _build(facts)

    assert report.decision == Decision.GO
    assert audit.decision == Decision.GO
    assert audit.guardrail_evidence.status == "pass"
    assert any(f.category == "statistical_significance" and f.impact == "supports_decision" for f in audit.supporting_facts)
    assert any(f.category == "practical_significance" for f in [audit.practical_significance_evidence])
    assert audit.practical_significance_evidence.impact == "supports_decision"


# --- E. Segmentation evidence ------------------------------------------------------


def test_segmentation_reliable_findings_are_surfaced_as_neutral_evidence():
    seg = SegmentationResult(
        ran=True,
        reason="Segmentation found reliable differences in device_type.",
        usable_dimensions=["device_type"],
        skipped_dimensions=[],
        dimension_results=[
            SegmentDimensionResult(
                dimension="device_type",
                segment_effects=[
                    SegmentEffect(
                        segment_value="mobile",
                        control_n=1000,
                        variant_n=1000,
                        sample_size_status=SegmentSampleSizeStatus.SUFFICIENT,
                        stat_result=_stat(),
                    )
                ],
                multiple_testing_method="Holm-Bonferroni across 1 segment test",
                reliable_segment_values=["mobile"],
                has_reliable_segment_effect=True,
                has_heterogeneous_effect=False,
            )
        ],
        min_segment_size=30,
    )
    facts = _facts(segmentation_result=seg)
    _, audit = _build(facts)

    assert audit.segmentation_evidence.impact == "neutral"
    assert "device_type" in audit.segmentation_evidence.value


def test_segmentation_no_usable_dimensions_is_explained():
    seg = SegmentationResult(
        ran=False,
        reason="Segmentation was not informative because no usable low-cardinality dimensions were available.",
        usable_dimensions=[],
        skipped_dimensions=[],
        dimension_results=[],
        min_segment_size=30,
    )
    facts = _facts(segmentation_result=seg)
    _, audit = _build(facts)

    assert audit.segmentation_evidence.status == "not_available"
    assert "no usable" in audit.segmentation_evidence.detail.lower()


def test_invalid_experiment_without_segmentation_says_so():
    facts = _facts(has_conflicting_variant_duplicates=True, segmentation_result=None)
    _, audit = _build(facts)

    assert audit.segmentation_evidence.status == "not_available"
    assert audit.segmentation_evidence.value == "Not run"


# --- F. Deterministic behavior -----------------------------------------------------


def test_same_facts_produce_the_same_audit_trail():
    facts_a = _facts()
    facts_b = _facts()

    _, audit_a = _build(facts_a)
    _, audit_b = _build(facts_b)

    assert audit_a.model_dump() == audit_b.model_dump()


def test_audit_trail_never_overrides_or_recomputes_the_decision():
    """The audit trail's `decision` must always mirror the report's canonical decision."""
    for facts in (
        _facts(has_conflicting_variant_duplicates=True),
        _facts(stat_results=[_stat(significant=False, p_value=0.9)]),
        _facts(),
    ):
        report, audit = _build(facts)
        assert audit.decision == report.decision
        assert audit.headline == report.decision.value.replace("_", " ")
