"""
Regression tests for decision-aware / blocking-reason-aware Evidence &
Sources filtering (spec section 11, cases A-D). Case E (actual
retriever exception) is covered in
tests/api/test_routes_experiments_execution_steps.py, since it depends
on GraphState / the execution trace, not ReportFacts directly.
"""

from app.graph.report_generator import ReportFacts, TemplateReportGenerator
from app.rag.retriever import DocumentChunk, RetrievedChunk
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.quality import QualityCheck
from app.schemas.statistics import HypothesisTestType, PowerAnalysisResult, StatResult


def _kb_facts(
    *,
    srm_passed=True,
    has_conflicting_variant_duplicates=False,
    quality_checks=None,
    kb_results=None,
):
    """Minimal ReportFacts fixture (no stat results) for KB relevance tests."""
    dataset = DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=294478,
        metric_label="Conversion Rate",
        metric_selection_reason="test",
    )
    return ReportFacts(
        user_prompt="Should we ship this?",
        dataset=dataset,
        quality_checks=quality_checks or [],
        srm_passed=srm_passed,
        stat_results=[],
        test_selections=[],
        power_analysis=None,
        mde_display="2.8% (relative)",
        sample_size_note="294,478 users observed",
        has_conflicting_variant_duplicates=has_conflicting_variant_duplicates,
        kb_results=kb_results,
    )


_MDE_CHUNK = RetrievedChunk(
    chunk=DocumentChunk(
        source="kohavi.md",
        heading="Minimum Detectable Effect (MDE) and Power",
        content="MDE and power determine the smallest effect a test can reliably detect.",
    ),
    score=0.81,
)
_CHI_SQUARE_CHUNK = RetrievedChunk(
    chunk=DocumentChunk(
        source="kohavi.md",
        heading="Binary Metrics: Chi-square vs Fisher",
        content="Use chi-square for large samples and Fisher's exact test for small ones.",
    ),
    score=0.77,
)
_ASSIGNMENT_CHUNK = RetrievedChunk(
    chunk=DocumentChunk(
        source="kohavi.md",
        heading="Treatment Assignment Integrity",
        content="Randomization integrity requires each user be assigned to exactly one variant.",
    ),
    score=0.65,
)
_SRM_CHUNK = RetrievedChunk(
    chunk=DocumentChunk(
        source="kohavi.md",
        heading="Sample Ratio Mismatch (SRM)",
        content="SRM indicates a broken randomization or logging pipeline.",
    ),
    score=0.9,
)
_OUTLIER_CHUNK = RetrievedChunk(
    chunk=DocumentChunk(
        source="kohavi.md",
        heading="Outlier Detection",
        content="Extreme observations can distort the mean and inflate variance.",
    ),
    score=0.7,
)


def test_conflicting_variant_assignment_filters_out_generic_evidence():
    """Case A (spec 11.A): conflicting variant assignment INVALID must
    drop unrelated MDE/Power and chi-square evidence and fall back to
    the honest empty-evidence state, not fabricate a citation."""
    facts = _kb_facts(
        has_conflicting_variant_duplicates=True,
        kb_results=[_MDE_CHUNK, _CHI_SQUARE_CHUNK],
    )

    report = TemplateReportGenerator().generate(facts)

    assert report.experiment_validity == "INVALID"
    assert report.knowledge_base_references == []
    assert report.knowledge_base_blocking_issue == "Conflicting Variant Assignment"
    assert not any(
        "Minimum Detectable Effect" in rec or "Chi-square" in rec
        for rec in report.recommendations
    )


def test_conflicting_variant_assignment_keeps_actually_relevant_evidence():
    facts = _kb_facts(
        has_conflicting_variant_duplicates=True,
        kb_results=[_MDE_CHUNK, _ASSIGNMENT_CHUNK],
    )

    report = TemplateReportGenerator().generate(facts)

    assert [r.heading for r in report.knowledge_base_references] == ["Treatment Assignment Integrity"]


def test_srm_failure_shows_srm_relevant_evidence():
    """Case B (spec 11.B): SRM-relevant evidence may still be shown."""
    facts = _kb_facts(srm_passed=False, kb_results=[_MDE_CHUNK, _SRM_CHUNK])

    report = TemplateReportGenerator().generate(facts)

    assert report.experiment_validity == "INVALID"
    assert [r.heading for r in report.knowledge_base_references] == ["Sample Ratio Mismatch (SRM)"]


def test_srm_failure_with_no_relevant_evidence_uses_system_fallback_or_empty_state():
    """When retrieval found nothing SRM-relevant, an approved built-in
    fallback (if any) must be clearly marked `is_system_fallback=True`,
    never presented as a real retrieved citation."""
    facts = _kb_facts(srm_passed=False, kb_results=[_MDE_CHUNK])

    report = TemplateReportGenerator().generate(facts)

    assert report.knowledge_base_blocking_issue == "Sample Ratio Mismatch (SRM)"
    for ref in report.knowledge_base_references:
        assert ref.is_system_fallback is True


def test_critical_data_quality_failure_filters_out_generic_evidence():
    """Case C (spec 11.C): critical quality failure (Outlier Detection)
    must not surface unrelated MDE/Power evidence."""
    quality_checks = [
        QualityCheck(
            label="Outlier Detection",
            passed=False,
            critical=True,
            detail="1,895 users assigned to more than one variant.",
        )
    ]
    facts = _kb_facts(quality_checks=quality_checks, kb_results=[_MDE_CHUNK, _OUTLIER_CHUNK])

    report = TemplateReportGenerator().generate(facts)

    assert report.knowledge_base_blocking_issue == "Outlier Detection"
    assert [r.heading for r in report.knowledge_base_references] == ["Outlier Detection"]


def test_valid_experiment_keeps_generic_evidence_unfiltered():
    """Case D (spec 11.D): existing behavior for VALID experiments must
    be unchanged — generic methodology documents can still appear."""
    stat = StatResult(
        metric="Conversion Rate",
        test_type=HypothesisTestType.CHI_SQUARE,
        test_name="Chi-square test",
        statistic=123.4,
        selection_reason="Binary metric with sufficient expected cell counts.",
        control="11.87%",
        variant="17.95%",
        delta="+51.2% (rel)",
        p_value=0.000001,
        significant=True,
        ci_lower="+5.82pp",
        ci_upper="+6.33pp",
    )
    power = PowerAnalysisResult(
        minimum_detectable_effect_relative=2.8,
        required_sample_size=10000,
        observed_sample_size=294478,
        achieved_power=0.999999,
        alpha=0.05,
        is_sufficiently_powered=True,
    )
    dataset = DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=294478,
        metric_label="Conversion Rate",
        metric_selection_reason="test",
    )
    facts = ReportFacts(
        user_prompt="Should we ship this?",
        dataset=dataset,
        quality_checks=[],
        srm_passed=True,
        stat_results=[stat],
        test_selections=[],
        power_analysis=power,
        mde_display="2.8% (relative)",
        sample_size_note="294,478 users observed",
        kb_results=[_MDE_CHUNK, _CHI_SQUARE_CHUNK],
    )

    report = TemplateReportGenerator().generate(facts)

    assert report.experiment_validity != "INVALID"
    assert {r.heading for r in report.knowledge_base_references} == {
        "Minimum Detectable Effect (MDE) and Power",
        "Binary Metrics: Chi-square vs Fisher",
    }
    assert report.knowledge_base_blocking_issue is None
