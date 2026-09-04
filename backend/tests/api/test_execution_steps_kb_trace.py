"""
Regression test for spec section 9/10/11: the execution trace's
Knowledge Base Retrieval step must use the SAME final, decision-
filtered reference list the report shows in Evidence & Sources — never
the raw, unfiltered `kb_results` retrieved before the INVALID-specific
relevance gate ran. Before this fix, `_build_execution_steps` built its
"Retrieved N reference(s)..." line straight from `state["kb_results"]`,
so an INVALID report could correctly show "No sufficiently relevant
evidence found" while the trace still claimed "Retrieved 2
reference(s), top match: Minimum Detectable Effect (MDE) and Power" —
exactly the bug reported for the 1,895-conflicting-variant dataset.
"""

from app.api.routes_experiments import _build_execution_steps
from app.graph.report_generator import ReportFacts, TemplateReportGenerator
from app.rag.retriever import DocumentChunk, RetrievedChunk
from app.schemas.dataset import DatasetInfo, DatasetType

_MDE_CHUNK = RetrievedChunk(
    chunk=DocumentChunk(
        source="kohavi.md",
        heading="Minimum Detectable Effect (MDE) and Power",
        content="MDE and power determine the smallest effect a test can reliably detect.",
    ),
    score=0.81,
)
_ASSIGNMENT_CHUNK = RetrievedChunk(
    chunk=DocumentChunk(
        source="kohavi.md",
        heading="Treatment Assignment Integrity",
        content="Randomization integrity requires each user be assigned to exactly one variant.",
    ),
    score=0.65,
)


def _dataset():
    return DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=1895,
        metric_label="Conversion Rate",
        metric_selection_reason="test",
    )


def _state(kb_results):
    dataset = _dataset()
    facts = ReportFacts(
        user_prompt="Should we ship this?",
        dataset=dataset,
        quality_checks=[],
        srm_passed=True,
        stat_results=[],
        test_selections=[],
        power_analysis=None,
        mde_display="2.8% (relative)",
        sample_size_note="1,895 users observed",
        has_conflicting_variant_duplicates=True,
        kb_results=kb_results,
    )
    report = TemplateReportGenerator().generate(facts)
    return {
        "dataset": dataset,
        "plan": {"intent_label": "Full Experiment Review", "llm_status": "not_used"},
        "report": report,
        "srm_result": None,  # exits right after the KB step, same as a pure conceptual run
        "kb_results": kb_results,
        "kb_error": None,
    }, report


def test_execution_trace_reflects_filtered_evidence_not_raw_retrieval():
    """Retrieval found only the unrelated MDE chunk; the decision-aware
    gate must filter it out, and the trace must say so — never
    'Retrieved 1 reference(s)'."""
    state, report = _state(kb_results=[_MDE_CHUNK])

    assert report.knowledge_base_references == []  # sanity: report itself is already correct

    steps = {s.id: s for s in _build_execution_steps(state)}
    kb_step = steps["knowledge_base"]

    assert "Retrieved" not in kb_step.detail
    assert "No sufficiently relevant evidence" in kb_step.detail
    assert "Conflicting Variant Assignment" in kb_step.detail
    assert kb_step.status.value == "SUCCESS"


def test_execution_trace_reflects_filtered_evidence_when_some_survives():
    """Retrieval found one relevant and one irrelevant chunk; the trace
    must count and name only the surviving, relevant one."""
    state, report = _state(kb_results=[_MDE_CHUNK, _ASSIGNMENT_CHUNK])

    assert [r.heading for r in report.knowledge_base_references] == ["Treatment Assignment Integrity"]

    steps = {s.id: s for s in _build_execution_steps(state)}
    kb_step = steps["knowledge_base"]

    assert "Retrieved 1 reference(s)" in kb_step.detail
    assert "Treatment Assignment Integrity" in kb_step.detail
    assert "Minimum Detectable Effect" not in kb_step.detail


def test_execution_trace_still_reports_actual_retriever_failure():
    """Case E (spec 11.E): an actual retriever exception must still show
    as a failure, never confused with the filtered-to-empty case above."""
    state, _ = _state(kb_results=[])
    state["kb_error"] = "index not built"

    steps = {s.id: s for s in _build_execution_steps(state)}
    kb_step = steps["knowledge_base"]

    assert "Knowledge base retrieval failed" in kb_step.detail
    assert "index not built" in kb_step.detail
    assert kb_step.status.value == "WARNING"
