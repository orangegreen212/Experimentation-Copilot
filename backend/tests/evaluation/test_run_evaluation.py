"""
Tests for evaluation/runners/run_evaluation.py.

Covers task spec section 11: "regression comparison" — proves
`compare_versions()` correctly flags a regression when a "candidate"
generator is worse than a "baseline" generator, and correctly reports
no regressions when comparing the real system against itself (the
Phase-4-vs-Phase-4 self-comparison this exact runner will do until a
Phase 5 generator exists to compare against).
"""

from __future__ import annotations

from app.graph.report_generator import TemplateReportGenerator
from app.schemas.report import Decision
from evaluation.datasets.golden_dataset import GOLDEN_CASES
from evaluation.runners.run_evaluation import compare_versions, evaluate_version


class _RegressedGenerator:
    """Simulates a WORSE future version: silently drops guardrail
    enforcement (a realistic regression class this framework exists to
    catch), only for cases that included a failing guardrail."""

    def generate(self, facts):
        report = TemplateReportGenerator().generate(facts)
        has_failing_guardrail = any(
            g.significant and (g.delta or "").strip().startswith("-") for g in facts.guardrail_results
        )
        if has_failing_guardrail and report.decision.value == "NO_GO":
            return report.model_copy(update={"decision": Decision.GO})
        return report


def test_self_comparison_shows_no_regressions():
    baseline = evaluate_version("baseline", TemplateReportGenerator(), cases=GOLDEN_CASES, include_rag=False)
    candidate = evaluate_version("candidate_identical", TemplateReportGenerator(), cases=GOLDEN_CASES, include_rag=False)
    comparison = compare_versions(baseline, candidate)
    assert comparison.regressions == []


def test_comparison_detects_a_real_regression():
    baseline = evaluate_version("baseline", TemplateReportGenerator(), cases=GOLDEN_CASES, include_rag=False)
    candidate = evaluate_version("regressed_candidate", _RegressedGenerator(), cases=GOLDEN_CASES, include_rag=False)
    comparison = compare_versions(baseline, candidate)
    assert "unsafe_go_rate" in comparison.regressions or "guardrail_override_rate" in comparison.regressions
    assert comparison.deltas["unsafe_go_rate"] > 0 or comparison.deltas["guardrail_override_rate"] > 0


def test_evaluate_version_produces_all_required_fields():
    result = evaluate_version("phase4_current", TemplateReportGenerator(), cases=GOLDEN_CASES, include_rag=False)
    assert result.n_golden_cases == len(GOLDEN_CASES)
    assert result.decision_accuracy == 1.0
    assert result.safety_invariant_violations == 0
    assert all(status in ("PASS", "FAIL", "UNSAFE") for status in result.case_statuses.values())
