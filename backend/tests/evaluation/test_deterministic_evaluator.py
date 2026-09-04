"""
Tests for evaluation/evaluators/deterministic.py.

Covers task spec section 11 requirements: "correct decision passes",
"wrong decision fails", "underpowered experiment", "non-significant
result", "evaluation aggregation" — all against the REAL
TemplateReportGenerator + the real GOLDEN_CASES dataset, not a fake
"always pass" generator (per spec: "Do not create fake 'always pass'
tests").
"""

from __future__ import annotations

from app.graph.report_generator import ReportGenerator, TemplateReportGenerator
from evaluation.datasets.golden_dataset import GOLDEN_CASES, get_case
from evaluation.evaluators.deterministic import evaluate_case, evaluate_dataset


class _AlwaysGoGenerator:
    """A deliberately WRONG generator used only to prove the evaluator
    can fail a bad system — never used to grade the real application."""

    def generate(self, facts):
        real_report = TemplateReportGenerator().generate(facts)
        from app.schemas.report import Decision

        # Corrupt every decision to GO, keep everything else untouched.
        return real_report.model_copy(update={"decision": Decision.GO})


def test_correct_decision_passes():
    case = get_case("valid_positive_go")
    result = evaluate_case(case, TemplateReportGenerator())
    assert result.status == "PASS"
    assert result.all_checked_fields_passed


def test_wrong_decision_fails():
    case = get_case("srm_failure_invalid")  # expects INVALID
    result = evaluate_case(case, _AlwaysGoGenerator())
    assert result.status == "UNSAFE"  # SRM-fail case forced to GO is a safety violation, not a mere FAIL
    assert not result.all_checked_fields_passed
    assert any(c.field == "decision" and not c.passed for c in result.checks)


def test_underpowered_experiment_case_is_inconclusive():
    case = get_case("underpowered_null_inconclusive")
    result = evaluate_case(case, TemplateReportGenerator())
    assert result.status == "PASS"
    decision_check = next(c for c in result.checks if c.field == "decision")
    assert decision_check.actual == "INCONCLUSIVE"
    power_check = next(c for c in result.checks if c.field == "power_sufficient")
    assert power_check.actual is False


def test_non_significant_result_case_is_inconclusive():
    case = get_case("non_significant_inconclusive")
    result = evaluate_case(case, TemplateReportGenerator())
    assert result.status == "PASS"
    sig_check = next(c for c in result.checks if c.field == "significant")
    assert sig_check.actual is False
    decision_check = next(c for c in result.checks if c.field == "decision")
    assert decision_check.actual == "INCONCLUSIVE"


def test_evaluation_aggregation_on_real_generator():
    report = evaluate_dataset(TemplateReportGenerator(), GOLDEN_CASES)
    assert report.n_cases == len(GOLDEN_CASES)
    assert report.decision_accuracy == 1.0
    assert report.accuracy == 1.0
    assert report.unsafe_go_rate == 0.0
    assert report.guardrail_override_rate == 0.0
    assert report.invalid_experiment_approval_rate == 0.0
    # Confusion matrix must have exactly one entry per (expected, actual) pair
    # observed, and every diagonal cell should be non-empty for a perfect run.
    for expected_class, actual_counts in report.confusion_matrix.items():
        assert expected_class in actual_counts  # perfect run -> expected==actual present


def test_evaluation_aggregation_detects_a_broken_generator():
    report = evaluate_dataset(_AlwaysGoGenerator(), GOLDEN_CASES)
    assert report.decision_accuracy < 1.0
    assert report.unsafe_go_rate > 0.0  # forcing GO on SRM/guardrail-fail cases must show up as unsafe
