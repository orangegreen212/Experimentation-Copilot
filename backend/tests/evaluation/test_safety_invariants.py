"""
Tests for evaluation/evaluators/safety_invariants.py.

Covers task spec section 11: "SRM safety invariant", "guardrail safety
invariant", "invalid experiment safety invariant". Each test proves
BOTH that the invariant holds on the real system AND that the checker
itself can detect a violation (constructed via `model_copy`, not a
fake always-pass double) — otherwise a checker that always returns
"no violations" would pass silently.
"""

from __future__ import annotations

from app.graph.report_generator import TemplateReportGenerator
from app.schemas.report import Decision
from evaluation.datasets.golden_dataset import GOLDEN_CASES, get_case
from evaluation.evaluators.safety_invariants import (
    check_invariants,
    evaluate_invariants_over_dataset,
)


def test_srm_safety_invariant_holds_on_real_system():
    case = get_case("srm_failure_invalid")
    report = TemplateReportGenerator().generate(case.facts)
    violations = check_invariants(case.facts, report)
    assert violations == []


def test_srm_safety_invariant_detects_violation():
    case = get_case("srm_failure_invalid")
    report = TemplateReportGenerator().generate(case.facts)
    corrupted = report.model_copy(update={"decision": Decision.GO})
    violations = check_invariants(case.facts, corrupted)
    names = [v.invariant for v in violations]
    assert "srm_fail_never_go" in names
    assert "invalid_experiment_must_be_invalid" in names


def test_guardrail_safety_invariant_holds_on_real_system():
    case = get_case("critical_guardrail_failure_no_go")
    report = TemplateReportGenerator().generate(case.facts)
    violations = check_invariants(case.facts, report)
    assert violations == []


def test_guardrail_safety_invariant_detects_violation():
    case = get_case("critical_guardrail_failure_no_go")
    report = TemplateReportGenerator().generate(case.facts)
    corrupted = report.model_copy(update={"decision": Decision.GO})
    violations = check_invariants(case.facts, corrupted)
    names = [v.invariant for v in violations]
    assert "critical_guardrail_never_go" in names


def test_invalid_experiment_safety_invariant_holds_on_real_system():
    case = get_case("conflicting_variant_duplicates_invalid")
    report = TemplateReportGenerator().generate(case.facts)
    violations = check_invariants(case.facts, report)
    assert violations == []


def test_invalid_experiment_safety_invariant_detects_violation():
    case = get_case("invalid_malformed_data")
    report = TemplateReportGenerator().generate(case.facts)
    corrupted = report.model_copy(update={"decision": Decision.INCONCLUSIVE})
    violations = check_invariants(case.facts, corrupted)
    names = [v.invariant for v in violations]
    assert "invalid_experiment_must_be_invalid" in names


def test_zero_violations_across_full_golden_dataset():
    """The real system, over every golden case, must have ZERO safety
    invariant violations — this is the target asserted by the task spec
    ("target for safety violations should be zero")."""
    report = evaluate_invariants_over_dataset(TemplateReportGenerator(), GOLDEN_CASES)
    assert report.n_violations == 0
    assert report.violation_rate == 0.0
