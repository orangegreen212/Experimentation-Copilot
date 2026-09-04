"""
Deterministic evaluators — Layer 1/3 of the evaluation strategy
("statistical correctness" + "decision correctness").

These evaluators NEVER call an LLM and NEVER use semantic similarity.
Every comparison here is `expected_field == actual_field` on
structured data (per task spec section 2/4: "Do NOT use semantic
similarity for the actual decision").

`evaluate_case` runs ONE `GoldenCase` through a `ReportGenerator` and
produces a `CaseResult` recording, for every checked dimension,
whether the actual structured output matched the hand-specified
expectation. `evaluate_dataset` aggregates many `CaseResult`s into
accuracy / confusion-matrix / per-class / FPR-FNR / safety-rate
metrics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.graph.report_generator import ReportGenerator
from app.schemas.report import Decision, ExperimentReport
from evaluation.datasets.golden_dataset import GoldenCase, GOLDEN_CASES


# ----------------------------------------------------------------------
# Per-case result
# ----------------------------------------------------------------------

@dataclass
class FieldCheck:
    field: str
    expected: object
    actual: object

    @property
    def passed(self) -> bool:
        return self.expected is None or self.expected == self.actual


@dataclass
class CaseResult:
    case_id: str
    category: str
    status: str  # "PASS" | "FAIL" | "UNSAFE"
    checks: list[FieldCheck]
    report: ExperimentReport
    reason: str = ""

    @property
    def all_checked_fields_passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _effect_direction(report: ExperimentReport) -> str:
    """Derive the observed primary-metric direction from `report.stats`
    without re-parsing free text beyond the existing display-string
    convention (`delta` starts with '+'/'-') — same convention the
    production `report_generator.py` itself already uses internally.
    """
    if not report.stats:
        return "none"
    primary = report.stats[0]
    if not primary.significant:
        # Direction is only meaningful for a statistically significant
        # result in this dataset's `expected.effect_direction` convention
        # — a non-significant point estimate's sign is noise, not
        # evidence of direction (matches determine_decision()'s own
        # treatment: the `not primary_stat.significant` branch never
        # inspects sign at all).
        return "none"
    delta = (primary.delta or "").strip()
    if delta.startswith("-"):
        return "negative"
    if delta.startswith("+"):
        return "positive"
    return "none"


def evaluate_case(case: GoldenCase, generator: ReportGenerator) -> CaseResult:
    report = generator.generate(case.facts)
    exp = case.expected

    checks = [
        FieldCheck("decision", exp.decision, report.decision.value),
        FieldCheck("validity", exp.validity, report.experiment_validity.value),
        FieldCheck(
            "significant",
            exp.significant,
            report.stats[0].significant if report.stats else None,
        ),
        FieldCheck(
            "power_sufficient",
            exp.power_sufficient,
            None,  # power_analysis isn't on ExperimentReport directly; see note below
        ),
        FieldCheck("guardrail_status", exp.guardrail_status, report.guardrail_status.value),
        FieldCheck("effect_direction", exp.effect_direction, _effect_direction(report)),
        FieldCheck("srm_passed", exp.srm_passed, not report.srm_warning),
    ]

    # power_sufficient isn't exposed on the final ExperimentReport (only
    # mde/sample_size_note display strings are) — check it against the
    # INPUT facts instead, since TemplateReportGenerator's contract is
    # to consume (never recompute) power_analysis. This still validates
    # "did the system correctly account for power in its decision" via
    # the `decision` field itself (see e.g. underpowered_null_inconclusive).
    if exp.power_sufficient is not None and case.facts.power_analysis is not None:
        checks[3] = FieldCheck(
            "power_sufficient", exp.power_sufficient, case.facts.power_analysis.is_sufficiently_powered
        )
    else:
        checks[3] = FieldCheck("power_sufficient", None, None)

    failed = [c for c in checks if not c.passed]

    # UNSAFE: a safety-relevant field (decision, validity) is wrong AND
    # the direction of the error is dangerous (see safety_invariants.py
    # for the exhaustive invariant list — this is the summary label).
    unsafe = False
    if any(c.field == "decision" for c in failed):
        if exp.decision == "INVALID" and report.decision != Decision.INVALID:
            unsafe = True
        if exp.decision in ("NO_GO", "INCONCLUSIVE") and report.decision == Decision.GO:
            unsafe = True

    status = "UNSAFE" if unsafe else ("PASS" if not failed else "FAIL")
    reason = "" if not failed else "; ".join(f"{c.field}: expected={c.expected!r} got={c.actual!r}" for c in failed)

    return CaseResult(case_id=case.id, category=case.category, status=status, checks=checks, report=report, reason=reason)


# ----------------------------------------------------------------------
# Aggregate metrics
# ----------------------------------------------------------------------

@dataclass
class DeterministicEvalReport:
    n_cases: int
    accuracy: float  # fraction of cases with ALL checked fields correct
    decision_accuracy: float  # fraction with `decision` correct specifically
    confusion_matrix: dict[str, dict[str, int]]  # expected -> actual -> count (Decision only)
    per_decision_class_accuracy: dict[str, float]
    false_positive_rate: float  # expected NOT GO, actual GO  (unsafe "ship" recommendation)
    false_negative_rate: float  # expected GO, actual NOT GO  (missed a real win — not unsafe, but a quality issue)
    unsafe_go_rate: float  # expected INVALID/NO_GO/INCONCLUSIVE, actual GO
    guardrail_override_rate: float  # a failing guardrail present in facts, actual decision GO or GO_WITH_CAUTION anyway
    invalid_experiment_approval_rate: float  # expected INVALID, actual anything other than INVALID
    case_results: list[CaseResult] = field(default_factory=list)


def evaluate_dataset(
    generator: ReportGenerator, cases: list[GoldenCase] = GOLDEN_CASES
) -> DeterministicEvalReport:
    results = [evaluate_case(c, generator) for c in cases]

    n = len(results)
    all_correct = sum(1 for r in results if r.all_checked_fields_passed)
    decision_correct = sum(1 for r in results if all(c.passed for c in r.checks if c.field == "decision"))

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()

    unsafe_go = 0
    guardrail_override = 0
    invalid_approved = 0
    fp = 0  # expected non-GO, actual GO
    fn = 0  # expected GO, actual non-GO
    n_expected_non_go = 0
    n_expected_go = 0

    for case, r in zip(cases, results):
        exp_decision = case.expected.decision
        actual_decision = next(c.actual for c in r.checks if c.field == "decision")
        confusion[exp_decision][actual_decision] += 1
        per_class_total[exp_decision] += 1
        if exp_decision == actual_decision:
            per_class_correct[exp_decision] += 1

        if exp_decision in ("INVALID", "NO_GO", "INCONCLUSIVE") and actual_decision == "GO":
            unsafe_go += 1

        if exp_decision != "GO":
            n_expected_non_go += 1
            if actual_decision == "GO":
                fp += 1
        else:
            n_expected_go += 1
            if actual_decision != "GO":
                fn += 1

        has_failing_guardrail = any(
            g.significant and (g.delta or "").strip().startswith("-") for g in case.facts.guardrail_results
        )
        if has_failing_guardrail and actual_decision in ("GO", "GO_WITH_CAUTION"):
            guardrail_override += 1

        if exp_decision == "INVALID" and actual_decision != "INVALID":
            invalid_approved += 1

    per_class_accuracy = {
        cls: (per_class_correct[cls] / total if total else 0.0) for cls, total in per_class_total.items()
    }

    return DeterministicEvalReport(
        n_cases=n,
        accuracy=all_correct / n if n else 0.0,
        decision_accuracy=decision_correct / n if n else 0.0,
        confusion_matrix={k: dict(v) for k, v in confusion.items()},
        per_decision_class_accuracy=per_class_accuracy,
        false_positive_rate=(fp / n_expected_non_go) if n_expected_non_go else 0.0,
        false_negative_rate=(fn / n_expected_go) if n_expected_go else 0.0,
        unsafe_go_rate=unsafe_go / n if n else 0.0,
        guardrail_override_rate=guardrail_override / n if n else 0.0,
        invalid_experiment_approval_rate=invalid_approved / n if n else 0.0,
        case_results=results,
    )
