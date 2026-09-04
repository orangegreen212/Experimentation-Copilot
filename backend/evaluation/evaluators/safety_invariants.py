"""
Decision safety invariants — Layer 2 of the evaluation strategy.

These are explicit, deterministic assertions over the ALREADY-COMPUTED
`ExperimentReport`. They are intentionally simpler and more
conservative than `evaluate_case`'s golden-case field comparisons:
each invariant here should hold for EVERY possible input, not just
the specific golden cases, so they are written as pure predicate
functions applied to (facts, report) pairs and can be reused directly
inside `pytest` (see tests/evaluation/test_safety_invariants.py).

Per task spec section 3, target for safety violations is ZERO. An
invariant violation is always reported as UNSAFE, never as a mere
FAIL, to keep it visually distinct in the evaluation report.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.graph.report_generator import ReportFacts
from app.schemas.report import Decision, ExperimentReport


@dataclass
class InvariantViolation:
    invariant: str
    detail: str


def _invariant_srm_fail_never_go(facts: ReportFacts, report: ExperimentReport) -> InvariantViolation | None:
    if not facts.srm_passed and report.decision == Decision.GO:
        return InvariantViolation(
            "srm_fail_never_go",
            "SRM failed (facts.srm_passed=False) but decision=GO.",
        )
    return None


def _invariant_critical_guardrail_never_go(facts: ReportFacts, report: ExperimentReport) -> InvariantViolation | None:
    failing_guardrail = any(
        g.significant and (g.delta or "").strip().startswith("-") for g in facts.guardrail_results
    )
    if failing_guardrail and report.decision == Decision.GO:
        return InvariantViolation(
            "critical_guardrail_never_go",
            "A guardrail metric significantly regressed but decision=GO.",
        )
    return None


def _invariant_invalid_experiment_must_be_invalid(
    facts: ReportFacts, report: ExperimentReport
) -> InvariantViolation | None:
    critical_quality_failure = any((not qc.passed) and qc.critical for qc in facts.quality_checks)
    invalid_input = (not facts.srm_passed) or facts.has_conflicting_variant_duplicates or critical_quality_failure
    if invalid_input and report.decision != Decision.INVALID:
        return InvariantViolation(
            "invalid_experiment_must_be_invalid",
            f"Input had a critical validity failure but decision={report.decision.value}, not INVALID.",
        )
    return None


def _invariant_underpowered_not_confident_positive(
    facts: ReportFacts, report: ExperimentReport
) -> InvariantViolation | None:
    if (
        facts.power_analysis is not None
        and not facts.power_analysis.is_sufficiently_powered
        and report.stats
        and report.stats[0].significant
    ):
        # A significant result under low power is a real statistical
        # signal — this invariant only guards against representing it
        # with unwarranted certainty (HIGH confidence / GO with no
        # caveat), not against a significant finding existing at all.
        if report.recommendation_confidence.value == "HIGH" and report.decision == Decision.GO:
            return InvariantViolation(
                "underpowered_not_confident_positive",
                "Underpowered but reported as HIGH-confidence GO.",
            )
    return None


def _invariant_non_significant_not_claimed_significant(
    facts: ReportFacts, report: ExperimentReport
) -> InvariantViolation | None:
    if report.stats and not report.stats[0].significant:
        if report.decision in (Decision.GO, Decision.NO_GO):
            # GO/NO_GO both imply a significant, actionable primary-metric
            # result in this system's decision model — see
            # determine_decision(): only the `not primary_stat.significant`
            # branch produces INCONCLUSIVE, and every GO/NO_GO branch is
            # downstream of the significance check having already passed.
            return InvariantViolation(
                "non_significant_not_claimed_significant",
                f"Primary metric was not statistically significant but decision={report.decision.value}.",
            )
    return None


INVARIANTS = [
    _invariant_srm_fail_never_go,
    _invariant_critical_guardrail_never_go,
    _invariant_invalid_experiment_must_be_invalid,
    _invariant_underpowered_not_confident_positive,
    _invariant_non_significant_not_claimed_significant,
]


@dataclass
class InvariantEvalReport:
    n_checked: int
    n_violations: int
    violations: list[tuple[str, InvariantViolation]]  # (case_id, violation)

    @property
    def violation_rate(self) -> float:
        return self.n_violations / self.n_checked if self.n_checked else 0.0


def check_invariants(facts: ReportFacts, report: ExperimentReport) -> list[InvariantViolation]:
    """Run every invariant against one (facts, report) pair."""
    violations = []
    for inv in INVARIANTS:
        result = inv(facts, report)
        if result is not None:
            violations.append(result)
    return violations


def evaluate_invariants_over_dataset(generator, cases) -> InvariantEvalReport:
    """Run every invariant over every golden case's (facts, report) pair."""
    all_violations: list[tuple[str, InvariantViolation]] = []
    n_checked = 0
    for case in cases:
        report = generator.generate(case.facts)
        n_checked += len(INVARIANTS)
        for v in check_invariants(case.facts, report):
            all_violations.append((case.id, v))
    return InvariantEvalReport(n_checked=n_checked, n_violations=len(all_violations), violations=all_violations)
