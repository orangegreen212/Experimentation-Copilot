"""
Evaluation runner — orchestrates every layer and produces both a
machine-readable JSON result and a human-readable summary (task spec
section 10).

VERSION-AGNOSTIC BY DESIGN (task spec section 9): `evaluate_version()`
takes a `ReportGenerator` instance as a parameter, not a hardcoded
import — the SAME `GOLDEN_CASES` dataset and the SAME evaluators run
against whichever generator is passed in. Today that's always
`TemplateReportGenerator` from this Phase 3/4 snapshot; a future
Phase 5/6 snapshot evaluates by importing ITS OWN `ReportGenerator`
implementation (which must satisfy the same `ReportGenerator` Protocol
— see `app/graph/report_generator.py`) and calling
`evaluate_version(name="phase5", generator=Phase5ReportGenerator())`
with no change to this file. `compare_versions()` then diffs two
`VersionEvalResult`s metric-by-metric.

Every PASS/FAIL/UNSAFE/NOT_APPLICABLE distinction required by the
task spec is preserved end-to-end into the JSON output — see
`_case_status_payload`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.graph.report_generator import ReportGenerator, TemplateReportGenerator
from evaluation.datasets.golden_dataset import GOLDEN_CASES
from evaluation.evaluators.agent_evaluator import EXPECTED_TRAJECTORIES, evaluate_agent
from evaluation.evaluators.deterministic import evaluate_dataset
from evaluation.evaluators.rag_evaluator import evaluate_rag
from evaluation.evaluators.safety_invariants import evaluate_invariants_over_dataset

_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "value") and not isinstance(obj, (str, int, float, bool)):
        return obj.value  # enums
    return obj


@dataclass
class VersionEvalResult:
    version_name: str
    timestamp: str
    n_golden_cases: int
    decision_accuracy: float
    overall_accuracy: float
    unsafe_go_rate: float
    guardrail_override_rate: float
    invalid_experiment_approval_rate: float
    false_positive_rate: float
    false_negative_rate: float
    safety_invariant_violations: int
    safety_invariant_checks: int
    rag_avg_precision_at_k: float
    rag_avg_recall_at_k: float
    rag_avg_answer_faithfulness: float
    case_statuses: dict[str, str]  # case_id -> PASS | FAIL | UNSAFE
    failures: list[str]


def _case_status_payload(det_report, invariant_report) -> tuple[dict[str, str], list[str]]:
    statuses = {r.case_id: r.status for r in det_report.case_results}
    failures = [f"[{r.case_id}] {r.reason}" for r in det_report.case_results if r.status != "PASS"]
    for case_id, violation in invariant_report.violations:
        statuses[case_id] = "UNSAFE"
        failures.append(f"[{case_id}] SAFETY INVARIANT VIOLATION: {violation.invariant} — {violation.detail}")
    return statuses, failures


def evaluate_version(
    version_name: str,
    generator: ReportGenerator,
    cases=GOLDEN_CASES,
    include_rag: bool = True,
) -> VersionEvalResult:
    det_report = evaluate_dataset(generator, cases)
    invariant_report = evaluate_invariants_over_dataset(generator, cases)
    statuses, failures = _case_status_payload(det_report, invariant_report)

    if include_rag:
        rag_report = evaluate_rag()
        rag_p, rag_r, rag_f = (
            rag_report.avg_precision_at_k,
            rag_report.avg_recall_at_k,
            rag_report.avg_answer_faithfulness,
        )
    else:
        rag_p = rag_r = rag_f = 0.0  # NOT_APPLICABLE — see run_full_evaluation for the explicit marker

    return VersionEvalResult(
        version_name=version_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        n_golden_cases=det_report.n_cases,
        decision_accuracy=det_report.decision_accuracy,
        overall_accuracy=det_report.accuracy,
        unsafe_go_rate=det_report.unsafe_go_rate,
        guardrail_override_rate=det_report.guardrail_override_rate,
        invalid_experiment_approval_rate=det_report.invalid_experiment_approval_rate,
        false_positive_rate=det_report.false_positive_rate,
        false_negative_rate=det_report.false_negative_rate,
        safety_invariant_violations=invariant_report.n_violations,
        safety_invariant_checks=invariant_report.n_checked,
        rag_avg_precision_at_k=rag_p,
        rag_avg_recall_at_k=rag_r,
        rag_avg_answer_faithfulness=rag_f,
        case_statuses=statuses,
        failures=failures,
    )


@dataclass
class VersionComparison:
    baseline: str
    candidate: str
    deltas: dict[str, float]
    regressions: list[str]  # metrics that got WORSE on the candidate
    improvements: list[str]


_HIGHER_IS_BETTER = {
    "decision_accuracy", "overall_accuracy", "rag_avg_precision_at_k",
    "rag_avg_recall_at_k", "rag_avg_answer_faithfulness",
}
_LOWER_IS_BETTER = {
    "unsafe_go_rate", "guardrail_override_rate", "invalid_experiment_approval_rate",
    "false_positive_rate", "false_negative_rate", "safety_invariant_violations",
}


def compare_versions(baseline: VersionEvalResult, candidate: VersionEvalResult) -> VersionComparison:
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    improvements: list[str] = []

    for metric in _HIGHER_IS_BETTER | _LOWER_IS_BETTER:
        base_val = getattr(baseline, metric)
        cand_val = getattr(candidate, metric)
        delta = cand_val - base_val
        deltas[metric] = delta
        if delta == 0:
            continue
        worse = (metric in _HIGHER_IS_BETTER and delta < 0) or (metric in _LOWER_IS_BETTER and delta > 0)
        (regressions if worse else improvements).append(metric)

    return VersionComparison(
        baseline=baseline.version_name, candidate=candidate.version_name,
        deltas=deltas, regressions=regressions, improvements=improvements,
    )


def _print_summary(result: VersionEvalResult) -> None:
    print(f"=== Evaluation: {result.version_name} ({result.timestamp}) ===")
    print(f"Golden cases:                     {result.n_golden_cases}")
    print(f"Decision accuracy:                {result.decision_accuracy * 100:.1f}%")
    print(f"Overall (all-fields) accuracy:    {result.overall_accuracy * 100:.1f}%")
    print(f"Unsafe GO rate:                   {result.unsafe_go_rate * 100:.1f}%  (target: 0%)")
    print(f"Guardrail override rate:          {result.guardrail_override_rate * 100:.1f}%  (target: 0%)")
    print(f"Invalid-experiment approval rate: {result.invalid_experiment_approval_rate * 100:.1f}%  (target: 0%)")
    print(f"False positive rate:              {result.false_positive_rate * 100:.1f}%")
    print(f"False negative rate:              {result.false_negative_rate * 100:.1f}%")
    print(f"Safety invariant violations:      {result.safety_invariant_violations}/{result.safety_invariant_checks}")
    print(f"RAG precision@k:                  {result.rag_avg_precision_at_k * 100:.1f}%")
    print(f"RAG recall@k:                     {result.rag_avg_recall_at_k * 100:.1f}%")
    print(f"RAG answer faithfulness:          {result.rag_avg_answer_faithfulness * 100:.1f}%")
    if result.failures:
        print("\nFailures / violations:")
        for f in result.failures:
            print(f"  - {f}")
    else:
        print("\nNo failures.")


def run_full_evaluation(
    version_name: str = "phase4_current",
    generator: ReportGenerator | None = None,
    write_json: bool = True,
) -> VersionEvalResult:
    generator = generator or TemplateReportGenerator()
    result = evaluate_version(version_name, generator)
    _print_summary(result)

    # Agent/trajectory evaluation is NOT_APPLICABLE unless the caller
    # supplies real ExecutionStep traces (this runner has no live
    # dataset/graph invocation — see agent_evaluator.py's module
    # docstring) — reported explicitly rather than silently skipped.
    print(
        f"\nAgent/trajectory evaluation: NOT_APPLICABLE in this standalone run "
        f"(requires live ExecutionStep traces — see evaluation/evaluators/agent_evaluator.py "
        f"and tests/evaluation/test_agent_evaluator.py for the synthetic-trace regression guard "
        f"covering all {len(EXPECTED_TRAJECTORIES)} route shapes)."
    )

    if write_json:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _REPORTS_DIR / f"{version_name}.json"
        out_path.write_text(json.dumps(_to_jsonable(result), indent=2, default=str))
        print(f"\nJSON result written to {out_path}")

    return result


def main() -> None:
    run_full_evaluation()


if __name__ == "__main__":
    main()
