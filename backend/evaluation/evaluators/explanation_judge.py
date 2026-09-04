"""
LLM explanation evaluation — Layer 4 of the evaluation strategy.

SCOPE BOUNDARY (per task spec section 5, load-bearing):

    LLM-as-a-Judge MUST NOT determine statistical correctness.
    Statistical correctness comes from deterministic ground truth.

This module ONLY grades the natural-language `executive_summary` /
`decision_reason` text against the STRUCTURED facts that already
determined the decision (see `evaluators/deterministic.py` and
`evaluators/safety_invariants.py` for where correctness itself is
decided). It never re-derives or overrides `decision`,
`experiment_validity`, or any other structured field.

Five rubric dimensions (G-Eval/rubric-based, scored 1-5), matching the
task spec's Layer 4 list exactly:
  A. Groundedness       — does the explanation agree with the facts?
  B. Completeness       — does it mention the evidence needed to
                           justify the decision (decision, validity,
                           significance, guardrail status)?
  C. Unsupported claims  — does it introduce any fact not present in
                           the structured evidence? (lower is better;
                           reported as a count, not a 1-5 score)
  D. Decision consistency — does the explanation's stated conclusion
                           agree with the ACTUAL structured `decision`?
  E. Clarity             — understandable to a product/data
                           decision-maker? (1-5 rubric)

DESIGN — testable without a live network call:
`judge_explanation()` takes a `judge_fn` callable (defaults to
`_default_judge_fn`, which wraps `app.llm.client.get_llm()`) so tests
can inject a fake judge exactly the way `tests/llm/test_llm_report_
generator.py` already mocks `get_llm()` — no new mocking pattern is
introduced. `unsupported_claims_heuristic()` is a fully deterministic,
non-LLM pre-check that flags any number in the explanation not
present in the structured facts; it is a cheap first line of defense
run BEFORE the LLM judge, not a replacement for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

from pydantic import BaseModel, Field

from app.schemas.report import ExperimentReport


class _JudgeScores(BaseModel):
    groundedness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    decision_consistency: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    unsupported_claims: list[str] = Field(default_factory=list)
    rationale: str = ""


class JudgeFn(Protocol):
    def __call__(self, system_prompt: str, user_prompt: str) -> _JudgeScores: ...


def _structured_facts_summary(report: ExperimentReport) -> str:
    """The ONLY facts the judge is allowed to check the explanation
    against — deliberately plain text built straight from structured
    fields, never from the explanation itself (that would be circular).
    """
    lines = [
        f"decision: {report.decision.value}",
        f"experiment_validity: {report.experiment_validity.value}",
        f"guardrail_status: {report.guardrail_status.value}",
        f"practical_significance: {report.practical_significance}",
        f"srm_warning: {report.srm_warning}",
    ]
    for s in report.stats:
        lines.append(
            f"stat: metric={s.metric} p_value={s.p_value} significant={s.significant} "
            f"delta={s.delta} control={s.control} variant={s.variant}"
        )
    for qc in report.quality_checks:
        lines.append(f"quality_check: {qc.label} passed={qc.passed} critical={qc.critical}")
    return "\n".join(lines)


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*%?")


def unsupported_claims_heuristic(explanation: str, facts_summary: str) -> list[str]:
    """
    Deterministic pre-check (NOT the judge): every numeric token in the
    explanation should appear, in some form, in the structured facts
    summary. This catches only the crudest fabrications (a number that
    exists nowhere in the facts) — it's intentionally conservative
    (few false positives) since numeric formatting differs (e.g. "8.4%"
    vs "8.4"), and is meant as a fast pre-filter ahead of the LLM
    judge's `unsupported_claims`, not a replacement for it.
    """
    facts_numbers = set(_NUMBER_RE.findall(facts_summary))
    facts_numbers_normalized = {n.rstrip("%").replace(",", "") for n in facts_numbers}
    suspicious = []
    for token in _NUMBER_RE.findall(explanation):
        normalized = token.rstrip("%").replace(",", "")
        if not normalized or normalized in ("-", "+"):
            continue
        if normalized not in facts_numbers_normalized:
            suspicious.append(token)
    return suspicious


@dataclass
class ExplanationEvalResult:
    case_id: str
    groundedness: int
    completeness: int
    decision_consistency: int
    clarity: int
    unsupported_claims: list[str]
    heuristic_unsupported_numbers: list[str]
    rationale: str

    @property
    def is_grounded(self) -> bool:
        return self.groundedness >= 4 and not self.unsupported_claims and not self.heuristic_unsupported_numbers


_JUDGE_SYSTEM_PROMPT = """You are grading whether an automatically-generated experiment-report \
explanation is faithful to a fixed set of structured facts. You are NOT grading whether the \
underlying statistical decision is correct — that is decided elsewhere and is out of scope. \
Score each dimension 1 (worst) to 5 (best):
- groundedness: does the explanation agree with the structured facts, without contradicting any of them?
- completeness: does it mention the evidence needed to justify the decision (decision, validity, significance, guardrails)?
- decision_consistency: does the explanation's narrative conclusion match the structured `decision` field exactly (e.g. does not say "consider shipping" when decision=NO_GO)?
- clarity: would a product/data decision-maker understand the explanation and know what to do next?
List any unsupported_claims: specific facts/numbers/qualifiers asserted in the explanation that do NOT appear in the structured facts.
Respond only with the requested structured fields."""


def _default_judge_fn(system_prompt: str, user_prompt: str) -> _JudgeScores:
    """Live judge — wraps app.llm.client.get_llm(), the project's single
    LLM-construction chokepoint (see that module's docstring). Never
    called by default in tests; tests inject a fake `judge_fn` instead
    (same pattern as tests/llm/test_llm_report_generator.py's `_fake_llm`).
    """
    from app.llm.client import get_llm

    llm = get_llm().with_structured_output(_JudgeScores)
    return llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])


def judge_explanation(
    case_id: str,
    report: ExperimentReport,
    explanation: str,
    judge_fn: Callable[[str, str], _JudgeScores] = _default_judge_fn,
) -> ExplanationEvalResult:
    facts_summary = _structured_facts_summary(report)
    heuristic_flags = unsupported_claims_heuristic(explanation, facts_summary)

    user_prompt = (
        f"STRUCTURED FACTS (ground truth — the explanation must not contradict or go beyond these):\n"
        f"{facts_summary}\n\n"
        f"EXPLANATION TO GRADE:\n{explanation}\n"
    )
    scores = judge_fn(_JUDGE_SYSTEM_PROMPT, user_prompt)

    return ExplanationEvalResult(
        case_id=case_id,
        groundedness=scores.groundedness,
        completeness=scores.completeness,
        decision_consistency=scores.decision_consistency,
        clarity=scores.clarity,
        unsupported_claims=scores.unsupported_claims,
        heuristic_unsupported_numbers=heuristic_flags,
        rationale=scores.rationale,
    )


@dataclass
class ExplanationEvalSummary:
    n: int
    avg_groundedness: float
    avg_completeness: float
    avg_decision_consistency: float
    avg_clarity: float
    unsupported_claim_rate: float  # fraction of cases with >=1 unsupported claim (LLM- or heuristic-flagged)
    results: list[ExplanationEvalResult]


def summarize(results: list[ExplanationEvalResult]) -> ExplanationEvalSummary:
    n = len(results)
    if n == 0:
        return ExplanationEvalSummary(0, 0, 0, 0, 0, 0, [])
    flagged = sum(1 for r in results if r.unsupported_claims or r.heuristic_unsupported_numbers)
    return ExplanationEvalSummary(
        n=n,
        avg_groundedness=sum(r.groundedness for r in results) / n,
        avg_completeness=sum(r.completeness for r in results) / n,
        avg_decision_consistency=sum(r.decision_consistency for r in results) / n,
        avg_clarity=sum(r.clarity for r in results) / n,
        unsupported_claim_rate=flagged / n,
        results=results,
    )
