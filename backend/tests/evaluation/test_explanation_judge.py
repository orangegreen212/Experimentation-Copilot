"""
Tests for evaluation/evaluators/explanation_judge.py.

Covers task spec section 11: "explanation groundedness",
"unsupported claim detection". No live LLM call — a fake `judge_fn`
is injected, the same mocking pattern already used in
tests/llm/test_llm_report_generator.py's `_fake_llm`.
"""

from __future__ import annotations

from app.graph.report_generator import TemplateReportGenerator
from evaluation.datasets.golden_dataset import get_case
from evaluation.evaluators.explanation_judge import (
    _JudgeScores,
    judge_explanation,
    unsupported_claims_heuristic,
)


def _fake_judge_factory(scores: _JudgeScores):
    def _fn(system_prompt: str, user_prompt: str) -> _JudgeScores:
        return scores

    return _fn


def test_grounded_explanation_scores_well():
    case = get_case("valid_positive_go")
    report = TemplateReportGenerator().generate(case.facts)
    explanation = (
        "The primary metric was statistically significant with p=0.0003 and the effect "
        "was practically significant, and the guardrail metric passed, so the recommendation is GO."
    )
    judge_fn = _fake_judge_factory(
        _JudgeScores(groundedness=5, completeness=5, decision_consistency=5, clarity=5, unsupported_claims=[])
    )
    result = judge_explanation(case.id, report, explanation, judge_fn=judge_fn)
    assert result.is_grounded
    assert result.groundedness == 5
    assert result.unsupported_claims == []


def test_ungrounded_explanation_flagged_by_judge():
    case = get_case("srm_failure_invalid")
    report = TemplateReportGenerator().generate(case.facts)
    # Explanation fabricates a specific p-value and a "GO" recommendation
    # that contradicts the actual INVALID decision.
    explanation = "The result was significant at p=0.0001 and we recommend shipping immediately (GO)."
    judge_fn = _fake_judge_factory(
        _JudgeScores(
            groundedness=1,
            completeness=2,
            decision_consistency=1,
            clarity=3,
            unsupported_claims=["p=0.0001", "recommend shipping (GO)"],
            rationale="Contradicts the actual INVALID decision and invents a p-value never computed.",
        )
    )
    result = judge_explanation(case.id, report, explanation, judge_fn=judge_fn)
    assert not result.is_grounded
    assert result.decision_consistency == 1
    assert len(result.unsupported_claims) == 2


def test_unsupported_claims_heuristic_flags_numbers_not_in_facts():
    facts_summary = "decision: GO\nstat: metric=Conversion Rate p_value=0.0003 significant=True delta=+8.4% (rel)"
    explanation_with_fabrication = "The lift was +47.2% and highly significant."
    flagged = unsupported_claims_heuristic(explanation_with_fabrication, facts_summary)
    assert any("47.2" in token for token in flagged)


def test_unsupported_claims_heuristic_does_not_flag_numbers_present_in_facts():
    facts_summary = "decision: GO\nstat: metric=Conversion Rate p_value=0.0003 significant=True delta=+8.4% (rel)"
    explanation_grounded = "The lift was +8.4% and highly significant."
    flagged = unsupported_claims_heuristic(explanation_grounded, facts_summary)
    assert flagged == []
