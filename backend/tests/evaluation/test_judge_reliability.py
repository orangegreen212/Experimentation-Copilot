"""Tests for evaluation/evaluators/judge_reliability.py."""

from __future__ import annotations

from evaluation.evaluators.judge_reliability import (
    pairwise_order_robustness,
    repeat_judge,
)


def test_repeat_judge_detects_stable_scores():
    result = repeat_judge(lambda: 5, n_runs=5)
    assert result.mean == 5.0
    assert result.stdev == 0.0
    assert result.agreement_rate == 1.0


def test_repeat_judge_detects_noisy_scores():
    scores = iter([5, 3, 5, 2, 5])
    result = repeat_judge(lambda: next(scores), n_runs=5)
    assert result.stdev > 0.0
    assert result.agreement_rate == 3 / 5  # three 5's out of five runs


def test_pairwise_order_robustness_detects_consistent_judge():
    def judge_fn(first: str, second: str):
        # Always prefers whichever candidate is literally "better_report".
        if first == "better_report":
            return "A"
        if second == "better_report":
            return "B"
        return "TIE"

    result = pairwise_order_robustness("case1", "better_report", "worse_report", judge_fn)
    assert result.consistent


def test_pairwise_order_robustness_detects_position_biased_judge():
    def position_biased_judge_fn(first: str, second: str):
        # Always prefers whichever candidate is presented FIRST, regardless of content.
        return "A"

    result = pairwise_order_robustness("case1", "candidate_x", "candidate_y", position_biased_judge_fn)
    assert not result.consistent
