"""
Judge reliability — Layer 6 of the evaluation strategy.

"Do not blindly trust a single LLM judge" (task spec section 6). Two
checks:

  1. `repeat_judge()` — run the SAME judge call multiple times with a
     FIXED prompt and measure score variance / agreement. High
     variance on the identical input means the judge itself is noisy,
     which caps how much weight `explanation_judge.py`'s scores should
     be given.

  2. `pairwise_order_robustness()` — for a pairwise comparison judge
     (grading candidate A vs candidate B), check whether swapping
     which one is presented first changes the verdict. A judge that
     flips its answer purely because of presentation order is
     position-biased, not actually comparing content.

Both are pure functions over injected `judge_fn` callables — same
testability pattern as explanation_judge.py, no live network call
required for tests.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Literal


# ----------------------------------------------------------------------
# 1. Repeated-run variance / agreement
# ----------------------------------------------------------------------

@dataclass
class RepeatJudgeReport:
    n_runs: int
    scores: list[int]
    mean: float
    stdev: float
    # Fraction of runs that agree with the majority/modal score --
    # a simple, interpretable stand-in for inter-rater agreement when
    # there's only one "rater" run repeatedly.
    agreement_rate: float


def repeat_judge(
    judge_fn: Callable[[], int],
    n_runs: int = 5,
) -> RepeatJudgeReport:
    """
    `judge_fn` takes no arguments and returns a single integer score
    (e.g. `groundedness`) for a FIXED, already-bound prompt — callers
    partial-apply the specific case/explanation before passing this in
    so this function's only concern is repetition and variance.
    """
    scores = [judge_fn() for _ in range(n_runs)]
    mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    modal = statistics.mode(scores)
    agreement = sum(1 for s in scores if s == modal) / len(scores)
    return RepeatJudgeReport(n_runs=n_runs, scores=scores, mean=mean, stdev=stdev, agreement_rate=agreement)


# ----------------------------------------------------------------------
# 2. Pairwise position-order robustness
# ----------------------------------------------------------------------

Verdict = Literal["A", "B", "TIE"]


@dataclass
class PositionRobustnessReport:
    case_id: str
    verdict_original_order: Verdict
    verdict_swapped_order: Verdict
    # After swapping which candidate is labeled "A" vs "B", the winning
    # CANDIDATE (not the winning LABEL) should stay the same. Consistent
    # means the underlying preference didn't flip due to presentation
    # order alone.
    consistent: bool


def pairwise_order_robustness(
    case_id: str,
    candidate_a: str,
    candidate_b: str,
    pairwise_judge_fn: Callable[[str, str], Verdict],
) -> PositionRobustnessReport:
    """
    `pairwise_judge_fn(first, second)` returns which of `first`/`second`
    it prefers, labeled "A" (=first) / "B" (=second) / "TIE" — i.e. it
    always grades positionally, never by candidate identity. This
    function calls it once in original order and once with the two
    candidates swapped, then translates both verdicts back to
    candidate identity to check consistency.
    """
    verdict_original = pairwise_judge_fn(candidate_a, candidate_b)
    verdict_swapped_raw = pairwise_judge_fn(candidate_b, candidate_a)

    # Translate swapped-order verdict back into "which candidate won"
    # terms: in the swapped call, "A" position holds candidate_b.
    if verdict_swapped_raw == "A":
        winner_swapped = "B"  # candidate_b won, but it was slotted as "A"
    elif verdict_swapped_raw == "B":
        winner_swapped = "A"  # candidate_a won, but it was slotted as "B"
    else:
        winner_swapped = "TIE"

    consistent = verdict_original == winner_swapped

    return PositionRobustnessReport(
        case_id=case_id,
        verdict_original_order=verdict_original,
        verdict_swapped_order=verdict_swapped_raw,
        consistent=consistent,
    )


@dataclass
class JudgeStabilityReport:
    repeat_reports: list[RepeatJudgeReport]
    position_reports: list[PositionRobustnessReport]

    @property
    def avg_agreement_rate(self) -> float:
        if not self.repeat_reports:
            return 1.0
        return sum(r.agreement_rate for r in self.repeat_reports) / len(self.repeat_reports)

    @property
    def position_consistency_rate(self) -> float:
        if not self.position_reports:
            return 1.0
        return sum(1 for r in self.position_reports if r.consistent) / len(self.position_reports)
