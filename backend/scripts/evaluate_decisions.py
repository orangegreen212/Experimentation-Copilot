"""
Decision-quality evaluation — the level above retrieval quality.

Top-1/Hit@3/MRR/NDCG@3 (see evaluate_retrieval.py) only answer "did we
find the right methodology chunk?" They say nothing about whether the
agent actually reaches the METHODOLOGICALLY CORRECT ship/no-ship
recommendation for a given set of experiment facts, or whether it
stays grounded in those facts rather than fabricating something.

This is an expert-labeled evaluation set: 15 hand-built experiment
scenarios (SRM failure, broken variant assignment, underpowered null
result, borderline p-value, negative significant effect, stacked
quality issues, funnel-only, and more — see SCENARIOS below),
each with an expected decision an experienced Product Analyst would
reach. It exercises `TemplateReportGenerator` directly (not the LLM
path) — see the module docstring rationale below for why.

Three metrics, matching the "Decision / Numbers / Safety" rows of the
evaluation strategy:

  - Decision Accuracy: does `report.confidence` match the expected
    label (SHIP-worthy / DO-NOT-SHIP / INCONCLUSIVE)?
  - Numerical Consistency: are `report.stats`/`report.mde`/
    `report.sample_size_note` byte-identical to what was fed in via
    `ReportFacts` — i.e. is the report incapable of silently changing
    a number? (This is checked structurally, not by comparing against
    a second computation, since TemplateReportGenerator's contract
    IS to copy facts through unchanged — see report_generator.py.)
  - Unsupported Recommendation Rate: when `kb_results` is empty, does
    the report avoid inventing a methodology citation anyway? (Should
    always be 0% — this is the deterministic path's whole point.)

WHY TemplateReportGenerator, not the live LLM: this needs to be a fast,
deterministic, CI-safe regression guard (like evaluate_retrieval.py),
not a live network call to OpenRouter graded by another LLM. The
`_assess_confidence`/`_recommendations` decision RULES this evaluates
are the same deterministic logic `LLMReportGenerator` reuses via
composition for the confidence level (see report_generator.py's
`LLMReportGenerator.generate`) — so this evaluation exercises the part
of the decision that's guaranteed identical on both paths. What the
LLM path adds beyond this is narration quality, not decision logic;
grading narration quality well would need a live LLM-as-a-judge setup,
which the discussion in this thread deliberately scoped out for now
("I would not build an LLM-as-a-judge as the only metric... a small
expert-labeled set + deterministic checks").

Run standalone:

    python3 scripts/evaluate_decisions.py

Also exercised as a regression guard in
tests/graph/test_decision_eval.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.report_generator import ReportFacts, TemplateReportGenerator  # noqa: E402
from app.rag.retriever import DocumentChunk, RetrievedChunk  # noqa: E402
from app.schemas.dataset import DatasetInfo, DatasetType  # noqa: E402
from app.schemas.quality import QualityCheck  # noqa: E402
from app.schemas.report import ConfidenceLevel  # noqa: E402
from app.schemas.statistics import HypothesisTestType, PowerAnalysisResult, StatResult  # noqa: E402


def _dataset(users: int = 12_400) -> DatasetInfo:
    return DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST, variants=2, users=users, metric_label="Conversion Rate",
        metric_selection_reason="Selected by the deterministic outcome-column priority — no competing outcome metrics were available in this dataset.",
    )


def _srm_pass() -> QualityCheck:
    return QualityCheck(label="Sample Ratio Mismatch (SRM)", passed=True, detail="p=0.83")


def _srm_fail() -> QualityCheck:
    return QualityCheck(label="Sample Ratio Mismatch (SRM)", passed=False, detail="p=0.0002")


def _stat(p_value: float, significant: bool, delta: str = "+8.4% (rel)", delta_relative: float = 8.4) -> StatResult:
    return StatResult(
        metric="Conversion Rate", test_type=HypothesisTestType.CHI_SQUARE, test_name="Chi-square test",
        statistic=12.4, selection_reason="binary metric", control="4.21%", variant="4.56%",
        delta=delta, delta_relative=delta_relative, p_value=p_value, significant=significant,
        ci_lower="-0.10pp", ci_upper="+0.80pp",
    )


def _power(achieved: float, sufficient: bool, mde: float = 1.8) -> PowerAnalysisResult:
    return PowerAnalysisResult(
        minimum_detectable_effect_relative=mde, required_sample_size=8200, observed_sample_size=12_400,
        achieved_power=achieved, alpha=0.05, is_sufficiently_powered=sufficient,
    )


def _kb(source: str = "kohavi.md", heading: str = "Minimum Detectable Effect (MDE) and Power") -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=DocumentChunk(source=source, heading=heading, content="..."), score=0.4)]


@dataclass
class Scenario:
    name: str
    facts: ReportFacts
    expected_confidence: ConfidenceLevel
    # One of Decision's values (GO / GO_WITH_CAUTION / NO_GO / INCONCLUSIVE / INVALID)
    # — the actual decision a Product
    # Analyst would reach, independent of the confidence LABEL. This is what
    # closes the original gap: confidence answers "can this measurement be
    # trusted", NOT "what should we do about it" — a scenario can legitimately
    # be HIGH confidence and still be a "do_not_ship" (a clean, well-powered,
    # significant REGRESSION is exactly that).
    #
    # Checked directly against the structured `report.decision` field
    # (the `Decision` enum, app/schemas/report.py) — NOT by parsing
    # `report.recommendations` text. This is the single source of truth for
    # "what was decided", never a regex over free text standing in for it.
    expected_decision: str
    description: str = ""


SCENARIOS: list[Scenario] = [
    Scenario(
        "srm_failure",
        ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_fail()],
            srm_passed=False, stat_results=[], test_selections=[], power_analysis=None,
            mde_display="N/A", sample_size_note="N/A", kb_results=_kb("kohavi.md", "Sample Ratio Mismatch"),
        ),
        ConfidenceLevel.LOW,
        "INVALID",
        "SRM failed -> randomization broken -> do not trust the experiment regardless of any p-value.",
    ),
    Scenario(
        "conflicting_variant_duplicates",
        ReportFacts(
            user_prompt="ship it?", dataset=_dataset(), quality_checks=[_srm_pass()], srm_passed=True,
            stat_results=[], test_selections=[], power_analysis=None, mde_display="N/A",
            sample_size_note="N/A", has_conflicting_variant_duplicates=True,
        ),
        ConfidenceLevel.LOW,
        "INVALID",
        "Users assigned to both variants -> broken assignment pipeline -> same severity as SRM failure.",
    ),
    Scenario(
        "underpowered_null_result",
        ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.42, significant=False, delta="-1.2% (rel)", delta_relative=-1.2)],
            test_selections=[], power_analysis=_power(achieved=0.24, sufficient=False),
            mde_display="2.8%", sample_size_note="12,400 users", kb_results=_kb(),
        ),
        ConfidenceLevel.MEDIUM,
        "INCONCLUSIVE",
        "p=0.42, power=24% -> null result is INCONCLUSIVE (do not conclude 'no effect'), not a confident no-ship.",
    ),
    Scenario(
        "adequately_powered_null_result",
        ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.71, significant=False, delta="+0.1% (rel)", delta_relative=0.1)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users",
        ),
        ConfidenceLevel.HIGH,
        "INCONCLUSIVE",
        "Adequately powered AND null -> a genuinely trustworthy 'no effect' read, not just 'inconclusive.'",
    ),
    Scenario(
        "significant_and_adequately_powered",
        ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.0003, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
        ),
        ConfidenceLevel.HIGH,
        "GO_WITH_CAUTION",
        "Significant + adequately powered + quality checks pass -> ship-worthy, high confidence.",
    ),
    Scenario(
        "significant_but_underpowered",
        ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.02, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.55, sufficient=False),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
        ),
        ConfidenceLevel.MEDIUM,
        "GO_WITH_CAUTION",
        "Significant, but the experiment was still underpowered for the target MDE -> caution, not full confidence.",
    ),
    Scenario(
        "quality_issue_present",
        ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(),
            quality_checks=[_srm_pass(), QualityCheck(label="Null Values", passed=False, detail="3.1% missing")],
            srm_passed=True, stat_results=[_stat(p_value=0.01, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users",
        ),
        ConfidenceLevel.MEDIUM,
        "GO_WITH_CAUTION",
        "A non-SRM data quality issue (nulls) downgrades confidence even with a significant, powered result.",
    ),
    Scenario(
        "quality_only_no_hypothesis_test",
        ReportFacts(
            user_prompt="Check the SRM for this dataset", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[], test_selections=[], power_analysis=None,
            mde_display="N/A", sample_size_note="N/A",
        ),
        ConfidenceLevel.MEDIUM,
        "INCONCLUSIVE",
        "No hypothesis test was requested/run -> no ship/no-ship confidence should be implied either way.",
    ),
    Scenario(
        "negative_significant_effect",
        ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.001, significant=True, delta="-4.2% (rel)", delta_relative=-4.2)],
            test_selections=[], power_analysis=_power(achieved=0.94, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
        ),
        ConfidenceLevel.HIGH,
        "NO_GO",
        "A statistically significant NEGATIVE effect is still a confident, well-powered result -- just a "
        "confident 'do not ship' rather than 'ship.' Confidence in the READ stays high; direction is separate "
        "from confidence, and TemplateReportGenerator must not conflate 'significant' with 'positive.'",
    ),
    Scenario(
        "multiple_quality_failures",
        ReportFacts(
            user_prompt="Evaluate this experiment", dataset=_dataset(),
            quality_checks=[
                _srm_pass(),
                QualityCheck(label="Null Values", passed=False, detail="4.8% missing"),
                QualityCheck(label="Duplicate Users", passed=False, detail="1.2% duplicated"),
            ],
            srm_passed=True, stat_results=[_stat(p_value=0.002, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.88, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users",
        ),
        ConfidenceLevel.MEDIUM,
        "GO_WITH_CAUTION",
        "Two independent non-SRM quality issues stacked on top of an otherwise clean significant result -- "
        "still not LOW (SRM/randomization is intact, so the experiment is still analyzable), but the "
        "accumulated data-quality risk should keep it below HIGH.",
    ),
    Scenario(
        "borderline_p_value",
        ReportFacts(
            user_prompt="Is this result statistically significant?", dataset=_dataset(),
            quality_checks=[_srm_pass()], srm_passed=True,
            stat_results=[_stat(p_value=0.048, significant=True)], test_selections=[],
            power_analysis=_power(achieved=0.81, sufficient=True), mde_display="1.8%",
            sample_size_note="12,400 users", kb_results=_kb(),
        ),
        ConfidenceLevel.HIGH,
        "GO_WITH_CAUTION",
        "p=0.048 crosses the pre-registered alpha=0.05 threshold and the experiment is adequately powered -- "
        "'barely significant' is still significant; the report must not silently downgrade confidence just "
        "because the p-value is close to the boundary (that would be an undisclosed second threshold).",
    ),
    Scenario(
        "small_sample_underpowered_and_not_significant",
        ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(users=640),
            quality_checks=[_srm_pass()], srm_passed=True,
            stat_results=[_stat(p_value=0.61, significant=False, delta="+3.1% (rel)", delta_relative=3.1)],
            test_selections=[], power_analysis=_power(achieved=0.09, sufficient=False, mde=14.5),
            mde_display="14.5%", sample_size_note="640 users -- far below the ~8,200 required", kb_results=_kb(),
        ),
        ConfidenceLevel.MEDIUM,
        "INCONCLUSIVE",
        "Severely underpowered (9% achieved power) AND not significant -- this is a 'we learned essentially "
        "nothing' result, not evidence either way; must stay INCONCLUSIVE, not escalate to a confident LOW "
        "(LOW is reserved for broken data/randomization, not merely 'small sample').",
    ),
    Scenario(
        "srm_failure_with_significant_looking_stats_present",
        ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(),
            quality_checks=[_srm_fail()], srm_passed=False,
            # Deliberately non-empty stat_results, simulating a caller that computed them anyway --
            # the report generator must still treat this as untrustworthy, not average the two signals.
            stat_results=[_stat(p_value=0.0001, significant=True, delta="+9.1% (rel)", delta_relative=9.1)],
            test_selections=[], power_analysis=_power(achieved=0.95, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb("kohavi.md", "Sample Ratio Mismatch"),
        ),
        ConfidenceLevel.LOW,
        "INVALID",
        "Even an eye-catching, well-powered, highly-significant-looking result must not override an SRM "
        "failure -- broken randomization invalidates the comparison regardless of how clean the p-value "
        "looks; this is the single most safety-critical case in the whole decision layer (Twyman's Law).",
    ),
    Scenario(
        "no_srm_check_run_at_all",
        ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(),
            quality_checks=[], srm_passed=True,  # no SRM QualityCheck present, but srm_passed defaults True
            stat_results=[_stat(p_value=0.01, significant=True)], test_selections=[],
            power_analysis=_power(achieved=0.9, sufficient=True), mde_display="1.8%",
            sample_size_note="12,400 users",
        ),
        ConfidenceLevel.HIGH,
        "GO_WITH_CAUTION",
        "Sanity check on the eval harness itself: srm_passed=True with an otherwise-clean, well-powered, "
        "significant result should read the same as the explicit-SRM-check-present case above -- confidence "
        "shouldn't silently depend on whether an SRM QualityCheck object happens to be present in the list.",
    ),
    Scenario(
        "funnel_only_no_experiment_capability",
        ReportFacts(
            user_prompt="Where are users dropping off in the funnel?", dataset=_dataset(),
            quality_checks=[], srm_passed=True, stat_results=[], test_selections=[], power_analysis=None,
            mde_display="N/A", sample_size_note="N/A", validation_ran=False,
        ),
        ConfidenceLevel.MEDIUM,
        "INCONCLUSIVE",
        "A funnel-only request (no validation/experiment capability selected) never ran a hypothesis test -- "
        "same as the quality-only case, ship/no-ship confidence must not be implied from funnel data alone.",
    ),
]


@dataclass
class DecisionEvalResult:
    # Renamed from the original `decision_accuracy` — that field only ever
    # checked `report.confidence`, which answers "can this measurement be
    # trusted", not "what should we do about it". Kept as its own metric
    # since it's still meaningful, just correctly named now.
    confidence_accuracy: float
    # The metric this eval was missing: does `report.recommendations`
    # actually reach the correct ship/do_not_ship/inconclusive call,
    # classified via the structured `report.decision` field. This is the metric that would
    # have caught the negative-significant-effect "Consider shipping" bug —
    # confidence_accuracy alone did not, since that scenario's confidence
    # (HIGH) was already correct even while the recommendation text was wrong.
    decision_accuracy: float
    numerical_consistency: float
    unsupported_recommendation_rate: float
    n_scenarios: int
    failures: list[str] = field(default_factory=list)


def evaluate_decisions(scenarios: list[Scenario] = SCENARIOS) -> DecisionEvalResult:
    generator = TemplateReportGenerator()

    confidence_correct = 0
    decision_correct = 0
    numbers_consistent = 0
    unsupported_citations = 0
    failures: list[str] = []

    for s in scenarios:
        report = generator.generate(s.facts)

        if report.confidence == s.expected_confidence:
            confidence_correct += 1
        else:
            failures.append(
                f"[{s.name}] expected confidence={s.expected_confidence.value}, "
                f"got {report.confidence.value} — {s.description}"
            )

        actual_decision = report.decision.value
        if actual_decision == s.expected_decision:
            decision_correct += 1
        else:
            failures.append(
                f"[{s.name}] expected decision={s.expected_decision!r}, got {actual_decision!r} "
                f"— recommendations={report.recommendations!r}"
            )

        # Numerical Consistency: TemplateReportGenerator's contract is to
        # copy facts through byte-identical, never derive/alter a number --
        # EXCEPT the deliberate `validation_ran=False` placeholder path
        # (no dataset was ever evaluated, so mde/sample_size_note are
        # replaced with a fixed "N/A — conceptual question" string by
        # design, not silently altered). stat_results must still always
        # pass through untouched, in every case.
        stats_ok = report.stats == s.facts.stat_results
        if s.facts.validation_ran:
            numbers_ok = stats_ok and report.mde == s.facts.mde_display and report.sample_size_note == s.facts.sample_size_note
        else:
            numbers_ok = stats_ok and s.facts.stat_results == []
        if numbers_ok:
            numbers_consistent += 1
        else:
            failures.append(f"[{s.name}] numerical consistency violated")

        # Unsupported Recommendation Rate: no kb_results -> no methodology
        # citation should appear anywhere in recommendations.
        if not s.facts.kb_results:
            cited_without_source = any(
                ("Methodology guidance" in r or ".md" in r) for r in report.recommendations
            )
            if cited_without_source:
                unsupported_citations += 1
                failures.append(f"[{s.name}] cited methodology with no retrieved kb_results")

    n = len(scenarios)
    return DecisionEvalResult(
        confidence_accuracy=confidence_correct / n,
        decision_accuracy=decision_correct / n,
        numerical_consistency=numbers_consistent / n,
        unsupported_recommendation_rate=unsupported_citations / n,
        n_scenarios=n,
        failures=failures,
    )


def main() -> None:
    result = evaluate_decisions()
    print(f"Scenarios evaluated:          {result.n_scenarios}")
    print(f"Confidence Accuracy:          {result.confidence_accuracy * 100:.1f}%")
    print(f"Decision Accuracy:            {result.decision_accuracy * 100:.1f}%")
    print(f"Numerical Consistency:        {result.numerical_consistency * 100:.1f}%")
    print(f"Unsupported Recommendation Rate: {result.unsupported_recommendation_rate * 100:.1f}%")
    if result.failures:
        print("\nFailures:")
        for f in result.failures:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
