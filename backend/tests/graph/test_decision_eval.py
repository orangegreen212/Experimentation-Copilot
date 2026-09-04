"""
Regression guard for decision quality — runs the same 15-scenario
expert-labeled eval set as scripts/evaluate_decisions.py and asserts
all four metrics stay at their required floor. Unlike retrieval
quality (where <100% is expected and documented), all four metrics
here are held to 100%: this is deterministic Python logic being
checked against deterministic rules, not a lexical-similarity search
over free text, so there's no legitimate reason for any of these to
regress even slightly. If one of these ever fails, treat it as a real
bug in `TemplateReportGenerator`'s decision rules, not something to
relax the threshold for.

IMPORTANT — `decision_accuracy` vs `confidence_accuracy`: these test
DIFFERENT things and both matter. `confidence_accuracy` answers "can
this measurement be trusted" (SRM/quality/power) — the LEGACY
`confidence` field (see schemas/report.py module docstring; no new
logic is built on it). `decision_accuracy` answers "what should we
actually DO about it", by classifying the structured `report.decision`
field (the `Decision` enum: GO / GO_WITH_CAUTION / NO_GO /
INCONCLUSIVE / INVALID), NOT free text. A scenario can be HIGH
confidence and still be `NO_GO` — a clean, well-powered, statistically
significant REGRESSION is exactly that case. Earlier versions of this
eval only checked confidence, and a scenario with exactly that shape
(significant negative effect, HIGH confidence) was passing at 100%
"decision accuracy" while the underlying recommendation text literally
said "Consider shipping" for a -4.2% regression — a real bug that
confidence-only checking could not have caught. `decision_accuracy`
exists specifically to catch that class of bug going forward.
"""

from scripts.evaluate_decisions import evaluate_decisions


def test_confidence_accuracy_is_perfect():
    result = evaluate_decisions()
    assert result.confidence_accuracy == 1.0, (
        f"Confidence accuracy dropped to {result.confidence_accuracy:.1%}. Failures: {result.failures}"
    )


def test_decision_accuracy_is_perfect():
    """
    Checks the ACTUAL GO/GO_WITH_CAUTION/NO_GO/INCONCLUSIVE/INVALID call
    via the structured `report.decision` field (the `Decision` enum),
    not by parsing `report.recommendations` text — see module docstring
    for why this is a distinct, necessary metric.
    """
    result = evaluate_decisions()
    assert result.decision_accuracy == 1.0, (
        f"Decision accuracy dropped to {result.decision_accuracy:.1%}. Failures: {result.failures}"
    )


def test_negative_significant_effect_recommends_no_go():
    """
    Direct regression test for the specific bug this field was added to
    catch: a statistically significant NEGATIVE effect must produce
    `decision == Decision.NO_GO`, checked on the structured field
    directly — and, separately, the recommendation text must not read
    as a green light either, since the two are meant to always agree.
    """
    from scripts.evaluate_decisions import SCENARIOS
    from app.graph.report_generator import TemplateReportGenerator
    from app.schemas.report import Decision

    scenario = next(s for s in SCENARIOS if s.name == "negative_significant_effect")
    report = TemplateReportGenerator().generate(scenario.facts)
    assert report.decision == Decision.NO_GO
    assert report.recommendations[0].strip().upper().startswith("NO-GO")


def test_numerical_consistency_is_perfect():
    """The core architectural guarantee: TemplateReportGenerator must never alter a computed number."""
    result = evaluate_decisions()
    assert result.numerical_consistency == 1.0, (
        f"Numerical consistency dropped to {result.numerical_consistency:.1%}. Failures: {result.failures}"
    )


def test_no_unsupported_recommendations():
    """A methodology citation must never appear when kb_results is empty."""
    result = evaluate_decisions()
    assert result.unsupported_recommendation_rate == 0.0, (
        f"Unsupported recommendation rate is {result.unsupported_recommendation_rate:.1%}. Failures: {result.failures}"
    )


def test_scenario_set_covers_the_key_cases():
    """
    Sanity check on the eval harness itself, not just its output — makes
    sure nobody accidentally shrinks SCENARIOS down to a handful of easy
    cases while this test keeps passing on a near-empty set.
    """
    from scripts.evaluate_decisions import SCENARIOS

    names = {s.name for s in SCENARIOS}
    assert len(SCENARIOS) >= 15
    assert "srm_failure" in names
    assert "underpowered_null_result" in names
    assert "srm_failure_with_significant_looking_stats_present" in names
