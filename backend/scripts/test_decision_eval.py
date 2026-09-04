"""
Regression guard for decision quality — runs the same 15-scenario
expert-labeled eval set as scripts/evaluate_decisions.py and asserts
all three metrics stay at their required floor. Unlike retrieval
quality (where <100% is expected and documented), decision accuracy,
numerical consistency, and unsupported-recommendation-rate are held to
100% here: this is deterministic Python logic being checked against
deterministic rules, not a lexical-similarity search over free text,
so there's no legitimate reason for any of these three to regress even
slightly. If one of these ever fails, treat it as a real bug in
`TemplateReportGenerator`'s decision rules, not something to relax the
threshold for.
"""

from scripts.evaluate_decisions import evaluate_decisions


def test_decision_accuracy_is_perfect():
    result = evaluate_decisions()
    assert result.decision_accuracy == 1.0, (
        f"Decision accuracy dropped to {result.decision_accuracy:.1%}. Failures: {result.failures}"
    )


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
