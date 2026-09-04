"""
Regression tests for `_hypothesis_test_skip_reason` in
app/api/routes_experiments.py — the "why did Experiment produce no
hypothesis test" explanation.

Before this fix, the ONLY reason ever given was:

    Not needed for intent "<intent label>"

which is wrong whenever a validity gate (conflicting variant
assignments, SRM failure, or another critical quality failure) is the
REAL reason testing was skipped — Full Experiment Review, in
particular, normally DOES require hypothesis testing.
"""

from app.api.routes_experiments import _hypothesis_test_skip_reason
from app.schemas.quality import QualityCheck, SRMResult


def _srm(passed: bool, p_value: float = 0.5) -> SRMResult:
    return SRMResult(
        observed_control=100,
        observed_variant=100,
        expected_ratio=0.5,
        chi2_statistic=0.0,
        p_value=p_value,
        passed=passed,
    )


def test_conflicting_variant_assignments_gives_validity_reason_with_exact_count():
    state = {
        "has_conflicting_variant_duplicates": True,
        "quality_checks": [
            QualityCheck(
                label="Duplicate User Variant Conflicts",
                passed=False,
                detail=(
                    "1,541 user(s) found assigned to MORE THAN ONE variant — this is not a "
                    "harmless duplicate, it indicates a broken randomization/assignment "
                    "pipeline. Results below cannot be trusted until this is fixed."
                ),
            )
        ],
    }
    plan = {"intent_label": "Full Experiment Review"}
    reason = _hypothesis_test_skip_reason(state, plan, srm=_srm(passed=True))

    assert "1,541" in reason
    assert "validity" in reason.lower()
    # Must NOT say the old, misleading reason:
    assert "Not needed for intent" not in reason
    assert "Full Experiment Review" not in reason
    # Must not invent a different number:
    assert "1,967" not in reason


def test_conflicting_variant_reason_takes_priority_over_srm_wording():
    """SRM failure and variant conflicts must never be conflated into one explanation."""
    state = {
        "has_conflicting_variant_duplicates": True,
        "quality_checks": [
            QualityCheck(
                label="Duplicate User Variant Conflicts",
                passed=False,
                detail="1,541 user(s) found assigned to MORE THAN ONE variant.",
            )
        ],
    }
    plan = {"intent_label": "Full Experiment Review"}
    reason = _hypothesis_test_skip_reason(state, plan, srm=_srm(passed=False, p_value=0.001))

    assert "1,541" in reason
    assert "Sample Ratio Mismatch" not in reason
    assert "SRM" not in reason


def test_srm_failure_alone_gives_srm_specific_reason():
    state = {"has_conflicting_variant_duplicates": False, "quality_checks": []}
    plan = {"intent_label": "Full Experiment Review"}
    reason = _hypothesis_test_skip_reason(state, plan, srm=_srm(passed=False, p_value=0.002))

    assert "SRM" in reason or "Sample Ratio Mismatch" in reason
    assert "0.002" in reason
    assert "Not needed for intent" not in reason


def test_no_validity_failure_falls_back_to_intent_wording_unchanged():
    """
    Regression guard: when there really is no validity/quality-gate
    reason (e.g. a pure conceptual/knowledge-base question), the
    original intent-based wording must still be produced exactly as
    before this fix.
    """
    state = {"has_conflicting_variant_duplicates": False, "quality_checks": []}
    plan = {"intent_label": "Knowledge Base Question"}
    reason = _hypothesis_test_skip_reason(state, plan, srm=None)

    assert reason == 'Not needed for intent "Knowledge Base Question"'
