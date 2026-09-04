"""
Product improvement — tests for `build_decision_narrative` (deterministic
explanation layer over the existing canonical decision model). No LLM,
no new statistics: verifies each decision branch only ever restates
facts it was given.
"""

from app.schemas.report import Decision, ExperimentValidity, GuardrailStatus
from app.schemas.statistics import HypothesisTestType, PowerAnalysisResult, StatResult
from app.stats.decision_narrative import build_decision_narrative


def _stat(metric="Conversion Rate", significant=True, delta="+51.2% (rel)", is_omnibus=False, is_winner=False):
    return StatResult(
        metric=metric,
        test_type=HypothesisTestType.CHI_SQUARE,
        test_name="Chi-square test",
        statistic=123.4,
        selection_reason="Binary metric with sufficient expected cell counts.",
        control="11.87%",
        variant="17.95%",
        delta=delta,
        p_value=0.0,
        significant=significant,
        ci_lower="+5.82pp",
        ci_upper="+6.33pp",
        is_omnibus=is_omnibus,
        is_winner=is_winner,
    )


def _power(is_sufficiently_powered=True, required_sample_size=1000):
    return PowerAnalysisResult(
        minimum_detectable_effect_relative=0.05,
        required_sample_size=required_sample_size,
        observed_sample_size=500,
        achieved_power=0.5 if not is_sufficiently_powered else 0.9,
        is_sufficiently_powered=is_sufficiently_powered,
    )


class TestGoWithCaution:
    def test_contains_all_four_required_sections(self):
        narrative = build_decision_narrative(
            decision=Decision.GO_WITH_CAUTION,
            decision_reason="Conversion Rate is statistically significant...",
            experiment_validity=ExperimentValidity.VALID,
            guardrail_status=GuardrailStatus.NOT_AVAILABLE,
            practical_significance=True,
            stat_results=[_stat()],
            guardrail_results=[],
            available_metrics=["Conversion Rate", "Purchase Amount", "Session Duration"],
            power_analysis=None,
        )
        assert narrative.why_this_decision
        assert narrative.what_prevents_full_go
        assert any("guardrail" in s.lower() for s in narrative.what_prevents_full_go)
        assert narrative.monitoring.primary_metric == "Conversion Rate"
        assert narrative.monitoring.guardrails_evaluated == []
        assert set(narrative.monitoring.potential_monitoring_metrics) == {"Purchase Amount", "Session Duration"}
        assert narrative.recommended_next_step

    def test_does_not_call_unavailable_guardrails_evaluated(self):
        narrative = build_decision_narrative(
            decision=Decision.GO_WITH_CAUTION,
            decision_reason="reason",
            experiment_validity=ExperimentValidity.VALID,
            guardrail_status=GuardrailStatus.NOT_AVAILABLE,
            practical_significance=True,
            stat_results=[_stat()],
            guardrail_results=[],
            available_metrics=["Conversion Rate"],
            power_analysis=None,
        )
        assert narrative.monitoring.guardrails_evaluated == []
        assert "Guardrail metrics are unavailable." in narrative.what_prevents_full_go


class TestGo:
    def test_go_has_no_prevents_full_go(self):
        narrative = build_decision_narrative(
            decision=Decision.GO,
            decision_reason="reason",
            experiment_validity=ExperimentValidity.VALID,
            guardrail_status=GuardrailStatus.PASS,
            practical_significance=True,
            stat_results=[_stat()],
            guardrail_results=[_stat(metric="Purchase Amount", significant=False)],
            available_metrics=["Conversion Rate", "Purchase Amount"],
            power_analysis=_power(is_sufficiently_powered=True),
        )
        assert narrative.what_prevents_full_go == []
        assert narrative.monitoring.guardrails_evaluated == ["Purchase Amount"]
        assert "Statistical power is sufficient." in narrative.why_this_decision


class TestInconclusive:
    def test_underpowered_includes_sample_size_recommendation(self):
        narrative = build_decision_narrative(
            decision=Decision.INCONCLUSIVE,
            decision_reason="No statistically significant difference.",
            experiment_validity=ExperimentValidity.VALID,
            guardrail_status=GuardrailStatus.NOT_AVAILABLE,
            practical_significance=None,
            stat_results=[_stat(significant=False)],
            guardrail_results=[],
            available_metrics=["Conversion Rate"],
            power_analysis=_power(is_sufficiently_powered=False, required_sample_size=48213),
        )
        assert any("48,213" in s for s in narrative.what_would_change_decision)
        assert "power" in narrative.why_this_decision[0].lower()

    def test_not_underpowered_falls_back_to_decision_reason(self):
        narrative = build_decision_narrative(
            decision=Decision.INCONCLUSIVE,
            decision_reason="No hypothesis test was run for this request.",
            experiment_validity=ExperimentValidity.VALID,
            guardrail_status=GuardrailStatus.NOT_AVAILABLE,
            practical_significance=None,
            stat_results=[],
            guardrail_results=[],
            available_metrics=[],
            power_analysis=None,
        )
        assert narrative.why_this_decision == ["No hypothesis test was run for this request."]
        assert not any("48,213" in s for s in narrative.what_would_change_decision)


class TestInvalid:
    def test_invalid_never_recommends_rollout(self):
        narrative = build_decision_narrative(
            decision=Decision.INVALID,
            decision_reason="Do not ship — the experiment failed a critical validity check.",
            experiment_validity=ExperimentValidity.INVALID,
            guardrail_status=GuardrailStatus.NOT_AVAILABLE,
            practical_significance=None,
            stat_results=[_stat()],
            guardrail_results=[],
            available_metrics=["Conversion Rate"],
            power_analysis=None,
        )
        combined = " ".join([narrative.recommended_next_step, *narrative.why_this_decision]).lower()
        assert "do not" in narrative.recommended_next_step.lower() or "not roll out" in combined
        assert "rollout" not in narrative.recommended_next_step.lower() or "do not" in narrative.recommended_next_step.lower()


class TestNoGo:
    def test_guardrail_failure_named_when_available(self):
        narrative = build_decision_narrative(
            decision=Decision.NO_GO,
            decision_reason="Guardrail metric failed.",
            experiment_validity=ExperimentValidity.VALID,
            guardrail_status=GuardrailStatus.FAIL,
            practical_significance=True,
            stat_results=[_stat()],
            guardrail_results=[_stat(metric="Support Tickets", delta="-9.0% (rel)", significant=True)],
            available_metrics=["Conversion Rate", "Support Tickets"],
            power_analysis=None,
        )
        assert any("Support Tickets" in s for s in narrative.why_this_decision)
        assert "ship" in narrative.recommended_next_step.lower()
        assert "do not" in narrative.recommended_next_step.lower()
