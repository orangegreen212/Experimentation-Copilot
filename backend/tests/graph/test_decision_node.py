"""
Regression tests for decision_node's outer safety net: if report
generation itself raises an unexpected exception — a bug in report
assembly, not in the underlying data or statistics — the node
must not propagate a 500. It must return a safe, minimal report built
from `build_emergency_fallback_report`, preserving every deterministic
value that was already computed.
"""

from app.graph.nodes.decision_node import decision_node
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.quality import QualityCheck
from app.schemas.statistics import HypothesisTestType, StatResult


def _base_state(**overrides):
    stat = StatResult(
        metric="Conversion Rate",
        test_type=HypothesisTestType.CHI_SQUARE,
        test_name="Chi-square test",
        statistic=123.4,
        selection_reason="Binary metric with sufficient expected cell counts.",
        control="11.87%",
        variant="17.95%",
        delta="+51.2% (rel)",
        p_value=0.0,
        significant=True,
        ci_lower="+5.82pp",
        ci_upper="+6.33pp",
    )
    dataset = DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=294478,
        metric_label="Conversion Rate",
        metric_selection_reason="Selected by the deterministic outcome-column priority — no competing outcome metrics were available in this dataset.",
    )
    state = {
        "dataset_id": "does-not-matter-for-this-test",
        "user_prompt": "Analyze this experiment end-to-end",
        "dataset": dataset,
        "srm_result": None,
        "quality_checks": [QualityCheck(label="Missing Values", passed=True, detail="0% missing")],
        "stat_results": [stat],
        "test_selection": None,
        "power_analysis": None,
        "kb_results": [],
        "funnel_result": None,
    }
    state.update(overrides)
    return state


def test_decision_node_falls_back_to_emergency_report_when_generator_raises(monkeypatch):
    class _BoomGenerator:
        def generate(self, facts):
            raise AttributeError("simulated bug in report assembly")

    import sys
    mod = sys.modules["app.graph.nodes.decision_node"]
    monkeypatch.setattr(mod, "get_report_generator", lambda: _BoomGenerator())

    state = decision_node(_base_state())
    report = state["report"]

    # Deterministic results must survive the failure untouched.
    assert report.stats[0].control == "11.87%"
    assert report.stats[0].variant == "17.95%"
    assert report.quality_checks[0].label == "Missing Values"
    # Never fabricate a positive-looking result out of a narration failure.
    assert "internal error" in report.executive_summary.lower() or "failed" in report.executive_summary.lower()
    assert not any(rec.strip().upper().startswith("SHIP") for rec in report.recommendations)


def test_decision_node_emergency_fallback_is_low_confidence_when_data_also_failed_quality(monkeypatch):
    class _BoomGenerator:
        def generate(self, facts):
            raise RuntimeError("simulated bug")

    import sys
    mod = sys.modules["app.graph.nodes.decision_node"]
    monkeypatch.setattr(mod, "get_report_generator", lambda: _BoomGenerator())

    state = decision_node(_base_state(
        quality_checks=[QualityCheck(label="Sample Ratio Mismatch (SRM)", passed=False, critical=True, detail="p < 0.001")],
    ))
    # srm_passed defaults to True in ReportFacts construction when srm_result is None,
    # so drive the LOW-confidence branch via a critical quality-check failure instead.
    report = state["report"]
    assert report.confidence == "LOW"
    assert not any(rec.strip().upper().startswith("SHIP") for rec in report.recommendations)


def test_decision_node_succeeds_normally_when_generator_does_not_raise():
    """Sanity check: the try/except safety net doesn't change the normal happy path."""
    state = decision_node(_base_state())
    report = state["report"]
    assert report.stats[0].control == "11.87%"
    assert "internal error" not in report.executive_summary.lower()


def test_decision_node_stamps_decision_narrative_on_normal_report():
    """Product improvement: decision_narrative is populated for a normal (non-error) run."""
    state = decision_node(_base_state())
    report = state["report"]
    assert report.decision_narrative is not None
    assert report.decision_narrative.why_this_decision
    assert report.decision_narrative.monitoring.primary_metric == "Conversion Rate"
    assert report.decision_narrative.recommended_next_step


def test_decision_node_stamps_decision_narrative_on_emergency_fallback(monkeypatch):
    """decision_narrative must still be present even when report generation itself raised."""
    class _BoomGenerator:
        def generate(self, facts):
            raise RuntimeError("simulated bug")

    import sys
    mod = sys.modules["app.graph.nodes.decision_node"]
    monkeypatch.setattr(mod, "get_report_generator", lambda: _BoomGenerator())

    state = decision_node(_base_state())
    report = state["report"]
    assert report.decision_narrative is not None
