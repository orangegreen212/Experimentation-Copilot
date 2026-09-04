"""
Integration test: `decision_node` must attach a `decision_audit` to
the final report on both the normal generator path AND the
emergency-fallback path, since `build_decision_audit_trail` is called
once, after both are resolved.
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


def test_decision_node_attaches_decision_audit_on_normal_path():
    state = decision_node(_base_state())
    report = state["report"]

    assert report.decision_audit is not None
    assert report.decision_audit.decision == report.decision
    assert report.decision_audit.headline == report.decision.value.replace("_", " ")


def test_decision_node_attaches_decision_audit_on_emergency_fallback_path(monkeypatch):
    class _BoomGenerator:
        def generate(self, facts):
            raise RuntimeError("simulated bug in report assembly")

    import sys

    mod = sys.modules["app.graph.nodes.decision_node"]
    monkeypatch.setattr(mod, "get_report_generator", lambda: _BoomGenerator())

    state = decision_node(_base_state())
    report = state["report"]

    # The emergency-fallback report still gets a decision_audit — same
    # attachment point covers both paths (see decision_node.py).
    assert report.decision_audit is not None
    assert report.decision_audit.decision == report.decision
