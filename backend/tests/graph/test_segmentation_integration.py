"""
Phase 5 integration test — proves segmentation is genuinely wired
through the real graph (experiment_node -> decision_node -> report),
not just correct in isolation (see tests/stats/test_segmentation.py
for the isolated unit tests). Also proves the existing validity/SRM
guardrails still take priority over — and are never overridden by —
segmentation, per the explicit requirement that segmentation must
only ever be supporting evidence.
"""

import numpy as np
import pandas as pd

from app.core.dataset_store import store_dataset
from app.graph.graph_builder import experiment_review_graph
from app.graph.nodes.decision_node import decision_node
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.quality import QualityCheck
from app.schemas.settings import AnalysisSettings
from app.schemas.statistics import HypothesisTestType, StatResult


def _binary_dataset_with_device_effect(n_per_segment=150, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for variant in ["control", "treatment"]:
        for device, lift in [("mobile", 0.25 if variant == "treatment" else 0.0), ("desktop", 0.0)]:
            rate = 0.10 + lift
            metric = rng.binomial(1, rate, n_per_segment)
            for i in range(n_per_segment):
                rows.append(
                    {
                        "user_id": f"{variant}_{device}_{i}",
                        "variant": variant,
                        "device": device,
                        "converted": int(metric[i]),
                    }
                )
    return pd.DataFrame(rows)


def test_segmentation_result_is_present_on_final_report_two_arm():
    df = _binary_dataset_with_device_effect()
    dataset_id = store_dataset(df)
    state = experiment_review_graph.invoke(
        {"dataset_id": dataset_id, "user_prompt": "Should we ship variant B?", "settings": AnalysisSettings()}
    )
    assert state.get("segmentation_result") is not None
    assert state["segmentation_result"].ran is True
    assert "device" in state["segmentation_result"].usable_dimensions

    report = state["report"]
    assert report.segmentation is not None
    assert report.segmentation.ran is True


def test_segmentation_flags_reliable_device_effect_end_to_end():
    df = _binary_dataset_with_device_effect(n_per_segment=300, seed=11)
    dataset_id = store_dataset(df)
    state = experiment_review_graph.invoke(
        {"dataset_id": dataset_id, "user_prompt": "Should we ship variant B?", "settings": AnalysisSettings()}
    )
    device_dim = next(
        d for d in state["segmentation_result"].dimension_results if d.dimension == "device"
    )
    assert "mobile" in device_dim.reliable_segment_values


def test_segmentation_skipped_but_reported_when_no_usable_dimension():
    """A dataset with no categorical dimension beyond user/variant/metric still gets a SegmentationResult, with ran=False."""
    rng = np.random.default_rng(5)
    rows = []
    for variant in ["control", "treatment"]:
        metric = rng.binomial(1, 0.10, 100)
        for i in range(100):
            rows.append({"user_id": f"{variant}_{i}", "variant": variant, "converted": int(metric[i])})
    df = pd.DataFrame(rows)
    dataset_id = store_dataset(df)
    state = experiment_review_graph.invoke(
        {"dataset_id": dataset_id, "user_prompt": "Should we ship variant B?", "settings": AnalysisSettings()}
    )
    assert state["segmentation_result"].ran is False
    assert state["segmentation_result"].reason


def test_segmentation_never_overrides_srm_failure_routing():
    """
    A dataset with a severe SRM failure must still skip straight to
    Decision (route_after_validation) without ever reaching
    experiment_node/segmentation — segmentation must not be able to
    mask or override an invalid experiment. This is the existing Part 1
    guardrail; Phase 5 must not weaken it.
    """
    rng = np.random.default_rng(9)
    # deliberately skewed allocation -> SRM failure (expected ~50/50)
    rows = []
    for i in range(950):
        rows.append({"user_id": f"c_{i}", "variant": "control", "converted": int(rng.binomial(1, 0.1))})
    for i in range(50):
        rows.append({"user_id": f"t_{i}", "variant": "treatment", "converted": int(rng.binomial(1, 0.1))})
    df = pd.DataFrame(rows)
    dataset_id = store_dataset(df)
    state = experiment_review_graph.invoke(
        {"dataset_id": dataset_id, "user_prompt": "Should we ship variant B?", "settings": AnalysisSettings()}
    )
    assert state["srm_result"].passed is False
    # experiment_node (and therefore segmentation) never ran
    assert "segmentation_result" not in state
    assert state["report"].segmentation is None


def test_decision_node_handles_missing_segmentation_result_gracefully():
    """
    Defensive: decision_node must not crash when segmentation_result is
    absent from state (e.g. any future routing path that reaches
    decision_node without going through experiment_node).
    """
    stat = StatResult(
        metric="Conversion Rate",
        test_type=HypothesisTestType.CHI_SQUARE,
        test_name="Chi-square test",
        statistic=1.0,
        selection_reason="test",
        control="10%",
        variant="12%",
        delta="+20% (rel)",
        p_value=0.2,
        significant=False,
        ci_lower="-1pp",
        ci_upper="+3pp",
    )
    dataset = DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=1000,
        metric_label="Conversion Rate",
        metric_selection_reason="test",
    )
    state = {
        "dataset_id": "irrelevant",
        "user_prompt": "test",
        "dataset": dataset,
        "srm_result": None,
        "quality_checks": [QualityCheck(label="Missing Values", passed=True, detail="0% missing")],
        "stat_results": [stat],
        "test_selection": None,
        "power_analysis": None,
        "kb_results": [],
        "funnel_result": None,
        # note: no "segmentation_result" key at all
    }
    result_state = decision_node(state)
    assert result_state["report"].segmentation is None
