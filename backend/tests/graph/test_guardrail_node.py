"""
guardrail_node.py — integration tests (guardrail root-cause fix).

Exercises the exact four end-to-end scenarios required by the design
audit: no guardrails requested, requested-but-absent, partially
available, and fully evaluated — verifying `state["guardrail_results"]`
is ACTUALLY populated (the root cause: previously nothing ever wrote
this key at all).
"""

import sys

import pandas as pd
import pytest

from app.graph.nodes.guardrail_node import derive_guardrail_request_state, guardrail_node
from app.schemas.guardrails import GuardrailRequestState
from app.schemas.settings import AnalysisSettings
from app.stats.dataset_classifier import ExperimentColumns


def _two_arm_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [f"U{i}" for i in range(40)],
            "group": ["control"] * 20 + ["treatment"] * 20,
            "converted": [0, 1] * 20,
            "purchase_amount": [10.0 + i * 0.1 for i in range(40)],
            "bounce_rate": [0.5 - i * 0.001 for i in range(40)],
        }
    )


def _state(df: pd.DataFrame, guardrail_metrics: list[str], monkeypatch, primary_col="converted"):
    columns = ExperimentColumns(
        user_col="user_id",
        variant_col="group",
        metric_col=primary_col,
        metric_type=None,  # not read by guardrail_node
    )
    settings = AnalysisSettings(guardrail_metrics=guardrail_metrics)

    # `sys.modules[...]`, not `import app.graph.nodes.guardrail_node as
    # mod`: `app/graph/nodes/__init__.py` does
    # `from .guardrail_node import guardrail_node`, which rebinds the
    # `guardrail_node` ATTRIBUTE on the `app.graph.nodes` package to
    # the function — shadowing the submodule reference that `import
    # ... as mod` would otherwise resolve via attribute access. Going
    # through `sys.modules` reads the actual module object directly
    # and is unaffected by that shadowing (same fix already used in
    # tests/graph/test_decision_node.py for the identical situation).
    mod = sys.modules["app.graph.nodes.guardrail_node"]

    monkeypatch.setattr(mod, "get_dataset", lambda dataset_id: df)

    return {
        "dataset_id": "does-not-matter",
        "settings": settings,
        "dataset": object(),  # only checked for `is not None` in guardrail_node
        "experiment_columns": columns,
    }


class TestDeriveGuardrailRequestState:
    def test_no_resolutions_is_not_specified(self):
        assert derive_guardrail_request_state([]) == GuardrailRequestState.NOT_SPECIFIED

    def test_all_unresolved_is_requested_not_found(self):
        from app.schemas.guardrails import GuardrailResolution

        resolutions = [GuardrailResolution(requested_name="Revenue", resolved=False)]
        assert derive_guardrail_request_state(resolutions) == GuardrailRequestState.REQUESTED_NOT_FOUND

    def test_mixed_is_partially_available(self):
        from app.schemas.guardrails import GuardrailResolution

        resolutions = [
            GuardrailResolution(requested_name="Purchase Amount", resolved=True, resolved_metric_label="Purchase Amount"),
            GuardrailResolution(requested_name="Revenue", resolved=False),
        ]
        assert derive_guardrail_request_state(resolutions) == GuardrailRequestState.PARTIALLY_AVAILABLE

    def test_all_resolved_is_available(self):
        from app.schemas.guardrails import GuardrailResolution

        resolutions = [
            GuardrailResolution(requested_name="Purchase Amount", resolved=True, resolved_metric_label="Purchase Amount"),
        ]
        assert derive_guardrail_request_state(resolutions) == GuardrailRequestState.AVAILABLE


class TestGuardrailNodeEndToEnd:
    def test_scenario_a_no_guardrails_requested(self, monkeypatch):
        state = _state(_two_arm_dataframe(), [], monkeypatch)
        result = guardrail_node(state)

        assert result["guardrail_request_state"] == GuardrailRequestState.NOT_SPECIFIED
        assert result["guardrail_results"] == []
        assert result["requested_guardrails"] == []

    def test_scenario_b_requested_but_absent(self, monkeypatch):
        state = _state(_two_arm_dataframe(), ["Revenue", "Bounce Rate Requested Typo"], monkeypatch)
        result = guardrail_node(state)

        assert result["guardrail_request_state"] == GuardrailRequestState.REQUESTED_NOT_FOUND
        assert result["guardrail_results"] == []
        assert {r.requested_name for r in result["guardrail_resolutions"]} == {"Revenue", "Bounce Rate Requested Typo"}
        assert all(not r.resolved for r in result["guardrail_resolutions"])

    def test_scenario_c_partially_available(self, monkeypatch):
        state = _state(_two_arm_dataframe(), ["Purchase Amount", "Revenue"], monkeypatch)
        result = guardrail_node(state)

        assert result["guardrail_request_state"] == GuardrailRequestState.PARTIALLY_AVAILABLE
        # Only the resolved one gets a real StatResult.
        assert len(result["guardrail_results"]) == 1
        assert result["guardrail_results"][0].metric == "Purchase Amount"

    def test_scenario_d_multiple_available_guardrails_evaluated(self, monkeypatch):
        state = _state(_two_arm_dataframe(), ["Purchase Amount", "Bounce Rate"], monkeypatch)
        result = guardrail_node(state)

        assert result["guardrail_request_state"] == GuardrailRequestState.AVAILABLE
        assert len(result["guardrail_results"]) == 2
        metrics = {g.metric for g in result["guardrail_results"]}
        assert metrics == {"Purchase Amount", "Bounce Rate"}

        # Directionality: Bounce Rate is lower-is-better, Purchase Amount is higher-is-better.
        by_metric = {g.metric: g for g in result["guardrail_results"]}
        assert by_metric["Bounce Rate"].higher_is_better is False
        assert by_metric["Purchase Amount"].higher_is_better is True

    def test_never_lets_primary_metric_be_its_own_guardrail(self, monkeypatch):
        state = _state(_two_arm_dataframe(), ["Conversion Rate"], monkeypatch)
        result = guardrail_node(state)
        assert result["guardrail_request_state"] == GuardrailRequestState.REQUESTED_NOT_FOUND
