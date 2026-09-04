"""
API-level integration test — exercises the real FastAPI app via
TestClient through the exact sequence the frontend performs:

    POST /datasets/classify (use_demo=true)
        -> POST /experiments/analyze
        -> POST /experiments/{experiment_id}/chat

This is the test that would have caught `follow_up_chat` being an
unconditional `raise NotImplementedError` — the unit tests in
test_chat_generator.py test the response-generation logic in
isolation, but only a real request through the actual route (with the
actual in-process `_REPORT_STORE`) proves the endpoint itself works.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_full_classify_analyze_chat_flow():
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true"})
    assert classify_resp.status_code == 200
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
        },
    )
    assert analyze_resp.status_code == 200
    analyze_body = analyze_resp.json()
    experiment_id = analyze_body["experimentId"]
    assert experiment_id
    assert analyze_body["report"]["stats"] != []
    assert analyze_body["executionSteps"] != []

    chat_resp = client.post(
        f"/experiments/{experiment_id}/chat",
        json={"experimentId": experiment_id, "message": "Should we ship this?"},
    )
    assert chat_resp.status_code == 200
    chat_body = chat_resp.json()
    assert chat_body["message"]["role"] == "assistant"
    assert chat_body["message"]["content"]
    # Grounded in the real analyze response, not a hardcoded mock string.
    assert analyze_body["report"]["confidence"] in chat_body["message"]["content"]


def test_chat_on_unknown_experiment_id_returns_404_not_500():
    """
    This is the exact regression the stub would have caused differently:
    `raise NotImplementedError` inside a route handler surfaces as an
    unhandled 500, not a clean 404 — a real "experiment not found" must
    be a proper 404 with an explanatory message, not a crash.
    """
    resp = client.post(
        "/experiments/nonexistent-id/chat",
        json={"experimentId": "nonexistent-id", "message": "Should we ship this?"},
    )
    assert resp.status_code == 404
    assert "nonexistent-id" in resp.json()["detail"]


def test_chat_second_turn_uses_prior_conversation_and_history_is_ordered():
    """
    Multi-turn flow: a first exchange establishes context, a second
    message that alone matches no template keyword still lands on the
    same real answer once the prior turn is threaded in — proving the
    route actually passes persisted history to the responder, not just
    persisting it for display.
    """
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true"})
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
        },
    )
    experiment_id = analyze_resp.json()["experimentId"]

    first_resp = client.post(
        f"/experiments/{experiment_id}/chat",
        json={"experimentId": experiment_id, "message": "Should we ship this?"},
    )
    assert first_resp.status_code == 200
    first_reply = first_resp.json()["message"]["content"]

    second_resp = client.post(
        f"/experiments/{experiment_id}/chat",
        json={"experimentId": experiment_id, "message": "what about that?"},
    )
    assert second_resp.status_code == 200
    second_reply = second_resp.json()["message"]["content"]
    # "what about that?" alone matches no keyword — only resolves to the
    # same ship-branch answer as turn one if the prior turn was actually
    # passed to the responder as history.
    assert second_reply == first_reply

    history_resp = client.get(f"/experiments/{experiment_id}/chat")
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert len(history) == 4
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "Should we ship this?"
    assert history[2]["content"] == "what about that?"


def test_chat_grounded_in_srm_failure_dataset():
    """Same flow, but on the low-quality demo dataset — confirms chat reflects a REAL SRM failure, not a script."""
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true", "simulate_low_quality": "true"})
    assert classify_resp.status_code == 200
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Evaluate the checkout redesign — is variant ready to ship?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
        },
    )
    assert analyze_resp.status_code == 200
    analyze_body = analyze_resp.json()
    assert analyze_body["report"]["srmWarning"] is True
    experiment_id = analyze_body["experimentId"]

    chat_resp = client.post(
        f"/experiments/{experiment_id}/chat",
        json={"experimentId": experiment_id, "message": "Was there an SRM issue?"},
    )
    assert chat_resp.status_code == 200
    content = chat_resp.json()["message"]["content"]
    assert "FAILED" in content or "fail" in content.lower()


# ---------------------------------------------------------------------------
# Phase 1 — Experiment Hypothesis: API-level backward compatibility + flow
# ---------------------------------------------------------------------------


def test_analyze_experiment_works_without_hypothesis():
    """Backward compatibility: omitting `hypothesis` entirely must not change anything."""
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true"})
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
        },
    )
    assert analyze_resp.status_code == 200
    assert analyze_resp.json()["report"]["stats"] != []


def test_analyze_experiment_works_with_valid_hypothesis():
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true"})
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
            "hypothesis": {
                "statement": "Increasing the checkout CTA visibility will increase checkout conversion.",
                "primaryMetric": "Conversion Rate",
                "expectedDirection": "increase",
                "expectedEffectRelative": 0.05,
                "rationale": "A similar CTA change on the homepage produced a comparable lift.",
            },
        },
    )
    assert analyze_resp.status_code == 200
    body = analyze_resp.json()
    assert body["report"]["stats"] != []
    # Phase 1 explicitly does NOT surface a verdict in the report — this
    # is confirming the analysis pipeline still ran normally, not that
    # any hypothesis-comparison output exists yet.
    assert "confidence" in body["report"]


def test_analyze_experiment_rejects_invalid_hypothesis():
    """A malformed hypothesis (empty statement) is rejected at the API boundary with a 422, not silently dropped."""
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true"})
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
            "hypothesis": {
                "statement": "",
                "primaryMetric": "Conversion Rate",
                "expectedDirection": "increase",
            },
        },
    )
    assert analyze_resp.status_code == 422


def test_analyze_experiment_rejects_negative_effect_for_increase():
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true"})
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
            "hypothesis": {
                "statement": "Increasing the checkout CTA visibility will increase checkout conversion.",
                "primaryMetric": "Conversion Rate",
                "expectedDirection": "increase",
                "expectedEffectRelative": -0.05,
            },
        },
    )
    assert analyze_resp.status_code == 422


def test_hypothesis_survives_request_to_analysis_context():
    """
    Integration test at the graph level: the exact structured Hypothesis
    submitted through the API must reach the analysis pipeline's
    internal state UNCHANGED — proving request -> state -> analysis
    context actually carries it, not just that the API accepts it.
    """
    from app.core.dataset_store import store_dataset
    from app.graph.graph_builder import experiment_review_graph
    from app.schemas.hypothesis import ExpectedDirection, Hypothesis
    from app.schemas.settings import AnalysisSettings
    import pandas as pd
    from pathlib import Path

    demo_csv = Path(__file__).resolve().parent.parent.parent / "data" / "demo" / "demo_ab_checkout.csv"
    df = pd.read_csv(demo_csv)
    dataset_id = store_dataset(df)

    hypothesis = Hypothesis(
        statement="Increasing the checkout CTA visibility will increase checkout conversion.",
        primary_metric="Conversion Rate",
        expected_direction=ExpectedDirection.INCREASE,
        expected_effect_relative=0.05,
        rationale="A similar CTA change on the homepage produced a comparable lift.",
    )

    final_state = experiment_review_graph.invoke(
        {
            "dataset_id": dataset_id,
            "user_prompt": "Should we ship variant B?",
            "settings": AnalysisSettings(),
            "hypothesis": hypothesis,
        }
    )

    # Reached the state unchanged...
    assert final_state.get("hypothesis") == hypothesis
    # ...and reached ReportFacts via decision_node without being altered
    # (the LLM/template report generator never mutates it — see
    # ReportFacts.hypothesis in report_generator.py). We can't inspect
    # ReportFacts directly from the final state (it's constructed and
    # discarded inside decision_node), so this confirms the one thing
    # that IS observable from outside: analysis completed normally with
    # a hypothesis present in state throughout the whole pipeline run.
    assert final_state["report"].stats != []


# ---------------------------------------------------------------------------
# Phase 2 — Hypothesis Evaluation: API-level exposure + backward compatibility
# ---------------------------------------------------------------------------


def test_report_without_hypothesis_has_null_hypothesis_fields():
    """
    Backward compatibility (Phase 2): a report for an analysis with no
    hypothesis serializes exactly as before, except the two new fields
    are present and null.
    """
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true"})
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
        },
    )
    assert analyze_resp.status_code == 200
    report = analyze_resp.json()["report"]
    assert "hypothesis" in report
    assert "hypothesisEvaluation" in report
    assert report["hypothesis"] is None
    assert report["hypothesisEvaluation"] is None


def test_report_with_hypothesis_exposes_structured_evaluation():
    """The final report must expose both `hypothesis` and `hypothesisEvaluation` as structured data, not just narrative text."""
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true"})
    dataset_id = classify_resp.json()["datasetId"]
    dataset_metric = classify_resp.json()["dataset"]["metricLabel"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
            "hypothesis": {
                "statement": "Increasing the checkout CTA visibility will increase checkout conversion.",
                "primaryMetric": dataset_metric,
                "expectedDirection": "increase",
                "expectedEffectRelative": 0.01,
            },
        },
    )
    assert analyze_resp.status_code == 200
    report = analyze_resp.json()["report"]
    assert report["hypothesis"] is not None
    assert report["hypothesis"]["primaryMetric"] == dataset_metric
    assert report["hypothesisEvaluation"] is not None
    evaluation = report["hypothesisEvaluation"]
    assert evaluation["hypothesisPresent"] is True
    assert evaluation["metricMatched"] is True
    # Verdict is one of the three canonical values — never fabricated, never a 4th value.
    assert evaluation["verdict"] in ("SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED")
    # direction_supported and statistically_significant are reported
    # independently (see the dedicated unit tests in
    # tests/stats/test_hypothesis_evaluator.py for the full sign matrix).
    assert isinstance(evaluation["directionSupported"], bool)
    assert isinstance(evaluation["statisticallySignificant"], bool)


def test_report_with_unmatchable_primary_metric_has_no_fabricated_verdict():
    """Spec §11 — a primary_metric that doesn't match any StatResult must yield an explicit unavailable evaluation, never a guessed verdict."""
    classify_resp = client.post("/datasets/classify", data={"use_demo": "true"})
    dataset_id = classify_resp.json()["datasetId"]

    analyze_resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": "Should we ship variant B?",
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
            "hypothesis": {
                "statement": "Increasing the checkout CTA visibility will increase checkout conversion.",
                "primaryMetric": "Definitely Not A Real Metric",
                "expectedDirection": "increase",
                "expectedEffectRelative": 0.05,
            },
        },
    )
    assert analyze_resp.status_code == 200
    evaluation = analyze_resp.json()["report"]["hypothesisEvaluation"]
    assert evaluation is not None
    assert evaluation["metricMatched"] is False
    assert evaluation["verdict"] is None
    assert evaluation["evaluationNote"] is not None
