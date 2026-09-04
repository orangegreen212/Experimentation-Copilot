"""
Phase 8 — Production Readiness, Observability & Run Trace.

Exercises the real FastAPI app (same pattern as test_experiments_api.py)
to prove the execution trace / run metadata reflect what the graph
ACTUALLY did, never a second, duplicated execution.
"""

from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def _classify_demo(simulate_low_quality: bool = False) -> str:
    resp = client.post(
        "/datasets/classify",
        data={"use_demo": "true", "simulate_low_quality": str(simulate_low_quality).lower()},
    )
    assert resp.status_code == 200
    return resp.json()["datasetId"]


def _analyze(dataset_id: str, prompt: str) -> dict:
    resp = client.post(
        "/experiments/analyze",
        json={
            "datasetId": dataset_id,
            "prompt": prompt,
            "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
        },
    )
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# A. Successful run — execution trace contains expected major stages
# ---------------------------------------------------------------------------


def test_successful_run_execution_trace_has_expected_stages():
    dataset_id = _classify_demo()
    body = _analyze(dataset_id, "Should we ship variant B?")

    steps = body["executionSteps"]
    step_ids = [s["id"] for s in steps]
    assert step_ids[0] == "classifier"
    assert "planner" in step_ids
    assert "validation" in step_ids
    assert "experiment" in step_ids
    assert step_ids[-1] == "decision"

    # Every step carries a real status now, not just detail text.
    for step in steps:
        assert step["status"] in {"SUCCESS", "SKIPPED", "WARNING", "FAILED"}
    assert all(s["status"] == "SUCCESS" for s in steps if s["id"] in {"classifier", "validation", "experiment", "decision"})


def test_successful_run_has_run_metadata():
    dataset_id = _classify_demo()
    body = _analyze(dataset_id, "Should we ship variant B?")

    run_metadata = body["report"]["runMetadata"]
    assert run_metadata is not None
    assert run_metadata["runId"]
    assert run_metadata["timestamp"]
    assert run_metadata["userCount"] > 0
    assert run_metadata["variantCount"] >= 2
    assert run_metadata["analysisMode"]
    assert run_metadata["executionStatus"] in {"SUCCESS", "SKIPPED", "WARNING", "FAILED"}
    # No raw dataset / sensitive user info anywhere in the payload.
    assert "rows" not in run_metadata
    assert "dataframe" not in run_metadata


# ---------------------------------------------------------------------------
# B. Segmentation — represented, and skips are not reported as FAILED
# ---------------------------------------------------------------------------


def test_segmentation_step_present_and_not_failed():
    dataset_id = _classify_demo()
    body = _analyze(dataset_id, "Should we ship variant B?")

    steps = {s["id"]: s for s in body["executionSteps"]}
    assert "segmentation" in steps
    assert steps["segmentation"]["status"] != "FAILED"
    assert steps["segmentation"]["status"] in {"SUCCESS", "SKIPPED"}


def test_successful_run_is_not_reported_as_skipped_when_only_segmentation_was_skipped():
    """
    Regression test for the executionStatus rollup bug: the demo
    dataset legitimately has no usable segmentation dimensions (a
    normal, healthy outcome, not a problem), and report generation
    succeeds outright — the top-level Run Status must read SUCCESS,
    not SKIPPED, so it never looks like the whole analysis was
    skipped. See tests/api/test_run_metadata_execution_status.py for
    the full unit-level coverage of the rollup rule itself.
    """
    dataset_id = _classify_demo()
    body = _analyze(dataset_id, "Should we ship variant B?")

    steps = {s["id"]: s for s in body["executionSteps"]}
    assert steps["segmentation"]["status"] == "SKIPPED"
    assert body["report"].get("reportFallbackReason") in (None, "")
    assert body["report"]["runMetadata"]["executionStatus"] == "SUCCESS"


# ---------------------------------------------------------------------------
# C. Knowledge base empty result vs failure
# ---------------------------------------------------------------------------


def test_knowledge_base_empty_result_is_not_an_error():
    """A conceptual question with a real (non-matching) query should show
    'no relevant reference' with SUCCESS status, never a failure."""
    dataset_id = _classify_demo()
    body = _analyze(dataset_id, "What is the airspeed velocity of an unladen swallow?")

    steps = {s["id"]: s for s in body["executionSteps"]}
    if "knowledge_base" in steps:
        kb_step = steps["knowledge_base"]
        if "No sufficiently relevant evidence" in kb_step["detail"]:
            assert kb_step["status"] == "SUCCESS"


def test_knowledge_base_retrieval_failure_is_a_warning_not_silent():
    """When the retriever itself blows up, this must show up as a
    WARNING with a clear reason — never silently look like an empty
    (but legitimate) result, and never crash the request."""
    dataset_id = _classify_demo()

    with patch("app.graph.nodes.knowledge_base_node.get_retriever", side_effect=RuntimeError("index not built")):
        body = _analyze(dataset_id, "What is CUPED?")

    steps = {s["id"]: s for s in body["executionSteps"]}
    assert "knowledge_base" in steps
    assert steps["knowledge_base"]["status"] == "WARNING"
    assert "index not built" in steps["knowledge_base"]["detail"]
    assert "Traceback" not in steps["knowledge_base"]["detail"]


# ---------------------------------------------------------------------------
# D. LLM planner fallback is visible in the trace
# ---------------------------------------------------------------------------


def test_llm_planner_fallback_is_visible_when_it_actually_occurs():
    dataset_id = _classify_demo()

    with patch("app.core.config.app_settings.planner_backend", "llm"), patch(
        "app.llm.client.get_llm", side_effect=RuntimeError("no api key")
    ):
        body = _analyze(dataset_id, "Should we ship variant B?")

    steps = {s["id"]: s for s in body["executionSteps"]}
    assert steps["planner"]["status"] == "WARNING"
    assert "fell back to keyword-based planning" in steps["planner"]["detail"]


# ---------------------------------------------------------------------------
# E. Critical failure remains visible — never reported as a clean success
# ---------------------------------------------------------------------------


def test_srm_failure_is_visible_as_failed_validation_step():
    dataset_id = _classify_demo(simulate_low_quality=True)
    body = _analyze(dataset_id, "Evaluate the checkout redesign — is variant ready to ship?")

    steps = {s["id"]: s for s in body["executionSteps"]}
    assert body["report"]["srmWarning"] is True
    assert steps["validation"]["status"] == "FAILED"
    # The pipeline still completed and produced a report — not a 500.
    assert steps["decision"]["status"] in {"SUCCESS", "WARNING"}
    assert body["report"]["decision"] in {"INVALID", "INCONCLUSIVE", "NO_GO"}


# ---------------------------------------------------------------------------
# F. Run metadata / dataset facts serialize correctly
# ---------------------------------------------------------------------------


def test_run_metadata_dataset_facts_match_report_dataset():
    dataset_id = _classify_demo()
    body = _analyze(dataset_id, "Should we ship variant B?")

    run_metadata = body["report"]["runMetadata"]
    stats = body["report"]["stats"]
    assert run_metadata["variantCount"] >= 2
    assert run_metadata["userCount"] > 0
    if stats:
        assert run_metadata["primaryMetric"]


# ---------------------------------------------------------------------------
# G. API serialization — existing response fields untouched
# ---------------------------------------------------------------------------


def test_existing_response_fields_unchanged():
    dataset_id = _classify_demo()
    body = _analyze(dataset_id, "Should we ship variant B?")

    assert body["experimentId"]
    assert body["report"]["stats"] != []
    assert body["executionSteps"] != []
    assert "relatedExperiments" in body


# ---------------------------------------------------------------------------
# H. Backward compatibility — reopening a persisted experiment still works
# ---------------------------------------------------------------------------


def test_reopened_experiment_still_has_execution_steps_and_report():
    dataset_id = _classify_demo()
    body = _analyze(dataset_id, "Should we ship variant B?")
    experiment_id = body["experimentId"]

    detail_resp = client.get(f"/experiments/{experiment_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["executionSteps"] != []
    assert detail["report"]["runMetadata"] is not None


# ---------------------------------------------------------------------------
# System / health visibility
# ---------------------------------------------------------------------------


def test_system_info_exposes_safe_operational_fields():
    resp = client.get("/system/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "knowledgeBaseAvailable" in body
    assert isinstance(body["knowledgeBaseAvailable"], bool)
    assert body["availableModelsCount"] > 0
    # Nothing secret leaks.
    blob = str(body).lower()
    for forbidden in ("api_key", "apikey", "secret", "password", "token=", "connection_string"):
        assert forbidden not in blob
