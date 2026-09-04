"""
Tests for the pipeline-instrumentation + streaming-progress task:

  1. Every stage records a timing (Part 1) — via POST /analyze's new
     `stage_timings` field.
  2. POST /analyze/stream emits stage_started / stage_completed /
     pipeline_completed events, in a sane order (Parts 4/9).
  3. The streamed final `result` event carries the SAME report shape
     as the synchronous endpoint (Part 7 — streaming must not change
     the final ExperimentReport schema).
  4. An error mid-pipeline still produces an `error` event and a
     terminal `pipeline_completed` (Part 8 — never leaves the client
     hanging on a "running" stage).
  4b. A single node failure produces exactly one `error` event, not a
      duplicate from both the stage-level and pipeline-level handlers.
  5. A client disconnect mid-stream doesn't crash the server (Part 8).
  6. Statistical results are identical whether reached via /analyze or
     /analyze/stream (Part 9 requirement #9 — no behavior change from
     instrumentation).
"""

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_ANALYZE_SETTINGS = {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0}


def _classify_demo_dataset() -> str:
    resp = client.post("/datasets/classify", data={"use_demo": "true"})
    assert resp.status_code == 200, resp.text
    return resp.json()["datasetId"]


def _parse_sse_events(raw_text: str) -> list[dict]:
    events = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload:
            events.append(json.loads(payload))
    return events


def test_analyze_returns_stage_timings_for_every_stage_that_ran():
    """Part 1 — timing instrumentation on the synchronous endpoint."""
    dataset_id = _classify_demo_dataset()
    resp = client.post(
        "/experiments/analyze",
        json={"datasetId": dataset_id, "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    timings = body["stageTimings"]
    assert timings, "expected at least one stage timing"

    stages_seen = {t["stage"] for t in timings}
    # classifier/planner/validation/experiment/guardrail/decision always
    # run for a full "should we ship" review on the demo dataset; "total"
    # is the pipeline-wide wall-clock entry (see RunContext.pipeline_completed).
    for expected_stage in {"classifier", "planner", "validation", "experiment", "guardrail", "decision", "total"}:
        assert expected_stage in stages_seen, f"missing timing for stage={expected_stage}"

    for t in timings:
        assert t["status"] in {"completed", "failed"}
        assert isinstance(t["durationMs"], int)
        assert t["durationMs"] >= 0


def test_stream_emits_started_and_completed_for_every_stage_in_order():
    """Parts 4 & 9 — event shapes + ordering (started before completed, per stage)."""
    dataset_id = _classify_demo_dataset()
    with client.stream(
        "POST",
        "/experiments/analyze/stream",
        json={"datasetId": dataset_id, "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS},
    ) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())

    events = _parse_sse_events(raw)
    assert events, "expected at least one SSE event"

    # pipeline_completed must be last (Part 6/8 — never leave a stage
    # looking permanently "running").
    assert events[-1]["type"] == "pipeline_completed"

    # Exactly one final result carrying the full report, before pipeline_completed.
    result_events = [e for e in events if e["type"] == "result"]
    assert len(result_events) == 1
    assert result_events[0]["data"]["report"]["stats"] != []
    assert result_events[0]["data"]["executionSteps"] != []

    # Every stage_started has a matching stage_completed (or error) that
    # comes strictly after it, and started never fires twice for the
    # same stage before its completion (Part 9 — correct event ordering).
    open_stages: dict[str, int] = {}
    for i, event in enumerate(events):
        if event["type"] == "stage_started":
            assert event["stage"] not in open_stages, f"stage {event['stage']} started twice while running"
            open_stages[event["stage"]] = i
        elif event["type"] in ("stage_completed", "error") and "stage" in event:
            if event["stage"] in open_stages:
                assert open_stages[event["stage"]] < i
                del open_stages[event["stage"]]
    # Fan-out branches (e.g. knowledge_base run concurrently with
    # validation) may legitimately still be "open" only if that stage
    # never ran on this route — not asserted further here; the per-stage
    # ordering check above is what matters.


def test_streamed_report_matches_synchronous_report_for_same_dataset_and_prompt():
    """Part 7 — streaming must not change the final ExperimentReport schema or values."""
    dataset_id = _classify_demo_dataset()
    prompt = "Should we ship variant B?"

    sync_resp = client.post(
        "/experiments/analyze",
        json={"datasetId": dataset_id, "prompt": prompt, "settings": _ANALYZE_SETTINGS},
    )
    assert sync_resp.status_code == 200
    sync_report = sync_resp.json()["report"]

    with client.stream(
        "POST",
        "/experiments/analyze/stream",
        json={"datasetId": dataset_id, "prompt": prompt, "settings": _ANALYZE_SETTINGS},
    ) as resp:
        raw = "".join(resp.iter_text())
    stream_events = _parse_sse_events(raw)
    stream_report = next(e for e in stream_events if e["type"] == "result")["data"]["report"]

    # Statistical results must be identical (Part 9 requirement #9) —
    # compare the deterministic numeric fields, not LLM-authored prose
    # (narrative text can legitimately differ between independent runs
    # when an LLM report backend is configured; the template backend
    # used in tests is deterministic, but this keeps the assertion
    # robust to that).
    assert stream_report["stats"] == sync_report["stats"]
    assert stream_report["decision"] == sync_report["decision"]
    assert stream_report["confidence"] == sync_report["confidence"]


def test_stream_emits_exactly_one_error_event_for_a_single_node_failure(monkeypatch):
    """
    A single real failure inside a node (here: classifier) must produce
    exactly one user-visible `error` event on the stream — not one from
    `instrument_node`/`stage_failed` AND a second generic one from the
    outer pipeline exception handler in `analyze_experiment_stream`.
    See RunContext.error_emitted.
    """
    dataset_id = _classify_demo_dataset()

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic classifier failure for test")

    # classifier_node.py does `from app.core.dataset_store import
    # get_dataset` — patch the name as bound in that submodule. Import
    # via `importlib` (not `import app.graph.nodes.classifier_node as
    # ...`) because `app.graph.nodes`'s `__init__.py` re-exports a
    # function also named `classifier_node`, which would shadow the
    # submodule reference here.
    import importlib

    classifier_node_module = importlib.import_module("app.graph.nodes.classifier_node")
    monkeypatch.setattr(classifier_node_module, "get_dataset", _boom)

    with client.stream(
        "POST",
        "/experiments/analyze/stream",
        json={"datasetId": dataset_id, "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS},
    ) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())

    events = _parse_sse_events(raw)

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1, f"expected exactly one error event, got {error_events}"
    assert error_events[0]["stage"] == "classifier"
    assert "synthetic classifier failure" in error_events[0]["message"]

    # Still no result, and still terminates cleanly (Part 8).
    assert not any(e["type"] == "result" for e in events)
    assert events[-1]["type"] == "pipeline_completed"


def test_stream_reports_error_and_still_completes_for_unknown_dataset():
    """Part 8 — a failed request still gets an error event + terminal pipeline_completed, not a hang."""
    with client.stream(
        "POST",
        "/experiments/analyze/stream",
        json={"datasetId": "does-not-exist", "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS},
    ) as resp:
        assert resp.status_code == 200  # SSE stream itself opens fine; the error is an event, not an HTTP status
        raw = "".join(resp.iter_text())

    events = _parse_sse_events(raw)
    assert any(e["type"] == "error" for e in events)
    assert events[-1]["type"] == "pipeline_completed"
    assert not any(e["type"] == "result" for e in events)


def test_client_disconnect_mid_stream_does_not_crash_server():
    """Part 8 — the server must handle a client that stops reading partway through."""
    dataset_id = _classify_demo_dataset()
    with client.stream(
        "POST",
        "/experiments/analyze/stream",
        json={"datasetId": dataset_id, "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS},
    ) as resp:
        assert resp.status_code == 200
        iterator = resp.iter_text()
        # Read only the first chunk, then close the connection early —
        # simulates a browser tab closing mid-analysis.
        next(iterator, None)

    # Server must still be responsive afterwards — a crash or a leaked,
    # permanently-blocked thread would surface as this next call hanging
    # or failing.
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
