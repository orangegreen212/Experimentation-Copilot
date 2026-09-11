"""
API-level tests for duplicate-execution suppression (see
app/core/execution_dedup.py and tests/core/test_execution_dedup.py,
which cover the same guarantees at the unit level).

These tests drive the ACTUAL FastAPI routes (`/experiments/analyze`,
`/experiments/analyze/stream`) with genuinely concurrent HTTP requests
(via `httpx.AsyncClient` + `ASGITransport`, not `TestClient`, which is
synchronous and can't produce real overlap) and patch
`experiment_review_graph.invoke` to block on a `threading.Event` so
the test controls exactly when the "in-flight" window opens and
closes — the same technique used to test the dedup primitive itself,
now exercised through real request/response cycles, real
`ExperimentStore` persistence, and both entry points at once.

Covers the scenarios named in the storage/reliability audit:
  - user double-clicks Analyze
  - a second identical request arrives while the first is still running
  - /analyze and /analyze/stream given the same logical request share
    one execution
  - two DIFFERENT requests (different prompt) are never blocked by
    each other
  - a request that arrives AFTER the identical one already finished is
    a fresh, independent execution (not silently cached forever)
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
from httpx import ASGITransport

from app.core.experiment_store import get_experiment_store
from app.graph.graph_builder import experiment_review_graph
from app.main import app

_ANALYZE_SETTINGS = {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0}


def _classify_demo_dataset_sync() -> str:
    """Uses the existing synchronous TestClient — classification is
    fast and not itself under test here."""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        resp = client.post("/datasets/classify", data={"use_demo": "true"})
        assert resp.status_code == 200, resp.text
        return resp.json()["datasetId"]


class _GatedInvoke:
    """
    Replaces `experiment_review_graph.invoke` with a version that
    blocks on a `threading.Event` until the test releases it, so the
    test can deterministically create a window where a request is
    "still running" before letting it (and only it) actually finish.
    Runs on a real OS thread under the hood (the route wraps `.invoke`
    in `asyncio.to_thread`), so a plain `threading.Event` — not an
    `asyncio.Event` — is the correct primitive here.
    """

    def __init__(self, real_invoke):
        self._real_invoke = real_invoke
        self.call_count = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.started.set()
        self.release.wait(timeout=5)
        return self._real_invoke(*args, **kwargs)


@pytest.fixture
def gated_invoke(monkeypatch):
    gate = _GatedInvoke(experiment_review_graph.invoke)
    monkeypatch.setattr(experiment_review_graph, "invoke", gate)
    return gate


@pytest.mark.asyncio
async def test_concurrent_identical_analyze_requests_share_one_execution(gated_invoke):
    """Double-click / frontend-retry scenario: two byte-identical
    POST /experiments/analyze requests overlap in time -> the graph
    runs exactly once and both HTTP responses carry the SAME
    experiment_id (one ExperimentStore row, not two)."""
    dataset_id = _classify_demo_dataset_sync()
    payload = {"datasetId": dataset_id, "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS}

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        task_a = asyncio.create_task(client.post("/experiments/analyze", json=payload))
        # Wait until the leader's graph.invoke() has actually started
        # before firing the "duplicate" request, so it reliably joins
        # the in-flight execution rather than racing to start its own.
        await asyncio.to_thread(gated_invoke.started.wait, 5)
        task_b = asyncio.create_task(client.post("/experiments/analyze", json=payload))
        await asyncio.sleep(0.05)
        gated_invoke.release.set()

        resp_a, resp_b = await asyncio.gather(task_a, task_b)

    assert resp_a.status_code == 200, resp_a.text
    assert resp_b.status_code == 200, resp_b.text
    assert gated_invoke.call_count == 1, "the pipeline must run exactly once for two identical concurrent requests"
    assert resp_a.json()["experimentId"] == resp_b.json()["experimentId"]

    # Exactly one persisted row for this pair of requests.
    matching = [
        e for e in get_experiment_store().list() if e.experiment_id == resp_a.json()["experimentId"]
    ]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_analyze_and_analyze_stream_share_one_execution_for_identical_request(gated_invoke):
    """The two ENTRY POINTS (sync /analyze and streaming
    /analyze/stream) must share the dedup registry — a user hitting
    Analyze twice must be caught even if, say, a frontend retry uses a
    different code path than the original click."""
    dataset_id = _classify_demo_dataset_sync()
    payload = {"datasetId": dataset_id, "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS}

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        sync_task = asyncio.create_task(client.post("/experiments/analyze", json=payload))
        await asyncio.to_thread(gated_invoke.started.wait, 5)

        async def _read_stream_result() -> dict:
            async with client.stream("POST", "/experiments/analyze/stream", json=payload) as resp:
                assert resp.status_code == 200
                events = []
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        import json as _json

                        events.append(_json.loads(line[len("data:"):].strip()))
                return events

        stream_task = asyncio.create_task(_read_stream_result())
        await asyncio.sleep(0.05)
        gated_invoke.release.set()

        sync_resp, stream_events = await asyncio.gather(sync_task, stream_task)

    assert gated_invoke.call_count == 1
    assert sync_resp.status_code == 200, sync_resp.text

    result_events = [e for e in stream_events if e["type"] == "result"]
    assert len(result_events) == 1
    # The streaming caller was the follower here (it joined after the
    # sync request's graph.invoke() had already started) — it must
    # see the duplicate-notice event and the SAME experiment_id.
    assert any(e["type"] == "duplicate_of_running_analysis" for e in stream_events)
    assert result_events[0]["data"]["experimentId"] == sync_resp.json()["experimentId"]


@pytest.mark.asyncio
async def test_concurrent_different_requests_are_not_blocked_by_each_other(gated_invoke):
    """Two DIFFERENT analyses (different prompt => different dedup
    key) must both actually run — dedup must never behave like a
    global lock across unrelated experiments."""
    dataset_id = _classify_demo_dataset_sync()
    payload_a = {"datasetId": dataset_id, "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS}
    payload_b = {"datasetId": dataset_id, "prompt": "Is there a guardrail regression?", "settings": _ANALYZE_SETTINGS}

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        task_a = asyncio.create_task(client.post("/experiments/analyze", json=payload_a))
        task_b = asyncio.create_task(client.post("/experiments/analyze", json=payload_b))
        # Both requests must be able to enter graph.invoke() concurrently
        # -- neither is gated behind the other's key.
        await asyncio.sleep(0.2)
        assert gated_invoke.call_count == 2, "different requests must not block on the same dedup key"
        gated_invoke.release.set()

        resp_a, resp_b = await asyncio.gather(task_a, task_b)

    assert resp_a.status_code == 200 and resp_b.status_code == 200
    assert resp_a.json()["experimentId"] != resp_b.json()["experimentId"]


@pytest.mark.asyncio
async def test_concurrency_limit_returns_a_clean_503_not_a_hang(gated_invoke, monkeypatch):
    """
    Concurrency-audit finding: with `max_concurrent_analyses` DIFFERENT
    analyses already running, a further one must fail fast with a
    clean 503 — never queue indefinitely (which, combined with a
    platform request timeout, is indistinguishable from a hang) and
    never silently degrade every other request's latency.
    """
    from app.core.config import app_settings

    monkeypatch.setattr(app_settings, "max_concurrent_analyses", 1)
    dataset_id = _classify_demo_dataset_sync()
    payload_a = {"datasetId": dataset_id, "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS}
    payload_b = {"datasetId": dataset_id, "prompt": "Is there a guardrail regression?", "settings": _ANALYZE_SETTINGS}

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        task_a = asyncio.create_task(client.post("/experiments/analyze", json=payload_a))
        await asyncio.to_thread(gated_invoke.started.wait, 5)  # slot A is now held

        resp_b = await client.post("/experiments/analyze", json=payload_b)
        assert resp_b.status_code == 503, resp_b.text

        gated_invoke.release.set()
        resp_a = await task_a

    assert resp_a.status_code == 200, resp_a.text


@pytest.mark.asyncio
async def test_concurrent_requests_differing_only_in_settings_are_never_merged(gated_invoke):
    """
    The audit's 'most dangerous possible regression' check: dataset X
    + prompt Y with cuped=False vs cuped=True must never be
    deduplicated, even when they arrive concurrently — cuped changes
    the actual statistical computation (variance reduction), so
    silently reusing one's result for the other would silently
    corrupt the analysis the user asked for.
    """
    dataset_id = _classify_demo_dataset_sync()
    payload_a = {
        "datasetId": dataset_id, "prompt": "Should we ship variant B?",
        "settings": {**_ANALYZE_SETTINGS, "cuped": False},
    }
    payload_b = {
        "datasetId": dataset_id, "prompt": "Should we ship variant B?",
        "settings": {**_ANALYZE_SETTINGS, "cuped": True},
    }

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        task_a = asyncio.create_task(client.post("/experiments/analyze", json=payload_a))
        task_b = asyncio.create_task(client.post("/experiments/analyze", json=payload_b))
        await asyncio.sleep(0.2)
        assert gated_invoke.call_count == 2, "different settings must never share one execution"
        gated_invoke.release.set()

        resp_a, resp_b = await asyncio.gather(task_a, task_b)

    assert resp_a.status_code == 200 and resp_b.status_code == 200
    assert resp_a.json()["experimentId"] != resp_b.json()["experimentId"]


@pytest.mark.asyncio
async def test_sequential_identical_requests_after_completion_each_run_fresh(gated_invoke):
    """
    A request that arrives AFTER an identical one has already fully
    completed is a deliberate new run, not a duplicate — this product
    persists one ExperimentRecord per real analyze call (History
    supports multiple runs of the same definition/dataset). Dedup only
    suppresses genuinely OVERLAPPING duplicate work, never caches a
    result indefinitely.
    """
    dataset_id = _classify_demo_dataset_sync()
    payload = {"datasetId": dataset_id, "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS}

    gated_invoke.release.set()  # no gating needed — run fully sequentially

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        resp_1 = await client.post("/experiments/analyze", json=payload)
        resp_2 = await client.post("/experiments/analyze", json=payload)

    assert resp_1.status_code == 200 and resp_2.status_code == 200
    assert gated_invoke.call_count == 2, "a call after the previous one fully completed must run fresh"
    assert resp_1.json()["experimentId"] != resp_2.json()["experimentId"]
