"""
Tests for app/core/dataset_cleanup_scheduler.py.

Covers the safety properties demanded by the storage-growth audit:
  - recent (< retention window) unused datasets are preserved
  - old, genuinely unreferenced datasets are removed
  - a dataset referenced by a persisted experiment is preserved
    regardless of age
  - a dataset referenced by a definition's data source is preserved
    regardless of age
  - a sweep failure (e.g. DB unavailable) is caught and logged, never
    raised — the scheduler loop and the app itself must survive it
  - repeated sweeps are idempotent (nothing left to delete the second
    time around)
  - two sweeps triggered concurrently never run their delete logic at
    the same time (the module's `_sweep_lock`)
  - `start_background_cleanup()` / `stop_background_cleanup()` have a
    safe, idempotent lifecycle: starting twice reuses the same task,
    stopping actually cancels it, and the disabled-by-config case
    starts nothing at all
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

import app.core.dataset_cleanup_scheduler as scheduler_module
from app.core.config import app_settings
from app.core.dataset_cleanup_scheduler import (
    run_cleanup_sweep,
    start_background_cleanup,
    stop_background_cleanup,
)
from app.core.dataset_store import DatasetModel, _get_session_factory, dataset_exists, store_dataset
from app.main import app

_ANALYZE_SETTINGS = {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0}


def _sample_df(tag: str) -> pd.DataFrame:
    # A unique-content DataFrame per call so store_dataset()'s
    # content-hash dedup never collapses two test datasets into one.
    # Needs a real outcome/metric column (not just user_id/group) or
    # validation_node rejects it before an ExperimentRecord is ever
    # created — see test_old_dataset_referenced_by_an_experiment_is_preserved,
    # which needs a real, persisted analyze run.
    return pd.DataFrame(
        {
            "user_id": [f"{tag}-{i}" for i in range(20)],
            "group": (["control", "treatment"] * 10),
            "converted": ([0, 1] * 10),
        }
    )


def _backdate(dataset_id: str, hours: float) -> None:
    """Rewrites a stored dataset's created_at directly via the ORM —
    the only way to construct an "old" row in a test without actually
    waiting, since store_dataset() always stamps `now()`."""
    session_factory = _get_session_factory()
    with session_factory() as session:
        session.execute(
            update(DatasetModel)
            .where(DatasetModel.dataset_id == dataset_id)
            .values(created_at=datetime.now(timezone.utc) - timedelta(hours=hours))
        )
        session.commit()


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    """Every test gets a clean module-level `_scheduler_task` and
    default retention/interval settings, regardless of what an earlier
    test in this file changed."""
    yield
    scheduler_module._scheduler_task = None


# ---------------------------------------------------------------------------
# run_cleanup_sweep — what gets deleted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_unused_dataset_is_preserved():
    dataset_id = store_dataset(_sample_df("recent"))
    deleted = await run_cleanup_sweep()
    assert dataset_id not in deleted
    assert dataset_exists(dataset_id)


@pytest.mark.asyncio
async def test_old_unused_dataset_is_removed():
    dataset_id = store_dataset(_sample_df("old-unused"))
    _backdate(dataset_id, hours=app_settings.dataset_retention_hours + 1)
    deleted = await run_cleanup_sweep()
    assert dataset_id in deleted
    assert not dataset_exists(dataset_id)


@pytest.mark.asyncio
async def test_old_dataset_referenced_by_an_experiment_is_preserved():
    dataset_id = store_dataset(_sample_df("old-but-referenced"))
    _backdate(dataset_id, hours=app_settings.dataset_retention_hours + 1)

    with TestClient(app) as client:
        resp = client.post(
            "/experiments/analyze",
            json={"datasetId": dataset_id, "prompt": "Should we ship variant B?", "settings": _ANALYZE_SETTINGS},
        )
        assert resp.status_code == 200, resp.text

    deleted = await run_cleanup_sweep()
    assert dataset_id not in deleted
    assert dataset_exists(dataset_id)


@pytest.mark.asyncio
async def test_old_dataset_referenced_by_a_definition_data_source_is_preserved():
    dataset_id = store_dataset(_sample_df("old-but-definition-source"))
    _backdate(dataset_id, hours=app_settings.dataset_retention_hours + 1)

    with TestClient(app) as client:
        resp = client.post(
            "/experiment-definitions",
            json={
                "name": "Checkout redesign",
                "dataSource": {"type": "existing_dataset", "datasetId": dataset_id, "datasetName": "old-but-definition-source"},
            },
        )
        assert resp.status_code == 200, resp.text

    deleted = await run_cleanup_sweep()
    assert dataset_id not in deleted
    assert dataset_exists(dataset_id)


@pytest.mark.asyncio
async def test_in_flight_analyze_dataset_is_protected_from_a_concurrent_sweep():
    """
    Direct regression test for the cleanup-race scenario: an analyze
    call has STARTED using a dataset (registered via
    mark_dataset_in_use) but has not yet persisted an ExperimentRecord
    referencing it — a concurrent sweep must still treat it as
    referenced and must NOT delete it.
    """
    from app.core.dataset_store import mark_dataset_in_use, mark_dataset_no_longer_in_use

    dataset_id = store_dataset(_sample_df("race-protected"))
    _backdate(dataset_id, hours=app_settings.dataset_retention_hours + 1)

    mark_dataset_in_use(dataset_id)
    try:
        deleted = await run_cleanup_sweep()
        assert dataset_id not in deleted
        assert dataset_exists(dataset_id)
    finally:
        mark_dataset_no_longer_in_use(dataset_id)

    # Once no longer in flight (and still unreferenced by any real
    # experiment/definition), it's fair game again.
    deleted = await run_cleanup_sweep()
    assert dataset_id in deleted


@pytest.mark.asyncio
async def test_sweep_is_idempotent():
    dataset_id = store_dataset(_sample_df("idempotent"))
    _backdate(dataset_id, hours=app_settings.dataset_retention_hours + 1)

    first = await run_cleanup_sweep()
    second = await run_cleanup_sweep()

    assert dataset_id in first
    assert second == [], "nothing should be left to delete on the second sweep"


@pytest.mark.asyncio
async def test_sweep_failure_is_caught_and_never_raised(monkeypatch):
    def _boom():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(scheduler_module, "_referenced_dataset_ids", _boom)
    deleted = await run_cleanup_sweep()  # must not raise
    assert deleted == []


@pytest.mark.asyncio
async def test_concurrent_sweeps_never_overlap(monkeypatch):
    """Two callers trigger a sweep at the same time -> the second
    waits for the first instead of running its delete logic
    concurrently (the module's `_sweep_lock`)."""
    concurrent_calls = 0
    max_concurrent = 0

    real_referenced_ids = scheduler_module._referenced_dataset_ids

    def _tracking_referenced_ids():
        nonlocal concurrent_calls, max_concurrent
        concurrent_calls += 1
        max_concurrent = max(max_concurrent, concurrent_calls)
        try:
            return real_referenced_ids()
        finally:
            concurrent_calls -= 1

    monkeypatch.setattr(scheduler_module, "_referenced_dataset_ids", _tracking_referenced_ids)

    await asyncio.gather(run_cleanup_sweep(), run_cleanup_sweep())
    assert max_concurrent == 1, "two sweeps must never execute their delete logic concurrently"


# ---------------------------------------------------------------------------
# start_background_cleanup / stop_background_cleanup — task lifecycle
# ---------------------------------------------------------------------------


def test_start_is_idempotent_and_reuses_the_running_task(monkeypatch):
    monkeypatch.setattr(app_settings, "enable_background_dataset_cleanup", True)
    monkeypatch.setattr(app_settings, "dataset_cleanup_interval_seconds", 3600.0)

    async def _run():
        start_background_cleanup()
        first_task = scheduler_module._scheduler_task
        start_background_cleanup()  # second call must be a no-op
        second_task = scheduler_module._scheduler_task
        assert first_task is second_task
        await stop_background_cleanup()
        assert scheduler_module._scheduler_task is None

    asyncio.run(_run())


def test_disabled_flag_starts_no_task(monkeypatch):
    monkeypatch.setattr(app_settings, "enable_background_dataset_cleanup", False)

    async def _run():
        start_background_cleanup()
        assert scheduler_module._scheduler_task is None

    asyncio.run(_run())


def test_stop_without_start_is_a_safe_no_op():
    async def _run():
        await stop_background_cleanup()  # must not raise

    asyncio.run(_run())


def test_background_loop_actually_ticks_and_survives_a_failing_sweep(monkeypatch):
    """End-to-end lifecycle check: with a very short interval, the loop
    calls run_cleanup_sweep on schedule, and a failing sweep doesn't
    kill the loop (the next tick still happens)."""
    monkeypatch.setattr(app_settings, "enable_background_dataset_cleanup", True)
    monkeypatch.setattr(app_settings, "dataset_cleanup_interval_seconds", 0.01)

    call_count = 0

    async def _flaky_sweep():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated transient failure")
        return []

    monkeypatch.setattr(scheduler_module, "run_cleanup_sweep", _flaky_sweep)

    async def _run():
        start_background_cleanup()
        # Give the loop a handful of ticks; it must survive tick #1's
        # exception and keep going.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if call_count >= 3:
                break
        await stop_background_cleanup()

    asyncio.run(_run())
    assert call_count >= 3, "the loop must keep ticking after an earlier sweep raised"
