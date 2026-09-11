"""
Periodic background sweep for orphaned dataset rows.

WHY THIS EXISTS: `POST /datasets/cleanup` (routes_datasets.py) already
deletes every dataset row not referenced by an experiment or a
definition's data source — but nothing called it automatically (see
that route's own docstring, which candidly says so). Every abandoned
upload, superseded re-selection in the Data Source picker, or dataset
previewed but never analyzed stayed in the `datasets` table forever.
This module is what actually calls it, periodically, without a human
in the loop.

SAFETY PROPERTIES (each one demanded by the storage-growth audit):

  - Never deletes anything less than `dataset_retention_hours` old
    (default 24h) — see `dataset_store.delete_unused_datasets`'s
    `older_than` param. A dataset upload that hasn't been linked to an
    experiment/definition yet because the analyze call is still in
    flight, or the user simply hasn't clicked Analyze yet, is never at
    risk; no legitimate flow between an upload and either outcome
    takes anywhere near 24h.
  - Never deletes a row referenced by `experiments.dataset_id` (any
    persisted analysis run, regardless of age) or by an
    `experiment_definitions.data_source.dataset_id` (a definition's
    currently-connected data source) — computed as a fresh snapshot on
    every sweep, exactly as `/datasets/cleanup` already does.
  - A single `asyncio.Lock` ensures at most one sweep runs at a time
    in this process — a slow sweep (e.g. Postgres under load) can
    never overlap with the next scheduled tick, which could otherwise
    race on the same rows.
  - A sweep that raises (DB unavailable, transient network error,
    anything) is caught, logged, and the loop keeps ticking on
    schedule — one failed sweep must never kill the scheduler task
    (and, since it isn't awaited by any request, must never surface as
    an unhandled-exception crash of the app either).
  - Started and stopped via FastAPI's `lifespan` context manager (see
    main.py), not a bare fire-and-forget task created at import time:
    exactly one task per process lifetime, cleanly cancelled on
    shutdown, so a dev-server reload or a test importing `app.main`
    repeatedly never accumulates orphaned scheduler tasks.

DEPLOYMENT SCOPE — READ BEFORE ENABLING IN A NEW ENVIRONMENT:

This scheduler is in-process, in-memory state, exactly like
`core/rate_limit.py`'s counters and `core/dataset_store.py`'s
request-scoped cache — and it inherits the exact same limitation
those two already document. It is safe and effective under EXACTLY
the deployment this project's `render.yaml` actually configures: one
long-lived Uvicorn worker (`uvicorn app.main:app`, no `--workers`
flag), one process, one event loop, for the life of the dyno.

It is NOT safe to rely on if either of these change:

  1. **Multiple Uvicorn workers on Render** (`--workers N`): each
     worker is a separate process with its own event loop, so N
     independent sweep loops would start, each unaware of the others.
     They wouldn't corrupt data (the delete itself is a plain SQL
     DELETE against the shared Postgres database and is safe to run
     redundantly — deleting an already-deleted id is just a no-op),
     but you'd get up to N redundant sweeps every interval for no
     benefit, and the "only one sweep at a time" guarantee above only
     holds PER PROCESS, not cluster-wide.
  2. **The Vercel serverless deployment** (`api/index.py` /
     `vercel.json` — this same FastAPI app can be deployed there
     instead of/alongside Render). A serverless function has no
     persistent process between invocations: Vercel may run zero,
     one, or many concurrent container instances of this app
     depending on traffic, freezing and cold-starting them
     unpredictably. An `asyncio` loop started in one invocation's
     lifespan is NOT guaranteed to keep ticking once that invocation
     finishes handling its request — there is no guarantee it ever
     fires again, and no way for it to coordinate with any other
     concurrently-running instance. On Vercel, **do not depend on
     this scheduler at all** — call `POST /datasets/cleanup` from an
     external scheduler instead (a Vercel Cron Job, or any cron
     hitting the deployed URL). The route existed for exactly this
     manual/external-trigger use case before this module existed, and
     remains the deployment-agnostic mechanism.

`AppSettings.enable_background_dataset_cleanup` defaults to True,
which is correct for the Render single-worker deployment this project
actually ships with today. Set it to False (env var
`ENABLE_BACKGROUND_DATASET_CLEANUP=false`) if you deploy with multiple
workers or on Vercel, and drive `/datasets/cleanup` externally instead
— running this loop there isn't dangerous, just redundant/unreliable,
so this defaults on rather than silently doing nothing by default.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from app.core.config import app_settings
from app.core.dataset_store import delete_unused_datasets, in_flight_dataset_ids
from app.core.experiment_definition_store import get_experiment_definition_store
from app.core.experiment_store import get_experiment_store

log = logging.getLogger("core.dataset_cleanup_scheduler")

# Guarantees at most one sweep runs at a time within this process,
# regardless of how `run_cleanup_sweep` is invoked (the periodic loop
# below, or a test calling it directly/concurrently).
_sweep_lock = asyncio.Lock()


def _referenced_dataset_ids() -> set[str]:
    """Same computation as `routes_datasets.cleanup_unused_datasets` —
    factored out so both the manual route and this automatic sweep
    stay in exact agreement about what "referenced" means — PLUS every
    dataset_id currently registered as in-use by an in-progress
    analyze call (`core/dataset_store.in_flight_dataset_ids()`), which
    the manual route does NOT check: a human explicitly calling
    `/datasets/cleanup` already knows what they're doing, but this
    automatic sweep must never race a request that started before this
    sweep's snapshot was taken — see `core/dataset_store.py`'s
    `mark_dataset_in_use` docstring for the exact scenario this
    closes."""
    referenced_ids = {row.dataset_id for row in get_experiment_store().list()}
    referenced_ids |= {
        d.data_source.dataset_id
        for d in get_experiment_definition_store().list()
        if d.data_source is not None and d.data_source.dataset_id
    }
    referenced_ids |= in_flight_dataset_ids()
    return referenced_ids


async def run_cleanup_sweep() -> list[str]:
    """
    Runs one sweep: delete unreferenced dataset rows older than
    `AppSettings.dataset_retention_hours`. Safe to call concurrently —
    a second caller simply waits for the first sweep to finish rather
    than running its own overlapping one, then still gets its own
    accurate `deleted_ids` list for logging purposes (harmless
    redundant work, not a race).

    Never raises: a failure (DB unavailable, etc.) is logged and an
    empty list is returned, so a caller (the periodic loop, or a test)
    never needs its own try/except to stay alive.
    """
    async with _sweep_lock:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=app_settings.dataset_retention_hours)
        try:
            # Both the referenced-ids lookup and the delete itself are
            # synchronous SQLAlchemy calls; run them off the event loop
            # thread so a slow Postgres round trip (many rows, cold
            # connection) never blocks concurrently-running requests —
            # the same reasoning as `_execute_analysis`'s
            # `asyncio.to_thread` use in routes_experiments.py.
            def _do_sweep() -> list[str]:
                referenced_ids = _referenced_dataset_ids()
                return delete_unused_datasets(referenced_ids=referenced_ids, older_than=cutoff)

            deleted = await asyncio.to_thread(_do_sweep)
        except Exception:  # noqa: BLE001 — a sweep must never crash the scheduler or the app
            log.exception("[DatasetCleanupScheduler] Sweep failed — will retry on the next tick.")
            return []

        if deleted:
            log.info(
                "[DatasetCleanupScheduler] Swept %d orphaned dataset row(s) older than %.0fh: %s",
                len(deleted),
                app_settings.dataset_retention_hours,
                deleted,
            )
        return deleted


async def _sweep_loop() -> None:
    """Ticks `run_cleanup_sweep()` every
    `AppSettings.dataset_cleanup_interval_seconds`, forever, until
    cancelled. `run_cleanup_sweep()` already catches everything
    internally and is never expected to raise — but this loop belt-
    and-suspenders catches around the call anyway, so a future change
    to `run_cleanup_sweep` that accidentally lets an exception through
    can never take the whole scheduler (and therefore all FUTURE
    sweeps) down with it. This loop's only intended exit path is
    cancellation (app shutdown)."""
    while True:
        await asyncio.sleep(app_settings.dataset_cleanup_interval_seconds)
        try:
            await run_cleanup_sweep()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — see docstring: defense in depth only
            log.exception("[DatasetCleanupScheduler] Unexpected error escaped run_cleanup_sweep().")


_scheduler_task: "asyncio.Task | None" = None


def start_background_cleanup() -> None:
    """
    Starts the periodic sweep task for this process, if enabled and
    not already running. Called from `main.py`'s lifespan startup —
    idempotent (a second call while a task is already running is a
    no-op) so it's safe even if a caller invokes it more than once,
    e.g. across repeated test app construction in the same process.
    """
    global _scheduler_task
    if not app_settings.enable_background_dataset_cleanup:
        log.info("[DatasetCleanupScheduler] Disabled via ENABLE_BACKGROUND_DATASET_CLEANUP=false.")
        return
    if _scheduler_task is not None and not _scheduler_task.done():
        return
    _scheduler_task = asyncio.create_task(_sweep_loop())


async def stop_background_cleanup() -> None:
    """Cancels the periodic sweep task, if running, and waits for it
    to actually finish unwinding. Called from `main.py`'s lifespan
    shutdown so a dev-server reload / repeated app construction in
    tests never accumulates a second, orphaned scheduler task running
    alongside a new one."""
    global _scheduler_task
    if _scheduler_task is None:
        return
    _scheduler_task.cancel()
    try:
        await _scheduler_task
    except asyncio.CancelledError:
        pass
    _scheduler_task = None
