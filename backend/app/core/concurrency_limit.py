"""
Bounds how many FULL analysis pipeline executions
(`experiment_review_graph.invoke`, run via `asyncio.to_thread` in
routes_experiments.py) can run at the same time in this process.

WHY THIS EXISTS (concurrency audit finding): a pipeline execution is
real CPU-bound work (pandas/statsmodels computation across every stat
node) plus, when `report_backend`/`planner_backend` is "llm", a
synchronous outbound HTTP call. Nothing previously limited how many of
these could run concurrently — each accepted `/experiments/analyze` or
`/analyze/stream` request (after `core/execution_dedup.py`'s
suppression of genuine duplicates) got its own worker thread from
`asyncio.to_thread`'s default executor (up to `min(32, cpu_count()+4)`
threads). On this project's actual target — Render's free tier, a
small/shared-CPU instance (see render.yaml) — a handful of genuinely
DIFFERENT concurrent analyses (5 users, each running their own
experiment) would all fight over the same one-or-few CPU cores, so
every one of them gets slower rather than any one of them failing
fast. That's the mechanism behind "it works sometimes and doesn't
work sometimes": under load, individual requests don't error, they
just quietly take long enough to trip Render's platform-level request
timeout, which looks identical to "the LLM didn't connect" from the
frontend's perspective even though the actual cause is CPU contention,
not the LLM.

This is deliberately a SEMAPHORE around executions, not a rate
limiter (`core/rate_limit.py` already exists and solves a different
problem — request RATE per identity/route, regardless of whether
anything is still running). A single user is essentially never
affected (their own requests are already deduplicated by
execution_dedup before reaching here); this only engages once
`AppSettings.max_concurrent_analyses` DIFFERENT analyses are
genuinely running at once.

WHEN THE LIMIT IS HIT: `TooManyConcurrentAnalysesError` is raised
immediately (no queueing) — routes_experiments.py turns this into a
clean 503 with a clear message. Queueing was deliberately rejected:
a queued request would keep the HTTP connection open while doing
nothing, and on Render's own request timeout, an indefinitely-queued
request fails anyway — just later, and looking exactly like the
opaque hang this audit is trying to eliminate. Failing fast lets the
frontend show something legible ("try again in a moment") instead.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from app.core.config import app_settings


class TooManyConcurrentAnalysesError(RuntimeError):
    """Raised when `AppSettings.max_concurrent_analyses` pipeline
    executions are already running in this process. Routes translate
    this to HTTP 503."""


# A single process-wide counter + lock. Re-reads
# `app_settings.max_concurrent_analyses` on every call (rather than
# snapshotting it once) so a test/operator changing the setting takes
# effect immediately — this mirrors how `core/rate_limit.py`'s buckets
# and `core/execution_dedup.py`'s registry are also plain
# process-local state; see those modules' docstrings for the same
# accepted single-worker-deployment trade-off (render.yaml runs one
# Uvicorn worker; this does not coordinate across multiple workers or
# multiple Vercel serverless instances — it only needs to protect THIS
# process's own CPU, which is exactly the resource each of those
# workers/instances has its own of anyway).
#
# WHY NOT `asyncio.Semaphore`: a semaphore is the obvious first
# reach for this, but there is no built-in non-blocking "try_acquire",
# and the natural-looking `asyncio.wait_for(semaphore.acquire(),
# timeout=0)` substitute is unsafe — cancelling an `acquire()` that
# raced to actually start waiting internally can leave the semaphore's
# waiter queue in a state where a SUBSEQUENT, unrelated `release()`
# wakes the (already-cancelled) waiter instead of incrementing the
# available count, so a later caller's acquire can incorrectly block
# forever. Verified empirically while writing this module — not a
# hypothetical. A plain counter guarded by `asyncio.Lock` has none of
# that internal waiter-queue machinery and is trivially correct.
_lock = asyncio.Lock()
_current_count = 0


@asynccontextmanager
async def limit_concurrent_analyses():
    """
    Usage:

        async with limit_concurrent_analyses():
            ... run the pipeline ...

    Raises `TooManyConcurrentAnalysesError` immediately (does not
    block/queue — see this module's docstring for why queueing was
    rejected) if `AppSettings.max_concurrent_analyses` executions are
    already in flight in this process.
    """
    global _current_count
    async with _lock:
        if _current_count >= app_settings.max_concurrent_analyses:
            raise TooManyConcurrentAnalysesError(
                f"{app_settings.max_concurrent_analyses} analyses are already running on this instance."
            )
        _current_count += 1

    try:
        yield
    finally:
        async with _lock:
            _current_count -= 1
