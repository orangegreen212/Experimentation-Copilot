"""
Unit tests for app/core/concurrency_limit.py.

Covers:
  - up to `max_concurrent_analyses` executions can run at once
  - the next one is rejected immediately (never queued/blocked)
  - a slot is released on both success and exception, so the limiter
    never "leaks" a permanently-unavailable slot
  - the safe counter+lock implementation genuinely allows a second
    caller in after the first releases (regression test for the
    asyncio.Semaphore+wait_for(timeout=0) footgun found while building
    this — see the module's own docstring for what went wrong with
    that approach)
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.concurrency_limit import TooManyConcurrentAnalysesError, limit_concurrent_analyses
from app.core.config import app_settings


@pytest.fixture(autouse=True)
def _default_limit(monkeypatch):
    monkeypatch.setattr(app_settings, "max_concurrent_analyses", 2)


@pytest.mark.asyncio
async def test_up_to_the_limit_runs_concurrently():
    entered = 0
    max_entered = 0
    release = asyncio.Event()

    async def hold():
        nonlocal entered, max_entered
        async with limit_concurrent_analyses():
            entered += 1
            max_entered = max(max_entered, entered)
            await release.wait()
            entered -= 1

    t1 = asyncio.create_task(hold())
    t2 = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    assert max_entered == 2
    release.set()
    await asyncio.gather(t1, t2)


@pytest.mark.asyncio
async def test_exceeding_the_limit_is_rejected_immediately_not_queued():
    release = asyncio.Event()

    async def hold():
        async with limit_concurrent_analyses():
            await release.wait()

    t1 = asyncio.create_task(hold())
    t2 = asyncio.create_task(hold())
    await asyncio.sleep(0.05)  # both slots now taken

    with pytest.raises(TooManyConcurrentAnalysesError):
        async with limit_concurrent_analyses():
            pytest.fail("must not be able to enter a third concurrent execution")

    release.set()
    await asyncio.gather(t1, t2)


@pytest.mark.asyncio
async def test_slot_is_released_after_a_failing_execution():
    """A slot must not leak if the wrapped work raises — otherwise a
    single failed analysis would permanently shrink capacity."""
    with pytest.raises(ValueError):
        async with limit_concurrent_analyses():
            raise ValueError("boom")

    # The slot from the failed attempt above must be available again.
    entered = False
    async with limit_concurrent_analyses():
        entered = True
    assert entered


@pytest.mark.asyncio
async def test_a_new_caller_gets_in_after_an_earlier_one_releases():
    """Direct regression test for the asyncio.Semaphore +
    wait_for(timeout=0) bug found while implementing this module: a
    caller rejected while the limit was full must be able to succeed
    once a slot frees up — not deadlock forever."""
    release = asyncio.Event()

    async def hold():
        async with limit_concurrent_analyses():
            await release.wait()

    t1 = asyncio.create_task(hold())
    t2 = asyncio.create_task(hold())
    await asyncio.sleep(0.05)

    with pytest.raises(TooManyConcurrentAnalysesError):
        async with limit_concurrent_analyses():
            pass

    release.set()
    await asyncio.gather(t1, t2)

    # Must succeed promptly now that both slots are free — a hang here
    # would mean the limiter's internal state got corrupted.
    async def acquire_once():
        async with limit_concurrent_analyses():
            return "ok"

    result = await asyncio.wait_for(acquire_once(), timeout=1)
    assert result == "ok"
