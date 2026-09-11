"""
Unit tests for app/core/execution_dedup.py.

These test the dedup PRIMITIVE directly (no HTTP, no DB, no LangGraph)
so the concurrency behavior itself — which is the part that's easy to
get subtly wrong (races, stuck locks, leaked state) — is pinned down
precisely and fast, independent of anything about the analysis
pipeline. `tests/api/test_execution_dedup_api.py` covers the same
guarantees through the actual `/experiments/analyze*` routes.

Scenarios covered (see this module's and execution_dedup.py's
docstrings for why each matters):
  1. two calls with the SAME key, overlapping in time -> `work()` runs
     exactly once; the second (non-leader) caller gets the same result
     without ever calling `work()` itself.
  2. two calls with DIFFERENT keys, overlapping in time -> both run
     concurrently, neither blocks the other (no global lock).
  3. a failed leader -> the exception is raised to every concurrent
     follower waiting on that same key.
  4. after a key's execution finishes (success OR failure), it is
     removed from the registry -> a LATER call with the same key is a
     fresh leader and runs `work()` again (a failure never "poisons"
     the key forever; a completed run never permanently caches — see
     each test's own docstring for why that's the correct behavior for
     this product, not an oversight).
  5. `make_key()` is order-independent and distinguishes any differing
     field.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.execution_dedup import make_key, run_deduplicated


# ---------------------------------------------------------------------------
# make_key
# ---------------------------------------------------------------------------


def test_make_key_is_stable_and_order_independent():
    key_a = make_key(dataset_id="d1", prompt="p", settings={"cuped": True}, definition_id=None)
    key_b = make_key(settings={"cuped": True}, definition_id=None, prompt="p", dataset_id="d1")
    assert key_a == key_b


def test_make_key_differs_on_any_field():
    base = dict(dataset_id="d1", prompt="p", settings={"cuped": True}, definition_id=None)
    baseline = make_key(**base)

    assert make_key(**{**base, "dataset_id": "d2"}) != baseline
    assert make_key(**{**base, "prompt": "different prompt"}) != baseline
    assert make_key(**{**base, "settings": {"cuped": False}}) != baseline
    assert make_key(**{**base, "definition_id": "def-1"}) != baseline


def test_make_key_covers_every_analysis_settings_field():
    """
    Regression test for the audit's 'most dangerous possible
    regression' concern: two requests that differ in ANY field that
    can change the pipeline's output must never collide. This
    constructs the exact key `_analysis_dedup_key` builds
    (routes_experiments.py) using two settings objects that differ in
    every individual field AnalysisSettings declares, one at a time,
    and asserts each one changes the key on its own.
    """
    from app.schemas.settings import AnalysisSettings

    base_settings = AnalysisSettings(
        cuped=False,
        bootstrap=False,
        model="claude-sonnet",
        cost_usd=0,
        confidence_level=0.95,
        statistical_power=0.8,
        guardrail_metrics=["Revenue"],
        analysis_mode=None,
        stratification_column=None,
    )
    baseline = make_key(
        dataset_id="d1", assignment_dataset_id=None, prompt="Should we ship?",
        settings=base_settings.model_dump(by_alias=True), hypothesis=None, definition_id=None,
    )

    variants = {
        "cuped": base_settings.model_copy(update={"cuped": True}),
        "bootstrap": base_settings.model_copy(update={"bootstrap": True}),
        "model": base_settings.model_copy(update={"model": "gpt-4o-mini"}),
        "confidence_level": base_settings.model_copy(update={"confidence_level": 0.99}),
        "statistical_power": base_settings.model_copy(update={"statistical_power": 0.9}),
        "guardrail_metrics": base_settings.model_copy(update={"guardrail_metrics": ["Bounce Rate"]}),
        "analysis_mode": base_settings.model_copy(update={"analysis_mode": "stratified", "stratification_column": "country"}),
    }
    for field_name, variant_settings in variants.items():
        variant_key = make_key(
            dataset_id="d1", assignment_dataset_id=None, prompt="Should we ship?",
            settings=variant_settings.model_dump(by_alias=True), hypothesis=None, definition_id=None,
        )
        assert variant_key != baseline, f"changing only {field_name!r} must change the dedup key"


def test_make_key_differs_on_hypothesis():
    """A different (or newly-added) hypothesis must never be treated
    as the same request — it's carried into the report as structured
    context and could change the decision narrative."""
    key_no_hypothesis = make_key(dataset_id="d1", prompt="p", settings={}, hypothesis=None, definition_id=None)
    key_with_hypothesis = make_key(
        dataset_id="d1", prompt="p", settings={}, definition_id=None,
        hypothesis={"statement": "X will increase conversion", "primaryMetric": "conversion"},
    )
    key_with_different_hypothesis = make_key(
        dataset_id="d1", prompt="p", settings={}, definition_id=None,
        hypothesis={"statement": "X will DECREASE conversion", "primaryMetric": "conversion"},
    )
    assert len({key_no_hypothesis, key_with_hypothesis, key_with_different_hypothesis}) == 3


# ---------------------------------------------------------------------------
# run_deduplicated — concurrency behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_identical_key_runs_work_exactly_once():
    """
    Simulates: the user double-clicks Analyze, or the frontend retries
    a request that appeared to hang, or a page refresh re-submits the
    same analysis while the first is still running server-side. Both
    HTTP requests arrive close enough together that the second joins
    the first while it's still "in flight" — this is the exact
    scenario `core/execution_dedup.py`'s module docstring names.
    """
    call_count = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_work():
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()
        return "the-result"

    key = "same-analysis"

    leader_task = asyncio.create_task(run_deduplicated(key, slow_work))
    await started.wait()  # ensure the leader is actually "running" before the follower joins
    follower_task = asyncio.create_task(run_deduplicated(key, slow_work))

    # Give the follower a tick to register itself against the same key
    # (rather than racing to start its own second execution).
    await asyncio.sleep(0)
    release.set()

    leader_result = await leader_task
    follower_result = await follower_task

    assert call_count == 1, "work() must run exactly once for two concurrent identical requests"
    assert leader_result == (True, "the-result")
    assert follower_result == (False, "the-result")


@pytest.mark.asyncio
async def test_concurrent_different_keys_both_run_independently():
    """
    Two DIFFERENT analyses (different dataset/prompt/settings) must
    never block each other — this module intentionally has no global
    lock, only a per-key one, so unrelated experiments/users are
    unaffected by dedup.
    """
    call_count = 0
    release = asyncio.Event()

    async def slow_work():
        nonlocal call_count
        call_count += 1
        await release.wait()
        return "ok"

    task_a = asyncio.create_task(run_deduplicated("key-a", slow_work))
    task_b = asyncio.create_task(run_deduplicated("key-b", slow_work))
    await asyncio.sleep(0)

    # Both must have actually started (neither is blocked waiting on
    # the other's key) before either is allowed to finish.
    assert call_count == 2
    release.set()

    result_a = await task_a
    result_b = await task_b
    assert result_a == (True, "ok")
    assert result_b == (True, "ok")


@pytest.mark.asyncio
async def test_failed_leader_propagates_to_concurrent_followers():
    """A previous execution failed: every caller that was waiting on
    the SAME in-flight (now-failed) execution must see that failure —
    not a hang, not a silently different result."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing_work():
        started.set()
        await release.wait()
        raise ValueError("boom")

    key = "will-fail"
    leader_task = asyncio.create_task(run_deduplicated(key, failing_work))
    await started.wait()
    follower_task = asyncio.create_task(run_deduplicated(key, failing_work))
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(ValueError, match="boom"):
        await leader_task
    with pytest.raises(ValueError, match="boom"):
        await follower_task


@pytest.mark.asyncio
async def test_failed_execution_does_not_permanently_block_retry():
    """
    'A failed execution must not permanently block a future legitimate
    retry.' After a key's leader raises, the key must be removed from
    the registry so a SUBSEQUENT (non-overlapping) call with the same
    key is a fresh leader and actually retries the work — not an
    immediate re-raise of the stale failure, and not a permanent
    no-op.
    """
    attempts = 0

    async def flaky_work():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient failure")
        return "succeeded-on-retry"

    key = "retry-me"
    with pytest.raises(RuntimeError, match="transient failure"):
        await run_deduplicated(key, flaky_work)

    # A later, non-overlapping call with the identical key must run
    # work() again, fresh — the earlier failure must not "poison" the
    # key forever.
    is_leader, result = await run_deduplicated(key, flaky_work)
    assert attempts == 2
    assert is_leader is True
    assert result == "succeeded-on-retry"


@pytest.mark.asyncio
async def test_completed_execution_key_is_released_for_a_later_call():
    """
    'A completed execution should not unnecessarily execute the
    expensive pipeline again if the request is logically identical' —
    for a request that arrives WHILE the identical one is still
    running (see test_concurrent_identical_key_runs_work_exactly_once
    above). Once a run has actually finished, its key is intentionally
    released rather than cached forever: a later, separate call to
    re-run the exact same analysis is a deliberate new execution (this
    product's own History model supports multiple runs of the same
    definition/dataset — see ExperimentStore.list_by_definition), not
    a duplicate to be silently suppressed. This test pins down that
    "released after completion" behavior explicitly, so it can't
    regress into either extreme (permanent caching, or a leaked key
    that blocks retries).
    """
    call_count = 0

    async def work():
        nonlocal call_count
        call_count += 1
        return "result"

    key = "run-then-run-again"
    is_leader_1, result_1 = await run_deduplicated(key, work)
    is_leader_2, result_2 = await run_deduplicated(key, work)

    assert (is_leader_1, result_1) == (True, "result")
    assert (is_leader_2, result_2) == (True, "result")
    assert call_count == 2, "a call after the previous one fully completed must run fresh"


@pytest.mark.asyncio
async def test_no_leaked_asyncio_future_warnings_on_leader_only_failure(recwarn):
    """
    A leader whose work() raises, with NO follower ever joining, must
    not leave an asyncio Future with an unretrieved exception (which
    would otherwise log a spurious "exception was never retrieved"
    warning when garbage collected) — see run_deduplicated's own
    comment on why `future.exception()` is called defensively.
    """
    async def failing_work():
        raise ValueError("solo failure")

    with pytest.raises(ValueError):
        await run_deduplicated("solo-fail-key", failing_work)

    import gc

    gc.collect()
    assert not any("never retrieved" in str(w.message) for w in recwarn.list)
