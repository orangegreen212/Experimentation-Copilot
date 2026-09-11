"""
In-process duplicate-execution suppression for `/experiments/analyze`
and `/experiments/analyze/stream`.

PROBLEM THIS FIXES: neither route had any notion of execution identity
before this module existed. Every accepted HTTP request ran the full
LangGraph pipeline (one blocking `ChatOpenAI` call when
`REPORT_BACKEND=llm`/`PLANNER_BACKEND=llm`, plus one
`ExperimentStore.create()` insert) unconditionally — there was nothing
in `routes_experiments.py` that could recognize "this is the same
analysis as the one already running" and reuse its result. Two
concrete, easy-to-trigger paths turn that into duplicate LLM spend
*and* duplicate Experiment History rows for what the user experiences
as ONE analysis:

  1. A double-submit: the user's browser retries a POST that appeared
     to hang (common on a slow OpenRouter model), or the user clicks
     Analyze twice in quick succession before the frontend's
     `phase === 'running'` disable takes effect.
  2. A refresh-and-resubmit: the SSE endpoint deliberately does NOT
     cancel the backend pipeline on client disconnect (see
     `analyze_experiment_stream`'s docstring in routes_experiments.py
     — work in flight is intentionally left to finish and persist).
     If the user refreshes the page mid-analysis and clicks Analyze
     again, the first run is still executing server-side; the second
     click starts a genuinely independent second run of the identical
     request.

Neither case is a malicious retry storm (rate_limit.py already handles
that) — both are ordinary usage that this module turns from "two full
pipeline executions, two DB rows, two LLM calls" into "one execution,
shared by every caller that asked for the exact same thing while it
was in flight."

SCOPE / WHAT THIS DOES NOT DO: this is process-local, in-memory state
— exactly the same trade-off `core/rate_limit.py` already documents
and accepts for this single-instance deployment (Render, one Uvicorn
worker per render.yaml). It does not persist across a restart, and it
does not coordinate across horizontally-scaled instances. It also only
dedupes requests that are BYTE-FOR-BYTE identical in the fields that
determine the pipeline's output (dataset, assignment dataset, prompt,
settings, hypothesis, and — for definition-linked runs — the
definition id); it deliberately does not attempt fuzzy/semantic
matching of "is this basically the same question."

WHAT IT DOES NOT CHANGE: the pipeline itself, `ExperimentStore`, or
what gets persisted for a solo (non-duplicate) request — a request
with no in-flight twin runs exactly as it did before this module
existed, with the exact same latency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

# key -> the shared Future every concurrent caller for that key awaits.
# Populated by the first ("leader") caller for a key and removed again
# once that caller's execution finishes (success or failure) — never
# grows across the life of the process, only for as long as an
# execution with that exact key is actually in flight.
_inflight: dict[str, "asyncio.Future[Any]"] = {}

# Guards read-modify-write access to `_inflight` itself (checking for
# an existing entry and inserting a new one must be atomic, or two
# concurrent callers could each conclude they're the leader).
_registry_lock = asyncio.Lock()


def make_key(**fields: Any) -> str:
    """
    Deterministic key from the fields that fully determine the
    pipeline's output for one analysis request. Order-independent
    (sorted keys) and stable across Pydantic model shapes — callers
    pass already-plain values (str / dict / None), not model
    instances, so this has no dependency on any particular schema's
    `__repr__`/hashing.
    """
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def run_deduplicated(key: str, work: Callable[[], Awaitable[T]]) -> tuple[bool, T]:
    """
    Runs `work()` at most once per `key` among concurrent callers.

    Returns `(is_leader, result)`:
      - The first caller for a given key (`is_leader=True`) actually
        invokes `work()` and its result (or exception) is shared with
        every other caller that arrives with the same key while it is
        still running.
      - Every subsequent concurrent caller for that same key
        (`is_leader=False`) does NOT call `work()` at all — it just
        awaits the leader's result. `is_leader` is returned so the
        streaming endpoint can tell a follower apart from the leader
        and emit a distinct progress event for it (see
        `analyze_experiment_stream`).

    A caller whose key is not currently in flight always becomes the
    leader, so a normal, non-duplicate request behaves exactly as if
    this function didn't exist.
    """
    async with _registry_lock:
        future = _inflight.get(key)
        is_leader = future is None
        if is_leader:
            future = asyncio.get_event_loop().create_future()
            _inflight[key] = future

    assert future is not None  # for type-checkers; always set above

    if not is_leader:
        return False, await future

    try:
        result = await work()
    except BaseException as exc:  # noqa: BLE001 — must propagate to the leader's own caller
        future.set_exception(exc)
        # Immediately mark the exception "retrieved" from asyncio's
        # perspective. If no follower ever joined this key, nothing
        # else would ever call `.exception()`/`.result()` on this
        # future, and asyncio logs a spurious "exception was never
        # retrieved" warning when it's garbage-collected. This does
        # not affect what any follower awaiting the same future sees
        # (it still raises the same exception for them, exactly once
        # per follower).
        future.exception()
        raise
    else:
        future.set_result(result)
        return True, result
    finally:
        async with _registry_lock:
            # Only the leader ever removes its own key — a follower
            # never touches `_inflight`.
            if _inflight.get(key) is future:
                del _inflight[key]
