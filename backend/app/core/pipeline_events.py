"""
Pipeline instrumentation — per-stage timing for every graph node, plus
optional real-time progress events for the streaming `/analyze/stream`
endpoint (see routes_experiments.py).

DESIGN CONSTRAINTS (mirrors dataset_store.py's own "no raw DataFrame
in GraphState" rule):
  - Nothing here changes what a node computes. `instrument_node()`
    wraps a node's callable; the wrapped function calls the ORIGINAL
    node with the SAME state it always received. Timing/emission is
    pure side-channel bookkeeping around that call.
  - A `RunContext` is plumbed in through LangGraph's own `config`
    parameter (`config["configurable"]["run_context"]`), never through
    GraphState. GraphState is traced/serialized by LangSmith at every
    node; a RunContext holds a live callback and an event queue,
    neither of which is JSON-serializable, so it must never enter the
    state dict. `config["configurable"]` is not traced as node
    input/output, so this is the same seam LangGraph itself recommends
    for run-scoped, non-data objects (thread IDs, callbacks, etc).
  - A run invoked without a RunContext (e.g. existing tests that call
    `experiment_review_graph.invoke(state)` with no `config`, or
    `config=` without `configurable.run_context`) behaves exactly as
    before: node logic is unchanged, only the timing log line is
    skipped.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.core.logging import get_node_logger
from app.schemas.base import CamelModel

log = get_node_logger("Pipeline")

# (stage_started message, stage_completed message) — human-readable text
# for Part 4's streaming events. Falls back to the raw stage key for any
# node not listed here, so adding a node never requires touching this map.
STAGE_MESSAGES: dict[str, tuple[str, str]] = {
    "classifier": ("Classifying dataset", "Dataset classified"),
    "planner": ("Planning analysis", "Analysis plan ready"),
    "knowledge_base": ("Retrieving knowledge base context", "Knowledge base retrieval completed"),
    "funnel": ("Analyzing funnel", "Funnel analysis completed"),
    "validation": ("Validating experiment", "Experiment validation completed"),
    "experiment": ("Running statistical analysis", "Statistical analysis completed"),
    "guardrail": ("Checking guardrail metrics", "Guardrail check completed"),
    "decision": ("Generating report", "Report generation completed"),
}


class StageTiming(CamelModel):
    """
    One row of Part 1's required instrumentation record. A CamelModel
    (like every other schema in this app — see app/schemas/base.py) so
    `stage_timings` on `AnalyzeExperimentResponse` serializes as
    camelCase (`durationMs`, not `duration_ms`), consistent with every
    other field on the wire.
    """

    stage: str
    status: str  # "completed" | "failed"
    duration_ms: int
    error: Optional[str] = None


@dataclass
class RunContext:
    """
    One per `/experiments/analyze` (or `/analyze/stream`) invocation.

    `emit`, when provided, is called with a plain dict for every
    stage_started / stage_completed / error / pipeline_completed event
    (Part 4's event shapes). The streaming endpoint supplies an `emit`
    that pushes onto an asyncio.Queue (via `loop.call_soon_threadsafe`,
    since the graph itself runs on a worker thread); the synchronous
    endpoint passes `emit=None` and only collects `timings`.
    """

    run_id: str
    emit: Optional[Callable[[dict], None]] = None
    timings: list[StageTiming] = field(default_factory=list)
    _pipeline_start: float = field(default_factory=time.monotonic)
    # Set the moment any stage reports a failure (see `stage_failed`).
    # Used by the caller (routes_experiments.py's streaming endpoint) to
    # avoid emitting a second, generic "pipeline"-level error event for
    # a failure that a node already reported at the stage level — a
    # single real failure must produce exactly one user-visible error
    # event. Plain bool assignment/read is GIL-atomic, so this is safe
    # even though `stage_failed` can be called from the graph's worker
    # thread while the streaming coroutine reads it from the event loop.
    error_emitted: bool = False

    def _safe_emit(self, event: dict) -> None:
        if self.emit is None:
            return
        try:
            self.emit(event)
        except Exception:  # pragma: no cover - never let a UI event break the pipeline
            log.exception("[Pipeline] failed to emit event stage=%s", event.get("stage"))

    def stage_started(self, stage: str) -> None:
        message = STAGE_MESSAGES.get(stage, (stage, stage))[0]
        self._safe_emit({"type": "stage_started", "stage": stage, "message": message})

    def stage_completed(self, stage: str, duration_ms: int) -> None:
        self.timings.append(StageTiming(stage=stage, status="completed", duration_ms=duration_ms))
        message = STAGE_MESSAGES.get(stage, (stage, stage))[1]
        # SSE event payload uses camelCase (durationMs) to match every
        # other field the frontend already receives from this API (see
        # CamelModel) — this dict is NOT a StageTiming, it's the Part 4
        # wire event shape, kept as a plain dict since it's JSON sent
        # directly to the browser, not a stored/returned schema object.
        self._safe_emit(
            {"type": "stage_completed", "stage": stage, "message": message, "durationMs": duration_ms}
        )

    def stage_failed(self, stage: str, duration_ms: int, error: str) -> None:
        self.timings.append(
            StageTiming(stage=stage, status="failed", duration_ms=duration_ms, error=error)
        )
        self.error_emitted = True
        self._safe_emit({"type": "error", "stage": stage, "message": error})

    def pipeline_completed(self) -> None:
        """
        Records the total pipeline wall-clock time as a `StageTiming`
        (so it's included in `stage_timings`/the returned response) —
        does NOT itself emit an SSE event. Deliberately synchronous and
        side-effect-only: the caller (routes_experiments.py) decides
        when/how to actually send the `pipeline_completed` wire event,
        because on the streaming path that event must be awaited
        directly on the same coroutine as the final `result` event to
        guarantee ordering — routing it through `emit`'s
        `call_soon_threadsafe` bridge (used by every other event here,
        since those originate on the worker thread) would race with a
        directly-`await`ed queue put and could let `pipeline_completed`
        arrive before, or get dropped relative to, `result`.
        """
        total_ms = int((time.monotonic() - self._pipeline_start) * 1000)
        self.timings.append(StageTiming(stage="total", status="completed", duration_ms=total_ms))

    def timings_as_dicts(self) -> list[dict]:
        return [json.loads(t.model_dump_json(by_alias=True)) for t in self.timings]


def instrument_node(stage: str):
    """
    Wraps a LangGraph node function so every invocation records a
    StageTiming and (if a RunContext with `emit` is present) fires the
    stage_started/stage_completed/error events from Part 4.

    LangGraph inspects a node callable's signature and passes a second
    positional `config` argument automatically when the callable
    accepts one — the wrapped function opts into that; the ORIGINAL
    node function's own signature (`fn(state) -> GraphState`) is left
    completely untouched, so every existing direct unit test that
    calls e.g. `classifier_node(state)` still works unmodified.
    """

    def decorator(fn):
        def wrapped(state, config=None):
            ctx: Optional[RunContext] = None
            if config:
                ctx = (config.get("configurable") or {}).get("run_context")

            t0 = time.monotonic()
            if ctx:
                ctx.stage_started(stage)

            try:
                result = fn(state)
            except Exception as exc:
                duration_ms = int((time.monotonic() - t0) * 1000)
                if ctx:
                    ctx.stage_failed(stage, duration_ms, str(exc))
                log.error("[Pipeline] stage=%s status=failed duration_ms=%d error=%s", stage, duration_ms, exc)
                raise

            duration_ms = int((time.monotonic() - t0) * 1000)
            if ctx:
                ctx.stage_completed(stage, duration_ms)
            log.info("[Pipeline] stage=%s status=completed duration_ms=%d", stage, duration_ms)
            return result

        wrapped.__name__ = getattr(fn, "__name__", stage)
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        return wrapped

    return decorator
