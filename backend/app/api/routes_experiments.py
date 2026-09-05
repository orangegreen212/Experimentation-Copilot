"""
POST /experiments/analyze
GET  /experiments
GET  /experiments/{experiment_id}
DELETE /experiments/{experiment_id}
GET  /experiments/{experiment_id}/chat
POST /experiments/{experiment_id}/chat
POST /experiments/{experiment_id}/chat/stream

Synchronous analyze (no SSE/polling).
The frontend's fake-timer stepper stays as-is in the UI for now; this
endpoint returns the FINAL `executionSteps` (real `detail` text
reflecting what the graph actually did) alongside the report.

Experiment reports are persisted through `ExperimentStore` (see
app/core/experiment_store.py) instead of an in-process-only
`_REPORT_STORE` dict, so History survives a process restart / cold
start — required for serverless (Vercel) deployment.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.dataset_store import dataset_exists, dataset_request_scope
from app.graph.chat_generator import (
    DEFAULT_MAX_HISTORY_MESSAGES,
    build_chat_message,
    finalize_streamed_chat_message,
    stream_chat_response,
)
from app.core.config import app_settings
from app.core.experiment_store import ExperimentStore, get_experiment_store
from app.core.logging import get_node_logger
from app.core.pipeline_events import RunContext
from app.core.rate_limit import rate_limit
from app.graph.graph_builder import experiment_review_graph
from app.schemas.base import CamelModel
from app.schemas.chat import ChatMessage, ChatRole, FollowUpChatRequest, FollowUpChatResponse
from app.schemas.execution import ExecutionStep, ExecutionStepGroup, ExecutionStepStatus, RunMetadata
from app.schemas.experiment_history import ExperimentDetail, ExperimentSummary, RelatedExperiment
from app.schemas.hypothesis import Hypothesis
from app.schemas.report import ExperimentReport
from app.schemas.settings import AnalysisSettings
from app.stats.dataset_classifier import DatasetClassificationError, describe_dataset_structure
from app.stats.hypothesis_tests import test_selection_to_quality_check_detail

router = APIRouter(prefix="/experiments", tags=["experiments"])
log = get_node_logger("API")


class AnalyzeExperimentRequest(CamelModel):
    dataset_id: str
    prompt: str
    settings: AnalysisSettings
    # Display name for History (e.g. the uploaded/demo file name from
    # /datasets/classify's response). Falls back to dataset_id if the
    # frontend doesn't have one on hand.
    dataset_name: str = ""
    # Optional separate experiment-assignment dataset (e.g. `user_id |
    # variant`) — the `dataset_id` returned by a SECOND, independent
    # call to POST /datasets/classify (same upload mechanism as the
    # primary dataset, no parallel upload path). None (the default) is
    # fully backward compatible: every existing single-file analyze
    # request is completely unaffected. See app/graph/state.py and
    # app/stats/dataset_classifier.py's enrich_with_assignment for how
    # this flows through classifier_node.
    assignment_dataset_id: str | None = None
    # Phase 1 — Experiment Hypothesis. Optional and purely additive:
    # omitting it (the default) is 100% backward compatible with every
    # existing analysis flow. When provided, it's carried through
    # unmodified as structured context — see app/graph/state.py and
    # ReportFacts.hypothesis (report_generator.py) for how it flows
    # from here into the analysis pipeline. This phase does not yet
    # compute anything from it (no verdict, no expected-vs-observed
    # comparison — see Hypothesis's own docstring for full scope).
    hypothesis: Hypothesis | None = None


class AnalyzeExperimentResponse(CamelModel):
    experiment_id: str
    report: ExperimentReport
    execution_steps: list[ExecutionStep]
    # Structured retrieval of prior runs against the same dataset name
    # (see ExperimentStore.list_related) — plain filtered lookup, not
    # LLM/semantic memory. Empty when this is the first run for the
    # dataset.
    related_experiments: list[RelatedExperiment] = []
    # Part 1 of the pipeline-instrumentation task — per-stage timing
    # (stage/status/duration_ms/error), always populated (never used
    # to make any decision, purely observational). See
    # app/core/pipeline_events.py.
    stage_timings: list[dict] = []


def _to_related(record) -> RelatedExperiment:
    return RelatedExperiment(
        experiment_id=record.experiment_id,
        created_at=record.created_at,
        user_prompt=record.user_prompt,
        decision=record.decision,
        confidence=record.confidence,
        primary_metric=record.primary_metric,
    )


def _store() -> ExperimentStore:
    return get_experiment_store()


async def _execute_analysis(
    request: "AnalyzeExperimentRequest",
    run_context: RunContext,
    *,
    definition_id: str | None = None,
) -> AnalyzeExperimentResponse:
    """
    Shared body for the synchronous `/analyze` endpoint and the
    streaming `/analyze/stream` endpoint (Part 5 — SSE was chosen over
    WebSockets: progress only ever flows server -> client, so a
    bidirectional channel buys nothing here and adds a second protocol
    to maintain; see `analyze_experiment_stream`'s docstring). Both
    endpoints compute and persist the exact same
    `AnalyzeExperimentResponse` — the streaming endpoint just also
    reports progress while this runs (Part 7: the streamed channel is
    for progress only, this function is the single source of truth for
    the final response on both paths).

    `run_context` carries per-run stage timings (Part 1) and, when the
    caller wants live progress, an `emit` callback (Part 4) — see
    app/core/pipeline_events.py. The non-streaming endpoint passes a
    RunContext with `emit=None`, so timings are still collected (and
    returned in `stage_timings`) even without streaming.

    `definition_id` (Phase 8 — Experiment Platform layer) is None for
    every call from the two routes below (the existing, definition-less
    analysis flow is completely unaffected). It is set only when this
    is invoked from
    `routes_experiment_definitions.analyze_experiment_definition`,
    which composes an `AnalyzeExperimentRequest` from a saved
    `ExperimentDefinition`'s `data_source` + primary hypothesis and
    calls straight into this SAME function — deliberately not a
    parallel/duplicated analysis path (see that route's docstring).
    Threading it through here is what lets the resulting persisted run
    link back to its definition (`ExperimentStore.list_by_definition`,
    Phase 10 — History).
    """
    if not dataset_exists(request.dataset_id):
        raise HTTPException(status_code=404, detail=f"Unknown dataset_id: {request.dataset_id}")
    if request.assignment_dataset_id and not dataset_exists(request.assignment_dataset_id):
        raise HTTPException(
            status_code=404, detail=f"Unknown assignment_dataset_id: {request.assignment_dataset_id}"
        )

    initial_state = {
        "dataset_id": request.dataset_id,
        "assignment_dataset_id": request.assignment_dataset_id,
        "user_prompt": request.prompt,
        "settings": request.settings,
        # Phase 1 — canonical location for the hypothesis to enter the
        # pipeline. None when the analyst didn't provide one (fully
        # backward compatible — see AnalyzeExperimentRequest.hypothesis).
        "hypothesis": request.hypothesis,
    }

    # Request-scoped dataset cache: classifier_node, validation_node,
    # and experiment_node (and funnel_node, when routed there) each
    # call get_dataset(request.dataset_id) independently. Without this
    # scope every one of those is a fresh DB read + full
    # pd.read_json(orient="table") deserialize of the same DataFrame —
    # measured at ~5.5-6.5s each on a 294K-row dataset, which is what
    # was tripping Render's request timeout. Inside this scope, only
    # the first call actually hits the DB; the rest reuse that
    # DataFrame from memory. Always torn down on exit (including on
    # exception) by dataset_request_scope() itself — see its docstring
    # in app/core/dataset_store.py.
    with dataset_request_scope():
        try:
            # PRODUCTION AVAILABILITY FIX (Render health-check timeout):
            # `experiment_review_graph.invoke(...)` is entirely
            # synchronous — pandas/statsmodels CPU work, and, when
            # REPORT_BACKEND/PLANNER_BACKEND="llm", a *blocking*
            # `ChatOpenAI(...).invoke()` HTTP call (see
            # app/llm/client.py's `request_timeout=30s`). This app runs
            # as a single Uvicorn worker (render.yaml), so calling
            # `.invoke()` directly here — on the event loop thread —
            # blocks EVERY other coroutine on this process for the
            # entire duration, including `GET /health`. Render polls
            # `/health` every ~5s; once it can't be scheduled for a few
            # consecutive polls, Render kills and restarts the
            # instance mid-request, which is exactly what the "no
            # further application logs, then a fresh uvicorn boot ~30s
            # later" pattern in the incident logs shows. This reproduces
            # regardless of dataset size because the worst-case blocker
            # (the LLM HTTP round trip) doesn't scale with row count.
            # `asyncio.to_thread` moves the blocking call to a worker
            # thread so the event loop stays free to keep answering
            # `/health` (and any other concurrent request) while the
            # analysis runs — `asyncio.to_thread` also copies the
            # calling task's contextvars into the worker thread, which
            # is what lets `dataset_request_scope()` above (itself
            # contextvar-based) still apply inside the thread the graph
            # actually runs on. No change to what is computed or how.
            start = time.monotonic()
            final_state = await asyncio.to_thread(
                experiment_review_graph.invoke,
                initial_state,
                config={
                    "run_name": "Experiment Review",
                    "tags": ["experiment-review-copilot"],
                    "metadata": {
                        "dataset_id": request.dataset_id,
                        "prompt": request.prompt[:200],
                        "planner_backend": app_settings.planner_backend,
                        "report_backend": app_settings.report_backend,
                        "cuped": request.settings.cuped,
                        "bootstrap": request.settings.bootstrap,
                    },
                    # Part 1/4 — every node (see graph_builder.py's
                    # instrument_node(...) wrapping) reads this back out
                    # to record timing and, if `run_context.emit` is
                    # set, fire stage_started/stage_completed events.
                    # Never read by any node's own business logic.
                    "configurable": {"run_context": run_context},
                },
            )
            log.info("[API] graph.invoke completed in %.2fs (off event loop thread)", time.monotonic() - start)
        except DatasetClassificationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    execution_steps = _build_execution_steps(final_state)
    report = final_state["report"]

    # Phase 8 — structured run metadata, built here (not in
    # decision_node) because `dataset_name` is only known at the API
    # layer (see AnalyzeExperimentRequest.dataset_name — GraphState
    # deliberately never carries display-only strings). Stamped onto
    # the already-generated report via model_copy, same pattern as
    # hypothesis/decision_support/segmentation in decision_node.py, so
    # it persists through ExperimentStore and reopens correctly later.
    run_metadata = _build_run_metadata(
        request=request, final_state=final_state, execution_steps=execution_steps, report=report
    )
    report = report.model_copy(update={"run_metadata": run_metadata})

    record = _store().create(
        dataset_id=request.dataset_id,
        dataset_name=request.dataset_name or request.dataset_id,
        user_prompt=request.prompt,
        report=report,
        execution_steps=execution_steps,
        definition_id=definition_id,
    )

    related = _store().list_related(
        record.dataset_id, exclude_experiment_id=record.experiment_id, limit=5
    )

    # Part 1 — record total pipeline wall-clock time as the final
    # timing row (see RunContext.pipeline_completed's docstring for why
    # this does NOT also send the SSE `pipeline_completed` event itself
    # — the caller below does that once it actually has this response).
    run_context.pipeline_completed()

    return AnalyzeExperimentResponse(
        experiment_id=record.experiment_id,
        report=report,
        execution_steps=execution_steps,
        related_experiments=[_to_related(r) for r in related],
        stage_timings=run_context.timings_as_dicts(),
    )


@router.post(
    "/analyze",
    response_model=AnalyzeExperimentResponse,
    dependencies=[Depends(rate_limit("analyze", max_requests=10))],
)
async def analyze_experiment(request: AnalyzeExperimentRequest) -> AnalyzeExperimentResponse:
    """
    Run the full LangGraph pipeline synchronously, persist the
    completed report to Experiment History, and return it + the final
    execution step trace. Same response shape/behavior as before
    instrumentation was added — `stage_timings` is a purely additive
    field (Part 1). Nothing streams here; see
    `analyze_experiment_stream` for that.
    """
    run_context = RunContext(run_id=str(uuid.uuid4()))
    return await _execute_analysis(request, run_context)


@router.post(
    "/analyze/stream",
    dependencies=[Depends(rate_limit("analyze", max_requests=10))],
)
async def analyze_experiment_stream(request: AnalyzeExperimentRequest) -> StreamingResponse:
    """
    Part 4/5/6 — same pipeline and same final `AnalyzeExperimentResponse`
    as POST /analyze, but streamed as Server-Sent Events so the
    frontend can render real per-stage progress instead of a single
    blocking wait.

    WHY SSE over a raw streaming Response or WebSockets: the pipeline
    is strictly server -> client progress with a single terminal
    result — there is no client -> server message once the request is
    open, so plain one-directional `text/event-stream` covers it
    completely over a normal POST request, with no protocol upgrade
    and no extra client library, unlike WebSockets. It also degrades
    gracefully: any HTTP client that doesn't speak SSE can still read
    the same bytes as a plain streamed response and parse the `data:`
    lines itself.

    Event shapes (Part 4):
      {"type": "stage_started",   "stage": ..., "message": ...}
      {"type": "stage_completed", "stage": ..., "message": ..., "duration_ms": ...}
      {"type": "error",           "stage": ..., "message": ...}
      {"type": "result", "data": <AnalyzeExperimentResponse JSON>}   (final, once, on success)
      {"type": "pipeline_completed"}                                (always sent last)

    The graph still runs via `asyncio.to_thread` inside
    `_execute_analysis`, so the event loop stays free to keep sending
    events; `RunContext.emit` bridges events from that worker thread
    back to this coroutine's `asyncio.Queue` via
    `loop.call_soon_threadsafe` (an `asyncio.Queue` is not safe to
    write to directly from another thread).
    """
    loop = asyncio.get_event_loop()
    event_queue: "asyncio.Queue[dict]" = asyncio.Queue()
    _SENTINEL: dict = {"type": "__sentinel__"}

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, event)

    run_context = RunContext(run_id=str(uuid.uuid4()), emit=emit)

    async def run_and_finish() -> None:
        try:
            response = await _execute_analysis(request, run_context)
        except HTTPException as exc:
            # Only emit a generic "pipeline"-level error if no node
            # already reported this failure at the stage level (see
            # `RunContext.error_emitted`'s docstring). A dataset
            # classification failure, for instance, raises inside
            # `classifier`, which `instrument_node` already turns into
            # one stage-level `error` event before re-raising —
            # emitting a second, generic one here for the exact same
            # failure would show the user two error events for one
            # real problem. A 404 on an unknown dataset_id, by
            # contrast, is raised before any node runs at all, so
            # `error_emitted` is still False and this is the only
            # error event sent.
            if not run_context.error_emitted:
                await event_queue.put({"type": "error", "stage": "pipeline", "message": str(exc.detail)})
        except Exception as exc:  # pragma: no cover - defensive: never leave the client hanging
            log.exception("[API] streaming analyze failed")
            if not run_context.error_emitted:
                await event_queue.put({"type": "error", "stage": "pipeline", "message": str(exc)})
        else:
            # Awaited directly on this same coroutine, in this exact
            # order — result THEN pipeline_completed — rather than via
            # `RunContext.emit`'s `call_soon_threadsafe` bridge (used
            # by every stage_started/stage_completed event above, since
            # those originate on the worker thread running the graph).
            # Bridging this pair the same way would race a
            # thread-scheduled callback against a directly-awaited
            # queue put and could deliver them out of order — see
            # RunContext.pipeline_completed's docstring.
            await event_queue.put(
                {"type": "result", "data": json.loads(response.model_dump_json(by_alias=True))}
            )
        finally:
            # Sent unconditionally — success, a handled error, or a
            # client disconnect (see event_stream()'s try/finally
            # below) — so the frontend never sees a stage stuck
            # "running" forever (Part 6/8's requirement to not leave
            # the UI permanently showing a running stage).
            await event_queue.put({"type": "pipeline_completed"})
            await event_queue.put(_SENTINEL)

    async def event_stream():
        task = asyncio.create_task(run_and_finish())
        try:
            while True:
                event = await event_queue.get()
                if event is _SENTINEL:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            # Part 8 — client disconnect: FastAPI/Starlette stops
            # iterating this generator when the connection drops. The
            # pipeline itself (running on the worker thread via
            # asyncio.to_thread inside `_execute_analysis`) is
            # deliberately NOT cancelled — work already in flight
            # finishes and is still persisted to Experiment History via
            # the normal `_store().create(...)` call, exactly as a
            # synchronous /analyze request would; only the progress
            # events stop being read. Awaiting the task here (shielded
            # from this generator's own cancellation) just ensures its
            # exceptions, if any slipped past the try/except above,
            # don't surface as an unretrieved-exception warning after
            # we've stopped reading the queue.
            if not task.done():
                await asyncio.shield(asyncio.gather(task, return_exceptions=True))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/related/by-dataset", response_model=list[RelatedExperiment])
async def get_related_by_dataset(dataset_id: str) -> list[RelatedExperiment]:
    """
    Prior runs against this exact dataset, looked up right after
    classification — before the user even hits Run. Same structured
    ExperimentStore.list_related() lookup used post-analyze; exposed
    separately so the New Experiment screen can show "reviewed before"
    context up front instead of only after a fresh run completes.

    Keyed by dataset_id (see list_related's docstring for why —
    dataset_name is a display label, not an identity, and two unrelated
    uploads can share a file name).
    """
    related = _store().list_related(dataset_id, limit=5)
    return [_to_related(r) for r in related]


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments() -> list[ExperimentSummary]:
    """Experiment History list — most recent first, no report body."""
    return [
        ExperimentSummary(
            experiment_id=r.experiment_id,
            created_at=r.created_at,
            dataset_name=r.dataset_name,
            user_prompt=r.user_prompt,
            decision=r.decision,
            confidence=r.confidence,
            primary_metric=r.primary_metric,
        )
        for r in _store().list_summaries()
    ]


@router.get("/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(experiment_id: str) -> ExperimentDetail:
    """Reopen a previous experiment: full report + steps + saved chat."""
    record = _store().get(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {experiment_id}")
    related = _store().list_related(
        record.dataset_id, exclude_experiment_id=experiment_id, limit=5
    )
    return ExperimentDetail(
        experiment_id=record.experiment_id,
        created_at=record.created_at,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        user_prompt=record.user_prompt,
        report=record.report,
        execution_steps=record.execution_steps,
        chat_messages=_store().list_chat_messages(experiment_id),
        related_experiments=[_to_related(r) for r in related],
    )


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str) -> dict[str, bool]:
    deleted = _store().delete(experiment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {experiment_id}")
    return {"deleted": True}


def _build_run_metadata(
    request: "AnalyzeExperimentRequest",
    final_state: dict,
    execution_steps: list[ExecutionStep],
    report: ExperimentReport | None = None,
) -> RunMetadata:
    """
    Phase 8 — structured run metadata (see RunMetadata's docstring).
    Every value here is read from facts already computed elsewhere in
    the pipeline; nothing is recomputed. Deliberately excludes the raw
    dataset and any per-user data.
    """
    dataset = final_state["dataset"]
    plan = final_state["plan"]

    # Worst status across the real trace — SUCCESS < SKIPPED < WARNING
    # < FAILED (SKIPPED is intentional, so it never outranks a real
    # WARNING/FAILED; it does outrank plain SUCCESS so a skip is still
    # visible at the run level) — EXCEPT for optional/exploratory
    # capability stages (Segmentation, Funnel Analysis) whose SKIPPED
    # status is itself a normal, healthy outcome — e.g. "no usable
    # segmentation dimensions" — that never affects the primary
    # decision (see their own "exploratory ... does not override the
    # primary experiment decision" framing in the report itself, and
    # _segmentation_step's docstring). Counting one of THEIR skips as
    # the run's worst status made a fully successful run (LLM report
    # generated, `report.report_fallback_reason` empty, every other
    # step SUCCESS) surface as top-level `Run Status: SKIPPED` — read
    # by a user as "the whole analysis was skipped", when in fact only
    # a single, optional, non-decision-affecting capability was. Every
    # OTHER SKIPPED step (e.g. the hypothesis test itself not having
    # run) still counts here, since that reflects the primary result
    # being absent, not merely a supplementary one.
    severity = {
        ExecutionStepStatus.SUCCESS: 0,
        ExecutionStepStatus.SKIPPED: 1,
        ExecutionStepStatus.WARNING: 2,
        ExecutionStepStatus.FAILED: 3,
    }
    non_critical_skip_step_ids = {"segmentation", "funnel"}
    worst = max(
        (
            s.status
            for s in execution_steps
            if not (s.status == ExecutionStepStatus.SKIPPED and s.id in non_critical_skip_step_ids)
        ),
        key=lambda s: severity[s],
        default=ExecutionStepStatus.SUCCESS,
    )

    settings = request.settings
    llm_status = plan.get("llm_status", "not_used")
    selected_model = plan.get("llm_requested_model") if llm_status != "not_used" else None

    return RunMetadata(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_name=request.dataset_name or request.dataset_id,
        dataset_classification=dataset.type.value,
        user_count=dataset.users,
        variant_count=dataset.variants,
        primary_metric=dataset.metric_label,
        analysis_mode=plan["intent_label"],
        selected_model=selected_model or (settings.model if settings is not None else None),
        planner_backend=app_settings.planner_backend,
        report_backend=app_settings.report_backend,
        execution_status=worst,
        # Echoed from the report itself — see LLMUsage's docstring
        # (app/schemas/execution.py) and ExperimentReport.llm_usage's
        # comment (app/schemas/report.py) for why `report` is the
        # source of truth here, not something recomputed from
        # `final_state`. `report` is optional (defaults to None) only
        # so existing unit tests that construct RunMetadata directly
        # from execution_steps — without a full ExperimentReport —
        # don't have to fabricate one; the real request path
        # (analyze_experiment/analyze_experiment_stream below) always
        # passes it.
        llm_usage=report.llm_usage if report is not None else None,
    )


def _hypothesis_test_skip_reason(state: dict, plan: dict, srm) -> str:
    """
    Why the Experiment stage produced no hypothesis test, when neither a
    `test_selection` nor `stat_results` are present in the final state.

    This used to unconditionally say `Not needed for intent "<intent>"`,
    which is wrong whenever the REAL reason is that a validity gate
    (SRM failure or conflicting variant assignments) stopped the
    pipeline before Experiment ever ran — Full Experiment Review, for
    example, normally DOES require hypothesis testing. Each validity
    reason is reported using the exact structured fact already computed
    by validation_node (e.g. the quality-check's own count), never a
    number recomputed here.
    """
    if state.get("has_conflicting_variant_duplicates"):
        quality_checks = state.get("quality_checks") or []
        conflict_check = next(
            (qc for qc in quality_checks if qc.label == "Duplicate User Variant Conflicts"), None
        )
        conflict_detail = conflict_check.detail if conflict_check is not None else "conflicting variant assignments were detected"
        return (
            "Statistical testing was skipped because a critical experiment-validity check "
            f"failed: {conflict_detail} Comparing the variants would produce an unreliable result."
        )

    if srm is not None and not srm.passed:
        return (
            "Statistical testing was skipped because the Sample Ratio Mismatch (SRM) check "
            f"failed (p={srm.p_value:.3f}) — the observed variant split deviates from the "
            "expected allocation, so comparing the variants would produce an unreliable result."
        )

    critical_quality_failures = [qc for qc in (state.get("quality_checks") or []) if not qc.passed and qc.critical]
    if critical_quality_failures:
        labels = ", ".join(qc.label for qc in critical_quality_failures)
        return f"Statistical testing was skipped because of a critical data-quality failure: {labels}."

    return f'Not needed for intent "{plan["intent_label"]}"'


def _planner_step_detail(plan: dict) -> str:
    """
    Builds the execution-trace detail for the planner step, making a
    failed/fallback LLM planner call visible instead of looking identical
    to a normal keyword-routed request. `llm_status` comes from
    planner_strategy.py:
      - "not_used": KeywordPlanner ran (PLANNER_BACKEND=keyword, the
        default) — no LLM was involved at all, detail is unchanged.
      - "success": LLMPlanner's call to the requested model actually
        succeeded — the resolved model id is shown so "the selected
        model actually ran" is verifiable, not just assumed.
      - "fallback": the requested model failed and KeywordPlanner ran
        instead — this is called out explicitly, with the real error,
        so it never silently looks like the selected model worked.
    """
    base = f'Intent identified as "{plan["intent_label"]}"'
    status = plan.get("llm_status", "not_used")
    if status == "success":
        return f"{base} — LLM model: {plan.get('llm_requested_model')} (succeeded)"
    if status == "fallback":
        return (
            f"{base} — LLM model: {plan.get('llm_requested_model')} FAILED "
            f"({plan.get('llm_error')}); fell back to keyword-based planning"
        )
    return base


def _decision_step(report) -> ExecutionStep:
    """
    The Decision/Report Generation step's status reflects
    what actually happened, not just "we produced some report":
      - FAILED: never used here — a hard report-generation exception is
        always caught and turned into the emergency fallback report
        (see build_emergency_fallback_report), which is a WARNING, not
        a FAILED step — the pipeline as a whole still completed.
      - WARNING: the report generator silently fell back (LLM report
        generation failed, or the emergency fallback path was used) —
        see ExperimentReport.report_fallback_reason.
      - SUCCESS: otherwise, including ordinary NO_GO/INVALID decisions
        — an experiment correctly being told "don't ship this" is a
        successful run of the pipeline, not a failure of the pipeline.
    """
    if report.report_fallback_reason:
        return ExecutionStep(
            id="decision",
            label="Report Generation",
            group=ExecutionStepGroup.DECISION_ENGINE,
            detail=f"Confidence: {report.confidence.value} ({report.confidence_stars} stars) — {report.report_fallback_reason}",
            status=ExecutionStepStatus.WARNING,
        )
    return ExecutionStep(
        id="decision",
        label="Report Generation",
        group=ExecutionStepGroup.DECISION_ENGINE,
        detail=f"Confidence: {report.confidence.value} ({report.confidence_stars} stars)",
        status=ExecutionStepStatus.SUCCESS,
    )


def _segmentation_step(state: dict) -> ExecutionStep | None:
    """
    Segmentation is only represented in the execution trace
    when experiment_node actually ran (two-arm path);
    absent entirely on validation-only / SRM-failed / multi-arm runs,
    same conditional-inclusion pattern the rest of this function uses.
    """
    segmentation = state.get("segmentation_result")
    if segmentation is None:
        return None
    if not segmentation.ran:
        return ExecutionStep(
            id="segmentation",
            label="Segmentation — Skipped",
            group=ExecutionStepGroup.CAPABILITY,
            detail=segmentation.reason,
            status=ExecutionStepStatus.SKIPPED,
        )
    return ExecutionStep(
        id="segmentation",
        label="Segmentation Analysis",
        group=ExecutionStepGroup.CAPABILITY,
        detail=f"{len(segmentation.dimension_results)} dimension(s) analyzed — {segmentation.reason}",
        status=ExecutionStepStatus.SUCCESS,
    )


def _build_execution_steps(state: dict) -> list[ExecutionStep]:
    """
    Builds the post-hoc execution trace from the final graph state.
    Real `detail` text drawn from what each node actually computed —
    not the frontend's hardcoded mock strings. Steps are conditionally
    included based on which nodes the graph actually ran: a pure
    conceptual question skips Validation and Experiment
    entirely and only shows Classifier → Planner → Knowledge Base →
    Decision.

    Every step also carries a real `status`
    (ExecutionStepStatus) alongside its `detail` text — see
    ExecutionStep.status's docstring for how this differs from the
    frontend's fake-timer StepStatus.
    """
    dataset = state["dataset"]
    plan = state["plan"]
    report = state["report"]
    srm = state.get("srm_result")
    test_selection = state.get("test_selection")
    kb_results = state.get("kb_results")
    kb_error = state.get("kb_error")
    funnel_result = state.get("funnel_result")
    funnel_skip_reason = state.get("funnel_skip_reason")

    llm_status = plan.get("llm_status", "not_used")
    planner_status = ExecutionStepStatus.WARNING if llm_status == "fallback" else ExecutionStepStatus.SUCCESS

    steps = [
        ExecutionStep(
            id="classifier",
            label="Dataset Classification",
            group=ExecutionStepGroup.CLASSIFIER,
            detail=(
                f"Detected {describe_dataset_structure(dataset, state.get('experiment_columns') is not None)}"
                f" — {dataset.users:,} users, {dataset.variants} variants"
            ),
            status=ExecutionStepStatus.SUCCESS,
        ),
        ExecutionStep(
            id="planner",
            label="Intent Planning",
            group=ExecutionStepGroup.PLANNER,
            detail=_planner_step_detail(plan),
            status=planner_status,
        ),
    ]

    if funnel_result is not None:
        step_names = " → ".join(s.name for s in funnel_result.steps)
        steps.append(
            ExecutionStep(
                id="funnel",
                label="Funnel Analysis",
                group=ExecutionStepGroup.CAPABILITY,
                detail=(
                    f"{step_names} — largest drop-off {funnel_result.largest_dropoff_from} → "
                    f"{funnel_result.largest_dropoff_to} ({funnel_result.largest_dropoff_rate:.1%})"
                ),
                status=ExecutionStepStatus.SUCCESS,
            )
        )
        if srm is None:
            steps.append(_decision_step(report))
            return steps

    elif funnel_skip_reason is not None:
        steps.append(
            ExecutionStep(
                id="funnel",
                label="Funnel Analysis — Skipped",
                group=ExecutionStepGroup.CAPABILITY,
                detail=funnel_skip_reason,
                status=ExecutionStepStatus.SKIPPED,
            )
        )
        steps.append(_decision_step(report))
        return steps

    if kb_results is not None:
        # PHASE 8 — distinguish an actual retrieval FAILURE (kb_error
        # set) from a legitimate empty result (kb_error is None); see
        # knowledge_base_node.py and GraphState.kb_error's docstring.
        #
        # The trace must reflect the
        # same final, decision-filtered reference list the report
        # itself shows in Evidence & Sources — not the raw, unfiltered
        # `kb_results` retrieved before the INVALID-specific relevance
        # gate ran (requirement #9/#10/#11: single source of truth).
        # `report.knowledge_base_references` is that final list;
        # `report.knowledge_base_blocking_issue` names what the gate
        # filtered against when everything was dropped for an INVALID
        # decision, so the trace can be as specific as the UI is.
        final_kb_references = report.knowledge_base_references
        if kb_error is not None:
            kb_detail = f"Knowledge base retrieval failed ({kb_error}) — degraded gracefully with no references."
            kb_status = ExecutionStepStatus.WARNING
        elif final_kb_references:
            top = final_kb_references[0]
            kb_detail = f"Retrieved {len(final_kb_references)} reference(s), top match: {top.heading} (score={top.relevance_score:.2f})"
            kb_status = ExecutionStepStatus.SUCCESS
        else:
            # Honest about WHY nothing
            # is shown: candidates may have been retrieved and
            # discarded for scoring below stats_thresholds.kb_relevance_threshold,
            # or (decision-aware RAG) retrieved but filtered out as
            # irrelevant to the specific INVALID reason — not
            # necessarily "the KB has nothing on this topic at all".
            if report.knowledge_base_blocking_issue:
                kb_detail = (
                    "No sufficiently relevant evidence found in the knowledge base "
                    f"for: {report.knowledge_base_blocking_issue}"
                )
            else:
                kb_detail = "No sufficiently relevant evidence found in the knowledge base"
            kb_status = ExecutionStepStatus.SUCCESS
        steps.append(
            ExecutionStep(
                id="knowledge_base",
                label="Knowledge Base Retrieval",
                group=ExecutionStepGroup.CAPABILITY,
                detail=kb_detail,
                status=kb_status,
            )
        )
        # knowledge_base is NOT an exclusive path — validation
        # + experiment run right alongside it for a normal "should we
        # ship?" prompt (see graph_builder.py / test_graph_builder.py's
        # TestMethodologyRag), so this must never `return steps` here and
        # silently drop the Validation/Experiment/Segmentation steps from
        # the trace whenever knowledge_base ran. A pure conceptual
        # question (no derived dataset, srm never ran) still exits here.
        if srm is None:
            steps.append(_decision_step(report))
            return steps

    steps.append(
        ExecutionStep(
            id="validation",
            label="Data Quality Validation",
            group=ExecutionStepGroup.CAPABILITY,
            detail=f"SRM {'passed' if srm.passed else 'FAILED'} (p={srm.p_value:.3f})",
            status=ExecutionStepStatus.SUCCESS if srm.passed else ExecutionStepStatus.FAILED,
        )
    )

    if test_selection is not None:
        steps.append(
            ExecutionStep(
                id="experiment",
                label=test_selection_to_quality_check_detail(test_selection),
                group=ExecutionStepGroup.CAPABILITY,
                detail=test_selection.reason,
                status=ExecutionStepStatus.SUCCESS,
            )
        )
    elif state.get("stat_results"):
        stat_results = state["stat_results"]
        omnibus = stat_results[0]
        pairwise_count = sum(1 for result in stat_results if not result.is_omnibus)
        steps.append(
            ExecutionStep(
                id="experiment",
                label=omnibus.test_name,
                group=ExecutionStepGroup.CAPABILITY,
                detail=(
                    f"{omnibus.test_name} across {dataset.variants} variants "
                    f"(p={omnibus.p_value:.4g}); {pairwise_count} corrected pairwise comparison(s) "
                    "run when the omnibus test was significant."
                ),
                status=ExecutionStepStatus.SUCCESS,
            )
        )
    else:
        skip_reason = _hypothesis_test_skip_reason(state, plan, srm)
        # A validity-gate skip (SRM failure / conflicting duplicates /
        # critical quality failure) is a genuinely different situation
        # from "the planner just didn't request a hypothesis test" —
        # the former is downstream of an actual FAILED validation step
        # above, so marking it WARNING (rather than a plain SKIPPED)
        # keeps it visually distinct from an intentional, healthy skip.
        gated = bool(state.get("has_conflicting_variant_duplicates")) or (srm is not None and not srm.passed)
        steps.append(
            ExecutionStep(
                id="experiment",
                label="Hypothesis Test — Skipped",
                group=ExecutionStepGroup.CAPABILITY,
                detail=skip_reason,
                status=ExecutionStepStatus.WARNING if gated else ExecutionStepStatus.SKIPPED,
            )
        )

    segmentation_step = _segmentation_step(state)
    if segmentation_step is not None:
        steps.append(segmentation_step)

    steps.append(_decision_step(report))
    return steps


@router.get("/{experiment_id}/chat", response_model=list[ChatMessage])
async def get_chat_history(experiment_id: str) -> list[ChatMessage]:
    """Persisted conversation for an experiment, reloaded on reopen."""
    if _store().get(experiment_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {experiment_id}")
    return _store().list_chat_messages(experiment_id)


@router.post(
    "/{experiment_id}/chat",
    response_model=FollowUpChatResponse,
    dependencies=[Depends(rate_limit("chat", max_requests=20))],
)
async def follow_up_chat(experiment_id: str, request: FollowUpChatRequest) -> FollowUpChatResponse:
    """
    Answer a follow-up question grounded in the stored report + stats
    for this experiment, AND the prior conversation for this
    experiment_id. Never recomputes numbers — the LLM (or the
    deterministic TemplateChatResponder, depending on REPORT_BACKEND)
    only synthesizes an answer from already-computed facts in the
    stored ExperimentReport, plus already-persisted chat turns. See
    app/graph/chat_generator.py for the responder implementations and
    the grounding guarantee. The raw dataset is never loaded or
    touched by this route.
    """
    store = _store()
    record = store.get(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {experiment_id}")

    # Prior turns for THIS experiment, oldest-first, read before the
    # new user message is added below so it isn't duplicated in the
    # history passed to the responder. Trimmed to a reasonable, fixed
    # window — no unbounded prompt growth over a long conversation.
    prior_history = store.list_chat_messages(experiment_id)[-DEFAULT_MAX_HISTORY_MESSAGES:]

    user_message = ChatMessage(
        id=f"u-{uuid.uuid4().hex[:12]}", role=ChatRole.USER, content=request.message
    )
    store.add_chat_message(experiment_id, user_message)

    try:
        assistant_message = build_chat_message(record.report, request.message, prior_history, model=request.model)
    except Exception as exc:  # noqa: BLE001 — LLM failures already fall back internally (chat_generator.py);
        # reaching here means even the deterministic TemplateChatResponder failed, which is a real bug,
        # not a "no LLM configured" situation — log the full detail server-side but never
        # echo raw exception text (which may include provider/internal detail) to the client.
        log.error("[API] Chat response generation failed for experiment_id=%s: %s", experiment_id, exc)
        raise HTTPException(status_code=500, detail="Failed to generate a chat response. Please try again.") from exc

    store.add_chat_message(experiment_id, assistant_message)

    return FollowUpChatResponse(message=assistant_message)


_STREAM_DONE = object()  # sentinel passed to next(iterator, default) in follow_up_chat_stream


@router.post(
    "/{experiment_id}/chat/stream",
    dependencies=[Depends(rate_limit("chat", max_requests=20))],
)
async def follow_up_chat_stream(experiment_id: str, request: FollowUpChatRequest) -> StreamingResponse:
    """
    Same grounding/persistence contract as POST /{experiment_id}/chat
    (see that docstring), streamed as Server-Sent Events so the client
    can render tokens as they arrive instead of waiting for the whole
    answer — same SSE approach and same rationale (one-directional,
    single terminal result, degrades gracefully for any HTTP client)
    as POST /experiments/analyze/stream.

    Event shapes:
      {"type": "token", "content": "..."}                     (zero or more, in order)
      {"type": "done", "message": <ChatMessage JSON>}          (terminal, on success)
      {"type": "error", "message": "..."}                      (terminal, only on failure)

    The user's message is persisted up front, exactly like the
    non-streaming route. The assistant's message is persisted once the
    generator finishes — on a clean run that's after every token; on a
    MID-stream LLM failure (see `stream_chat_response`'s docstring)
    it's whatever text was actually generated before the failure, so a
    dropped connection never silently loses the turn from history the
    way not persisting anything would.
    """
    store = _store()
    record = store.get(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {experiment_id}")

    prior_history = store.list_chat_messages(experiment_id)[-DEFAULT_MAX_HISTORY_MESSAGES:]

    user_message = ChatMessage(
        id=f"u-{uuid.uuid4().hex[:12]}", role=ChatRole.USER, content=request.message
    )
    store.add_chat_message(experiment_id, user_message)

    async def event_stream():
        chunks: list[str] = []
        try:
            # `stream_chat_response` is a plain (sync) generator — the
            # underlying LangChain `.stream()` call is blocking network
            # I/O per chunk, so it's iterated via `asyncio.to_thread`
            # per step rather than directly in this coroutine, which
            # would otherwise stall the event loop for every network
            # read the same way a synchronous `requests.get` would.
            iterator = stream_chat_response(record.report, request.message, prior_history, model=request.model)
            while True:
                chunk = await asyncio.to_thread(next, iterator, _STREAM_DONE)
                if chunk is _STREAM_DONE:
                    break
                chunks.append(chunk)
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        except Exception as exc:  # noqa: BLE001 — must still persist partial content + tell the client
            log.error("[API] Streaming chat response failed for experiment_id=%s: %s", experiment_id, exc)
            yield f"data: {json.dumps({'type': 'error', 'message': 'The response was interrupted. Please try again.'})}\n\n"

        assistant_message = finalize_streamed_chat_message("".join(chunks))
        store.add_chat_message(experiment_id, assistant_message)
        yield f"data: {json.dumps({'type': 'done', 'message': json.loads(assistant_message.model_dump_json(by_alias=True))})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
