"""
Execution pipeline schemas.

Mirrors lib/types.ts:

    export type StepStatus = 'pending' | 'running' | 'done';

    export interface ExecutionStep {
      id: string;
      label: string;
      group: 'Classifier' | 'Planner' | 'Capability' | 'Decision Engine';
      detail: string;
      status?: ExecutionStepStatus;
    }

Per the sprint decision: NO live streaming this sprint. We still model
this contract because the synchronous /experiments/analyze response
should return the final `steps` (real `detail` text reflecting what
actually happened) so the frontend's ExecutionStepper can render real
data instead of the hardcoded EXECUTION_STEPS mock.

PHASE 8 UPDATE — `ExecutionStep.status` added: this is a DIFFERENT
concept from `StepStatus` below. `StepStatus` is the frontend's
fake-timer "is this row's spinner still going" concept
(pending/running/done), driven client-side before the synchronous
/analyze response even arrives — that stays a purely frontend concern,
unchanged. `ExecutionStepStatus` is the REAL, backend-computed outcome
of a pipeline stage once the graph has actually finished — did it
succeed, get intentionally skipped, produce a warning (e.g. a
graceful fallback occurred), or fail outright — and it travels all
the way from GraphState to the frontend (see
routes_experiments.py's `_build_execution_steps`). It defaults to
SUCCESS so nothing that only ever set `detail` before Phase 8 breaks.
"""

from enum import Enum

from app.schemas.base import CamelModel


class StepStatus(str, Enum):
    """
    Kept here (not deleted) because it still mirrors types.ts's
    `StepStatus` and will be the payload shape for a future streaming
    channel — just not a field on `ExecutionStep` itself.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


class ExecutionStepGroup(str, Enum):
    CLASSIFIER = "Classifier"
    PLANNER = "Planner"
    CAPABILITY = "Capability"
    DECISION_ENGINE = "Decision Engine"


class ExecutionStepStatus(str, Enum):
    """
    Real (not fake-timer) outcome of a single pipeline stage, computed
    from the actual final GraphState — see module docstring above.
    """

    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    WARNING = "WARNING"
    FAILED = "FAILED"


class ExecutionStep(CamelModel):
    """Mirrors types.ts's ExecutionStep, plus Phase 8's `status` field."""

    id: str
    label: str
    group: ExecutionStepGroup
    detail: str
    status: ExecutionStepStatus = ExecutionStepStatus.SUCCESS


class LLMUsage(CamelModel):
    """
    Token/cost accounting for a single LLM call made during report
    generation (LLMReportGenerator's OpenRouter call — the one call in
    the pipeline actually worth metering; the Planner's keyword-vs-LLM
    choice is comparatively tiny). Every field is optional because not
    every OpenRouter model/response includes every value:
    `prompt_tokens`/`completion_tokens`/`total_tokens` come from the
    OpenAI-standard `usage` object nearly every provider returns;
    `cost_usd` is an OpenRouter-specific extension that is only present
    when the account/response actually includes it, so it is commonly
    None even when token counts are present. None (the whole object,
    on `ExperimentReport.llm_usage`) means no LLM call happened for
    this run at all (template/keyword backends, or the safety-gate/
    conceptual paths that never call an LLM) — that is different from
    an LLM call happening but returning no usage data, which this
    object would represent with all-None fields.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


class RunMetadata(CamelModel):
    """
    Phase 8 — structured run/execution metadata for a single
    `/experiments/analyze` invocation. Deliberately does NOT store the
    raw dataset or any sensitive user information — every field here
    is a small scalar summary already available elsewhere in the
    pipeline (DatasetInfo, AnalysisSettings, AppSettings, PlannerOutput).

    This is observational only: nothing here is read by any node to
    make a decision, and nothing here duplicates or re-derives a
    statistical result — nothing is computed twice.
    """

    run_id: str
    timestamp: str  # ISO 8601, set once when the run completes
    dataset_name: str
    dataset_classification: str
    user_count: int
    variant_count: int
    primary_metric: str | None = None
    analysis_mode: str  # the Planner's `intent_label`
    selected_model: str | None = None  # only meaningful when an LLM was actually invoked
    planner_backend: str  # "keyword" | "llm" (server config)
    report_backend: str  # "template" | "llm" (server config)
    execution_status: ExecutionStepStatus  # worst status across all execution_steps

    # Token/cost accounting for the report-generation LLM call, echoed
    # straight from `ExperimentReport.llm_usage` (see LLMUsage's
    # docstring above for why every field is independently optional).
    # None whenever no LLM call was attempted for this run at all —
    # e.g. REPORT_BACKEND=template, or the safety-gate/conceptual
    # paths in LLMReportGenerator.generate() that return the
    # deterministic template report without ever calling an LLM.
    llm_usage: LLMUsage | None = None
