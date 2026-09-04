"""
Experiment History API schemas.

These wrap `ExperimentRecord` (app/core/experiment_store.py) for the
wire — kept separate from the store's plain record class so the store
itself has no Pydantic/FastAPI dependency.
"""

from datetime import datetime

from app.schemas.base import CamelModel
from app.schemas.chat import ChatMessage
from app.schemas.execution import ExecutionStep
from app.schemas.report import ExperimentReport


class ExperimentSummary(CamelModel):
    """One row in the History list — no report body, keeps the list light."""

    experiment_id: str
    created_at: datetime
    dataset_name: str
    user_prompt: str
    decision: str
    confidence: str
    primary_metric: str


class RelatedExperiment(CamelModel):
    """
    One prior run against the same dataset — plain structured retrieval
    from ExperimentStore.list_related(), not LLM/semantic memory. Shown
    to the analyst as short factual context ("this dataset was reviewed
    before, decision was X") when starting a new analysis.
    """

    experiment_id: str
    created_at: datetime
    user_prompt: str
    decision: str
    confidence: str
    primary_metric: str


class ExperimentDetail(CamelModel):
    """Full reopened experiment — report + steps + persisted chat."""

    experiment_id: str
    created_at: datetime
    dataset_id: str
    dataset_name: str
    user_prompt: str
    report: ExperimentReport
    execution_steps: list[ExecutionStep]
    chat_messages: list[ChatMessage]
    related_experiments: list[RelatedExperiment] = []


class SystemInfo(CamelModel):
    """
    Read-only backend configuration the frontend is allowed to display.

    The LLM model is configured via `.env` (see app/core/config.py) —
    there is no runtime model-switching endpoint. This exists so the UI
    can show the user what's actually running instead of a dropdown
    with no effect.
    """

    llm_provider: str
    llm_model: str
    planner_backend: str
    report_backend: str
    # --- Phase 8: safe, cheap operational visibility --------------------
    # Deliberately excludes anything from the "DO NOT expose" list in
    # the Phase 8 spec (API keys, secrets, credentials, connection
    # strings). `knowledge_base_available` calls the existing,
    # in-memory, @lru_cache'd get_retriever() — no external I/O, no new
    # health-check machinery — so this stays cheap on every page load.
    knowledge_base_available: bool
    available_models_count: int
