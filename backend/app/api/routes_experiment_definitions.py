"""
POST   /experiment-definitions
GET    /experiment-definitions
GET    /experiment-definitions/{definition_id}
PATCH  /experiment-definitions/{definition_id}
DELETE /experiment-definitions/{definition_id}
POST   /experiment-definitions/{definition_id}/analyze
GET    /experiment-definitions/{definition_id}/runs

Phase 1 of the Experiment Platform layer (see Stage 0 architecture
doc): plain CRUD over `ExperimentDefinition`, the new pre-analysis
planning entity.

Phase 8 ("Data Source -> existing analysis engine") adds the single
`analyze` endpoint below. It deliberately does NOT introduce a second
analysis pipeline: it only maps a definition's `data_source` +
primary hypothesis onto the SAME `AnalyzeExperimentRequest` that
`POST /experiments/analyze` already accepts, and calls straight into
that route module's shared `_execute_analysis`. `app/graph/*` and
`app/stats/*` are still never imported here directly.

Phase 10 ("History") adds `GET /{definition_id}/runs` — every
AnalysisRun launched from a definition, most recent first, via
`ExperimentStore.list_by_definition`.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from app.api.routes_experiments import (
    AnalyzeExperimentRequest,
    AnalyzeExperimentResponse,
    _execute_analysis,
)
from app.core.experiment_definition_store import (
    ExperimentDefinitionStore,
    get_experiment_definition_store,
)
from app.core.experiment_store import ExperimentStore, get_experiment_store
from app.core.pipeline_events import RunContext
from app.schemas.base import CamelModel
from app.schemas.experiment_definition import (
    ExperimentDefinition,
    ExperimentDefinitionCreateRequest,
    ExperimentDefinitionSummary,
    ExperimentDefinitionUpdateRequest,
    HypothesisRole,
)
from app.schemas.experiment_history import ExperimentSummary
from app.schemas.settings import AnalysisSettings

router = APIRouter(prefix="/experiment-definitions", tags=["experiment-definitions"])


def _store() -> ExperimentDefinitionStore:
    return get_experiment_definition_store()


def _run_store() -> ExperimentStore:
    return get_experiment_store()


@router.post("", response_model=ExperimentDefinition)
async def create_experiment_definition(
    request: ExperimentDefinitionCreateRequest,
) -> ExperimentDefinition:
    """Creates a new ExperimentDefinition (status defaults to DRAFT)."""
    return _store().create(request)


@router.get("", response_model=list[ExperimentDefinitionSummary])
async def list_experiment_definitions() -> list[ExperimentDefinitionSummary]:
    """Experiment Library list — most recently updated first."""
    return _store().list_summaries()


@router.get("/{definition_id}", response_model=ExperimentDefinition)
async def get_experiment_definition(definition_id: str) -> ExperimentDefinition:
    record = _store().get(definition_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment definition id: {definition_id}")
    return record


@router.patch("/{definition_id}", response_model=ExperimentDefinition)
async def update_experiment_definition(
    definition_id: str, request: ExperimentDefinitionUpdateRequest
) -> ExperimentDefinition:
    """
    Partial update — only fields present in the request body are
    changed (see ExperimentDefinitionUpdateRequest's docstring for the
    exact exclude_unset semantics). Used by the Design/Variants/
    Targeting/Metrics forms to save incrementally without resending
    the whole definition each time.
    """
    updated = _store().update(definition_id, request)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment definition id: {definition_id}")
    return updated


@router.delete("/{definition_id}")
async def delete_experiment_definition(definition_id: str) -> dict[str, bool]:
    deleted = _store().delete(definition_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown experiment definition id: {definition_id}")
    return {"deleted": True}


class AnalyzeDefinitionRequest(CamelModel):
    """
    Body for POST /experiment-definitions/{id}/analyze.

    Deliberately tiny — everything the analysis actually needs
    (dataset, hypothesis) already lives on the saved
    `ExperimentDefinition` (see Phase 7/8's Metrics/Data Source forms).
    Only the two things that are still a per-run choice on the
    existing `/experiments/analyze` flow (CUPED/bootstrap toggles,
    model) are passed in here, same shape as `AnalyzeExperimentRequest.
    settings`.
    """

    settings: AnalysisSettings
    # Optional override for the free-text prompt sent to the pipeline.
    # When omitted, falls back to the definition's objective, then its
    # problem statement, then a generic default — see
    # `_prompt_from_definition` below.
    prompt: str | None = None


def _prompt_from_definition(definition: ExperimentDefinition, override: str | None) -> str:
    if override and override.strip():
        return override.strip()
    if definition.objective:
        return definition.objective
    if definition.problem_statement:
        return definition.problem_statement
    return f"Analyze the {definition.name} experiment and recommend a decision."


@router.post("/{definition_id}/analyze", response_model=AnalyzeExperimentResponse)
async def analyze_experiment_definition(
    definition_id: str, request: AnalyzeDefinitionRequest
) -> AnalyzeExperimentResponse:
    """
    Phase 8 — "Data Source -> existing analysis engine".

        ExperimentDefinition
                |
          selected dataset (definition.data_source)
                |
          EXISTING ANALYSIS ENGINE (/experiments/analyze's _execute_analysis)
                |
          Decision Scientist
                |
             Report

    This endpoint does no analysis itself: it reads the definition,
    pulls `data_source.dataset_id` and the one PRIMARY hypothesis (if
    any) off it, builds a normal `AnalyzeExperimentRequest`, and calls
    the exact same `_execute_analysis` that `POST /experiments/analyze`
    uses — so CUPED/bootstrap/guardrails/segmentation/decision logic
    all behave identically regardless of which route the run started
    from. The only thing added is `definition_id`, threaded through so
    the persisted run links back to this definition (Phase 10 —
    `GET /{definition_id}/runs`).
    """
    definition = _store().get(definition_id)
    if definition is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment definition id: {definition_id}")

    if not definition.data_source or not definition.data_source.dataset_id:
        raise HTTPException(
            status_code=422,
            detail="This experiment has no data source yet — connect a dataset before analyzing.",
        )

    primary_hypothesis = next(
        (h.hypothesis for h in definition.hypotheses if h.role == HypothesisRole.PRIMARY),
        None,
    )

    analyze_request = AnalyzeExperimentRequest(
        dataset_id=definition.data_source.dataset_id,
        dataset_name=definition.data_source.dataset_name or definition.name,
        prompt=_prompt_from_definition(definition, request.prompt),
        settings=request.settings,
        hypothesis=primary_hypothesis,
    )
    run_context = RunContext(run_id=str(uuid.uuid4()))
    return await _execute_analysis(analyze_request, run_context, definition_id=definition_id)


@router.get("/{definition_id}/runs", response_model=list[ExperimentSummary])
async def list_experiment_definition_runs(definition_id: str) -> list[ExperimentSummary]:
    """
    Phase 10 — History. Every AnalysisRun launched from this
    definition (via the `analyze` endpoint above), most recent first —
    a definition can have several (e.g. "initial analysis", "extended
    data", "final analysis"), each independently reopenable through
    the existing `GET /experiments/{experiment_id}`.
    """
    if _store().get(definition_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment definition id: {definition_id}")

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
        for r in _run_store().list_by_definition(definition_id)
    ]
