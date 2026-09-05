"""
POST   /experiment-definitions
GET    /experiment-definitions
GET    /experiment-definitions/{definition_id}
PATCH  /experiment-definitions/{definition_id}
DELETE /experiment-definitions/{definition_id}

Phase 1 of the Experiment Platform layer (see Stage 0 architecture
doc): plain CRUD over `ExperimentDefinition`, the new pre-analysis
planning entity. Deliberately does NOT touch the existing analysis
pipeline in this phase — there is no "analyze this definition"
endpoint yet (that's Phase 6). This router only persists and returns
definitions; `app/graph/*` and `app/stats/*` are untouched and never
imported here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.experiment_definition_store import (
    ExperimentDefinitionStore,
    get_experiment_definition_store,
)
from app.schemas.experiment_definition import (
    ExperimentDefinition,
    ExperimentDefinitionCreateRequest,
    ExperimentDefinitionSummary,
    ExperimentDefinitionUpdateRequest,
)

router = APIRouter(prefix="/experiment-definitions", tags=["experiment-definitions"])


def _store() -> ExperimentDefinitionStore:
    return get_experiment_definition_store()


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
