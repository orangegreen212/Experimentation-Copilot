"""
GET /system/info
GET /system/models

Read-only backend configuration for the UI, plus the curated list of
selectable LLM models (see AppSettings.available_llm_models /
app.llm.client.resolve_model()). The model IS user-selectable at
runtime now — per experiment run (AnalyzeExperimentRequest.settings.model)
and per follow-up chat message (FollowUpChatRequest.model) — but only
from this fixed, server-curated list; the frontend never posts an
arbitrary model string, and the backend re-validates it anyway
(resolve_model() falls back to AppSettings.llm_model for anything not
in this list).
"""

from fastapi import APIRouter

from app.core.config import app_settings
from app.core.logging import get_node_logger
from app.rag.retriever import get_retriever
from app.schemas.experiment_history import SystemInfo
from app.schemas.system import AvailableModel, AvailableModelsResponse

router = APIRouter(prefix="/system", tags=["system"])
log = get_node_logger("System")


@router.get("/info", response_model=SystemInfo)
async def get_system_info() -> SystemInfo:
    # PHASE 8 — cheap, safe operational visibility. get_retriever() is
    # @lru_cache'd and builds an in-memory TF-IDF index over the static
    # local knowledge-base markdown files — no network call, no
    # per-request cost after the first hit, so this is safe to call on
    # every /system/info request. If it ever does fail (e.g. missing
    # knowledge-base files in this deployment), that's real operational
    # information the UI should know, not something to hide.
    try:
        get_retriever()
        kb_available = True
    except Exception as exc:  # noqa: BLE001 — reporting availability must never itself fail the endpoint
        log.warning("[System] Knowledge base unavailable (%s).", exc)
        kb_available = False

    return SystemInfo(
        llm_provider=app_settings.llm_provider,
        llm_model=app_settings.llm_model,
        planner_backend=app_settings.planner_backend,
        report_backend=app_settings.report_backend,
        knowledge_base_available=kb_available,
        available_models_count=len(app_settings.available_llm_models),
    )


@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models() -> AvailableModelsResponse:
    # `default_model` always resolves to a value that is itself a member
    # of `models` below — see the startup guardrail in app/core/config.py
    # that warns if `AppSettings.llm_model` ever drifts outside
    # `available_llm_models`, which would otherwise let the UI's
    # "Backend default" label point at an id the dropdown never offers.
    #
    # NOTE on live validation: this deliberately does NOT make a live
    # OpenRouter request per model to "verify" availability on every
    # page load — that would cost real tokens/latency on every render
    # for zero functional benefit (a model can still rate-limit or fail
    # on the very next call regardless of a health check a moment
    # earlier). Instead, real invocation failures are now surfaced
    # explicitly and per-request via `plan.llm_status` /
    # `plan.llm_error` on the Planner step of the execution trace (see
    # routes_experiments.py's `_planner_step_detail`) — a model that
    # fails is never silently reported as having worked.
    return AvailableModelsResponse(
        models=[AvailableModel(id=m["id"], label=m["label"]) for m in app_settings.available_llm_models],
        default_model=app_settings.llm_model,
    )
