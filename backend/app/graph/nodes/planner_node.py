"""
Planner node — thin adapter, no routing logic here. All decision logic
lives in app/graph/planner_strategy.py behind the `Planner` Protocol
(same Strategy pattern as decision_node.py + report_generator.py).
This node just calls get_planner().plan() and logs the result.
"""

from app.core.dataset_store import get_dataset
from app.core.logging import get_node_logger
from app.graph.planner_strategy import (
    get_planner,
    plan_from_explicit_settings,
    plan_from_free_text_stratification_request,
)
from app.graph.state import GraphState
from app.schemas.settings import AnalysisSettings

log = get_node_logger("Planner")


def _settings_with_stratification(settings: AnalysisSettings | None, column: str | None) -> AnalysisSettings:
    """
    Project a detected free-text stratification request onto the SAME
    AnalysisSettings fields the structured-UI path already uses
    (analysis_mode / stratification_column), so experiment_node needs
    no changes at all to pick it up — it already reads exactly these
    two fields. `column` may be None (stratified analysis was clearly
    requested, but no "by/using/on <column>" clause could be parsed);
    experiment_node's existing `and settings.stratification_column`
    gate already treats a falsy column as "nothing to run", which is
    the correct behavior here too (no column to run stratification on).
    """
    base = settings if settings is not None else AnalysisSettings()
    return base.model_copy(update={"analysis_mode": "stratified", "stratification_column": column})


def planner_node(state: GraphState) -> GraphState:
    settings = state.get("settings")
    user_prompt = state.get("user_prompt", "")

    # 1) Structured explicit UI setting (e.g. a dropdown) — unchanged,
    # still takes absolute priority when present.
    explicit_plan = plan_from_explicit_settings(
        getattr(settings, "analysis_mode", None) if settings is not None else None,
        getattr(settings, "stratification_column", None) if settings is not None else None,
    )
    if explicit_plan is not None:
        log.info(
            "[Planner] Explicit analysis mode selected — %s (capabilities: %s) — "
            "bypassing keyword/LLM intent detection.",
            explicit_plan["intent_label"],
            ", ".join(explicit_plan["run_capability_nodes"]),
        )
        return {**state, "plan": explicit_plan}

    # 2) An explicit FREE-TEXT request (e.g. "Stratified Analysis by
    # landing_page" typed into the ordinary prompt box) must be detected
    # deterministically and given the same priority as (1) above —
    # checked BEFORE KeywordPlanner/LLMPlanner ever run, so the word
    # "analysis" alone can never cause it to fall through to Full
    # Experiment Review. See
    # plan_from_free_text_stratification_request's docstring.
    dataset_columns = None
    dataset_id = state.get("dataset_id")
    if dataset_id:
        try:
            dataset_columns = list(get_dataset(dataset_id).columns)
        except Exception:  # noqa: BLE001 — a lookup failure here must never block routing;
            # the column-validation step is a best-effort enhancement, not a hard
            # dependency (check_stratification_eligibility validates again downstream).
            dataset_columns = None

    free_text_result = plan_from_free_text_stratification_request(user_prompt, dataset_columns)
    if free_text_result is not None:
        plan, resolved_column = free_text_result
        log.info(
            "[Planner] Explicit free-text stratification request detected in prompt "
            "(resolved column=%r) — bypassing keyword/LLM intent detection.",
            resolved_column,
        )
        updated_settings = _settings_with_stratification(settings, resolved_column)
        return {**state, "plan": plan, "settings": updated_settings}

    planner = get_planner()

    # The request-scoped model override (Settings.model in the UI ->
    # AnalysisSettings.model -> here) tells LLMPlanner which model the
    # user picked — see planner_strategy.py's Planner Protocol docstring.
    requested_model = getattr(settings, "model", None) if settings is not None else None

    plan = planner.plan(user_prompt, state["dataset"], requested_model)

    log.info(
        "[Planner] Intent detected — %s (capabilities: %s, llm_status=%s)",
        plan["intent_label"],
        ", ".join(plan["run_capability_nodes"]),
        plan.get("llm_status", "not_used"),
    )

    return {**state, "plan": plan}
