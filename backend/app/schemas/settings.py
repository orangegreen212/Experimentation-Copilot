"""
Analysis settings schema.

Mirrors lib/types.ts:

    export interface Settings {
      cuped: boolean;
      bootstrap: boolean;
      model: string;
      costUsd: number;
    }

`cuped` / `bootstrap` are read by the graph to decide whether to run
`stats/variance_reduction.py` before hypothesis testing. `model` selects
the OpenRouter model (Stage 8) for the planner/decision LLM calls.
`costUsd` is accumulated client-side today (per workspace-view.tsx /
page.tsx); the backend will echo per-run cost in the analyze response
so the frontend doesn't have to hardcode it, but ownership of the
running total stays on the frontend per the current app design.
"""

from pydantic import Field

from app.schemas.base import CamelModel


class AnalysisSettings(CamelModel):
    cuped: bool = False
    bootstrap: bool = False
    model: str = "claude-sonnet"
    cost_usd: float = 0.0

    # Guardrail metrics EXPLICITLY selected by the user for this
    # analysis (e.g. ["Revenue", "Bounce Rate"]) — structured data,
    # never parsed out of free-text `prompt`. Optional and purely
    # additive: omitted/empty (the default) means no guardrails were
    # requested and is 100% backward compatible with every existing
    # request. Matched against the dataset's actual
    # `DatasetInfo.available_metrics` deterministically (exact match
    # only — see app.stats.dataset_classifier.resolve_guardrail_metrics)
    # by the guardrail analysis node, never by the LLM. Distinct from
    # `DatasetInfo.guardrail_candidates`, which is only an automatically
    # detected SUGGESTION and is never treated as a request on its own.
    guardrail_metrics: list[str] = Field(default_factory=list)

    # Explicit analysis-mode selection from the UI (e.g. the user picked
    # "Stratified Analysis by <column>" from a dropdown). None means no
    # explicit mode was selected — every existing flow is unaffected.
    # When set, this ALWAYS takes priority over free-form intent
    # detection (keyword or LLM planner) for what to display/route as —
    # see app/graph/nodes/planner_node.py. The only currently-supported
    # explicit value is "stratified"; anything else is treated the same
    # as None (ignored) rather than raising, so an unrecognized/future
    # value from a newer frontend build degrades gracefully instead of
    # 500ing the request.
    analysis_mode: str | None = None
    # The baseline categorical column to stratify by, required when
    # analysis_mode == "stratified". Ignored otherwise.
    stratification_column: str | None = None
