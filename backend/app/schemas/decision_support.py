"""
Decision Support schemas — Phase 3.

Answers "given the experiment result and the hypothesis, what does
this mean for the business?" This is explicitly NOT a second
statistical analysis: every numeric field here is either copied
verbatim from an existing deterministic fact (`StatResult`,
`HypothesisEvaluation`, `DatasetInfo`) or derived from those facts
via plain arithmetic performed in `app/stats/decision_support.py`.
No LLM is involved anywhere in producing these values — see that
module's docstring for the full rule set.

`DecisionSupport` is optional on `ExperimentReport` and is `None`
whenever there is no hypothesis (Phase 3 spec §7) — it never affects
experiments run without one, and the existing report shape/behavior
for those runs is unchanged.
"""

from __future__ import annotations

from app.schemas.base import CamelModel
from app.schemas.hypothesis_evaluation import HypothesisVerdict


class AdditionalMetricComparison(CamelModel):
    """
    One non-primary metric analyzed alongside the primary metric
    (Phase 3 spec §4). Copied directly from the matching `StatResult`
    row — no new statistical test, no recomputation.
    """

    metric: str
    baseline_value: float | None
    observed_value: float | None
    absolute_change: float | None
    relative_change: float | None
    statistically_significant: bool
    direction: str  # "increase" | "decrease" | "no_change"


class GuardrailFinding(CamelModel):
    """
    Deterministic status of one guardrail metric (Phase 3 spec §5).
    Built only from the existing `StatResult` rows in
    `ReportFacts.guardrail_results` — never a new test, never a
    reinterpretation of `GuardrailStatus`/`Decision`, both of which
    remain computed exclusively by `determine_decision()`.
    """

    metric: str
    observed_value: float | None
    relative_change: float | None
    statistically_significant: bool
    violated: bool


class DecisionSupport(CamelModel):
    """
    Structured, deterministic decision-support facts for one
    experiment report. See module docstring for the "no second
    statistical analysis" boundary this enforces.

    `available=False` means impact/expected-vs-observed numbers could
    not be reliably computed (e.g. missing baseline) — `warnings`
    explains why. This is never silently left blank; per Phase 3 spec
    §3/§10, "not enough information" is always preferred over a
    fabricated or misleading number.
    """

    available: bool

    # --- expected vs observed (spec §2) ---------------------------------
    primary_metric: str | None = None
    baseline_value: float | None = None
    observed_value: float | None = None
    observed_effect_absolute: float | None = None
    observed_effect_relative: float | None = None
    expected_effect_relative: float | None = None
    expected_value: float | None = None
    effect_achievement_ratio: float | None = None
    statistical_significance: bool | None = None
    hypothesis_verdict: HypothesisVerdict | None = None
    business_interpretation: str | None = None

    # --- absolute business impact (spec §3) -----------------------------
    # "unavailable" (impact_calculation_method == "unavailable") unless
    # the conservative conditions in stats/decision_support.py are all
    # satisfied — see that module for the exact rule. Never inferred
    # from a formatted string, never inferred for revenue/AOV/other
    # continuous metrics.
    impact_calculation_method: str  # "population_scaled" | "unavailable"
    baseline_expected_count: float | None = None
    observed_count: float | None = None
    incremental_count: float | None = None

    # --- additional metrics / trade-offs (spec §4) ----------------------
    additional_metrics: list[AdditionalMetricComparison] = []

    # --- guardrails (spec §5) --------------------------------------------
    guardrail_findings: list[GuardrailFinding] = []
    guardrail_violated: bool = False

    warnings: list[str] = []
