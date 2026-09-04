"""
Statistical result schemas.

`StatResult` mirrors lib/types.ts (extended per project decision to
stop hardcoding a test name in the frontend — see StatResult below for
the added fields).

NOTE on types: the frontend deliberately types control/variant/delta/
ciLower/ciUpper as *strings* (pre-formatted for display, e.g. "4.21%",
"+8.4% (rel)", "+0.18pp"). Formatting decisions (percent vs currency
vs pp, rounding) are business logic that belongs in `stats/`, not in
the schema. The schema just enforces "this must already be a display
string by the time it reaches the API boundary."

Everything else in this file (TestSelectionResult, PowerAnalysisResult)
is INTERNAL — used by graph nodes and the decision node's prompt
context, never sent to the frontend directly. They carry raw floats
because internal consumers (the decision LLM prompt, tests) need exact
numbers, not formatted strings.
"""

from enum import Enum

from pydantic import Field

from app.schemas.base import CamelModel


class HypothesisTestType(str, Enum):
    """
    Which test the deterministic selector (stats/hypothesis_tests.py
    select_test()) chose. The LLM never decides this — see that
    module's docstring for the full decision tree.
    """

    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    CHI_SQUARE = "chi_square"
    FISHERS_EXACT = "fishers_exact"
    ONE_WAY_ANOVA = "one_way_anova"
    KRUSKAL_WALLIS = "kruskal_wallis"


# Human-readable display names, single source of truth so the frontend
# never re-derives a label from the enum value itself.
TEST_TYPE_DISPLAY_NAMES: dict[HypothesisTestType, str] = {
    HypothesisTestType.WELCH_T_TEST: "Welch's t-test",
    HypothesisTestType.MANN_WHITNEY_U: "Mann-Whitney U test",
    HypothesisTestType.CHI_SQUARE: "Chi-square test",
    HypothesisTestType.FISHERS_EXACT: "Fisher's exact test",
    HypothesisTestType.ONE_WAY_ANOVA: "One-way ANOVA",
    HypothesisTestType.KRUSKAL_WALLIS: "Kruskal-Wallis test",
}


class StatResult(CamelModel):
    """
    One row in the frontend's stats table (e.g. Conversion Rate, AOV).

    DECISION: the frontend must never hardcode or guess which
    hypothesis test produced a row — that caused a real trust bug (a
    Chi-square result displayed under a "Welch's t-test" caption). The
    backend now tells the frontend which test ran (`testType`,
    `testName`), the raw test statistic (`statistic`), and WHY that
    test was selected (`selectionReason` — copied straight from
    `TestSelectionResult.reason`, so there are not two separate copies
    of this explanation to keep in sync).
    """

    metric: str
    test_type: HypothesisTestType
    test_name: str
    statistic: float
    selection_reason: str
    control: str
    variant: str
    delta: str
    # Phase 2 — the SAME relative effect already computed to build the
    # `delta` display string above, exposed as a raw fraction (e.g.
    # 0.084 for "+8.4% (rel)") for deterministic downstream consumers
    # (see app/stats/hypothesis_evaluator.py) that need the number, not
    # the formatted text. NEVER parse `delta` to get this — it is set
    # from the exact same local variable used to format `delta`, in
    # the same function, at the same time (see hypothesis_tests.py).
    # None when the control-arm baseline is zero (relative effect is
    # mathematically undefined there) or for omnibus/multi-arm rows,
    # which have no single well-defined pairwise relative effect.
    observed_relative_effect: float | None = None
    p_value: float
    significant: bool
    ci_lower: str
    ci_upper: str
    # Multi-arm metadata. Optional so existing two-arm API consumers remain unchanged.
    comparison: str | None = None
    is_omnibus: bool = False
    adjusted_p_value: float | None = None
    multiple_testing_method: str | None = None
    reference_arm: str | None = None
    arm: str | None = None
    practical_significant: bool | None = None
    is_winner: bool = False
    # Guardrail directionality — "does higher = better for this metric?"
    # Defaults True (matches the implicit assumption every non-guardrail
    # row already made). Only ever set to False by guardrail_node.py, for
    # rows it built from a lower-is-better metric (e.g. Bounce Rate,
    # churn, latency — see app.stats.dataset_classifier.
    # infer_guardrail_direction). The primary metric's own StatResult
    # rows are unaffected — this field is not read anywhere in the
    # primary-metric decision path, only in the guardrail evaluation
    # branch of determine_decision().
    higher_is_better: bool = True


class MetricType(str, Enum):
    """
    Classifies a metric for the purpose of choosing an outlier-detection
    strategy (Stage 3 decision — see check_outliers in quality_checks.py):

      - BINARY: 0/1 conversion-style metrics. Outlier detection is not
        meaningful here and is skipped entirely.
      - CONTINUOUS_MONETARY: revenue/order-value-style metrics where a
        zero represents "no conversion happened" rather than a true
        observation of the revenue distribution. Outlier detection
        runs on positive values only, using IQR (robust to skew).
      - CONTINUOUS_GENERAL: other continuous metrics with no
        structural zero (e.g. session duration, page views). Outlier
        detection runs on all values using IQR.
    """

    BINARY = "binary"
    CONTINUOUS_MONETARY = "continuous_monetary"
    CONTINUOUS_GENERAL = "continuous_general"


class NormalityCheckResult(CamelModel):
    """Shapiro-Wilk result per arm — internal, feeds test selection + QualityCheck."""

    control_statistic: float
    control_p_value: float
    control_normal: bool
    variant_statistic: float
    variant_p_value: float
    variant_normal: bool


class TestSelectionResult(CamelModel):
    """
    Record of which hypothesis test was chosen and why — produced by
    the single deterministic selector `select_test()`.

    `normality` is None when the selector skipped the normality check
    (either because the metric is binary, or because both arms had
    n >= 30 and the large-sample rule applied instead).
    """

    test_type: HypothesisTestType
    reason: str
    normality: NormalityCheckResult | None = None
    large_sample_rule_applied: bool = False


class PowerAnalysisResult(CamelModel):
    """
    Internal power/MDE numbers.

    The frontend only ever sees the two pre-formatted strings on
    ExperimentReport (`mde`, `sampleSizeNote`) — this schema is what
    stats/power_analysis.py returns internally before that formatting
    happens, and what the decision LLM prompt is grounded in.
    """

    # The MDE at the observed sample size — independent of the effect
    # actually observed (solved purely from n/alpha/target_power; see
    # `_solve_mde`). "What could we have detected."
    minimum_detectable_effect_relative: float
    # The sample size (per arm) that WOULD be required to reach
    # target_power for the effect size ACTUALLY OBSERVED in this
    # experiment (see `_solve_required_n`, called with
    # `observed_effect_size` — never the MDE). "What would it take to
    # confirm what we saw," not "what would it take to detect the
    # MDE." Never label this as being "for the MDE" in display text —
    # see `format_sample_size_note`'s docstring.
    required_sample_size: int
    observed_sample_size: int
    achieved_power: float = Field(ge=0.0, le=1.0)
    alpha: float = 0.05
    is_sufficiently_powered: bool


class VarianceReductionResult(CamelModel):
    """
    Internal result of CUPED and/or bootstrap adjustment, when enabled
    via Settings.cuped / Settings.bootstrap.

    Applied BEFORE hypothesis testing when enabled; this schema records
    what changed so the decision node can explain the effect (e.g. "CI
    tightened by ~35%") without doing any math itself.
    """

    method: str  # "cuped" | "bootstrap"
    variance_before: float
    variance_after: float
    variance_reduction_pct: float

