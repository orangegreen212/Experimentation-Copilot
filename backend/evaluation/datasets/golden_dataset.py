"""
Golden evaluation dataset — deterministic ground truth for the
evaluation framework.

WHY THIS FILE EXISTS
---------------------
The application under test (Phases 1-4: validation, SRM, power
analysis, hypothesis evaluation, decision support, guardrails, RAG,
LangGraph workflow) is a decision-support system, not a chatbot. Its
correctness is defined by deterministic statistical/decision rules
(see `app/graph/report_generator.py::determine_decision` and
`experiment_validity`), so the ground truth for evaluating it must
ALSO be deterministic and human-specified — never generated or
labeled by an LLM (see task spec, "Do NOT define ground truth by
asking an LLM").

Each `GoldenCase` bundles:
  - `facts`: a `ReportFacts` object — the exact, already-computed
    input the production `ReportGenerator` strategies consume. This
    is real production input shape, not a mock of the evaluation
    framework's own invention.
  - `expected`: an `ExpectedFacts` object — every structured fact a
    correct system must produce for this input, specified by hand.

This dataset is versioned independently of the application. The SAME
cases (and the same `evaluate_report` runner) are meant to evaluate
this Phase 3/4 snapshot AND, later, a Phase 5 (segmentation) snapshot
or any future version — see runners/run_evaluation.py's
`evaluate_generator()`, which takes a `ReportGenerator` as a
parameter for exactly this reason. Nothing here imports or assumes
segmentation exists.

CATEGORY COVERAGE (per task spec section 1, A-P):
  A. Valid positive experiment            -> valid_positive_go,
                                              valid_positive_go_with_caution_no_guardrails
  B. Valid negative experiment             -> valid_negative_no_go
  C. Non-significant experiment            -> non_significant_inconclusive
  D. Underpowered experiment               -> underpowered_null_inconclusive
  E. SRM failure                           -> srm_failure_invalid
  F. Critical guardrail failure            -> critical_guardrail_failure_no_go
  G. Multiple guardrail issues             -> multiple_guardrail_issues
  H. Invalid/malformed data                -> invalid_malformed_data
  I. Zero/near-zero conversions            -> near_zero_conversions
  J. Large effect, insufficient sample     -> large_effect_insufficient_sample
  K. Small effect, large sample            -> small_effect_large_sample
  L. Metric directionality edge case       -> directionality_edge_case_decrease_metric
  M. Missing values                        -> missing_values_data_quality
  N. Data quality problems (outliers)      -> data_quality_outliers
  O. Metric metadata edge cases            -> metric_metadata_ambiguous_selection
  P. Conflicting primary vs guardrail      -> conflicting_primary_vs_guardrail

Two additional structural cases are included because they are
safety-critical regression guards specific to this codebase (Twyman's
Law: a good-looking result must never override a validity failure),
matching the pattern already established in
`scripts/evaluate_decisions.py`:
  - srm_failure_with_significant_looking_stats
  - conflicting_variant_duplicates_invalid
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.graph.report_generator import ReportFacts
from app.rag.retriever import DocumentChunk, RetrievedChunk
from app.schemas.dataset import DatasetInfo, DatasetType
from app.schemas.quality import QualityCheck
from app.schemas.report import Decision, ExperimentValidity, GuardrailStatus
from app.schemas.statistics import HypothesisTestType, PowerAnalysisResult, StatResult


# --------------------------------------------------------------------
# Builders (mirrors scripts/evaluate_decisions.py's helpers so both
# evaluation surfaces stay visually/semantically consistent)
# --------------------------------------------------------------------

def _dataset(users: int = 12_400, metric_label: str = "Conversion Rate", available_metrics: list[str] | None = None) -> DatasetInfo:
    return DatasetInfo(
        type=DatasetType.AGGREGATED_AB_TEST,
        variants=2,
        users=users,
        metric_label=metric_label,
        available_metrics=available_metrics or [metric_label],
        metric_selection_reason="Selected by the deterministic outcome-column priority — no competing outcome metrics were available in this dataset.",
    )


def _srm_pass() -> QualityCheck:
    return QualityCheck(label="Sample Ratio Mismatch (SRM)", passed=True, detail="p=0.83")


def _srm_fail() -> QualityCheck:
    return QualityCheck(label="Sample Ratio Mismatch (SRM)", passed=False, detail="p=0.0002", critical=True)


def _stat(
    p_value: float,
    significant: bool,
    delta: str = "+8.4% (rel)",
    delta_relative: float = 8.4,
    metric: str = "Conversion Rate",
    control: str = "4.21%",
    variant: str = "4.56%",
) -> StatResult:
    return StatResult(
        metric=metric, test_type=HypothesisTestType.CHI_SQUARE, test_name="Chi-square test",
        statistic=12.4, selection_reason="binary metric", control=control, variant=variant,
        delta=delta, delta_relative=delta_relative, p_value=p_value, significant=significant,
        ci_lower="-0.10pp", ci_upper="+0.80pp",
    )


def _power(achieved: float, sufficient: bool, mde: float = 1.8) -> PowerAnalysisResult:
    return PowerAnalysisResult(
        minimum_detectable_effect_relative=mde, required_sample_size=8200, observed_sample_size=12_400,
        achieved_power=achieved, alpha=0.05, is_sufficiently_powered=sufficient,
    )


def _kb(source: str = "kohavi.md", heading: str = "Minimum Detectable Effect (MDE) and Power") -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=DocumentChunk(source=source, heading=heading, content="..."), score=0.4)]


# --------------------------------------------------------------------
# Expected facts — hand-specified ground truth
# --------------------------------------------------------------------

@dataclass
class ExpectedFacts:
    """
    Every structured fact a correct `ExperimentReport` must show for a
    `GoldenCase`. Fields left as `None` are not checked (e.g. a case
    that is only about validity has no meaningful expected guardrail
    status). This is intentionally a plain dataclass, not a Pydantic
    model — it never needs to be serialized/validated on its own, only
    compared field-by-field against a real `ExperimentReport`.
    """

    decision: str  # Decision enum value — REQUIRED, every case has one
    validity: str | None = None  # ExperimentValidity enum value
    significant: bool | None = None  # primary metric's StatResult.significant
    power_sufficient: bool | None = None
    guardrail_status: str | None = None  # GuardrailStatus enum value
    effect_direction: str | None = None  # "positive" | "negative" | "none"
    srm_passed: bool | None = None


@dataclass
class GoldenCase:
    id: str
    category: str  # letter A-P, or "structural" for extra safety-regression cases
    description: str
    facts: ReportFacts
    expected: ExpectedFacts
    tags: list[str] = field(default_factory=list)


GOLDEN_CASES: list[GoldenCase] = [
    # A. Valid positive experiment ------------------------------------------------
    GoldenCase(
        id="valid_positive_go",
        category="A",
        description=(
            "Clean data, SRM passes, significant positive effect, adequately powered, AND a "
            "guardrail metric that was actually evaluated and passed -> a clean GO. (Per "
            "determine_decision(), GO requires guardrail_status == PASS specifically -- "
            "GuardrailStatus.NOT_AVAILABLE is deliberately never treated as equivalent to PASS, "
            "so a run with no guardrails evaluated tops out at GO_WITH_CAUTION; see "
            "valid_positive_go_with_caution_no_guardrails below for that case.)"
        ),
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.0003, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users — exceeds requirement", kb_results=_kb(),
            guardrail_results=[
                _stat(p_value=0.61, significant=False, delta="+0.3% (rel)", delta_relative=0.3, metric="Page Load Latency", control="820ms", variant="817ms"),
            ],
        ),
        expected=ExpectedFacts(
            decision="GO", validity="VALID", significant=True, power_sufficient=True,
            guardrail_status="PASS", effect_direction="positive", srm_passed=True,
        ),
    ),
    GoldenCase(
        id="valid_positive_go_with_caution_no_guardrails",
        category="A",
        description=(
            "Same clean positive result as valid_positive_go, but NO guardrail metrics were "
            "evaluated at all -> GO_WITH_CAUTION, not GO. Locks in the "
            "GuardrailStatus.NOT_AVAILABLE != PASS distinction as a first-class expected "
            "behavior, not an accidental omission."
        ),
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.0003, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users — exceeds requirement", kb_results=_kb(),
        ),
        expected=ExpectedFacts(
            decision="GO_WITH_CAUTION", validity="VALID", significant=True, power_sufficient=True,
            guardrail_status="NOT_AVAILABLE", effect_direction="positive", srm_passed=True,
        ),
    ),
    # B. Valid negative experiment -------------------------------------------------
    GoldenCase(
        id="valid_negative_no_go",
        category="B",
        description="Clean data, significant NEGATIVE effect that clears the practical-significance threshold -> NO_GO (never a positive-sounding recommendation).",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.0002, significant=True, delta="-9.1% (rel)", delta_relative=-9.1, control="4.60%", variant="4.18%")],
            test_selections=[], power_analysis=_power(achieved=0.94, sufficient=True, mde=1.8),
            mde_display="1.8%", sample_size_note="12,400 users — exceeds requirement", kb_results=_kb(),
        ),
        expected=ExpectedFacts(
            decision="NO_GO", validity="VALID", significant=True, power_sufficient=True,
            guardrail_status="NOT_AVAILABLE", effect_direction="negative", srm_passed=True,
        ),
    ),
    # C. Non-significant experiment --------------------------------------------------
    GoldenCase(
        id="non_significant_inconclusive",
        category="C",
        description="Adequately powered but p-value above alpha -> INCONCLUSIVE, must not be reported as a confident null.",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.71, significant=False, delta="+0.1% (rel)", delta_relative=0.1)],
            test_selections=[], power_analysis=_power(achieved=0.9, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
        ),
        expected=ExpectedFacts(
            decision="INCONCLUSIVE", validity="VALID", significant=False, power_sufficient=True,
            effect_direction="none", srm_passed=True,
        ),
    ),
    # D. Underpowered experiment -----------------------------------------------------
    GoldenCase(
        id="underpowered_null_inconclusive",
        category="D",
        description="Non-significant AND underpowered -> INCONCLUSIVE; a null result here must not be read as evidence of 'no effect'.",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.42, significant=False, delta="-1.2% (rel)", delta_relative=-1.2)],
            test_selections=[], power_analysis=_power(achieved=0.24, sufficient=False),
            mde_display="2.8%", sample_size_note="12,400 users", kb_results=_kb(),
        ),
        expected=ExpectedFacts(
            decision="INCONCLUSIVE", validity="VALID", significant=False, power_sufficient=False,
            effect_direction="none", srm_passed=True,
        ),
    ),
    # E. SRM failure -------------------------------------------------------------------
    GoldenCase(
        id="srm_failure_invalid",
        category="E",
        description="Randomization broken (SRM failed) -> INVALID regardless of any downstream number.",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_fail()],
            srm_passed=False, stat_results=[], test_selections=[], power_analysis=None,
            mde_display="N/A", sample_size_note="N/A", kb_results=_kb("kohavi.md", "Sample Ratio Mismatch"),
        ),
        expected=ExpectedFacts(decision="INVALID", validity="INVALID", srm_passed=False),
    ),
    # F. Critical guardrail failure -----------------------------------------------------
    GoldenCase(
        id="critical_guardrail_failure_no_go",
        category="F",
        description="Primary metric significant+practical, but a guardrail metric fails -> NO_GO despite the positive primary result.",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.0003, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
            guardrail_results=[
                _stat(p_value=0.001, significant=True, delta="-4.2% (rel)", delta_relative=-4.2, metric="Page Load Latency", control="820ms", variant="856ms"),
            ],
        ),
        expected=ExpectedFacts(
            decision="NO_GO", validity="VALID", significant=True, power_sufficient=True,
            guardrail_status="FAIL", effect_direction="positive", srm_passed=True,
        ),
    ),
    # G. Multiple guardrail issues -----------------------------------------------------
    GoldenCase(
        id="multiple_guardrail_issues",
        category="G",
        description="Two guardrail metrics regress (one significant fail, one non-significant warning) -> overall guardrail status is FAIL (worst-case wins), decision is NO_GO.",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.0003, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
            guardrail_results=[
                _stat(p_value=0.001, significant=True, delta="-4.2% (rel)", delta_relative=-4.2, metric="Page Load Latency", control="820ms", variant="856ms"),
                _stat(p_value=0.09, significant=False, delta="-1.1% (rel)", delta_relative=-1.1, metric="Refund Rate", control="2.10%", variant="2.12%"),
            ],
        ),
        expected=ExpectedFacts(
            decision="NO_GO", validity="VALID", significant=True, power_sufficient=True,
            guardrail_status="FAIL", effect_direction="positive", srm_passed=True,
        ),
    ),
    # H. Invalid/malformed data ---------------------------------------------------------
    GoldenCase(
        id="invalid_malformed_data",
        category="H",
        description="Critical data-quality failure (not SRM) -> INVALID, same severity class as SRM.",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[
                _srm_pass(),
                QualityCheck(label="Required Columns Present", passed=False, detail="Missing 'variant' column in 40% of rows", critical=True),
            ],
            srm_passed=True, stat_results=[], test_selections=[], power_analysis=None,
            mde_display="N/A", sample_size_note="N/A",
        ),
        expected=ExpectedFacts(decision="INVALID", validity="INVALID", srm_passed=True),
    ),
    # I. Zero/near-zero conversions --------------------------------------------------
    GoldenCase(
        id="near_zero_conversions",
        category="I",
        description="Near-zero conversion rates: test still runs, but result is non-significant given the tiny observed counts -> INCONCLUSIVE, not a confident call either way.",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(users=20_000), quality_checks=[_srm_pass()],
            srm_passed=True,
            stat_results=[_stat(p_value=0.88, significant=False, delta="+0.001% (rel)", delta_relative=0.001, control="0.010%", variant="0.010%")],
            test_selections=[], power_analysis=_power(achieved=0.05, sufficient=False, mde=180.0),
            mde_display="180%", sample_size_note="20,000 users — effectively zero base rate", kb_results=_kb(),
        ),
        expected=ExpectedFacts(
            decision="INCONCLUSIVE", validity="VALID", significant=False, power_sufficient=False,
            effect_direction="none", srm_passed=True,
        ),
    ),
    # J. Large effect, insufficient sample --------------------------------------------
    GoldenCase(
        id="large_effect_insufficient_sample",
        category="J",
        description="Eye-catching effect size but the sample is too small for it to reach significance -> INCONCLUSIVE, not GO — a large point estimate on a tiny sample is not evidence.",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(users=300), quality_checks=[_srm_pass()],
            srm_passed=True,
            stat_results=[_stat(p_value=0.31, significant=False, delta="+35.0% (rel)", delta_relative=35.0, control="4.00%", variant="5.40%")],
            test_selections=[], power_analysis=_power(achieved=0.18, sufficient=False, mde=40.0),
            mde_display="40%", sample_size_note="300 users -- far below requirement", kb_results=_kb(),
        ),
        expected=ExpectedFacts(
            decision="INCONCLUSIVE", validity="VALID", significant=False, power_sufficient=False,
            effect_direction="none", srm_passed=True,
        ),
    ),
    # K. Small effect, large sample -----------------------------------------------------
    GoldenCase(
        id="small_effect_large_sample",
        category="K",
        description="Large sample makes a tiny, practically meaningless effect statistically significant -> GO_WITH_CAUTION (statistically real, but below the practical-significance/MDE threshold is NO_GO; here it's borderline/undetermined practical significance).",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(users=500_000), quality_checks=[_srm_pass()],
            srm_passed=True,
            stat_results=[_stat(p_value=0.001, significant=True, delta="+0.3% (rel)", delta_relative=0.3, control="4.000%", variant="4.012%")],
            test_selections=[], power_analysis=_power(achieved=0.99, sufficient=True, mde=0.5),
            mde_display="0.5%", sample_size_note="500,000 users", kb_results=_kb(),
        ),
        expected=ExpectedFacts(
            decision="NO_GO", validity="VALID", significant=True, power_sufficient=True,
            effect_direction="positive", srm_passed=True,
        ),
    ),
    # L. Metric directionality edge case (a "lower is better" metric with a decrease) --
    GoldenCase(
        id="directionality_edge_case_decrease_metric",
        category="L",
        description="Guardrail metric where a NEGATIVE relative_change string (a decrease) is actually the desired direction is out of scope for this deterministic layer today (guardrail evaluation in this codebase treats any significant decrease as a violation) -- this case documents and locks that known, conservative behavior rather than silently assuming metric-specific directionality.",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.0003, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
            guardrail_results=[
                _stat(p_value=0.0005, significant=True, delta="-12.0% (rel)", delta_relative=-12.0, metric="Support Ticket Rate", control="3.0%", variant="2.6%"),
            ],
        ),
        expected=ExpectedFacts(
            decision="NO_GO", validity="VALID", significant=True, power_sufficient=True,
            guardrail_status="FAIL", effect_direction="positive", srm_passed=True,
        ),
        tags=["known-limitation"],
    ),
    # M. Missing values -----------------------------------------------------------------
    GoldenCase(
        id="missing_values_data_quality",
        category="M",
        description="Non-critical missing-values warning (not critical) -> CAUTION, not INVALID; decision can still proceed but must reflect reduced trust (MEDIUM confidence path).",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[
                _srm_pass(),
                QualityCheck(label="Missing Values", passed=False, detail="3.2% of rows have a null metric value", critical=False),
            ],
            srm_passed=True, stat_results=[_stat(p_value=0.0003, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
        ),
        expected=ExpectedFacts(
            decision="GO_WITH_CAUTION", validity="CAUTION", significant=True, power_sufficient=True,
            effect_direction="positive", srm_passed=True,
        ),
    ),
    # N. Data quality problems (outliers, informational) -------------------------------
    GoldenCase(
        id="data_quality_outliers",
        category="N",
        description="An informational-only quality check failing (e.g. large-sample normality rejection) must NOT downgrade validity at all -> VALID (still GO_WITH_CAUTION here since no guardrails were evaluated, matching valid_positive_go_with_caution_no_guardrails — the point of this case is validity=VALID despite the failed informational check, not guardrail behavior).",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[
                _srm_pass(),
                QualityCheck(label="Normality (Shapiro-Wilk)", passed=False, detail="p=1e-28 at n>=30/arm (expected at this sample size)", critical=False, informational=True),
            ],
            srm_passed=True, stat_results=[_stat(p_value=0.0003, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
        ),
        expected=ExpectedFacts(
            decision="GO_WITH_CAUTION", validity="VALID", significant=True, power_sufficient=True,
            guardrail_status="NOT_AVAILABLE", effect_direction="positive", srm_passed=True,
        ),
    ),
    # O. Metric metadata edge cases (ambiguous multi-metric selection) -----------------
    GoldenCase(
        id="metric_metadata_ambiguous_selection",
        category="O",
        description="Multiple numeric metrics were available and none was explicitly requested -- decision must still be reachable. Policy update: ambiguous-metric selection alone no longer caps the decision at GO_WITH_CAUTION (the metric was still selected by a deterministic, documented priority rule, and remains statistically significant on its own terms) -- with a passing guardrail this now reaches a clean GO at HIGH confidence, with the ambiguity surfaced only as an informational note in decision_reason (checked by the explanation evaluator, not this deterministic layer).",
        facts=ReportFacts(
            user_prompt="How did the experiment do?",
            dataset=_dataset(available_metrics=["Conversion Rate", "Average Order Value", "Session Duration"]),
            quality_checks=[_srm_pass()], srm_passed=True,
            stat_results=[_stat(p_value=0.0003, significant=True)],
            test_selections=[], power_analysis=_power(achieved=0.91, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
            guardrail_results=[
                _stat(p_value=0.61, significant=False, delta="+0.3% (rel)", delta_relative=0.3, metric="Page Load Latency", control="820ms", variant="817ms"),
            ],
        ),
        expected=ExpectedFacts(
            decision="GO", validity="VALID", significant=True, power_sufficient=True,
            guardrail_status="PASS", effect_direction="positive", srm_passed=True,
        ),
    ),
    # P. Conflicting evidence between primary metric and guardrails --------------------
    GoldenCase(
        id="conflicting_primary_vs_guardrail",
        category="P",
        description="Primary metric strongly positive AND practically significant, guardrail shows only a non-significant warning-level regression -> GO_WITH_CAUTION, not a clean GO and not NO_GO (guardrail didn't fail, but isn't clean either).",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_pass()],
            srm_passed=True, stat_results=[_stat(p_value=0.0001, significant=True, delta="+9.4% (rel)", delta_relative=9.4)],
            test_selections=[], power_analysis=_power(achieved=0.95, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb(),
            guardrail_results=[
                _stat(p_value=0.22, significant=False, delta="-0.8% (rel)", delta_relative=-0.8, metric="Page Load Latency", control="820ms", variant="827ms"),
            ],
        ),
        expected=ExpectedFacts(
            decision="GO_WITH_CAUTION", validity="VALID", significant=True, power_sufficient=True,
            guardrail_status="WARNING", effect_direction="positive", srm_passed=True,
        ),
    ),
    # ---- structural safety regressions (see module docstring) ------------------------
    GoldenCase(
        id="srm_failure_with_significant_looking_stats",
        category="structural",
        description="Twyman's Law regression guard: even an eye-catching, well-powered, highly-significant-looking result must not override an SRM failure.",
        facts=ReportFacts(
            user_prompt="Should we ship variant B?", dataset=_dataset(), quality_checks=[_srm_fail()],
            srm_passed=False,
            stat_results=[_stat(p_value=0.0001, significant=True, delta="+9.1% (rel)", delta_relative=9.1)],
            test_selections=[], power_analysis=_power(achieved=0.95, sufficient=True),
            mde_display="1.8%", sample_size_note="12,400 users", kb_results=_kb("kohavi.md", "Sample Ratio Mismatch"),
        ),
        expected=ExpectedFacts(decision="INVALID", validity="INVALID", srm_passed=False),
        tags=["safety-critical"],
    ),
    GoldenCase(
        id="conflicting_variant_duplicates_invalid",
        category="structural",
        description="Users assigned to both variants (broken assignment pipeline) -> INVALID, same severity as SRM failure.",
        facts=ReportFacts(
            user_prompt="ship it?", dataset=_dataset(), quality_checks=[_srm_pass()], srm_passed=True,
            stat_results=[], test_selections=[], power_analysis=None, mde_display="N/A",
            sample_size_note="N/A", has_conflicting_variant_duplicates=True,
        ),
        expected=ExpectedFacts(decision="INVALID", validity="INVALID", srm_passed=True),
        tags=["safety-critical"],
    ),
]


def get_case(case_id: str) -> GoldenCase:
    for c in GOLDEN_CASES:
        if c.id == case_id:
            return c
    raise KeyError(f"No golden case with id={case_id!r}")


def cases_by_category(category: str) -> list[GoldenCase]:
    return [c for c in GOLDEN_CASES if c.category == category]
