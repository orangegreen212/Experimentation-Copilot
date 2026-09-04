"""
Experiment report schemas.

Mirrors lib/types.ts:

    export type ConfidenceLevel = 'HIGH' | 'MEDIUM' | 'LOW';

    export interface ExperimentReport {
      confidence: ConfidenceLevel;
      confidenceReason: string;
      confidenceStars: number;
      srmWarning: boolean;
      executiveSummary: string;
      qualityChecks: QualityCheck[];
      stats: StatResult[];
      mde: string;
      sampleSizeNote: string;
      recommendations: string[];
      nextSteps: string[];
      experimentValidity: ExperimentValidity;
      guardrailStatus: GuardrailStatus;
      practicalSignificance: boolean | null;
      decision: Decision;
      decisionReason: string;
      recommendationConfidence: ConfidenceLevel;
    }

This is the FINAL output of the LangGraph pipeline (decision_node,
Stage 7). `executiveSummary`, `recommendations`, `nextSteps`, and
`confidenceReason` are LLM-generated text — but only ever generated
FROM the already-computed fields below them (qualityChecks, stats,
mde, sampleSizeNote), never computed by the LLM itself.

LEGACY FIELD NOTE — `confidence` / `confidenceStars` / `confidenceReason`:
Despite the name, `_assess_confidence()` (report_generator.py) has never
computed statistical confidence — it's a data/experiment RELIABILITY
assessment (SRM, quality checks, duplicate users, power/MDE
definedness). Kept for backward compatibility (persisted history rows,
existing frontend rendering) but no NEW decision logic is built on top
of it — see `determine_decision()`, which is the single source of
truth for `decision` and never reads `confidence`. If you're adding
new logic and are tempted to branch on `confidence == HIGH`, use
`experiment_validity` and/or `recommendation_confidence` instead.

`decision` is the canonical, deterministic ship/no-ship-equivalent
signal (`determine_decision()` in report_generator.py) — the LLM may
explain it in `executive_summary`/`recommendations` text, but never
computes or overrides it (enforced by the server-side gate in
`LLMReportGenerator`, which compares any LLM-proposed framing against
this field rather than string-matching "SHIP").
"""

from enum import Enum

from app.schemas.quality import QualityCheck
from app.schemas.statistics import StatResult
from app.schemas.hypothesis import Hypothesis
from app.schemas.hypothesis_evaluation import HypothesisEvaluation
from app.schemas.decision_support import DecisionSupport
from app.schemas.segmentation import SegmentationResult
from app.schemas.stratification import StratificationResult
from app.schemas.base import CamelModel
from app.schemas.execution import LLMUsage, RunMetadata
from app.schemas.decision_narrative import DecisionNarrative
from app.schemas.guardrails import GuardrailRequestState, GuardrailResolution


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ExperimentValidity(str, Enum):
    """
    "Can we trust this experiment at all?" — SRM, critical quality
    checks, conflicting variant duplicates. Computed the same way
    `confidence` used to be computed; this is the field new logic
    should read instead of `confidence`.
    """

    VALID = "VALID"
    CAUTION = "CAUTION"
    INVALID = "INVALID"


class GuardrailStatus(str, Enum):
    """
    The EVALUATION outcome of whichever guardrail metric(s) actually
    got resolved and tested. NOT_AVAILABLE means "no guardrail metric
    was evaluated" — it is NEVER equivalent to PASS, and — as of the
    guardrail root-cause fix — it no longer distinguishes WHY nothing
    was evaluated (never requested vs. requested-but-not-found vs.
    resolved-but-not-yet-evaluated). That distinction now lives in the
    separate, independent `GuardrailRequestState`
    (app/schemas/guardrails.py) — always read both together: e.g.
    `guardrail_request_state=PARTIALLY_AVAILABLE` +
    `guardrail_status=FAIL` is a normal, coherent combination meaning
    "some requested guardrails weren't found, but the ones that were
    found include a failure."
    """

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class Decision(str, Enum):
    """
    The single canonical ship/no-ship-equivalent signal. Always
    produced by `determine_decision()` — a pure function of
    ExperimentValidity, significance, practical_significance, and
    GuardrailStatus. Never read from or written by the LLM.
    """

    INVALID = "INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"
    NO_GO = "NO_GO"
    GO_WITH_CAUTION = "GO_WITH_CAUTION"
    GO = "GO"


# Imported here (after `Decision` is defined above, not at module top)
# so `app/schemas/decision_audit.py` can import `Decision` back from
# this module without a circular-import failure — see that module's
# docstring. `DecisionAuditTrail` is purely additive (Phase 7); it
# does not change anything defined above this line.
from app.schemas.decision_audit import DecisionAuditTrail  # noqa: E402


class KnowledgeBaseReference(CamelModel):
    """
    One knowledge-base excerpt shown in Evidence & Sources (Stage 9 —
    Agentic RAG). Only populated when the Planner decided a
    conceptual/explanatory answer was needed and routed to the
    `knowledge_base` node — a normal "evaluate my experiment" run has
    an empty list here, and the frontend should not render this
    section when the list is empty.

    `is_system_fallback` distinguishes the two kinds of entry that can
    appear here:
      - False (default): an ACTUAL retrieved chunk from the knowledge
        base, with a real cosine `relevance_score` against the query.
      - True: a built-in, hand-written methodology note supplied by
        this application itself (never fetched from the KB, never
        attributed to `source`/`heading` metadata that implies
        retrieval) — used ONLY as an approved fallback when real
        retrieval found nothing relevant for a topic the report
        structurally needs to explain (e.g. what SRM is, on an
        SRM-failed report). `relevance_score` is meaningless for a
        fallback entry (always 0.0) since no similarity search
        produced it. The frontend MUST render these visibly
        differently from retrieved evidence — never merge them
        silently into "sources found".
    """

    source: str  # e.g. "kohavi.md", or "system-methodology" for a fallback entry
    heading: str  # e.g. "Sample Ratio Mismatch (SRM)"
    excerpt: str
    relevance_score: float
    is_system_fallback: bool = False


class ExperimentReport(CamelModel):
    confidence: ConfidenceLevel  # LEGACY — data/experiment reliability; see module docstring. Do not build new logic on this.
    confidence_reason: str
    confidence_stars: int
    srm_warning: bool
    executive_summary: str
    quality_checks: list[QualityCheck]
    stats: list[StatResult]
    mde: str
    sample_size_note: str
    recommendations: list[str]
    next_steps: list[str]
    knowledge_base_references: list[KnowledgeBaseReference] = []

    # RELEVANCE-THRESHOLD TASK: distinguishes "knowledge_base_node ran
    # but nothing cleared stats_thresholds.kb_relevance_threshold"
    # (True, references == []) from "the knowledge base was never
    # queried for this request at all" (False, references == []) —
    # both look identical on `knowledge_base_references` alone. The
    # frontend's Evidence & Sources card uses this to show an honest
    # "No sufficiently relevant evidence found" message only in the
    # former case, instead of silently showing nothing either way.
    knowledge_base_attempted: bool = False

    # Distinguishes "retrieval was attempted and genuinely FAILED" from
    # "retrieval ran and legitimately found nothing above threshold" —
    # this distinction lives in the report the user actually reads, not
    # only the execution trace. Set from GraphState.kb_error via
    # ReportFacts.kb_error. None means no retrieval failure occurred
    # (retrieval either wasn't attempted, or ran without error — check
    # `knowledge_base_attempted` / `knowledge_base_references` to tell
    # those two apart). Never fabricated: only ever the short
    # exception type/message knowledge_base_node.py itself caught.
    knowledge_base_retrieval_error: str | None = None

    # The human-readable name of the
    # SPECIFIC validity-blocking failure (e.g. "Outlier Detection",
    # "Sample Ratio Mismatch (SRM)") that Evidence & Sources was
    # filtered against, set whenever `experiment_validity == INVALID`
    # for one of the three recognized reasons — see
    # `app/rag/blocking_topics.py` and `report_generator.
    # relevant_kb_results_for_decision`. None whenever the experiment
    # is not INVALID, or INVALID for a reason this module doesn't
    # recognize. Used by the frontend to say WHAT evidence was missing
    # ("No sufficiently relevant evidence found for: Outlier
    # Detection.") instead of only a generic empty-state message; never
    # used to fabricate or imply a citation that wasn't actually found.
    knowledge_base_blocking_issue: str | None = None

    # Optional deterministic bootstrap cross-check facts. These are populated by
    # Python when Bootstrap is enabled; the LLM never computes or supplies them.
    bootstrap_ci_lower: float | None = None
    bootstrap_ci_upper: float | None = None
    bootstrap_iterations: int | None = None

    # --- canonical decision model (new) -------------------------------
    experiment_validity: ExperimentValidity
    guardrail_status: GuardrailStatus
    practical_significance: bool | None  # True / False / None(=unknown) — never silently defaulted to True
    decision: Decision
    decision_reason: str
    recommendation_confidence: ConfidenceLevel

    # --- Guardrail REQUEST state (guardrail root-cause fix) --------------
    # Independent of `guardrail_status` above — see GuardrailStatus's
    # updated docstring for how the two combine. Set directly by
    # decision_node.py from state["guardrail_request_state"] /
    # state["requested_guardrails"] / state["guardrail_resolutions"]
    # (guardrail_node.py) — never computed or altered by a
    # ReportGenerator/LLM. `guardrail_request_state` defaults to
    # NOT_SPECIFIED and `requested_guardrails`/`guardrail_resolutions`
    # default to `[]` so every report shape from before this field
    # existed still validates unchanged and reads as "no guardrails
    # requested" (never as "requested but unavailable").
    guardrail_request_state: GuardrailRequestState = GuardrailRequestState.NOT_SPECIFIED
    requested_guardrails: list[str] = []
    guardrail_resolutions: list[GuardrailResolution] = []

    # --- Phase 2: hypothesis + its deterministic evaluation ------------
    # Both are set directly by decision_node.py (never by a
    # ReportGenerator, template or LLM-backed) from the exact
    # `ReportFacts.hypothesis` / `ReportFacts.hypothesis_evaluation`
    # objects — see decision_node.py's docstring. `hypothesis` is None
    # when the analyst didn't provide one (Phase 1 backward
    # compatibility, unchanged). `hypothesis_evaluation` is None both
    # when there's no hypothesis AND when evaluation was unavailable
    # for a present hypothesis (see HypothesisEvaluation's docstring
    # for the distinction — check `hypothesis_evaluation.metric_matched`
    # /`evaluation_note` for the latter case's explanation).
    hypothesis: Hypothesis | None = None
    hypothesis_evaluation: HypothesisEvaluation | None = None

    # --- Phase 3: deterministic decision support ------------------------
    # Set directly by decision_node.py from build_decision_support() —
    # never by a ReportGenerator/LLM. None whenever `hypothesis` is
    # None (Phase 3 spec §7); reports without a hypothesis are
    # otherwise completely unchanged from Phase 1/2 behavior.
    decision_support: DecisionSupport | None = None

    # --- Phase 5: deterministic segmentation (supporting evidence only) --
    # Set directly by decision_node.py from ReportFacts.segmentation_result
    # — never computed or altered by a ReportGenerator/LLM. None when
    # experiment_node never ran (e.g. SRM-failed / validation-only runs).
    segmentation: SegmentationResult | None = None

    # --- TRUE stratified analysis (separate section from Segmentation) --
    # Set directly by decision_node.py from state["stratification_result"]
    # — never computed or altered by a ReportGenerator/LLM. None
    # whenever stratified analysis wasn't requested (AnalysisSettings.
    # analysis_mode != "stratified") or experiment_node never ran (e.g.
    # SRM-failed / validation-only runs). See app/schemas/stratification.py
    # for why this is deliberately its own section, not folded into
    # `segmentation` above.
    stratification: StratificationResult | None = None

    # --- Phase 7: deterministic Decision Audit Trail ---------------------
    # Set directly by decision_node.py from build_decision_audit_trail()
    # (app/stats/decision_audit.py) — never computed or altered by a
    # ReportGenerator/LLM. Explanatory only: `decision_audit.decision`
    # always mirrors `decision` above; this field never introduces a
    # second, independent decision. None only for report shapes that
    # bypass the canonical decision flow entirely (e.g. the pure
    # funnel-only / conceptual-question reports, which have no
    # `decision` to audit).
    decision_audit: DecisionAuditTrail | None = None

    # --- Product improvement: deterministic Decision Narrative ---------
    # Set directly by decision_node.py from build_decision_narrative()
    # (app/stats/decision_narrative.py) — never computed or altered by
    # a ReportGenerator/LLM. Purely explanatory: adapts to `decision`
    # but never changes it. None only for report paths that never
    # reached determine_decision() at all (e.g. funnel-only / purely
    # conceptual reports).
    decision_narrative: DecisionNarrative | None = None

    # --- Phase 8: production readiness / observability ------------------
    # Set directly by decision_node.py — never by a ReportGenerator/LLM.
    # `run_metadata` is the structured run summary (see RunMetadata's
    # docstring); it is always populated once decision_node has run.
    run_metadata: RunMetadata | None = None
    # `report_fallback_reason` is populated ONLY when LLMReportGenerator
    # actually attempted an LLM call for this run and it failed, causing
    # a silent-until-now fallback to TemplateReportGenerator (see
    # report_generator.py). None on every other path, including the
    # normal case where REPORT_BACKEND=template was configured and no
    # LLM was ever attempted — that is not a fallback, it's the
    # configured behavior, so it must not look like a degraded run.
    report_fallback_reason: str | None = None
    # Set ONLY by LLMReportGenerator, ONLY when it actually attempted
    # (and got a response from) an LLM call for this run — see
    # LLMUsage's docstring. None on every path that never called an
    # LLM (template/keyword backends, safety-gate/conceptual paths).
    # `_build_run_metadata` (routes_experiments.py) copies this onto
    # `RunMetadata.llm_usage` for the frontend's Run Information card
    # — this field is the source of truth; RunMetadata's copy is a
    # convenience echo, never computed independently.
    llm_usage: LLMUsage | None = None
