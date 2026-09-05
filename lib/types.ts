export type ConfidenceLevel = 'HIGH' | 'MEDIUM' | 'LOW';

export type StepStatus = 'pending' | 'running' | 'done';

/**
 * Phase 8 — the REAL, backend-computed outcome of a pipeline stage,
 * distinct from `StepStatus` above (which is the frontend's
 * fake-timer "is this row's spinner still going" concept, driven
 * client-side before the /analyze response even arrives).
 */
export type ExecutionStepStatus = 'SUCCESS' | 'SKIPPED' | 'WARNING' | 'FAILED';

export interface ExecutionStep {
  id: string;
  label: string;
  group: 'Classifier' | 'Planner' | 'Capability' | 'Decision Engine';
  detail: string;
  /** Optional only for backward compatibility with pre-Phase-8 persisted rows; the backend always sends it today. */
  status?: ExecutionStepStatus;
}

/**
 * Pipeline-streaming progress events (POST /experiments/analyze/stream,
 * see backend/app/api/routes_experiments.py). Purely additive to the
 * existing ExecutionStep contract above — `stage` here is the SAME
 * string as `ExecutionStep.id` for every graph-node-backed stage
 * (classifier/planner/validation/experiment/funnel/guardrail/
 * knowledge_base/decision), so a `stage_started`/`stage_completed`
 * event can be matched directly against the eventual real
 * `ExecutionStep` with that `id` once the final `result` event
 * arrives — no separate mapping table needed on the frontend.
 */
export type PipelineStreamEvent =
  | { type: 'stage_started'; stage: string; message: string }
  | { type: 'stage_completed'; stage: string; message: string; durationMs: number }
  | { type: 'error'; stage: string; message: string }
  | {
      type: 'result';
      data: {
        experimentId: string;
        report: ExperimentReport;
        executionSteps: ExecutionStep[];
        relatedExperiments: RelatedExperiment[];
        stageTimings: { stage: string; status: string; durationMs: number; error: string | null }[];
      };
    }
  | { type: 'pipeline_completed' };

export interface QualityCheck {
  label: string;
  passed: boolean;
  detail: string;
}

export type HypothesisTestType = 'welch_t_test' | 'mann_whitney_u' | 'chi_square' | 'fishers_exact';

export interface StatResult {
  metric: string;
  testType: HypothesisTestType;
  testName: string;
  statistic: number;
  selectionReason: string;
  control: string;
  variant: string;
  delta: string;
  pValue: number;
  significant: boolean;
  ciLower: string;
  ciUpper: string;
  // Multi-arm metadata — optional so existing two-arm consumers are
  // unaffected. Already sent by the backend (app/schemas/statistics.py)
  // but previously missing from this type.
  observedRelativeEffect?: number | null;
  comparison?: string | null;
  isOmnibus?: boolean;
  adjustedPValue?: number | null;
  multipleTestingMethod?: string | null;
  referenceArm?: string | null;
  arm?: string | null;
  practicalSignificant?: boolean | null;
  isWinner?: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

/**
 * SSE event shapes from POST /experiments/{id}/chat/stream (see
 * backend/app/api/routes_experiments.py's `follow_up_chat_stream`).
 * `token` may arrive many times in a row before the terminal `done`;
 * `error` can arrive mid-stream (see that route's docstring on partial
 * content) and is always followed by a `done` carrying whatever text
 * was actually generated — it does not replace `done`.
 */
export type ChatStreamEvent =
  | { type: 'token'; content: string }
  | { type: 'error'; message: string }
  | { type: 'done'; message: ChatMessage };

export interface KnowledgeBaseReference {
  source: string;
  heading: string;
  excerpt: string;
  relevanceScore: number;
}

/**
 * Canonical ship/no-ship-equivalent signal — always produced by
 * `determine_decision()` on the backend (app/schemas/report.py).
 * Never read from or written by the LLM.
 */
export type Decision = 'INVALID' | 'INCONCLUSIVE' | 'NO_GO' | 'GO_WITH_CAUTION' | 'GO';

/**
 * "Can we trust this experiment at all?" — SRM, critical quality
 * checks, conflicting variant duplicates. See ExperimentValidity in
 * app/schemas/report.py.
 */
export type ExperimentValidity = 'VALID' | 'CAUTION' | 'INVALID';

/**
 * NOT_AVAILABLE means "no guardrail metric was evaluated" — it is
 * NEVER equivalent to PASS. See GuardrailStatus in app/schemas/report.py.
 */
export type GuardrailStatus = 'PASS' | 'WARNING' | 'FAIL' | 'NOT_AVAILABLE';

/**
 * Guardrail REQUEST/availability state — independent of GuardrailStatus
 * above (which is only the evaluation outcome). See GuardrailRequestState
 * in app/schemas/guardrails.py. NOT_SPECIFIED (no guardrails requested)
 * must never be conflated with REQUESTED_NOT_FOUND (requested, but no
 * matching dataset column) — that conflation was the original bug.
 */
export type GuardrailRequestState =
  | 'NOT_SPECIFIED'
  | 'REQUESTED_NOT_FOUND'
  | 'PARTIALLY_AVAILABLE'
  | 'AVAILABLE';

/** One requested guardrail's resolution outcome. See GuardrailResolution in app/schemas/guardrails.py. */
export interface GuardrailResolution {
  requestedName: string;
  resolved: boolean;
  resolvedMetricLabel?: string | null;
}

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
  knowledgeBaseReferences: KnowledgeBaseReference[];
  // True when the KB was actually queried for this request (result may
  // still be empty if nothing cleared the relevance threshold) — lets
  // the UI distinguish that from "KB wasn't queried at all" instead of
  // treating both as identical empty arrays.
  knowledgeBaseAttempted?: boolean;
  // Set only when knowledge_base_node's retriever call itself raised
  // (index missing, I/O error, etc.) — never fabricated, never set
  // just because nothing cleared the relevance threshold. `null`/
  // `undefined` means "no retrieval failure occurred": either the KB
  // wasn't queried at all (`knowledgeBaseAttempted` false) or it ran
  // without error (`knowledgeBaseAttempted` true, references may
  // still legitimately be empty). See
  // ExperimentReport.knowledge_base_retrieval_error on the backend.
  knowledgeBaseRetrievalError?: string | null;
  // DECISION-AWARE RAG: the human-readable name of the specific
  // validity-blocking failure (e.g. "Outlier Detection", "Sample Ratio
  // Mismatch (SRM)") that Evidence & Sources was filtered against, set
  // whenever the backend's INVALID-specific relevance gate ran. Null/
  // undefined whenever the experiment isn't INVALID, or INVALID for a
  // reason the gate doesn't recognize. See
  // ExperimentReport.knowledge_base_blocking_issue on the backend.
  knowledgeBaseBlockingIssue?: string | null;
  bootstrapCiLower?: number | null;
  bootstrapCiUpper?: number | null;
  bootstrapIterations?: number | null;
  /** Phase 3 — deterministic decision support. Null whenever no
   *  hypothesis was provided (see DecisionSupport docstring on the
   *  backend); existing reports without a hypothesis are otherwise
   *  completely unaffected. */
  decisionSupport?: DecisionSupport | null;

  /**
   * Phase 1/2 — the analyst's stated hypothesis and its deterministic
   * SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED evaluation (see
   * HypothesisEvaluation above). Both are set directly by
   * decision_node.py from app/stats/hypothesis_evaluator.py — never
   * computed or altered by a ReportGenerator/LLM. `hypothesis` is null
   * when the analyst didn't provide one. `hypothesisEvaluation` is
   * null both when there's no hypothesis and when evaluation was
   * unavailable for a present one — see that type's docstring.
   *
   * IMPORTANT: this is a distinct concept from `decision` /
   * `decisionReason` below — the hypothesis verdict answers "was the
   * stated hypothesis supported?" while `decision` answers "what
   * should the business do?" (e.g. a SUPPORTED hypothesis can still
   * carry a GO_WITH_CAUTION decision when a guardrail warns). Never
   * collapse the two into a single field when rendering.
   */
  hypothesis?: Hypothesis | null;
  hypothesisEvaluation?: HypothesisEvaluation | null;

  // --- canonical decision model — already sent by the backend
  // (app/schemas/report.py) but previously missing from this type,
  // so the frontend silently dropped them. Optional here only to
  // stay backward compatible with any already-persisted history rows
  // that predate this field; the backend always sends them today.
  experimentValidity?: ExperimentValidity;
  guardrailStatus?: GuardrailStatus;
  practicalSignificance?: boolean | null;
  decision?: Decision;
  decisionReason?: string;
  recommendationConfidence?: ConfidenceLevel;

  // --- Guardrail REQUEST state (guardrail root-cause fix) ---------------
  // Independent of guardrailStatus above — always read both together.
  // Defaults (NOT_SPECIFIED / []) on any report predating this field.
  guardrailRequestState?: GuardrailRequestState;
  requestedGuardrails?: string[];
  guardrailResolutions?: GuardrailResolution[];

  /**
   * Phase 5 — deterministic segmentation, supporting evidence only.
   * Set directly by decision_node.py from ReportFacts.segmentation_result
   * — never computed or altered by a ReportGenerator/LLM. Null when
   * experiment_node never ran (e.g. SRM-failed / validation-only runs).
   * Never consulted by determine_decision() on the backend, and must
   * never be treated as authoritative here either.
   */
  segmentation?: SegmentationResult | null;

  /**
   * TRUE stratified analysis (deliberately separate from `segmentation`
   * above — never to be confused with or labeled as "Segment
   * Analysis"). Set directly by decision_node.py from
   * state.stratification_result (when experiment_node actually ran)
   * or a NOT_RUN fact (when the SAME validity gate that blocks the
   * ordinary hypothesis test — SRM / conflicting variant assignment /
   * a critical quality failure — also blocked stratified inference).
   * Null only when stratified analysis was never requested at all.
   * See app/schemas/stratification.py.
   */
  stratification?: StratificationResult | null;

  /**
   * Phase 7 — deterministic Decision Audit Trail. Set directly by
   * decision_node.py from build_decision_audit_trail()
   * (app/stats/decision_audit.py) — never computed or altered by a
   * ReportGenerator/LLM. `decisionAudit.decision` always mirrors
   * `decision` above; this is explanatory only and never a second,
   * independent decision. Optional for backward compatibility with
   * any already-persisted history rows that predate this field.
   */
  decisionAudit?: DecisionAuditTrail | null;

  /**
   * Product improvement — deterministic Decision Narrative (Why this
   * decision / What prevents a full GO / What to monitor / Recommended
   * next step). Set directly by decision_node.py from
   * build_decision_narrative() — no LLM involved, purely explanatory,
   * never overrides `decision` itself. Optional/null for older
   * persisted history rows that predate this field.
   */
  decisionNarrative?: DecisionNarrative | null;

  /**
   * Phase 8 — structured run metadata (see RunMetadata's docstring on
   * the backend). Always populated once decision_node has run; only
   * absent on history rows persisted before Phase 8.
   */
  runMetadata?: RunMetadata | null;
  /**
   * Phase 8 — populated ONLY when LLMReportGenerator actually
   * attempted an LLM call for this run and it failed, silently
   * falling back to the deterministic template report until now. Null
   * on every other path, including the normal case where no LLM was
   * ever attempted — that's configured behavior, not a fallback.
   */
  reportFallbackReason?: string | null;
}

/** Why a candidate column was NOT used as a segmentation dimension. */
export type SegmentSkipReason =
  | 'high_cardinality'
  | 'too_few_distinct_values'
  | 'non_categorical'
  | 'is_identifier_or_metric_column'
  | 'excessive_missing_values';

export interface SkippedDimension {
  column: string;
  reason: SegmentSkipReason;
  detail: string;
}

/** Whether a single segment value had enough data to test at all. */
export type SegmentSampleSizeStatus = 'sufficient' | 'insufficient';

/**
 * One segment value (e.g. dimension="device", value="mobile") within
 * one segmentation dimension. `statResult` is null when
 * `sampleSizeStatus` is 'insufficient' — a test is never run on a
 * segment too small to trust.
 */
export interface SegmentEffect {
  segmentValue: string;
  controlN: number;
  variantN: number;
  sampleSizeStatus: SegmentSampleSizeStatus;
  statResult?: StatResult | null;
  skipDetail?: string | null;
}

/** Full within-dimension analysis (e.g. all device values). */
export interface SegmentDimensionResult {
  dimension: string;
  segmentEffects: SegmentEffect[];
  multipleTestingMethod: string;
  /** segment_value(s) whose adjusted p-value is still significant. */
  reliableSegmentValues: string[];
  /** True iff at least one segment is reliably different from the overall result. */
  hasHeterogeneousEffect: boolean;
}

/**
 * Top-level Phase 5 fact, attached to the report as supporting
 * evidence only — never consulted by the backend's canonical
 * decision, and must never be rendered as if it overrides it.
 */
export interface SegmentationResult {
  /** False when segmentation could not run at all (e.g. no usable dimensions). */
  ran: boolean;
  /** Human-readable summary of what ran / why not, always populated. */
  reason: string;
  usableDimensions: string[];
  skippedDimensions: SkippedDimension[];
  dimensionResults: SegmentDimensionResult[];
  /** The guardrail threshold actually applied, for transparency. */
  minSegmentSize: number;
}

/**
 * TRUE stratified analysis types — mirrors app/schemas/stratification.py.
 * Deliberately separate from SegmentationResult above (see that
 * module's docstring): stratification estimates ONE combined treatment
 * effect using pre-specified strata weighting; segmentation explores
 * whether the effect differs across segments. Never render one as the
 * other.
 */

/** Closed set of reasons a requested stratification variable cannot be used. */
export type StratificationIneligibilityReason =
  | 'column_not_found'
  | 'is_treatment_assignment'
  | 'perfectly_associated_with_assignment'
  | 'no_strata_with_both_variants'
  | 'multi_arm_not_supported'
  | 'all_strata_too_sparse';

/**
 * Descriptive summary of one stratum value — reported regardless of
 * whether the overall variable ends up eligible, so the report can
 * always show "N observations, X% of total" for transparency.
 * `controlOutcomeRate`/`variantOutcomeRate` are the stratum's own
 * descriptive per-arm rate/mean (only populated when the stratum has
 * both variants) — the "stratum-level estimate" alongside the single
 * combined `StratifiedEstimate` below.
 */
export interface StratumSummary {
  stratumValue: string;
  controlN: number;
  variantN: number;
  totalN: number;
  proportionOfTotal: number;
  hasBothVariants: boolean;
  /** Both arms meet the minimum-per-arm threshold to be used in the combined estimate. */
  sufficient: boolean;
  controlOutcomeRate?: number | null;
  variantOutcomeRate?: number | null;
}

/**
 * Whether the requested stratification variable can be used for TRUE
 * stratified inference on THIS dataset, and why. Always populated when
 * present — `reason` is a plain-English explanation regardless of
 * outcome (e.g. the `landing_page` rejection: perfectly associated
 * with treatment assignment).
 */
export interface StratificationEligibility {
  stratificationColumn: string;
  eligible: boolean;
  reason: string;
  ineligibilityReason?: StratificationIneligibilityReason | null;
  strata: StratumSummary[];
  totalN: number;
  missingCount: number;
  missingProportion: number;
  minArmSize: number;
  sparseStratumValues: string[];
}

/**
 * The combined (across-strata) treatment-effect estimate — never a
 * naive average of per-stratum rates/means. Produced by an
 * inverse-variance-weighted fixed-effect combination of each stratum's
 * own control-vs-variant effect.
 */
export interface StratifiedEstimate {
  method: string;
  metricLabel: string;
  strataUsed: number;
  effectEstimate: number;
  standardError: number;
  ciLower: number;
  ciUpper: number;
  pValue: number;
  significant: boolean;
}

/**
 * Whether stratified inference actually executed. `not_run` is
 * distinct from `eligibility.eligible === false`: the latter means the
 * experiment ran and the variable itself was determined unusable;
 * `not_run` means the experiment never ran at all because it failed
 * the SAME validity gate (SRM / conflicting variant assignment /
 * critical quality failure) the ordinary hypothesis test is subject
 * to. In that case `eligibility`/`estimate` are both null/absent and
 * `notRunReason` explains why.
 */
export type StratificationStatus = 'ran' | 'not_run';

/**
 * Top-level fact rendered in its OWN "Stratified Analysis" section —
 * deliberately separate from `SegmentationResult`. `eligibility`/
 * `estimate` are absent whenever `status === 'not_run'`. `estimate` is
 * also absent whenever `eligibility.eligible` is false even when
 * `status === 'ran'`.
 */
export interface StratificationResult {
  status: StratificationStatus;
  stratificationColumn: string;
  eligibility?: StratificationEligibility | null;
  estimate?: StratifiedEstimate | null;
  /** Populated ONLY when status === 'not_run'. */
  notRunReason?: string | null;
}

export interface HistorySession {
  id: string;
  name: string;
  date: string;
  confidence: ConfidenceLevel;
  metric: string;
  summary: string;
  report: ExperimentReport;
}

export interface DatasetInfo {
  type: string;
  variants: number;
  users: number;
  metricLabel: string;
  /** Already-detected metric columns — reused by the Hypothesis form's
   *  primary-metric picker so it never becomes a second, independent
   *  metric-selection system. */
  availableMetrics?: string[];
  /** Deterministic explanation of why metricLabel was picked over any
   *  other available metric — see DatasetInfo.metric_selection_reason
   *  on the backend. */
  metricSelectionReason?: string;

  // --- Full classification visibility (semantic fields) ---------------
  // Additive to the fields above. `userIdColumn` / `variantColumn` are
  // real column names when resolved; everything else is a display-only
  // "Candidates" list — a naming/shape heuristic, never a promise of
  // eligibility. See backend/app/stats/dataset_classifier.py.
  /** Real column name used as the experiment-unit identifier, if resolved. */
  userIdColumn?: string | null;
  /** Real column name used as the variant/arm column, if resolved. */
  variantColumn?: string | null;
  /** Distinct arm values found in variantColumn, e.g. ['control', 'treatment']. */
  variantValues?: string[];
  /** Same value as metricLabel, under the name used by the classification card. */
  primaryMetric?: string;
  /** availableMetrics minus the primary metric. */
  additionalMetrics?: string[];
  /** Candidate low-cardinality, non-numeric columns that MIGHT be usable
   *  for stratified analysis — never a promise of eligibility. */
  stratificationCandidates?: string[];
  /** Candidate metric(s), among availableMetrics, whose name suggests a
   *  regression-watch metric rather than the primary metric. Naming
   *  heuristic only — no statistical guardrail check is run. */
  guardrailCandidates?: string[];
  /** Candidate pre-experiment CUPED covariate, or empty if none found. */
  covariateCandidates?: string[];
}

export interface Settings {
  cuped: boolean;
  bootstrap: boolean;
  /**
   * Optional per-run LLM model override — one of the ids returned by
   * GET /system/models (see AvailableModel / getAvailableModels() in
   * lib/api.ts). Omitting it (undefined) means "use the backend's
   * configured default", exactly like before this field existed.
   * Never a free-typed string — always chosen from the dropdown.
   */
  model?: string;
  /**
   * Guardrail metrics EXPLICITLY selected by the user for this analysis
   * (e.g. ['Revenue', 'Bounce Rate']) — chosen from DatasetInfo.
   * availableMetrics, never free-typed. Optional/omittable — every
   * existing request without this field is unaffected and reads as
   * "no guardrails requested". Distinct from DatasetInfo.
   * guardrailCandidates, which is only an automatically detected
   * SUGGESTION, never a request on its own. See AnalysisSettings.
   * guardrail_metrics on the backend.
   */
  guardrailMetrics?: string[];
}

/** One selectable entry in the model dropdown — GET /system/models. */
export interface AvailableModel {
  id: string;
  label: string;
}

/** GET /system/models response. */
export interface AvailableModelsResponse {
  models: AvailableModel[];
  defaultModel: string;
}

/**
 * Phase 1 — structured Experiment Hypothesis, captured before the
 * result is known. Optional everywhere: omitting it entirely must not
 * change any existing analysis flow (see experiment-config.tsx /
 * workspace-view.tsx and the backend's AnalyzeExperimentRequest.hypothesis
 * for the same "purely additive" guarantee on the wire).
 *
 * `expectedEffectRelative` is a RELATIVE fraction, never percentage
 * points: 0.05 means "+5% relative" (a 10% baseline becomes 10.5%),
 * NOT "+5 percentage points" (which would be 15%). See
 * hypothesis-form.tsx for how this is surfaced to the user
 * unambiguously.
 *
 * This phase does not compute or display a verdict — no
 * SUPPORTED/NOT_SUPPORTED comparison exists yet.
 */
export type ExpectedDirection = 'increase' | 'decrease' | 'no_change';

export interface Hypothesis {
  statement: string;
  primaryMetric: string;
  expectedDirection: ExpectedDirection;
  expectedEffectRelative?: number | null;
  rationale?: string | null;
}

/**
 * Phase 2 — deterministic comparison of `Hypothesis` against the
 * matched `StatResult`. Computed once, server-side, by
 * app/stats/hypothesis_evaluator.py — no LLM ever calculates or
 * overrides `verdict`; the frontend only renders these fields as-is.
 * See HypothesisEvaluation in app/schemas/hypothesis_evaluation.py
 * for the authoritative field list — do not invent fields here.
 *
 * `verdict` (and the other comparison fields) are null when
 * evaluation itself was unavailable — check `metricMatched` /
 * `evaluationNote` for why, never treat a null verdict as REJECTED.
 */
export interface HypothesisEvaluation {
  hypothesisPresent: boolean;
  expectedDirection: ExpectedDirection;
  expectedEffectRelative?: number | null;
  observedEffectRelative?: number | null;
  directionSupported?: boolean | null;
  statisticallySignificant?: boolean | null;
  effectAchievementRatio?: number | null;
  verdict?: HypothesisVerdict | null;
  metricMatched: boolean;
  evaluationNote?: string | null;
}

// ---------------------------------------------------------------------------
// Sample size planning — pre-experiment "Create Experiment" screen.
// See backend/app/schemas/hypothesis.py's SampleSizePlanRequest/Response
// for the authoritative field list; mirror it exactly here.
// ---------------------------------------------------------------------------

export type PlanningMetricType = 'binary' | 'continuous_monetary' | 'continuous_general';

export interface SampleSizePlanRequest {
  metricType: PlanningMetricType;
  baselineRate: number;
  baselineStd?: number | null;
  mdeRelativePct: number;
  numVariants: number;
  dailyTrafficPerArm?: number | null;
}

export interface SampleSizePlan {
  requiredNPerArm: number;
  requiredNTotal: number;
  alpha: number;
  targetPower: number;
  effectSize: number;
}

export interface SampleSizePlanResponse {
  plan: SampleSizePlan;
  estimatedDays?: number | null;
}

/**
 * A fully specified experiment plan, captured on the "Create Experiment"
 * screen before any dataset is selected. Kept client-side only (not
 * persisted to the backend) — once the analyst picks a dataset, its own
 * hypothesis (see Hypothesis above) becomes the source of truth for
 * that specific run; this plan exists to inform that choice, e.g. by
 * carrying the statement/primaryMetric/guardrails forward as sensible
 * defaults into HypothesisForm.
 */
export interface ExperimentPlan {
  statement: string;
  primaryMetric: string;
  expectedDirection: ExpectedDirection;
  guardrailMetricNames: string[];
  sampleSizeRequest?: SampleSizePlanRequest | null;
  sampleSizeResult?: SampleSizePlanResponse | null;
}

/**
 * Phase 3 — deterministic Decision Support. Every field here is
 * either copied straight from an existing deterministic backend fact
 * or computed by plain arithmetic in app/stats/decision_support.py —
 * no LLM ever computes any of these numbers, and the frontend must
 * never (re)calculate any of them either; it only renders what the
 * backend supplies. See DecisionSupport in app/schemas/decision_support.py
 * for the authoritative field list — do not invent fields here.
 */
export type HypothesisVerdict = 'SUPPORTED' | 'PARTIALLY_SUPPORTED' | 'NOT_SUPPORTED';

export interface AdditionalMetricComparison {
  metric: string;
  baselineValue?: number | null;
  observedValue?: number | null;
  absoluteChange?: number | null;
  relativeChange?: number | null;
  statisticallySignificant: boolean;
  direction: 'increase' | 'decrease' | 'no_change';
}

export interface GuardrailFinding {
  metric: string;
  observedValue?: number | null;
  relativeChange?: number | null;
  statisticallySignificant: boolean;
  violated: boolean;
}

export interface DecisionSupport {
  available: boolean;

  primaryMetric?: string | null;
  baselineValue?: number | null;
  observedValue?: number | null;
  observedEffectAbsolute?: number | null;
  observedEffectRelative?: number | null;
  expectedEffectRelative?: number | null;
  expectedValue?: number | null;
  effectAchievementRatio?: number | null;
  statisticalSignificance?: boolean | null;
  hypothesisVerdict?: HypothesisVerdict | null;
  businessInterpretation?: string | null;

  /** 'population_scaled' | 'unavailable' — never inferred/parsed on the frontend. */
  impactCalculationMethod: 'population_scaled' | 'unavailable';
  baselineExpectedCount?: number | null;
  observedCount?: number | null;
  incrementalCount?: number | null;

  additionalMetrics: AdditionalMetricComparison[];

  guardrailFindings: GuardrailFinding[];
  guardrailViolated: boolean;

  warnings: string[];
}

/**
 * Phase 7 — Decision Audit Trail. Every field is either copied
 * straight from an existing deterministic backend fact or a short
 * deterministic sentence built from those facts
 * (app/stats/decision_audit.py) — never computed or (re)decided in
 * the frontend. `decision` always mirrors `ExperimentReport.decision`;
 * this is explanatory only, never a second decision engine. See
 * DecisionAuditTrail in app/schemas/decision_audit.py for the
 * authoritative field list — do not invent fields here.
 */
export type AuditStatus = 'pass' | 'warning' | 'fail' | 'info' | 'not_available';
export type AuditImpact = 'supports_decision' | 'blocks_decision' | 'limits_confidence' | 'neutral';
export type AuditCategory =
  | 'validity'
  | 'data_quality'
  | 'statistical_significance'
  | 'practical_significance'
  | 'power'
  | 'guardrails'
  | 'segmentation';

export interface AuditFact {
  status: AuditStatus;
  category: AuditCategory;
  label: string;
  value: string;
  impact: AuditImpact;
  detail?: string | null;
}

export interface DecisionAuditTrail {
  decision: Decision;
  headline: string;
  rationale: string[];
  supportingFacts: AuditFact[];
  warnings: AuditFact[];
  criticalChecks: AuditFact[];
  statisticalEvidence: AuditFact[];
  practicalSignificanceEvidence?: AuditFact | null;
  powerEvidence?: AuditFact | null;
  guardrailEvidence: AuditFact;
  segmentationEvidence: AuditFact;
  decisionImpact: string;
}

/** Product improvement — see DecisionNarrative in app/schemas/decision_narrative.py. */
export interface MonitoringInfo {
  primaryMetric: string | null;
  guardrailsEvaluated: string[];
  potentialMonitoringMetrics: string[];
}

/** Product improvement — see DecisionNarrative in app/schemas/decision_narrative.py. */
export interface DecisionNarrative {
  whyThisDecision: string[];
  whatPreventsFullGo: string[];
  whatWouldChangeDecision: string[];
  monitoring: MonitoringInfo;
  recommendedNextStep: string;
}

/**
 * Token/cost accounting for the report-generation LLM call — see
 * LLMUsage's docstring in app/schemas/execution.py. Every field is
 * independently optional (not every provider/response includes every
 * value); the object itself is only present when an LLM call actually
 * happened for the run at all.
 */
export interface LLMUsage {
  promptTokens?: number | null;
  completionTokens?: number | null;
  totalTokens?: number | null;
  costUsd?: number | null;
}

/** Phase 8 — see RunMetadata's docstring in app/schemas/execution.py. */
export interface RunMetadata {
  runId: string;
  timestamp: string;
  datasetName: string;
  datasetClassification: string;
  userCount: number;
  variantCount: number;
  primaryMetric?: string | null;
  analysisMode: string;
  selectedModel?: string | null;
  plannerBackend: string;
  reportBackend: string;
  executionStatus: ExecutionStepStatus;
  /** None whenever no LLM call was attempted for this run at all. */
  llmUsage?: LLMUsage | null;
}

/** Read-only backend config — the model is set via `.env`, never by the UI. */
export interface SystemInfo {
  llmProvider: string;
  llmModel: string;
  plannerBackend: string;
  reportBackend: string;
  /** Phase 8 — safe operational visibility only; never secrets/keys. */
  knowledgeBaseAvailable: boolean;
  availableModelsCount: number;
}

/** One row in persisted Experiment History (GET /experiments). */
export interface ExperimentSummary {
  experimentId: string;
  createdAt: string;
  datasetName: string;
  userPrompt: string;
  decision: string;
  confidence: ConfidenceLevel;
  primaryMetric: string;
}

/** A reopened experiment (GET /experiments/{id}). */
export interface ExperimentDetail {
  experimentId: string;
  createdAt: string;
  datasetId: string;
  datasetName: string;
  userPrompt: string;
  report: ExperimentReport;
  executionSteps: ExecutionStep[];
  chatMessages: ChatMessage[];
  relatedExperiments: RelatedExperiment[];
}

/**
 * A prior run against the same dataset — plain structured retrieval
 * from the backend (ExperimentStore.list_related), not semantic/LLM
 * memory. Shown as short factual context, e.g. "reviewed before,
 * decision was X".
 */
export interface RelatedExperiment {
  experimentId: string;
  createdAt: string;
  userPrompt: string;
  decision: string;
  confidence: ConfidenceLevel;
  primaryMetric: string;
}

// ---------------------------------------------------------------------------
// ExperimentDefinition — Experiment Platform layer, Phase 1/2.
// Mirrors backend/app/schemas/experiment_definition.py exactly. This is
// the NEW pre-analysis planning entity (Experiment Library / Design /
// Variants / Targeting / Metrics) — a completely separate concept from
// `ExperimentSummary`/`ExperimentDetail` above, which represent an
// already-completed analysis run. See that schema's module docstring
// for the full architectural boundary (planning metadata only; never
// read by the stats engine in this phase).
// ---------------------------------------------------------------------------

export type ExperimentStatus =
  | 'draft'
  | 'ready'
  | 'running'
  | 'completed'
  | 'needs_investigation'
  | 'invalid'
  | 'shipped'
  | 'archived';

export type HypothesisRole = 'primary' | 'secondary';

export interface RoledHypothesis {
  role: HypothesisRole;
  hypothesis: Hypothesis;
}

export type MetricRole = 'primary' | 'secondary' | 'guardrail';

export interface ExperimentMetric {
  name: string;
  role: MetricRole;
  type: PlanningMetricType;
  description?: string | null;
  fieldDefinition?: string | null;
}

export interface Variant {
  id: string;
  name: string;
  description?: string | null;
  isControl: boolean;
  allocationPct: number;
}

export interface Targeting {
  countries: string[];
  platforms: string[];
  devices: string[];
  userType?: string | null;
  acquisitionChannel?: string | null;
  userSegment?: string | null;
  trafficAllocationPct?: number | null;
}

/** Mirrors RandomizationUnit in schemas/experiment_definition.py.
 *  Descriptive planning metadata only — see that enum's docstring;
 *  nothing in this phase actually assigns users based on this value. */
export type RandomizationUnit = 'user' | 'session' | 'device';

export interface Exposure {
  assignedUsers?: number | null;
  exposedUsers?: number | null;
}

export type DataSourceType = 'uploaded_csv' | 'existing_dataset' | 'public_dataset';

export interface DataSourceRef {
  type: DataSourceType;
  datasetId?: string | null;
  datasetName?: string | null;
}

/** Shared fields for create/update requests and the full record — same
 *  split as ExperimentDefinitionBase in the backend schema. */
export interface ExperimentDefinitionFields {
  name: string;
  productArea?: string | null;
  owner?: string | null;
  team?: string | null;
  status: ExperimentStatus;
  problemStatement?: string | null;
  objective?: string | null;
  hypotheses: RoledHypothesis[];
  variants: Variant[];
  targeting: Targeting;
  randomizationUnit: RandomizationUnit;
  metrics: ExperimentMetric[];
  exposure: Exposure;
  expectedDurationDays?: number | null;
  targetSampleSize?: number | null;
  mdeRelativePct?: number | null;
  dataSource?: DataSourceRef | null;
}

export interface ExperimentDefinition extends ExperimentDefinitionFields {
  id: string;
  createdAt: string;
  updatedAt: string;
}

/** Lightweight row for the Experiment Library list — mirrors
 *  ExperimentDefinitionSummary in the backend schema exactly. */
export interface ExperimentDefinitionSummary {
  id: string;
  name: string;
  status: ExperimentStatus;
  productArea?: string | null;
  owner?: string | null;
  primaryMetric?: string | null;
  createdAt: string;
  updatedAt: string;
}

/** Body for POST /experiment-definitions. Every field beyond `name` is
 *  optional — `status` defaults to 'draft' server-side if omitted. */
export type ExperimentDefinitionCreateRequest = Partial<
  Omit<ExperimentDefinitionFields, 'name' | 'status'>
> & {
  name: string;
  status?: ExperimentStatus;
};

/** Body for PATCH /experiment-definitions/{id} — every field optional,
 *  only fields present in the object are changed server-side. */
export type ExperimentDefinitionUpdateRequest = Partial<ExperimentDefinitionFields>;

