'use client';

import { useState, type ReactNode } from 'react';
import {
  Star,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Lightbulb,
  ArrowRight,
  FileBarChart,
  ShieldCheck,
  Target,
  HelpCircle,
  Rows3,
  ListChecks,
  Info,
  BookOpen,
  ChevronDown,
  FlaskConical,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import type {
  ExperimentReport,
  DecisionSupport,
  StratificationResult,
  DecisionAuditTrail,
  AuditFact,
  DecisionNarrative,
  KnowledgeBaseReference,
  StatResult,
  Hypothesis,
  HypothesisEvaluation,
} from '@/lib/types';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { downloadReportPdf } from '@/lib/pdf';
import {
  selectPrimaryStat,
  primaryEffectParts,
  formatPrimaryEffect,
  formatPercent,
  formatNumber,
  hypothesisResultLabel,
  findHypothesisStatResult,
  verdictBadgeClass,
} from '@/lib/report-format';
import { ExperimentHeader } from '@/components/report/experiment-header';
import { SummaryCards } from '@/components/report/summary-cards';
import { KpiGrid } from '@/components/report/kpi-grid';
import { HeroCard } from '@/components/report/hero-card';
import { GuardrailSection } from '@/components/report/guardrail-section';
import { RecommendationCard } from '@/components/report/recommendation-card';
import { SegmentAnalysisSection } from '@/components/report/segment-analysis-section';
import { CopilotSummary } from '@/components/report/copilot-summary';
import { PrimaryMetricChart } from '@/components/report/primary-metric-chart';
import { SrmPanel } from '@/components/report/srm-panel';

interface ReportCardProps {
  report: ExperimentReport;
  /** Optional context used only to label the PDF export — never sent anywhere. */
  datasetName?: string;
  experimentId?: string;
  prompt?: string;
}

/**
 * Plain-language description for a technical metric, shown via a small
 * (i) icon next to the label — Karolina's feedback: the numbers make
 * sense to her, but unlabeled statistical jargon (MDE, SRM, power, ...)
 * does not. Purely presentational — never affects any computed value.
 */
function InfoTooltip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Info
          className="h-3 w-3 shrink-0 cursor-help text-muted-foreground hover:text-muted-foreground"
          aria-label="More info"
        />
      </TooltipTrigger>
      <TooltipContent className="max-w-[260px] text-xs">{text}</TooltipContent>
    </Tooltip>
  );
}

/**
 * Karolina's feedback item 1 — the Executive Summary text must adapt to
 * the actual experiment outcome, not just restate one generic template.
 * Built entirely from fields the backend already computed and already
 * sends today (`report.decision`, `report.experimentValidity`,
 * `stats[0].significant/delta/ciLower/ciUpper/pValue`,
 * `report.decisionReason`) — no new backend computation, no numbers
 * invented client-side. `report.executiveSummary` (the longer backend
 * narrative, with metric-selection reasoning etc.) remains available
 * further down inside Statistical Results, so nothing is lost — this
 * is just the 2-5 sentence version for the very top of the report.
 */
/**
 * Mirrors the backend's `select_primary_stat()` (app/graph/report_generator.py)
 * EXACTLY — same precedence, same fallback order — so the frontend never
 * treats an omnibus row (`isOmnibus`, e.g. "Effect: Omnibus", CI [N/A,
 * N/A] for a 3+-arm experiment) as the decision-facing effect. Before
 * this, `report.stats[0]` was used directly, which happens to BE the
 * omnibus row whenever one exists (it's always listed first) — that
 * produced "Conversion Rate increased by Omnibus... 95% CI [N/A, N/A]"
 * in the Executive Summary for any multi-arm experiment, even though
 * a specific, fully-computed pairwise comparison (the one the
 * deterministic Decision section already correctly describes) was
 * right there in `report.stats[1..]`.
 */
type SummaryState =
  | 'invalid'
  | 'no_test'
  | 'underpowered'
  | 'non_significant'
  | 'significant';

function classifySummaryState(report: ExperimentReport): SummaryState {
  if (report.decision === 'INVALID' || report.experimentValidity === 'INVALID') {
    return 'invalid';
  }
  const primary = selectPrimaryStat(report);
  if (!primary) return 'no_test';
  if (primary.significant) return 'significant';
  // Non-significant: the backend appends "...also underpowered..." to
  // decisionReason specifically (and only) for the underpowered case —
  // see determine_decision() in report_generator.py — so this is a
  // read of an existing computed fact, not a new inference.
  const underpowered = (report.decisionReason ?? '').toLowerCase().includes('underpowered');
  return underpowered ? 'underpowered' : 'non_significant';
}

function directionWord(delta: string): 'increased' | 'decreased' {
  return delta.trim().startsWith('-') ? 'decreased' : 'increased';
}

function formatPValueClause(pValue: number | undefined): string {
  if (pValue == null) return '';
  return ` (p ${pValue < 0.001 ? '< 0.001' : pValue.toFixed(3)})`;
}

/**
 * `StatResult.ciLower`/`ciUpper` are display STRINGS, not numbers —
 * for the omnibus row (`is_omnibus=True`, hypothesis_tests.py) they
 * are the literal string `"N/A"` (statistically correct — an omnibus
 * chi-square/ANOVA/Kruskal-Wallis test has no single delta, so no
 * single CI). `Boolean("N/A")` is `true` (any non-empty string is
 * truthy), so a naive `ciLower && ciUpper` check treats that literal
 * "N/A" as if it were a real bound — producing sentences like "the
 * 95% CI [N/A, N/A] includes zero", which is not just ugly, it's
 * factually wrong (there IS no interval, not an interval of zero
 * width). This is the actual check for "is there a usable interval
 * to report".
 */
function isRealCi(value: string | undefined | null): boolean {
  if (!value) return false;
  return value.trim().toUpperCase() !== 'N/A';
}

const RECOMMENDATION_LABEL: Record<string, string> = {
  GO: 'Ship this experiment.',
  // GO_WITH_CAUTION's text is decided per-report by
  // recommendationLabelFor() below — it must never imply guardrails
  // were evaluated when they weren't (guardrail root-cause fix).
  NO_GO: 'Do not ship — the effect is not large enough, or is in the wrong direction, to justify this change.',
  INCONCLUSIVE: 'Do not ship yet — this result is inconclusive.',
  INVALID: 'Do not use this result for a ship decision.',
};

/**
 * GO_WITH_CAUTION reads very differently depending on WHY it's
 * cautious: "keep monitoring the guardrail metrics closely" is only
 * true once guardrails were actually evaluated (they could still
 * regress later) — if they were requested but never found in the
 * dataset, there's nothing to "keep monitoring", and saying so implies
 * evaluation happened when it didn't (doc2 §2 / doc3 §12). See
 * report.guardrailRequestState (guardrailRequestState.ts equivalent —
 * GuardrailRequestState in app/schemas/guardrails.py).
 */
function recommendationLabelFor(report: ExperimentReport): string | undefined {
  if (!report.decision) return undefined;
  if (report.decision !== 'GO_WITH_CAUTION') return RECOMMENDATION_LABEL[report.decision];

  if (report.guardrailStatus === 'PASS' || report.guardrailStatus === 'WARNING') {
    return 'Ship with caution — keep monitoring the guardrail metrics closely.';
  }
  if (report.guardrailRequestState === 'REQUESTED_NOT_FOUND') {
    return 'Ship with caution — the requested guardrail metrics could not be evaluated, so potential negative effects on those metrics are unverified.';
  }
  if (
    report.guardrailRequestState === 'AVAILABLE' ||
    report.guardrailRequestState === 'PARTIALLY_AVAILABLE'
  ) {
    // BUG FIX: this combination means the guardrail(s) resolved to a
    // real column but couldn't be statistically evaluated (e.g.
    // multi-arm) — a distinct fact from "not found in the dataset".
    // Previously fell through to the generic "no guardrails were
    // evaluated" wording below, which reads as if nothing was even
    // requested/found — directly contradicting a badge that (correctly)
    // shows the request resolved.
    return 'Ship with caution — the requested guardrail metrics were found but could not be statistically evaluated for this experiment, so potential negative effects on those metrics are unverified.';
  }
  return 'Ship with caution — this recommendation is based on the primary metric only; no guardrail metrics were evaluated.';
}

function buildStateAwareSummary(report: ExperimentReport, state: SummaryState): string {
  const primary = selectPrimaryStat(report);
  const recommendation = recommendationLabelFor(report);

  if (state === 'invalid') {
    const effectClause = primary ? ` The observed change (${primary.delta}) cannot be treated as a reliable result.` : '';
    return (
      `This experiment failed a critical validity check (data quality or randomization), ` +
      `so the result is not decision-safe.${effectClause} Fix the underlying issue and rerun before deciding.`
    );
  }

  if (state === 'no_test') {
    return (
      `Data quality checks were completed on ${report.qualityChecks.length} check(s), but no statistical ` +
      `test was run for this request. See Data Quality below for details.`
    );
  }

  if (!primary) {
    return report.executiveSummary;
  }

  const direction = directionWord(primary.delta);
  const hasCi = Boolean(primary.ciLower && primary.ciUpper);
  const ciInline = hasCi ? `the 95% CI [${primary.ciLower}, ${primary.ciUpper}]` : '';
  const ciSentence = hasCi ? ` The 95% CI is [${primary.ciLower}, ${primary.ciUpper}].` : '';

  if (state === 'underpowered') {
    return (
      `${primary.metric} ${direction} by ${primary.delta}.${ciSentence} The experiment does not have ` +
      `enough statistical power to support a reliable conclusion${formatPValueClause(primary.pValue)}. ` +
      `The experiment itself is valid — this is a sample-size issue, not a data-quality one. ` +
      `${recommendation ?? 'Continue the experiment to reach sufficient power before deciding.'}`
    );
  }

  if (state === 'non_significant') {
    return (
      `${primary.metric} ${direction} by ${primary.delta}, but ` +
      `${hasCi ? `${ciInline} includes zero` : 'this is not statistically significant'}` +
      `${formatPValueClause(primary.pValue)}. The experiment does not provide sufficient ` +
      `evidence of a real effect. ${recommendation ?? ''}`
    );
  }

  // significant
  const effectDisplay = formatPrimaryEffect(primary);

  // A user-supplied hypothesis threshold (a pre-registered business
  // number) supports the stronger "large enough to matter practically"
  // claim. Without one, `_practicalSignificance` fell back to the
  // post-hoc MDE — a sample-size-derived number, not a business
  // threshold — so the wording needs to say that explicitly rather than
  // implying practical significance is settled. See
  // `report_generator.py::_practical_significance_threshold` (backend)
  // for the same precedence rule this mirrors.
  const isPreRegisteredThreshold =
    report.decisionSupport?.available === true &&
    report.decisionSupport?.expectedEffectRelative != null &&
    report.decisionSupport?.primaryMetric === primary.metric;

  let practicalClause = '';
  let practicalSentence = '';
  if (report.practicalSignificance === true) {
    if (isPreRegisteredThreshold) {
      practicalClause = ' and is large enough to matter practically';
    } else {
      const mdeMatch = report.mde.match(/[\d.]+/);
      const mdePercent = mdeMatch ? mdeMatch[0] : report.mde;
      practicalSentence =
        ` The observed lift exceeds the post-hoc MDE of ${mdePercent}%, although practical ` +
        `significance should ultimately be assessed against a predefined business threshold.`;
    }
  } else if (report.practicalSignificance === false) {
    practicalClause = ', though it is smaller than the practical-significance threshold';
  }

  return (
    `${primary.metric} ${direction} by ${effectDisplay} compared with Control.${ciSentence} ` +
    `This is a statistically significant effect${formatPValueClause(primary.pValue)}${practicalClause}.` +
    `${practicalSentence} ` +
    `${recommendation ?? ''}`
  );
}

function decisionBadgeVariant(decision: string | null | undefined) {
  if (decision === 'GO') return 'border-success/25 bg-success/[0.08] text-success';
  if (decision === 'GO_WITH_CAUTION') return 'border-warning/25 bg-warning/[0.08] text-warning';
  if (decision === 'NO_GO' || decision === 'INVALID') return 'border-destructive/25 bg-destructive/[0.08] text-destructive';
  return 'border-border bg-secondary text-muted-foreground'; // INCONCLUSIVE / unknown
}

/**
 * The very first thing in the report — Karolina's #1 ask. Header line 3
 * (state-aware, concise summary) plus the key-result cards: Delta, 95%
 * CI, p-value, Recommendation — sourced via `selectPrimaryStat(report)`
 * (never the raw `report.stats[0]`, which is the omnibus row on any
 * multi-arm experiment) and `report.decision`, the same decision-facing
 * values the deterministic Decision section below already describes.
 */
function TopExecutiveSummary({ report }: { report: ExperimentReport }) {
  const primary = selectPrimaryStat(report);
  const state = classifySummaryState(report);
  const summaryText = buildStateAwareSummary(report, state);
  const isInvalid = state === 'invalid';

  return (
    <Card className="border-border shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <FileBarChart className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">Executive Summary</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-[13px] leading-relaxed text-foreground">{summaryText}</p>

        {primary && (
          <div className="flex flex-col items-stretch gap-3 sm:flex-row sm:items-center">
            <div
              className={cn(
                'flex-1 rounded-md border px-4 py-3 text-center',
                isInvalid ? 'border-border bg-secondary' : 'border-border bg-secondary'
              )}
            >
              <p className="text-[10px] font-medium text-muted-foreground">
                Effect
              </p>
              <p
                className={cn(
                  'mt-0.5 text-xl font-semibold',
                  isInvalid
                    ? 'text-muted-foreground'
                    : primary.significant
                      ? 'text-success'
                      : 'text-foreground'
                )}
              >
                {primaryEffectParts(primary).primary}
              </p>
              {primaryEffectParts(primary).secondary && (
                <p className="mt-0.5 text-[11px] font-medium text-muted-foreground">
                  {primaryEffectParts(primary).secondary}
                </p>
              )}
            </div>
            <div className="flex-1 rounded-md border border-border bg-secondary px-4 py-3 text-center">
              <p className="text-[10px] font-medium text-muted-foreground">
                95% CI
              </p>
              <p className="mt-0.5 font-mono text-base font-semibold text-foreground">
                [{primary.ciLower}, {primary.ciUpper}]
              </p>
            </div>
            <div className="flex-1 rounded-md border border-border bg-secondary px-4 py-3 text-center">
              <p className="text-[10px] font-medium text-muted-foreground">
                p-value
              </p>
              <p className="mt-0.5 font-mono text-base font-semibold text-foreground">
                {primary.pValue < 0.001 ? '<0.001' : primary.pValue.toFixed(3)}
              </p>
            </div>
            {report.decision && (
              <div className="flex-1 rounded-md border border-border px-4 py-3 text-center">
                <p className="text-[10px] font-medium text-muted-foreground">
                  Recommendation
                </p>
                <Badge
                  variant="outline"
                  className={cn('mt-1 px-3 py-1 text-sm font-bold', decisionBadgeVariant(report.decision))}
                >
                  {report.decision.replace(/_/g, ' ')}
                </Badge>
              </div>
            )}
          </div>
        )}

        {isInvalid && (
          <p className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-destructive" />
            The figures above are shown for reference only and must not be used as a business result.
          </p>
        )}
      </CardContent>
    </Card>
  );
}


function ConfidenceBanner({ report }: { report: ExperimentReport }) {
  const isLow = report.confidence === 'LOW';
  const isHigh = report.confidence === 'HIGH';

  return (
    <div
      className={cn(
        'flex flex-col gap-3 rounded-lg border p-5 sm:flex-row sm:items-center sm:justify-between',
        isHigh && 'border-success/25 bg-success/[0.08]',
        !isHigh && !isLow && 'border-border bg-secondary',
        isLow && 'border-destructive/25 bg-destructive/[0.08]'
      )}
    >
      <div className="flex items-center gap-4">
        <div
          className={cn(
            'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg',
            isHigh && 'bg-success/[0.15] text-success',
            !isHigh && !isLow && 'bg-border-strong text-muted-foreground',
            isLow && 'bg-destructive/[0.15] text-destructive'
          )}
        >
          <ShieldCheck className="h-5 w-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold tracking-tight text-foreground">
              {/* BUG FIX: this used to say "Recommendation Confidence:"
                  while displaying `report.confidence` — the deterministic
                  data-quality/statistical confidence in the RESULTS
                  (matches the PDF export's "Confidence in Results"
                  section). `report.recommendationConfidence` is a
                  SEPARATE field (the decision-level confidence in the
                  GO/GO_WITH_CAUTION/NO_GO call itself, e.g. downgraded to
                  MEDIUM by a guardrail warning even when result confidence
                  is HIGH) — see DecisionStrip below, which now renders it.
                  Mislabeling this one caused the PDF and the site to show
                  two different-looking "confidence" numbers for the same
                  report. */}
              Confidence in Results:{' '}
              <span
                className={cn(
                  isHigh && 'text-success',
                  !isHigh && !isLow && 'text-foreground',
                  isLow && 'text-destructive'
                )}
              >
                {report.confidence}
              </span>
            </h2>
            <span
              className={cn(
                isHigh && 'text-success',
                !isHigh && !isLow && 'text-muted-foreground',
                isLow && 'text-destructive'
              )}
            >
              {Array.from({ length: 5 }).map((_, i) => (
                <Star
                  key={i}
                  className={cn(
                    'inline h-3.5 w-3.5',
                    i < report.confidenceStars
                      ? 'fill-current'
                      : 'fill-transparent opacity-25'
                  )}
                />
              ))}
            </span>
          </div>
          <p className="mt-0.5 max-w-2xl text-[13px] text-muted-foreground">
            {report.confidenceReason}
          </p>
        </div>
      </div>
      {report.srmWarning && (
        <Badge
          variant="outline"
          className="w-fit shrink-0 gap-1.5 border-destructive/25 bg-destructive/[0.08] text-destructive"
        >
          <AlertTriangle className="h-3.5 w-3.5" />
          SRM Warning
        </Badge>
      )}
    </div>
  );
}

/**
 * Karolina's feedback (item 10): plain-language descriptions for
 * technical quality-check labels, matched loosely by substring since
 * the backend's exact label wording (e.g. "Sample Ratio Mismatch (SRM)")
 * can vary slightly across checks.
 */
const QUALITY_LABEL_TOOLTIPS: { match: string; text: string }[] = [
  {
    match: 'srm',
    text: 'Sample Ratio Mismatch — checks whether users were allocated to experiment groups according to the expected allocation ratio. A mismatch can indicate a bug in the randomization and may invalidate the results.',
  },
  {
    match: 'sample ratio',
    text: 'Sample Ratio Mismatch — checks whether users were allocated to experiment groups according to the expected allocation ratio. A mismatch can indicate a bug in the randomization and may invalidate the results.',
  },
  {
    match: 'power',
    text: 'The probability of detecting an effect of the specified size if that effect is truly present. High power reduces the chance of missing a real effect, but does not by itself confirm that the treatment works.',
  },
];

function QualityRow({
  label,
  passed,
  detail,
}: {
  label: string;
  passed: boolean;
  detail: string;
}) {
  const tooltip = QUALITY_LABEL_TOOLTIPS.find((t) => label.toLowerCase().includes(t.match))?.text;
  return (
    <div className="flex items-start gap-3 rounded-md border border-border px-3 py-2.5">
      {passed ? (
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />
      ) : (
        <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
      )}
      <div className="min-w-0">
        <p className="flex items-center gap-1 text-[13px] font-medium text-foreground">
          {label}
          {tooltip && <InfoTooltip text={tooltip} />}
        </p>
        <p className="text-xs text-muted-foreground">{detail}</p>
      </div>
    </div>
  );
}

function StatRow({
  stat,
}: {
  stat: ExperimentReport['stats'][number];
}) {
  return (
    <div className="grid grid-cols-12 items-center gap-2 rounded-md border border-border px-3 py-2.5 text-[13px]">
      <div className="col-span-12 sm:col-span-3">
        <p className="flex items-center gap-1 font-medium text-foreground">
          {stat.metric}
          {stat.isOmnibus && (
            <InfoTooltip text="Omnibus test — tests whether there is evidence of a difference among the experiment groups overall. It does not provide one single effect size or confidence interval, which is why those are shown as N/A here." />
          )}
        </p>
      </div>
      <div className="col-span-4 sm:col-span-2">
        <span className="text-[11px] font-medium text-muted-foreground">
          Control
        </span>
        <p className="font-data font-medium text-foreground">{stat.control}</p>
      </div>
      <div className="col-span-4 sm:col-span-2">
        <span className="text-[11px] font-medium text-muted-foreground">
          Variant
        </span>
        <p className="font-data font-medium text-foreground">{stat.variant}</p>
      </div>
      <div className="col-span-4 sm:col-span-2">
        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground">
          Delta
          <InfoTooltip text="The absolute difference between the treatment and control results." />
        </span>
        <p
          className={cn(
            'font-data font-semibold',
            stat.significant ? 'text-success' : 'text-muted-foreground'
          )}
        >
          {stat.delta}
        </p>
      </div>
      <div className="col-span-6 sm:col-span-1">
        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground">
          p-value
          <InfoTooltip text="The probability of observing a result at least this extreme if there were no true difference between the groups. It is not the probability that the treatment has no effect." />
        </span>
        <p
          className={cn(
            'w-fit rounded px-1.5 py-0.5 font-data text-[12px] font-semibold',
            stat.pValue < 0.05
              ? 'bg-success/[0.12] text-success'
              : 'bg-secondary text-muted-foreground'
          )}
        >
          {stat.pValue < 0.001 ? '<0.001' : stat.pValue.toFixed(3)}
        </p>
      </div>
      <div className="col-span-6 sm:col-span-2">
        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground">
          95% CI
          <InfoTooltip text="A range of plausible values for the true effect, estimated from the experiment data. It reflects uncertainty in the estimate, not a probability that the true effect falls in this specific range." />
        </span>
        <p className="font-data text-[12px] text-foreground">
          [{stat.ciLower}, {stat.ciUpper}]
        </p>
      </div>
    </div>
  );
}

function DecisionMetricStat({
  label,
  value,
  emphasis,
  tooltip,
}: {
  label: string;
  value: string;
  emphasis?: 'neutral' | 'positive' | 'negative';
  tooltip?: string;
}) {
  return (
    <div className="rounded-md border border-border bg-secondary px-3 py-2.5">
      <p className="flex items-center gap-1 text-[10px] font-medium text-muted-foreground">
        {label}
        {tooltip && <InfoTooltip text={tooltip} />}
      </p>
      <p
        className={cn(
          'mt-0.5 text-[13px] font-semibold',
          emphasis === 'positive' && 'text-success',
          emphasis === 'negative' && 'text-destructive',
          (!emphasis || emphasis === 'neutral') && 'text-foreground'
        )}
      >
        {value}
      </p>
    </div>
  );
}

/**
 * Explicit Hypothesis Evaluation — Karolina/Phase-2 ask: the report
 * must say SUPPORTED/REJECTED out loud, and that verdict must be
 * visually and structurally separate from the business Decision
 * strip below it. Nothing here is calculated in the browser — every
 * field comes straight from `report.hypothesisEvaluation`
 * (app/stats/hypothesis_evaluator.py), and the verdict badge never
 * reuses the Decision vocabulary (GO / GO WITH CAUTION / NO-GO).
 */
function HypothesisEvaluationSection({ report }: { report: ExperimentReport }) {
  const hypothesis = report.hypothesis;
  if (!hypothesis) return null;

  const evaluation = report.hypothesisEvaluation;
  const verdict = evaluation?.verdict ?? null;
  const unavailable = !evaluation || verdict == null;
  const matchedStat = findHypothesisStatResult(hypothesis, report.stats);

  return (
    <Card className="border-border shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">Hypothesis Evaluation</CardTitle>
          <Badge
            variant="outline"
            className={cn('ml-auto px-3 py-1 text-[13px] font-bold', verdictBadgeClass(verdict))}
          >
            {hypothesisResultLabel(verdict)}
          </Badge>
        </div>
        <CardDescription>
          Whether the stated hypothesis was supported by the experiment&apos;s statistics —
          a separate question from the business recommendation below.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-md border border-border bg-secondary px-3 py-2.5">
          <p className="text-[10px] font-medium text-muted-foreground">Hypothesis</p>
          <p className="mt-0.5 text-[13px] text-foreground">{hypothesis.statement}</p>
        </div>

        {unavailable ? (
          <p className="text-[13px] text-muted-foreground">
            {evaluation?.evaluationNote ??
              'This hypothesis could not be evaluated against the computed statistics.'}
          </p>
        ) : (
          <>
            <div>
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                Why
              </p>
              <ul className="space-y-1.5 text-[13px] text-foreground">
                <li>
                  <span className="font-medium text-foreground">Observed effect:</span>{' '}
                  {formatPercent(evaluation.observedEffectRelative)} relative
                </li>
                {evaluation.expectedEffectRelative != null && (
                  <li>
                    <span className="font-medium text-foreground">Expected effect:</span>{' '}
                    {formatPercent(evaluation.expectedEffectRelative)} relative
                  </li>
                )}
                <li>
                  <span className="font-medium text-foreground">Statistical significance:</span>{' '}
                  {evaluation.statisticallySignificant ? 'Yes' : 'No'}
                  {matchedStat != null &&
                    `, p ${matchedStat.pValue < 0.001 ? '< 0.001' : `= ${matchedStat.pValue.toFixed(3)}`}`}
                </li>
                {matchedStat != null && (
                  <li>
                    <span className="font-medium text-foreground">95% CI:</span> [{matchedStat.ciLower}
                    , {matchedStat.ciUpper}]
                  </li>
                )}
                {report.experimentValidity && (
                  <li>
                    <span className="font-medium text-foreground">Experiment validity:</span>{' '}
                    {report.experimentValidity}
                  </li>
                )}
              </ul>
            </div>

            <p className="text-[13px] leading-relaxed text-foreground">
              {verdict === 'SUPPORTED' &&
                'The experiment provides sufficient statistical evidence to support the hypothesis' +
                  (evaluation.expectedEffectRelative != null
                    ? ', and the observed effect meets or exceeds the pre-specified expected-effect threshold.'
                    : '.')}
              {verdict === 'PARTIALLY_SUPPORTED' &&
                'The effect was statistically significant and in the expected direction, but fell short of the pre-specified expected-effect threshold.'}
              {verdict === 'NOT_SUPPORTED' &&
                'The stated hypothesis was not supported by this experiment. This does not by itself mean the treatment was harmful — see the statistics above for the actual direction and significance.'}
            </p>

            {/* Underpowered caveat — only when the hypothesis was REJECTED
                (not significant) AND the same deterministic power check the
                Decision Audit Trail already computes (app/stats/decision_
                audit.py's _power_evidence) flagged this run as underpowered.
                This never changes the verdict itself (still REJECTED — a
                verdict is never fabricated/softened client-side); it only
                surfaces the already-computed caveat that a non-significant
                result here may reflect insufficient power rather than a
                genuinely null effect, right next to the badge that would
                otherwise read as a flat disproof. */}
            {verdict === 'NOT_SUPPORTED' &&
              report.decisionAudit?.powerEvidence &&
              report.decisionAudit.powerEvidence.status === 'warning' && (
                <div className="flex items-start gap-2.5 rounded-md border border-warning/25 bg-warning/[0.08] px-3 py-2.5 text-[13px] text-warning">
                  <HelpCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <p>
                    This run was underpowered ({report.decisionAudit.powerEvidence.value}
                    {report.decisionAudit.powerEvidence.detail
                      ? ` — ${report.decisionAudit.powerEvidence.detail}`
                      : ''}
                    ). A non-significant result here may mean the sample was too small to
                    detect a real effect of this size, not that no effect exists — REJECTED
                    reflects &quot;not supported by this experiment&quot;, not &quot;disproven&quot;.
                  </p>
                </div>
              )}
          </>
        )}

        {/* Explicit separation: hypothesis verdict, experiment validity,
            guardrails, and the final business recommendation are four
            distinct signals — never collapsed into one field. */}
        <Separator className="bg-border" />
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <DecisionMetricStat label="Hypothesis" value={hypothesisResultLabel(verdict)} />
          <DecisionMetricStat label="Experiment Validity" value={report.experimentValidity ?? 'N/A'} />
          <DecisionMetricStat
            label="Guardrails"
            value={report.guardrailStatus ? report.guardrailStatus.replace(/_/g, ' ') : 'N/A'}
          />
          <DecisionMetricStat
            label="Final Recommendation"
            value={report.decision ? report.decision.replace(/_/g, ' ') : 'N/A'}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function DecisionSupportSection({
  decisionSupport,
  guardrailRequestState,
  guardrailResolutions,
}: {
  decisionSupport: DecisionSupport;
  guardrailRequestState?: ExperimentReport['guardrailRequestState'];
  guardrailResolutions?: ExperimentReport['guardrailResolutions'];
}) {
  const ds = decisionSupport;

  return (
    <Card className="border-border shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Target className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">Decision Support</CardTitle>
        </div>
        <CardDescription>
          What this result means for the business, derived deterministically from the
          experiment's statistics — never calculated in the browser.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {!ds.available && ds.warnings.length > 0 && (
          <div className="flex items-start gap-2.5 rounded-md border border-warning/25 bg-warning/[0.08] px-3 py-2.5 text-[13px] text-warning">
            <HelpCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="space-y-1">
              {ds.warnings.map((w, i) => (
                <p key={i}>{w}</p>
              ))}
            </div>
          </div>
        )}

        {ds.available && (
          <>
            {/* Expected / Observed / Achievement / Significance — kept visually distinct */}
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              <DecisionMetricStat
                label="Expected Effect"
                value={formatPercent(ds.expectedEffectRelative)}
              />
              <DecisionMetricStat
                label="Observed Effect"
                value={formatPercent(ds.observedEffectRelative)}
                emphasis={
                  ds.observedEffectRelative == null
                    ? undefined
                    : ds.observedEffectRelative >= 0
                    ? 'positive'
                    : 'negative'
                }
              />
              <DecisionMetricStat
                label="Achievement"
                value={ds.effectAchievementRatio != null ? formatPercent(ds.effectAchievementRatio, 0) : 'N/A'}
              />
              <DecisionMetricStat
                label="Statistically Significant"
                tooltip="Whether the observed effect is unlikely to be due to chance alone. This is a statistical judgment only — it does not by itself mean the effect is large enough to matter for the business."
                value={
                  ds.statisticalSignificance == null
                    ? 'N/A'
                    : ds.statisticalSignificance
                    ? 'Yes'
                    : 'No'
                }
                emphasis={
                  ds.statisticalSignificance == null
                    ? undefined
                    : ds.statisticalSignificance
                    ? 'positive'
                    : 'neutral'
                }
              />
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              <DecisionMetricStat
                label="Baseline"
                value={ds.baselineValue != null ? formatNumber(ds.baselineValue) : 'N/A'}
              />
              <DecisionMetricStat
                label="Expected Value"
                value={ds.expectedValue != null ? formatNumber(ds.expectedValue) : 'N/A'}
              />
            </div>

            {ds.businessInterpretation && (
              <p className="text-[13px] leading-relaxed text-foreground">
                {ds.businessInterpretation}
              </p>
            )}

            {/* Business impact — backend-supplied value only, never computed here */}
            <div className="rounded-md border border-border px-3 py-2.5">
              <p className="text-[10px] font-medium text-muted-foreground">
                Business Impact
              </p>
              {ds.impactCalculationMethod === 'population_scaled' && ds.incrementalCount != null ? (
                <p className="mt-0.5 text-[13px] font-semibold text-foreground">
                  {ds.incrementalCount >= 0 ? '+' : ''}
                  {ds.incrementalCount.toLocaleString(undefined, { maximumFractionDigits: 0 })}{' '}
                  <span className="font-normal text-muted-foreground">
                    (baseline {ds.baselineExpectedCount?.toLocaleString(undefined, { maximumFractionDigits: 0 })}{' '}
                    → observed {ds.observedCount?.toLocaleString(undefined, { maximumFractionDigits: 0 })})
                  </span>
                </p>
              ) : (
                <p className="mt-0.5 text-[13px] text-muted-foreground">Not available</p>
              )}
              {ds.warnings.length > 0 && ds.available && (
                <div className="mt-2 space-y-1">
                  {ds.warnings.map((w, i) => (
                    <p key={i} className="text-[11px] text-muted-foreground">
                      {w}
                    </p>
                  ))}
                </div>
              )}
            </div>
          </>
        )}

        {/* Additional metrics */}
        {ds.additionalMetrics.length > 0 && (
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Additional Metrics
            </p>
            <div className="space-y-2">
              {ds.additionalMetrics.map((m) => (
                <div
                  key={m.metric}
                  className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2.5 text-[13px]"
                >
                  <span className="font-medium text-foreground">{m.metric}</span>
                  <div className="flex items-center gap-3">
                    <span
                      className={cn(
                        'font-semibold',
                        m.direction === 'increase' && 'text-success',
                        m.direction === 'decrease' && 'text-destructive',
                        m.direction === 'no_change' && 'text-muted-foreground'
                      )}
                    >
                      {formatPercent(m.relativeChange)}
                    </span>
                    <Badge
                      variant="outline"
                      className={cn(
                        'text-[10px]',
                        m.statisticallySignificant
                          ? 'border-border text-foreground'
                          : 'border-border text-muted-foreground'
                      )}
                    >
                      {m.statisticallySignificant ? 'Significant' : 'Not significant'}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Guardrails "requested but not found" note — the PASS/VIOLATED
            list itself now lives in OverviewHero above, so this only
            still renders the not-found/not-specified case to avoid
            showing the same list twice. */}
        {ds.guardrailFindings.length === 0 && (
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Guardrails
            </p>
            {guardrailRequestState === 'REQUESTED_NOT_FOUND' ||
            guardrailRequestState === 'PARTIALLY_AVAILABLE' ? (
              <div className="space-y-1">
                <p className="text-[13px] text-muted-foreground">Requested — not found</p>
                {(guardrailResolutions ?? [])
                  .filter((r) => !r.resolved)
                  .map((r) => (
                    <p key={r.requestedName} className="text-[11px] text-muted-foreground">
                      {r.requestedName} — no matching metric in this dataset
                    </p>
                  ))}
              </div>
            ) : (
              <p className="text-[13px] text-muted-foreground">Not specified</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function decisionBadgeClass(decision: string | null | undefined) {
  if (decision === 'GO') return 'border-success/25 bg-success/[0.08] text-success';
  if (decision === 'GO_WITH_CAUTION') return 'border-warning/25 bg-warning/[0.08] text-warning';
  if (decision === 'NO_GO' || decision === 'INVALID') return 'border-destructive/25 bg-destructive/[0.08] text-destructive';
  return 'border-border bg-secondary text-muted-foreground'; // INCONCLUSIVE / unknown
}

function guardrailBadgeClass(status: string | null | undefined) {
  if (status === 'PASS') return 'border-success/25 bg-success/[0.08] text-success';
  if (status === 'WARNING') return 'border-warning/25 bg-warning/[0.08] text-warning';
  if (status === 'FAIL') return 'border-destructive/25 bg-destructive/[0.08] text-destructive';
  return 'border-border bg-secondary text-muted-foreground'; // NOT_AVAILABLE
}

/**
 * Guardrail root-cause fix — the Decision Strip badge must show the
 * REQUEST/availability state (NOT_SPECIFIED / REQUESTED_NOT_FOUND /
 * PARTIALLY_AVAILABLE), never just PASS/WARNING/FAIL/NOT_AVAILABLE,
 * which collapses "never asked" and "asked but not found" into one
 * indistinguishable label — exactly the original bug.
 * `guardrailRequestState` is optional (older/edge-case report shapes) —
 * falls back to the plain evaluation status when absent.
 */
function guardrailBadgeLabel(report: ExperimentReport): string | null {
  const state = report.guardrailRequestState;
  if (state === 'NOT_SPECIFIED' || !state) {
    return report.guardrailStatus ? 'Not specified' : null;
  }
  if (state === 'REQUESTED_NOT_FOUND') return 'Requested — not found';
  // AVAILABLE / PARTIALLY_AVAILABLE
  if (report.guardrailStatus && report.guardrailStatus !== 'NOT_AVAILABLE') {
    return report.guardrailStatus.replace(/_/g, ' ');
  }
  if (state === 'PARTIALLY_AVAILABLE') return 'Partially available';
  // BUG FIX: reaching here means the guardrail(s) resolved to a real
  // dataset column (AVAILABLE) but guardrailStatus is still
  // NOT_AVAILABLE — i.e. evaluation itself never ran (e.g. multi-arm —
  // see guardrail_node.py). Previously this silently fell through to a
  // hardcoded 'Evaluated', directly contradicting the decision text
  // ("no guardrail metrics were evaluated") right next to it.
  return 'Available — not evaluated';
}

function guardrailStripBadgeClass(report: ExperimentReport): string {
  const state = report.guardrailRequestState;
  if (state === 'REQUESTED_NOT_FOUND') return 'border-warning/25 bg-warning/[0.08] text-warning';
  if (state === 'PARTIALLY_AVAILABLE') return 'border-warning/25 bg-warning/[0.08] text-warning';
  if (
    state === 'AVAILABLE' &&
    (!report.guardrailStatus || report.guardrailStatus === 'NOT_AVAILABLE')
  ) {
    return 'border-warning/25 bg-warning/[0.08] text-warning'; // resolved but not evaluated
  }
  return guardrailBadgeClass(report.guardrailStatus);
}

/**
 * Product improvement — compact, decision-adaptive explanation
 * (Why this decision / What prevents a full GO / What to monitor /
 * Recommended next step). Renders only backend-computed
 * `DecisionNarrative` facts — no text is authored in this component.
 */
function DecisionNarrativeSection({ narrative }: { narrative: DecisionNarrative }) {
  const { monitoring } = narrative;
  const hasMonitoringContent =
    monitoring.primaryMetric ||
    monitoring.guardrailsEvaluated.length > 0 ||
    monitoring.potentialMonitoringMetrics.length > 0;

  return (
    <Card className="border-border shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Lightbulb className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">Decision Narrative</CardTitle>
        </div>
        <CardDescription>What this decision means for the product team</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2.5">
      {narrative.whyThisDecision.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            Why this decision
          </p>
          <ul className="mt-1 space-y-1">
            {narrative.whyThisDecision.map((line, i) => (
              <li key={i} className="flex items-start gap-2 text-[12px] text-muted-foreground">
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground/50" />
                {line}
              </li>
            ))}
          </ul>
        </div>
      )}

      {narrative.whatPreventsFullGo.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            What prevents a full GO
          </p>
          <ul className="mt-1 space-y-1">
            {narrative.whatPreventsFullGo.map((line, i) => (
              <li key={i} className="flex items-start gap-2 text-[12px] text-muted-foreground">
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-warning" />
                {line}
              </li>
            ))}
          </ul>
        </div>
      )}

      {narrative.whatWouldChangeDecision.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            What would change the decision
          </p>
          <ul className="mt-1 space-y-1">
            {narrative.whatWouldChangeDecision.map((line, i) => (
              <li key={i} className="flex items-start gap-2 text-[12px] text-muted-foreground">
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground/50" />
                {line}
              </li>
            ))}
          </ul>
        </div>
      )}

      {hasMonitoringContent && (
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            What to monitor
          </p>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {monitoring.primaryMetric && (
              <Badge variant="outline" className="border-border text-[10px] text-foreground">
                {monitoring.primaryMetric} (primary)
              </Badge>
            )}
            {monitoring.guardrailsEvaluated.map((m) => (
              <Badge key={m} variant="outline" className="border-copilot/25 bg-copilot-bg text-[10px] text-copilot">
                {m} (guardrail)
              </Badge>
            ))}
            {monitoring.potentialMonitoringMetrics.map((m) => (
              <Badge key={m} variant="outline" className="border-border text-[10px] text-muted-foreground">
                {m}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {narrative.recommendedNextStep && (
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            Recommended next step
          </p>
          <p className="mt-1 text-[12px] text-foreground">{narrative.recommendedNextStep}</p>
        </div>
      )}
      </CardContent>
    </Card>
  );
}

/**
 * Evidence & Sources (Agentic RAG, Stage 9/10) — renders the
 * knowledge-base excerpts the backend actually retrieved
 * (`report.knowledgeBaseReferences`). Purely a display of already-
 * retrieved evidence: no source is invented here, and nothing in this
 * component can influence `report.decision` — it only explains the
 * context the decision was made alongside.
 *
 * Renders nothing when the knowledge base was never queried for this
 * request (`attempted` false — e.g. datasets predating this field, or
 * a request that never routed to the knowledge_base node), so a
 * normal run without KB involvement is unaffected.
 *
 * Distinguishes three states — never collapses "FAILED" into "no
 * evidence found" (see knowledge_base_retrieval_error's docstring in
 * app/schemas/report.py, which is where this data actually comes
 * from — this component only decides how to render it):
 *   1. FAILED       — `retrievalError` is set: the retriever itself
 *                      raised (index missing, I/O error, etc.). Shown
 *                      first, distinctly, since this is an
 *                      infrastructure problem, not a legitimate
 *                      "nothing relevant" result.
 *   2. NO EVIDENCE  — attempted, no error, but `references` is empty:
 *                      retrieval ran fine and genuinely found nothing
 *                      above the relevance threshold.
 *   3. EVIDENCE     — attempted, references present: the normal case.
 */
export function EvidenceSection({
  references,
  attempted,
  retrievalError,
  blockingIssue,
}: {
  references: KnowledgeBaseReference[];
  attempted?: boolean;
  retrievalError?: string | null;
  blockingIssue?: string | null;
}) {
  // Collapsed by default, same pattern as DecisionAuditTrailSection below —
  // this is supporting/citation detail, not the decision itself, so it
  // shouldn't dominate the page on first render. Nothing is removed, only
  // hidden behind the same click-to-expand toggle.
  const [expanded, setExpanded] = useState(false);

  if (!attempted) return null;

  if (retrievalError) {
    return (
      <Card className="border-border shadow-none">
        <CardHeader
          className="cursor-pointer select-none pb-3"
          onClick={() => setExpanded((v) => !v)}
        >
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-foreground" />
            <CardTitle className="text-[15px] tracking-tight">Evidence &amp; Sources</CardTitle>
            <Badge variant="outline" className="ml-auto border-destructive/25 text-[10px] font-semibold text-destructive">
              Retrieval failed
            </Badge>
            <ChevronDown
              className={cn('h-4 w-4 shrink-0 text-muted-foreground transition-transform', expanded && 'rotate-180')}
            />
          </div>
          <CardDescription>{expanded ? 'click to collapse' : 'click to expand'}</CardDescription>
        </CardHeader>
        {expanded && (
          <CardContent className="space-y-1">
            <p className="text-[12px] font-medium text-destructive">Knowledge base retrieval failed.</p>
            <p className="text-[12px] text-muted-foreground">
              The knowledge base could not be queried, so no methodology evidence was used for this
              decision. The decision above was based only on deterministic validation rules.
            </p>
            <p className="mt-1 text-[10px] text-muted-foreground">Retrieval error: {retrievalError}</p>
          </CardContent>
        )}
      </Card>
    );
  }

  if (!references || references.length === 0) {
    return (
      <Card className="border-border shadow-none">
        <CardHeader
          className="cursor-pointer select-none pb-3"
          onClick={() => setExpanded((v) => !v)}
        >
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-foreground" />
            <CardTitle className="text-[15px] tracking-tight">Evidence &amp; Sources</CardTitle>
            <ChevronDown
              className={cn('ml-auto h-4 w-4 shrink-0 text-muted-foreground transition-transform', expanded && 'rotate-180')}
            />
          </div>
          <CardDescription>{expanded ? 'click to collapse' : 'click to expand'}</CardDescription>
        </CardHeader>
        {expanded && (
          <CardContent>
            <p className="text-[12px] text-muted-foreground">
              {blockingIssue
                ? `No sufficiently relevant evidence found for: ${blockingIssue}.`
                : "No sufficiently relevant evidence found."}
            </p>
          </CardContent>
        )}
      </Card>
    );
  }

  return (
    <Card className="border-border shadow-none">
      <CardHeader
        className="cursor-pointer select-none pb-3"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">
            Evidence &amp; Sources <span className="font-data font-normal text-muted-foreground">· {references.length} items</span>
          </CardTitle>
          <ChevronDown
            className={cn('ml-auto h-4 w-4 shrink-0 text-muted-foreground transition-transform', expanded && 'rotate-180')}
          />
        </div>
        <CardDescription>
          Experimentation guidance retrieved from the knowledge base, relevant to this decision —{' '}
          {expanded ? 'click to collapse' : 'click to expand'}
        </CardDescription>
      </CardHeader>
      {expanded && (
      <CardContent className="space-y-2">
        {references.map((ref, i) => (
          <div key={`${ref.source}-${ref.heading}-${i}`} className="rounded-md border border-border bg-secondary px-3 py-2">
            <div className="flex flex-wrap items-center justify-between gap-1.5">
              <p className="text-[12px] font-medium text-foreground">{ref.heading}</p>
              <Badge variant="outline" className="border-border text-[10px] text-muted-foreground">
                {ref.source}
              </Badge>
            </div>
            <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">{ref.excerpt}</p>
            <p className="mt-1 text-[10px] text-muted-foreground">Relevance score: {ref.relevanceScore.toFixed(2)}</p>
          </div>
        ))}
      </CardContent>
      )}
    </Card>
  );
}

/**
 * Compact strip surfacing the backend's canonical decision fields
 * (Decision / ExperimentValidity / GuardrailStatus). These already
 * arrived over the wire but had no home in the UI before this
 * integration — never computed here, only rendered.
 */
function DecisionStrip({ report }: { report: ExperimentReport }) {
  if (!report.decision) return null;
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-4 py-3">
      <span className="text-[10px] font-medium text-muted-foreground">
        Decision
      </span>
      <Badge variant="outline" className={cn('text-[11px] font-semibold', decisionBadgeClass(report.decision))}>
        {report.decision.replace(/_/g, ' ')}
      </Badge>
      {guardrailBadgeLabel(report) && (
        <Badge variant="outline" className={cn('text-[10px]', guardrailStripBadgeClass(report))}>
          Guardrails: {guardrailBadgeLabel(report)}
        </Badge>
      )}
      {report.experimentValidity && (
        <Badge variant="outline" className="border-border text-[10px] text-muted-foreground">
          Validity: {report.experimentValidity}
        </Badge>
      )}
      {/* BUG FIX: previously never rendered on the site at all — only
          in the PDF export (lib/pdf.ts) — so the two outputs silently
          disagreed about which "confidence" number belonged to the
          decision itself. See ConfidenceBanner above for the other
          half of this fix. */}
      {report.recommendationConfidence && (
        <Badge variant="outline" className="border-border text-[10px] text-muted-foreground">
          Recommendation Confidence: {report.recommendationConfidence}
        </Badge>
      )}
      {report.decisionReason && (
        <p className="mt-1 basis-full text-[12px] text-muted-foreground">{report.decisionReason}</p>
      )}
      {/* Canonical list of what was actually requested/resolved — only
          when there's something to say beyond the badge (guardrail
          root-cause fix, doc3 §8: one canonical place, not repeated
          across Executive Summary / Decision Narrative / Next Steps). */}
      {report.guardrailResolutions && report.guardrailResolutions.length > 0 && (
        <div className="mt-1 basis-full space-y-1">
          {report.guardrailResolutions.map((r) => (
            <p key={r.requestedName} className="text-[11px] text-muted-foreground">
              <span className="font-medium text-foreground">{r.requestedName}</span>
              {r.resolved ? ' — resolved' : ' — not found in this dataset'}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

function auditStatusBadgeClass(status: AuditFact['status']) {
  if (status === 'pass') return 'border-success/25 bg-success/[0.08] text-success';
  if (status === 'warning') return 'border-warning/25 bg-warning/[0.08] text-warning';
  if (status === 'fail') return 'border-destructive/25 bg-destructive/[0.08] text-destructive';
  if (status === 'not_available') return 'border-border bg-secondary text-muted-foreground';
  return 'border-border bg-secondary text-muted-foreground'; // info
}

function auditStatusIcon(status: AuditFact['status']) {
  if (status === 'pass') return <CheckCircle2 className="h-3.5 w-3.5 text-success" />;
  if (status === 'fail') return <XCircle className="h-3.5 w-3.5 text-destructive" />;
  if (status === 'warning') return <AlertTriangle className="h-3.5 w-3.5 text-warning" />;
  return <HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />;
}

function AuditFactRow({ fact }: { fact: AuditFact }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-border px-3 py-2">
      <span className="mt-0.5 shrink-0">{auditStatusIcon(fact.status)}</span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[12px] font-medium text-foreground">{fact.label}</span>
          <Badge variant="outline" className={cn('text-[10px]', auditStatusBadgeClass(fact.status))}>
            {fact.value}
          </Badge>
        </div>
        {fact.detail && <p className="mt-0.5 text-[11px] text-muted-foreground">{fact.detail}</p>}
      </div>
    </div>
  );
}

/**
 * Phase 7 — Decision Audit Trail. Renders only backend-computed
 * `DecisionAuditTrail` facts (app/schemas/decision_audit.py); no
 * statistics or wording are generated in the frontend, and
 * `decisionAudit.decision` is never re-derived here — it always
 * mirrors `report.decision`. Collapsible (item 5): all the same
 * information stays in the DOM/available, just hidden behind a toggle
 * so it doesn't dominate the page — nothing is removed, only its
 * default visual weight is reduced.
 */
function DecisionAuditTrailSection({ audit }: { audit: DecisionAuditTrail }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <Card className="border-border shadow-none">
      <CardHeader
        className="cursor-pointer select-none pb-3"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <ListChecks className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">Decision Audit Trail</CardTitle>
          <Badge variant="outline" className={cn('ml-auto text-[10px] font-semibold', decisionBadgeClass(audit.decision))}>
            {audit.headline}
          </Badge>
          <ChevronDown
            className={cn('h-4 w-4 shrink-0 text-muted-foreground transition-transform', expanded && 'rotate-180')}
          />
        </div>
        <CardDescription>
          Why the system reached this exact decision — {expanded ? 'click to collapse' : 'click to expand'}
        </CardDescription>
      </CardHeader>
      {expanded && (
        <CardContent className="space-y-4">
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            Why this decision
          </p>
          <ul className="space-y-1.5">
            {audit.rationale.map((r, i) => (
              <li key={i} className="flex items-start gap-2.5 text-[13px]">
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-primary" />
                <span className="text-foreground">{r}</span>
              </li>
            ))}
          </ul>
        </div>

        {audit.supportingFacts.length > 0 && (
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Evidence supporting decision
            </p>
            <div className="space-y-1.5">
              {audit.supportingFacts.map((f, i) => (
                <AuditFactRow key={`${f.category}-${f.label}-${i}`} fact={f} />
              ))}
            </div>
          </div>
        )}

        {audit.warnings.length > 0 && (
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Warnings / limitations
            </p>
            <div className="space-y-1.5">
              {audit.warnings.map((f, i) => (
                <AuditFactRow key={`${f.category}-${f.label}-${i}`} fact={f} />
              ))}
            </div>
          </div>
        )}

        <Separator className="bg-border" />
        <div>
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            Decision impact
          </p>
          <p className="text-[13px] leading-relaxed text-foreground">{audit.decisionImpact}</p>
        </div>
        </CardContent>
      )}
    </Card>
  );
}

/**
 * TRUE Stratified Analysis. Deliberately its OWN section, never merged
 * into or labeled as "Segment Analysis" above — see StratificationResult's
 * docstring in lib/types.ts. Renders only backend-computed facts:
 *   - `status === 'not_run'`: the experiment failed the SAME validity
 *     gate the ordinary hypothesis test is subject to (SRM /
 *     conflicting variant assignment / critical quality failure), so
 *     stratified inference was never attempted — no numbers are shown,
 *     only the reason.
 *   - `status === 'ran'` and `eligibility.eligible === false`: the
 *     requested variable itself was rejected (e.g. perfectly
 *     associated with treatment assignment) — reason shown, no estimate.
 *   - `status === 'ran'` and eligible: the full combined estimate plus
 *     a per-stratum breakdown.
 */
function StratificationAnalysisSection({
  stratification,
}: {
  stratification: StratificationResult | null | undefined;
}) {
  if (!stratification) return null;

  const eligibility = stratification.eligibility;
  const estimate = stratification.estimate;

  return (
    <Card className="border-border shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Rows3 className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">Stratified Analysis</CardTitle>
        </div>
        <CardDescription>
          Variable: <span className="font-medium text-foreground">{stratification.stratificationColumn}</span>
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {stratification.status === 'not_run' && (
          <div className="rounded-md border border-border bg-secondary px-3 py-4">
            <Badge variant="outline" className="mb-2 text-[10px] text-muted-foreground">
              Status: NOT RUN
            </Badge>
            <p className="text-[13px] text-muted-foreground">
              {stratification.notRunReason ??
                'Stratified inference was not performed because the experiment is invalid.'}
            </p>
          </div>
        )}

        {stratification.status === 'ran' && eligibility && !eligibility.eligible && (
          <div className="rounded-md border border-border bg-secondary px-3 py-4">
            <Badge variant="outline" className="mb-2 text-[10px] text-muted-foreground">
              Not Eligible
            </Badge>
            <p className="text-[13px] text-muted-foreground">{eligibility.reason}</p>
          </div>
        )}

        {stratification.status === 'ran' && eligibility && eligibility.eligible && estimate && (
          <>
            <p className="text-[13px] text-muted-foreground">{eligibility.reason}</p>

            <div className="rounded-md border border-border p-3">
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                Combined Stratified Estimate
              </p>
              <div className="grid grid-cols-2 gap-2 text-[13px] sm:grid-cols-4">
                <div>
                  <p className="text-muted-foreground">Effect</p>
                  <p className="font-medium text-foreground">{estimate.effectEstimate.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Std. Error</p>
                  <p className="font-medium text-foreground">{estimate.standardError.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">95% CI</p>
                  <p className="font-medium text-foreground">
                    [{estimate.ciLower.toFixed(4)}, {estimate.ciUpper.toFixed(4)}]
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">p-value</p>
                  <p className="font-medium text-foreground">{estimate.pValue.toFixed(4)}</p>
                </div>
              </div>
              <p className="mt-2 text-[11px] text-muted-foreground">
                {estimate.method} — {estimate.strataUsed} stratum/strata used.
              </p>
            </div>

            <div>
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                Per-Stratum Breakdown
              </p>
              <div className="space-y-1.5">
                {eligibility.strata.map((s) => (
                  <div
                    key={s.stratumValue}
                    className="grid grid-cols-2 gap-2 text-[12px] sm:grid-cols-5 sm:items-center"
                  >
                    <span className="font-medium text-foreground">{s.stratumValue}</span>
                    <span className="text-muted-foreground">
                      control n={s.controlN}, variant n={s.variantN}
                    </span>
                    <span className="text-muted-foreground">
                      {s.controlOutcomeRate != null ? `control: ${s.controlOutcomeRate.toFixed(4)}` : '—'}
                    </span>
                    <span className="text-muted-foreground">
                      {s.variantOutcomeRate != null ? `variant: ${s.variantOutcomeRate.toFixed(4)}` : '—'}
                    </span>
                    <span className="text-muted-foreground">{!s.sufficient ? 'excluded (too few obs.)' : ''}</span>
                  </div>
                ))}
              </div>
            </div>

            {eligibility.sparseStratumValues.length > 0 && (
              <p className="text-[11px] text-muted-foreground">
                Excluded for insufficient observations: {eligibility.sparseStratumValues.join(', ')}
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Phase 8 — compact "what actually happened" summary. Deliberately a
 * handful of scalar facts, not an execution log: the full step-by-step
 * trace already lives in the Execution Pipeline stepper elsewhere in
 * the UI, so this section never duplicates it — it complements it
 * with the run-level facts (dataset, model, backend) that don't
 * belong to any single step.
 */
function RunInformationSection({ report }: { report: ExperimentReport }) {
  const rm = report.runMetadata;
  if (!rm) return null;

  const rows: { label: string; value: string }[] = [
    { label: 'Dataset', value: `${rm.datasetName} (${rm.datasetClassification})` },
    { label: 'Analysis Mode', value: rm.analysisMode },
    { label: 'Users / Variants', value: `${rm.userCount.toLocaleString()} users, ${rm.variantCount} variants` },
    // NOTE: `rm.selectedModel` is the model REQUESTED for this run
    // (Settings.model) — it is NOT necessarily what the Planner stage
    // used. With the default PLANNER_BACKEND=keyword, the Planner never
    // calls an LLM at all; `selectedModel` only reflects the model the
    // *report generator* attempted (see backend's `_build_run_metadata`).
    // Labeling this row "Planner" implied the Planner ran that model,
    // which was misleading whenever it differed from what actually
    // failed/succeeded during report generation (see the confidence
    // banner above, which reports the real report-generation outcome).
    { label: 'Requested LLM Model', value: rm.selectedModel ? rm.selectedModel : `none (${rm.plannerBackend} planner)` },
    { label: 'Report Backend', value: rm.reportBackend },
    { label: 'Run Status', value: rm.executionStatus },
  ];

  // Only present when an LLM call actually happened for this run and
  // returned usage data — see LLMUsage's docstring
  // (app/schemas/execution.py). Rendered as its own row rather than
  // folded into `rows` above because either half (tokens, cost) can be
  // missing independently, so the label/value text needs its own
  // conditional formatting.
  const usage = rm.llmUsage;
  const tokenParts: string[] = [];
  if (usage?.promptTokens != null && usage?.completionTokens != null) {
    tokenParts.push(`${usage.promptTokens.toLocaleString()} in / ${usage.completionTokens.toLocaleString()} out`);
  } else if (usage?.totalTokens != null) {
    tokenParts.push(`${usage.totalTokens.toLocaleString()} total`);
  }
  if (usage?.costUsd != null) {
    tokenParts.push(`$${usage.costUsd.toFixed(4)}`);
  }
  const usageValue = tokenParts.join(' · ');

  return (
    <Card className="border-border shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Info className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">Run Information</CardTitle>
          <Badge
            variant="outline"
            className={cn(
              'ml-auto text-[10px]',
              rm.executionStatus === 'FAILED' && 'border-destructive/25 text-destructive',
              rm.executionStatus === 'WARNING' && 'border-warning/25 text-warning',
              (rm.executionStatus === 'SUCCESS' || rm.executionStatus === 'SKIPPED') && 'border-border text-muted-foreground'
            )}
          >
            {rm.executionStatus}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-2 sm:grid-cols-2">
          {rows.map((row) => (
            <div key={row.label} className="rounded-md border border-border bg-secondary px-3 py-2">
              <p className="text-[10px] font-medium text-muted-foreground">{row.label}</p>
              <p className="mt-0.5 text-[13px] text-foreground">{row.value}</p>
            </div>
          ))}
          {usageValue && (
            <div className="rounded-md border border-border bg-secondary px-3 py-2">
              <p className="text-[10px] font-medium text-muted-foreground">LLM Usage</p>
              <p className="mt-0.5 text-[13px] text-foreground">{usageValue}</p>
            </div>
          )}
        </div>
        {report.reportFallbackReason && (
          <p className="mt-2.5 flex items-start gap-2 rounded-md border border-warning/25 bg-warning/[0.08] px-3 py-2 text-xs text-warning">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            {report.reportFallbackReason}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export function ReportCard({ report, datasetName, experimentId, prompt }: ReportCardProps) {
  return (
    <TooltipProvider delayDuration={150}>
    <div className="space-y-3 animate-fade-in">
      {/* 1. Experiment title / overview — now a full top bar (status,
          experiment ID, date, users, variants) with the download action
          moved in from the old standalone button above it. */}
      <ExperimentHeader
        report={report}
        datasetName={datasetName}
        experimentId={experimentId}
        prompt={prompt}
        onDownload={() => downloadReportPdf(report, { datasetName, experimentId, prompt })}
      />

      {/* 1.5. Compact "at a glance" strip — badges + Primary Metric /
          Variants / Data Quality cards. Sits above the more detailed
          HeroCard/KpiGrid below; same underlying data, just the fast
          30-second read requested for the top of the report. */}
      <SummaryCards report={report} />

      {/* 2, 3 & 3.5. Dashboard-style overview hero — verdict circle,
          effect/CI/p-value, four status tiles, guardrails and the
          recommendation panel, all in one glanceable block above every
          technical section (Karolina's #1 ask, decision-first).
          Same data as before — now composed from components/report/*
          instead of one inline OverviewHero function; nothing here is
          recalculated. */}
      <div className="space-y-3">
        <HeroCard report={report} />
        <KpiGrid report={report} />
        <div className="grid gap-3 lg:grid-cols-[1.4fr_1fr]">
          <GuardrailSection report={report} />
          <RecommendationCard report={report} />
        </div>
        <CopilotSummary
          summary={buildStateAwareSummary(report, classifySummaryState(report))}
          onAskCopilot={() =>
            document.getElementById('follow-up-chat')?.scrollIntoView({ behavior: 'smooth' })
          }
        />
      </div>

      {/* 4. Decision Analytics — confidence, deterministic Decision
          Support (business impact / additional metrics), and the
          human-readable decision narrative. Guardrails and the headline
          decision now live in OverviewHero above; nothing here is
          removed or recalculated, only grouped and moved up. */}
      <ConfidenceBanner report={report} />
      {report.decisionSupport && (
        <DecisionSupportSection
          decisionSupport={report.decisionSupport}
          guardrailRequestState={report.guardrailRequestState}
          guardrailResolutions={report.guardrailResolutions}
        />
      )}
      {report.decisionNarrative && <DecisionNarrativeSection narrative={report.decisionNarrative} />}

      {/* 5. Statistical Analysis */}
      {(() => {
        const primaryStat = report.stats.find((s) => s.metric === report.hypothesis?.primaryMetric) ?? report.stats[0];
        return primaryStat ? <PrimaryMetricChart stat={primaryStat} /> : null;
      })()}
      <Card className="border-border shadow-none">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <FileBarChart className="h-4 w-4 text-foreground" />
            <CardTitle className="text-[15px] tracking-tight">
              Statistical Results
            </CardTitle>
          </div>
          <CardDescription>
            {(() => {
              const testNames = Array.from(
                new Set(report.stats.map((s) => s.testName))
              );
              if (testNames.length === 0) {
                return 'Statistical test results with 95% confidence intervals';
              }
              return `${testNames.join(' / ')} results with 95% confidence intervals`;
            })()}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {report.stats.map((s) => (
            <StatRow key={s.metric} stat={s} />
          ))}

          {report.bootstrapCiLower != null && report.bootstrapCiUpper != null && (
            <>
              <Separator className="my-3 bg-border" />
              <div className="rounded-md border border-border bg-secondary px-3 py-2.5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-[10px] font-medium text-muted-foreground">
                      Bootstrap Cross-check
                    </p>
                    <p className="mt-0.5 text-[13px] text-foreground">
                      95% CI for the difference: [{report.bootstrapCiLower.toFixed(4)}, {report.bootstrapCiUpper.toFixed(4)}]
                    </p>
                  </div>
                  <Badge variant="outline" className="shrink-0 border-border text-[10px] text-muted-foreground">
                    {report.bootstrapIterations?.toLocaleString() ?? '10,000'} iterations
                  </Badge>
                </div>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Non-parametric cross-check; the primary hypothesis test and decision remain unchanged.
                </p>
              </div>
            </>
          )}

          <Separator className="my-3 bg-border" />
          <div className="grid gap-2 sm:grid-cols-2">
            <div className="rounded-md border border-border bg-secondary px-3 py-2.5">
              <span className="inline-flex items-center gap-1 text-[10px] font-medium text-muted-foreground">
                MDE
                <InfoTooltip text="Minimum Detectable Effect — the smallest effect the experiment is designed to reliably detect at the chosen significance level and power. It is a design target, not the effect actually observed." />
              </span>
              <p className="mt-0.5 text-[13px] text-foreground">{report.mde}</p>
            </div>
            <div className="rounded-md border border-border bg-secondary px-3 py-2.5">
              <p className="text-[10px] font-medium text-muted-foreground">
                Sample Size
              </p>
              <p className="mt-0.5 text-[13px] text-foreground">
                {report.sampleSizeNote}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 6. Data Quality & Validity */}
      <Card className="border-border shadow-none">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-foreground" />
            <CardTitle className="text-[15px] tracking-tight">
              Data Quality &amp; Assumptions
            </CardTitle>
            <Badge
              variant="outline"
              className={cn(
                'ml-auto text-[10px] font-semibold',
                report.qualityChecks.length > 0 && report.qualityChecks.every((c) => c.passed)
                  ? 'border-success/25 bg-success/[0.08] text-success'
                  : 'border-warning/25 bg-warning/[0.08] text-warning'
              )}
            >
              {report.qualityChecks.length === 0
                ? 'No checks run'
                : report.qualityChecks.every((c) => c.passed)
                ? 'All checks passed'
                : 'Has warnings'}
            </Badge>
          </div>
          <CardDescription>
            Automated checks on randomization, outliers, and missing data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 sm:grid-cols-2">
            {report.qualityChecks.map((c) => (
              <QualityRow key={c.label} {...c} />
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Assignment Quality (SRM) — visual split of Control/Treatment vs
          expected, parsed from the existing SRM QualityCheck row above;
          adds nothing the backend didn't already compute. */}
      <SrmPanel qualityChecks={report.qualityChecks} />

      {/* 7. Segmentation Analysis — exploratory, supporting evidence only;
          kept visually and hierarchically secondary to the primary
          decision above. */}
      <SegmentAnalysisSection segmentation={report.segmentation} />

      {/* TRUE stratified analysis — its own section, never merged into or
          labeled as Segment Analysis above. */}
      <StratificationAnalysisSection stratification={report.stratification} />

      {/* 9. Decision Audit Trail (Phase 7) — explanatory only;
          decisionAudit.decision always mirrors report.decision above and
          never overrides it. Collapsible — see DecisionAuditTrailSection. */}
      {report.decisionAudit && <DecisionAuditTrailSection audit={report.decisionAudit} />}

      {/* Evidence & Sources (Stage 9/10 Agentic RAG) — only renders when the
          backend actually retrieved knowledge-base references; see
          EvidenceSection's docstring. Never affects report.decision. */}
      <EvidenceSection
        references={report.knowledgeBaseReferences}
        attempted={report.knowledgeBaseAttempted}
        retrievalError={report.knowledgeBaseRetrievalError}
        blockingIssue={report.knowledgeBaseBlockingIssue}
      />

      {/* Supplementary: Strategic Recommendations */}
      <Card className="border-border shadow-none">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Lightbulb className="h-4 w-4 text-foreground" />
            <CardTitle className="text-[15px] tracking-tight">
              Strategic Recommendations &amp; Next Steps
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Recommendations
            </p>
            <ul className="space-y-1.5">
              {report.recommendations.map((r, i) => (
                <li key={i} className="flex items-start gap-2.5 text-[13px]">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-primary" />
                  <span className="text-foreground">{r}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Next Steps
            </p>
            <ul className="space-y-1.5">
              {report.nextSteps.map((s, i) => (
                <li key={i} className="flex items-start gap-2.5 text-[13px]">
                  <ArrowRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  <span className="text-foreground">{s}</span>
                </li>
              ))}
            </ul>
          </div>
        </CardContent>
      </Card>

      {/* Supplementary: Run Information (Phase 8) — compact, a few scalar facts, never a debugging log */}
      <RunInformationSection report={report} />
    </div>
    </TooltipProvider>
  );
}
