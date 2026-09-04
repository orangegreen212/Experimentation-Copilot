import type { ExperimentReport, StatResult, Hypothesis } from '@/lib/types';

/** Picks the stat row the hero/summary blocks should headline: the
 *  backend-flagged winner first, then the first significant pairwise
 *  result, then the first pairwise result, then whatever is first. */
export function selectPrimaryStat(report: ExperimentReport): StatResult | undefined {
  const results = report.stats;
  const winners = results.filter((r) => r.isWinner);
  if (winners.length > 0) return winners[0];
  const pairwise = results.filter((r) => !r.isOmnibus);
  if (pairwise.length > 0) {
    const significantPairwise = pairwise.filter((r) => r.significant);
    return significantPairwise.length > 0 ? significantPairwise[0] : pairwise[0];
  }
  return results[0];
}

/**
 * Executive Summary polish: for conversion-rate metrics, `StatResult.delta`
 * is the RELATIVE change only ("+20.6% (rel)") — the absolute
 * percentage-point change is easier to read at a glance but was never
 * surfaced on its own. This derives it from `control`/`variant`, which
 * the backend already sends as formatted percentage strings (e.g.
 * "4.21%") for exactly these two test types — no new statistic is
 * computed here, this only re-presents numbers the backend already
 * calculated (see hypothesis_tests.py `_compute_binary_result`).
 *
 * Non-conversion-rate metrics are returned unchanged: `delta` there
 * has no absolute-pp counterpart to pair it with.
 */
export function primaryEffectParts(stat: StatResult): { primary: string; secondary?: string } {
  const isConversionRate = stat.testType === 'chi_square' || stat.testType === 'fishers_exact';
  if (!isConversionRate) return { primary: stat.delta };

  const controlPct = parseFloat(stat.control);
  const variantPct = parseFloat(stat.variant);
  if (Number.isNaN(controlPct) || Number.isNaN(variantPct)) return { primary: stat.delta };

  const absolutePp = variantPct - controlPct;
  const absoluteLabel = `${absolutePp >= 0 ? '+' : ''}${absolutePp.toFixed(2)}pp`;
  const relativeLabel = stat.delta.replace(/\s*\(rel\)\s*$/i, ' relative');
  return { primary: absoluteLabel, secondary: `(${relativeLabel})` };
}

/** Inline "+0.61pp (+20.6% relative)" form, used in prose (the Effect card renders the two parts separately). */
export function formatPrimaryEffect(stat: StatResult): string {
  const { primary, secondary } = primaryEffectParts(stat);
  return secondary ? `${primary} ${secondary}` : primary;
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value == null) return 'N/A';
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value == null) return 'N/A';
  return value.toFixed(digits);
}

/**
 * Human-readable status word for a hypothesis verdict. Deliberately
 * distinct vocabulary from the business Decision (`GO` / `GO WITH
 * CAUTION` / `NO_GO` / ...) so the two concepts are never visually
 * confused — see ExperimentReport.hypothesisEvaluation's docstring.
 * `verdict` itself is never computed here; this only relabels the
 * already-deterministic backend value for display.
 */
export function hypothesisResultLabel(verdict: string | null | undefined): string {
  if (verdict === 'SUPPORTED') return 'SUPPORTED';
  if (verdict === 'PARTIALLY_SUPPORTED') return 'PARTIALLY SUPPORTED';
  if (verdict === 'NOT_SUPPORTED') return 'REJECTED';
  return 'EVALUATION UNAVAILABLE';
}

export function verdictBadgeClass(verdict: string | null | undefined) {
  if (verdict === 'SUPPORTED') return 'border-green-200 bg-green-50 text-green-700';
  if (verdict === 'PARTIALLY_SUPPORTED') return 'border-amber-200 bg-amber-50 text-amber-700';
  if (verdict === 'NOT_SUPPORTED') return 'border-red-200 bg-red-50 text-red-700';
  return 'border-black/10 bg-neutral-50 text-neutral-500';
}

/** Matches HypothesisEvaluator's own matching rule (app/stats/hypothesis_evaluator.py):
 *  exact metric-name match, skipping omnibus rows — never fuzzy, never
 *  recomputed here. Used only to surface the p-value / CI already
 *  computed for that metric next to the hypothesis verdict. */
export function findHypothesisStatResult(
  hypothesis: Hypothesis,
  stats: StatResult[]
): StatResult | undefined {
  return stats.find((s) => !s.isOmnibus && s.metric === hypothesis.primaryMetric);
}

export type DecisionTone = 'go' | 'caution' | 'no' | 'neutral';

/** Maps report.decision to the same go/caution/no/neutral tone vocabulary
 *  used across every hero-style block (tiles, guardrails, recommendation). */
export function decisionToneFor(decision: string | null | undefined): DecisionTone {
  if (decision === 'GO') return 'go';
  if (decision === 'GO_WITH_CAUTION') return 'caution';
  if (decision === 'NO_GO' || decision === 'INVALID') return 'no';
  return 'neutral';
}
