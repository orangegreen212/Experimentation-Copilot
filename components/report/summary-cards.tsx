import { type ReactNode } from 'react';
import { CheckCircle2, AlertTriangle, Target, Users } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ExperimentReport } from '@/lib/types';
import { selectPrimaryStat, primaryEffectParts } from '@/lib/report-format';

/**
 * "At a glance" summary — sits directly under ExperimentHeader, above
 * the detailed HeroCard/KpiGrid/etc. below it. Purely a compact
 * re-presentation of facts the backend already computed
 * (`selectPrimaryStat`, `report.qualityChecks`, `report.hypothesis`) —
 * nothing here is a new statistic, and nothing below this component
 * changes: this is additive, not a replacement for the detailed
 * sections a reader can still scroll into.
 *
 * Deliberately does NOT show a "days running" badge some reference
 * designs include: this product analyzes an already-collected,
 * static dataset snapshot, not a live/ongoing experiment — there is
 * no real "days running" fact to report, and fabricating one would
 * violate the same "never invent a number" rule every other card in
 * this file follows. `users exposed` below is real
 * (`RunMetadata.userCount`, the actual observed row count).
 */
export function SummaryCards({ report }: { report: ExperimentReport }) {
  const rm = report.runMetadata;
  const primary = selectPrimaryStat(report);
  const checks = report.qualityChecks ?? [];
  const failedChecks = checks.filter((c) => !c.passed);
  const qualityPassing = checks.length > 0 && failedChecks.length === 0;

  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-2">
        {rm && (
          <Badge tone="neutral">
            {rm.userCount.toLocaleString()} users exposed
          </Badge>
        )}
        {primary && (
          <Badge tone={primary.significant ? 'go' : 'neutral'}>
            {primary.significant ? (
              <CheckCircle2 className="h-3 w-3" />
            ) : (
              <AlertTriangle className="h-3 w-3" />
            )}
            {primary.significant ? 'Statistically significant' : 'Not statistically significant'}
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {/* Primary Metric */}
        <div className="rounded-xl border border-black/10 bg-white p-4">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-neutral-400">
            <Target className="h-3.5 w-3.5" />
            Primary Metric
          </div>
          <p className="mt-1.5 truncate text-[15px] font-semibold text-black">
            {primary?.metric ?? report.hypothesis?.primaryMetric ?? 'N/A'}
          </p>
          {report.hypothesis?.expectedEffectRelative != null && (
            <p className="mt-0.5 text-xs text-neutral-400">
              Hypothesized to {report.hypothesis.expectedDirection.replace('_', ' ')} by{' '}
              {Math.abs(report.hypothesis.expectedEffectRelative * 100).toFixed(1)}% (relative)
            </p>
          )}
        </div>

        {/* Variants */}
        <div className="rounded-xl border border-black/10 bg-white p-4">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-neutral-400">
            <Users className="h-3.5 w-3.5" />
            Variants
          </div>
          {primary ? (
            <div className="mt-1.5 flex items-end gap-3">
              <div>
                <p className="text-[15px] font-semibold text-black">{primary.control}</p>
                <p className="flex items-center gap-1 text-[11px] text-neutral-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-neutral-400" />
                  control
                </p>
              </div>
              <div>
                <p className="text-[15px] font-semibold text-black">{primary.variant}</p>
                <p className="flex items-center gap-1 text-[11px] text-neutral-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
                  {primary.arm ?? 'treatment'}
                </p>
              </div>
              <span
                className={cn(
                  'ml-auto rounded-full px-1.5 py-0.5 text-[11px] font-semibold',
                  primary.significant ? 'bg-green-50 text-green-700' : 'bg-neutral-100 text-neutral-500'
                )}
              >
                {primaryEffectParts(primary).secondary ?? primary.delta}
              </span>
            </div>
          ) : (
            <p className="mt-1.5 text-sm text-neutral-400">N/A</p>
          )}
        </div>

        {/* Data Quality */}
        <div className="rounded-xl border border-black/10 bg-white p-4">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-neutral-400">
            {qualityPassing ? (
              <CheckCircle2 className="h-3.5 w-3.5" />
            ) : (
              <AlertTriangle className="h-3.5 w-3.5" />
            )}
            Data Quality
          </div>
          <p
            className={cn(
              'mt-1.5 text-[15px] font-semibold',
              qualityPassing ? 'text-green-700' : failedChecks.length > 0 ? 'text-amber-700' : 'text-neutral-400'
            )}
          >
            {checks.length === 0 ? 'N/A' : qualityPassing ? 'Passing' : `${failedChecks.length} check(s) flagged`}
          </p>
          <p className="mt-0.5 truncate text-xs text-neutral-400">
            {checks.length === 0
              ? 'No automated checks were run.'
              : qualityPassing
              ? `All ${checks.length} automated data checks passed.`
              : failedChecks.map((c) => c.label).join(', ')}
          </p>
        </div>
      </div>
    </div>
  );
}

function Badge({ tone, children }: { tone: 'go' | 'neutral'; children: ReactNode }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium',
        tone === 'go' ? 'border-green-200 bg-green-50 text-green-700' : 'border-black/10 bg-neutral-50 text-neutral-500'
      )}
    >
      {children}
    </span>
  );
}
