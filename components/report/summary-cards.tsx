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
            <span className="font-data">{rm.userCount.toLocaleString()}</span> users exposed
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

      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
        {/* Primary Metric */}
        <div className="rounded-lg border border-border bg-surface p-3.5">
          <div className="flex items-center gap-1.5 text-[12px] font-medium text-muted-foreground">
            <Target className="h-3.5 w-3.5" />
            Primary metric
          </div>
          <p className="mt-1.5 truncate text-[15px] font-semibold text-foreground">
            {primary?.metric ?? report.hypothesis?.primaryMetric ?? 'N/A'}
          </p>
          {report.hypothesis?.expectedEffectRelative != null && (
            <p className="mt-0.5 text-[12px] text-muted-foreground">
              Hypothesized to {report.hypothesis.expectedDirection.replace('_', ' ')} by{' '}
              <span className="font-data">
                {Math.abs(report.hypothesis.expectedEffectRelative * 100).toFixed(1)}%
              </span>{' '}
              (relative)
            </p>
          )}
        </div>

        {/* Variants */}
        <div className="rounded-lg border border-border bg-surface p-3.5">
          <div className="flex items-center gap-1.5 text-[12px] font-medium text-muted-foreground">
            <Users className="h-3.5 w-3.5" />
            Variants
          </div>
          {primary ? (
            <div className="mt-1.5 flex items-end gap-3">
              <div>
                <p className="font-data text-[15px] font-semibold text-foreground">{primary.control}</p>
                <p className="flex items-center gap-1 text-[12px] text-muted-foreground">
                  <span className="h-1.5 w-1.5 rounded-full bg-border-strong" />
                  control
                </p>
              </div>
              <div>
                <p className="font-data text-[15px] font-semibold text-primary">{primary.variant}</p>
                <p className="flex items-center gap-1 text-[12px] text-muted-foreground">
                  <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                  {primary.arm ?? 'treatment'}
                </p>
              </div>
              <span
                className={cn(
                  'ml-auto rounded-full px-1.5 py-0.5 font-data text-[12px] font-semibold',
                  primary.significant ? 'bg-success/[0.1] text-success' : 'bg-secondary text-muted-foreground'
                )}
              >
                {primaryEffectParts(primary).secondary ?? primary.delta}
              </span>
            </div>
          ) : (
            <p className="mt-1.5 text-sm text-muted-foreground">N/A</p>
          )}
        </div>

        {/* Data Quality */}
        <div className="rounded-lg border border-border bg-surface p-3.5">
          <div className="flex items-center gap-1.5 text-[12px] font-medium text-muted-foreground">
            {qualityPassing ? (
              <CheckCircle2 className="h-3.5 w-3.5" />
            ) : (
              <AlertTriangle className="h-3.5 w-3.5" />
            )}
            Data quality
          </div>
          <p
            className={cn(
              'mt-1.5 text-[15px] font-semibold',
              qualityPassing ? 'text-success' : failedChecks.length > 0 ? 'text-warning' : 'text-muted-foreground'
            )}
          >
            {checks.length === 0 ? 'N/A' : qualityPassing ? 'Passing' : `${failedChecks.length} check(s) flagged`}
          </p>
          <p className="mt-0.5 truncate text-[12px] text-muted-foreground">
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
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[12px] font-medium',
        tone === 'go' ? 'border-success/25 bg-success/[0.08] text-success' : 'border-border bg-secondary text-muted-foreground'
      )}
    >
      {children}
    </span>
  );
}
