import { type ReactNode } from 'react';
import { CheckCircle2, AlertTriangle, Users } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ExperimentReport } from '@/lib/types';
import { selectPrimaryStat } from '@/lib/report-format';

/**
 * Slim "at a glance" badge strip above the Decision Summary. This used
 * to also carry Primary Metric / Variants / Data Quality tiles, but
 * those duplicated HeroCard (primary metric + control/variant/effect)
 * and the collapsed Statistical Details section (data quality) one
 * screen down — per the redesign brief's explicit instruction to merge
 * duplicated information rather than preserve it because it already
 * existed. What's left here is real information that appears nowhere
 * else: how many real users this result is based on, and a one-glance
 * significance flag before the reader even reaches HeroCard.
 */
export function SummaryCards({ report }: { report: ExperimentReport }) {
  const rm = report.runMetadata;
  const primary = selectPrimaryStat(report);

  if (!rm && !primary) return null;

  return (
    <div className="flex flex-wrap items-center gap-2">
      {rm && (
        <Badge tone="neutral">
          <Users className="h-3 w-3" />
          <span className="font-data">{rm.userCount.toLocaleString()}</span> users exposed
        </Badge>
      )}
      {primary && (
        <Badge tone={primary.significant ? 'go' : 'neutral'}>
          {primary.significant ? <CheckCircle2 className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
          {primary.significant ? 'Statistically significant' : 'Not statistically significant'}
        </Badge>
      )}
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
