import { CircleCheck, TriangleAlert, CircleX, CircleHelp } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import type { ExperimentReport } from '@/lib/types';
import { decisionToneFor, toneClasses } from '@/lib/report-format';

const TONE_ICON = {
  go: CircleCheck,
  caution: TriangleAlert,
  no: CircleX,
  neutral: CircleHelp,
} as const;

export function RecommendationCard({ report }: { report: ExperimentReport }) {
  const decision = report.decision;
  const tone = decisionToneFor(decision);
  const t = toneClasses(tone);
  const Icon = TONE_ICON[tone];

  return (
    <div className={cn('flex items-start gap-4 rounded-xl border border-border bg-surface p-5')}>
      <span className={cn('h-full min-h-[3.25rem] w-1 shrink-0 rounded-full', t.bar)} aria-hidden />
      <Icon className={cn('mt-0.5 h-5 w-5 shrink-0', t.icon)} />
      <div className="min-w-0 flex-1">
        <p className="text-[12px] font-medium text-muted-foreground">Recommended decision</p>
        <p className={cn('font-data text-xl font-semibold tracking-tight', t.text)}>
          {decision ? decision.replace(/_/g, ' ') : 'N/A'}
        </p>
        <p className="mt-1.5 text-[13px] leading-relaxed text-muted-foreground">
          {report.decisionReason ?? 'See the decision narrative below for the full reasoning.'}
        </p>
        {report.recommendationConfidence && (
          <Badge variant="outline" className="mt-3 border-border bg-secondary text-[11px] font-medium text-muted-foreground">
            Confidence: {report.recommendationConfidence}
          </Badge>
        )}
      </div>
    </div>
  );
}
