import { CheckCircle2, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ExperimentReport } from '@/lib/types';
import { hypothesisResultLabel, toneClasses } from '@/lib/report-format';

export function HypothesisCard({ report }: { report: ExperimentReport }) {
  const hypothesis = report.hypothesis;
  const verdict = report.hypothesisEvaluation?.verdict ?? null;
  const isRejected = verdict === 'NOT_SUPPORTED';
  const tone =
    verdict === 'SUPPORTED' ? 'go' : verdict === 'PARTIALLY_SUPPORTED' ? 'caution' : isRejected ? 'no' : 'neutral';
  const t = toneClasses(tone);

  return (
    <div className="flex items-start gap-3.5">
      <div
        className={cn(
          'flex h-11 w-11 shrink-0 items-center justify-center rounded-full border',
          t.border,
          t.bg,
          t.icon
        )}
      >
        {isRejected ? <XCircle className="h-5 w-5" /> : <CheckCircle2 className="h-5 w-5" />}
      </div>
      <div className="min-w-0">
        <p className="text-[12px] font-medium text-muted-foreground">Hypothesis evaluation</p>
        <p className={cn('font-data text-[22px] font-semibold leading-tight tracking-tight', t.text)}>
          {hypothesisResultLabel(verdict)}
        </p>
        {hypothesis && (
          <p className="mt-1.5 text-[13px] leading-relaxed text-muted-foreground">{hypothesis.statement}</p>
        )}
      </div>
    </div>
  );
}
