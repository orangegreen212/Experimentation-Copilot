import { CheckCircle2, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ExperimentReport } from '@/lib/types';
import { hypothesisResultLabel } from '@/lib/report-format';

export function HypothesisCard({ report }: { report: ExperimentReport }) {
  const hypothesis = report.hypothesis;
  const verdict = report.hypothesisEvaluation?.verdict ?? null;
  const isSupported = verdict === 'SUPPORTED';
  const isPartial = verdict === 'PARTIALLY_SUPPORTED';
  const isRejected = verdict === 'NOT_SUPPORTED';

  return (
    <div className="flex items-start gap-4">
      <div
        className={cn(
          'flex h-14 w-14 shrink-0 items-center justify-center rounded-full border-4',
          isSupported && 'border-green-100 bg-green-50 text-green-600',
          isPartial && 'border-amber-100 bg-amber-50 text-amber-600',
          isRejected && 'border-red-100 bg-red-50 text-red-600',
          !isSupported && !isPartial && !isRejected && 'border-neutral-100 bg-neutral-50 text-neutral-400'
        )}
      >
        {isRejected ? <XCircle className="h-6 w-6" /> : <CheckCircle2 className="h-6 w-6" />}
      </div>
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-wide text-neutral-400">
          Hypothesis Evaluation
        </p>
        <p
          className={cn(
            'text-2xl font-bold tracking-tight',
            isSupported && 'text-green-600',
            isPartial && 'text-amber-600',
            isRejected && 'text-red-600',
            !isSupported && !isPartial && !isRejected && 'text-neutral-500'
          )}
        >
          {hypothesisResultLabel(verdict)}
        </p>
        {hypothesis && (
          <p className="mt-1 text-[13px] leading-relaxed text-neutral-600">{hypothesis.statement}</p>
        )}
      </div>
    </div>
  );
}
