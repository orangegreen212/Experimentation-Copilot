import { ArrowRight } from 'lucide-react';
import type { ExperimentReport } from '@/lib/types';
import { selectPrimaryStat, primaryEffectParts, formatPercent } from '@/lib/report-format';

export function EffectCard({ report }: { report: ExperimentReport }) {
  const primary = selectPrimaryStat(report);
  const evaluation = report.hypothesisEvaluation;
  if (!primary) return null;

  return (
    <>
      <div>
        <p className="text-[12px] font-medium text-muted-foreground">Observed effect</p>
        <p
          className={
            'font-data text-[32px] font-semibold leading-tight tracking-tight ' +
            (primary.significant ? 'text-primary' : 'text-foreground')
          }
        >
          {primaryEffectParts(primary).primary}
        </p>
        {primaryEffectParts(primary).secondary && (
          <p className="mt-0.5 font-data text-[12px] font-medium text-muted-foreground">
            {primaryEffectParts(primary).secondary}
          </p>
        )}
        <div className="mt-2.5 flex items-center gap-2 text-[13px]">
          <span className="font-data font-semibold text-foreground">{primary.control}</span>
          <ArrowRight className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="font-data font-semibold text-primary">{primary.variant}</span>
        </div>
      </div>

      <div className="space-y-1.5 rounded-lg bg-secondary p-3.5 text-[13px]">
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">p-value</span>
          <span className="font-data font-semibold text-foreground">
            {primary.pValue < 0.001 ? '<0.001' : primary.pValue.toFixed(3)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">95% CI</span>
          <span className="font-data font-semibold text-foreground">
            [{primary.ciLower}, {primary.ciUpper}]
          </span>
        </div>
        {evaluation?.expectedEffectRelative != null && (
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Expected effect</span>
            <span className="font-data font-semibold text-foreground">
              {formatPercent(evaluation.expectedEffectRelative)}
            </span>
          </div>
        )}
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">MDE (relative)</span>
          <span className="font-data font-semibold text-foreground">{report.mde}</span>
        </div>
      </div>
    </>
  );
}
