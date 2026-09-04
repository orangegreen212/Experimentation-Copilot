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
        <p className="text-[10px] font-semibold uppercase tracking-wide text-neutral-400">
          Observed Effect
        </p>
        <p
          className={
            'text-3xl font-bold tracking-tight ' +
            (primary.significant ? 'text-indigo-600' : 'text-neutral-700')
          }
        >
          {primaryEffectParts(primary).primary}
        </p>
        {primaryEffectParts(primary).secondary && (
          <p className="mt-0.5 text-[11px] font-medium text-neutral-400">
            {primaryEffectParts(primary).secondary}
          </p>
        )}
        <div className="mt-2 flex items-center gap-2 text-[13px]">
          <span className="font-semibold text-black">{primary.control}</span>
          <ArrowRight className="h-3.5 w-3.5 text-neutral-400" />
          <span className="font-semibold text-indigo-600">{primary.variant}</span>
        </div>
      </div>

      <div className="space-y-2 rounded-xl bg-neutral-50 p-3.5 text-[13px]">
        <div className="flex items-center justify-between">
          <span className="text-neutral-500">p-value</span>
          <span className="font-mono font-semibold text-black">
            {primary.pValue < 0.001 ? '<0.001' : primary.pValue.toFixed(3)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-neutral-500">95% CI</span>
          <span className="font-mono font-semibold text-black">
            [{primary.ciLower}, {primary.ciUpper}]
          </span>
        </div>
        {evaluation?.expectedEffectRelative != null && (
          <div className="flex items-center justify-between">
            <span className="text-neutral-500">Expected Effect</span>
            <span className="font-mono font-semibold text-black">
              {formatPercent(evaluation.expectedEffectRelative)}
            </span>
          </div>
        )}
        <div className="flex items-center justify-between">
          <span className="text-neutral-500">MDE (Relative)</span>
          <span className="font-mono font-semibold text-black">{report.mde}</span>
        </div>
      </div>
    </>
  );
}
