import { CheckCircle2, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import type { ExperimentReport } from '@/lib/types';

export function GuardrailSection({ report }: { report: ExperimentReport }) {
  const guardrails = report.decisionSupport?.guardrailFindings ?? [];

  return (
    <div className="rounded-2xl border border-black/10 bg-white p-5">
      <p className="mb-3 text-[13px] font-semibold tracking-tight text-black">Guardrail Metrics</p>
      {guardrails.length === 0 ? (
        <p className="text-[13px] text-neutral-500">No guardrail metrics evaluated.</p>
      ) : (
        <div className="space-y-2">
          {guardrails.map((g) => (
            <div
              key={g.metric}
              className="flex items-center justify-between gap-3 rounded-lg border border-black/5 bg-neutral-50 px-3.5 py-2.5"
            >
              <div className="flex items-center gap-2.5">
                <div
                  className={cn(
                    'flex h-7 w-7 items-center justify-center rounded-full',
                    g.violated ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'
                  )}
                >
                  {g.violated ? <XCircle className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
                </div>
                <span className="text-[13px] font-medium text-black">{g.metric}</span>
              </div>
              <Badge
                variant="outline"
                className={cn(
                  'text-[10px] font-semibold',
                  g.violated
                    ? 'border-red-200 bg-red-50 text-red-700'
                    : 'border-green-200 bg-green-50 text-green-700'
                )}
              >
                {g.violated ? 'VIOLATED' : 'PASS'}
              </Badge>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
