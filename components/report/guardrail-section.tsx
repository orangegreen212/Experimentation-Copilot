import { CheckCircle2, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import type { ExperimentReport } from '@/lib/types';

export function GuardrailSection({ report }: { report: ExperimentReport }) {
  const guardrails = report.decisionSupport?.guardrailFindings ?? [];

  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <p className="mb-3 text-[13px] font-semibold tracking-tight text-foreground">Guardrail Metrics</p>
      {guardrails.length === 0 ? (
        <p className="text-[13px] text-muted-foreground">No guardrail metrics evaluated.</p>
      ) : (
        <div className="space-y-2">
          {guardrails.map((g) => (
            <div
              key={g.metric}
              className="flex items-center justify-between gap-3 rounded-lg border border-border bg-secondary px-3.5 py-2.5"
            >
              <div className="flex items-center gap-2.5">
                <div
                  className={cn(
                    'flex h-7 w-7 items-center justify-center rounded-full',
                    g.violated ? 'bg-destructive/[0.12] text-destructive' : 'bg-success/[0.12] text-success'
                  )}
                >
                  {g.violated ? <XCircle className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
                </div>
                <div>
                  <span className="block text-[13px] font-medium text-foreground">{g.metric}</span>
                  {(g.observedValue != null || g.relativeChange != null) && (
                    <span className="block font-data text-[11px] text-muted-foreground">
                      {g.observedValue != null && <>observed {g.observedValue.toLocaleString()}</>}
                      {g.observedValue != null && g.relativeChange != null && ' · '}
                      {g.relativeChange != null && (
                        <>
                          {g.relativeChange >= 0 ? '+' : ''}
                          {g.relativeChange.toFixed(2)}% change
                        </>
                      )}
                    </span>
                  )}
                </div>
              </div>
              <Badge
                variant="outline"
                className={cn(
                  'shrink-0 text-[10px] font-semibold',
                  g.violated
                    ? 'border-destructive/25 bg-destructive/[0.08] text-destructive'
                    : 'border-success/25 bg-success/[0.08] text-success'
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

