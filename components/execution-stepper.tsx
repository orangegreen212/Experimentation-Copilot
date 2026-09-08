'use client';

import { Check, Loader2, Circle, AlertTriangle, X, MinusCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ExecutionStep, StepStatus } from '@/lib/types';

interface StepperProps {
  steps: ExecutionStep[];
  statuses: Record<string, StepStatus>;
}

const GROUP_ORDER = ['Classifier', 'Planner', 'Capability', 'Decision Engine'] as const;

/**
 * Phase 8 — real (backend-computed) icon/color for a finished step,
 * distinct from the `done` (frontend fake-timer) check below. A
 * WARNING (e.g. a graceful LLM fallback) or FAILED step must look
 * visually distinct from an ordinary SUCCESS, and a SKIPPED step must
 * look intentional rather than broken — never all rendered as the
 * same green check.
 */
function RealStatusIcon({ status }: { status: ExecutionStep['status'] }) {
  switch (status) {
    case 'WARNING':
      return <AlertTriangle className="h-4 w-4 text-warning" />;
    case 'FAILED':
      return <X className="h-4 w-4 text-destructive" />;
    case 'SKIPPED':
      return <MinusCircle className="h-4 w-4 text-muted-foreground" />;
    case 'SUCCESS':
    default:
      return <Check className="h-4 w-4 text-success" />;
  }
}

export function ExecutionStepper({ steps, statuses }: StepperProps) {
  const grouped = GROUP_ORDER.map((g) => ({
    group: g,
    items: steps.filter((s) => s.group === g),
  }));

  return (
    <div className="space-y-5">
      {grouped.map(({ group, items }) => (
        <div key={group}>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            {group}
          </p>
          <div className="space-y-px">
            {items.map((step) => {
              const status = statuses[step.id] ?? 'pending';
              return (
                <div
                  key={step.id}
                  className={cn(
                    'flex items-start gap-3 border-l-2 px-3 py-2 transition-colors',
                    status === 'running' && 'border-black bg-secondary',
                    status === 'done' && step.status === 'FAILED' && 'border-destructive/30 bg-destructive/[0.08]',
                    status === 'done' && step.status === 'WARNING' && 'border-warning/30 bg-warning/[0.08]',
                    status === 'done' && (step.status === 'SUCCESS' || step.status === 'SKIPPED' || !step.status) && 'border-border-strong',
                    status === 'pending' && 'border-border opacity-50'
                  )}
                >
                  <div className="mt-0.5 shrink-0">
                    {status === 'done' && <RealStatusIcon status={step.status} />}
                    {status === 'running' && (
                      <Loader2 className="h-4 w-4 animate-spin text-foreground" />
                    )}
                    {status === 'pending' && (
                      <Circle className="h-4 w-4 text-muted-foreground/50" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p
                        className={cn(
                          'text-[13px] font-medium leading-tight',
                          status === 'done' && 'text-foreground',
                          status === 'running' && 'text-foreground',
                          status === 'pending' && 'text-muted-foreground'
                        )}
                      >
                        {step.label}
                      </p>
                      {status === 'done' && step.status === 'SKIPPED' && (
                        <span className="rounded-full bg-secondary px-1.5 py-0.5 text-[9px] font-medium text-muted-foreground">
                          Skipped
                        </span>
                      )}
                      {status === 'done' && step.status === 'WARNING' && (
                        <span className="rounded-full bg-warning/[0.15] px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-warning">
                          Warning
                        </span>
                      )}
                      {status === 'done' && step.status === 'FAILED' && (
                        <span className="rounded-full bg-destructive/[0.15] px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-destructive">
                          Failed
                        </span>
                      )}
                    </div>
                    {status !== 'pending' && (
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {step.detail}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
