import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import type { DecisionTone } from '@/lib/report-format';

export function KpiTile({
  icon,
  label,
  value,
  tone,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  tone: DecisionTone;
}) {
  return (
    <div className="rounded-2xl border border-black/10 bg-white p-4">
      <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wide text-neutral-400">
        <span
          className={cn(
            'flex h-6 w-6 items-center justify-center rounded-md',
            tone === 'go' && 'bg-green-50 text-green-600',
            tone === 'caution' && 'bg-amber-50 text-amber-600',
            tone === 'no' && 'bg-red-50 text-red-600',
            tone === 'neutral' && 'bg-indigo-50 text-indigo-600'
          )}
        >
          {icon}
        </span>
        {label}
      </div>
      <p
        className={cn(
          'mt-2 truncate text-lg font-bold tracking-tight',
          tone === 'go' && 'text-green-600',
          tone === 'caution' && 'text-amber-600',
          tone === 'no' && 'text-red-600',
          tone === 'neutral' && 'text-black'
        )}
      >
        {value}
      </p>
    </div>
  );
}
