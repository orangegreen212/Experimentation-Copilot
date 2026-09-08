import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import type { DecisionTone } from '@/lib/report-format';
import { toneClasses } from '@/lib/report-format';

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
  const t = toneClasses(tone);
  return (
    <div className="rounded-lg border border-border bg-surface p-3.5">
      <div className="flex items-center gap-2 text-[12px] font-medium text-muted-foreground">
        <span className={cn('flex h-5 w-5 items-center justify-center', t.icon)}>{icon}</span>
        {label}
      </div>
      <p className={cn('mt-2 truncate font-data text-[17px] font-semibold tracking-tight', t.text)}>
        {value}
      </p>
    </div>
  );
}
