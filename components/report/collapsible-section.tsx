'use client';

import { useState, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

/**
 * Shared Tier-3/4 progressive-disclosure wrapper: a compact, always-
 * visible summary line the user can scan in passing, expanding to the
 * full detail only on demand. Used to bundle sections that are
 * necessary for a responsible decision but shouldn't compete with the
 * decision itself for visual weight — statistical methodology, data
 * quality checks, segmentation, audit trail.
 */
export function CollapsibleSection({
  icon,
  title,
  meta,
  defaultOpen = false,
  children,
}: {
  icon?: ReactNode;
  title: string;
  /** Short scannable teaser shown next to the title when collapsed, e.g. "Significant · 95% confidence". */
  meta?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-border bg-surface">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2.5 px-4 py-3 text-left"
        aria-expanded={open}
      >
        {icon && <span className="text-muted-foreground">{icon}</span>}
        <span className="text-[13px] font-semibold tracking-tight text-foreground">{title}</span>
        {meta && <span className="text-[12px] text-muted-foreground">{meta}</span>}
        <ChevronDown
          className={cn('ml-auto h-4 w-4 shrink-0 text-muted-foreground transition-transform', open && 'rotate-180')}
        />
      </button>
      {open && <div className="space-y-3 border-t border-border p-4 pt-3.5">{children}</div>}
    </div>
  );
}
