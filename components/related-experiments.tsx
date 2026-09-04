'use client';

import { History } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import type { ConfidenceLevel, RelatedExperiment } from '@/lib/types';

const CONFIDENCE_BADGE: Record<ConfidenceLevel, string> = {
  HIGH: 'border-green-200 bg-green-50 text-green-700',
  MEDIUM: 'border-black/10 bg-neutral-100 text-neutral-600',
  LOW: 'border-red-200 bg-red-50 text-red-700',
};

interface RelatedExperimentsProps {
  items: RelatedExperiment[];
}

/**
 * Plain structured retrieval from ExperimentStore.list_related() —
 * prior runs against the same dataset name. Not semantic/LLM memory,
 * just a factual list: "this dataset was reviewed before, here's what
 * was decided." Renders nothing when there's no prior history.
 */
export function RelatedExperiments({ items }: RelatedExperimentsProps) {
  if (items.length === 0) return null;

  return (
    <div className="rounded-lg border border-black/10 bg-neutral-50/50 p-4">
      <div className="mb-2 flex items-center gap-2">
        <History className="h-3.5 w-3.5 text-neutral-500" />
        <p className="text-[13px] font-medium text-black">
          Previously reviewed — {items.length} prior run{items.length > 1 ? 's' : ''} on this dataset
        </p>
      </div>
      <ul className="space-y-1.5">
        {items.map((item) => (
          <li key={item.experimentId} className="flex items-center gap-2 text-xs">
            <span className="shrink-0 text-neutral-400">
              {new Date(item.createdAt).toLocaleDateString()}
            </span>
            <Badge
              variant="outline"
              className={cn('shrink-0 text-[10px]', CONFIDENCE_BADGE[item.confidence])}
            >
              {item.confidence}
            </Badge>
            <span className="truncate text-neutral-600">{item.userPrompt}</span>
            <span className="ml-auto shrink-0 font-medium text-neutral-500">
              {item.decision === 'ship' ? 'Shipped' : 'Held'}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
