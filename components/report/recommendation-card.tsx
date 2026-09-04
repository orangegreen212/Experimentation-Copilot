import { AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import type { ExperimentReport } from '@/lib/types';
import { decisionToneFor } from '@/lib/report-format';

export function RecommendationCard({ report }: { report: ExperimentReport }) {
  const decision = report.decision;
  const tone = decisionToneFor(decision);

  return (
    <div
      className={cn(
        'flex flex-col items-start rounded-2xl border p-5',
        tone === 'go' && 'border-green-100 bg-green-50/60',
        tone === 'caution' && 'border-amber-100 bg-amber-50/60',
        tone === 'no' && 'border-red-100 bg-red-50/60',
        tone === 'neutral' && 'border-black/10 bg-neutral-50'
      )}
    >
      <div
        className={cn(
          'flex h-11 w-11 items-center justify-center rounded-full',
          tone === 'go' && 'bg-green-100 text-green-600',
          tone === 'caution' && 'bg-amber-100 text-amber-600',
          tone === 'no' && 'bg-red-100 text-red-600',
          tone === 'neutral' && 'bg-neutral-200 text-neutral-500'
        )}
      >
        <AlertTriangle className="h-5 w-5" />
      </div>
      <p
        className={cn(
          'mt-3 text-lg font-bold tracking-tight',
          tone === 'go' && 'text-green-700',
          tone === 'caution' && 'text-amber-700',
          tone === 'no' && 'text-red-700',
          tone === 'neutral' && 'text-neutral-600'
        )}
      >
        {decision ? decision.replace(/_/g, ' ') : 'N/A'}
      </p>
      <p className="mt-1 text-[13px] leading-relaxed text-neutral-600">
        {report.decisionReason ?? 'See Decision Narrative below for the full reasoning.'}
      </p>
      {report.recommendationConfidence && (
        <Badge variant="outline" className="mt-3 border-black/10 bg-white text-[10px] text-neutral-600">
          Recommendation Confidence: {report.recommendationConfidence}
        </Badge>
      )}
    </div>
  );
}
