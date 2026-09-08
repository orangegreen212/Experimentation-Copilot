import type { ExperimentReport } from '@/lib/types';
import { HypothesisCard } from './hypothesis-card';
import { EffectCard } from './effect-card';

export function HeroCard({ report }: { report: ExperimentReport }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-5 sm:p-6">
      <div className="grid gap-6 divide-y divide-border lg:grid-cols-2 lg:divide-x lg:divide-y-0 lg:gap-8">
        <div className="lg:pr-8">
          <HypothesisCard report={report} />
        </div>
        <div className="space-y-4 pt-6 lg:pl-8 lg:pt-0">
          <EffectCard report={report} />
        </div>
      </div>
    </div>
  );
}
