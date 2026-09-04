import type { ExperimentReport } from '@/lib/types';
import { HypothesisCard } from './hypothesis-card';
import { EffectCard } from './effect-card';

export function HeroCard({ report }: { report: ExperimentReport }) {
  return (
    <div className="rounded-2xl border border-black/10 bg-white p-5 shadow-sm sm:p-6">
      <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr_0.9fr]">
        <HypothesisCard report={report} />
        <EffectCard report={report} />
      </div>
    </div>
  );
}
