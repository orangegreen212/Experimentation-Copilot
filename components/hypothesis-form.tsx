'use client';

import { ChevronDown, FlaskConical } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import type { DatasetInfo, ExpectedDirection, Hypothesis } from '@/lib/types';

interface HypothesisFormProps {
  dataset: DatasetInfo;
  /** null = analyst hasn't opted in to providing a hypothesis this run. */
  value: Hypothesis | null;
  onChange: (h: Hypothesis | null) => void;
}

const DIRECTION_OPTIONS: { value: ExpectedDirection; label: string }[] = [
  { value: 'increase', label: 'Increase' },
  { value: 'decrease', label: 'Decrease' },
  { value: 'no_change', label: 'No change' },
];

/**
 * Fraction -> percent for the input box. Plain `value * 100` reintroduces
 * binary floating-point noise (0.07 * 100 === 7.000000000000001 in JS), which
 * then shows up character-for-character in the controlled <input> while the
 * user is typing. Rounding to 8 decimal places removes that noise while still
 * preserving any real precision a user might type (e.g. 5.25%).
 */
function relativeToPercentInput(value: number): number {
  return Math.round(value * 100 * 1e8) / 1e8;
}

/**
 * Phase 1 — structured Experiment Hypothesis section, shown inline in
 * the existing New Experiment setup flow (workspace-view.tsx), right
 * after the classifier banner establishes what metrics are available.
 * Not a new page, not a second metric-selection system: the "Primary
 * metric" dropdown is populated straight from `dataset.availableMetrics`
 * / `dataset.metricLabel` — the same detected metrics the rest of the
 * app already uses.
 *
 * Entirely optional and collapsed by default: closing this section (or
 * never opening it) means `value` stays `null` and nothing new is sent
 * to the backend — see lib/api.ts's analyzeExperiment, which only
 * includes `hypothesis` in the request body when it's non-null.
 *
 * Phase 1 does NOT show or compute a verdict here — there is nothing
 * yet to compare the hypothesis against. This form only captures,
 * validates client-side for basic completeness, and hands off a
 * structured `Hypothesis` object; the backend's own Pydantic
 * validation (app/schemas/hypothesis.py) remains the source of truth.
 */
export function HypothesisForm({ dataset, value, onChange }: HypothesisFormProps) {
  const enabled = value !== null;
  const availableMetrics =
    dataset.availableMetrics && dataset.availableMetrics.length > 0
      ? dataset.availableMetrics
      : [dataset.metricLabel];

  const draft: Hypothesis = value ?? {
    statement: '',
    primaryMetric: dataset.metricLabel,
    expectedDirection: 'increase',
    expectedEffectRelative: null,
    rationale: null,
  };

  const update = (patch: Partial<Hypothesis>) => onChange({ ...draft, ...patch });

  const setEnabled = (next: boolean) => {
    if (next) {
      onChange(draft);
    } else {
      onChange(null);
    }
  };

  return (
    <Card className="border-black/10 shadow-none">
      <Collapsible open={enabled} onOpenChange={setEnabled}>
        <CollapsibleTrigger asChild>
          <button type="button" className="w-full text-left">
            <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
              <div>
                <div className="flex items-center gap-2">
                  <FlaskConical className="h-4 w-4 text-black" />
                  <CardTitle className="text-[15px] tracking-tight">Hypothesis (optional)</CardTitle>
                </div>
                <CardDescription>
                  Capture what you expected before seeing the result
                </CardDescription>
              </div>
              <ChevronDown
                className={cn(
                  'h-4 w-4 shrink-0 text-neutral-400 transition-transform',
                  enabled && 'rotate-180'
                )}
              />
            </CardHeader>
          </button>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="space-y-4 pt-0">
            <div>
              <Label htmlFor="hypothesis-statement" className="text-[13px] font-medium text-black">
                Hypothesis statement
              </Label>
              <Textarea
                id="hypothesis-statement"
                value={draft.statement}
                onChange={(e) => update({ statement: e.target.value })}
                placeholder="e.g. Increasing the checkout CTA visibility will increase checkout conversion."
                className="mt-1.5 min-h-[60px] resize-none border-black/10 placeholder:text-neutral-400"
                rows={2}
                maxLength={500}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label htmlFor="hypothesis-metric" className="text-[13px] font-medium text-black">
                  Primary metric
                </Label>
                <Select
                  value={draft.primaryMetric}
                  onValueChange={(v) => update({ primaryMetric: v })}
                >
                  <SelectTrigger id="hypothesis-metric" className="mt-1.5 border-black/10">
                    <SelectValue placeholder="Select a metric" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableMetrics.map((metric) => (
                      <SelectItem key={metric} value={metric}>
                        {metric}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="hypothesis-direction" className="text-[13px] font-medium text-black">
                  Expected direction
                </Label>
                <Select
                  value={draft.expectedDirection}
                  onValueChange={(v) => update({ expectedDirection: v as ExpectedDirection })}
                >
                  <SelectTrigger id="hypothesis-direction" className="mt-1.5 border-black/10">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {DIRECTION_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div>
              <Label htmlFor="hypothesis-effect" className="text-[13px] font-medium text-black">
                Expected effect
              </Label>
              <div className="mt-1.5 flex items-center gap-2">
                <Input
                  id="hypothesis-effect"
                  type="number"
                  inputMode="decimal"
                  step="0.1"
                  min={0}
                  value={
                    draft.expectedEffectRelative === null || draft.expectedEffectRelative === undefined
                      ? ''
                      : relativeToPercentInput(draft.expectedEffectRelative)
                  }
                  onChange={(e) => {
                    const raw = e.target.value;
                    update({
                      expectedEffectRelative: raw === '' ? null : Number(raw) / 100,
                    });
                  }}
                  placeholder="5"
                  className="w-32 border-black/10 placeholder:text-neutral-400"
                />
                <span className="text-xs text-neutral-500">
                  % relative — e.g. 5 means baseline 10% &rarr; 10.5%, NOT 15%
                </span>
              </div>
              {draft.expectedEffectRelative !== null && draft.expectedEffectRelative !== undefined && (
                <p className="mt-1 text-xs font-medium text-neutral-600">
                  Expected effect: {draft.expectedDirection === 'decrease' ? '-' : '+'}
                  {(draft.expectedEffectRelative * 100).toFixed(1)}% relative
                </p>
              )}
            </div>

            <div>
              <Label htmlFor="hypothesis-rationale" className="text-[13px] font-medium text-black">
                Business rationale <span className="font-normal text-neutral-400">(optional)</span>
              </Label>
              <Textarea
                id="hypothesis-rationale"
                value={draft.rationale ?? ''}
                onChange={(e) => update({ rationale: e.target.value === '' ? null : e.target.value })}
                placeholder="Why do you expect this effect?"
                className="mt-1.5 min-h-[60px] resize-none border-black/10 placeholder:text-neutral-400"
                rows={2}
                maxLength={2000}
              />
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}
