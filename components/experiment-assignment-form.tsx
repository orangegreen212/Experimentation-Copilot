'use client';

/**
 * Phase 6 — Allocation / Randomization ("Assignment").
 *
 * Stage 0 architecture doc §6: "здесь мы не строим настоящий feature
 * flag/randomization service" — this section only captures the
 * INTENDED randomization unit (RandomizationUnit — see
 * schemas/experiment_definition.py) and displays the expected split
 * already configured in Variants (experiment-variants-form.tsx) as a
 * simple stacked bar. It does not duplicate variant editing — that
 * stays the single source of truth in the Variants section above this
 * one; this component only reads `definition.variants`.
 *
 * The doc is explicit that the EXISTING SRM check (app/stats/srm.py)
 * is what verifies real observed traffic against this expectation,
 * once analysis actually runs — this section is the "expected" half
 * only, never a live/observed number.
 */

import { useEffect, useState } from 'react';
import { Loader2, Shuffle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { updateExperimentDefinition, ApiError } from '@/lib/api';
import type { ExperimentDefinition, RandomizationUnit } from '@/lib/types';

interface ExperimentAssignmentFormProps {
  definition: ExperimentDefinition;
  onSaved: (updated: ExperimentDefinition) => void;
}

const UNIT_OPTIONS: { value: RandomizationUnit; label: string }[] = [
  { value: 'user', label: 'User' },
  { value: 'session', label: 'Session' },
  { value: 'device', label: 'Device' },
];

// Distinct fills for up to a handful of treatment arms; Control always
// renders in neutral gray regardless of position (see the render loop
// below), so this palette only needs to cover non-control variants.
const SEGMENT_COLORS = ['bg-indigo-500', 'bg-emerald-500', 'bg-amber-500', 'bg-rose-500', 'bg-sky-500'];

export function ExperimentAssignmentForm({ definition, onSaved }: ExperimentAssignmentFormProps) {
  const [randomizationUnit, setRandomizationUnit] = useState<RandomizationUnit>(
    definition.randomizationUnit
  );
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    setRandomizationUnit(definition.randomizationUnit);
    setSaveError(null);
  }, [definition.id, definition.updatedAt]);

  const dirty = randomizationUnit !== definition.randomizationUnit;
  const totalAllocation = definition.variants.reduce((sum, v) => sum + (v.allocationPct || 0), 0);

  const handleUnitChange = async (value: RandomizationUnit) => {
    setRandomizationUnit(value);
    setSaving(true);
    setSaveError(null);
    try {
      const updated = await updateExperimentDefinition(definition.id, { randomizationUnit: value });
      onSaved(updated);
    } catch (e) {
      setSaveError(e instanceof ApiError ? e.message : 'Could not save the randomization unit.');
      setRandomizationUnit(definition.randomizationUnit);
    } finally {
      setSaving(false);
    }
  };

  let treatmentIndex = 0;

  return (
    <Card className="border-black/10 shadow-none">
      <CardHeader className="space-y-0">
        <div className="flex items-center gap-2">
          <Shuffle className="h-4 w-4 text-black" />
          <div>
            <CardTitle className="text-[15px] tracking-tight">Assignment</CardTitle>
            <CardDescription>
              Expected allocation, per the Variants section — SRM checks the actual split at
              analysis time.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pt-0">
        <div className="max-w-xs">
          <Label className="text-xs font-medium text-neutral-500">Randomization unit</Label>
          <div className="mt-1.5 flex items-center gap-2">
            <Select
              value={randomizationUnit}
              onValueChange={(v) => handleUnitChange(v as RandomizationUnit)}
              disabled={saving}
            >
              <SelectTrigger className="h-8 border-black/10 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {UNIT_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value} className="text-sm">
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {saving && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-neutral-400" />}
          </div>
          {saveError && <p className="mt-1.5 text-xs text-red-600">{saveError}</p>}
        </div>

        <div>
          <Label className="text-xs font-medium text-neutral-500">Expected allocation</Label>
          {definition.variants.length === 0 ? (
            <p className="mt-1.5 text-sm text-neutral-400">
              No variants configured yet — add them in the Variants section above.
            </p>
          ) : (
            <div className="mt-2 space-y-2">
              <div className="flex h-3 w-full overflow-hidden rounded-full bg-neutral-100">
                {definition.variants.map((v) => {
                  const color = v.isControl
                    ? 'bg-neutral-400'
                    : SEGMENT_COLORS[treatmentIndex++ % SEGMENT_COLORS.length];
                  return (
                    <div
                      key={v.id}
                      className={cn('h-full', color)}
                      style={{ width: `${Math.max(v.allocationPct, 0)}%` }}
                      title={`${v.name}: ${v.allocationPct}%`}
                    />
                  );
                })}
              </div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-neutral-500">
                {definition.variants.map((v, i) => (
                  <span key={v.id} className="flex items-center gap-1.5">
                    <span
                      className={cn(
                        'h-2 w-2 shrink-0 rounded-full',
                        v.isControl
                          ? 'bg-neutral-400'
                          : SEGMENT_COLORS[
                              definition.variants.slice(0, i).filter((x) => !x.isControl).length %
                                SEGMENT_COLORS.length
                            ]
                      )}
                    />
                    {v.name} <span className="font-medium text-neutral-700">{v.allocationPct}%</span>
                  </span>
                ))}
                <span
                  className={cn(
                    'ml-auto font-medium',
                    Math.abs(totalAllocation - 100) <= 0.5 ? 'text-neutral-500' : 'text-red-600'
                  )}
                >
                  Total: {totalAllocation.toFixed(1)}%
                </span>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
