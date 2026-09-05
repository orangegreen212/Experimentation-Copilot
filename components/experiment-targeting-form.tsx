'use client';

/**
 * Phase 5 — Targeting.
 *
 * Same "не настоящий production targeting" boundary as Variants
 * (Stage 0 architecture doc §5): every field here is descriptive
 * planning metadata on the `ExperimentDefinition`, never enforced
 * against real traffic. Mirrors `Targeting`
 * (schemas/experiment_definition.py) exactly via lib/types.ts.
 *
 * List fields (countries/platforms/devices) use the same
 * add-on-Enter chip pattern as ExperimentSetup's guardrail-metric
 * input (experiment-setup.tsx) for a consistent feel across the two
 * "planning" surfaces.
 */

import { useEffect, useState } from 'react';
import { Loader2, Plus, X, Crosshair } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { updateExperimentDefinition, ApiError } from '@/lib/api';
import type { ExperimentDefinition, Targeting } from '@/lib/types';

interface ExperimentTargetingFormProps {
  definition: ExperimentDefinition;
  onSaved: (updated: ExperimentDefinition) => void;
}

function emptyTargeting(): Targeting {
  return {
    countries: [],
    platforms: [],
    devices: [],
    userType: null,
    acquisitionChannel: null,
    userSegment: null,
    trafficAllocationPct: null,
  };
}

function isDirty(draft: Targeting, saved: ExperimentDefinition): boolean {
  return JSON.stringify(draft) !== JSON.stringify(saved.targeting ?? emptyTargeting());
}

/** Small reusable add-on-Enter chip list, shared by countries/platforms/devices. */
function ChipListField({
  label,
  placeholder,
  values,
  onChange,
}: {
  label: string;
  placeholder: string;
  values: string[];
  onChange: (next: string[]) => void;
}) {
  const [draft, setDraft] = useState('');

  const add = () => {
    const trimmed = draft.trim();
    if (trimmed && !values.includes(trimmed)) {
      onChange([...values, trimmed]);
    }
    setDraft('');
  };

  const remove = (v: string) => onChange(values.filter((x) => x !== v));

  return (
    <div>
      <Label className="text-xs font-medium text-neutral-500">{label}</Label>
      <div className="mt-1.5 flex gap-2">
        <Input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              add();
            }
          }}
          placeholder={placeholder}
          className="h-8 border-black/10 text-sm placeholder:text-neutral-400"
          maxLength={100}
        />
        <Button type="button" variant="outline" size="sm" onClick={add} className="h-8 shrink-0 border-black/15 px-2">
          <Plus className="h-3.5 w-3.5" />
        </Button>
      </div>
      {values.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {values.map((v) => (
            <span
              key={v}
              className="flex items-center gap-1 rounded-full border border-black/10 bg-neutral-50 px-2.5 py-1 text-xs text-black"
            >
              {v}
              <button type="button" onClick={() => remove(v)} aria-label={`Remove ${v}`}>
                <X className="h-3 w-3 text-neutral-400 hover:text-black" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export function ExperimentTargetingForm({ definition, onSaved }: ExperimentTargetingFormProps) {
  const [targeting, setTargeting] = useState<Targeting>(definition.targeting ?? emptyTargeting());
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    setTargeting(definition.targeting ?? emptyTargeting());
    setSaveError(null);
  }, [definition.id, definition.updatedAt]);

  const dirty = isDirty(targeting, definition);
  const invalid =
    targeting.trafficAllocationPct !== null &&
    targeting.trafficAllocationPct !== undefined &&
    (targeting.trafficAllocationPct < 0 || targeting.trafficAllocationPct > 100);

  const patch = (p: Partial<Targeting>) => setTargeting((prev) => ({ ...prev, ...p }));

  const handleSave = async () => {
    if (invalid) return;
    setSaving(true);
    setSaveError(null);
    try {
      const updated = await updateExperimentDefinition(definition.id, { targeting });
      onSaved(updated);
    } catch (e) {
      setSaveError(e instanceof ApiError ? e.message : 'Could not save targeting.');
    } finally {
      setSaving(false);
    }
  };

  const handleDiscard = () => {
    setTargeting(definition.targeting ?? emptyTargeting());
    setSaveError(null);
  };

  return (
    <Card className="border-black/10 shadow-none">
      <CardHeader className="space-y-0">
        <div className="flex items-center gap-2">
          <Crosshair className="h-4 w-4 text-black" />
          <div>
            <CardTitle className="text-[15px] tracking-tight">Target audience</CardTitle>
            <CardDescription>Configuration metadata, not a production targeting engine.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pt-0">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <ChipListField
            label="Country"
            placeholder="e.g. US — press Enter"
            values={targeting.countries}
            onChange={(v) => patch({ countries: v })}
          />
          <ChipListField
            label="Platform"
            placeholder="e.g. Web — press Enter"
            values={targeting.platforms}
            onChange={(v) => patch({ platforms: v })}
          />
          <ChipListField
            label="Device"
            placeholder="e.g. Desktop — press Enter"
            values={targeting.devices}
            onChange={(v) => patch({ devices: v })}
          />
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <Label className="text-xs font-medium text-neutral-500">User type</Label>
            <Input
              value={targeting.userType ?? ''}
              onChange={(e) => patch({ userType: e.target.value === '' ? null : e.target.value })}
              placeholder="e.g. New users"
              className="mt-1.5 h-8 border-black/10 text-sm placeholder:text-neutral-400"
              maxLength={100}
            />
          </div>
          <div>
            <Label className="text-xs font-medium text-neutral-500">Acquisition channel</Label>
            <Input
              value={targeting.acquisitionChannel ?? ''}
              onChange={(e) => patch({ acquisitionChannel: e.target.value === '' ? null : e.target.value })}
              placeholder="e.g. Paid Search"
              className="mt-1.5 h-8 border-black/10 text-sm placeholder:text-neutral-400"
              maxLength={100}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <Label className="text-xs font-medium text-neutral-500">
              User segment <span className="font-normal text-neutral-400">(optional)</span>
            </Label>
            <Input
              value={targeting.userSegment ?? ''}
              onChange={(e) => patch({ userSegment: e.target.value === '' ? null : e.target.value })}
              placeholder="e.g. High-value customers"
              className="mt-1.5 h-8 border-black/10 text-sm placeholder:text-neutral-400"
              maxLength={100}
            />
          </div>
          <div>
            <Label className="text-xs font-medium text-neutral-500">Traffic allocation %</Label>
            <Input
              type="number"
              inputMode="decimal"
              step="0.1"
              min={0}
              max={100}
              value={targeting.trafficAllocationPct ?? ''}
              onChange={(e) =>
                patch({
                  trafficAllocationPct: e.target.value === '' ? null : Number(e.target.value),
                })
              }
              placeholder="e.g. 20"
              className="mt-1.5 h-8 border-black/10 text-sm placeholder:text-neutral-400"
            />
          </div>
        </div>

        {invalid && (
          <p className="text-xs text-red-600">Traffic allocation must be between 0 and 100.</p>
        )}

        <div className="flex items-center justify-end gap-2 border-t border-black/5 pt-3">
          {saveError && <p className="mr-auto text-xs text-red-600">{saveError}</p>}
          {dirty && !saving && (
            <Button size="sm" variant="ghost" onClick={handleDiscard} className="text-neutral-500">
              Discard changes
            </Button>
          )}
          <Button
            size="sm"
            onClick={handleSave}
            disabled={!dirty || saving || invalid}
            className="gap-1.5 bg-indigo-600 text-white hover:bg-indigo-700"
          >
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            Save targeting
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
