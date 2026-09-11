'use client';

/**
 * Phase 4 — Variants.
 *
 * Deliberately simple (Stage 0 architecture doc §"Phase 4 — Variants":
 * "делаем просто и красиво, без попытки стать Optimizely") — this is
 * NOT a real feature-flag/randomization service, just descriptive
 * planning metadata on the `ExperimentDefinition` (same architectural
 * boundary as ExperimentDesignForm's hypotheses — see that file's
 * docstring and schemas/experiment_definition.py's module docstring).
 * The existing SRM check (app/stats/srm.py) is what actually verifies
 * real observed traffic split matches an expectation, at analysis
 * time — this form only captures what that expectation *is*.
 *
 * Mirrors `Variant` (schemas/experiment_definition.py) exactly via
 * lib/types.ts. Server-side validation
 * (ExperimentDefinitionBase._validate_variants) requires, once any
 * variants exist: exactly one Control, allocations summing to ~100%,
 * unique ids — this form enforces the same invariants client-side so
 * the analyst sees the problem before hitting Save, not after a 422.
 */

import { useEffect, useState } from 'react';
import { Loader2, Plus, Trash2, GitBranch, CheckCircle2, AlertCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';
import { updateExperimentDefinition, ApiError } from '@/lib/api';
import type { ExperimentDefinition, Variant } from '@/lib/types';

interface ExperimentVariantsFormProps {
  definition: ExperimentDefinition;
  onSaved: (updated: ExperimentDefinition) => void;
}

const ALLOCATION_TOLERANCE_PCT = 0.5; // matches backend's _ALLOCATION_TOLERANCE_PCT

function makeId(): string {
  // Matches the shape of the backend's default_factory (uuid4().hex[:8])
  // closely enough for a client-generated draft id — the server
  // re-validates uniqueness regardless.
  return Math.random().toString(16).slice(2, 10);
}

function emptyVariant(isControl: boolean): Variant {
  return {
    id: makeId(),
    name: isControl ? 'Control' : '',
    description: null,
    isControl,
    allocationPct: 0,
  };
}

/** Splits 100% evenly across `n` variants, giving any remainder to the
 *  first one so the total is always exactly 100 (never 99.99...). */
function evenSplit(n: number): number[] {
  if (n <= 0) return [];
  const base = Math.floor((100 / n) * 100) / 100;
  const amounts = new Array(n).fill(base);
  const remainder = Math.round((100 - base * n) * 100) / 100;
  amounts[0] = Math.round((amounts[0] + remainder) * 100) / 100;
  return amounts;
}

function isDirty(draft: Variant[], saved: ExperimentDefinition): boolean {
  return JSON.stringify(draft) !== JSON.stringify(saved.variants);
}

export function ExperimentVariantsForm({ definition, onSaved }: ExperimentVariantsFormProps) {
  const [variants, setVariants] = useState<Variant[]>(definition.variants);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    setVariants(definition.variants);
    setSaveError(null);
  }, [definition.id, definition.updatedAt]);

  const dirty = isDirty(variants, definition);

  const controlCount = variants.filter((v) => v.isControl).length;
  const totalAllocation = variants.reduce((sum, v) => sum + (v.allocationPct || 0), 0);
  const allocationOk = Math.abs(totalAllocation - 100) <= ALLOCATION_TOLERANCE_PCT;
  const hasEmptyName = variants.some((v) => !v.name.trim());
  const invalid =
    variants.length > 0 && (controlCount !== 1 || !allocationOk || hasEmptyName);

  const updateVariant = (index: number, patch: Partial<Variant>) => {
    setVariants((prev) => prev.map((v, i) => (i === index ? { ...v, ...patch } : v)));
  };

  const setControl = (index: number) => {
    setVariants((prev) => prev.map((v, i) => ({ ...v, isControl: i === index })));
  };

  const addVariant = () => {
    setVariants((prev) => {
      const next = [...prev, emptyVariant(prev.length === 0)];
      // Re-balance allocation evenly across every variant whenever one
      // is added — matches the doc mockup's "Add variant" behavior
      // (50/25/25 style even splits) rather than leaving the new arm
      // at 0% for the analyst to fix by hand every time.
      const splits = evenSplit(next.length);
      return next.map((v, i) => ({ ...v, allocationPct: splits[i] }));
    });
  };

  const removeVariant = (index: number) => {
    setVariants((prev) => {
      const removed = prev[index];
      let next = prev.filter((_, i) => i !== index);
      if (removed?.isControl && next.length > 0 && !next.some((v) => v.isControl)) {
        next = next.map((v, i) => (i === 0 ? { ...v, isControl: true } : v));
      }
      return next;
    });
  };

  const rebalanceEvenly = () => {
    setVariants((prev) => {
      const splits = evenSplit(prev.length);
      return prev.map((v, i) => ({ ...v, allocationPct: splits[i] }));
    });
  };

  const handleSave = async () => {
    if (invalid) return;
    setSaving(true);
    setSaveError(null);
    try {
      const updated = await updateExperimentDefinition(definition.id, { variants });
      onSaved(updated);
    } catch (e) {
      setSaveError(e instanceof ApiError ? e.message : 'Could not save the variants.');
    } finally {
      setSaving(false);
    }
  };

  const handleDiscard = () => {
    setVariants(definition.variants);
    setSaveError(null);
  };

  return (
    <Card className="border-border shadow-none">
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <div className="flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-foreground" />
          <div>
            <CardTitle className="text-[15px] tracking-tight">Variants</CardTitle>
            <CardDescription>Exactly one Control; allocations must sum to 100%.</CardDescription>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {variants.length > 1 && (
            <Button size="sm" variant="ghost" className="text-muted-foreground" onClick={rebalanceEvenly}>
              Split evenly
            </Button>
          )}
          <Button size="sm" variant="outline" className="gap-1.5" onClick={addVariant}>
            <Plus className="h-3.5 w-3.5" />
            Add variant
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {variants.length === 0 && (
          <div className="rounded-md border border-dashed border-border-strong bg-secondary/60 py-8 text-center text-[13px] text-muted-foreground">
            No variants yet — add a Control and at least one Treatment.
          </div>
        )}

        {variants.map((v, index) => (
          <div
            key={v.id}
            className={cn(
              'space-y-3 rounded-lg border p-4',
              v.isControl ? 'border-border-strong bg-secondary' : 'border-border'
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setControl(index)}
                  title="Mark as Control"
                  className={cn(
                    'flex h-4 w-4 shrink-0 items-center justify-center rounded-full border transition-colors',
                    v.isControl ? 'border-primary bg-primary' : 'border-border-strong hover:border-primary/50'
                  )}
                >
                  {v.isControl && <span className="h-1.5 w-1.5 rounded-full bg-surface" />}
                </button>
                <Badge
                  variant="outline"
                  className={cn(
                    'text-[10px] uppercase tracking-wide',
                    v.isControl
                      ? 'border-border-strong bg-secondary text-muted-foreground'
                      : 'border-primary/25 bg-accent text-primary'
                  )}
                >
                  {v.isControl ? 'Control' : 'Treatment'}
                </Badge>
              </div>
              <button
                type="button"
                onClick={() => removeVariant(index)}
                title="Remove variant"
                className="rounded-md p-1.5 text-muted-foreground/50 transition-colors hover:bg-destructive/[0.08] hover:text-destructive"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-[2fr_1fr]">
              <div>
                <Label className="text-xs font-medium text-muted-foreground">Name</Label>
                <Input
                  value={v.name}
                  onChange={(e) => updateVariant(index, { name: e.target.value })}
                  placeholder={v.isControl ? 'Control' : 'e.g. Treatment A'}
                  className={cn(
                    'mt-1 h-8 text-sm placeholder:text-muted-foreground',
                    !v.name.trim() ? 'border-destructive focus-visible:ring-destructive' : 'border-border'
                  )}
                />
                {!v.name.trim() && (
                  <p className="mt-1 text-[11px] text-destructive">Name is required.</p>
                )}
              </div>
              <div>
                <Label className="text-xs font-medium text-muted-foreground">Allocation %</Label>
                <Input
                  type="number"
                  inputMode="decimal"
                  step="0.1"
                  min={0}
                  max={100}
                  value={v.allocationPct}
                  onChange={(e) =>
                    updateVariant(index, { allocationPct: e.target.value === '' ? 0 : Number(e.target.value) })
                  }
                  className="mt-1 h-8 border-border text-sm"
                />
              </div>
            </div>

            <div>
              <Label className="text-xs font-medium text-muted-foreground">
                Description <span className="font-normal text-muted-foreground">(optional)</span>
              </Label>
              <Input
                value={v.description ?? ''}
                onChange={(e) => updateVariant(index, { description: e.target.value === '' ? null : e.target.value })}
                placeholder="e.g. New landing page + CTA"
                className="mt-1 h-8 border-border text-sm placeholder:text-muted-foreground"
              />
            </div>
          </div>
        ))}

        {variants.length > 0 && (
          <div className="flex items-center gap-2 border-t border-border pt-3 text-xs">
            {allocationOk ? (
              <CheckCircle2 className="h-3.5 w-3.5 text-success" />
            ) : (
              <AlertCircle className="h-3.5 w-3.5 text-destructive" />
            )}
            <span className={cn('font-medium', allocationOk ? 'text-muted-foreground' : 'text-destructive')}>
              Total allocation: {totalAllocation.toFixed(1)}%
            </span>
            {controlCount !== 1 && (
              <span className="text-destructive">
                — exactly one variant must be marked Control (currently {controlCount})
              </span>
            )}
            {hasEmptyName && (
              <span className="text-destructive">
                — every variant needs a name
                {(() => {
                  const missing = variants
                    .map((v, i) => (!v.name.trim() ? (v.isControl ? 'Control' : `#${i + 1}`) : null))
                    .filter(Boolean);
                  return missing.length ? ` (missing: ${missing.join(', ')})` : '';
                })()}
              </span>
            )}
          </div>
        )}

        <div className="flex items-center justify-end gap-2 pt-1">
          {saveError && <p className="mr-auto text-xs text-destructive">{saveError}</p>}
          {dirty && !saving && (
            <Button size="sm" variant="ghost" onClick={handleDiscard} className="text-muted-foreground">
              Discard changes
            </Button>
          )}
          <Button
            size="sm"
            onClick={handleSave}
            disabled={!dirty || saving || invalid}
            className="gap-1.5 bg-primary text-white hover:bg-primary/90"
          >
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            Save variants
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
