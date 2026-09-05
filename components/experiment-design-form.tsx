'use client';

/**
 * Phase 3 — Experiment Design.
 *
 * Edits the planning fields the Phase 2 "New Experiment" dialog
 * deliberately left out: problem statement, objective, and a list of
 * hypotheses (exactly one PRIMARY, any number of SECONDARY) — see
 * experiment-library.tsx's dialog docstring ("Problem/objective/
 * hypotheses/variants/targeting/metrics are edited via the Experiment
 * Design form in a later phase" — this is that later phase).
 *
 * Deliberately separate from HypothesisForm (hypothesis-form.tsx):
 * that form captures ONE hypothesis scoped to an already-loaded
 * dataset (primary metric is a dropdown of DatasetInfo.availableMetrics)
 * for a single /experiments/analyze run. This form edits a LIST of
 * hypotheses on an ExperimentDefinition that may not have a dataset
 * attached at all yet (Phase 8, Data Source, comes later) — so primary
 * metric here is free text, same reasoning ExperimentSetup
 * (experiment-setup.tsx) already uses for its pre-dataset metric field.
 *
 * Reuses the `Hypothesis` shape unmodified (schemas/hypothesis.py via
 * lib/types.ts) inside each `RoledHypothesis` — only `role` is new
 * here. Saves via PATCH /experiment-definitions/{id}
 * (updateExperimentDefinition), same partial-update contract the
 * status dropdown in experiment-library.tsx already uses.
 */

import { useEffect, useState } from 'react';
import { Loader2, Plus, Trash2, Target, ListChecks } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { updateExperimentDefinition, ApiError } from '@/lib/api';
import type {
  ExperimentDefinition,
  ExpectedDirection,
  Hypothesis,
  HypothesisRole,
  RoledHypothesis,
} from '@/lib/types';

interface ExperimentDesignFormProps {
  definition: ExperimentDefinition;
  /** Called with the freshly-saved record after a successful save, so
   *  the caller (ExperimentLibrary) can update its own state. */
  onSaved: (updated: ExperimentDefinition) => void;
}

const DIRECTION_OPTIONS: { value: ExpectedDirection; label: string }[] = [
  { value: 'increase', label: 'Increase' },
  { value: 'decrease', label: 'Decrease' },
  { value: 'no_change', label: 'No change' },
];

function emptyHypothesis(): Hypothesis {
  return {
    statement: '',
    primaryMetric: '',
    expectedDirection: 'increase',
    expectedEffectRelative: null,
    rationale: null,
  };
}

/** Fraction -> percent for the input box (see hypothesis-form.tsx for
 *  why this rounds rather than doing a plain `value * 100`). */
function relativeToPercentInput(value: number): number {
  return Math.round(value * 100 * 1e8) / 1e8;
}

function isDirty(
  draft: { problemStatement: string; objective: string; hypotheses: RoledHypothesis[] },
  saved: ExperimentDefinition
): boolean {
  return (
    draft.problemStatement !== (saved.problemStatement ?? '') ||
    draft.objective !== (saved.objective ?? '') ||
    JSON.stringify(draft.hypotheses) !== JSON.stringify(saved.hypotheses)
  );
}

export function ExperimentDesignForm({ definition, onSaved }: ExperimentDesignFormProps) {
  const [problemStatement, setProblemStatement] = useState(definition.problemStatement ?? '');
  const [objective, setObjective] = useState(definition.objective ?? '');
  const [hypotheses, setHypotheses] = useState<RoledHypothesis[]>(definition.hypotheses);

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Re-sync local drafts whenever a different definition is selected
  // (or after a save round-trips a fresh record back in) — otherwise
  // switching definitions in the library list would keep showing the
  // previous one's in-progress edits.
  useEffect(() => {
    setProblemStatement(definition.problemStatement ?? '');
    setObjective(definition.objective ?? '');
    setHypotheses(definition.hypotheses);
    setSaveError(null);
  }, [definition.id, definition.updatedAt]);

  const dirty = isDirty({ problemStatement, objective, hypotheses }, definition);

  const primaryCount = hypotheses.filter((h) => h.role === 'primary').length;
  const hasEmptyStatement = hypotheses.some((h) => !h.hypothesis.statement.trim());
  const hasEmptyMetric = hypotheses.some((h) => !h.hypothesis.primaryMetric.trim());
  const invalid =
    hypotheses.length > 0 && (primaryCount !== 1 || hasEmptyStatement || hasEmptyMetric);

  const updateHypothesis = (index: number, patch: Partial<Hypothesis>) => {
    setHypotheses((prev) =>
      prev.map((h, i) => (i === index ? { ...h, hypothesis: { ...h.hypothesis, ...patch } } : h))
    );
  };

  const setRole = (index: number, role: HypothesisRole) => {
    setHypotheses((prev) =>
      prev.map((h, i) => {
        if (i === index) return { ...h, role };
        // Exactly one PRIMARY, enforced client-side the same way the
        // backend validator requires — promoting one row to PRIMARY
        // demotes any other PRIMARY row rather than leaving an invalid
        // two-primary state for the analyst to fix by hand.
        if (role === 'primary' && h.role === 'primary') return { ...h, role: 'secondary' };
        return h;
      })
    );
  };

  const addHypothesis = () => {
    setHypotheses((prev) => [
      ...prev,
      { role: prev.length === 0 ? 'primary' : 'secondary', hypothesis: emptyHypothesis() },
    ]);
  };

  const removeHypothesis = (index: number) => {
    setHypotheses((prev) => {
      const removed = prev[index];
      const next = prev.filter((_, i) => i !== index);
      // If the removed row was PRIMARY and others remain, promote the
      // first remaining one so the list never sits without a PRIMARY
      // while the analyst decides what to do next.
      if (removed?.role === 'primary' && next.length > 0 && !next.some((h) => h.role === 'primary')) {
        next[0] = { ...next[0], role: 'primary' };
      }
      return next;
    });
  };

  const handleSave = async () => {
    if (invalid) return;
    setSaving(true);
    setSaveError(null);
    try {
      const updated = await updateExperimentDefinition(definition.id, {
        problemStatement: problemStatement.trim() || null,
        objective: objective.trim() || null,
        hypotheses,
      });
      onSaved(updated);
    } catch (e) {
      setSaveError(e instanceof ApiError ? e.message : 'Could not save the experiment design.');
    } finally {
      setSaving(false);
    }
  };

  const handleDiscard = () => {
    setProblemStatement(definition.problemStatement ?? '');
    setObjective(definition.objective ?? '');
    setHypotheses(definition.hypotheses);
    setSaveError(null);
  };

  return (
    <div className="space-y-4">
      <Card className="border-black/10 shadow-none">
        <CardContent className="space-y-4 py-5">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <Label htmlFor="design-problem" className="text-[13px] font-medium text-black">
                Problem statement
              </Label>
              <Textarea
                id="design-problem"
                value={problemStatement}
                onChange={(e) => setProblemStatement(e.target.value)}
                placeholder="What problem are we trying to solve?"
                className="mt-1.5 min-h-[80px] resize-none border-black/10 placeholder:text-neutral-400"
                rows={3}
                maxLength={2000}
              />
            </div>
            <div>
              <Label htmlFor="design-objective" className="text-[13px] font-medium text-black">
                Objective
              </Label>
              <Textarea
                id="design-objective"
                value={objective}
                onChange={(e) => setObjective(e.target.value)}
                placeholder="What do we want to improve?"
                className="mt-1.5 min-h-[80px] resize-none border-black/10 placeholder:text-neutral-400"
                rows={3}
                maxLength={2000}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="border-black/10 shadow-none">
        <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4 text-black" />
            <div>
              <CardTitle className="text-[15px] tracking-tight">Hypotheses</CardTitle>
              <CardDescription>
                Exactly one PRIMARY hypothesis; add as many SECONDARY as you need.
              </CardDescription>
            </div>
          </div>
          <Button size="sm" variant="outline" className="gap-1.5" onClick={addHypothesis}>
            <Plus className="h-3.5 w-3.5" />
            Add hypothesis
          </Button>
        </CardHeader>
        <CardContent className="space-y-3 pt-0">
          {hypotheses.length === 0 && (
            <div className="flex items-center gap-2 rounded-md border border-dashed border-black/15 bg-neutral-50/60 py-8 text-center text-[13px] text-neutral-400">
              <ListChecks className="mx-auto h-4 w-4 text-neutral-300" />
              <span className="mx-auto">
                No hypotheses yet — add one to start defining what you expect this experiment to do.
              </span>
            </div>
          )}

          {hypotheses.map((h, index) => (
            <div
              key={index}
              className={cn(
                'space-y-3 rounded-lg border p-4',
                h.role === 'primary' ? 'border-indigo-200 bg-indigo-50/30' : 'border-black/10'
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className={cn(
                      'text-[10px] uppercase tracking-wide',
                      h.role === 'primary'
                        ? 'border-indigo-200 bg-indigo-50 text-indigo-700'
                        : 'border-neutral-200 bg-neutral-50 text-neutral-500'
                    )}
                  >
                    {h.role === 'primary' ? 'Primary' : 'Secondary'}
                  </Badge>
                  <span className="text-xs text-neutral-400">H{index + 1}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Select value={h.role} onValueChange={(v) => setRole(index, v as HypothesisRole)}>
                    <SelectTrigger className="h-7 w-[110px] text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="primary" className="text-xs">
                        Primary
                      </SelectItem>
                      <SelectItem value="secondary" className="text-xs">
                        Secondary
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <button
                    type="button"
                    onClick={() => removeHypothesis(index)}
                    title="Remove hypothesis"
                    className="rounded-md p-1.5 text-neutral-300 transition-colors hover:bg-red-50 hover:text-red-600"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              <Textarea
                value={h.hypothesis.statement}
                onChange={(e) => updateHypothesis(index, { statement: e.target.value })}
                placeholder="e.g. Redesigning the landing page will increase signup conversion."
                className="min-h-[54px] resize-none border-black/10 text-sm placeholder:text-neutral-400"
                rows={2}
                maxLength={500}
              />

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs font-medium text-neutral-500">Primary metric</Label>
                  <Input
                    value={h.hypothesis.primaryMetric}
                    onChange={(e) => updateHypothesis(index, { primaryMetric: e.target.value })}
                    placeholder="e.g. Signup Conversion"
                    className="mt-1 h-8 border-black/10 text-sm placeholder:text-neutral-400"
                  />
                </div>
                <div>
                  <Label className="text-xs font-medium text-neutral-500">Expected direction</Label>
                  <Select
                    value={h.hypothesis.expectedDirection}
                    onValueChange={(v) => updateHypothesis(index, { expectedDirection: v as ExpectedDirection })}
                  >
                    <SelectTrigger className="mt-1 h-8 border-black/10 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {DIRECTION_OPTIONS.map((opt) => (
                        <SelectItem key={opt.value} value={opt.value} className="text-sm">
                          {opt.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs font-medium text-neutral-500">
                    Expected effect{' '}
                    <span className="font-normal text-neutral-400">(optional, % relative)</span>
                  </Label>
                  <Input
                    type="number"
                    inputMode="decimal"
                    step="0.1"
                    min={0}
                    value={
                      h.hypothesis.expectedEffectRelative === null ||
                      h.hypothesis.expectedEffectRelative === undefined
                        ? ''
                        : relativeToPercentInput(h.hypothesis.expectedEffectRelative)
                    }
                    onChange={(e) => {
                      const raw = e.target.value;
                      updateHypothesis(index, {
                        expectedEffectRelative: raw === '' ? null : Number(raw) / 100,
                      });
                    }}
                    placeholder="5"
                    className="mt-1 h-8 border-black/10 text-sm placeholder:text-neutral-400"
                  />
                </div>
                <div>
                  <Label className="text-xs font-medium text-neutral-500">
                    Rationale <span className="font-normal text-neutral-400">(optional)</span>
                  </Label>
                  <Input
                    value={h.hypothesis.rationale ?? ''}
                    onChange={(e) =>
                      updateHypothesis(index, { rationale: e.target.value === '' ? null : e.target.value })
                    }
                    placeholder="Why do you expect this effect?"
                    className="mt-1 h-8 border-black/10 text-sm placeholder:text-neutral-400"
                    maxLength={2000}
                  />
                </div>
              </div>
            </div>
          ))}

          {hypotheses.length > 0 && primaryCount !== 1 && (
            <p className="text-xs text-red-600">
              Exactly one hypothesis must be marked Primary (currently {primaryCount}).
            </p>
          )}
        </CardContent>
      </Card>

      <div className="flex items-center justify-end gap-2">
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
          Save design
        </Button>
      </div>
    </div>
  );
}
