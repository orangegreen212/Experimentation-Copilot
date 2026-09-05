'use client';

/**
 * Phase 7 — Metrics.
 *
 * Mirrors `ExperimentMetric` (schemas/experiment_definition.py) exactly
 * via lib/types.ts. Same architectural boundary as Variants/Targeting
 * (see experiment-variants-form.tsx's docstring): this is descriptive
 * planning metadata only. The statistical engine derives its own
 * primary/guardrail metrics straight from the dataset at analysis time
 * — this form is NOT a second, independent metric-selection mechanism,
 * it just lets the analyst state up front what they INTEND to measure,
 * grouped by role the same way the Stage 0 mockup lays it out:
 *
 *   PRIMARY / SECONDARY / GUARDRAILS
 *
 * Server-side validation (ExperimentDefinitionBase._validate_metrics)
 * only enforces "at most one PRIMARY metric" — this form enforces the
 * same invariant client-side so the analyst sees the problem before
 * Save, not after a 422.
 */

import { useEffect, useState } from 'react';
import { Loader2, Plus, Trash2, Target, ShieldAlert, ListChecks, AlertCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
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
import type { ExperimentDefinition, ExperimentMetric, MetricRole, PlanningMetricType } from '@/lib/types';

interface ExperimentMetricsFormProps {
  definition: ExperimentDefinition;
  onSaved: (updated: ExperimentDefinition) => void;
}

const METRIC_TYPE_OPTIONS: { value: PlanningMetricType; label: string }[] = [
  { value: 'binary', label: 'Conversion rate' },
  { value: 'continuous_monetary', label: 'Mean — monetary' },
  { value: 'continuous_general', label: 'Mean — general' },
];

const METRIC_TYPE_LABELS: Record<PlanningMetricType, string> = {
  binary: 'Conversion rate',
  continuous_monetary: 'Mean',
  continuous_general: 'Mean',
};

const ROLE_SECTIONS: { role: MetricRole; title: string; icon: typeof Target; description: string }[] = [
  { role: 'primary', title: 'Primary', icon: Target, description: 'The one metric the decision hinges on.' },
  { role: 'secondary', title: 'Secondary', icon: ListChecks, description: 'Supporting context, not decisive on its own.' },
  { role: 'guardrail', title: 'Guardrails', icon: ShieldAlert, description: 'Must not regress, whatever the primary result.' },
];

function emptyMetric(role: MetricRole): ExperimentMetric {
  return { name: '', role, type: 'binary', description: null, fieldDefinition: null };
}

function isDirty(draft: ExperimentMetric[], saved: ExperimentDefinition): boolean {
  return JSON.stringify(draft) !== JSON.stringify(saved.metrics);
}

export function ExperimentMetricsForm({ definition, onSaved }: ExperimentMetricsFormProps) {
  const [metrics, setMetrics] = useState<ExperimentMetric[]>(definition.metrics);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    setMetrics(definition.metrics);
    setSaveError(null);
  }, [definition.id, definition.updatedAt]);

  const dirty = isDirty(metrics, definition);

  const primaryCount = metrics.filter((m) => m.role === 'primary').length;
  const hasEmptyName = metrics.some((m) => !m.name.trim());
  const invalid = primaryCount > 1 || hasEmptyName;

  const updateMetric = (index: number, patch: Partial<ExperimentMetric>) => {
    setMetrics((prev) => prev.map((m, i) => (i === index ? { ...m, ...patch } : m)));
  };

  const addMetric = (role: MetricRole) => {
    setMetrics((prev) => [...prev, emptyMetric(role)]);
  };

  const removeMetric = (index: number) => {
    setMetrics((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSave = async () => {
    if (invalid) return;
    setSaving(true);
    setSaveError(null);
    try {
      const updated = await updateExperimentDefinition(definition.id, { metrics });
      onSaved(updated);
    } catch (e) {
      setSaveError(e instanceof ApiError ? e.message : 'Could not save the metrics.');
    } finally {
      setSaving(false);
    }
  };

  const handleDiscard = () => {
    setMetrics(definition.metrics);
    setSaveError(null);
  };

  return (
    <Card className="border-black/10 shadow-none">
      <CardHeader className="space-y-0">
        <div className="flex items-center gap-2">
          <Target className="h-4 w-4 text-black" />
          <div>
            <CardTitle className="text-[15px] tracking-tight">Metrics</CardTitle>
            <CardDescription>At most one metric may be marked Primary.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-5 pt-0">
        {ROLE_SECTIONS.map(({ role, title, icon: Icon, description }) => {
          const roleMetrics = metrics
            .map((m, i) => ({ m, i }))
            .filter(({ m }) => m.role === role);

          return (
            <div key={role} className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5">
                  <Icon className="h-3.5 w-3.5 text-neutral-400" />
                  <span className="text-[11px] font-semibold uppercase tracking-wide text-neutral-500">
                    {title}
                  </span>
                  <span className="text-xs text-neutral-400">— {description}</span>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 gap-1 px-2 text-xs text-neutral-500"
                  onClick={() => addMetric(role)}
                >
                  <Plus className="h-3.5 w-3.5" />
                  Add metric
                </Button>
              </div>

              {roleMetrics.length === 0 && (
                <div className="rounded-md border border-dashed border-black/15 bg-neutral-50/60 py-4 text-center text-xs text-neutral-400">
                  No {title.toLowerCase()} metrics yet.
                </div>
              )}

              <div className="space-y-2">
                {roleMetrics.map(({ m, i }) => (
                  <div
                    key={i}
                    className={cn(
                      'space-y-3 rounded-lg border p-3',
                      role === 'primary' ? 'border-indigo-200 bg-indigo-50/40' : 'border-black/10'
                    )}
                  >
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-[2fr_1fr_auto]">
                      <div>
                        <Label className="text-xs font-medium text-neutral-500">Name</Label>
                        <Input
                          value={m.name}
                          onChange={(e) => updateMetric(i, { name: e.target.value })}
                          placeholder="e.g. Signup Conversion"
                          className="mt-1 h-8 border-black/10 text-sm placeholder:text-neutral-400"
                        />
                      </div>
                      <div>
                        <Label className="text-xs font-medium text-neutral-500">Type</Label>
                        <Select
                          value={m.type}
                          onValueChange={(v) => updateMetric(i, { type: v as PlanningMetricType })}
                        >
                          <SelectTrigger className="mt-1 h-8 border-black/10 text-xs">
                            <SelectValue>{METRIC_TYPE_LABELS[m.type]}</SelectValue>
                          </SelectTrigger>
                          <SelectContent>
                            {METRIC_TYPE_OPTIONS.map((opt) => (
                              <SelectItem key={opt.value} value={opt.value} className="text-xs">
                                {opt.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="flex items-end justify-end">
                        <button
                          type="button"
                          onClick={() => removeMetric(i)}
                          title="Remove metric"
                          className="rounded-md p-1.5 text-neutral-300 transition-colors hover:bg-red-50 hover:text-red-600"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      <div>
                        <Label className="text-xs font-medium text-neutral-500">
                          Description <span className="font-normal text-neutral-400">(optional)</span>
                        </Label>
                        <Input
                          value={m.description ?? ''}
                          onChange={(e) =>
                            updateMetric(i, { description: e.target.value === '' ? null : e.target.value })
                          }
                          placeholder="What this measures"
                          className="mt-1 h-8 border-black/10 text-sm placeholder:text-neutral-400"
                        />
                      </div>
                      <div>
                        <Label className="text-xs font-medium text-neutral-500">
                          Definition <span className="font-normal text-neutral-400">(optional)</span>
                        </Label>
                        <Input
                          value={m.fieldDefinition ?? ''}
                          onChange={(e) =>
                            updateMetric(i, { fieldDefinition: e.target.value === '' ? null : e.target.value })
                          }
                          placeholder="e.g. signup_event"
                          className="mt-1 h-8 border-black/10 text-sm placeholder:text-neutral-400"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}

        {primaryCount > 1 && (
          <div className="flex items-center gap-2 border-t border-black/5 pt-3 text-xs text-red-600">
            <AlertCircle className="h-3.5 w-3.5" />
            At most one metric may be marked Primary (currently {primaryCount}).
          </div>
        )}

        <div className="flex items-center justify-end gap-2 pt-1">
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
            Save metrics
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
