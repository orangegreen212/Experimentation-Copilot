'use client';

import { useEffect, useState } from 'react';
import { Cpu, Target, Gauge, Wand2, Check } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { getAvailableModels } from '@/lib/api';
import { getWorkspaceSettings, saveWorkspaceSettings, type WorkspaceDefaults } from '@/lib/workspace-settings';
import type { AvailableModel } from '@/lib/types';

const EMPTY_DEFAULTS: WorkspaceDefaults = {
  cuped: false,
  bootstrap: false,
};

/**
 * Workspace-level analysis defaults — NOT a duplicate of
 * ExperimentConfig (components/experiment-config.tsx), which stays the
 * per-run "applies to this evaluation only" panel on the Overview
 * screen. This screen sets what a brand-new experiment starts with;
 * `app/page.tsx` seeds its `settings` state from here on load, and the
 * per-run panel still lets you change anything for that one run.
 *
 * Scoped to fields that already have a real effect at analyze-time
 * (model / confidence level / statistical power / CUPED / bootstrap —
 * see WorkspaceDefaults' doc comment). Winsorization / Bonferroni /
 * a configurable testing method are NOT included: nothing in the
 * backend (AnalysisSettings, experiment_node.py) reads them yet, and
 * shipping a toggle with no effect would be actively misleading.
 */
export function SettingsView() {
  const [defaults, setDefaults] = useState<WorkspaceDefaults>(EMPTY_DEFAULTS);
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [modelsError, setModelsError] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<number | null>(null);

  useEffect(() => {
    getAvailableModels()
      .then((res) => setModels(res.models))
      .catch(() => setModelsError(true));

    getWorkspaceSettings()
      .then((saved) => {
        if (saved) setDefaults({ ...EMPTY_DEFAULTS, ...saved });
      })
      .catch(() => setLoadError(true))
      .finally(() => setLoading(false));
  }, []);

  const freeModels = models.filter((m) => m.id.endsWith(':free'));
  const update = (patch: Partial<WorkspaceDefaults>) => {
    setDefaults((prev) => ({ ...prev, ...patch }));
    setSavedAt(null);
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      await saveWorkspaceSettings(defaults);
      setSavedAt(Date.now());
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save settings.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <p className="text-xs text-neutral-400">Loading settings…</p>;
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <Card className="border-black/10 shadow-none">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Wand2 className="h-4 w-4 text-black" />
            <CardTitle className="text-[15px] tracking-tight">Settings</CardTitle>
          </div>
          <CardDescription>Configure default analysis behavior for your experiments.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-0">
          {/* AI Model */}
          <div className="flex items-start justify-between gap-4 py-3.5">
            <div className="min-w-0">
              <p className="flex items-center gap-1.5 text-[13px] font-medium text-black">
                <Cpu className="h-3.5 w-3.5" />
                AI Model
                <Badge variant="outline" className="border-black/10 text-[10px] font-normal text-neutral-500">
                  Free models only
                </Badge>
              </p>
              <p className="text-xs text-neutral-400">
                {modelsError ? 'Could not load the model list.' : 'Used unless overridden for a specific run.'}
              </p>
            </div>
            <div className="shrink-0">
              {!modelsError && freeModels.length > 0 ? (
                <Select
                  value={defaults.model && freeModels.some((m) => m.id === defaults.model) ? defaults.model : freeModels[0].id}
                  onValueChange={(v) => update({ model: v })}
                >
                  <SelectTrigger className="h-8 w-[260px] border-black/15 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {freeModels.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        {m.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <span className="text-xs text-neutral-400">Unavailable</span>
              )}
            </div>
          </div>
          <div className="h-px bg-black/10" />

          {/* Statistical defaults */}
          <div className="flex items-start justify-between gap-4 py-3.5">
            <div className="min-w-0">
              <p className="flex items-center gap-1.5 text-[13px] font-medium text-black">
                <Target className="h-3.5 w-3.5" />
                Confidence Level
              </p>
              <p className="text-xs text-neutral-400">Default significance threshold for new experiments.</p>
            </div>
            <Select
              value={defaults.confidenceLevel != null ? String(Math.round(defaults.confidenceLevel * 100)) : '__default__'}
              onValueChange={(v) => update({ confidenceLevel: v === '__default__' ? undefined : Number(v) / 100 })}
            >
              <SelectTrigger className="h-8 w-[140px] shrink-0 border-black/15 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__default__">App default (95%)</SelectItem>
                <SelectItem value="90">90%</SelectItem>
                <SelectItem value="95">95%</SelectItem>
                <SelectItem value="99">99%</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="h-px bg-black/10" />

          <div className="flex items-start justify-between gap-4 py-3.5">
            <div className="min-w-0">
              <p className="flex items-center gap-1.5 text-[13px] font-medium text-black">
                <Gauge className="h-3.5 w-3.5" />
                Statistical Power
              </p>
              <p className="text-xs text-neutral-400">Default target power for new experiments.</p>
            </div>
            <Select
              value={defaults.statisticalPower != null ? String(Math.round(defaults.statisticalPower * 100)) : '__default__'}
              onValueChange={(v) => update({ statisticalPower: v === '__default__' ? undefined : Number(v) / 100 })}
            >
              <SelectTrigger className="h-8 w-[140px] shrink-0 border-black/15 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__default__">App default (80%)</SelectItem>
                <SelectItem value="80">80%</SelectItem>
                <SelectItem value="90">90%</SelectItem>
                <SelectItem value="95">95%</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="h-px bg-black/10" />

          {/* Data processing */}
          <ToggleRow
            label="CUPED Variance Reduction"
            description="Use pre-experiment covariates to reduce variance, by default, for new experiments."
            checked={defaults.cuped}
            onChange={(v) => update({ cuped: v })}
          />
          <div className="h-px bg-black/10" />
          <ToggleRow
            label="Bootstrap Resampling"
            description="Non-parametric confidence intervals via 10,000 bootstrap iterations, by default."
            checked={defaults.bootstrap}
            onChange={(v) => update({ bootstrap: v })}
          />
        </CardContent>
      </Card>

      <div className="flex items-center gap-3">
        <Button onClick={handleSave} disabled={saving} className="h-8 text-xs">
          {saving ? 'Saving…' : 'Save defaults'}
        </Button>
        {savedAt && (
          <span className="flex items-center gap-1 text-xs text-emerald-600">
            <Check className="h-3.5 w-3.5" /> Saved
          </span>
        )}
        {saveError && <span className="text-xs text-red-600">{saveError}</span>}
        {loadError && <span className="text-xs text-neutral-400">Could not load previously saved defaults.</span>}
      </div>

      <p className="text-xs text-neutral-400">
        These settings are used as defaults when creating a new experiment. You can still change any of
        them for a single run in Experiment Configuration on the Overview screen.
      </p>
    </div>
  );
}

function ToggleRow({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-start justify-between gap-4 py-3.5">
      <div className="min-w-0">
        <p className="text-[13px] font-medium text-black">{label}</p>
        <p className="text-xs text-neutral-400">{description}</p>
      </div>
      <Switch checked={checked} onCheckedChange={onChange} className="mt-1 shrink-0" />
    </div>
  );
}
