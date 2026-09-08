'use client';

import { useEffect, useState } from 'react';
import { Cpu, Wand2, Target, Gauge } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { getSystemInfo, getAvailableModels, ApiError } from '@/lib/api';
import type { AvailableModel, Settings, SystemInfo } from '@/lib/types';

interface ExperimentConfigProps {
  settings: Settings;
  onChange: (s: Settings) => void;
}

/**
 * Experiment-specific configuration, shown inline in the New Experiment
 * workflow (not a standalone Settings section — see sidebar.tsx).
 *
 * The LLM model IS now user-selectable at runtime, from a fixed,
 * server-curated list (GET /system/models — see AppSettings.
 * available_llm_models on the backend). This replaces the earlier
 * read-only display: switching backend .env models required a
 * redeploy, which made it painful to work around a rate-limited or
 * unavailable paid model — the dropdown lets an analyst pick one of
 * several free OpenRouter models (or the backend default) per run,
 * with no redeploy needed. The backend re-validates whatever is sent
 * here against the same allowlist, so this control can never send an
 * arbitrary model string.
 *
 * NOTE: a per-session cost limit was considered but dropped — the
 * backend does not currently compute or return a real per-run cost, so
 * a cost-limit control here would be a UI element with no actual
 * effect. Add it back once /experiments/analyze returns real cost data.
 */
export function ExperimentConfig({ settings, onChange }: ExperimentConfigProps) {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [infoError, setInfoError] = useState(false);
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [defaultModel, setDefaultModel] = useState<string | null>(null);
  const [modelsError, setModelsError] = useState(false);

  useEffect(() => {
    getSystemInfo()
      .then(setSystemInfo)
      .catch(() => setInfoError(true));

    getAvailableModels()
      .then((res) => {
        setModels(res.models);
        setDefaultModel(res.defaultModel);
      })
      .catch(() => setModelsError(true));
  }, []);

  // Free-only: the backend's curated list mixes paid (reliable) and
  // free (OpenRouter ":free" suffix, may queue/fail under load) models
  // — see config.py's available_llm_models comment on why paid ones
  // are listed at all. This UI is scoped to only ever offer/send a
  // free model, so paid entries (including the backend's own default,
  // which is paid) are filtered out entirely rather than merely
  // reordered.
  const freeModels = models.filter((m) => m.id.endsWith(':free'));

  const update = (patch: Partial<Settings>) => onChange({ ...settings, ...patch });

  // With no "use the backend default" option left (the default is
  // paid), a free model must be explicitly selected as soon as the
  // list loads — otherwise an unset `settings.model` would fall
  // through to the backend's paid default at analyze-time, silently
  // defeating the whole point of this filter.
  useEffect(() => {
    if (!settings.model && freeModels.length > 0) {
      update({ model: freeModels[0].id });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [freeModels.length]);

  return (
    <Card className="border-border shadow-none">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Wand2 className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">Experiment Configuration</CardTitle>
        </div>
        <CardDescription>Applies to this evaluation only</CardDescription>
      </CardHeader>
      <CardContent className="space-y-0">
        <ToggleRow
          label="CUPED Variance Reduction"
          description="Use pre-experiment covariates to reduce variance and tighten confidence intervals."
          checked={settings.cuped}
          onChange={(v) => update({ cuped: v })}
        />
        <div className="h-px bg-border" />
        <ToggleRow
          label="Bootstrap Resampling"
          description="Non-parametric confidence intervals via 10,000 bootstrap iterations."
          checked={settings.bootstrap}
          onChange={(v) => update({ bootstrap: v })}
        />
        <div className="h-px bg-border" />

        {/* Confidence level / statistical power — locked in for this
            run only (same contract as CUPED/bootstrap/model above);
            omitting either falls back to the backend's fixed default
            (95% / 80%) rather than sending an unvalidated value.

            Values round-trip through whole-percent strings ("90",
            "95"), never `String(0.90)` — JS stringifies 0.90 as "0.9",
            which silently failed to match a hardcoded "0.90" SelectItem
            value and made the picker look like it wasn't registering
            selections at all. */}
        <div className="flex items-start justify-between gap-4 py-3.5">
          <div className="min-w-0">
            <p className="flex items-center gap-1.5 text-[13px] font-medium text-foreground">
              <Target className="h-3.5 w-3.5" />
              Confidence Level
            </p>
            <p className="text-xs text-muted-foreground">
              Significance threshold for this run&apos;s hypothesis test and guardrails.
            </p>
          </div>
          <Select
            value={settings.confidenceLevel != null ? String(Math.round(settings.confidenceLevel * 100)) : '__default__'}
            onValueChange={(v) => update({ confidenceLevel: v === '__default__' ? undefined : Number(v) / 100 })}
          >
            <SelectTrigger className="h-8 w-[140px] shrink-0 border-border-strong text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__default__">Default (95%)</SelectItem>
              <SelectItem value="90">90%</SelectItem>
              <SelectItem value="95">95%</SelectItem>
              <SelectItem value="99">99%</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="h-px bg-border" />
        <div className="flex items-start justify-between gap-4 py-3.5">
          <div className="min-w-0">
            <p className="flex items-center gap-1.5 text-[13px] font-medium text-foreground">
              <Gauge className="h-3.5 w-3.5" />
              Statistical Power
            </p>
            <p className="text-xs text-muted-foreground">
              Target power used for this run&apos;s MDE / required-sample-size calculation.
            </p>
          </div>
          <Select
            value={settings.statisticalPower != null ? String(Math.round(settings.statisticalPower * 100)) : '__default__'}
            onValueChange={(v) => update({ statisticalPower: v === '__default__' ? undefined : Number(v) / 100 })}
          >
            <SelectTrigger className="h-8 w-[140px] shrink-0 border-border-strong text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__default__">Default (80%)</SelectItem>
              <SelectItem value="80">80%</SelectItem>
              <SelectItem value="90">90%</SelectItem>
              <SelectItem value="95">95%</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="h-px bg-border" />

        {/* Model selector — server-curated list only (GET /system/models),
            filtered to free (":free") models only per product decision:
            this Copilot only ever pays for the deterministic stats
            engine's compute, never the LLM interpretation layer. */}
        <div className="flex items-start justify-between gap-4 py-3.5">
          <div className="min-w-0">
            <p className="flex items-center gap-1.5 text-[13px] font-medium text-foreground">
              <Cpu className="h-3.5 w-3.5" />
              LLM Model
              <Badge variant="outline" className="border-border text-[10px] font-normal text-muted-foreground">
                Free models only
              </Badge>
            </p>
            <p className="text-xs text-muted-foreground">
              {modelsError
                ? 'Could not load the model list.'
                : 'Free OpenRouter models may queue or fail to respond under load — switch here if that happens.'}
            </p>
          </div>
          <div className="shrink-0">
            {modelsError && (
              <span className="text-xs text-muted-foreground">Unavailable</span>
            )}
            {!modelsError && freeModels.length > 0 ? (
              <Select
                value={settings.model && freeModels.some((m) => m.id === settings.model) ? settings.model : freeModels[0].id}
                onValueChange={(v) => update({ model: v })}
              >
                <SelectTrigger className="h-8 w-[260px] border-border-strong text-xs">
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
              !modelsError && (
                <span className="text-xs text-muted-foreground">No free models configured</span>
              )
            )}
          </div>
        </div>
      </CardContent>
    </Card>
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
        <p className="text-[13px] font-medium text-foreground">{label}</p>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
      <Switch checked={checked} onCheckedChange={onChange} className="mt-1 shrink-0" />
    </div>
  );
}
