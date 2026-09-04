'use client';

import { useEffect, useState } from 'react';
import { Cpu, Wand2 } from 'lucide-react';
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

  const update = (patch: Partial<Settings>) => onChange({ ...settings, ...patch });

  // Empty string sentinel = "use the backend default" (settings.model
  // left undefined) — the Radix Select needs a non-empty value for
  // every item, so the default option gets its own explicit value.
  const selectValue = settings.model ?? '__default__';

  return (
    <Card className="border-black/10 shadow-none">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Wand2 className="h-4 w-4 text-black" />
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
        <div className="h-px bg-black/10" />
        <ToggleRow
          label="Bootstrap Resampling"
          description="Non-parametric confidence intervals via 10,000 bootstrap iterations."
          checked={settings.bootstrap}
          onChange={(v) => update({ bootstrap: v })}
        />
        <div className="h-px bg-black/10" />

        {/* Model selector — server-curated list only (GET /system/models).
            Applies to this run's report generation and its follow-up chat. */}
        <div className="flex items-start justify-between gap-4 py-3.5">
          <div className="min-w-0">
            <p className="flex items-center gap-1.5 text-[13px] font-medium text-black">
              <Cpu className="h-3.5 w-3.5" />
              LLM Model
            </p>
            <p className="text-xs text-neutral-400">
              {modelsError
                ? 'Could not load the model list — using the backend default.'
                : 'Pick a free OpenRouter model if the default is rate-limited or unavailable.'}
            </p>
          </div>
          <div className="shrink-0">
            {modelsError && infoError && (
              <span className="text-xs text-neutral-400">Unavailable</span>
            )}
            {!modelsError && models.length > 0 ? (
              <Select
                value={selectValue}
                onValueChange={(v) => update({ model: v === '__default__' ? undefined : v })}
              >
                <SelectTrigger className="h-8 w-[260px] border-black/15 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__default__">
                    Backend default{defaultModel ? ` (${defaultModel})` : ''}
                  </SelectItem>
                  {models
                    .filter((m) => m.id !== defaultModel)
                    .map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        {m.label}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            ) : (
              !modelsError &&
              systemInfo && (
                <Badge variant="outline" className="border-black/10 text-neutral-600">
                  {systemInfo.llmModel}
                </Badge>
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
        <p className="text-[13px] font-medium text-black">{label}</p>
        <p className="text-xs text-neutral-400">{description}</p>
      </div>
      <Switch checked={checked} onCheckedChange={onChange} className="mt-1 shrink-0" />
    </div>
  );
}
