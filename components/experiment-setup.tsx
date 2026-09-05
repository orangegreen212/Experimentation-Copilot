'use client';

/**
 * Pre-experiment "Create Experiment" screen — captures a hypothesis,
 * primary/guardrail metrics, and (optionally) computes the required
 * sample size BEFORE any dataset is selected.
 *
 * This is deliberately separate from HypothesisForm (hypothesis-form.tsx):
 * that form captures a hypothesis SCOPED TO an already-loaded dataset
 * (its primary-metric dropdown is populated from dataset.availableMetrics,
 * and the resulting Hypothesis is sent to /experiments/analyze alongside
 * that specific dataset). This screen runs BEFORE any dataset exists —
 * primary metric is free text, not a dropdown, and the output
 * (ExperimentPlan) is never sent to the backend as-is; it only pre-fills
 * HypothesisForm once a dataset is picked afterward (see onContinue).
 *
 * The sample-size calculator here calls POST /experiments/plan-sample-size
 * (app/api/routes_planning.py) — pure statsmodels math over an ASSUMED
 * baseline/effect, the inverse problem from the post-hoc power analysis
 * shown in a finished report.
 */

import { useState } from 'react';
import { ChevronDown, FlaskConical, Loader2, Plus, X, Calculator } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { planSampleSize, ApiError } from '@/lib/api';
import type {
  ExpectedDirection,
  ExperimentPlan,
  PlanningMetricType,
  SampleSizePlanResponse,
} from '@/lib/types';

interface ExperimentSetupProps {
  /** Called when the analyst is done planning and wants to move on to
   *  picking a dataset (upload / demo / real). The plan is handed up so
   *  the caller can pre-fill HypothesisForm once a dataset loads. */
  onContinue: (plan: ExperimentPlan) => void;
  /** Called when the analyst wants to skip planning entirely and go
   *  straight to picking a dataset, with no plan captured. */
  onSkip: () => void;
}

const DIRECTION_OPTIONS: { value: ExpectedDirection; label: string }[] = [
  { value: 'increase', label: 'Increase' },
  { value: 'decrease', label: 'Decrease' },
  { value: 'no_change', label: 'No change' },
];

const METRIC_TYPE_OPTIONS: { value: PlanningMetricType; label: string }[] = [
  { value: 'binary', label: 'Binary (conversion rate)' },
  { value: 'continuous_monetary', label: 'Continuous — monetary (e.g. order value)' },
  { value: 'continuous_general', label: 'Continuous — general (e.g. session duration)' },
];

export function ExperimentSetup({ onContinue, onSkip }: ExperimentSetupProps) {
  const [statement, setStatement] = useState('');
  const [primaryMetric, setPrimaryMetric] = useState('');
  const [expectedDirection, setExpectedDirection] = useState<ExpectedDirection>('increase');
  const [guardrails, setGuardrails] = useState<string[]>([]);
  const [guardrailDraft, setGuardrailDraft] = useState('');

  // Sample size calculator inputs
  const [metricType, setMetricType] = useState<PlanningMetricType>('binary');
  const [baselineRate, setBaselineRate] = useState('');
  const [baselineStd, setBaselineStd] = useState('');
  const [mdeRelativePct, setMdeRelativePct] = useState('');
  const [numVariants, setNumVariants] = useState('2');
  const [dailyTraffic, setDailyTraffic] = useState('');

  const [isCalculating, setIsCalculating] = useState(false);
  const [calcError, setCalcError] = useState<string | null>(null);
  const [result, setResult] = useState<SampleSizePlanResponse | null>(null);

  const addGuardrail = () => {
    const trimmed = guardrailDraft.trim();
    if (trimmed && !guardrails.includes(trimmed)) {
      setGuardrails([...guardrails, trimmed]);
    }
    setGuardrailDraft('');
  };

  const removeGuardrail = (name: string) => {
    setGuardrails(guardrails.filter((g) => g !== name));
  };

  const canCalculate =
    baselineRate.trim() !== '' &&
    mdeRelativePct.trim() !== '' &&
    (metricType === 'binary' || baselineStd.trim() !== '');

  const handleCalculate = async () => {
    setIsCalculating(true);
    setCalcError(null);
    setResult(null);
    try {
      const response = await planSampleSize({
        metricType,
        baselineRate: parseFloat(baselineRate),
        baselineStd: baselineStd.trim() ? parseFloat(baselineStd) : null,
        mdeRelativePct: parseFloat(mdeRelativePct),
        numVariants: parseInt(numVariants, 10) || 2,
        dailyTrafficPerArm: dailyTraffic.trim() ? parseInt(dailyTraffic, 10) : null,
      });
      setResult(response);
    } catch (err) {
      setCalcError(
        err instanceof ApiError ? err.message : 'Could not calculate the required sample size.'
      );
    } finally {
      setIsCalculating(false);
    }
  };

  const handleContinue = () => {
    onContinue({
      statement,
      primaryMetric,
      expectedDirection,
      guardrailMetricNames: guardrails,
      sampleSizeRequest: canCalculate
        ? {
            metricType,
            baselineRate: parseFloat(baselineRate),
            baselineStd: baselineStd.trim() ? parseFloat(baselineStd) : null,
            mdeRelativePct: parseFloat(mdeRelativePct),
            numVariants: parseInt(numVariants, 10) || 2,
            dailyTrafficPerArm: dailyTraffic.trim() ? parseInt(dailyTraffic, 10) : null,
          }
        : null,
      sampleSizeResult: result,
    });
  };

  return (
    <Card className="border-black/10 shadow-none">
      <CardHeader className="space-y-0">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-black" />
          <CardTitle className="text-[15px] tracking-tight">Create Experiment</CardTitle>
        </div>
        <CardDescription>
          Define the hypothesis and plan the sample size before selecting a dataset
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* Hypothesis */}
        <div>
          <Label htmlFor="setup-statement" className="text-[13px] font-medium text-black">
            Hypothesis statement
          </Label>
          <Textarea
            id="setup-statement"
            value={statement}
            onChange={(e) => setStatement(e.target.value)}
            placeholder="e.g. Changing the checkout CTA from blue to green will increase checkout conversion."
            className="mt-1.5 min-h-[60px] resize-none border-black/10 placeholder:text-neutral-400"
            rows={2}
            maxLength={500}
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label htmlFor="setup-metric" className="text-[13px] font-medium text-black">
              Primary metric
            </Label>
            <Input
              id="setup-metric"
              value={primaryMetric}
              onChange={(e) => setPrimaryMetric(e.target.value)}
              placeholder="e.g. Checkout conversion rate"
              className="mt-1.5 border-black/10 placeholder:text-neutral-400"
              maxLength={100}
            />
          </div>
          <div>
            <Label className="text-[13px] font-medium text-black">Expected direction</Label>
            <Select
              value={expectedDirection}
              onValueChange={(v) => setExpectedDirection(v as ExpectedDirection)}
            >
              <SelectTrigger className="mt-1.5 border-black/10">
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

        {/* Guardrails */}
        <div>
          <Label className="text-[13px] font-medium text-black">Guardrail metrics</Label>
          <div className="mt-1.5 flex gap-2">
            <Input
              value={guardrailDraft}
              onChange={(e) => setGuardrailDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  addGuardrail();
                }
              }}
              placeholder="e.g. Refund rate — press Enter to add"
              className="border-black/10 placeholder:text-neutral-400"
              maxLength={100}
            />
            <Button type="button" variant="outline" size="sm" onClick={addGuardrail} className="border-black/15 shrink-0">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
          {guardrails.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {guardrails.map((g) => (
                <span
                  key={g}
                  className="flex items-center gap-1 rounded-full border border-black/10 bg-neutral-50 px-2.5 py-1 text-xs text-black"
                >
                  {g}
                  <button type="button" onClick={() => removeGuardrail(g)} aria-label={`Remove ${g}`}>
                    <X className="h-3 w-3 text-neutral-400 hover:text-black" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Sample size calculator */}
        <Collapsible className="rounded-lg border border-black/10">
          <CollapsibleTrigger className="flex w-full items-center justify-between px-4 py-3 text-left">
            <div className="flex items-center gap-2">
              <Calculator className="h-4 w-4 text-neutral-500" />
              <span className="text-[13px] font-medium text-black">
                Sample Size / MDE Calculator
              </span>
            </div>
            <ChevronDown className="h-4 w-4 text-neutral-400 transition-transform data-[state=open]:rotate-180" />
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-3 border-t border-black/10 px-4 py-4">
            <p className="text-xs text-neutral-400">
              Estimate how many users this experiment needs, from an assumed
              baseline and the smallest relative effect worth detecting.
            </p>

            <div>
              <Label className="text-[13px] font-medium text-black">Metric type</Label>
              <Select value={metricType} onValueChange={(v) => setMetricType(v as PlanningMetricType)}>
                <SelectTrigger className="mt-1.5 border-black/10">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {METRIC_TYPE_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className={cn('grid gap-3', metricType === 'binary' ? 'grid-cols-2' : 'grid-cols-3')}>
              <div>
                <Label htmlFor="setup-baseline" className="text-[13px] font-medium text-black">
                  {metricType === 'binary' ? 'Baseline rate (0–1)' : 'Baseline mean'}
                </Label>
                <Input
                  id="setup-baseline"
                  type="number"
                  value={baselineRate}
                  onChange={(e) => setBaselineRate(e.target.value)}
                  placeholder={metricType === 'binary' ? 'e.g. 0.12' : 'e.g. 45.0'}
                  className="mt-1.5 border-black/10 placeholder:text-neutral-400"
                />
              </div>
              {metricType !== 'binary' && (
                <div>
                  <Label htmlFor="setup-std" className="text-[13px] font-medium text-black">
                    Baseline std. dev.
                  </Label>
                  <Input
                    id="setup-std"
                    type="number"
                    value={baselineStd}
                    onChange={(e) => setBaselineStd(e.target.value)}
                    placeholder="e.g. 20.0"
                    className="mt-1.5 border-black/10 placeholder:text-neutral-400"
                  />
                </div>
              )}
              <div>
                <Label htmlFor="setup-mde" className="text-[13px] font-medium text-black">
                  Min. detectable effect (%)
                </Label>
                <Input
                  id="setup-mde"
                  type="number"
                  value={mdeRelativePct}
                  onChange={(e) => setMdeRelativePct(e.target.value)}
                  placeholder="e.g. 10"
                  className="mt-1.5 border-black/10 placeholder:text-neutral-400"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label htmlFor="setup-variants" className="text-[13px] font-medium text-black">
                  Number of variants
                </Label>
                <Input
                  id="setup-variants"
                  type="number"
                  min={2}
                  max={10}
                  value={numVariants}
                  onChange={(e) => setNumVariants(e.target.value)}
                  className="mt-1.5 border-black/10"
                />
              </div>
              <div>
                <Label htmlFor="setup-traffic" className="text-[13px] font-medium text-black">
                  Daily traffic per arm (optional)
                </Label>
                <Input
                  id="setup-traffic"
                  type="number"
                  value={dailyTraffic}
                  onChange={(e) => setDailyTraffic(e.target.value)}
                  placeholder="e.g. 500"
                  className="mt-1.5 border-black/10 placeholder:text-neutral-400"
                />
              </div>
            </div>

            <Button
              type="button"
              size="sm"
              onClick={handleCalculate}
              disabled={!canCalculate || isCalculating}
            >
              {isCalculating ? (
                <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
              ) : (
                <Calculator className="mr-1.5 h-4 w-4" />
              )}
              Calculate required sample size
            </Button>

            {calcError && <p className="text-[13px] text-red-600">{calcError}</p>}

            {result && (
              <div className="rounded-md border border-black/10 bg-neutral-50 px-4 py-3 text-[13px] text-black">
                <p>
                  <span className="font-semibold">{result.plan.requiredNPerArm.toLocaleString()}</span>{' '}
                  users needed per arm ({result.plan.requiredNTotal.toLocaleString()} total across{' '}
                  {numVariants} variants), at {(result.plan.targetPower * 100).toFixed(0)}% power and
                  α={result.plan.alpha}.
                </p>
                {result.estimatedDays != null && (
                  <p className="mt-1 text-neutral-500">
                    At the given traffic, that's roughly{' '}
                    <span className="font-medium text-black">
                      {Math.ceil(result.estimatedDays)} days
                    </span>{' '}
                    to reach the required sample size per arm.
                  </p>
                )}
              </div>
            )}
          </CollapsibleContent>
        </Collapsible>

        <div className="flex items-center justify-between">
          <button
            type="button"
            className="text-xs text-neutral-400 underline hover:text-black"
            onClick={onSkip}
          >
            Skip planning, go straight to dataset
          </button>
          <Button onClick={handleContinue} size="sm">
            Continue to select dataset
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
