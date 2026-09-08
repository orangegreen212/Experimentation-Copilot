'use client';

/**
 * Phase 8 — Data Source.
 *
 * This is the connection point between the new planning layer
 * (`ExperimentDefinition`) and the datasets the existing analysis
 * engine already knows how to classify — the same three "real,
 * published experiment" datasets and the same upload path used by the
 * Overview tab's New Experiment flow (see routes_datasets.py's
 * `GET /datasets/real` / `POST /datasets/classify`). Nothing new is
 * computed here: choosing a source just runs it through the existing
 * classifier and saves the resulting `dataset_id`/`dataset_name` onto
 * `definition.data_source` — see `DataSourceRef`
 * (schemas/experiment_definition.py). That reference is what
 * `POST /experiment-definitions/{id}/analyze` (Phase 8 backend) later
 * reads to hand off to the EXISTING analysis engine:
 *
 *   ExperimentDefinition
 *           |
 *     selected dataset
 *           |
 *   EXISTING ANALYSIS ENGINE
 *           |
 *     Decision Scientist
 *           |
 *        Report
 *
 * Also snapshots the dataset's detected `metric_label`/
 * `available_metrics` onto `data_source` (not just the id/name) —
 * that's what lets ExperimentDesignForm render the primary-metric
 * field as a picker instead of free text once a dataset is connected,
 * so a hypothesis's `primary_metric` can no longer silently drift
 * from the dataset's real metric name (the bug this replaced: a typo
 * or paraphrase typed during Design, before any dataset existed to
 * check it against, surfacing only much later as "Hypothesis
 * Evaluation: UNAVAILABLE" deep in the report). The warning below is
 * a safety net for hypotheses written before a dataset was connected,
 * not the primary defense anymore.
 */

import { useEffect, useRef, useState } from 'react';
import { Loader2, Database, Upload, CheckCircle2, AlertTriangle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import {
  classifyDataset,
  listRealDatasets,
  updateExperimentDefinition,
  ApiError,
  type RealDatasetOption,
  type ClassifyDatasetResult,
} from '@/lib/api';
import type { ExperimentDefinition, HypothesisRole } from '@/lib/types';

interface ExperimentDataSourceFormProps {
  definition: ExperimentDefinition;
  onSaved: (updated: ExperimentDefinition) => void;
}

export function ExperimentDataSourceForm({ definition, onSaved }: ExperimentDataSourceFormProps) {
  const [options, setOptions] = useState<RealDatasetOption[]>([]);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [connectingKey, setConnectingKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [syncing, setSyncing] = useState(false);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [backfilling, setBackfilling] = useState(false);
  // Guards against re-triggering the backfill for the same connection
  // (e.g. if the reclassify itself fails, don't retry-loop on every
  // render) — keyed by datasetId so a genuinely NEW connection is
  // never skipped.
  const backfilledFor = useRef<string | null>(null);

  useEffect(() => {
    listRealDatasets()
      .then(setOptions)
      .catch(() => setOptionsError('Could not load the available datasets.'))
      .finally(() => setLoadingOptions(false));
  }, []);

  // Backward-compatible backfill: a definition whose data source was
  // connected BEFORE `metricLabel`/`availableMetrics` existed on
  // DataSourceRef has `dataSource.datasetId` but no `metricLabel` —
  // which means the Design step's primary-metric picker silently
  // falls back to free text and the mismatch warning below can never
  // fire, for a definition that may well already have a mismatched
  // hypothesis (exactly what was happening before this ran). Rather
  // than requiring the analyst to notice and manually re-click an
  // already-"Connected" dataset, re-run the same classify() this
  // dataset was originally connected with and save the resulting
  // metric metadata — visible via the small "Refreshing..." note
  // below, never fully silent, and only for `existing_dataset`
  // sources matched against the real-dataset list (an uploaded CSV's
  // original file isn't available to reclassify, so those are left
  // for a manual reconnect — see the docstring above).
  useEffect(() => {
    const ds = definition.dataSource;
    if (
      !ds ||
      ds.type !== 'existing_dataset' ||
      !ds.datasetId ||
      ds.metricLabel ||
      options.length === 0 ||
      backfilledFor.current === ds.datasetId
    ) {
      return;
    }
    const match = options.find((o) => o.label === ds.datasetName);
    if (!match) return;

    backfilledFor.current = ds.datasetId;
    setBackfilling(true);
    classifyDataset({ datasetKey: match.key })
      .then((result) =>
        updateExperimentDefinition(definition.id, {
          dataSource: {
            type: 'existing_dataset',
            datasetId: result.datasetId,
            datasetName: match.label,
            metricLabel: result.dataset.metricLabel,
            availableMetrics: result.dataset.availableMetrics ?? [result.dataset.metricLabel],
          },
        })
      )
      .then(onSaved)
      .catch(() => {
        // Non-fatal — the analyst can still reconnect manually via the
        // list below; no need to surface this as a hard error.
      })
      .finally(() => setBackfilling(false));
  }, [definition.dataSource, definition.id, onSaved, options]);

  const connectedDatasetId = definition.dataSource?.datasetId ?? null;
  const connectedName = definition.dataSource?.datasetName ?? null;
  // Persisted on `dataSource` (see DataSourceRef's docstring) rather than
  // held in local state, so it survives a reload — the connected
  // dataset's real metric name is what backs the Design step's
  // primary-metric picker (ExperimentDesignForm), not just this
  // mismatch warning.
  const connectedMetricLabel = definition.dataSource?.metricLabel ?? null;

  const primaryHypothesisIndex = definition.hypotheses.findIndex(
    (h) => h.role === ('primary' as HypothesisRole)
  );
  const primaryHypothesis = primaryHypothesisIndex >= 0 ? definition.hypotheses[primaryHypothesisIndex] : null;

  // Safety net for hypotheses written BEFORE a picker was available (or
  // edited some other way) — the picker in ExperimentDesignForm is what
  // now prevents this going forward, once a dataset is connected.
  const metricMismatch =
    connectedMetricLabel &&
    primaryHypothesis &&
    primaryHypothesis.hypothesis.primaryMetric.trim().toLowerCase() !== connectedMetricLabel.trim().toLowerCase()
      ? { hypothesisMetric: primaryHypothesis.hypothesis.primaryMetric, datasetMetric: connectedMetricLabel }
      : null;

  const saveDataSource = async (
    datasetId: string,
    datasetName: string,
    type: 'existing_dataset' | 'uploaded_csv',
    metricLabel: string,
    availableMetrics: string[]
  ) => {
    setError(null);
    try {
      const updated = await updateExperimentDefinition(definition.id, {
        dataSource: { type, datasetId, datasetName, metricLabel, availableMetrics },
      });
      onSaved(updated);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not connect this dataset.');
    }
  };

  const handleChooseReal = async (option: RealDatasetOption) => {
    setConnectingKey(option.key);
    try {
      const result: ClassifyDatasetResult = await classifyDataset({ datasetKey: option.key });
      await saveDataSource(
        result.datasetId,
        option.label,
        'existing_dataset',
        result.dataset.metricLabel,
        result.dataset.availableMetrics ?? [result.dataset.metricLabel]
      );
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not connect this dataset.');
    } finally {
      setConnectingKey(null);
    }
  };

  const handleUpload = async (file: File) => {
    setConnectingKey('__upload__');
    try {
      const result: ClassifyDatasetResult = await classifyDataset({ file });
      await saveDataSource(
        result.datasetId,
        result.fileName ?? file.name,
        'uploaded_csv',
        result.dataset.metricLabel,
        result.dataset.availableMetrics ?? [result.dataset.metricLabel]
      );
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not classify this file.');
    } finally {
      setConnectingKey(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleSyncMetric = async () => {
    if (!connectedMetricLabel || primaryHypothesisIndex < 0) return;
    setSyncing(true);
    setSyncError(null);
    try {
      const nextHypotheses = definition.hypotheses.map((h, i) =>
        i === primaryHypothesisIndex
          ? { ...h, hypothesis: { ...h.hypothesis, primaryMetric: connectedMetricLabel } }
          : h
      );
      const updated = await updateExperimentDefinition(definition.id, { hypotheses: nextHypotheses });
      onSaved(updated);
    } catch (e) {
      setSyncError(e instanceof ApiError ? e.message : 'Could not update the hypothesis.');
    } finally {
      setSyncing(false);
    }
  };

  return (
    <Card className="border-border shadow-none">
      <CardHeader className="space-y-0">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-foreground" />
          <div>
            <CardTitle className="text-[15px] tracking-tight">Data Source</CardTitle>
            <CardDescription>Choose the dataset this experiment will be analyzed against.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {connectedDatasetId && (
          <div className="flex items-center gap-2 rounded-md border border-success/25 bg-success/[0.08] px-3 py-2 text-[13px] text-success">
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
            Connected to <span className="font-medium">{connectedName || connectedDatasetId}</span>
            {backfilling && (
              <span className="ml-auto flex items-center gap-1.5 text-xs font-normal text-success/70">
                <Loader2 className="h-3 w-3 animate-spin" />
                Refreshing dataset metadata...
              </span>
            )}
          </div>
        )}

        {metricMismatch && (
          <div className="space-y-2 rounded-md border border-warning/25 bg-warning/[0.08] px-3 py-2 text-[13px] text-warning">
            <div className="flex items-start gap-2">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <p>
                Your primary hypothesis&apos;s metric is{' '}
                <span className="font-medium">&ldquo;{metricMismatch.hypothesisMetric}&rdquo;</span>, but this
                dataset&apos;s detected metric is{' '}
                <span className="font-medium">&ldquo;{metricMismatch.datasetMetric}&rdquo;</span>. These must
                match exactly for hypothesis evaluation to run — otherwise the report will show
                &ldquo;evaluation unavailable&rdquo; even though the analysis itself succeeds.
              </p>
            </div>
            <div className="flex items-center gap-2 pl-5">
              <Button size="sm" variant="outline" className="h-7 text-xs" onClick={handleSyncMetric} disabled={syncing}>
                {syncing && <Loader2 className="mr-1.5 h-3 w-3 animate-spin" />}
                Use &ldquo;{metricMismatch.datasetMetric}&rdquo; instead
              </Button>
              {syncError && <span className="text-xs text-destructive">{syncError}</span>}
            </div>
          </div>
        )}

        {optionsError && <p className="text-xs text-destructive">{optionsError}</p>}
        {loadingOptions && (
          <div className="flex items-center gap-2 py-4 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading available datasets...
          </div>
        )}

        <div className="space-y-2">
          {options.map((option) => {
            const isConnecting = connectingKey === option.key;
            const isConnected = connectedName === option.label;
            return (
              <button
                key={option.key}
                type="button"
                onClick={() => handleChooseReal(option)}
                disabled={connectingKey !== null}
                className={cn(
                  'flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-colors disabled:opacity-60',
                  isConnected
                    ? 'border-primary/40 bg-accent/60'
                    : 'border-border hover:border-border-strong hover:bg-secondary'
                )}
              >
                <span
                  className={cn(
                    'flex h-4 w-4 shrink-0 items-center justify-center rounded-full border',
                    isConnected ? 'border-primary bg-primary' : 'border-border-strong'
                  )}
                >
                  {isConnected && <span className="h-1.5 w-1.5 rounded-full bg-surface" />}
                </span>
                <span className="min-w-0 flex-1 text-[13px] text-foreground">{option.label}</span>
                {isConnecting && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground" />}
              </button>
            );
          })}
        </div>

        <div className="border-t border-border pt-3">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleUpload(file);
            }}
          />
          <Button
            size="sm"
            variant="outline"
            className="gap-1.5"
            disabled={connectingKey !== null}
            onClick={() => fileInputRef.current?.click()}
          >
            {connectingKey === '__upload__' ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Upload className="h-3.5 w-3.5" />
            )}
            Upload CSV instead
          </Button>
        </div>

        {error && <p className="text-xs text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}
