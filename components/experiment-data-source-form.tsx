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
 */

import { useEffect, useRef, useState } from 'react';
import { Loader2, Database, Upload, CheckCircle2 } from 'lucide-react';
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
} from '@/lib/api';
import type { ExperimentDefinition } from '@/lib/types';

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

  useEffect(() => {
    listRealDatasets()
      .then(setOptions)
      .catch(() => setOptionsError('Could not load the available datasets.'))
      .finally(() => setLoadingOptions(false));
  }, []);

  const connectedDatasetId = definition.dataSource?.datasetId ?? null;
  const connectedName = definition.dataSource?.datasetName ?? null;

  const saveDataSource = async (datasetId: string, datasetName: string, type: 'existing_dataset' | 'uploaded_csv') => {
    setError(null);
    try {
      const updated = await updateExperimentDefinition(definition.id, {
        dataSource: { type, datasetId, datasetName },
      });
      onSaved(updated);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not connect this dataset.');
    }
  };

  const handleChooseReal = async (option: RealDatasetOption) => {
    setConnectingKey(option.key);
    try {
      const result = await classifyDataset({ datasetKey: option.key });
      await saveDataSource(result.datasetId, option.label, 'existing_dataset');
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not connect this dataset.');
    } finally {
      setConnectingKey(null);
    }
  };

  const handleUpload = async (file: File) => {
    setConnectingKey('__upload__');
    try {
      const result = await classifyDataset({ file });
      await saveDataSource(result.datasetId, result.fileName ?? file.name, 'uploaded_csv');
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not classify this file.');
    } finally {
      setConnectingKey(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <Card className="border-black/10 shadow-none">
      <CardHeader className="space-y-0">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-black" />
          <div>
            <CardTitle className="text-[15px] tracking-tight">Data Source</CardTitle>
            <CardDescription>Choose the dataset this experiment will be analyzed against.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {connectedDatasetId && (
          <div className="flex items-center gap-2 rounded-md border border-green-200 bg-green-50 px-3 py-2 text-[13px] text-green-800">
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
            Connected to <span className="font-medium">{connectedName || connectedDatasetId}</span>
          </div>
        )}

        {optionsError && <p className="text-xs text-red-600">{optionsError}</p>}
        {loadingOptions && (
          <div className="flex items-center gap-2 py-4 text-xs text-neutral-400">
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
                    ? 'border-indigo-300 bg-indigo-50/60'
                    : 'border-black/10 hover:border-black/20 hover:bg-neutral-50'
                )}
              >
                <span
                  className={cn(
                    'flex h-4 w-4 shrink-0 items-center justify-center rounded-full border',
                    isConnected ? 'border-indigo-600 bg-indigo-600' : 'border-neutral-300'
                  )}
                >
                  {isConnected && <span className="h-1.5 w-1.5 rounded-full bg-white" />}
                </span>
                <span className="min-w-0 flex-1 text-[13px] text-black">{option.label}</span>
                {isConnecting && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-neutral-400" />}
              </button>
            );
          })}
        </div>

        <div className="border-t border-black/5 pt-3">
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

        {error && <p className="text-xs text-red-600">{error}</p>}
      </CardContent>
    </Card>
  );
}
