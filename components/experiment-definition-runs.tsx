'use client';

/**
 * Phase 9/10 — Visualization + History, scoped to one ExperimentDefinition.
 *
 * Phase 9 ("no new calculations in the frontend"): this component does
 * not compute anything. "Run analysis" calls
 * `analyzeExperimentDefinition()` (Phase 8), which hands off to the
 * EXISTING analysis engine, and the result — a normal
 * `AnalyzeExperimentResult` — is rendered with the SAME `<ReportCard />`
 * already used by the Overview tab and History tab. Nothing here
 * re-derives lift, CIs, guardrail verdicts, or segments; it only
 * displays what the engine already returned.
 *
 * Phase 10 ("one definition can have several AnalysisRuns"): the list
 * below is `GET /experiment-definitions/{id}/runs`
 * (ExperimentStore.list_by_definition), most recent first — e.g.
 *
 *   Landing Page Redesign
 *     Analysis #1 — initial analysis
 *     Analysis #2 — extended data
 *     Analysis #3 — final analysis
 *
 * Selecting a past run reopens it via the existing
 * `GET /experiments/{experiment_id}` (getExperiment), same as the
 * History tab — this is not a second persistence mechanism.
 */

import { useEffect, useState } from 'react';
import { FlaskConical, Loader2, ChevronRight, History as HistoryIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ReportCard } from '@/components/report-card';
import {
  analyzeExperimentDefinition,
  getExperiment,
  listExperimentDefinitionRuns,
  ApiError,
} from '@/lib/api';
import type { ConfidenceLevel, ExperimentDefinition, ExperimentDetail, ExperimentSummary, Settings } from '@/lib/types';

interface ExperimentDefinitionRunsProps {
  definition: ExperimentDefinition;
  /** Applies the same run-level toggles (CUPED/bootstrap/model) as the
   *  Overview tab's Experiment Configuration panel — kept at the parent
   *  so it's one shared control, not a second, divergent settings UI. */
  settings: Settings;
}

const CONFIDENCE_STYLES: Record<ConfidenceLevel, string> = {
  HIGH: 'border-green-200 bg-green-50 text-green-700',
  MEDIUM: 'border-black/10 bg-neutral-100 text-neutral-600',
  LOW: 'border-red-200 bg-red-50 text-red-700',
};

const DECISION_STYLES: Record<string, string> = {
  GO: 'border-green-200 bg-green-50 text-green-700',
  GO_WITH_CAUTION: 'border-amber-200 bg-amber-50 text-amber-700',
  NO_GO: 'border-red-200 bg-red-50 text-red-700',
  INCONCLUSIVE: 'border-neutral-200 bg-neutral-50 text-neutral-500',
  INVALID: 'border-red-200 bg-red-50 text-red-700',
};

function runLabel(index: number, total: number): string {
  // Oldest run is "#1 — initial analysis"; the most recent of several is
  // "final analysis"; anything in between is just numbered — matches
  // the Stage 0 doc's History mockup without inventing extra metadata
  // the backend doesn't actually track (no "run purpose" field exists).
  const position = total - index; // list is most-recent-first
  if (total === 1) return `Analysis #${position} — initial analysis`;
  if (position === 1) return `Analysis #${position} — initial analysis`;
  if (position === total) return `Analysis #${position} — final analysis`;
  return `Analysis #${position}`;
}

export function ExperimentDefinitionRuns({ definition, settings }: ExperimentDefinitionRunsProps) {
  const [runs, setRuns] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const hasDataSource = Boolean(definition.dataSource?.datasetId);

  const refetchRuns = () => {
    setLoading(true);
    listExperimentDefinitionRuns(definition.id)
      .then((list) => {
        setRuns(list);
        setLoadError(null);
      })
      .catch(() => setLoadError('Could not load past analysis runs.'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    refetchRuns();
    setSelectedId(null);
    setDetail(null);
  }, [definition.id]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    setDetailLoading(true);
    setDetailError(null);
    getExperiment(selectedId)
      .then(setDetail)
      .catch((e) => {
        setDetail(null);
        setDetailError(e instanceof ApiError ? e.message : 'Could not load this analysis run.');
      })
      .finally(() => setDetailLoading(false));
  }, [selectedId]);

  const handleRunAnalysis = async () => {
    setRunning(true);
    setRunError(null);
    try {
      const result = await analyzeExperimentDefinition({ definitionId: definition.id, settings });
      refetchRuns();
      setSelectedId(result.experimentId);
    } catch (e) {
      setRunError(e instanceof ApiError ? e.message : 'Could not run the analysis.');
    } finally {
      setRunning(false);
    }
  };

  return (
    <Card className="border-black/10 shadow-none">
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <div className="flex items-center gap-2">
          <HistoryIcon className="h-4 w-4 text-black" />
          <div>
            <CardTitle className="text-[15px] tracking-tight">Analysis Runs</CardTitle>
            <CardDescription>
              {hasDataSource
                ? 'Run this definition against its connected dataset through the existing analysis engine.'
                : 'Connect a data source above before running an analysis.'}
            </CardDescription>
          </div>
        </div>
        <Button
          size="sm"
          onClick={handleRunAnalysis}
          disabled={!hasDataSource || running}
          className="shrink-0 gap-1.5 bg-indigo-600 text-white hover:bg-indigo-700"
        >
          {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FlaskConical className="h-3.5 w-3.5" />}
          Run analysis
        </Button>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {runError && <p className="text-xs text-red-600">{runError}</p>}
        {loadError && <p className="text-xs text-red-600">{loadError}</p>}

        {loading && (
          <div className="flex items-center gap-2 py-4 text-xs text-neutral-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading past runs...
          </div>
        )}

        {!loading && runs.length === 0 && (
          <div className="rounded-md border border-dashed border-black/15 bg-neutral-50/60 py-6 text-center text-[13px] text-neutral-400">
            No analysis runs yet for this experiment.
          </div>
        )}

        <div className="space-y-1.5">
          {runs.map((run, i) => {
            const active = selectedId === run.experimentId;
            return (
              <button
                key={run.experimentId}
                type="button"
                onClick={() => setSelectedId(active ? null : run.experimentId)}
                className={cn(
                  'flex w-full items-center justify-between gap-3 rounded-lg border p-3 text-left transition-colors',
                  active ? 'border-black/20 bg-neutral-50' : 'border-black/10 hover:bg-neutral-50'
                )}
              >
                <div className="min-w-0">
                  <p className="truncate text-[13px] font-medium text-black">{runLabel(i, runs.length)}</p>
                  <p className="mt-0.5 truncate text-xs text-neutral-400">
                    {new Date(run.createdAt).toLocaleString()} · {run.primaryMetric}
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-1.5">
                  <Badge
                    variant="outline"
                    className={cn('text-[10px]', DECISION_STYLES[run.decision] ?? DECISION_STYLES.INCONCLUSIVE)}
                  >
                    {run.decision.replace(/_/g, ' ')}
                  </Badge>
                  <Badge
                    variant="outline"
                    className={cn('text-[10px]', CONFIDENCE_STYLES[run.confidence as ConfidenceLevel])}
                  >
                    {run.confidence}
                  </Badge>
                  <ChevronRight
                    className={cn('h-3.5 w-3.5 text-neutral-300 transition-transform', active && 'rotate-90')}
                  />
                </div>
              </button>
            );
          })}
        </div>

        {detailError && <p className="text-xs text-red-600">{detailError}</p>}
        {detailLoading && (
          <div className="flex items-center gap-2 py-4 text-xs text-neutral-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading report...
          </div>
        )}

        {!detailLoading && detail && (
          <div className="pt-2">
            <ReportCard
              report={detail.report}
              datasetName={detail.datasetName}
              experimentId={detail.experimentId}
              prompt={detail.userPrompt}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
