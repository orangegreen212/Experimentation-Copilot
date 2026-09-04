'use client';

import { useEffect, useState } from 'react';
import { Database, Loader2, Users, Layers3, FlaskConical, ChevronRight } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { listExperiments, getExperiment, ApiError } from '@/lib/api';
import type { ExperimentSummary } from '@/lib/types';

interface DatasetGroup {
  datasetName: string;
  experimentCount: number;
  lastRunAt: string;
  primaryMetric: string;
  /** experimentId of the most recent run against this dataset — used to
   *  fetch a classification blurb lazily (see below). */
  latestExperimentId: string;
}

interface DatasetsViewProps {
  /** Bump to force a refetch (e.g. after a new experiment is saved). */
  refreshKey?: number;
  /** Switches the app to the Experiments tab, optionally opening one run. */
  onViewExperiments?: (experimentId?: string) => void;
}

function groupByDataset(sessions: ExperimentSummary[]): DatasetGroup[] {
  const byName = new Map<string, ExperimentSummary[]>();
  for (const s of sessions) {
    const list = byName.get(s.datasetName) ?? [];
    list.push(s);
    byName.set(s.datasetName, list);
  }
  return Array.from(byName.entries())
    .map(([datasetName, runs]) => {
      const sorted = [...runs].sort(
        (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      );
      return {
        datasetName,
        experimentCount: runs.length,
        lastRunAt: sorted[0].createdAt,
        primaryMetric: sorted[0].primaryMetric,
        latestExperimentId: sorted[0].experimentId,
      };
    })
    .sort((a, b) => new Date(b.lastRunAt).getTime() - new Date(a.lastRunAt).getTime());
}

/** One card's classification blurb — fetched lazily per-dataset (only the
 *  latest experiment's report is loaded, not the whole history) so the
 *  list itself renders instantly from data we already had. */
function DatasetCard({
  group,
  onViewExperiments,
}: {
  group: DatasetGroup;
  onViewExperiments?: (experimentId?: string) => void;
}) {
  const [classification, setClassification] = useState<string | null>(null);
  const [stats, setStats] = useState<{ users: number; variants: number } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getExperiment(group.latestExperimentId)
      .then((detail) => {
        if (cancelled) return;
        const rm = detail.report.runMetadata;
        setClassification(rm?.datasetClassification ?? null);
        setStats(rm ? { users: rm.userCount, variants: rm.variantCount } : null);
      })
      .catch(() => {
        if (!cancelled) setClassification(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [group.latestExperimentId]);

  return (
    <Card className="border-black/10 shadow-none transition-colors hover:border-black/20">
      <CardContent className="space-y-3 p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2.5">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
              <Database className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-[13px] font-semibold text-black">{group.datasetName}</p>
              <p className="text-[11px] text-neutral-400">
                Last used {new Date(group.lastRunAt).toLocaleDateString()}
              </p>
            </div>
          </div>
          <Badge variant="outline" className="shrink-0 border-black/10 text-[10px] text-neutral-600">
            {group.experimentCount} experiment{group.experimentCount === 1 ? '' : 's'}
          </Badge>
        </div>

        <div className="min-h-[32px] text-[12px] leading-relaxed text-neutral-600">
          {loading ? (
            <span className="inline-flex items-center gap-1.5 text-neutral-400">
              <Loader2 className="h-3 w-3 animate-spin" />
              Loading classification...
            </span>
          ) : (
            classification || 'No classification summary available for this dataset.'
          )}
        </div>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-neutral-500">
          {stats && (
            <>
              <span className="inline-flex items-center gap-1.5">
                <Users className="h-3.5 w-3.5 text-neutral-400" />
                {stats.users.toLocaleString()} users
              </span>
              <span className="inline-flex items-center gap-1.5">
                <Layers3 className="h-3.5 w-3.5 text-neutral-400" />
                {stats.variants} variants
              </span>
            </>
          )}
          <span className="inline-flex items-center gap-1.5">
            <FlaskConical className="h-3.5 w-3.5 text-neutral-400" />
            {group.primaryMetric}
          </span>
        </div>

        {onViewExperiments && (
          <button
            onClick={() => onViewExperiments(group.latestExperimentId)}
            className="inline-flex items-center gap-1 text-[12px] font-medium text-indigo-600 hover:text-indigo-700"
          >
            View experiments
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        )}
      </CardContent>
    </Card>
  );
}

export function DatasetsView({ refreshKey, onViewExperiments }: DatasetsViewProps) {
  const [sessions, setSessions] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    listExperiments()
      .then((list) => {
        setSessions(list);
        setError(null);
      })
      .catch(() => {
        setSessions([]);
        setError('Could not load datasets. Please try again.');
      })
      .finally(() => setLoading(false));
  }, [refreshKey]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-16 text-sm text-neutral-400">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading datasets...
      </div>
    );
  }

  if (error) {
    return (
      <Card className="border-red-200 bg-red-50 shadow-none">
        <CardContent className="py-3 text-[13px] text-red-700">{error}</CardContent>
      </Card>
    );
  }

  const groups = groupByDataset(sessions);

  if (groups.length === 0) {
    return (
      <Card className="border-black/10 shadow-none">
        <CardContent className="flex flex-col items-center gap-2 py-20 text-center text-sm text-neutral-400">
          <Database className="h-6 w-6 text-neutral-300" />
          No datasets yet — run an experiment to see it here.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {groups.map((g) => (
        <DatasetCard key={g.datasetName} group={g} onViewExperiments={onViewExperiments} />
      ))}
    </div>
  );
}
