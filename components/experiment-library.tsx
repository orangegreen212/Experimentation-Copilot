'use client';

import { useEffect, useState } from 'react';
import {
  FolderKanban,
  Plus,
  Loader2,
  Trash2,
  ArrowRight,
  Beaker,
  CheckCircle2,
  Circle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  createExperimentDefinition,
  deleteExperimentDefinition,
  getExperimentDefinition,
  listExperimentDefinitions,
  updateExperimentDefinition,
  ApiError,
} from '@/lib/api';
import { ExperimentDesignForm } from '@/components/experiment-design-form';
import { ExperimentVariantsForm } from '@/components/experiment-variants-form';
import { ExperimentTargetingForm } from '@/components/experiment-targeting-form';
import { ExperimentAssignmentForm } from '@/components/experiment-assignment-form';
import { ExperimentMetricsForm } from '@/components/experiment-metrics-form';
import { ExperimentDataSourceForm } from '@/components/experiment-data-source-form';
import { getReadinessChecks, isStatusGated } from '@/lib/experiment-readiness';
import { ExperimentDefinitionRuns } from '@/components/experiment-definition-runs';
import type {
  ExperimentDefinition,
  ExperimentDefinitionSummary,
  ExperimentStatus,
  Settings,
} from '@/lib/types';

interface ExperimentLibraryProps {
  /** Bump to force a refetch (e.g. after an analysis run elsewhere saves something related). */
  refreshKey?: number;
  /** Same run-level toggles (CUPED/bootstrap/model) as the Overview
   *  tab's Experiment Configuration panel — passed through to
   *  ExperimentDefinitionRuns (Phase 8/9) so "Run analysis" here uses
   *  the same settings, not a second/divergent control. */
  settings: Settings;
  /** Called when the analyst chooses to continue a definition into the
   *  existing analysis workflow (Overview tab) instead of running it
   *  inline via the Analysis Runs section below. */
  onContinueToAnalysis?: (definitionId: string) => void;
}

const STATUS_LABELS: Record<ExperimentStatus, string> = {
  draft: 'Draft',
  ready: 'Ready',
  running: 'Running',
  completed: 'Completed',
  needs_investigation: 'Needs Investigation',
  invalid: 'Invalid',
  shipped: 'Shipped',
};

const STATUS_STYLES: Record<ExperimentStatus, string> = {
  draft: 'border-border bg-secondary text-muted-foreground',
  ready: 'border-primary/25 bg-accent text-primary',
  running: 'border-copilot/25 bg-copilot-bg text-copilot',
  completed: 'border-success/25 bg-success/[0.08] text-success',
  needs_investigation: 'border-warning/25 bg-warning/[0.08] text-warning',
  invalid: 'border-destructive/25 bg-destructive/[0.08] text-destructive',
  shipped: 'border-success/40 bg-success/[0.15] text-success',
};

const ALL_STATUSES: ExperimentStatus[] = [
  'draft',
  'ready',
  'running',
  'completed',
  'needs_investigation',
  'invalid',
  'shipped',
];



interface NewDefinitionFormState {
  name: string;
  productArea: string;
  owner: string;
  team: string;
}

const EMPTY_FORM: NewDefinitionFormState = { name: '', productArea: '', owner: '', team: '' };

export function ExperimentLibrary({ refreshKey, settings, onContinueToAnalysis }: ExperimentLibraryProps) {
  const [items, setItems] = useState<ExperimentDefinitionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ExperimentDefinition | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [createOpen, setCreateOpen] = useState(false);
  const [form, setForm] = useState<NewDefinitionFormState>(EMPTY_FORM);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [statusSaving, setStatusSaving] = useState(false);
  const [statusBlockedReason, setStatusBlockedReason] = useState<string | null>(null);

  const refetchList = () => {
    setLoading(true);
    listExperimentDefinitions()
      .then((list) => {
        setItems(list);
        setError(null);
        setSelectedId((prev) => (prev && list.some((d) => d.id === prev) ? prev : (list[0]?.id ?? null)));
      })
      .catch(() => {
        setItems([]);
        setError('Could not load the Experiment Library. Please try again.');
      })
      .finally(() => setLoading(false));
  };

  useEffect(refetchList, [refreshKey]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    setDetailLoading(true);
    setDetailError(null);
    setStatusBlockedReason(null);
    getExperimentDefinition(selectedId)
      .then(setDetail)
      .catch((e) => {
        setDetail(null);
        setDetailError(e instanceof ApiError ? e.message : 'Could not load this experiment.');
      })
      .finally(() => setDetailLoading(false));
  }, [selectedId]);

  const handleCreate = async () => {
    if (!form.name.trim()) {
      setCreateError('Experiment name is required.');
      return;
    }
    setCreating(true);
    setCreateError(null);
    try {
      const created = await createExperimentDefinition({
        name: form.name.trim(),
        productArea: form.productArea.trim() || undefined,
        owner: form.owner.trim() || undefined,
        team: form.team.trim() || undefined,
      });
      setCreateOpen(false);
      setForm(EMPTY_FORM);
      refetchList();
      setSelectedId(created.id);
    } catch (e) {
      setCreateError(e instanceof ApiError ? e.message : 'Could not create this experiment.');
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (confirmingId !== id) {
      setConfirmingId(id);
      return;
    }
    setConfirmingId(null);
    setDeletingId(id);
    try {
      await deleteExperimentDefinition(id);
      setItems((prev) => {
        const next = prev.filter((d) => d.id !== id);
        setSelectedId((prevSelected) => (prevSelected === id ? (next[0]?.id ?? null) : prevSelected));
        return next;
      });
      setDetail((prev) => (prev?.id === id ? null : prev));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not delete this experiment.');
    } finally {
      setDeletingId(null);
    }
  };

  const handleStatusChange = async (status: ExperimentStatus) => {
    if (!detail) return;
    setStatusBlockedReason(null);
    if (isStatusGated(status)) {
      const gaps = getReadinessChecks(detail).filter((c) => !c.met);
      if (gaps.length > 0) {
        setStatusBlockedReason(
          `Can't mark this "${STATUS_LABELS[status]}" yet — missing: ${gaps.map((g) => g.label).join(', ')}.`
        );
        return;
      }
    }
    setStatusSaving(true);
    try {
      const updated = await updateExperimentDefinition(detail.id, { status });
      setDetail(updated);
      setItems((prev) => prev.map((d) => (d.id === updated.id ? { ...d, status: updated.status } : d)));
    } catch (e) {
      setDetailError(e instanceof ApiError ? e.message : 'Could not update status.');
    } finally {
      setStatusSaving(false);
    }
  };

  return (
    <div className="flex h-full gap-6">
      {/* Definition list */}
      <div className="flex w-72 shrink-0 flex-col">
        <div className="mb-3 flex items-center gap-2">
          <FolderKanban className="h-4 w-4 text-foreground" />
          <h2 className="text-[13px] font-semibold text-foreground">Experiments</h2>
          <Badge variant="outline" className="ml-auto text-[10px] border-border text-muted-foreground">
            {!loading && !error ? items.length : '—'}
          </Badge>
        </div>

        <Button
          size="sm"
          className="mb-3 gap-1.5 bg-primary text-white hover:bg-primary/90"
          onClick={() => {
            setForm(EMPTY_FORM);
            setCreateError(null);
            setCreateOpen(true);
          }}
        >
          <Plus className="h-3.5 w-3.5" />
          New Experiment
        </Button>

        {loading && (
          <div className="flex items-center gap-2 py-6 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading experiments...
          </div>
        )}
        {!loading && error && (
          <Card className="border-destructive/25 bg-destructive/[0.08] shadow-none">
            <CardContent className="py-3 text-[13px] text-destructive">{error}</CardContent>
          </Card>
        )}
        {!loading && !error && items.length === 0 && (
          <Card className="border-border shadow-none">
            <CardContent className="py-8 text-center text-[13px] text-muted-foreground">
              No experiments yet. Create one to start planning.
            </CardContent>
          </Card>
        )}

        <div className="space-y-px">
          {items.map((item) => {
            const active = selectedId === item.id;
            const isDeleting = deletingId === item.id;
            const isConfirming = confirmingId === item.id;
            return (
              <div
                key={item.id}
                className={cn(
                  'group relative w-full border-l-2 px-3 py-3 text-left transition-colors',
                  active ? 'border-black bg-secondary' : 'border-transparent hover:bg-secondary'
                )}
              >
                <button onClick={() => setSelectedId(item.id)} className="block w-full text-left">
                  <div className="flex items-center justify-between pr-7">
                    <span className="truncate text-[15px] font-medium text-foreground">{item.name}</span>
                  </div>
                  <p className="mt-0.5 truncate text-sm text-muted-foreground">
                    {item.productArea || 'No product area set'}
                    {item.owner ? ` · ${item.owner}` : ''}
                  </p>
                  <div className="mt-1.5 flex items-center gap-2">
                    <Badge variant="outline" className={cn('gap-1 text-xs', STATUS_STYLES[item.status])}>
                      {STATUS_LABELS[item.status]}
                    </Badge>
                    {item.primaryMetric && (
                      <span className="truncate text-xs text-muted-foreground">{item.primaryMetric}</span>
                    )}
                  </div>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(item.id);
                  }}
                  disabled={isDeleting}
                  title={isConfirming ? 'Click again to confirm delete' : 'Delete this experiment'}
                  className={cn(
                    'absolute right-2 top-3 rounded-md p-1.5 transition-colors',
                    isConfirming
                      ? 'bg-destructive/[0.08] text-destructive'
                      : 'text-muted-foreground/50 hover:bg-destructive/[0.08] hover:text-destructive group-hover:text-muted-foreground'
                  )}
                >
                  {isDeleting ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Trash2 className="h-3.5 w-3.5" />
                  )}
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Detail panel */}
      <div className="min-w-0 flex-1 overflow-y-auto pr-1">
        {detailError && (
          <Card className="mb-3 border-destructive/25 bg-destructive/[0.08] shadow-none">
            <CardContent className="py-3 text-[13px] text-destructive">{detailError}</CardContent>
          </Card>
        )}
        {detailLoading && (
          <div className="flex items-center gap-2 py-6 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading experiment...
          </div>
        )}
        {!detailLoading && detail ? (
          <div className="space-y-4">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <h2 className="truncate text-lg font-semibold tracking-tight text-foreground">
                  {detail.name}
                </h2>
                <p className="mt-0.5 text-sm text-muted-foreground">
                  {detail.productArea || 'No product area'}
                  {detail.owner ? ` · Owner: ${detail.owner}` : ''}
                  {detail.team ? ` · Team: ${detail.team}` : ''}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <Select
                  value={detail.status}
                  onValueChange={(v) => handleStatusChange(v as ExperimentStatus)}
                  disabled={statusSaving}
                >
                  <SelectTrigger className="h-8 w-[190px] text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {ALL_STATUSES.map((s) => (
                      <SelectItem key={s} value={s} className="text-xs">
                        {STATUS_LABELS[s]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {statusBlockedReason && (
              <p className="-mt-2 text-xs text-warning">{statusBlockedReason}</p>
            )}

            {/* Readiness — gives Draft/Ready/Running/... actual teeth
                (see lib/experiment-readiness.ts) without a backend
                change: a compact checklist of the same pieces this
                wizard's other sections already collect, so it's clear
                at a glance what's blocking a status like Ready/Running/
                Shipped rather than discovering it only from the error
                above after picking one. */}
            {detail.status !== 'invalid' && detail.status !== 'needs_investigation' && (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 rounded-lg border border-border bg-secondary/60 px-3 py-2">
                {getReadinessChecks(detail).map((c) => (
                  <span
                    key={c.label}
                    className={cn(
                      'flex items-center gap-1.5 text-xs',
                      c.met ? 'text-success' : 'text-muted-foreground'
                    )}
                  >
                    {c.met ? (
                      <CheckCircle2 className="h-3.5 w-3.5" />
                    ) : (
                      <Circle className="h-3.5 w-3.5" />
                    )}
                    {c.label}
                  </span>
                ))}
              </div>
            )}

            <Card className="border-border shadow-none">
              <CardContent className="py-4">
                <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
                  <span>
                    Hypotheses: <span className="font-medium text-muted-foreground">{detail.hypotheses.length}</span>
                  </span>
                  <span>
                    Variants: <span className="font-medium text-muted-foreground">{detail.variants.length}</span>
                  </span>
                  <span>
                    Metrics: <span className="font-medium text-muted-foreground">{detail.metrics.length}</span>
                  </span>
                  <span>
                    Data source:{' '}
                    <span className="font-medium text-muted-foreground">
                      {detail.dataSource?.datasetName || detail.dataSource?.datasetId || 'Not connected yet'}
                    </span>
                  </span>
                </div>
              </CardContent>
            </Card>

            {/* Phase 3 — Experiment Design: problem/objective/hypotheses,
                editable inline and saved via PATCH. */}
            <ExperimentDesignForm
              definition={detail}
              onSaved={(updated) => {
                setDetail(updated);
                setItems((prev) => prev.map((d) => (d.id === updated.id ? { ...d, name: updated.name } : d)));
              }}
            />

            {/* Phase 4 — Variants: Control + Treatment arms with allocation. */}
            <ExperimentVariantsForm
              definition={detail}
              onSaved={(updated) => setDetail(updated)}
            />

            {/* Phase 5 — Targeting: descriptive audience-filter metadata. */}
            <ExperimentTargetingForm
              definition={detail}
              onSaved={(updated) => setDetail(updated)}
            />

            {/* Phase 6 — Assignment: randomization unit + expected split,
                read from the Variants section above. */}
            <ExperimentAssignmentForm
              definition={detail}
              onSaved={(updated) => setDetail(updated)}
            />

            {/* Phase 7 — Metrics: Primary/Secondary/Guardrail. */}
            <ExperimentMetricsForm
              definition={detail}
              onSaved={(updated) => setDetail(updated)}
            />

            {/* Phase 8 — Data Source: connects this definition to one of
                the existing datasets (or a fresh upload), via the same
                classify pipeline as the Overview tab. */}
            <ExperimentDataSourceForm
              definition={detail}
              onSaved={(updated) => setDetail(updated)}
            />

            {/* Phase 9/10 — Visualization + History: run this definition
                through the EXISTING analysis engine and browse its past
                AnalysisRuns, each rendered with the same <ReportCard />
                used everywhere else. */}
            <ExperimentDefinitionRuns
              definition={detail}
              settings={settings}
              onDefinitionUpdated={(updated) => setDetail(updated)}
            />

            <Card className="border-dashed border-border-strong bg-secondary/60 shadow-none">
              <CardContent className="flex items-center justify-between gap-4 py-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Beaker className="h-4 w-4 text-muted-foreground" />
                  Prefer the full Overview workflow (prompt, live progress, follow-up chat)?
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  className="shrink-0 gap-1.5"
                  onClick={() => onContinueToAnalysis?.(detail.id)}
                >
                  Continue to Analysis
                  <ArrowRight className="h-3.5 w-3.5" />
                </Button>
              </CardContent>
            </Card>
          </div>
        ) : (
          !detailLoading &&
          !loading &&
          !error && (
            <Card className="border-border shadow-none">
              <CardContent className="flex items-center justify-center py-20 text-sm text-muted-foreground">
                {items.length > 0
                  ? 'Select an experiment to view its details'
                  : 'No experiments yet — create one to start planning'}
              </CardContent>
            </Card>
          )
        )}
      </div>

      {/* New Experiment dialog — Phase 2 scope: basic identity fields
          only. Problem/objective/hypotheses/variants/targeting/metrics
          are edited via the Experiment Design form in a later phase. */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>New Experiment</DialogTitle>
            <DialogDescription>
              Creates a DRAFT experiment in the Library. You can fill in the rest — hypotheses,
              variants, targeting, metrics — afterward.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div className="space-y-1.5">
              <Label htmlFor="new-exp-name">Experiment name *</Label>
              <Input
                id="new-exp-name"
                placeholder="e.g. Landing Page Redesign"
                value={form.name}
                onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="new-exp-area">Product area</Label>
              <Input
                id="new-exp-area"
                placeholder="e.g. Conversion Optimization"
                value={form.productArea}
                onChange={(e) => setForm((f) => ({ ...f, productArea: e.target.value }))}
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label htmlFor="new-exp-owner">Owner</Label>
                <Input
                  id="new-exp-owner"
                  placeholder="e.g. jane@company.com"
                  value={form.owner}
                  onChange={(e) => setForm((f) => ({ ...f, owner: e.target.value }))}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="new-exp-team">Team</Label>
                <Input
                  id="new-exp-team"
                  placeholder="e.g. Growth"
                  value={form.team}
                  onChange={(e) => setForm((f) => ({ ...f, team: e.target.value }))}
                />
              </div>
            </div>
            {createError && <p className="text-xs text-destructive">{createError}</p>}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)} disabled={creating}>
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              disabled={creating}
              className="gap-1.5 bg-primary text-white hover:bg-primary/90"
            >
              {creating && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
