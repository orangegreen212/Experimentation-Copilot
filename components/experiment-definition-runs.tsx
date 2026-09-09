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
import {
  FlaskConical,
  Loader2,
  ChevronRight,
  History as HistoryIcon,
  Rocket,
  Undo2,
  RotateCcw,
  Trash2,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ReportCard } from '@/components/report-card';
import {
  analyzeExperimentDefinition,
  deleteExperiment,
  followUpChat,
  getExperiment,
  listExperimentDefinitionRuns,
  updateExperimentDefinition,
  ApiError,
} from '@/lib/api';
import type { ChatMessage, ConfidenceLevel, ExperimentDefinition, ExperimentDetail, ExperimentSummary, Settings } from '@/lib/types';

interface ExperimentDefinitionRunsProps {
  definition: ExperimentDefinition;
  /** Applies the same run-level toggles (CUPED/bootstrap/model) as the
   *  Overview tab's Experiment Configuration panel — kept at the parent
   *  so it's one shared control, not a second, divergent settings UI. */
  settings: Settings;
  /** Called after a Decision action (Ship/Roll back/Keep iterating)
   *  successfully PATCHes the definition's status, so the caller
   *  (ExperimentLibrary) can update its own copy — same pattern every
   *  other *-form.tsx component's `onSaved` already uses. */
  onDefinitionUpdated: (updated: ExperimentDefinition) => void;
}

const CONFIDENCE_STYLES: Record<ConfidenceLevel, string> = {
  HIGH: 'border-success/25 bg-success/[0.08] text-success',
  MEDIUM: 'border-border bg-secondary text-muted-foreground',
  LOW: 'border-destructive/25 bg-destructive/[0.08] text-destructive',
};

const DECISION_STYLES: Record<string, string> = {
  GO: 'border-success/25 bg-success/[0.08] text-success',
  GO_WITH_CAUTION: 'border-warning/25 bg-warning/[0.08] text-warning',
  NO_GO: 'border-destructive/25 bg-destructive/[0.08] text-destructive',
  INCONCLUSIVE: 'border-border bg-secondary text-muted-foreground',
  INVALID: 'border-destructive/25 bg-destructive/[0.08] text-destructive',
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

export function ExperimentDefinitionRuns({ definition, settings, onDefinitionUpdated }: ExperimentDefinitionRunsProps) {
  const [runs, setRuns] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Same click-to-confirm delete pattern as history-view.tsx's session
  // list — reuses the same deleteExperiment(experimentId) API call,
  // since each run IS an Experiment row (run.experimentId), just
  // scoped here to one ExperimentDefinition's run list instead of the
  // global history sidebar.
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  // Chat for the currently-open run — same pattern as history-view.tsx's
  // "Ask Copilot Anything" chat, just previously missing entirely from
  // this Definition-scoped runs view (the button scrolled to a
  // `#follow-up-chat` element that was never rendered here).
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

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
      .then((d) => {
        setDetail(d);
        setMessages(d.chatMessages);
      })
      .catch((e) => {
        setDetail(null);
        setMessages([]);
        setDetailError(e instanceof ApiError ? e.message : 'Could not load this analysis run.');
      })
      .finally(() => setDetailLoading(false));
  }, [selectedId]);

  const handleFollowUp = async (content: string) => {
    if (!selectedId) return;
    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content };
    setMessages((prev) => [...prev, userMsg]);
    setIsChatLoading(true);
    setChatError(null);
    try {
      const reply = await followUpChat({ experimentId: selectedId, message: content, model: settings.model });
      setMessages((prev) => [...prev, reply]);
    } catch (e) {
      setChatError(e instanceof ApiError ? e.message : 'Could not get a response. Please try again.');
      setMessages((prev) => prev.filter((m) => m.id !== userMsg.id));
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleDeleteRun = async (experimentId: string) => {
    if (confirmingId !== experimentId) {
      // First click just arms the confirmation — avoids an accidental
      // one-click delete on a row the user only meant to open.
      setConfirmingId(experimentId);
      return;
    }
    setConfirmingId(null);
    setDeletingId(experimentId);
    setDeleteError(null);
    try {
      await deleteExperiment(experimentId);
      setRuns((prev) => prev.filter((r) => r.experimentId !== experimentId));
      setSelectedId((prev) => (prev === experimentId ? null : prev));
    } catch (e) {
      setDeleteError(e instanceof ApiError ? e.message : 'Could not delete this analysis run.');
    } finally {
      setDeletingId(null);
    }
  };

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

  const [decisionSaving, setDecisionSaving] = useState<'shipped' | 'completed' | 'ready' | null>(null);
  const [decisionError, setDecisionError] = useState<string | null>(null);

  const handleDecision = async (status: 'shipped' | 'completed' | 'ready') => {
    setDecisionSaving(status);
    setDecisionError(null);
    try {
      const updated = await updateExperimentDefinition(definition.id, { status });
      onDefinitionUpdated(updated);
    } catch (e) {
      setDecisionError(e instanceof ApiError ? e.message : 'Could not update this experiment.');
    } finally {
      setDecisionSaving(null);
    }
  };

  return (
    <Card className="border-border shadow-none">
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <div className="flex items-center gap-2">
          <HistoryIcon className="h-4 w-4 text-foreground" />
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
          className="shrink-0 gap-1.5 bg-primary text-white hover:bg-primary/90"
        >
          {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FlaskConical className="h-3.5 w-3.5" />}
          Run analysis
        </Button>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {runError && <p className="text-xs text-destructive">{runError}</p>}
        {loadError && <p className="text-xs text-destructive">{loadError}</p>}
        {deleteError && <p className="text-xs text-destructive">{deleteError}</p>}

        {loading && (
          <div className="flex items-center gap-2 py-4 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading past runs...
          </div>
        )}

        {!loading && runs.length === 0 && (
          <div className="rounded-md border border-dashed border-border-strong bg-secondary/60 py-6 text-center text-[13px] text-muted-foreground">
            No analysis runs yet for this experiment.
          </div>
        )}

        <div className="space-y-1.5">
          {runs.map((run, i) => {
            const active = selectedId === run.experimentId;
            const isConfirming = confirmingId === run.experimentId;
            const isDeleting = deletingId === run.experimentId;
            return (
              <div key={run.experimentId} className="group relative">
                <button
                  type="button"
                  onClick={() => setSelectedId(active ? null : run.experimentId)}
                  className={cn(
                    'flex w-full items-center justify-between gap-3 rounded-lg border p-3 pr-10 text-left transition-colors',
                    active ? 'border-border-strong bg-secondary' : 'border-border hover:bg-secondary'
                  )}
                >
                  <div className="min-w-0">
                    <p className="truncate text-[13px] font-medium text-foreground">{runLabel(i, runs.length)}</p>
                    <p className="mt-0.5 truncate text-xs text-muted-foreground">
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
                      className={cn('h-3.5 w-3.5 text-muted-foreground/50 transition-transform', active && 'rotate-90')}
                    />
                  </div>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteRun(run.experimentId);
                  }}
                  disabled={isDeleting}
                  title={isConfirming ? 'Click again to confirm delete' : 'Delete this analysis run'}
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
                {isConfirming && !isDeleting && (
                  <p className="mt-1 text-xs font-medium text-destructive">
                    Click the trash icon again to permanently delete this run
                  </p>
                )}
              </div>
            );
          })}
        </div>

        {/* "Make a decision" — a real action, not just a read-only
            `decision` label buried in the report (own take on that
            idea, not Amplitude's flow: three explicit outcomes that
            actually move this definition's status, since nothing
            previously did — see lib/experiment-readiness.ts's
            docstring on why status had no teeth before this). Shown
            once at least one run exists to decide about; each button
            reflects/disables against the CURRENT status rather than
            assuming a fresh decision is always being made. */}
        {runs.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3">
            <span className="mr-1 text-[11px] font-medium text-muted-foreground">
              Decision
            </span>
            <Button
              size="sm"
              variant="outline"
              className="h-7 gap-1.5 border-success/25 text-xs text-success hover:bg-success/[0.08]"
              disabled={decisionSaving !== null || definition.status === 'shipped'}
              onClick={() => handleDecision('shipped')}
            >
              {decisionSaving === 'shipped' ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Rocket className="h-3 w-3" />
              )}
              {definition.status === 'shipped' ? 'Shipped' : 'Ship the winner'}
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-7 gap-1.5 text-xs text-muted-foreground"
              disabled={decisionSaving !== null || definition.status === 'completed'}
              onClick={() => handleDecision('completed')}
            >
              {decisionSaving === 'completed' ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Undo2 className="h-3 w-3" />
              )}
              Roll back
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-7 gap-1.5 text-xs text-muted-foreground"
              disabled={decisionSaving !== null || definition.status === 'ready'}
              onClick={() => handleDecision('ready')}
            >
              {decisionSaving === 'ready' ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <RotateCcw className="h-3 w-3" />
              )}
              Keep iterating
            </Button>
            {decisionError && <span className="text-xs text-destructive">{decisionError}</span>}
          </div>
        )}

        {detailError && <p className="text-xs text-destructive">{detailError}</p>}
        {detailLoading && (
          <div className="flex items-center gap-2 py-4 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading report...
          </div>
        )}

        {!detailLoading && detail && (
          <div className="space-y-4 pt-2">
            {chatError && <p className="text-xs text-destructive">{chatError}</p>}
            <ReportCard
              report={detail.report}
              datasetName={detail.datasetName}
              experimentId={detail.experimentId}
              prompt={detail.userPrompt}
              chat={{ messages, onSend: handleFollowUp, isLoading: isChatLoading }}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
