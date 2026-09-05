'use client';

import { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  FileSpreadsheet,
  CheckCircle2,
  Sparkles,
  Play,
  Wand2,
  Loader2,
  XCircle,
  Database,
  ChevronDown,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent } from '@/components/ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ExecutionStepper } from '@/components/execution-stepper';
import { ReportCard } from '@/components/report-card';
import { FollowUpChat } from '@/components/follow-up-chat';
import { HypothesisForm } from '@/components/hypothesis-form';
import { ExperimentSetup } from '@/components/experiment-setup';
import { DatasetClassificationCard } from '@/components/dataset-classification-card';
import {
  classifyDataset,
  refreshClassification,
  analyzeExperimentStream,
  streamChatResponse,
  listRealDatasets,
  ApiError,
} from '@/lib/api';
import type { RealDatasetOption } from '@/lib/api';
import type {
  DatasetInfo,
  ExecutionStep,
  ExecutionStepStatus,
  ExperimentReport,
  ExperimentPlan,
  StepStatus,
  ChatMessage,
  Hypothesis,
  Settings,
  PipelineStreamEvent,
} from '@/lib/types';

/**
 * Placeholder label/group for a stage while it's `running`, before the
 * final `result` event's real ExecutionStep (with its real `detail` text)
 * arrives — see `runEvaluation`'s stage_started handling below. `id` here
 * matches the real ExecutionStep.id for that same stage (see
 * backend/app/api/routes_experiments.py's `_build_execution_steps`), so
 * the placeholder is seamlessly replaced, never duplicated, once the real
 * step list comes in.
 */
const STAGE_PLACEHOLDER: Record<string, Pick<ExecutionStep, 'label' | 'group'>> = {
  classifier: { label: 'Dataset Classification', group: 'Classifier' },
  planner: { label: 'Intent Planning', group: 'Planner' },
  funnel: { label: 'Funnel Analysis', group: 'Capability' },
  knowledge_base: { label: 'Knowledge Base Retrieval', group: 'Capability' },
  validation: { label: 'Data Quality Validation', group: 'Capability' },
  experiment: { label: 'Statistical Analysis', group: 'Capability' },
  guardrail: { label: 'Guardrail Check', group: 'Capability' },
  decision: { label: 'Report Generation', group: 'Decision Engine' },
};

interface WorkspaceViewProps {
  onSessionSaved: (name: string, report: ExperimentReport, experimentId: string) => void;
  settings: Settings;
  // Guardrail multiselect (below) lives here, not in ExperimentConfig,
  // because it needs `dataset.availableMetrics` — which only exists once
  // a dataset is loaded inside THIS component. ExperimentConfig itself
  // stays dataset-agnostic (cuped/bootstrap/model apply before any
  // dataset is even picked). Optional so this stays backward compatible
  // with any other caller that doesn't need guardrail selection.
  onSettingsChange?: (s: Settings) => void;
}

type Phase = 'idle' | 'running' | 'done';

export function WorkspaceView({ onSessionSaved, settings, onSettingsChange }: WorkspaceViewProps) {
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  // The primary-file-ONLY classification result, saved at the moment the
  // primary file/demo is classified — before any assignment file exists.
  // `dataset` (above) is what the banner actually shows and gets
  // recomputed once an assignment file is attached (see
  // handleAssignmentFile); this is what it reverts to if that assignment
  // file is removed again (see clearAssignmentFile).
  const [primaryOnlyDataset, setPrimaryOnlyDataset] = useState<DatasetInfo | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [isDemo, setIsDemo] = useState(false);
  // Real, published experiment datasets fetched from GET /datasets/real —
  // populated once on mount so the picker (rendered next to the demo
  // button) doesn't have to guess dataset_key values that only the
  // backend knows about.
  const [realDatasets, setRealDatasets] = useState<RealDatasetOption[]>([]);
  const [fileName, setFileName] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [phase, setPhase] = useState<Phase>('idle');
  const [statuses, setStatuses] = useState<Record<string, StepStatus>>({});
  const [executionSteps, setExecutionSteps] = useState<ExecutionStep[]>([]);
  const [report, setReport] = useState<ExperimentReport | null>(null);
  const [experimentId, setExperimentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hypothesis, setHypothesis] = useState<Hypothesis | null>(null);
  // The plan captured on the pre-experiment "Create Experiment" screen
  // (see ExperimentSetup) — kept around so it can pre-fill the
  // hypothesis/guardrails once a dataset is subsequently loaded. Never
  // sent to the backend directly; only its fields feed into Hypothesis /
  // settings.guardrailMetrics below.
  const [experimentPlan, setExperimentPlan] = useState<ExperimentPlan | null>(null);
  // Whether the pre-experiment planning card is collapsed (either
  // because the analyst finished it via onContinue, or explicitly
  // skipped it). Separate from experimentPlan being null/non-null so
  // "skip without planning" and "plan captured" are distinguishable.
  const [showSetupSkipped, setShowSetupSkipped] = useState(false);
  // Optional separate experiment-assignment dataset (user_id | variant),
  // uploaded through the SAME classifyDataset() call as the primary
  // dataset — see handleAssignmentFile below. Undefined/null for every
  // existing single-file flow; only ever set when the analyst
  // explicitly uploads a second file.
  const [assignmentDatasetId, setAssignmentDatasetId] = useState<string | null>(null);
  const [assignmentFileName, setAssignmentFileName] = useState<string | null>(null);
  const [isClassifyingAssignment, setIsClassifyingAssignment] = useState(false);
  const [assignmentError, setAssignmentError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const assignmentFileInputRef = useRef<HTMLInputElement>(null);

  // Fetch the list of available real/published experiment datasets once,
  // on mount. Failure here just means the picker stays empty (upload and
  // demo remain fully usable) — not worth surfacing as a page-level error.
  useEffect(() => {
    listRealDatasets()
      .then(setRealDatasets)
      .catch(() => setRealDatasets([]));
  }, []);

  const resetForNewDataset = () => {
    setReport(null);
    setExperimentId(null);
    setMessages([]);
    setPhase('idle');
    setExecutionSteps([]);
    setStatuses({});
    setError(null);
    // A hypothesis is scoped to one dataset/run — a freshly loaded
    // dataset (new upload, new demo toggle) shouldn't carry over a
    // hypothesis written against the previous one's metrics.
    setHypothesis(null);
    // An assignment file is scoped to the primary dataset it was
    // matched against — a new primary dataset invalidates any
    // previously uploaded assignment mapping.
    setAssignmentDatasetId(null);
    setAssignmentFileName(null);
    setAssignmentError(null);
    // Same reasoning as the hypothesis reset above — a guardrail
    // selection is scoped to the metrics of ONE dataset; carrying it
    // over to a newly loaded dataset could silently request a metric
    // name that means something completely different there (or simply
    // doesn't exist), which is exactly the kind of silent
    // misattribution this feature exists to prevent.
    onSettingsChange?.({ ...settings, guardrailMetrics: [] });
  };

  // Pre-fills the hypothesis/guardrails from a plan captured on the
  // "Create Experiment" screen (if any) once a dataset has just been
  // classified. Guardrail names are matched loosely (case-insensitive
  // substring) against this SPECIFIC dataset's own guardrailCandidates —
  // never applied blindly — since a plan written before any dataset was
  // selected can't know what that dataset will actually call things;
  // see resetForNewDataset's comment above for why a mismatch here would
  // be a silent-misattribution risk.
  const applyExperimentPlan = (loadedDataset: DatasetInfo) => {
    if (!experimentPlan) return;
    setHypothesis({
      statement: experimentPlan.statement,
      primaryMetric: experimentPlan.primaryMetric || loadedDataset.metricLabel,
      expectedDirection: experimentPlan.expectedDirection,
      expectedEffectRelative:
        experimentPlan.sampleSizeRequest?.mdeRelativePct != null
          ? experimentPlan.sampleSizeRequest.mdeRelativePct / 100
          : null,
      rationale: null,
    });
    const candidates = loadedDataset.guardrailCandidates ?? [];
    const matched = experimentPlan.guardrailMetricNames
      .map((planned) =>
        candidates.find((c) => c.toLowerCase().includes(planned.toLowerCase()))
      )
      .filter((c): c is string => Boolean(c));
    if (matched.length > 0) {
      onSettingsChange?.({ ...settings, guardrailMetrics: matched });
    }
  };

  const loadDemo = async () => {
    setIsClassifying(true);
    setError(null);
    try {
      const result = await classifyDataset({ useDemo: true, simulateLowQuality: false });
      setIsDemo(true);
      setDataset(result.dataset);
      setPrimaryOnlyDataset(result.dataset);
      setDatasetId(result.datasetId);
      setFileName(result.fileName);
      resetForNewDataset();
      applyExperimentPlan(result.dataset);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to load the demo dataset.');
    } finally {
      setIsClassifying(false);
    }
  };

  // Real, published experiment dataset — a third source alongside "Upload
  // CSV" and "Load Demo". Goes through the exact same classifyDataset()
  // call/endpoint as those two, just with `datasetKey` instead of `file`
  // or `useDemo`, so every downstream step (banner, hypothesis form,
  // analysis) is unaffected by which source produced the dataset.
  const loadRealDataset = async (key: string) => {
    setIsClassifying(true);
    setError(null);
    try {
      const result = await classifyDataset({ datasetKey: key });
      setIsDemo(false);
      setDataset(result.dataset);
      setPrimaryOnlyDataset(result.dataset);
      setDatasetId(result.datasetId);
      setFileName(result.fileName);
      resetForNewDataset();
      applyExperimentPlan(result.dataset);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to load the real dataset.');
    } finally {
      setIsClassifying(false);
    }
  };

  const handleFile = async (file: File) => {
    setIsClassifying(true);
    setError(null);
    try {
      const result = await classifyDataset({ file });
      setIsDemo(false);
      setDataset(result.dataset);
      setPrimaryOnlyDataset(result.dataset);
      setDatasetId(result.datasetId);
      setFileName(result.fileName);
      resetForNewDataset();
      applyExperimentPlan(result.dataset);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not parse this file.');
    } finally {
      setIsClassifying(false);
    }
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  // Uploads a separate experiment-assignment file (e.g. `user_id |
  // variant`) through the SAME classifyDataset() call/endpoint the
  // primary dataset uses — no parallel upload mechanism.
  //
  // Once uploaded, the Classifier Banner is immediately RECOMPUTED via
  // refreshClassification() — a read-only re-classify of the already-
  // stored primary dataset merged with this assignment dataset (same
  // enrich_with_assignment() the real analysis uses). Without this, the
  // banner stayed frozen at the primary-only "0 Variants" result from
  // handleFile/loadDemo above, even though /experiments/analyze would go
  // on to correctly resolve 2 variants — a misleading UX gap between what
  // the banner showed and what the analysis actually used.
  const handleAssignmentFile = async (file: File) => {
    setIsClassifyingAssignment(true);
    setAssignmentError(null);
    try {
      const result = await classifyDataset({ file });
      setAssignmentDatasetId(result.datasetId);
      setAssignmentFileName(result.fileName);
      if (datasetId && fileName) {
        const refreshed = await refreshClassification({
          datasetId,
          assignmentDatasetId: result.datasetId,
          fileName,
        });
        setDataset(refreshed.dataset);
      }
    } catch (err) {
      setAssignmentError(err instanceof ApiError ? err.message : 'Could not parse this file.');
    } finally {
      setIsClassifyingAssignment(false);
    }
  };

  const onAssignmentFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleAssignmentFile(f);
  };

  const clearAssignmentFile = () => {
    setAssignmentDatasetId(null);
    setAssignmentFileName(null);
    setAssignmentError(null);
    // Revert the Classifier Banner back to the primary-only result — it
    // was recomputed to include the assignment dataset in
    // handleAssignmentFile above; removing that assignment file means
    // the banner should reflect what the analysis will now actually run
    // against again (primary alone).
    if (primaryOnlyDataset) setDataset(primaryOnlyDataset);
  };

  const runEvaluation = async () => {
    if (!dataset || !datasetId) return;
    setPhase('running');
    setReport(null);
    setExperimentId(null);
    setMessages([]);
    setError(null);
    setExecutionSteps([]);
    setStatuses({});

    // Live progress state for the stream — kept as plain locals (not
    // React state) inside this closure and flushed into
    // setExecutionSteps/setStatuses on every event, so each SSE event
    // renders immediately without waiting on stale closures over
    // executionSteps/statuses.
    let liveSteps: ExecutionStep[] = [];
    const liveStatuses: Record<string, StepStatus> = {};

    const handleStreamEvent = (event: PipelineStreamEvent) => {
      if (event.type === 'stage_started') {
        liveStatuses[event.stage] = 'running';
        const existingIndex = liveSteps.findIndex((s) => s.id === event.stage);
        const placeholder = STAGE_PLACEHOLDER[event.stage] ?? {
          label: event.stage,
          group: 'Capability' as const,
        };
        const step: ExecutionStep = { id: event.stage, detail: event.message, ...placeholder };
        liveSteps = existingIndex >= 0
          ? liveSteps.map((s, i) => (i === existingIndex ? step : s))
          : [...liveSteps, step];
        setExecutionSteps(liveSteps);
        setStatuses({ ...liveStatuses });
      } else if (event.type === 'stage_completed') {
        liveStatuses[event.stage] = 'done';
        liveSteps = liveSteps.map((s) => (s.id === event.stage ? { ...s, detail: event.message } : s));
        setExecutionSteps(liveSteps);
        setStatuses({ ...liveStatuses });
      } else if (event.type === 'error') {
        // Never leave a stage looking permanently "running" — mark it
        // done-with-FAILED so the stepper's real-status icon (see
        // execution-stepper.tsx's RealStatusIcon) shows it clearly,
        // rather than a spinner that never resolves.
        liveStatuses[event.stage] = 'done';
        liveSteps = liveSteps.map((s) =>
          s.id === event.stage ? { ...s, detail: event.message, status: 'FAILED' as ExecutionStepStatus } : s
        );
        setExecutionSteps(liveSteps);
        setStatuses({ ...liveStatuses });
      }
      // 'result' and 'pipeline_completed' are handled after the promise
      // resolves below — the real ExecutionStep list from 'result'
      // replaces these placeholders wholesale (real detail text + real
      // SUCCESS/SKIPPED/WARNING/FAILED status, not the placeholder
      // approximation above).
    };

    try {
      const result = await analyzeExperimentStream(
        {
          datasetId,
          datasetName: fileName ?? datasetId,
          prompt,
          settings,
          assignmentDatasetId,
          hypothesis,
        },
        { onEvent: handleStreamEvent }
      );

      // The final result's executionSteps carry the FINAL real `detail`
      // text describing what the graph actually did — this replaces the
      // live placeholders above, so every stage ends up 'done' with its
      // authoritative label/detail/status, not the approximation shown
      // while it was running.
      setExecutionSteps(result.executionSteps);
      const doneStatuses: Record<string, StepStatus> = {};
      result.executionSteps.forEach((s) => (doneStatuses[s.id] = 'done'));
      setStatuses(doneStatuses);

      setReport(result.report);
      setExperimentId(result.experimentId);
      setPhase('done');
      onSessionSaved(
        `Exp #${Math.floor(100 + Math.random() * 900)}: ${prompt.trim() || 'Untitled Experiment'}`,
        result.report,
        result.experimentId
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The analysis failed. Please try again.');
      setPhase('idle');
    }
  };

  const handleFollowUp = async (content: string) => {
    if (!experimentId) return;
    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content };
    setMessages((prev) => [...prev, userMsg]);
    setIsChatLoading(true);

    // Placeholder id for the assistant message while it streams in — swapped
    // for the backend-persisted id/content once the terminal `done` event
    // arrives, so chat history stays consistent with what's actually stored.
    const streamingId = `a-streaming-${Date.now()}`;
    let started = false;

    try {
      const finalMsg = await streamChatResponse(
        { experimentId, message: content, model: settings.model },
        {
          onToken: (chunk) => {
            if (!started) {
              // First token: drop the typing-dots indicator and start a
              // real message bubble instead of waiting for the whole answer.
              started = true;
              setIsChatLoading(false);
              setMessages((prev) => [...prev, { id: streamingId, role: 'assistant', content: chunk }]);
            } else {
              setMessages((prev) =>
                prev.map((m) => (m.id === streamingId ? { ...m, content: m.content + chunk } : m))
              );
            }
          },
          // A mid-stream interruption is always followed by `done` with
          // whatever partial text was generated (see streamChatResponse's
          // docstring), so there's nothing terminal to do here — the UI
          // already shows the partial answer and keeps it once `done`
          // swaps in the persisted id below.
          onError: (message) => {
            console.error('[chat stream] interrupted:', message);
          },
        }
      );

      // Swap the streaming placeholder for the backend-persisted message
      // (real id, content that exactly matches what was persisted) — never
      // a second request for "the real answer": `finalMsg` IS the answer.
      setMessages((prev) => (started ? prev.map((m) => (m.id === streamingId ? finalMsg : m)) : [...prev, finalMsg]));
    } catch (err) {
      const errorMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content:
          err instanceof ApiError
            ? `Sorry, that follow-up failed: ${err.message}`
            : 'Sorry, something went wrong answering that follow-up.',
      };
      setMessages((prev) => (started ? prev.filter((m) => m.id !== streamingId) : prev).concat(errorMsg));
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="space-y-4">
    {/* Upload/config/execution steps stay narrow (form-like); the report
        below breaks out to full width once it's ready, to match the
        reference dashboard layout — see the closing wrapper below. */}
    <div className="mx-auto max-w-3xl space-y-4">
      {/* Pre-experiment planning — only shown before any dataset is
          loaded; once `dataset` is set, the analyst has moved past
          planning into review, and hypothesis/guardrails are edited via
          HypothesisForm below instead (scoped to that specific
          dataset's real metric names). */}
      {!dataset && !showSetupSkipped && (
        <ExperimentSetup
          onContinue={(plan) => {
            setExperimentPlan(plan);
            setShowSetupSkipped(true);
          }}
          onSkip={() => setShowSetupSkipped(true)}
        />
      )}
      {!dataset && showSetupSkipped && experimentPlan && (
        <div className="flex items-center justify-between rounded-lg border border-black/10 bg-neutral-50 px-4 py-2.5 text-[13px] text-black">
          <span>
            Experiment plan captured — it will pre-fill the hypothesis once you pick a dataset below.
          </span>
          <button
            type="button"
            className="text-neutral-400 underline hover:text-black"
            onClick={() => setShowSetupSkipped(false)}
          >
            Edit
          </button>
        </div>
      )}
      {!dataset && !experimentPlan && showSetupSkipped && (
        <button
          type="button"
          className="text-left text-xs text-neutral-400 underline hover:text-black"
          onClick={() => setShowSetupSkipped(false)}
        >
          Plan this experiment first
        </button>
      )}

      {/* Upload Zone */}
      <div
        className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-black/15 bg-white py-12 text-center"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-neutral-100 text-black">
          <UploadCloud className="h-5 w-5" />
        </div>
        <div>
          <p className="text-[13px] font-medium text-black">
            Drop your CSV or Excel file here, or click to browse
          </p>
          <p className="text-xs text-neutral-400">
            Aggregated or raw A/B test data — the classifier will detect the
            format
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            disabled={isClassifying}
            className="border-black/15"
          >
            <FileSpreadsheet className="mr-1.5 h-4 w-4" />
            Upload File
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            className="hidden"
            onChange={onFileChange}
          />
          <Button size="sm" onClick={loadDemo} disabled={isClassifying}>
            {isClassifying ? (
              <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="mr-1.5 h-4 w-4" />
            )}
            Load Demo A/B Dataset
          </Button>
        </div>
      </div>

      {/* Real, published experiment datasets — collapsed by default so
          the primary upload/demo actions above stay visually dominant.
          Renders nothing if the backend has none registered or the
          fetch failed (see the useEffect above). */}
      {realDatasets.length > 0 && (
        <Collapsible className="rounded-lg border border-black/10 bg-white">
          <CollapsibleTrigger className="flex w-full items-center justify-between px-4 py-3 text-left">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-neutral-500" />
              <span className="text-[13px] font-medium text-black">
                Real Experiment Datasets
              </span>
              <Badge variant="outline" className="text-[10px]">
                {realDatasets.length}
              </Badge>
            </div>
            <ChevronDown className="h-4 w-4 text-neutral-400 transition-transform data-[state=open]:rotate-180" />
          </CollapsibleTrigger>
          <CollapsibleContent className="border-t border-black/10 px-4 py-3">
            <p className="mb-3 text-xs text-neutral-400">
              Run the Copilot against a genuine published randomized experiment
              instead of a synthetic demo.
            </p>
            <div className="flex flex-col gap-2">
              {realDatasets.map((ds) => (
                <button
                  key={ds.key}
                  onClick={() => loadRealDataset(ds.key)}
                  disabled={isClassifying}
                  className="flex items-center gap-2.5 rounded-md border border-black/10 px-3 py-2 text-left text-[13px] text-black transition-colors hover:bg-neutral-50 disabled:opacity-50"
                >
                  {isClassifying ? (
                    <Loader2 className="h-4 w-4 shrink-0 animate-spin text-neutral-400" />
                  ) : (
                    <FileSpreadsheet className="h-4 w-4 shrink-0 text-neutral-500" />
                  )}
                  {ds.label}
                </button>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Error banner */}
      {error && (
        <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50/50 px-4 py-3 animate-slide-up">
          <XCircle className="h-4 w-4 shrink-0 text-red-600" />
          <p className="text-[13px] text-red-700">{error}</p>
        </div>
      )}

      {/* Classifier Banner */}
      {dataset && (
        <div className="flex items-center gap-3 rounded-lg border border-green-200 bg-green-50/50 px-4 py-3 animate-slide-up">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-green-600" />
          <div className="flex flex-1 flex-wrap items-center gap-x-4 gap-y-1 text-[13px]">
            <span className="font-medium text-black">
              Detected: {dataset.type}
            </span>
            <span className="text-neutral-500">{dataset.variants} Variants</span>
            <span className="text-neutral-500">
              {dataset.users.toLocaleString()} Users
            </span>
            <span className="text-neutral-500">
              Metric: {dataset.metricLabel}
            </span>
          </div>
          <Badge variant="outline" className="gap-1 text-[10px] border-black/10 text-neutral-500">
            <FileSpreadsheet className="h-3 w-3" />
            {fileName}
          </Badge>
        </div>
      )}

      {/* Full Dataset Classification — every field the classifier resolved
          or flagged, laid out explicitly. "Candidates" fields are naming/
          shape heuristics, never a promise of statistical eligibility —
          see DatasetInfo docstring on the backend. */}
      {dataset && <DatasetClassificationCard dataset={dataset} />}

      {/* Optional Assignment File (user_id | variant) — only useful once
          a primary dataset is loaded; merged deterministically onto it
          via enrich_with_assignment when the analysis runs. */}
      {dataset && (
        <div className="flex flex-wrap items-center gap-3 rounded-lg border border-black/10 bg-white px-4 py-3 animate-slide-up">
          <span className="text-[13px] font-medium text-black">
            Assignment file (optional)
          </span>
          <span className="text-xs text-neutral-400">
            user_id | variant — only needed if the primary dataset has no variant column
          </span>
          <div className="ml-auto flex items-center gap-2">
            {assignmentFileName ? (
              <>
                <Badge variant="outline" className="gap-1 text-[10px] border-black/10 text-neutral-500">
                  <FileSpreadsheet className="h-3 w-3" />
                  {assignmentFileName}
                </Badge>
                <Button variant="outline" size="sm" onClick={clearAssignmentFile} className="border-black/15">
                  Remove
                </Button>
              </>
            ) : (
              <Button
                variant="outline"
                size="sm"
                onClick={() => assignmentFileInputRef.current?.click()}
                disabled={isClassifyingAssignment}
                className="border-black/15"
              >
                {isClassifyingAssignment ? (
                  <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                ) : (
                  <FileSpreadsheet className="mr-1.5 h-4 w-4" />
                )}
                Upload Assignment File
              </Button>
            )}
            <input
              ref={assignmentFileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls"
              className="hidden"
              onChange={onAssignmentFileChange}
            />
          </div>
        </div>
      )}
      {assignmentError && (
        <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50/50 px-4 py-3 animate-slide-up">
          <XCircle className="h-4 w-4 shrink-0 text-red-600" />
          <p className="text-[13px] text-red-700">{assignmentError}</p>
        </div>
      )}

      {/* Hypothesis (Phase 1 — optional, collapsed by default) */}
      {dataset && (
        <HypothesisForm dataset={dataset} value={hypothesis} onChange={setHypothesis} />
      )}

      {/* Guardrail metrics (guardrail root-cause fix) — explicitly
          selected by the analyst, from this dataset's OWN detected
          metric columns only; never free-typed, never invented. This
          is a different concept from the "Guardrail Candidates" shown
          in the Dataset Classification card below (a naming-heuristic
          SUGGESTION) — selecting here is what actually gets evaluated. */}
      {dataset && dataset.availableMetrics && dataset.availableMetrics.length > 1 && (
        <Card className="border-black/10 shadow-none">
          <CardContent className="space-y-2 py-4">
            <div>
              <p className="text-[13px] font-medium text-black">Guardrail Metrics</p>
              <p className="text-xs text-neutral-400">
                Optional — pick metrics to check for unintended regressions alongside the primary
                metric. Leave unselected to skip guardrail evaluation.
              </p>
            </div>
            <div className="flex flex-wrap gap-x-4 gap-y-2 pt-1">
              {dataset.availableMetrics
                .filter((m) => m !== dataset.metricLabel)
                .map((metric) => {
                  const selected = settings.guardrailMetrics ?? [];
                  const checked = selected.includes(metric);
                  const isCandidate = dataset.guardrailCandidates?.includes(metric);
                  return (
                    <label
                      key={metric}
                      className="flex cursor-pointer items-center gap-1.5 text-[13px] text-black"
                    >
                      <Checkbox
                        checked={checked}
                        onCheckedChange={(v) => {
                          const next = v
                            ? [...selected, metric]
                            : selected.filter((m) => m !== metric);
                          onSettingsChange?.({ ...settings, guardrailMetrics: next });
                        }}
                      />
                      {metric}
                      {isCandidate && (
                        <Badge
                          variant="outline"
                          className="border-black/10 text-[10px] text-neutral-500"
                        >
                          suggested
                        </Badge>
                      )}
                    </label>
                  );
                })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Prompt Input */}
      {dataset && (
        <div className="rounded-lg border border-black/10 bg-white p-4 animate-slide-up">
          <label className="mb-2 block text-[13px] font-medium text-black">
            What would you like to evaluate?
          </label>
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g. Evaluate the checkout redesign experiment — is the variant ready to ship?"
            className="min-h-[60px] resize-none border-black/10 placeholder:text-neutral-400"
            rows={2}
          />
          <div className="mt-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-neutral-400">
              {settings.cuped && (
                <Badge variant="outline" className="gap-1 text-[10px] border-black/10 text-neutral-600">
                  <Wand2 className="h-3 w-3" />
                  CUPED
                </Badge>
              )}
              {settings.bootstrap && (
                <Badge variant="outline" className="gap-1 text-[10px] border-black/10 text-neutral-600">
                  <Wand2 className="h-3 w-3" />
                  Bootstrap
                </Badge>
              )}
              {!settings.cuped && !settings.bootstrap && (
                <span>Standard analysis (no variance reduction)</span>
              )}
            </div>
            <Button onClick={runEvaluation} disabled={phase === 'running' || isClassifying}>
              {phase === 'running' ? (
                <>
                  <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                  Evaluating...
                </>
              ) : (
                <>
                  <Play className="mr-1.5 h-4 w-4" />
                  Evaluate Experiment
                </>
              )}
            </Button>
          </div>
        </div>
      )}

      {/* Execution Stepper */}
      {(phase === 'running' || phase === 'done') && (
        <div className="rounded-lg border border-black/10 bg-white p-4 animate-fade-in">
          <p className="mb-3 text-[13px] font-semibold text-black">
            Execution Pipeline
          </p>
          {phase === 'running' ? (
            <div className="flex items-center gap-2 py-6 text-[13px] text-neutral-500">
              <Loader2 className="h-4 w-4 animate-spin" />
              Running Classifier → Planner → Capability nodes → Decision Engine...
            </div>
          ) : (
            <ExecutionStepper steps={executionSteps} statuses={statuses} />
          )}
        </div>
      )}
    </div>

      {/* Report Card — intentionally outside the max-w-3xl wrapper above,
          so it can use the full available width like the reference
          dashboard. */}
      {phase === 'done' && report && (
        <div className="space-y-4">
          <ReportCard
            report={report}
            datasetName={fileName ?? undefined}
            experimentId={experimentId ?? undefined}
            prompt={prompt}
          />
          <FollowUpChat messages={messages} onSend={handleFollowUp} isLoading={isChatLoading} />
        </div>
      )}
    </div>
  );
}
