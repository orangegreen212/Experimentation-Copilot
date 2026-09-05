/**
 * API service layer — the ONLY place in the frontend that calls
 * fetch() or knows the backend's base URL. Components never import
 * fetch directly; they call these functions and get back the exact
 * TypeScript interfaces from lib/types.ts.
 *
 * The backend is the single source of truth: this file does no
 * business logic (no statistics, no confidence scoring, no
 * formatting) — it only sends requests and shapes responses into the
 * types the components already expect.
 */

import type {
  AvailableModelsResponse,
  ChatMessage,
  ChatStreamEvent,
  DatasetInfo,
  ExecutionStep,
  ExperimentDefinition,
  ExperimentDefinitionCreateRequest,
  ExperimentDefinitionSummary,
  ExperimentDefinitionUpdateRequest,
  ExperimentDetail,
  ExperimentReport,
  ExperimentSummary,
  Hypothesis,
  PipelineStreamEvent,
  RelatedExperiment,
  Settings,
  SystemInfo,
  SampleSizePlanRequest,
  SampleSizePlanResponse,
} from './types';

/**
 * Base URL for every backend call. Two modes:
 *
 * - LOCAL DEV: set NEXT_PUBLIC_API_URL=http://localhost:8000 in
 *   .env.local, matching the separately-running `uvicorn app.main:app`
 *   (run from backend/) — its routes have no /api prefix.
 * - PRODUCTION (Vercel): leave NEXT_PUBLIC_API_URL unset. Defaults to
 *   the relative path '/api', which vercel.json rewrites to
 *   api/index.py — the same FastAPI app, mounted under /api there.
 *   Same-origin, so no CORS needed in production either.
 */
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? '/api';

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === 'string') return body.detail;
    return JSON.stringify(body);
  } catch {
    return response.statusText || `Request failed with status ${response.status}`;
  }
}

// ---------------------------------------------------------------------------
// POST /datasets/classify
// ---------------------------------------------------------------------------

export interface ClassifyDatasetResult {
  dataset: DatasetInfo;
  datasetId: string;
  fileName: string;
}

interface ClassifyDatasetParams {
  file?: File;
  useDemo?: boolean;
  simulateLowQuality?: boolean;
  // Loads a real, published experiment dataset bundled on the backend
  // (see GET /datasets/real for the available keys) — a third source
  // alongside file upload and the synthetic demo. Mutually exclusive
  // with `file` and `useDemo`, same as those are with each other.
  datasetKey?: string;
}

export async function classifyDataset({
  file,
  useDemo,
  simulateLowQuality,
  datasetKey,
}: ClassifyDatasetParams): Promise<ClassifyDatasetResult> {
  const formData = new FormData();
  if (file) formData.append('file', file);
  if (useDemo) formData.append('use_demo', 'true');
  if (simulateLowQuality) formData.append('simulate_low_quality', 'true');
  if (datasetKey) formData.append('dataset_key', datasetKey);

  const response = await fetch(`${API_URL}/datasets/classify`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// GET /datasets/real — list available real/published experiment datasets
// ---------------------------------------------------------------------------

export interface RealDatasetOption {
  key: string;
  label: string;
}

export async function listRealDatasets(): Promise<RealDatasetOption[]> {
  const response = await fetch(`${API_URL}/datasets/real`);
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// POST /experiments/plan-sample-size
// ---------------------------------------------------------------------------

export async function planSampleSize(
  request: SampleSizePlanRequest
): Promise<SampleSizePlanResponse> {
  const response = await fetch(`${API_URL}/experiments/plan-sample-size`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// POST /datasets/classify — REFRESH mode (Classifier Banner recompute)
// ---------------------------------------------------------------------------

interface RefreshClassificationParams {
  datasetId: string;
  assignmentDatasetId?: string | null;
  fileName: string;
}

/**
 * Re-runs classification against an ALREADY-STORED primary dataset,
 * optionally merged with an already-stored assignment dataset — no file
 * re-upload. Used to keep the Classifier Banner in sync with what
 * /experiments/analyze will actually resolve once an assignment file is
 * attached (or removed) — see workspace-view.tsx's handleAssignmentFile /
 * clearAssignmentFile. Returns only `dataset` (the caller already has
 * `datasetId` and `fileName` and must not overwrite them with this call's
 * echoed-back values).
 */
export async function refreshClassification({
  datasetId,
  assignmentDatasetId,
  fileName,
}: RefreshClassificationParams): Promise<{ dataset: DatasetInfo }> {
  const formData = new FormData();
  formData.append('dataset_id', datasetId);
  if (assignmentDatasetId) formData.append('assignment_dataset_id', assignmentDatasetId);
  formData.append('file_name', fileName);

  const response = await fetch(`${API_URL}/datasets/classify`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// POST /experiments/analyze
// ---------------------------------------------------------------------------

export interface AnalyzeExperimentResult {
  experimentId: string;
  report: ExperimentReport;
  executionSteps: ExecutionStep[];
  relatedExperiments: RelatedExperiment[];
  // Part 1 of the pipeline-instrumentation task — per-stage timing.
  // Optional for backward compatibility with any cached/mocked
  // response shaped before this field existed.
  stageTimings?: { stage: string; status: string; durationMs: number; error: string | null }[];
}

interface AnalyzeExperimentParams {
  datasetId: string;
  datasetName: string;
  prompt: string;
  settings: Settings;
  // Optional separate experiment-assignment dataset id — the `datasetId`
  // returned by a second, independent classifyDataset() call against a
  // `user_id | variant`-shaped file. Omitted (the default) is fully
  // backward compatible with every existing single-file flow.
  assignmentDatasetId?: string | null;
  // Phase 1 — optional. Omitting it is fully backward compatible.
  hypothesis?: Hypothesis | null;
}

export async function analyzeExperiment({
  datasetId,
  datasetName,
  prompt,
  settings,
  assignmentDatasetId,
  hypothesis,
}: AnalyzeExperimentParams): Promise<AnalyzeExperimentResult> {
  const response = await fetch(`${API_URL}/experiments/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      datasetId,
      datasetName,
      prompt,
      settings: {
        cuped: settings.cuped,
        bootstrap: settings.bootstrap,
        // Only sent when the user actually picked something other
        // than the backend default in the model dropdown — omitted
        // (rather than sent as undefined) keeps the request body
        // identical to before this field existed for the default case.
        ...(settings.model ? { model: settings.model } : {}),
        // Guardrail root-cause fix — only sent when the analyst actually
        // selected at least one guardrail metric; omitted (rather than
        // sent as []) keeps the request body identical to before this
        // field existed for every existing flow, and the backend
        // already treats omitted/empty identically (NOT_SPECIFIED).
        ...(settings.guardrailMetrics && settings.guardrailMetrics.length > 0
          ? { guardrailMetrics: settings.guardrailMetrics }
          : {}),
      },
      // Only sent when an assignment file was actually uploaded —
      // omitted (rather than sent as null) keeps the request body
      // identical to before this field existed for every existing flow.
      ...(assignmentDatasetId ? { assignmentDatasetId } : {}),
      // Only sent when the analyst actually filled in the hypothesis
      // form — omitted (rather than sent as null) keeps the request
      // body identical to before this phase for every existing flow.
      ...(hypothesis ? { hypothesis } : {}),
    }),
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// POST /experiments/analyze/stream — same pipeline as analyzeExperiment()
// above, but with live per-stage progress (Server-Sent Events) instead of
// one blocking wait. Same request shape; the final result is identical to
// what analyzeExperiment() returns, just delivered as the stream's terminal
// `result` event instead of the whole-response body.
// ---------------------------------------------------------------------------

interface AnalyzeExperimentStreamCallbacks {
  onEvent?: (event: PipelineStreamEvent) => void;
}

/**
 * Runs the same analysis as `analyzeExperiment`, but streams progress via
 * SSE and resolves with the same final `AnalyzeExperimentResult` shape once
 * the pipeline's `result` event arrives. `onEvent` fires for every event as
 * it's parsed off the wire (stage_started/stage_completed/error/result/
 * pipeline_completed — see PipelineStreamEvent in lib/types.ts), so a caller
 * can render live progress without waiting for the promise to resolve.
 *
 * Throws an `ApiError` if the HTTP request itself fails (e.g. a 404 before
 * the stream even opens), or a plain `Error` if the pipeline reports an
 * `error` event and never produces a `result` (e.g. an unknown dataset id
 * caught inside the pipeline rather than at the route's top-level checks —
 * see routes_experiments.py's `analyze_experiment_stream`).
 */
export async function analyzeExperimentStream(
  { datasetId, datasetName, prompt, settings, assignmentDatasetId, hypothesis }: AnalyzeExperimentParams,
  { onEvent }: AnalyzeExperimentStreamCallbacks = {}
): Promise<AnalyzeExperimentResult> {
  const response = await fetch(`${API_URL}/experiments/analyze/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      datasetId,
      datasetName,
      prompt,
      settings: {
        cuped: settings.cuped,
        bootstrap: settings.bootstrap,
        ...(settings.model ? { model: settings.model } : {}),
        ...(settings.guardrailMetrics && settings.guardrailMetrics.length > 0
          ? { guardrailMetrics: settings.guardrailMetrics }
          : {}),
      },
      ...(assignmentDatasetId ? { assignmentDatasetId } : {}),
      ...(hypothesis ? { hypothesis } : {}),
    }),
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  if (!response.body) {
    throw new Error('Streaming is not supported by this browser/environment.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: AnalyzeExperimentResult | null = null;
  let lastError: string | null = null;

  try {
    // Server-Sent Events framing: each event is one or more `data: ...`
    // lines followed by a blank line. This backend only ever sends a
    // single `data:` line per event (see routes_experiments.py), so
    // splitting on blank-line-separated blocks and stripping the `data:`
    // prefix per line is sufficient — no need for a full SSE parser.
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() ?? '';

      for (const block of blocks) {
        const dataLines = block
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice('data:'.length).trim())
          .filter(Boolean);
        if (dataLines.length === 0) continue;

        let event: PipelineStreamEvent;
        try {
          event = JSON.parse(dataLines.join('')) as PipelineStreamEvent;
        } catch (parseError) {
          // A single malformed/unexpectedly-split frame (e.g. proxy
          // buffering cutting a chunk mid-JSON) must not abort the
          // whole analysis or masquerade as "no result" — it's a
          // transport hiccup, not a backend-reported failure, so it's
          // logged and skipped rather than surfaced as `lastError`
          // (which is reserved for legitimate backend `error` events
          // below). Any later well-formed `result`/`error` event in
          // the same stream is still processed normally.
          console.error('[analyzeExperimentStream] failed to parse SSE event, skipping:', block, parseError);
          continue;
        }

        onEvent?.(event);

        if (event.type === 'result') {
          result = event.data;
        } else if (event.type === 'error') {
          lastError = event.message;
        }
      }
    }
  } finally {
    // Deterministic cleanup on every exit path — normal completion,
    // an error thrown out of the loop (e.g. `reader.read()` itself
    // failing on a network drop), or the caller's own exception —
    // so the underlying byte stream is always released instead of
    // relying on GC to eventually close it.
    reader.releaseLock();
  }

  if (result) return result;
  throw new Error(lastError ?? 'The analysis stream ended without a result.');
}

// ---------------------------------------------------------------------------
// Experiment History — GET /experiments, GET /experiments/{id}
// ---------------------------------------------------------------------------

export async function listExperiments(): Promise<ExperimentSummary[]> {
  const response = await fetch(`${API_URL}/experiments`);
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

export async function getExperiment(experimentId: string): Promise<ExperimentDetail> {
  const response = await fetch(`${API_URL}/experiments/${experimentId}`);
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

/**
 * DELETE /experiments/{id} — removes a persisted experiment (and its
 * chat history) from Experiment History. Backend returns 404 if the
 * id is unknown (already deleted / never existed), which we surface
 * as a normal ApiError so the caller can decide how to handle it.
 */
export async function deleteExperiment(experimentId: string): Promise<void> {
  const response = await fetch(`${API_URL}/experiments/${experimentId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
}

/**
 * Structured lookup of prior runs against this exact dataset — used on
 * the New Experiment screen right after classification, before the
 * user hits Run, so "this dataset was reviewed before" context shows
 * up front. Same underlying data as AnalyzeExperimentResult.relatedExperiments.
 *
 * Takes datasetId, NOT the file name: dataset_name is a display label
 * only (two different uploads can share a file name) — the backend
 * keys relation by dataset_id, the id the classify step returns.
 */
export async function getRelatedExperiments(datasetId: string): Promise<RelatedExperiment[]> {
  const response = await fetch(
    `${API_URL}/experiments/related/by-dataset?dataset_id=${encodeURIComponent(datasetId)}`
  );
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// GET /system/info — read-only backend config (model, backends)
// ---------------------------------------------------------------------------

export async function getSystemInfo(): Promise<SystemInfo> {
  const response = await fetch(`${API_URL}/system/info`);
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// GET /system/models — curated, user-selectable LLM allowlist
// ---------------------------------------------------------------------------

export async function getAvailableModels(): Promise<AvailableModelsResponse> {
  const response = await fetch(`${API_URL}/system/models`);
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// POST /experiments/{experimentId}/chat
// ---------------------------------------------------------------------------

interface FollowUpChatParams {
  experimentId: string;
  message: string;
  // Optional per-message LLM override — same allowlist as Settings.model.
  model?: string;
}

export async function followUpChat({
  experimentId,
  message,
  model,
}: FollowUpChatParams): Promise<ChatMessage> {
  const response = await fetch(`${API_URL}/experiments/${experimentId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      experimentId,
      message,
      ...(model ? { model } : {}),
    }),
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }

  const data = await response.json();
  return data.message;
}

// ---------------------------------------------------------------------------
// POST /experiments/{experimentId}/chat/stream — same contract as
// followUpChat() above (grounded in the stored report, persisted on the
// backend), but delivers the answer as SSE tokens instead of one blocking
// wait. Uses fetch + ReadableStream, not EventSource: this is a POST with a
// JSON body (experimentId/message/model), and EventSource only supports GET.
// ---------------------------------------------------------------------------

interface StreamChatResponseParams {
  experimentId: string;
  message: string;
  model?: string;
}

interface StreamChatResponseCallbacks {
  /** Fires for every token chunk, in arrival order — append, don't replace. */
  onToken?: (content: string) => void;
  /**
   * Fires if the backend reports a mid-stream failure (see the route's
   * docstring: this can only happen AFTER at least one token already
   * arrived). Purely informational — a `done` event with whatever partial
   * text was generated always follows, so callers don't need to treat this
   * as terminal themselves.
   */
  onError?: (message: string) => void;
}

/**
 * Streaming counterpart to `followUpChat`. Resolves with the same final
 * `ChatMessage` shape once the stream's terminal `done` event arrives —
 * that message is already persisted on the backend, so callers must NOT
 * make a second `followUpChat`/`getExperiment` call to fetch "the real
 * answer"; the accumulated `onToken` text and the resolved `done.message`
 * describe the exact same content, just assembled two different ways.
 *
 * Throws an `ApiError` if the HTTP request itself fails (e.g. unknown
 * experimentId, before the stream even opens) — same as `followUpChat`.
 * Does not throw for the in-stream `error` event, since (per the route's
 * docstring) that event is always followed by a `done` with partial
 * content rather than the stream ending without one; use `onError` if the
 * caller wants to surface that interruption to the user.
 */
export async function streamChatResponse(
  { experimentId, message, model }: StreamChatResponseParams,
  { onToken, onError }: StreamChatResponseCallbacks = {}
): Promise<ChatMessage> {
  const response = await fetch(`${API_URL}/experiments/${experimentId}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      experimentId,
      message,
      ...(model ? { model } : {}),
    }),
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  if (!response.body) {
    throw new Error('Streaming is not supported by this browser/environment.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let doneMessage: ChatMessage | null = null;

  try {
    // Same SSE framing/buffering as analyzeExperimentStream() above: one
    // reader.read() call has no relationship to SSE event boundaries — a
    // single chunk can contain zero, one, or several complete `data: ...\n\n`
    // blocks, and the last block in any given chunk is frequently cut off
    // mid-event. `buffer` carries that trailing partial block forward to be
    // completed by the next chunk, so events are only ever parsed once a
    // full blank-line-terminated block has arrived — never by JSON.parse-ing
    // a raw chunk directly.
    while (true) {
      const { done: streamDone, value } = await reader.read();
      if (streamDone) break;
      buffer += decoder.decode(value, { stream: true });

      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() ?? '';

      for (const block of blocks) {
        const dataLines = block
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice('data:'.length).trim())
          .filter(Boolean);
        if (dataLines.length === 0) continue;

        let event: ChatStreamEvent;
        try {
          event = JSON.parse(dataLines.join('')) as ChatStreamEvent;
        } catch (parseError) {
          // A transport-level split hiccup, not a backend-reported failure
          // — log and skip this frame rather than aborting the whole
          // answer (same reasoning as analyzeExperimentStream() above).
          console.error('[streamChatResponse] failed to parse SSE event, skipping:', block, parseError);
          continue;
        }

        if (event.type === 'token') {
          onToken?.(event.content);
        } else if (event.type === 'error') {
          onError?.(event.message);
        } else if (event.type === 'done') {
          doneMessage = event.message;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (doneMessage) return doneMessage;
  throw new Error('The chat stream ended without a response.');
}

// ---------------------------------------------------------------------------
// ExperimentDefinition CRUD — Experiment Platform layer, Phase 2.
// Backend: app/api/routes_experiment_definitions.py. Deliberately
// separate from the /experiments/* calls above: a definition has its
// own lifecycle independent of any analysis run (see lib/types.ts's
// ExperimentDefinition docstring).
// ---------------------------------------------------------------------------

export async function createExperimentDefinition(
  request: ExperimentDefinitionCreateRequest
): Promise<ExperimentDefinition> {
  const response = await fetch(`${API_URL}/experiment-definitions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

export async function listExperimentDefinitions(): Promise<ExperimentDefinitionSummary[]> {
  const response = await fetch(`${API_URL}/experiment-definitions`);
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

export async function getExperimentDefinition(definitionId: string): Promise<ExperimentDefinition> {
  const response = await fetch(`${API_URL}/experiment-definitions/${definitionId}`);
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

export async function updateExperimentDefinition(
  definitionId: string,
  request: ExperimentDefinitionUpdateRequest
): Promise<ExperimentDefinition> {
  const response = await fetch(`${API_URL}/experiment-definitions/${definitionId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
  return response.json();
}

export async function deleteExperimentDefinition(definitionId: string): Promise<void> {
  const response = await fetch(`${API_URL}/experiment-definitions/${definitionId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status);
  }
}
