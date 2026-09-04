'use client';

import { useEffect, useState } from 'react';
import { History, ChevronRight, Loader2, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { ReportCard } from '@/components/report-card';
import { FollowUpChat } from '@/components/follow-up-chat';
import { RelatedExperiments } from '@/components/related-experiments';
import { listExperiments, getExperiment, deleteExperiment, followUpChat, ApiError } from '@/lib/api';
import type { ChatMessage, ConfidenceLevel, ExperimentDetail, ExperimentSummary } from '@/lib/types';

const CONFIDENCE_STYLES: Record<ConfidenceLevel, { badge: string; text: string }> = {
  HIGH: { badge: 'border-green-200 bg-green-50 text-green-700', text: 'text-green-600' },
  MEDIUM: { badge: 'border-black/10 bg-neutral-100 text-neutral-600', text: 'text-neutral-400' },
  LOW: { badge: 'border-red-200 bg-red-50 text-red-700', text: 'text-red-600' },
};

interface HistoryViewProps {
  /** Bump this to force a refetch of the list (e.g. after saving a new experiment). */
  refreshKey?: number;
  /** Pre-selects this experiment when the view first loads (e.g. jumped
   *  here from the Datasets tab's "View experiments" link). */
  initialExperimentId?: string;
}

export function HistoryView({ refreshKey, initialExperimentId }: HistoryViewProps) {
  const [sessions, setSessions] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(initialExperimentId ?? null);
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    listExperiments()
      .then((list) => {
        setSessions(list);
        setError(null);
        // Keep the current selection if it still exists; otherwise pick the newest.
        setSelectedId((prev) =>
          prev && list.some((s) => s.experimentId === prev) ? prev : (list[0]?.experimentId ?? null)
        );
      })
      .catch((e) => {
        // An API error is NOT an empty history. Keep the list empty but
        // expose a dedicated error state so the UI never reports 0 sessions
        // as if the backend had successfully returned an empty list.
        setSessions([]);
        setSelectedId(null);
        setError('Could not load experiment history. Please try again.');
      })
      .finally(() => setLoading(false));
  }, [refreshKey]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      setMessages([]);
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
        setDetailError(e instanceof ApiError ? e.message : 'Could not load this experiment.');
      })
      .finally(() => setDetailLoading(false));
  }, [selectedId]);

  const handleDelete = async (experimentId: string) => {
    if (confirmingId !== experimentId) {
      // First click just arms the confirmation — avoids an accidental
      // one-click delete on a row the user only meant to select.
      setConfirmingId(experimentId);
      return;
    }
    setConfirmingId(null);
    setDeletingId(experimentId);
    try {
      await deleteExperiment(experimentId);
      setSessions((prev) => {
        const next = prev.filter((s) => s.experimentId !== experimentId);
        setSelectedId((prevSelected) =>
          prevSelected === experimentId ? (next[0]?.experimentId ?? null) : prevSelected
        );
        return next;
      });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not delete this experiment.');
    } finally {
      setDeletingId(null);
    }
  };

  const handleFollowUp = async (content: string) => {
    if (!selectedId) return;
    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content };
    setMessages((prev) => [...prev, userMsg]);
    try {
      const reply = await followUpChat({ experimentId: selectedId, message: content });
      setMessages((prev) => [...prev, reply]);
    } catch (e) {
      const detailMsg =
        e instanceof ApiError ? e.message : 'Follow-up chat is currently unavailable.';
      setMessages((prev) => [
        ...prev,
        { id: `a-${Date.now()}`, role: 'assistant', content: detailMsg },
      ]);
    }
  };

  return (
    <div className="flex h-full gap-6">
      {/* Session list */}
      <div className="flex w-72 shrink-0 flex-col">
        <div className="mb-3 flex items-center gap-2">
          <History className="h-4 w-4 text-black" />
          <h2 className="text-[13px] font-semibold text-black">Past Sessions</h2>
          <Badge variant="outline" className="ml-auto text-[10px] border-black/10 text-neutral-500">
            {!loading && !error ? sessions.length : '—'}
          </Badge>
        </div>
        {loading && (
          <div className="flex items-center gap-2 py-6 text-xs text-neutral-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading experiment history...
          </div>
        )}
        {!loading && error && (
          <Card className="border-red-200 bg-red-50 shadow-none">
            <CardContent className="py-3 text-[13px] text-red-700">{error}</CardContent>
          </Card>
        )}
        <div className="space-y-px">
          {sessions.map((session) => {
            const styles = CONFIDENCE_STYLES[session.confidence];
            const active = selectedId === session.experimentId;
            const isDeleting = deletingId === session.experimentId;
            const isConfirming = confirmingId === session.experimentId;
            return (
              <div
                key={session.experimentId}
                className={cn(
                  'group relative w-full border-l-2 px-3 py-3 text-left transition-colors',
                  active ? 'border-black bg-neutral-50' : 'border-transparent hover:bg-neutral-50'
                )}
              >
                <button
                  onClick={() => {
                    setConfirmingId(null);
                    setSelectedId(session.experimentId);
                  }}
                  className="block w-full text-left"
                >
                  <div className="flex items-center justify-between pr-7">
                    <span className="truncate text-[15px] font-medium text-black">
                      {session.datasetName}
                    </span>
                    <ChevronRight
                      className={cn(
                        'h-4 w-4 shrink-0 transition-colors',
                        active ? 'text-black' : 'text-neutral-300'
                      )}
                    />
                  </div>
                  <p className="mt-0.5 truncate text-sm text-neutral-400">
                    {new Date(session.createdAt).toLocaleString()} · {session.primaryMetric}
                  </p>
                  <p className="mt-0.5 truncate text-sm text-neutral-500">{session.userPrompt}</p>
                  <div className="mt-1.5 flex items-center gap-2">
                    <Badge variant="outline" className={cn('gap-1 text-xs', styles.badge)}>
                      {session.confidence}
                    </Badge>
                  </div>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(session.experimentId);
                  }}
                  disabled={isDeleting}
                  title={isConfirming ? 'Click again to confirm delete' : 'Delete this experiment'}
                  className={cn(
                    'absolute right-2 top-3 rounded-md p-1.5 transition-colors',
                    isConfirming
                      ? 'bg-red-50 text-red-600'
                      : 'text-neutral-300 hover:bg-red-50 hover:text-red-600 group-hover:text-neutral-400'
                  )}
                >
                  {isDeleting ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Trash2 className="h-3.5 w-3.5" />
                  )}
                </button>
                {isConfirming && !isDeleting && (
                  <p className="mt-1 text-xs font-medium text-red-600">
                    Click the trash icon again to permanently delete
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Report detail */}
      <div className="min-w-0 flex-1 overflow-y-auto pr-1">
        {detailError && !error && (
          <Card className="mb-3 border-red-200 bg-red-50 shadow-none">
            <CardContent className="py-3 text-[13px] text-red-700">{detailError}</CardContent>
          </Card>
        )}
        {detailLoading && (
          <div className="flex items-center gap-2 py-6 text-xs text-neutral-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading report...
          </div>
        )}
        {!detailLoading && detail ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold tracking-tight text-black">
                {detail.datasetName}
              </h2>
              <span className="text-sm text-neutral-400">
                · {new Date(detail.createdAt).toLocaleString()}
              </span>
            </div>
            <ReportCard
              report={detail.report}
              datasetName={detail.datasetName}
              experimentId={detail.experimentId}
              prompt={detail.userPrompt}
            />
            <RelatedExperiments
              items={detail.relatedExperiments.filter((r) => r.experimentId !== detail.experimentId)}
            />
            <FollowUpChat messages={messages} onSend={handleFollowUp} />
          </div>
        ) : (
          !detailLoading &&
          !loading &&
          !error && (
            <Card className="border-black/10 shadow-none">
              <CardContent className="flex items-center justify-center py-20 text-sm text-neutral-400">
                {sessions.length > 0
                  ? 'Select a session to view its report'
                  : 'No sessions yet — run an evaluation in New Experiment to see it here'}
              </CardContent>
            </Card>
          )
        )}
      </div>
    </div>
  );
}
