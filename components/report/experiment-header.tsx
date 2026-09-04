import { Fingerprint, CalendarDays, Users, Layers3, Download } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import type { ExperimentReport } from '@/lib/types';

const STATUS_STYLE: Record<string, string> = {
  SUCCESS: 'border-green-200 bg-green-50 text-green-700',
  WARNING: 'border-amber-200 bg-amber-50 text-amber-700',
  FAILED: 'border-red-200 bg-red-50 text-red-700',
  SKIPPED: 'border-black/10 bg-neutral-50 text-neutral-500',
};

const STATUS_LABEL: Record<string, string> = {
  SUCCESS: 'Completed',
  WARNING: 'Completed with warnings',
  FAILED: 'Failed',
  SKIPPED: 'Skipped',
};

function formatTimestamp(iso: string | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

export function ExperimentHeader({
  report,
  datasetName,
  experimentId,
  prompt,
  onDownload,
}: {
  report: ExperimentReport;
  datasetName?: string;
  experimentId?: string;
  prompt?: string;
  onDownload?: () => void;
}) {
  const rm = report.runMetadata;
  const status = rm?.executionStatus ?? 'SUCCESS';
  const date = formatTimestamp(rm?.timestamp);

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-black/10 bg-white px-5 py-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0 space-y-2">
        <div className="flex flex-wrap items-center gap-2.5">
          <h1 className="truncate text-lg font-semibold tracking-tight text-black">
            {datasetName || prompt || 'Experiment Review'}
          </h1>
          <span
            className={cn(
              'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold',
              STATUS_STYLE[status] ?? STATUS_STYLE.SKIPPED
            )}
          >
            {STATUS_LABEL[status] ?? status}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-neutral-500">
          {experimentId && (
            <span className="inline-flex items-center gap-1.5">
              <Fingerprint className="h-3.5 w-3.5 text-neutral-400" />
              {experimentId}
            </span>
          )}
          {date && (
            <span className="inline-flex items-center gap-1.5">
              <CalendarDays className="h-3.5 w-3.5 text-neutral-400" />
              {date}
            </span>
          )}
          {rm && (
            <span className="inline-flex items-center gap-1.5">
              <Users className="h-3.5 w-3.5 text-neutral-400" />
              {rm.userCount.toLocaleString()} users
            </span>
          )}
          {rm && (
            <span className="inline-flex items-center gap-1.5">
              <Layers3 className="h-3.5 w-3.5 text-neutral-400" />
              {rm.variantCount} variants
            </span>
          )}
        </div>
      </div>

      {onDownload && (
        <Button
          variant="outline"
          size="sm"
          className="shrink-0 gap-1.5 border-black/15"
          onClick={onDownload}
        >
          <Download className="h-3.5 w-3.5" />
          Download Report
        </Button>
      )}
    </div>
  );
}
