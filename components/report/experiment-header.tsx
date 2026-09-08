import { Fingerprint, CalendarDays, Users, Layers3, Download } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import type { ExperimentReport } from '@/lib/types';

const STATUS_STYLE: Record<string, string> = {
  SUCCESS: 'border-success/25 bg-success/[0.08] text-success',
  WARNING: 'border-warning/25 bg-warning/[0.08] text-warning',
  FAILED: 'border-destructive/25 bg-destructive/[0.08] text-destructive',
  SKIPPED: 'border-border bg-secondary text-muted-foreground',
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
    <div className="flex flex-col gap-3 border-b border-border pb-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0 space-y-1.5">
        <div className="flex flex-wrap items-center gap-2.5">
          <h1 className="truncate text-[20px] font-semibold tracking-tight text-foreground">
            {datasetName || prompt || 'Experiment review'}
          </h1>
          <span
            className={cn(
              'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium',
              STATUS_STYLE[status] ?? STATUS_STYLE.SKIPPED
            )}
          >
            {STATUS_LABEL[status] ?? status}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12.5px] text-muted-foreground">
          {experimentId && (
            <span className="inline-flex items-center gap-1.5">
              <Fingerprint className="h-3.5 w-3.5" />
              <span className="font-data">{experimentId}</span>
            </span>
          )}
          {date && (
            <span className="inline-flex items-center gap-1.5">
              <CalendarDays className="h-3.5 w-3.5" />
              {date}
            </span>
          )}
          {rm && (
            <span className="inline-flex items-center gap-1.5">
              <Users className="h-3.5 w-3.5" />
              <span className="font-data">{rm.userCount.toLocaleString()}</span> users
            </span>
          )}
          {rm && (
            <span className="inline-flex items-center gap-1.5">
              <Layers3 className="h-3.5 w-3.5" />
              {rm.variantCount} variants
            </span>
          )}
        </div>
      </div>

      {onDownload && (
        <Button
          variant="outline"
          size="sm"
          className="shrink-0 gap-1.5 border-border"
          onClick={onDownload}
        >
          <Download className="h-3.5 w-3.5" />
          Download report
        </Button>
      )}
    </div>
  );
}
