import { Scale, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { QualityCheck } from '@/lib/types';

/**
 * The SRM detail string is built deterministically on the backend by
 * srm_result_to_quality_check() (app/stats/srm.py) in one fixed shape:
 *   "Observed split 49.9% / 50.1% vs expected 50/50 (p = 0.25) [...]"
 * This just parses that same string back into numbers for layout — it
 * never invents a split that isn't already in the check's own detail
 * text, and falls back to the plain text if the shape doesn't match.
 */
function parseSrmDetail(detail: string) {
  const match = detail.match(
    /Observed split ([\d.]+)% \/ ([\d.]+)% vs expected (\d+)\/(\d+)/
  );
  if (!match) return null;
  return {
    observedControl: Number(match[1]),
    observedVariant: Number(match[2]),
    expectedControl: Number(match[3]),
    expectedVariant: Number(match[4]),
  };
}

export function SrmPanel({ qualityChecks }: { qualityChecks: QualityCheck[] }) {
  const srmCheck = qualityChecks.find((c) => c.label === 'Sample Ratio Mismatch (SRM)');
  if (!srmCheck) return null;

  const parsed = parseSrmDetail(srmCheck.detail);

  return (
    <Card className="border-border shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Scale className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">Assignment Quality</CardTitle>
          <Badge
            variant="outline"
            className={cn(
              'ml-auto text-[10px] font-semibold',
              srmCheck.passed
                ? 'border-success/25 bg-success/[0.08] text-success'
                : 'border-destructive/25 bg-destructive/[0.08] text-destructive'
            )}
          >
            {srmCheck.passed ? (
              <span className="inline-flex items-center gap-1"><CheckCircle2 className="h-3 w-3" /> SRM PASS</span>
            ) : (
              <span className="inline-flex items-center gap-1"><AlertTriangle className="h-3 w-3" /> SRM DETECTED</span>
            )}
          </Badge>
        </div>
        <CardDescription>
          Whether users landed in Control/Treatment at the expected ratio
        </CardDescription>
      </CardHeader>
      <CardContent>
        {parsed ? (
          <div className="space-y-2">
            <SplitRow label="Control" expected={parsed.expectedControl} observed={parsed.observedControl} ok={srmCheck.passed} />
            <SplitRow label="Treatment" expected={parsed.expectedVariant} observed={parsed.observedVariant} ok={srmCheck.passed} />
            {!srmCheck.passed && (
              <p className="mt-2 text-[12px] text-destructive">
                The experiment may be affected by assignment imbalance.
              </p>
            )}
          </div>
        ) : (
          <p className="text-[12px] text-muted-foreground">{srmCheck.detail}</p>
        )}
      </CardContent>
    </Card>
  );
}

function SplitRow({
  label,
  expected,
  observed,
  ok,
}: {
  label: string;
  expected: number;
  observed: number;
  ok: boolean;
}) {
  return (
    <div className="rounded-md border border-border bg-secondary px-3 py-2">
      <div className="flex items-center justify-between text-[12px]">
        <span className="font-medium text-foreground">{label}</span>
        <span className="text-muted-foreground">
          expected <span className="font-data">{expected}%</span> · observed{' '}
          <span className={cn('font-data font-semibold', ok ? 'text-foreground' : 'text-destructive')}>{observed}%</span>
        </span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-border-strong">
        <div
          className={cn('h-full rounded-full', ok ? 'bg-primary' : 'bg-destructive')}
          style={{ width: `${Math.min(observed, 100)}%` }}
        />
      </div>
    </div>
  );
}
