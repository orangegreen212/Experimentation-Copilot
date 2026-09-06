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
    <Card className="border-black/10 shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Scale className="h-4 w-4 text-black" />
          <CardTitle className="text-[15px] tracking-tight">Assignment Quality</CardTitle>
          <Badge
            variant="outline"
            className={cn(
              'ml-auto text-[10px] font-semibold',
              srmCheck.passed
                ? 'border-green-200 bg-green-50 text-green-700'
                : 'border-red-200 bg-red-50 text-red-700'
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
              <p className="mt-2 text-[12px] text-red-600">
                The experiment may be affected by assignment imbalance.
              </p>
            )}
          </div>
        ) : (
          <p className="text-[12px] text-neutral-500">{srmCheck.detail}</p>
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
    <div className="rounded-md border border-black/10 bg-neutral-50 px-3 py-2">
      <div className="flex items-center justify-between text-[12px]">
        <span className="font-medium text-black">{label}</span>
        <span className="text-neutral-500">
          expected {expected}% · observed{' '}
          <span className={cn('font-semibold', ok ? 'text-neutral-700' : 'text-red-600')}>{observed}%</span>
        </span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-neutral-200">
        <div
          className={cn('h-full rounded-full', ok ? 'bg-indigo-500' : 'bg-red-500')}
          style={{ width: `${Math.min(observed, 100)}%` }}
        />
      </div>
    </div>
  );
}
