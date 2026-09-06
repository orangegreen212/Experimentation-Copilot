import { BarChart3 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import type { StatResult } from '@/lib/types';

/**
 * Pulls the first numeric value out of a formatted stat string, e.g.
 * "11.87%" -> 11.87, "+6.08pp" -> 6.08, "-1.24%" -> -1.24. Returns null
 * when nothing numeric is present rather than guessing — callers must
 * handle that (never render a fabricated 0).
 */
function parseNumeric(value: string | null | undefined): number | null {
  if (!value) return null;
  const match = value.match(/-?\d+(\.\d+)?/);
  if (!match) return null;
  const n = Number(match[0]);
  return Number.isFinite(n) ? n : null;
}

/**
 * Primary Metric visualization: a Control-vs-Variant bar chart plus a
 * horizontal confidence-interval plot, both driven only by fields
 * already present on StatResult (control, variant, ciLower, ciUpper,
 * pValue) — nothing here is computed beyond parsing the backend's own
 * formatted strings back into numbers for layout.
 */
export function PrimaryMetricChart({ stat }: { stat: StatResult }) {
  const control = parseNumeric(stat.control);
  const variant = parseNumeric(stat.variant);
  const ciLower = parseNumeric(stat.ciLower);
  const ciUpper = parseNumeric(stat.ciUpper);

  if (control == null || variant == null) return null;

  const maxVal = Math.max(control, variant, 0.0001);
  const controlPct = (control / maxVal) * 100;
  const variantPct = (variant / maxVal) * 100;

  return (
    <Card className="border-black/10 shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-black" />
          <CardTitle className="text-[15px] tracking-tight">{stat.metric} by Variant</CardTitle>
        </div>
        <CardDescription>Control vs. Treatment, with the effect's confidence interval below</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Bar chart */}
        <div className="flex items-end gap-8 px-2">
          <div className="flex flex-1 flex-col items-center gap-2">
            <span className="text-[15px] font-semibold text-black">{stat.control}</span>
            <div className="flex h-32 w-full items-end justify-center">
              <div
                className="w-16 rounded-t-md bg-neutral-300"
                style={{ height: `${Math.max(controlPct, 3)}%` }}
              />
            </div>
            <span className="text-[11px] font-medium uppercase tracking-wide text-neutral-500">Control</span>
          </div>
          <div className="flex flex-1 flex-col items-center gap-2">
            <span className="text-[15px] font-semibold text-black">{stat.variant}</span>
            <div className="flex h-32 w-full items-end justify-center">
              <div
                className={
                  'w-16 rounded-t-md ' + (stat.significant ? 'bg-indigo-600' : 'bg-neutral-300')
                }
                style={{ height: `${Math.max(variantPct, 3)}%` }}
              />
            </div>
            <span className="text-[11px] font-medium uppercase tracking-wide text-neutral-500">Treatment</span>
          </div>
        </div>

        {/* Confidence interval plot */}
        {ciLower != null && ciUpper != null && (
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-neutral-400">
              Treatment effect — {stat.delta} ({stat.significant ? 'statistically significant' : 'not significant'})
            </p>
            <CiPlot lower={ciLower} upper={ciUpper} pValue={stat.pValue} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/** Horizontal 95% CI plot: a track spanning [lower, upper] padded around
 *  zero, a dot at the midpoint, and a zero reference line. Purely a
 *  layout of the two numbers already on the stat — no new computation. */
function CiPlot({ lower, upper, pValue }: { lower: number; upper: number; pValue: number }) {
  const mid = (lower + upper) / 2;
  const span = Math.max(Math.abs(lower), Math.abs(upper), Math.abs(mid), 0.0001) * 1.4;
  const toPct = (v: number) => ((v + span) / (2 * span)) * 100;

  const lowerPct = toPct(lower);
  const upperPct = toPct(upper);
  const midPct = toPct(mid);
  const zeroPct = toPct(0);

  return (
    <div className="px-2">
      <div className="relative h-8">
        {/* zero reference line */}
        <div
          className="absolute top-0 h-full w-px bg-neutral-300"
          style={{ left: `${zeroPct}%` }}
        />
        {/* CI track */}
        <div
          className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-indigo-200"
          style={{ left: `${lowerPct}%`, width: `${Math.max(upperPct - lowerPct, 0.5)}%` }}
        />
        {/* midpoint dot */}
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-indigo-600 shadow"
          style={{ left: `${midPct}%` }}
        />
      </div>
      <div className="mt-1 flex items-center justify-between text-[10px] text-neutral-400">
        <span>{lower >= 0 ? '+' : ''}{lower.toFixed(2)}</span>
        <span>0</span>
        <span>{upper >= 0 ? '+' : ''}{upper.toFixed(2)}</span>
      </div>
      <p className="mt-1 text-[11px] text-neutral-500">
        95% CI [{lower >= 0 ? '+' : ''}{lower.toFixed(2)}, {upper >= 0 ? '+' : ''}{upper.toFixed(2)}] · p-value {pValue < 0.001 ? '<0.001' : pValue.toFixed(3)}
      </p>
    </div>
  );
}
