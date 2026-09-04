import { Layers } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { SegmentationResult, SegmentDimensionResult } from '@/lib/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

function segmentBadgeClass(significant: boolean, isReliable: boolean) {
  if (isReliable) return 'border-green-200 bg-green-50 text-green-700';
  if (significant) return 'border-amber-200 bg-amber-50 text-amber-700';
  return 'border-black/10 bg-neutral-50 text-neutral-500';
}

function SegmentDimensionCard({ dim }: { dim: SegmentDimensionResult }) {
  return (
    <div className="rounded-md border border-black/10 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <p className="text-[13px] font-semibold text-black">{dim.dimension}</p>
        {dim.hasHeterogeneousEffect && (
          <Badge variant="outline" className="border-amber-200 bg-amber-50 text-[10px] text-amber-700">
            Heterogeneous effect detected
          </Badge>
        )}
        <span className="ml-auto text-[10px] text-neutral-400">{dim.multipleTestingMethod}</span>
      </div>
      <div className="space-y-2">
        {dim.segmentEffects.map((seg) => {
          const isReliable = dim.reliableSegmentValues.includes(seg.segmentValue);
          if (seg.sampleSizeStatus === 'insufficient') {
            return (
              <div
                key={seg.segmentValue}
                className="flex items-center justify-between gap-3 rounded-md border border-black/10 bg-neutral-50 px-3 py-2 text-[12px]"
              >
                <span className="font-medium text-black">{seg.segmentValue}</span>
                <span className="text-neutral-400">
                  Insufficient sample (n={seg.controlN}/{seg.variantN})
                  {seg.skipDetail ? ` — ${seg.skipDetail}` : ''}
                </span>
              </div>
            );
          }
          const s = seg.statResult;
          return (
            <div
              key={seg.segmentValue}
              className="grid grid-cols-12 items-center gap-2 rounded-md border border-black/10 px-3 py-2 text-[12px]"
            >
              <div className="col-span-12 sm:col-span-3">
                <span className="font-medium text-black">{seg.segmentValue}</span>
                <p className="text-[10px] text-neutral-400">
                  n={seg.controlN}/{seg.variantN}
                </p>
              </div>
              {s ? (
                <>
                  <div className="col-span-6 sm:col-span-2">
                    <span className="text-[10px] uppercase text-neutral-400">Control</span>
                    <p className="text-black">{s.control}</p>
                  </div>
                  <div className="col-span-6 sm:col-span-2">
                    <span className="text-[10px] uppercase text-neutral-400">Variant</span>
                    <p className="text-black">{s.variant}</p>
                  </div>
                  <div className="col-span-6 sm:col-span-2">
                    <span className="text-[10px] uppercase text-neutral-400">Delta</span>
                    <p className="font-semibold text-black">{s.delta}</p>
                  </div>
                  <div className="col-span-6 sm:col-span-1">
                    <span className="text-[10px] uppercase text-neutral-400">p-value</span>
                    <p className="font-mono text-[11px]">
                      {s.pValue < 0.001 ? '<0.001' : s.pValue.toFixed(3)}
                      {s.adjustedPValue != null && (
                        <span className="ml-1 text-neutral-400">
                          (adj. {s.adjustedPValue < 0.001 ? '<0.001' : s.adjustedPValue.toFixed(3)})
                        </span>
                      )}
                    </p>
                  </div>
                  <div className="col-span-12 sm:col-span-2 sm:text-right">
                    <Badge
                      variant="outline"
                      className={cn('text-[10px]', segmentBadgeClass(s.significant, isReliable))}
                    >
                      {isReliable ? 'Reliable effect' : s.significant ? 'Significant (unadjusted)' : 'Not significant'}
                    </Badge>
                  </div>
                </>
              ) : (
                <div className="col-span-9 text-neutral-400">No test result available</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Phase 5 — Segment Analysis. Renders only backend-computed
 * `SegmentationResult` facts; no statistics are calculated in
 * TypeScript and the LLM is never asked to compute anything here.
 */
export function SegmentAnalysisSection({
  segmentation,
}: {
  segmentation: SegmentationResult | null | undefined;
}) {
  return (
    <Card className="border-black/10 shadow-none">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-black" />
          <CardTitle className="text-[15px] tracking-tight">Segment Analysis</CardTitle>
        </div>
        <CardDescription>
          Segment analysis is exploratory and does not override the primary experiment decision.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {(!segmentation || !segmentation.ran || segmentation.dimensionResults.length === 0) && (
          <div className="rounded-md border border-black/10 bg-neutral-50 px-3 py-4 text-center">
            <p className="text-[13px] text-neutral-500">
              {segmentation?.reason ?? 'No segment analysis is available for this experiment.'}
            </p>
          </div>
        )}

        {segmentation && segmentation.ran && segmentation.dimensionResults.length > 0 && (
          <>
            <p className="text-[13px] text-neutral-600">{segmentation.reason}</p>
            {segmentation.dimensionResults.map((dim) => (
              <SegmentDimensionCard key={dim.dimension} dim={dim} />
            ))}
          </>
        )}

        {segmentation && segmentation.skippedDimensions.length > 0 && (
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-neutral-400">
              Skipped Dimensions
            </p>
            <div className="space-y-1.5">
              {segmentation.skippedDimensions.map((d) => (
                <div key={d.column} className="text-[12px] text-neutral-500">
                  <span className="font-medium text-neutral-700">{d.column}</span> — {d.detail}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
