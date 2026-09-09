import { Layers } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { SegmentationResult, SegmentDimensionResult } from '@/lib/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

function segmentBadgeClass(significant: boolean, isReliable: boolean) {
  if (isReliable) return 'border-success/25 bg-success/[0.08] text-success';
  if (significant) return 'border-warning/25 bg-warning/[0.08] text-warning';
  return 'border-border bg-secondary text-muted-foreground';
}

function SegmentDimensionCard({ dim }: { dim: SegmentDimensionResult }) {
  return (
    <div className="rounded-md border border-border p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <p className="text-[13px] font-semibold text-foreground">{dim.dimension}</p>
        {dim.hasHeterogeneousEffect && (
          <Badge variant="outline" className="border-warning/25 bg-warning/[0.08] text-[10px] text-warning">
            Heterogeneous effect detected
          </Badge>
        )}
        <span className="ml-auto text-[10px] text-muted-foreground">{dim.multipleTestingMethod}</span>
      </div>
      <div className="space-y-2">
        {dim.segmentEffects.map((seg) => {
          const isReliable = dim.reliableSegmentValues.includes(seg.segmentValue);
          if (seg.sampleSizeStatus === 'insufficient') {
            return (
              <div
                key={seg.segmentValue}
                className="flex items-center justify-between gap-3 rounded-md border border-border bg-secondary px-3 py-2 text-[12px]"
              >
                <span className="font-medium text-foreground">{seg.segmentValue}</span>
                <span className="text-muted-foreground">
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
              className="grid grid-cols-12 items-center gap-2 rounded-md border border-border px-3 py-2 text-[12px]"
            >
              <div className="col-span-12 sm:col-span-3">
                <span className="font-medium text-foreground">{seg.segmentValue}</span>
                <p className="text-[10px] text-muted-foreground">
                  n={seg.controlN}/{seg.variantN}
                </p>
              </div>
              {s ? (
                <>
                  <div className="col-span-6 sm:col-span-2">
                    <span className="text-[10px] uppercase text-muted-foreground">Control</span>
                    <p className="text-foreground">{s.control}</p>
                  </div>
                  <div className="col-span-6 sm:col-span-2">
                    <span className="text-[10px] uppercase text-muted-foreground">Variant</span>
                    <p className="text-foreground">{s.variant}</p>
                  </div>
                  <div className="col-span-6 sm:col-span-2">
                    <span className="text-[10px] uppercase text-muted-foreground">Delta</span>
                    <p className="font-semibold text-foreground">{s.delta}</p>
                  </div>
                  <div className="col-span-6 sm:col-span-1">
                    <span className="text-[10px] uppercase text-muted-foreground">p-value</span>
                    <p className="font-mono text-[11px]">
                      {s.pValue < 0.001 ? '<0.001' : s.pValue.toFixed(3)}
                      {s.adjustedPValue != null && (
                        <span className="ml-1 text-muted-foreground">
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
                <div className="col-span-9 text-muted-foreground">No test result available</div>
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
    <Card className="border-none shadow-none bg-transparent px-0">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-foreground" />
          <CardTitle className="text-[15px] tracking-tight">Segment Analysis</CardTitle>
        </div>
        <CardDescription>
          Segment analysis is exploratory and does not override the primary experiment decision.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {(!segmentation || !segmentation.ran || segmentation.dimensionResults.length === 0) && (
          <div className="rounded-md border border-border bg-secondary px-3 py-4 text-center">
            <p className="text-[13px] text-muted-foreground">
              {segmentation?.reason ?? 'No segment analysis is available for this experiment.'}
            </p>
          </div>
        )}

        {segmentation && segmentation.ran && segmentation.dimensionResults.length > 0 && (
          <>
            <p className="text-[13px] text-muted-foreground">{segmentation.reason}</p>
            {segmentation.dimensionResults.map((dim) => (
              <SegmentDimensionCard key={dim.dimension} dim={dim} />
            ))}
          </>
        )}

        {segmentation && segmentation.skippedDimensions.length > 0 && (
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Skipped Dimensions
            </p>
            <div className="space-y-1.5">
              {segmentation.skippedDimensions.map((d) => (
                <div key={d.column} className="text-[12px] text-muted-foreground">
                  <span className="font-medium text-foreground">{d.column}</span> — {d.detail}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
