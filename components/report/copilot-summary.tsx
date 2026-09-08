import { Sparkles, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function CopilotSummary({
  summary,
  onAskCopilot,
}: {
  summary: string;
  /** Scrolls to / focuses the existing Follow-up Q&A chat further down the page. */
  onAskCopilot?: () => void;
}) {
  return (
    <div className="rounded-xl border border-copilot/20 bg-copilot-bg p-5">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-copilot" />
          <p className="text-[13px] font-semibold tracking-tight text-foreground">Copilot interpretation</p>
        </div>
        <span className="rounded border border-copilot/25 bg-surface px-1.5 py-0.5 text-[11px] font-medium text-copilot">
          AI
        </span>
      </div>
      <p className="text-[13px] leading-relaxed text-foreground">{summary}</p>
      <Button
        size="sm"
        variant="outline"
        onClick={onAskCopilot}
        className="mt-4 w-full justify-between border-copilot/25 bg-surface text-copilot hover:bg-copilot-bg hover:text-copilot"
      >
        Ask Copilot anything
        <ArrowRight className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
