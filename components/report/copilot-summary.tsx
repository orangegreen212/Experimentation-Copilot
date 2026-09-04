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
    <div className="rounded-2xl border border-indigo-100 bg-indigo-50/40 p-5">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-indigo-600" />
          <p className="text-[13px] font-semibold tracking-tight text-black">Copilot Summary</p>
        </div>
        <span className="rounded border border-indigo-200 bg-white px-1.5 py-0.5 text-[10px] font-medium text-indigo-600">
          AI
        </span>
      </div>
      <p className="text-[13px] leading-relaxed text-neutral-700">{summary}</p>
      <Button
        size="sm"
        onClick={onAskCopilot}
        className="mt-4 w-full justify-between bg-indigo-600 text-white hover:bg-indigo-700"
      >
        Ask Copilot Anything
        <ArrowRight className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
