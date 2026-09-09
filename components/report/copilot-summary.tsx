import { Sparkles } from 'lucide-react';
import { FollowUpChat } from '@/components/follow-up-chat';
import type { ChatMessage } from '@/lib/types';

/**
 * Unified Copilot panel: interpretation + follow-up Q&A as ONE
 * conversational surface, per the redesign brief §8. Previously this
 * was two disconnected components (a static summary card, and a
 * `FollowUpChat` rendered separately — sometimes with a large gap,
 * sometimes on a totally different screen scroll position — by each
 * of the three screens that render a report). Now the report screen
 * always passes `chat` through and this renders the thread + input
 * directly beneath the interpretation, no border between them.
 *
 * `caveats` is real data (`decisionNarrative.whatPreventsFullGo`) —
 * not invented structure. It's the one part of the "What
 * happened/Why/Recommendation/Caveats" shape from the brief that the
 * backend gives us as a distinct list; the rest of that narrative is
 * already woven into `summary` as prose by `buildStateAwareSummary`,
 * so it isn't re-split into fake headers here.
 */
export function CopilotSummary({
  summary,
  caveats,
  chat,
}: {
  summary: string;
  caveats?: string[];
  chat?: {
    messages: ChatMessage[];
    onSend: (content: string) => void;
    isLoading?: boolean;
  };
}) {
  return (
    <div className="rounded-xl border border-copilot/20 bg-copilot-bg">
      <div className="p-5 pb-4">
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

        {caveats && caveats.length > 0 && (
          <div className="mt-3 space-y-1 border-t border-copilot/15 pt-3">
            <p className="text-[11px] font-medium text-muted-foreground">Caveats</p>
            <ul className="space-y-1">
              {caveats.map((c, i) => (
                <li key={i} className="flex items-start gap-1.5 text-[12.5px] leading-relaxed text-muted-foreground">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground/50" />
                  {c}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {chat && (
        <FollowUpChat
          variant="embedded"
          messages={chat.messages}
          onSend={chat.onSend}
          isLoading={chat.isLoading}
        />
      )}
    </div>
  );
}
