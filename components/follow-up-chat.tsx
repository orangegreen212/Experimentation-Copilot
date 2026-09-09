'use client';

import { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';
import type { ChatMessage } from '@/lib/types';

interface FollowUpChatProps {
  messages: ChatMessage[];
  onSend: (content: string) => void;
  /** True while a follow-up request is in flight — shows a "typing…"
   *  indicator and disables input so the UI doesn't look stuck while
   *  waiting on the LLM response. */
  isLoading?: boolean;
  /**
   * 'embedded' (default) renders just the thread + input, meant to sit
   * directly beneath CopilotSummary's interpretation as one continuous
   * panel — no border/header of its own. 'standalone' keeps the old
   * self-contained card, kept for any future spot this needs to live
   * on its own outside a report.
   */
  variant?: 'embedded' | 'standalone';
  suggestions?: string[];
}

const DEFAULT_SUGGESTIONS = [
  'Why did the variant win?',
  'Which segment contributed most?',
  'Are there any risks?',
  'What should we test next?',
];

export function FollowUpChat({
  messages,
  onSend,
  isLoading = false,
  variant = 'embedded',
  suggestions = DEFAULT_SUGGESTIONS,
}: FollowUpChatProps) {
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = (text?: string) => {
    if (isLoading) return;
    const trimmed = (text ?? input).trim();
    if (!trimmed) return;
    onSend(trimmed);
    setInput('');
  };

  const thread = (messages.length > 0 || isLoading) && (
    <div ref={scrollRef} className="max-h-64 space-y-3 overflow-y-auto px-4 py-3">
      {messages.map((m) => (
        <div key={m.id} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
          <div
            className={
              m.role === 'user'
                ? 'max-w-[85%] rounded-lg rounded-br-sm bg-primary px-3 py-2 text-[13px] text-primary-foreground'
                : 'max-w-[85%] rounded-lg rounded-bl-sm bg-secondary px-3 py-2 text-[13px] text-foreground'
            }
          >
            {m.content}
          </div>
        </div>
      ))}
      {isLoading && (
        <div className="flex justify-start">
          <div className="flex items-center gap-1 rounded-lg rounded-bl-sm bg-secondary px-3 py-2">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground/50" />
          </div>
        </div>
      )}
    </div>
  );

  const inputRow = (
    <div className="flex flex-col gap-2 px-4 py-3">
      {messages.length === 0 && !isLoading && (
        <div className="flex flex-wrap gap-1.5">
          {suggestions.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => handleSubmit(s)}
              disabled={isLoading}
              className="rounded-full border border-border bg-surface px-2.5 py-1 text-[11.5px] text-muted-foreground transition-colors hover:border-primary/30 hover:bg-accent hover:text-accent-foreground"
            >
              {s}
            </button>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
          placeholder="Ask a question about this report…"
          className="min-h-[40px] resize-none border-border bg-surface placeholder:text-muted-foreground"
          rows={1}
          disabled={isLoading}
        />
        <Button size="icon" onClick={() => handleSubmit()} disabled={!input.trim() || isLoading} className="shrink-0">
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );

  if (variant === 'standalone') {
    return (
      <div id="follow-up-chat" className="rounded-lg border border-border bg-surface">
        <div className="flex items-center gap-2 border-b border-border px-4 py-3">
          <h3 className="text-[13px] font-semibold text-foreground">Ask a question about this report</h3>
        </div>
        {thread}
        {inputRow}
      </div>
    );
  }

  return (
    <div id="follow-up-chat" className={cn(messages.length > 0 || isLoading ? 'border-t border-copilot/15' : '')}>
      {thread}
      {inputRow}
    </div>
  );
}
