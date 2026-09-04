'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, MessageSquare } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import type { ChatMessage } from '@/lib/types';

interface FollowUpChatProps {
  messages: ChatMessage[];
  onSend: (content: string) => void;
  /** True while a follow-up request is in flight — shows a "typing…"
   *  indicator and disables input so the UI doesn't look stuck while
   *  waiting on the LLM response. */
  isLoading?: boolean;
}

export function FollowUpChat({ messages, onSend, isLoading = false }: FollowUpChatProps) {
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = () => {
    if (isLoading) return;
    const trimmed = input.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setInput('');
  };

  return (
    <div id="follow-up-chat" className="rounded-lg border border-black/10 bg-white">
      <div className="flex items-center gap-2 border-b border-black/10 px-4 py-3">
        <MessageSquare className="h-4 w-4 text-black" />
        <h3 className="text-[13px] font-semibold text-black">Follow-up Q&amp;A</h3>
        <span className="text-xs text-neutral-400">
          Ask a question about this report...
        </span>
      </div>

      {(messages.length > 0 || isLoading) && (
        <div ref={scrollRef} className="max-h-64 space-y-3 overflow-y-auto px-4 py-3">
          {messages.map((m) => (
            <div
              key={m.id}
              className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}
            >
              <div
                className={
                  m.role === 'user'
                    ? 'max-w-[85%] rounded-lg rounded-br-sm bg-indigo-600 px-3 py-2 text-[13px] text-white'
                    : 'max-w-[85%] rounded-lg rounded-bl-sm bg-neutral-100 px-3 py-2 text-[13px] text-neutral-700'
                }
              >
                {m.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="flex items-center gap-1 rounded-lg rounded-bl-sm bg-neutral-100 px-3 py-2">
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.3s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.15s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400" />
              </div>
            </div>
          )}
        </div>
      )}

      <div className="flex items-end gap-2 px-4 py-3">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
          placeholder="e.g. What would the result look like with CUPED applied?"
          className="min-h-[40px] resize-none border-black/10 placeholder:text-neutral-400"
          rows={1}
          disabled={isLoading}
        />
        <Button size="icon" onClick={handleSubmit} disabled={!input.trim() || isLoading} className="shrink-0">
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
