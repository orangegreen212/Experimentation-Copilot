"""
Follow-up chat schemas.

Mirrors lib/types.ts:

    export interface ChatMessage {
      id: string;
      role: 'user' | 'assistant';
      content: string;
    }

Follow-up chat is explicitly GROUNDED: the assistant answers using the
already-computed ExperimentReport + underlying stats/quality objects
for this session, never by recomputing anything. See
`workspace-view.tsx`'s `generateReply()` mock for the kinds of
questions this must answer (CUPED effect, SRM explanation, ship
recommendation, sample/power questions) — Stage 7/8 will replace that
mock with an LLM call templated over the same grounding data.
"""

from enum import Enum

from app.schemas.base import CamelModel


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(CamelModel):
    id: str
    role: ChatRole
    content: str


class FollowUpChatRequest(CamelModel):
    experiment_id: str
    message: str
    # Optional per-request LLM override — see AnalysisSettings.model
    # (schemas/settings.py) and app.llm.client.resolve_model() for how
    # this is validated against the curated free-model allowlist.
    # None means "use the server-configured default model".
    model: str | None = None


class FollowUpChatResponse(CamelModel):
    message: ChatMessage
