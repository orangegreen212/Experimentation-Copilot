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

from pydantic import Field

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
    # RELIABILITY FIX (same audit finding/fix as
    # AnalyzeExperimentRequest.prompt in routes_experiments.py): this
    # is embedded verbatim into the LLM conversation on every chat
    # call (`conversation.append({"role": "user", "content":
    # message})` in graph/chat_generator.py's `_build_chat_conversation`)
    # and previously had no length limit. `history` is NOT a
    # client-supplied risk here — it's read server-side from
    # ExperimentStore and already trimmed to
    # DEFAULT_MAX_HISTORY_MESSAGES (routes_experiments.py) — so
    # `message` was the one unbounded input on this path.
    message: str = Field(max_length=4000)
    # Optional per-request LLM override — see AnalysisSettings.model
    # (schemas/settings.py) and app.llm.client.resolve_model() for how
    # this is validated against the curated free-model allowlist.
    # None means "use the server-configured default model".
    model: str | None = None


class FollowUpChatResponse(CamelModel):
    message: ChatMessage
