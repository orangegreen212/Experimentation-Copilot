"""
Schemas for GET /system/models.

Mirrors lib/types.ts:

    export interface AvailableModel {
      id: string;
      label: string;
    }

    export interface AvailableModelsResponse {
      models: AvailableModel[];
      defaultModel: string;
    }

`id` is the exact OpenRouter model string to send back as
`AnalysisSettings.model` / `FollowUpChatRequest.model` — never
constructed or guessed by the frontend, always one of these values
verbatim. See app.llm.client.resolve_model() for the server-side
re-validation that makes this allowlist authoritative even if a
client sent something else.
"""

from app.schemas.base import CamelModel


class AvailableModel(CamelModel):
    id: str
    label: str


class AvailableModelsResponse(CamelModel):
    models: list[AvailableModel]
    default_model: str
