"""
Shared Pydantic base configuration.

Every schema in this package inherits from `CamelModel` so that:
  - Python code stays idiomatic snake_case internally
  - JSON on the wire is camelCase, matching `lib/types.ts` exactly
  - The Next.js frontend requires ZERO changes to consume these responses

This is a deliberate architectural decision: the frontend contract
(types.ts) is treated as the source of truth for field naming.
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Base model: snake_case in Python, camelCase in JSON."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )
