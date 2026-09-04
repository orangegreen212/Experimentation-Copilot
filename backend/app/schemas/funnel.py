"""
Funnel analysis schemas — Product Analytics capability #2 (alongside
Experiment Review). Mirrors the existing pattern: a public,
frontend-facing result shape (`FunnelAnalysisResult`) built entirely
from numbers computed in `app/stats/funnel.py` — the LLM (when this
gets wired into report generation) narrates these numbers, never
computes them, exactly like `StatResult`.
"""

from app.schemas.base import CamelModel


class FunnelStepResult(CamelModel):
    """One step in the funnel, in the order it was detected/specified."""

    name: str
    users: int
    conversion_from_start: float  # fraction of step-1 users who reached this step (1.0 for step 1)
    conversion_from_previous: float  # fraction of the PREVIOUS step's users who reached this step (1.0 for step 1)


class FunnelAnalysisResult(CamelModel):
    """
    Full funnel breakdown — steps plus the single largest drop-off,
    called out explicitly since "where are we losing the most users"
    is usually the actual question being asked.
    """

    steps: list[FunnelStepResult]
    overall_conversion: float  # last step's users / first step's users
    largest_dropoff_from: str
    largest_dropoff_to: str
    largest_dropoff_rate: float  # fraction of largest_dropoff_from's users who did NOT reach largest_dropoff_to
