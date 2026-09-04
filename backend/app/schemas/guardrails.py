"""
Guardrail REQUEST state — Phase (guardrail root-cause fix).

Root cause of the original bug (see design audit): the system had
exactly one guardrail signal, `GuardrailStatus` (PASS / WARNING / FAIL
/ NOT_AVAILABLE), and NOT_AVAILABLE was overloaded to mean three
different things at once:

  1. the user never asked for a guardrail at all
  2. the user asked for a guardrail that doesn't exist in this dataset
  3. the user asked for a guardrail that exists, but it hasn't been
     evaluated yet for some other reason (e.g. the experiment itself
     is invalid)

This module adds the missing REQUEST/AVAILABILITY dimension so those
three cases are never collapsed into one value again.
`GuardrailStatus` (app/schemas/report.py) is left exactly as it was —
it now means ONLY "the evaluation outcome of whichever guardrails
actually got resolved and tested," never "were guardrails requested."

    request state (this file)      answers "could we find/run it?"
    GuardrailStatus (report.py)    answers "did it pass?"

Both are independent facts a report can (and should) show together —
e.g. request state = PARTIALLY_AVAILABLE *and* GuardrailStatus = FAIL
is a perfectly coherent, and common, combination.
"""

from enum import Enum

from app.schemas.base import CamelModel


class GuardrailRequestState(str, Enum):
    """
    Whether the user's explicitly requested guardrails (see
    `AnalysisSettings.guardrail_metrics`) could be resolved against
    this dataset's actual columns — deliberately independent of
    whether the resolved guardrail(s) then PASSED or FAILED
    evaluation (that's `GuardrailStatus`).
    """

    # No guardrails were requested at all. Never conflated with
    # REQUESTED_NOT_FOUND — this is the ONLY state a request with an
    # empty/omitted `guardrailMetrics` can ever produce.
    NOT_SPECIFIED = "NOT_SPECIFIED"
    # One or more guardrails were requested, but NONE of them matched
    # an actual column in this dataset (deterministic exact match —
    # never fuzzy, never semantic; see
    # app.stats.dataset_classifier.resolve_guardrail_metrics).
    REQUESTED_NOT_FOUND = "REQUESTED_NOT_FOUND"
    # Some requested guardrails matched a real column, others didn't.
    PARTIALLY_AVAILABLE = "PARTIALLY_AVAILABLE"
    # Every requested guardrail matched a real column. Does NOT by
    # itself imply the statistical evaluation ran (e.g. a multi-arm
    # experiment resolves guardrail columns but does not yet evaluate
    # them — see guardrail_node.py) — check `GuardrailStatus` for the
    # evaluation outcome.
    AVAILABLE = "AVAILABLE"


class GuardrailResolution(CamelModel):
    """
    One requested guardrail's resolution outcome, preserved verbatim
    so the report can list e.g. "Revenue — not found" /
    "Bounce Rate — not found" instead of silently dropping the
    request. `requested_name` is exactly what the user typed/selected
    — never rewritten, never silently substituted for a
    semantically-similar column.
    """

    requested_name: str
    resolved: bool
    # The dataset's own humanized metric label (matches
    # DatasetInfo.available_metrics / StatResult.metric) that this
    # request resolved to. None when `resolved` is False.
    resolved_metric_label: str | None = None
