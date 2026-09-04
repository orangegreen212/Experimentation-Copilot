"""
Decision Audit Trail schemas — Phase 7.

Answers "why did the system reach this exact decision?" for a
Product Manager, analyst, or reviewer, WITHOUT requiring them to
reverse-engineer the rest of the report. Every field here is either
copied verbatim from an already-computed structured fact
(`ExperimentReport`, `ReportFacts`, `StatResult`, `PowerAnalysisResult`,
`SegmentationResult`, `QualityCheck`, ...) or a short, deterministic,
plain-Python sentence built FROM those facts in
`app/stats/decision_audit.py`.

THIS IS NOT A SECOND DECISION ENGINE (Phase 7 spec §12):

  - `decision` here is always copied from the single canonical
    `Decision` already computed by `determine_decision()`
    (app/graph/report_generator.py) — never recomputed, never
    overridden.
  - No p-values, effect sizes, power, or significance are
    (re)calculated in this module — every number is read off an
    existing `StatResult` / `PowerAnalysisResult` / `SegmentationResult`
    row.
  - No LLM is involved anywhere in building this object.
  - Segmentation findings are supporting evidence only and never
    change `decision` — same rule `SegmentationResult` already
    follows (see app/schemas/segmentation.py).

`AuditFact` follows the project's existing "one structured row with a
human string" convention (compare `QualityCheck`, `GuardrailFinding`):
a small, frontend-renderable unit with a status, a category, a label,
a display value, and how it affects the decision — never raw numbers
the frontend would have to reinterpret.
"""

from __future__ import annotations

from enum import Enum

from app.schemas.base import CamelModel
from app.schemas.report import Decision


class AuditStatus(str, Enum):
    """Visual/semantic status of one audit fact — drives the ✓ / ⚠ / ✗ rendering."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    INFO = "info"
    NOT_AVAILABLE = "not_available"


class AuditImpact(str, Enum):
    """How this fact relates to the final `decision` — never implies a second decision."""

    SUPPORTS_DECISION = "supports_decision"
    BLOCKS_DECISION = "blocks_decision"
    LIMITS_CONFIDENCE = "limits_confidence"
    NEUTRAL = "neutral"


class AuditCategory(str, Enum):
    """Which part of the analysis this fact came from."""

    VALIDITY = "validity"
    DATA_QUALITY = "data_quality"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    PRACTICAL_SIGNIFICANCE = "practical_significance"
    POWER = "power"
    GUARDRAILS = "guardrails"
    SEGMENTATION = "segmentation"


class AuditFact(CamelModel):
    """
    One evidence row — e.g. {"status": "pass", "category": "validity",
    "label": "Experiment validity", "value": "VALID",
    "impact": "supports_decision"}. `detail` is optional supplementary
    text (mirrors `QualityCheck.detail`); `value` is always a short,
    already-formatted display string, never a raw float the frontend
    would have to format itself.
    """

    status: AuditStatus
    category: AuditCategory
    label: str
    value: str
    impact: AuditImpact
    detail: str | None = None


class DecisionAuditTrail(CamelModel):
    """
    Top-level Phase 7 fact, attached to the report alongside (never in
    place of) the existing `decision` / `decisionReason` fields. See
    module docstring for the "not a second decision engine" boundary.
    """

    # Copied verbatim from the report's own canonical `decision` — see
    # module docstring. Never recomputed here.
    decision: Decision
    headline: str  # e.g. "GO WITH CAUTION"

    # Deterministic bullet-point explanation ("Why this decision?"),
    # built from structured facts — never free-form LLM text.
    rationale: list[str]

    # Evidence explicitly supporting the decision (the "✓" list).
    supporting_facts: list[AuditFact]
    # Limitations / caveats that should prevent overconfidence (the "⚠" list).
    warnings: list[AuditFact]
    # Validity-specific checks (SRM, conflicting assignment, critical
    # data-quality issues) — always populated, even when all pass.
    critical_checks: list[AuditFact]

    # Category-specific evidence, broken out so the frontend can render
    # each section without re-filtering `supporting_facts`/`warnings`.
    statistical_evidence: list[AuditFact]
    practical_significance_evidence: AuditFact | None = None
    power_evidence: AuditFact | None = None
    guardrail_evidence: AuditFact
    segmentation_evidence: AuditFact

    # Short closing statement — "Decision impact" in the Phase 7 spec's
    # suggested UI structure.
    decision_impact: str
