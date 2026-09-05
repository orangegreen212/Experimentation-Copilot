"""
ExperimentDefinition schema — Phase 1 of the Experiment Platform layer.

This is the NEW pre-analysis planning entity described in the Stage 0
architecture doc. It is deliberately separate from `ExperimentReport`
/ `ExperimentRecord` (app/core/experiment_store.py): an
`ExperimentDefinition` can exist in DRAFT with no dataset attached at
all, long before any analysis runs. Once analysis does run, the
resulting persisted row in `experiments` (conceptually an
"AnalysisRun") links back here via `definition_id` — see
app/core/experiment_store.py's `ExperimentModel.definition_id`.

ARCHITECTURAL BOUNDARY (same rule as schemas/hypothesis.py): this file
is planning/configuration metadata only. Nothing here is read by the
statistical engine (`app/stats/*`) or the LangGraph pipeline
(`app/graph/*`) in this phase — those keep computing exactly what they
compute today, from the dataset alone. `ExperimentDefinition` fields
like `variants`, `targeting`, and `metrics` are descriptive/planning
context for the UI, not inputs the engine consumes. Wiring
`data_source_ref` + the primary hypothesis into the existing
`/experiments/analyze` request is a LATER phase (see Stage 0 doc,
Phase 6) — deliberately not implemented in this file.

No LLM is involved in constructing, validating, or normalizing any
model in this file — same rule as `Hypothesis` (see
schemas/hypothesis.py's docstring).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import Field, field_validator, model_validator

from app.schemas.base import CamelModel
from app.schemas.hypothesis import ExpectedDirection, Hypothesis
from app.schemas.statistics import MetricType

_MAX_NAME_LENGTH = 200
_MAX_TEXT_LENGTH = 2000
_ALLOCATION_TOLERANCE_PCT = 0.5  # percentage points; guards against float rounding, not real slack


class ExperimentStatus(str, Enum):
    """
    Lifecycle status of an `ExperimentDefinition`. Deliberately a closed
    set (mirrors `ExpectedDirection`/`MetricType`'s enum convention) —
    see Stage 0 architecture doc §"EXPERIMENT LIBRARY" for the intended
    meaning of each value. Nothing in this phase transitions status
    automatically; every transition is an explicit PATCH from the UI.
    """

    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    NEEDS_INVESTIGATION = "needs_investigation"
    INVALID = "invalid"
    SHIPPED = "shipped"
    ARCHIVED = "archived"


class HypothesisRole(str, Enum):
    """Distinguishes the one primary hypothesis from any secondary ones."""

    PRIMARY = "primary"
    SECONDARY = "secondary"


class RoledHypothesis(CamelModel):
    """
    One hypothesis on a definition, with its role attached.

    Reuses `Hypothesis` (schemas/hypothesis.py) unmodified for the
    statement/metric/direction/effect/rationale fields — this wrapper
    only adds `role` on top, rather than duplicating that model's
    validation logic. See `ExperimentDefinition._validate_hypotheses`
    for the "exactly one PRIMARY" rule enforced at the definition level
    (a lone `RoledHypothesis` doesn't know about its siblings, so that
    check can't live here).
    """

    role: HypothesisRole
    hypothesis: Hypothesis


class MetricRole(str, Enum):
    """Classifies a metric's role on the experiment (Stage 0 doc §8)."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    GUARDRAIL = "guardrail"


class ExperimentMetric(CamelModel):
    """
    A metric configured on the definition — descriptive/planning
    metadata only (see module docstring). The statistical engine
    derives its own primary/guardrail metrics from the dataset at
    analysis time; this is NOT a second, independent metric-selection
    mechanism, same boundary `Hypothesis.primary_metric` already
    documents.
    """

    name: str = Field(description='Display name, e.g. "Signup Conversion".')
    role: MetricRole
    type: MetricType
    description: str | None = Field(default=None, description="Optional free-text description.")
    field_definition: str | None = Field(
        default=None,
        description='Optional event/field this metric is computed from, e.g. "signup_event".',
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        stripped = (value or "").strip()
        if not stripped:
            raise ValueError("Metric name is required and must not be empty.")
        if len(stripped) > _MAX_NAME_LENGTH:
            raise ValueError(f"Metric name must be at most {_MAX_NAME_LENGTH} characters.")
        return stripped


class Variant(CamelModel):
    """
    One arm of the experiment. Allocation validation (exactly one
    control, allocations summing to ~100%) is enforced at the
    definition level in `ExperimentDefinition._validate_variants`,
    same reasoning as `RoledHypothesis` above — a single `Variant`
    can't see its siblings.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    description: str | None = None
    is_control: bool = False
    allocation_pct: float = Field(ge=0, le=100)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        stripped = (value or "").strip()
        if not stripped:
            raise ValueError("Variant name is required and must not be empty.")
        if len(stripped) > _MAX_NAME_LENGTH:
            raise ValueError(f"Variant name must be at most {_MAX_NAME_LENGTH} characters.")
        return stripped


class Targeting(CamelModel):
    """
    Lightweight experiment-configuration audience filter — explicitly
    NOT a production targeting engine (Stage 0 doc §5). Every field is
    optional free-form metadata; nothing here is validated against a
    real user population.
    """

    countries: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    user_type: str | None = Field(default=None, description='e.g. "new", "returning", "all".')
    acquisition_channel: str | None = None
    user_segment: str | None = None
    traffic_allocation_pct: float | None = Field(default=None, ge=0, le=100)


class Exposure(CamelModel):
    """
    Assigned vs. exposed users (Stage 0 doc §7) — populated once
    known, either from the dataset or entered manually. Both optional:
    this phase does not compute either value, only carries them.
    """

    assigned_users: int | None = Field(default=None, ge=0)
    exposed_users: int | None = Field(default=None, ge=0)


class DataSourceType(str, Enum):
    UPLOADED_CSV = "uploaded_csv"
    EXISTING_DATASET = "existing_dataset"
    PUBLIC_DATASET = "public_dataset"


class DataSourceRef(CamelModel):
    """
    Reference to the dataset this experiment will be analyzed against.
    `dataset_id` matches the id returned by POST /datasets/classify
    (see routes_datasets.py) — this is a reference, not a copy; the
    dataset itself is never duplicated into the definition.
    """

    type: DataSourceType = DataSourceType.EXISTING_DATASET
    dataset_id: str | None = None
    dataset_name: str | None = None


class ExperimentDefinitionBase(CamelModel):
    """Fields shared by create/update requests and the full record."""

    name: str
    product_area: str | None = None
    owner: str | None = None
    team: str | None = None
    status: ExperimentStatus = ExperimentStatus.DRAFT

    problem_statement: str | None = None
    objective: str | None = None

    hypotheses: list[RoledHypothesis] = Field(default_factory=list)
    variants: list[Variant] = Field(default_factory=list)
    targeting: Targeting = Field(default_factory=Targeting)
    metrics: list[ExperimentMetric] = Field(default_factory=list)
    exposure: Exposure = Field(default_factory=Exposure)

    expected_duration_days: int | None = Field(default=None, gt=0)
    target_sample_size: int | None = Field(default=None, gt=0)
    mde_relative_pct: float | None = Field(default=None, gt=0)

    data_source: DataSourceRef | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        stripped = (value or "").strip()
        if not stripped:
            raise ValueError("Experiment name is required and must not be empty.")
        if len(stripped) > _MAX_NAME_LENGTH:
            raise ValueError(f"Experiment name must be at most {_MAX_NAME_LENGTH} characters.")
        return stripped

    @field_validator("problem_statement", "objective")
    @classmethod
    def _validate_long_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if len(stripped) > _MAX_TEXT_LENGTH:
            raise ValueError(f"Text fields must be at most {_MAX_TEXT_LENGTH} characters.")
        return stripped

    @model_validator(mode="after")
    def _validate_hypotheses(self) -> "ExperimentDefinitionBase":
        if not self.hypotheses:
            return self
        primary_count = sum(1 for h in self.hypotheses if h.role == HypothesisRole.PRIMARY)
        if primary_count == 0:
            raise ValueError(
                "At least one hypothesis must be marked PRIMARY when hypotheses are provided."
            )
        if primary_count > 1:
            raise ValueError(
                f"Exactly one hypothesis may be marked PRIMARY, found {primary_count}."
            )
        return self

    @model_validator(mode="after")
    def _validate_variants(self) -> "ExperimentDefinitionBase":
        if not self.variants:
            return self
        control_count = sum(1 for v in self.variants if v.is_control)
        if control_count == 0:
            raise ValueError("Exactly one variant must be marked as Control when variants are provided.")
        if control_count > 1:
            raise ValueError(f"Exactly one variant may be marked as Control, found {control_count}.")

        ids = [v.id for v in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("Variant ids must be unique within an experiment.")

        total_allocation = sum(v.allocation_pct for v in self.variants)
        if abs(total_allocation - 100.0) > _ALLOCATION_TOLERANCE_PCT:
            raise ValueError(
                f"Variant allocations must sum to 100% (got {total_allocation:.2f}%)."
            )
        return self

    @model_validator(mode="after")
    def _validate_metrics(self) -> "ExperimentDefinitionBase":
        if not self.metrics:
            return self
        primary_count = sum(1 for m in self.metrics if m.role == MetricRole.PRIMARY)
        if primary_count > 1:
            raise ValueError(f"At most one metric may be marked PRIMARY, found {primary_count}.")
        return self


class ExperimentDefinitionCreateRequest(ExperimentDefinitionBase):
    """Body for POST /experiment-definitions. `status` defaults to DRAFT."""


class ExperimentDefinitionUpdateRequest(CamelModel):
    """
    Body for PATCH /experiment-definitions/{id}. Every field optional —
    only fields actually provided are changed (partial update); a field
    explicitly set to `null` in the request is NOT distinguished from
    "omitted" in this phase (full `exclude_unset` PATCH semantics are a
    later refinement, not required for Phase 1's CRUD contract).
    """

    name: str | None = None
    product_area: str | None = None
    owner: str | None = None
    team: str | None = None
    status: ExperimentStatus | None = None
    problem_statement: str | None = None
    objective: str | None = None
    hypotheses: list[RoledHypothesis] | None = None
    variants: list[Variant] | None = None
    targeting: Targeting | None = None
    metrics: list[ExperimentMetric] | None = None
    exposure: Exposure | None = None
    expected_duration_days: int | None = None
    target_sample_size: int | None = None
    mde_relative_pct: float | None = None
    data_source: DataSourceRef | None = None


class ExperimentDefinition(ExperimentDefinitionBase):
    """Full persisted record, as returned by the store/API."""

    id: str
    created_at: datetime
    updated_at: datetime


class ExperimentDefinitionSummary(CamelModel):
    """Lightweight row for the Experiment Library list view."""

    id: str
    name: str
    status: ExperimentStatus
    product_area: str | None = None
    owner: str | None = None
    primary_metric: str | None = None
    created_at: datetime
    updated_at: datetime
