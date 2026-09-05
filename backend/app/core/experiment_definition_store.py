"""
Persistent store for `ExperimentDefinition` — Phase 1 of the
Experiment Platform layer (see Stage 0 architecture doc).

Same design pattern as `app/core/experiment_store.py` (ABC interface +
SQLAlchemy implementation + lazy process-wide singleton), deliberately
kept as its OWN module/table rather than folded into
`ExperimentStore`: a definition's lifecycle (DRAFT -> ... -> ARCHIVED)
is independent of, and can outlive or precede, any single analysis
run. The two stores share the same underlying database (same
`app_settings.database_url`) but own separate tables and separate
engine/session-factory singletons, exactly like `ExperimentStore` and
`DatasetStore` already do.

The graph and the stats engine never see this module at all in this
phase — it is read/written only by `routes_experiment_definitions.py`.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import app_settings
from app.schemas.experiment_definition import (
    DataSourceRef,
    Exposure,
    ExperimentDefinition,
    ExperimentDefinitionSummary,
    ExperimentStatus,
    ExperimentMetric,
    MetricRole,
    RandomizationUnit,
    RoledHypothesis,
    Targeting,
    Variant,
)


class ExperimentDefinitionStore(ABC):
    """Interface the API routes depend on — never the DB directly."""

    @abstractmethod
    def create(self, definition_in) -> ExperimentDefinition:
        """Persists a new ExperimentDefinition (status defaults to DRAFT)."""

    @abstractmethod
    def get(self, definition_id: str) -> ExperimentDefinition | None:
        """Looks up one definition by id. Returns None if unknown."""

    @abstractmethod
    def list(self) -> list[ExperimentDefinition]:
        """Lists all definitions, most recently updated first."""

    @abstractmethod
    def list_summaries(self) -> list[ExperimentDefinitionSummary]:
        """Lightweight rows for the Experiment Library list view."""

    @abstractmethod
    def update(self, definition_id: str, update_in) -> ExperimentDefinition | None:
        """Applies a partial update. Returns None if unknown."""

    @abstractmethod
    def delete(self, definition_id: str) -> bool:
        """Deletes a definition. Returns False if unknown."""


class Base(DeclarativeBase):
    pass


class ExperimentDefinitionModel(Base):
    __tablename__ = "experiment_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    name: Mapped[str] = mapped_column(String(200))
    product_area: Mapped[str | None] = mapped_column(String(200), nullable=True)
    owner: Mapped[str | None] = mapped_column(String(200), nullable=True)
    team: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(32))

    problem_statement: Mapped[str | None] = mapped_column(Text, nullable=True)
    objective: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Structured sub-objects stored as JSON blobs, validated back through
    # their Pydantic models on read — same "persist verbatim, re-validate
    # on the way out" pattern as ExperimentModel.report_json in
    # experiment_store.py.
    hypotheses_json: Mapped[list] = mapped_column(JSON, default=list)
    variants_json: Mapped[list] = mapped_column(JSON, default=list)
    targeting_json: Mapped[dict] = mapped_column(JSON, default=dict)
    # NOTE: added after this table's first deploy. `Base.metadata.create_all()`
    # (see SQLExperimentDefinitionStore.__init__ below) only creates
    # tables that don't exist yet — it does NOT ALTER an existing table
    # to add new columns (this project has no Alembic/migration
    # tooling). A fresh DB picks this column up automatically; an
    # already-deployed Postgres DB needs a one-time manual
    # `ALTER TABLE experiment_definitions ADD COLUMN randomization_unit
    # VARCHAR(16) NOT NULL DEFAULT 'user'` before this column is read.
    randomization_unit: Mapped[str] = mapped_column(String(16), default=RandomizationUnit.USER.value)
    metrics_json: Mapped[list] = mapped_column(JSON, default=list)
    exposure_json: Mapped[dict] = mapped_column(JSON, default=dict)

    expected_duration_days: Mapped[int | None] = mapped_column(nullable=True)
    target_sample_size: Mapped[int | None] = mapped_column(nullable=True)
    mde_relative_pct: Mapped[float | None] = mapped_column(nullable=True)

    data_source_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


def _row_to_record(row: ExperimentDefinitionModel) -> ExperimentDefinition:
    return ExperimentDefinition(
        id=row.id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        name=row.name,
        product_area=row.product_area,
        owner=row.owner,
        team=row.team,
        status=ExperimentStatus(row.status),
        problem_statement=row.problem_statement,
        objective=row.objective,
        hypotheses=[RoledHypothesis.model_validate(h) for h in (row.hypotheses_json or [])],
        variants=[Variant.model_validate(v) for v in (row.variants_json or [])],
        targeting=Targeting.model_validate(row.targeting_json or {}),
        randomization_unit=RandomizationUnit(row.randomization_unit),
        metrics=[ExperimentMetric.model_validate(m) for m in (row.metrics_json or [])],
        exposure=Exposure.model_validate(row.exposure_json or {}),
        expected_duration_days=row.expected_duration_days,
        target_sample_size=row.target_sample_size,
        mde_relative_pct=row.mde_relative_pct,
        data_source=DataSourceRef.model_validate(row.data_source_json) if row.data_source_json else None,
    )


def _primary_metric_name(row: ExperimentDefinitionModel) -> str | None:
    for m in row.metrics_json or []:
        if m.get("role") == MetricRole.PRIMARY.value:
            return m.get("name")
    return None


def _row_to_summary(row: ExperimentDefinitionModel) -> ExperimentDefinitionSummary:
    return ExperimentDefinitionSummary(
        id=row.id,
        name=row.name,
        status=ExperimentStatus(row.status),
        product_area=row.product_area,
        owner=row.owner,
        primary_metric=_primary_metric_name(row),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SQLExperimentDefinitionStore(ExperimentDefinitionStore):
    """ExperimentDefinitionStore backed by SQLAlchemy — SQLite or Postgres."""

    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self._engine = create_engine(database_url, connect_args=connect_args)
        Base.metadata.create_all(self._engine)
        self._Session: sessionmaker[Session] = sessionmaker(bind=self._engine)

    def create(self, definition_in) -> ExperimentDefinition:
        definition_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        row = ExperimentDefinitionModel(
            id=definition_id,
            created_at=now,
            updated_at=now,
            name=definition_in.name,
            product_area=definition_in.product_area,
            owner=definition_in.owner,
            team=definition_in.team,
            status=definition_in.status.value,
            problem_statement=definition_in.problem_statement,
            objective=definition_in.objective,
            hypotheses_json=json.loads(
                json.dumps([h.model_dump(by_alias=True) for h in definition_in.hypotheses])
            ),
            variants_json=json.loads(
                json.dumps([v.model_dump(by_alias=True) for v in definition_in.variants])
            ),
            targeting_json=json.loads(definition_in.targeting.model_dump_json(by_alias=True)),
            randomization_unit=definition_in.randomization_unit.value,
            metrics_json=json.loads(
                json.dumps([m.model_dump(by_alias=True) for m in definition_in.metrics])
            ),
            exposure_json=json.loads(definition_in.exposure.model_dump_json(by_alias=True)),
            expected_duration_days=definition_in.expected_duration_days,
            target_sample_size=definition_in.target_sample_size,
            mde_relative_pct=definition_in.mde_relative_pct,
            data_source_json=(
                json.loads(definition_in.data_source.model_dump_json(by_alias=True))
                if definition_in.data_source is not None
                else None
            ),
        )
        with self._Session() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return _row_to_record(row)

    def get(self, definition_id: str) -> ExperimentDefinition | None:
        with self._Session() as session:
            row = session.get(ExperimentDefinitionModel, definition_id)
            return _row_to_record(row) if row else None

    def list(self) -> list[ExperimentDefinition]:
        with self._Session() as session:
            rows = session.scalars(
                select(ExperimentDefinitionModel).order_by(
                    ExperimentDefinitionModel.updated_at.desc()
                )
            ).all()
            return [_row_to_record(r) for r in rows]

    def list_summaries(self) -> list[ExperimentDefinitionSummary]:
        with self._Session() as session:
            rows = session.scalars(
                select(ExperimentDefinitionModel).order_by(
                    ExperimentDefinitionModel.updated_at.desc()
                )
            ).all()
            return [_row_to_summary(r) for r in rows]

    def update(self, definition_id: str, update_in) -> ExperimentDefinition | None:
        with self._Session() as session:
            row = session.get(ExperimentDefinitionModel, definition_id)
            if row is None:
                return None

            data = update_in.model_dump(exclude_unset=True, by_alias=False)

            if "name" in data and data["name"] is not None:
                row.name = data["name"]
            if "product_area" in data:
                row.product_area = data["product_area"]
            if "owner" in data:
                row.owner = data["owner"]
            if "team" in data:
                row.team = data["team"]
            if "status" in data and data["status"] is not None:
                row.status = update_in.status.value
            if "problem_statement" in data:
                row.problem_statement = data["problem_statement"]
            if "objective" in data:
                row.objective = data["objective"]
            if "hypotheses" in data and data["hypotheses"] is not None:
                row.hypotheses_json = json.loads(
                    json.dumps([h.model_dump(by_alias=True) for h in update_in.hypotheses])
                )
            if "variants" in data and data["variants"] is not None:
                row.variants_json = json.loads(
                    json.dumps([v.model_dump(by_alias=True) for v in update_in.variants])
                )
            if "targeting" in data and data["targeting"] is not None:
                row.targeting_json = json.loads(update_in.targeting.model_dump_json(by_alias=True))
            if "randomization_unit" in data and data["randomization_unit"] is not None:
                row.randomization_unit = update_in.randomization_unit.value
            if "metrics" in data and data["metrics"] is not None:
                row.metrics_json = json.loads(
                    json.dumps([m.model_dump(by_alias=True) for m in update_in.metrics])
                )
            if "exposure" in data and data["exposure"] is not None:
                row.exposure_json = json.loads(update_in.exposure.model_dump_json(by_alias=True))
            if "expected_duration_days" in data:
                row.expected_duration_days = data["expected_duration_days"]
            if "target_sample_size" in data:
                row.target_sample_size = data["target_sample_size"]
            if "mde_relative_pct" in data:
                row.mde_relative_pct = data["mde_relative_pct"]
            if "data_source" in data:
                row.data_source_json = (
                    json.loads(update_in.data_source.model_dump_json(by_alias=True))
                    if update_in.data_source is not None
                    else None
                )

            row.updated_at = datetime.now(timezone.utc)
            session.add(row)
            session.commit()
            session.refresh(row)
            return _row_to_record(row)

    def delete(self, definition_id: str) -> bool:
        with self._Session() as session:
            row = session.get(ExperimentDefinitionModel, definition_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True


_store: ExperimentDefinitionStore | None = None


def get_experiment_definition_store() -> ExperimentDefinitionStore:
    """
    Returns the process-wide ExperimentDefinitionStore singleton,
    constructing it on first use from `app_settings.database_url` —
    same lazy-singleton pattern, same production-SQLite guard rationale
    as `get_experiment_store()` (app/core/experiment_store.py), just
    against this module's own table.
    """
    global _store
    if _store is None:
        url = app_settings.database_url
        if app_settings.environment == "production" and url.startswith("sqlite"):
            raise RuntimeError(
                "DATABASE_URL is unset/sqlite in a production environment. "
                "Set DATABASE_URL to a hosted Postgres-compatible database "
                "(e.g. Neon, Supabase, or Vercel Postgres) — a local SQLite "
                "file does not survive between serverless invocations."
            )
        _store = SQLExperimentDefinitionStore(url)
    return _store
