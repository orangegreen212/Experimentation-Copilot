"""
Persistent Experiment History store.

This is a DIFFERENT concept from:
  - the methodology knowledge base (app/rag/retriever.py — Kohavi /
    Microsoft / Netflix / Booking write-ups, static reference docs)
  - chat memory (the follow-up conversation tied to one experiment)

`ExperimentStore` is only responsible for the persisted record of a
completed experiment run: what was asked, what the graph decided, and
enough of the report to reopen it later from History. Chat messages
for that experiment are a separate table (`ChatMessageRecord`) keyed
by the same `experiment_id`, exposed through the same store so the
route layer has one dependency instead of two.

The graph (`app/graph/`) and the API routes never talk to SQLAlchemy
directly — they only see this interface. That keeps the persistence
technology swappable (SQLite locally, Postgres in production) without
touching graph or route code, and makes the store trivial to fake in
tests.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import app_settings
from app.schemas.chat import ChatMessage, ChatRole
from app.schemas.execution import ExecutionStep
from app.schemas.report import ExperimentReport


# ---------------------------------------------------------------------------
# Plain dataclass-ish records — what the store hands back. Kept separate
# from the API's Pydantic response models (schemas/report.py etc.) so
# the store's contract doesn't depend on API request/response shape.
# ---------------------------------------------------------------------------


class ExperimentSummaryRecord:
    """Lightweight persisted experiment row used by the History list."""

    def __init__(
        self,
        experiment_id: str,
        created_at: datetime,
        dataset_name: str,
        user_prompt: str,
        decision: str,
        confidence: str,
        primary_metric: str,
    ) -> None:
        self.experiment_id = experiment_id
        self.created_at = created_at
        self.dataset_name = dataset_name
        self.user_prompt = user_prompt
        self.decision = decision
        self.confidence = confidence
        self.primary_metric = primary_metric


class ExperimentRecord:
    """One persisted experiment run, as returned by the store."""

    def __init__(
        self,
        experiment_id: str,
        created_at: datetime,
        dataset_id: str,
        dataset_name: str,
        user_prompt: str,
        report: ExperimentReport,
        decision: str,
        confidence: str,
        primary_metric: str,
        execution_steps: list[ExecutionStep],
        definition_id: str | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.created_at = created_at
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.user_prompt = user_prompt
        self.report = report
        self.decision = decision
        self.confidence = confidence
        self.primary_metric = primary_metric
        self.execution_steps = execution_steps
        # Phase 1 groundwork for the Experiment Platform layer — links
        # this analysis run back to the `ExperimentDefinition` it was
        # launched from (app/core/experiment_definition_store.py). None
        # for every run created via the existing /experiments/analyze
        # flow until a later phase wires an "analyze this definition"
        # entry point through; fully optional and backward compatible.
        self.definition_id = definition_id


class ExperimentStore(ABC):
    """Interface the graph/routes depend on — never the DB directly."""

    @abstractmethod
    def create(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        user_prompt: str,
        report: ExperimentReport,
        execution_steps: list[ExecutionStep],
        definition_id: str | None = None,
    ) -> ExperimentRecord:
        """Persists a completed experiment run and returns its record."""

    @abstractmethod
    def get(self, experiment_id: str) -> ExperimentRecord | None:
        """Looks up one experiment by id. Returns None if unknown."""

    @abstractmethod
    def list(self) -> list[ExperimentRecord]:
        """Lists all experiments, most recent first (report/steps included)."""

    @abstractmethod
    def list_summaries(self) -> list["ExperimentSummaryRecord"]:
        """Lists lightweight history rows without deserializing report JSON."""

    @abstractmethod
    def list_related(
        self, dataset_id: str, *, exclude_experiment_id: str | None = None, limit: int = 5
    ) -> list[ExperimentRecord]:
        """
        Structured retrieval of prior runs against the same dataset,
        most recent first — NOT semantic/LLM memory, just a plain filter
        over already-persisted records. Used to give the analyst (and,
        once Stage 8 lands, the LLM) short factual context like "this
        dataset was reviewed before and the decision was X" without any
        embeddings, vector store, or chat-history summarization.

        Keyed by `dataset_id`, NOT `dataset_name`. dataset_name is a
        display label only (the uploaded file name, or the demo file's
        name) and is NOT unique — two different uploads (or two demo
        loads) can legitimately share the same file name while being
        completely different datasets. dataset_id is the identity the
        classifier assigns per classify() call and is what actually
        distinguishes one dataset from another; relating experiments by
        name instead of id would silently merge unrelated datasets that
        happen to share a filename. See routes_experiments.py's
        `dataset_id` field on AnalyzeExperimentRequest — that's the
        value that must be passed here, never `dataset_name`.
        """

    @abstractmethod
    def delete(self, experiment_id: str) -> bool:
        """Deletes an experiment (and its chat history). Returns False if unknown."""

    @abstractmethod
    def add_chat_message(self, experiment_id: str, message: ChatMessage) -> None:
        """Appends one chat message to an experiment's conversation."""

    @abstractmethod
    def list_chat_messages(self, experiment_id: str) -> list[ChatMessage]:
        """Returns the persisted conversation for an experiment, in order."""


# ---------------------------------------------------------------------------
# SQLAlchemy implementation — works against SQLite (dev) or any
# Postgres-compatible DATABASE_URL (production) unchanged, since all
# the DB-specific bits (engine creation, column types) are isolated
# here behind the ExperimentStore interface above.
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class ExperimentModel(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Phase 1 groundwork — nullable link to `experiment_definitions`
    # (app/core/experiment_definition_store.py). Deliberately NOT a
    # SQLAlchemy ForeignKey/relationship across the two stores: each
    # store owns its own Base/engine construction independently (same
    # reasoning as ExperimentStore vs DatasetStore already being
    # separate), so this is a plain indexed string column, validated at
    # the application layer instead of the DB layer. Always None until
    # a later phase adds an "analyze this definition" entry point.
    definition_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    dataset_id: Mapped[str] = mapped_column(String(36), index=True)
    dataset_name: Mapped[str] = mapped_column(String(255))
    user_prompt: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[str] = mapped_column(String(16))
    primary_metric: Mapped[str] = mapped_column(String(255))
    # Full ExperimentReport + ExecutionStep list, stored as JSON blobs.
    # These are already-computed, already-validated Pydantic models by
    # the time they reach here — the store's job is to persist them
    # verbatim and hand back the same shape, not to re-derive anything.
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    execution_steps_json: Mapped[list[Any]] = mapped_column(JSON)


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Frontend-facing message id (ChatMessage.id, e.g. "u-172..."), kept
    # distinct from the DB primary key so client-generated ids round-trip.
    client_id: Mapped[str] = mapped_column(String(64))


def _row_to_record(row: ExperimentModel) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=row.experiment_id,
        created_at=row.created_at,
        dataset_id=row.dataset_id,
        dataset_name=row.dataset_name,
        user_prompt=row.user_prompt,
        report=ExperimentReport.model_validate(row.report_json),
        decision=row.decision,
        confidence=row.confidence,
        primary_metric=row.primary_metric,
        execution_steps=[ExecutionStep.model_validate(s) for s in row.execution_steps_json],
        definition_id=row.definition_id,
    )


class SQLExperimentStore(ExperimentStore):
    """ExperimentStore backed by SQLAlchemy — SQLite or Postgres."""

    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self._engine = create_engine(database_url, connect_args=connect_args)
        Base.metadata.create_all(self._engine)
        self._Session: sessionmaker[Session] = sessionmaker(bind=self._engine)

    def create(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        user_prompt: str,
        report: ExperimentReport,
        execution_steps: list[ExecutionStep],
        definition_id: str | None = None,
    ) -> ExperimentRecord:
        import uuid

        experiment_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        primary_metric = report.stats[0].metric if report.stats else "N/A"
        # Was: decision = "ship" if report.confidence.value == "HIGH" else "hold" — the exact
        # "confidence==HIGH implies ship" conflation the decision-model redesign exists to remove.
        # `report.decision` is the canonical, deterministic Decision enum (report_generator.determine_decision) —
        # never derived from the legacy `confidence` field. See schemas/report.py module docstring.
        decision = report.decision.value

        row = ExperimentModel(
            experiment_id=experiment_id,
            created_at=created_at,
            definition_id=definition_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            user_prompt=user_prompt,
            decision=decision,
            confidence=report.confidence.value,
            primary_metric=primary_metric,
            report_json=json.loads(report.model_dump_json(by_alias=True)),
            execution_steps_json=[
                json.loads(s.model_dump_json(by_alias=True)) for s in execution_steps
            ],
        )
        with self._Session() as session:
            session.add(row)
            session.commit()

        return ExperimentRecord(
            experiment_id=experiment_id,
            created_at=created_at,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            user_prompt=user_prompt,
            report=report,
            decision=decision,
            confidence=report.confidence.value,
            primary_metric=primary_metric,
            execution_steps=execution_steps,
            definition_id=definition_id,
        )

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        with self._Session() as session:
            row = session.get(ExperimentModel, experiment_id)
            return _row_to_record(row) if row else None

    def list(self) -> list[ExperimentRecord]:
        with self._Session() as session:
            rows = session.scalars(
                select(ExperimentModel).order_by(ExperimentModel.created_at.desc())
            ).all()
            return [_row_to_record(r) for r in rows]

    def list_summaries(self) -> list[ExperimentSummaryRecord]:
        # History list only needs metadata. Do not deserialize every saved
        # report here: one legacy/malformed report must not make the entire
        # History endpoint fail with a 500. The full report is validated only
        # when the user opens a specific experiment.
        with self._Session() as session:
            rows = session.scalars(
                select(ExperimentModel).order_by(ExperimentModel.created_at.desc())
            ).all()
            return [
                ExperimentSummaryRecord(
                    experiment_id=r.experiment_id,
                    created_at=r.created_at,
                    dataset_name=r.dataset_name,
                    user_prompt=r.user_prompt,
                    decision=r.decision,
                    confidence=r.confidence,
                    primary_metric=r.primary_metric,
                )
                for r in rows
            ]

    def list_related(
        self, dataset_id: str, *, exclude_experiment_id: str | None = None, limit: int = 5
    ) -> list[ExperimentRecord]:
        with self._Session() as session:
            stmt = (
                select(ExperimentModel)
                .filter(ExperimentModel.dataset_id == dataset_id)
                .order_by(ExperimentModel.created_at.desc())
                .limit(limit)
            )
            if exclude_experiment_id is not None:
                stmt = stmt.filter(ExperimentModel.experiment_id != exclude_experiment_id)
            rows = session.scalars(stmt).all()
            return [_row_to_record(r) for r in rows]

    def delete(self, experiment_id: str) -> bool:
        with self._Session() as session:
            row = session.get(ExperimentModel, experiment_id)
            if row is None:
                return False
            session.delete(row)
            session.query(ChatMessageModel).filter_by(experiment_id=experiment_id).delete()
            session.commit()
            return True

    def add_chat_message(self, experiment_id: str, message: ChatMessage) -> None:
        with self._Session() as session:
            session.add(
                ChatMessageModel(
                    experiment_id=experiment_id,
                    role=message.role.value,
                    message=message.content,
                    created_at=datetime.now(timezone.utc),
                    client_id=message.id,
                )
            )
            session.commit()

    def list_chat_messages(self, experiment_id: str) -> list[ChatMessage]:
        with self._Session() as session:
            rows = session.scalars(
                select(ChatMessageModel)
                .filter_by(experiment_id=experiment_id)
                .order_by(ChatMessageModel.created_at.asc())
            ).all()
            return [
                ChatMessage(id=r.client_id, role=ChatRole(r.role), content=r.message)
                for r in rows
            ]


_store: ExperimentStore | None = None


def get_experiment_store() -> ExperimentStore:
    """
    Returns the process-wide ExperimentStore singleton, constructing it
    on first use from `app_settings.database_url`.

    Refuses to start in production against SQLite: Vercel's filesystem
    is ephemeral/read-only outside /tmp, so a SQLite file would silently
    lose all history on every cold start — exactly the bug this store
    exists to fix. Point DATABASE_URL at a hosted Postgres-compatible
    database (Neon/Supabase/Vercel Postgres) for production.
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
        _store = SQLExperimentStore(url)
    return _store
