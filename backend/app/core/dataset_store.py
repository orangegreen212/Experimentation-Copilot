"""
Shared dataset store — `dataset_id` -> DataFrame.

Persisted to the same database as ExperimentStore (SQLite locally,
Postgres in production via DATABASE_URL) rather than kept in a
process-local dict. A process-local dict does not survive a restart,
a redeploy, or a request landing on a different worker/instance than
the one that received the original upload — on Render specifically,
free-tier dynos spin down on idle and lose all in-memory state, which
surfaced as spurious "Unknown dataset_id" 404s between an upload and
the analyze call that follows it. Persisting the same way experiment
history already does (see core/experiment_store.py) closes that gap.

CRITICAL (unchanged from before): the raw DataFrame must NEVER be
stored in `GraphState`. If it were, LangGraph would include it in
every node's traced input/output (LangSmith traces state at each
node, not just once at the root) — for a 290K-row dataset that's tens
of MB PER NODE, which both exceeds LangSmith's payload limit and
directly violates this project's "no raw data leaves Python, ever"
principle (see report_generator.py, planner_strategy.py docstrings —
this is the same guarantee, extended to observability, not just to
the LLM).

Instead, `GraphState` carries only the lightweight `dataset_id`
string. Any node that needs the actual data calls `get_dataset()`
here, fresh, each time — a DB lookup + deserialize, not a re-parse of
the original upload, and no different in shape from the in-memory
dict this replaces.

Public interface (`store_dataset` / `get_dataset`) is unchanged on
purpose — this was already the swap seam called out in the previous
version of this module's docstring; callers (classifier_node,
experiment_node, funnel_node, validation_node, routes_experiments,
routes_datasets) needed no changes.

PERFORMANCE (added after profiling a 294K-row dataset): a single
`/experiments/analyze` request calls `get_dataset()` up to 4 times —
once in the route as an existence check, then once each from
classifier_node, validation_node, and experiment_node — and every
call independently re-ran `pd.read_json(orient="table")` on the full
DataFrame. Measured at ~5.5-6.5s per deserialize on a 294K-row
dataset, 4 redundant deserializes accounted for the bulk of the
20+ second request time that was tripping Render's request timeout.

Fix: an optional, REQUEST-SCOPED in-memory cache (`dataset_request_scope()`
below), not a persistent/global one. Wrap the handling of one HTTP
request in `with dataset_request_scope():` and every `get_dataset()`
call for the same `dataset_id` within that block is served from
memory after the first real DB read + deserialize. The cache is
strictly request-scoped (implemented with `contextvars.ContextVar`,
which is coroutine/task-local, so concurrent requests never see each
other's cache) and is always torn down when the `with` block exits —
including via exception — so nothing leaks between requests or grows
unbounded across the life of the process.

This does NOT change:
  - the persistence format (still `to_json(orient="table")` /
    `read_json(orient="table")` — see round-trip fidelity findings
    that ruled out `orient="split"`)
  - restart/spin-down behavior (the DB row is still the source of
    truth; the cache only avoids re-reading it within one request)
  - `GraphState` (the DataFrame is still never put there — see the
    LangSmith-tracing note above, unchanged)
  - any classifier/statistics logic

Outside of a `dataset_request_scope()` block (e.g. a bare script or a
test that never opens the scope), `get_dataset()` behaves exactly as
before: every call is a fresh DB read + deserialize. The cache is
opt-in via the context manager, not the default for every caller.
"""

from __future__ import annotations

import base64
import gzip
import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import Iterator

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import app_settings
from app.core.logging import get_node_logger

# TEMPORARY DIAGNOSTIC INSTRUMENTATION — see store_dataset() below.
log = get_node_logger("DatasetStore")

# --- Compression (Variant A: base64(gzip(json)) in the existing Text
# column) -------------------------------------------------------------
#
# WHY: production instrumentation (see store_dataset()'s timing logs)
# showed the DB round-trip — not JSON serialization itself — as the
# dominant cost for large datasets (~294K rows -> ~79MB of
# `to_json(orient="table")` JSON text sent to Postgres). gzip shrinks
# that ~9x (~79MB -> ~8.8MB before base64; ~11.7MB after, since base64
# adds ~33% overhead) with negligible CPU cost, without touching the
# database schema, the column type, or the logical JSON format that
# `orient="table"` round-trips (dtype fidelity, etc.) — only what's
# physically sent over the wire and stored on disk changes.
#
# The column stays `Text`. A short, unambiguous prefix marks which
# rows are compressed so old, already-stored plain-JSON rows (written
# before this change) remain readable forever without a migration —
# see _decode_payload()'s docstring for the exact rule.
_COMPRESSED_PREFIX = "gzip+b64:"
_CSV_COMPRESSED_PREFIX = "csv+gzip+b64:"


def _encode_payload(json_str: str) -> str:
    """
    Compresses a `to_json(orient="table")` string for storage in the
    existing `Text` column: gzip -> base64 -> prefixed with
    `gzip+b64:` so `_decode_payload()` can recognize it unambiguously.
    Plain JSON (as produced by `to_json`) can start with `{` or `[`,
    neither of which collides with this prefix, so detection is safe.
    """
    compressed_bytes = gzip.compress(json_str.encode("utf-8"))
    encoded = base64.b64encode(compressed_bytes).decode("ascii")
    return _COMPRESSED_PREFIX + encoded


def _decode_payload(stored_value: str) -> str:
    """
    Reverses `_encode_payload()`, and is backward-compatible with rows
    written before compression was added: if `stored_value` starts
    with `gzip+b64:`, it's decoded (base64 -> gzip decompress) back to
    the original JSON string. Otherwise `stored_value` IS the original
    JSON string already (the pre-compression format) and is returned
    unchanged — no migration of old rows required, they are simply
    never re-encoded and read via this fallback path indefinitely.
    """
    if stored_value.startswith(_COMPRESSED_PREFIX):
        encoded = stored_value[len(_COMPRESSED_PREFIX):]
        compressed_bytes = base64.b64decode(encoded)
        return gzip.decompress(compressed_bytes).decode("utf-8")
    return stored_value



def _encode_csv_payload(raw_bytes: bytes) -> str:
    """Compress original CSV bytes for large-dataset persistence.

    Keeping the original CSV avoids the huge Python/JSON intermediate
    allocations produced by serializing a 294K+ row DataFrame to a
    column-oriented JSON object. The payload stays in the existing Text
    column via base64, so no database migration is required.
    """
    compressed = gzip.compress(raw_bytes, compresslevel=6)
    return _CSV_COMPRESSED_PREFIX + base64.b64encode(compressed).decode("ascii")


def _decode_csv_payload(stored_value: str) -> bytes:
    encoded = stored_value[len(_CSV_COMPRESSED_PREFIX):]
    return gzip.decompress(base64.b64decode(encoded))

# --- Storage format (Variant "columnar_v1") ---------------------------
#
# WHY: production instrumentation showed the memory spike was neither
# serialization nor the DB round-trip (those were already fixed by
# compression above) but `pd.read_json(orient="table")` ITSELF during
# deserialization: measured at ~1.17 GB peak for a 294K-row dataset
# whose JSON text is only ~79 MB (~14.8x blowup) — comfortably enough
# to exceed Render's 512 MB instance limit on its own. This happens
# because `orient="table"` parses JSON into an intermediate row-wise
# structure (conceptually one Python dict per row) before assembling
# columns, and pandas' own JSON reader carries substantial per-call
# overhead for large frames.
#
# Fix: store a hand-rolled COLUMN-oriented JSON shape instead —
# `{"format": "columnar_v1", "schema": {col: dtype_str}, "column_order":
# [...], "columns": {col: [values...]}}` — parsed with the STDLIB
# `json` module (a straight, lossless JSON<->Python mapping with no
# type-guessing) instead of pandas' JSON reader, then assembled into a
# DataFrame in one vectorized `pd.DataFrame(columns_dict)` call and had
# its dtypes reapplied explicitly from the stored schema.
#
# This is NOT the same risk as the earlier-rejected `orient="split"`:
# split's corruption (e.g. a leading-zero string ID "000001" silently
# becoming the int 1) came from PANDAS' OWN type-inference guessing at
# parse time, which runs whether or not a schema is reapplied
# afterward — by the time schema reapplication happens, the damage is
# already done. `json.loads` does no such guessing: a JSON string
# stays a Python str, full stop. Verified empirically (see
# tests/core/test_dataset_store.py::TestColumnarV1Format) against every
# case that broke `split`: leading-zero IDs, datetime-like strings,
# nullable Int64/boolean, None/NaN, float precision.
#
# Measured on the real 294,478-row dataset: peak memory during
# reconstruction dropped from ~1.17 GB (orient="table") to ~208 MB
# (columnar_v1) — a ~5.6x reduction — with byte-for-byte identical
# reconstructed values.
_COLUMNAR_V1_FORMAT = "columnar_v1"


def _serialize_dataframe(df: pd.DataFrame) -> str:
    """
    Serializes a DataFrame to the columnar_v1 JSON shape (see module
    docstring section above). Used by store_dataset() going forward;
    replaces the old `df.to_json(orient="table")` call. Uses the
    stdlib `json` module, not pandas' JSON writer.
    """
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    columns: dict[str, list] = {}
    for col in df.columns:
        dtype_str = schema[col]
        series = df[col]
        if dtype_str.startswith("datetime64"):
            # datetime isn't a native JSON type -- stringify explicitly,
            # preserving nulls as None (not the string "NaT").
            columns[col] = series.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").where(series.notna(), None).tolist()
        elif dtype_str in ("Int64", "boolean"):
            # pandas nullable extension dtypes: .tolist() on these can
            # leave pd.NA (not JSON-serializable) instead of None for
            # missing entries, so convert element-by-element explicitly.
            columns[col] = [
                None if pd.isna(v) else (bool(v) if dtype_str == "boolean" else int(v)) for v in series
            ]
        else:
            # object/str, int64, float64, bool: Series.tolist() already
            # converts numpy scalars (np.int64, np.float64, np.bool_) to
            # native Python types that `json` can serialize directly.
            # NaN (float64) / None (object) both need to become JSON
            # null explicitly -- plain `math.isnan`-style checks would
            # miss None, so pd.isna() (which handles both) is used.
            columns[col] = [None if pd.isna(v) else v for v in series.tolist()]

    payload = {
        "format": _COLUMNAR_V1_FORMAT,
        "schema": schema,
        "column_order": list(df.columns),
        "columns": columns,
    }
    return json.dumps(payload)


def _deserialize_dataframe(json_str: str) -> pd.DataFrame:
    """
    Reconstructs a DataFrame from a stored JSON string, dispatching on
    an explicit format marker rather than guessing from JSON shape:

      - `{"format": "columnar_v1", ...}` -> the new, low-memory path:
        `json.loads` (stdlib, no pandas JSON parser involved) + build
        the DataFrame from the column-oriented dict directly + reapply
        dtypes from the stored schema.
      - anything else (no "format" key, or a JSON object that doesn't
        match) -> treated as a pre-columnar_v1 record, written by an
        older version of store_dataset() as plain
        `df.to_json(orient="table")`. Read via the original
        `pd.read_json(orient="table")` path -- unchanged, no
        migration, exactly as readable as it always was. These records
        keep paying the higher memory cost this change was meant to
        avoid, but that is the pre-existing status quo for them, not a
        regression.
    """
    try:
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        # Old format: `df.to_json(orient="table")` output IS valid JSON
        # too, so a decode error here would mean something else is
        # wrong -- but pd.read_json can also accept some inputs
        # json.loads rejects (none expected in practice for this
        # module's own writes); fall through to the old path either way.
        return pd.read_json(StringIO(json_str), orient="table")

    if not isinstance(parsed, dict) or parsed.get("format") != _COLUMNAR_V1_FORMAT:
        # Pre-columnar_v1 record (plain orient="table" JSON, which is
        # also a JSON object but without our "format" marker key).
        return pd.read_json(StringIO(json_str), orient="table")

    # --- TEMPORARY DIAGNOSTIC INSTRUMENTATION -------------------------
    # Memory instrumentation around ONLY this new reconstruction path,
    # to verify the measured ~208MB local peak (vs ~1.17GB for the old
    # orient="table" path) holds on Render's actual 512MB instance.
    # Remove once confirmed in production.
    import tracemalloc

    tracemalloc.start()
    _t_start = time.perf_counter()
    # --- END TEMPORARY DIAGNOSTIC INSTRUMENTATION (setup) -------------

    schema: dict[str, str] = parsed["schema"]
    column_order: list[str] = parsed.get("column_order", list(parsed["columns"].keys()))
    columns: dict[str, list] = parsed["columns"]

    df = pd.DataFrame(columns)  # one vectorized array per column, no row-wise intermediate
    df = df[column_order]  # explicit column order, not whatever dict/JSON iteration happened to give

    for col in column_order:
        dtype_str = schema[col]
        if dtype_str.startswith("datetime64"):
            df[col] = pd.to_datetime(df[col])
        elif dtype_str in ("Int64", "boolean"):
            df[col] = df[col].astype(dtype_str)
        elif dtype_str != "object":
            df[col] = df[col].astype(dtype_str)
        # dtype_str == "object": already Python str/None from json.loads,
        # no coercion needed -- this is exactly the step that would have
        # corrupted a leading-zero ID under orient="split"'s own
        # auto-inference; here there is nothing to correct because
        # nothing was auto-inferred in the first place.

    # --- TEMPORARY DIAGNOSTIC INSTRUMENTATION -------------------------
    _t_end = time.perf_counter()
    _current, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log.info(
        "[DatasetStore] columnar_v1 deserialize completed in %.2fs, peak memory %.2f MB",
        _t_end - _t_start,
        _peak / 1e6,
    )
    # --- END TEMPORARY DIAGNOSTIC INSTRUMENTATION ----------------------

    return df


# (get_dataset falls back to always reading from the DB, same as
# before this change). A `dataset_request_scope()` block sets this to
# a fresh dict for the duration of that request/task only.
#
# ContextVar (not a plain module-level dict) specifically because it
# is coroutine/asyncio-Task-local: two concurrent `async def` request
# handlers each get their own independent view of this variable, so
# one request's cached DataFrame is never visible to another's, even
# though FastAPI/uvicorn run them in the same process/event loop.
_dataset_cache: ContextVar[dict[str, pd.DataFrame] | None] = ContextVar(
    "_dataset_cache", default=None
)


@contextmanager
def dataset_request_scope() -> Iterator[None]:
    """
    Opens a request-scoped dataset cache for the duration of the
    `with` block. Every `get_dataset(dataset_id)` call inside the
    block reuses the same deserialized DataFrame for a given
    `dataset_id` instead of re-reading and re-parsing it from the DB.

    Always tears the cache down on exit, including when the block
    raises — a failed analysis (e.g. a bad dataset, a graph node
    error) must not leak a cached DataFrame into whatever request
    happens to run next on this process.

    Callers: `routes_experiments.py`'s `analyze_experiment()` wraps
    its existence-check + `graph.invoke(...)` call in this scope, so
    all `get_dataset()` calls for one `/experiments/analyze` request
    (route + classifier_node + validation_node + experiment_node,
    and funnel_node when routed there) share one deserialize.
    """
    token = _dataset_cache.set({})
    try:
        yield
    finally:
        _dataset_cache.reset(token)


class Base(DeclarativeBase):
    pass


class DatasetModel(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # DataFrame serialized via to_json(orient="table"), which embeds
    # the column schema/dtypes alongside the data so read_json can
    # round-trip it without the caller guessing dtypes back.
    data_json: Mapped[str] = mapped_column(Text)


_engine = None
_Session: sessionmaker[Session] | None = None


def _get_session_factory() -> sessionmaker[Session]:
    """
    Lazily builds the engine/session factory from app_settings.database_url,
    same pattern (and same DB) as core/experiment_store.py's
    get_experiment_store(). Built lazily, not at import time, so test
    collection and other environments that never touch the store don't
    pay for a DB connection.
    """
    global _engine, _Session
    if _Session is None:
        url = app_settings.database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
        Base.metadata.create_all(_engine)
        _Session = sessionmaker(bind=_engine)
    return _Session


def store_dataset(df: pd.DataFrame, *, raw_csv_bytes: bytes | None = None) -> str:
    """Persist a DataFrame and return its dataset id.

    For uploaded CSVs, ``raw_csv_bytes`` uses the compact raw-CSV storage
    path. This is substantially safer for large files because it avoids
    materializing a second, much larger JSON representation of the DataFrame.
    Existing callers continue to use the columnar_v1 DataFrame path.
    """
    dataset_id = str(uuid.uuid4())

    if raw_csv_bytes is not None:
        _t_start = time.perf_counter()
        stored_value = _encode_csv_payload(raw_csv_bytes)
        log.info(
            "[DatasetStore] CSV payload compressed in %.2fs (%.2f MB -> %.2f MB)",
            time.perf_counter() - _t_start,
            len(raw_csv_bytes) / 1e6,
            len(stored_value) / 1e6,
        )
    else:
        serialized = _serialize_dataframe(df)
        stored_value = _encode_payload(serialized)

    row = DatasetModel(
        dataset_id=dataset_id,
        created_at=datetime.now(timezone.utc),
        data_json=stored_value,
    )
    session_factory = _get_session_factory()
    with session_factory() as session:
        session.add(row)
        session.commit()

    return dataset_id


def get_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Looks up a previously stored DataFrame. Raises HTTP 404 if unknown.

    If called inside a `dataset_request_scope()` block, reuses a
    DataFrame already deserialized earlier in the same request/task
    instead of re-reading and re-parsing it from the DB. Outside a
    scope, behaves exactly as before this change: a fresh DB read +
    `pd.read_json(orient="table")` every call.
    """
    cache = _dataset_cache.get()
    if cache is not None and dataset_id in cache:
        return cache[dataset_id]

    session_factory = _get_session_factory()
    with session_factory() as session:
        row = session.get(DatasetModel, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset_id: {dataset_id}")
    if row.data_json.startswith(_CSV_COMPRESSED_PREFIX):
        raw_csv = _decode_csv_payload(row.data_json)
        df = pd.read_csv(BytesIO(raw_csv))
    else:
        json_str = _decode_payload(row.data_json)
        df = _deserialize_dataframe(json_str)

    if cache is not None:
        cache[dataset_id] = df
    return df


def dataset_exists(dataset_id: str) -> bool:
    """
    Lightweight existence check — confirms a `dataset_id` is known
    without deserializing its (potentially tens-of-MB) `data_json`
    payload. Selects only the primary-key column.

    Used by `routes_experiments.py`'s pre-flight check, which
    previously called `get_dataset()` and discarded the result purely
    to raise a 404 for an unknown id — that discarded call was one of
    the 4 redundant full deserializes per `/experiments/analyze`
    request (see module docstring).
    """
    session_factory = _get_session_factory()
    with session_factory() as session:
        found_id = session.execute(
            select(DatasetModel.dataset_id).where(DatasetModel.dataset_id == dataset_id)
        ).scalar_one_or_none()
    return found_id is not None
