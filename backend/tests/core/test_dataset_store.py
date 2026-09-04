"""
Regression tests for the request-scoped dataset cache in
app/core/dataset_store.py.

Context: profiling a 294K-row dataset showed a single
`/experiments/analyze` request calling `get_dataset()` up to 4 times
(route existence-check + classifier_node + validation_node +
experiment_node), each independently re-running
`pd.read_json(orient="table")` on the same DataFrame — ~5.5-6.5s per
call, the dominant cost behind requests tripping Render's timeout.

`dataset_request_scope()` fixes this by caching the deserialized
DataFrame for the lifetime of one `with` block, keyed by
`dataset_id`, using a `contextvars.ContextVar` so concurrent
requests never share a cache.

These tests cover exactly what was asked for:
  1. one request deserializes the dataset only once
  2. subsequent get_dataset() calls in the same request reuse the
     same DataFrame (object identity, not just equal values)
  3. different requests/scopes do not share the cache
  4. cache cleanup happens even when the scope body raises
  5. existing (pre-change) persistence/restart behavior is unchanged
     for callers that never open a scope
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone

import pandas as pd
import pytest

from app.core import dataset_store
from app.core.dataset_store import (
    DatasetModel,
    _decode_payload,
    _deserialize_dataframe,
    _encode_payload,
    _get_session_factory,
    _serialize_dataframe,
    dataset_exists,
    dataset_request_scope,
    get_dataset,
    store_dataset,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [f"U{i}" for i in range(50)],
            "group": (["control", "treatment"] * 25),
            "converted": [i % 2 for i in range(50)],
        }
    )


@pytest.fixture
def stored_dataset_id() -> str:
    return store_dataset(_sample_df())


def _count_deserialize_calls(monkeypatch) -> list[int]:
    """
    Wraps _deserialize_dataframe (the actual deserialization entry
    point used by get_dataset(), regardless of whether a given record
    takes the columnar_v1 path or the pd.read_json(orient="table")
    backward-compat fallback) so we can assert on how many times the
    DataFrame was actually deserialized, not just how many times
    get_dataset() was called (the whole point of the cache is that
    these two numbers diverge).
    """
    calls = {"n": 0}
    real_deserialize = dataset_store._deserialize_dataframe

    def counting_deserialize(*args, **kwargs):
        calls["n"] += 1
        return real_deserialize(*args, **kwargs)

    monkeypatch.setattr(dataset_store, "_deserialize_dataframe", counting_deserialize)
    return calls


class TestDeserializeOncePerRequest:
    def test_multiple_get_dataset_calls_in_one_scope_deserialize_once(
        self, monkeypatch, stored_dataset_id
    ):
        calls = _count_deserialize_calls(monkeypatch)

        with dataset_request_scope():
            get_dataset(stored_dataset_id)  # simulates route existence-check-turned-real-read
            get_dataset(stored_dataset_id)  # simulates classifier_node
            get_dataset(stored_dataset_id)  # simulates validation_node
            get_dataset(stored_dataset_id)  # simulates experiment_node

        assert calls["n"] == 1, (
            f"expected exactly 1 deserialize for 4 get_dataset() calls "
            f"inside one scope, got {calls['n']}"
        )

    def test_without_a_scope_every_call_deserializes(self, monkeypatch, stored_dataset_id):
        """Sanity check that the counting harness itself is correct: with
        no scope open, the pre-existing (uncached) behavior is preserved."""
        calls = _count_deserialize_calls(monkeypatch)

        get_dataset(stored_dataset_id)
        get_dataset(stored_dataset_id)
        get_dataset(stored_dataset_id)

        assert calls["n"] == 3


class TestSameDataFrameReused:
    def test_repeated_calls_in_scope_return_the_same_object(self, stored_dataset_id):
        with dataset_request_scope():
            df1 = get_dataset(stored_dataset_id)
            df2 = get_dataset(stored_dataset_id)
            df3 = get_dataset(stored_dataset_id)

        assert df1 is df2
        assert df2 is df3

    def test_cached_dataframe_has_correct_values(self, stored_dataset_id):
        with dataset_request_scope():
            df = get_dataset(stored_dataset_id)
        original = _sample_df()
        assert list(df.columns) == list(original.columns)
        assert len(df) == len(original)
        assert df["user_id"].tolist() == original["user_id"].tolist()


class TestScopesDoNotShareCache:
    def test_different_scopes_do_not_share_the_dataframe_object(self, stored_dataset_id):
        with dataset_request_scope():
            df_first_request = get_dataset(stored_dataset_id)

        with dataset_request_scope():
            df_second_request = get_dataset(stored_dataset_id)

        # Different scopes must each do their own read — never hand a
        # DataFrame from one "request" to another, even though the
        # underlying data is identical.
        assert df_first_request is not df_second_request

    def test_different_scopes_each_trigger_their_own_deserialize(
        self, monkeypatch, stored_dataset_id
    ):
        calls = _count_deserialize_calls(monkeypatch)

        with dataset_request_scope():
            get_dataset(stored_dataset_id)
            get_dataset(stored_dataset_id)

        with dataset_request_scope():
            get_dataset(stored_dataset_id)
            get_dataset(stored_dataset_id)

        assert calls["n"] == 2, "each scope should deserialize once, independently"

    def test_no_cache_bleeds_outside_a_closed_scope(self, monkeypatch, stored_dataset_id):
        with dataset_request_scope():
            get_dataset(stored_dataset_id)

        # Scope is closed now — a call outside any scope must hit the
        # DB/deserialize again, not silently reuse the closed scope's
        # cached DataFrame.
        calls = _count_deserialize_calls(monkeypatch)
        get_dataset(stored_dataset_id)
        assert calls["n"] == 1


class TestCacheCleanupOnException:
    def test_cache_is_torn_down_when_scope_body_raises(self, monkeypatch, stored_dataset_id):
        class _BoomError(Exception):
            pass

        with pytest.raises(_BoomError):
            with dataset_request_scope():
                get_dataset(stored_dataset_id)
                raise _BoomError("simulated graph node failure mid-analysis")

        # The failed scope must not leak its cache into whatever runs
        # next in this process — verify a fresh scope (or no scope)
        # re-reads from the DB rather than reusing anything left over.
        calls = _count_deserialize_calls(monkeypatch)
        with dataset_request_scope():
            get_dataset(stored_dataset_id)
        assert calls["n"] == 1

    def test_a_later_successful_scope_after_a_failed_one_works_normally(self, stored_dataset_id):
        class _BoomError(Exception):
            pass

        with pytest.raises(_BoomError):
            with dataset_request_scope():
                get_dataset(stored_dataset_id)
                raise _BoomError("simulated failure")

        with dataset_request_scope():
            df = get_dataset(stored_dataset_id)
            assert len(df) == 50


class TestExistingBehaviorUnchanged:
    def test_get_dataset_unknown_id_raises_404_outside_scope(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            get_dataset("does-not-exist")
        assert exc_info.value.status_code == 404

    def test_get_dataset_unknown_id_raises_404_inside_scope(self):
        from fastapi import HTTPException

        with dataset_request_scope():
            with pytest.raises(HTTPException) as exc_info:
                get_dataset("does-not-exist")
        assert exc_info.value.status_code == 404

    def test_store_then_get_round_trips_without_a_scope(self):
        """Unchanged persistence behavior: store_dataset()/get_dataset()
        still work correctly for a caller that never opens a scope at
        all — the cache is strictly opt-in."""
        df = _sample_df()
        dataset_id = store_dataset(df)
        fetched = get_dataset(dataset_id)
        assert fetched["user_id"].tolist() == df["user_id"].tolist()
        assert fetched["group"].tolist() == df["group"].tolist()
        assert fetched["converted"].tolist() == df["converted"].tolist()

    def test_persistence_survives_a_simulated_process_restart(self, stored_dataset_id):
        """The whole point of persisting to the DB instead of an
        in-memory dict (see module docstring) is that a dataset_id
        survives the process's in-memory state being wiped. Simulate
        that by dropping the lazily-built engine/session singletons
        and confirming the dataset is still readable afterward."""
        dataset_store._engine = None
        dataset_store._Session = None

        df = get_dataset(stored_dataset_id)
        assert len(df) == 50


class TestDatasetExists:
    def test_true_for_known_id(self, stored_dataset_id):
        assert dataset_exists(stored_dataset_id) is True

    def test_false_for_unknown_id(self):
        assert dataset_exists("does-not-exist") is False

    def test_does_not_deserialize_the_dataframe(self, monkeypatch, stored_dataset_id):
        """The whole reason dataset_exists() was added: the route's
        pre-flight check should confirm the id exists without paying
        for a full deserialize of a potentially tens-of-MB payload."""
        calls = _count_deserialize_calls(monkeypatch)
        dataset_exists(stored_dataset_id)
        assert calls["n"] == 0


class TestCompression:
    """
    Variant A: base64(gzip(json)) stored in the existing Text column,
    with a `gzip+b64:` prefix so old plain-JSON rows stay readable
    without a migration. See _encode_payload()/_decode_payload()
    docstrings in dataset_store.py.
    """

    def test_newly_stored_dataset_reads_back_correctly(self, stored_dataset_id):
        """Requirement 1: a newly stored dataset can be read back correctly."""
        df = get_dataset(stored_dataset_id)
        original = _sample_df()
        assert list(df.columns) == list(original.columns)
        assert len(df) == len(original)
        assert df["user_id"].tolist() == original["user_id"].tolist()
        assert df["group"].tolist() == original["group"].tolist()
        assert df["converted"].tolist() == original["converted"].tolist()

    def test_stored_row_is_actually_compressed(self, stored_dataset_id):
        """Sanity check that store_dataset() is really writing the
        gzip+b64: prefixed form to the DB, not silently falling back
        to plain JSON."""
        session_factory = _get_session_factory()
        with session_factory() as session:
            row = session.get(DatasetModel, stored_dataset_id)
        assert row is not None
        assert row.data_json.startswith("gzip+b64:")

    def test_decompressed_json_is_identical_to_original_logical_data(self):
        """Requirement 2: decompressed JSON round-trips to the exact
        same logical DataFrame as the pre-compression to_json() output
        would have, for a dataframe with mixed dtypes (not just the
        simple 3-column fixture)."""
        df = pd.DataFrame(
            {
                "user_id": [f"U{i}" for i in range(30)],
                "group": (["control", "treatment"] * 15),
                "revenue": [round(i * 1.11, 2) for i in range(30)],
                "converted": [i % 2 for i in range(30)],
                "note": [None if i % 5 == 0 else f"n{i}" for i in range(30)],
            }
        )
        original_json = _serialize_dataframe(df)

        encoded = _encode_payload(original_json)
        decoded_json = _decode_payload(encoded)

        assert decoded_json == original_json, "decompressed JSON must be byte-for-byte identical to the pre-compression JSON"

        # and, one level up, the DataFrame reconstructed from it must
        # match the original — the same guarantee get_dataset() relies on.
        df_from_original = _deserialize_dataframe(original_json)
        df_from_decoded = _deserialize_dataframe(decoded_json)
        assert df_from_original.equals(df_from_decoded)

    def test_backward_compatible_with_existing_plain_json_row(self):
        """Requirement 3 + 7: a row written before compression was
        added (plain to_json(orient="table") text, no prefix, no
        columnar_v1 marker, no migration applied) must remain fully
        readable through get_dataset()."""
        df = _sample_df()
        plain_json = df.to_json(orient="table")  # NOT compressed, NOT columnar_v1 — simulates the oldest possible row

        dataset_id = str(uuid.uuid4())
        session_factory = _get_session_factory()
        with session_factory() as session:
            session.add(
                DatasetModel(
                    dataset_id=dataset_id,
                    created_at=datetime.now(timezone.utc),
                    data_json=plain_json,  # written directly, bypassing store_dataset()/_encode_payload()/_serialize_dataframe()
                )
            )
            session.commit()

        fetched = get_dataset(dataset_id)
        assert fetched["user_id"].tolist() == df["user_id"].tolist()
        assert fetched["group"].tolist() == df["group"].tolist()
        assert fetched["converted"].tolist() == df["converted"].tolist()

    def test_gzip_b64_prefix_detection_path(self):
        """Requirement 4: the gzip+b64: prefix is what triggers the
        decode path — a value without it is treated as plain JSON
        even if it happens to be valid base64/gzip-like text."""
        plain = '{"schema": {}, "data": []}'
        assert _decode_payload(plain) == plain  # no prefix -> returned as-is, not decoded

        compressed = _encode_payload(plain)
        assert compressed.startswith("gzip+b64:")
        assert compressed != plain
        assert _decode_payload(compressed) == plain

    def test_compression_actually_shrinks_a_large_repetitive_payload(self):
        """Not a strict requirement, but locks in that compression is
        doing real work (catches a future accidental no-op change to
        _encode_payload) — uses a payload large/repetitive enough for
        gzip's ratio to be meaningfully >1."""
        df = pd.DataFrame(
            {
                "user_id": [f"U{i}" for i in range(5000)],
                "group": (["control", "treatment"] * 2500),
                "converted": [i % 2 for i in range(5000)],
            }
        )
        json_str = _serialize_dataframe(df)
        encoded = _encode_payload(json_str)
        assert len(encoded) < len(json_str)


class TestColumnarV1Format:
    """
    Regression tests for the columnar_v1 storage format
    (_serialize_dataframe / _deserialize_dataframe), added to fix an
    OOM on Render: `pd.read_json(orient="table")` measured a ~1.17GB
    memory peak reconstructing a 294,478-row dataset (vs Render's
    512MB instance limit), because it parses JSON into a row-wise
    intermediate structure before assembling columns. columnar_v1
    parses with the stdlib `json` module (no pandas JSON parser
    involved, no type-guessing) and builds the DataFrame directly from
    a column-oriented dict, reapplying dtypes from an explicit stored
    schema.
    """

    def _large_ab_dataset(self, n_rows: int = 294_478) -> pd.DataFrame:
        """
        A synthetic dataset matching the shape of the real
        AB_Testing_Data.csv (294,478 rows) that triggered the OOM:
        string user_id/group/timestamp columns, int/float metric
        columns, and — critically — a leading-zero string ID column,
        since that's the exact case that silently corrupted under the
        earlier-rejected orient="split" approach.
        """
        return pd.DataFrame(
            {
                "user_id": [f"U{i}" for i in range(n_rows)],
                "leading_zero_id": [f"{i:07d}" for i in range(n_rows)],
                "group": (["control", "treatment"] * (n_rows // 2 + 1))[:n_rows],
                "converted": [i % 2 for i in range(n_rows)],
                "session_duration": [round((i % 97) * 0.37, 2) for i in range(n_rows)],
                "purchase_amount": [0.0 if i % 3 else round(i * 0.01, 2) for i in range(n_rows)],
                "note": [None if i % 11 == 0 else f"n{i}" for i in range(n_rows)],
            }
        )

    def test_large_dataset_reconstruction_matches_original_exactly(self):
        """
        The core regression test requested: uses a 294,478-row dataset
        (matching the real production dataset's scale) and verifies
        columns, row count, values, dtypes, leading zeros, missing
        values, and column order all survive the round-trip.
        """
        df = self._large_ab_dataset()

        serialized = _serialize_dataframe(df)
        parsed_preview = json.loads(serialized)
        assert parsed_preview["format"] == "columnar_v1"

        reconstructed = _deserialize_dataframe(serialized)

        # column order preserved
        assert list(reconstructed.columns) == list(df.columns)
        # row count preserved
        assert len(reconstructed) == len(df)
        # dtypes preserved
        assert (df.dtypes.astype(str) == reconstructed.dtypes.astype(str)).all()
        # leading zeros preserved (the exact case that broke orient="split")
        assert reconstructed["leading_zero_id"].tolist() == df["leading_zero_id"].tolist()
        assert reconstructed["leading_zero_id"].iloc[0] == "0000000"
        assert reconstructed["leading_zero_id"].iloc[1] == "0000001"
        # missing values preserved (position and non-null values)
        assert df["note"].isna().tolist() == reconstructed["note"].isna().tolist()
        # equivalent values across every column
        for col in df.columns:
            original_values = df[col].where(df[col].notna(), None).tolist()
            reconstructed_values = reconstructed[col].where(reconstructed[col].notna(), None).tolist()
            assert original_values == reconstructed_values, f"value mismatch in column {col!r}"

    def test_large_dataset_does_not_use_pandas_read_json(self, monkeypatch):
        """
        Requirement 4: columnar_v1 reconstruction must NOT call
        pandas' `pd.read_json` at all — that parser is exactly what
        produced the ~1.17GB peak this format exists to avoid.
        """
        df = self._large_ab_dataset(n_rows=50_000)  # smaller for test speed; format choice doesn't depend on size
        serialized = _serialize_dataframe(df)

        original_read_json = dataset_store.pd.read_json
        calls = {"n": 0}

        def counting_read_json(*args, **kwargs):
            calls["n"] += 1
            return original_read_json(*args, **kwargs)

        monkeypatch.setattr(dataset_store.pd, "read_json", counting_read_json)
        _deserialize_dataframe(serialized)
        assert calls["n"] == 0, "columnar_v1 path must not call pd.read_json"

    def test_end_to_end_store_and_get_dataset_on_large_dataset(self):
        """Full store_dataset() -> get_dataset() round-trip (through
        compression too) on the large dataset, not just the
        serialize/deserialize helpers directly."""
        df = self._large_ab_dataset(n_rows=50_000)  # smaller for test speed
        dataset_id = store_dataset(df)
        fetched = get_dataset(dataset_id)

        assert list(fetched.columns) == list(df.columns)
        assert len(fetched) == len(df)
        assert fetched["leading_zero_id"].tolist() == df["leading_zero_id"].tolist()
        assert fetched["converted"].tolist() == df["converted"].tolist()

    def test_object_dtype_column_round_trips(self):
        df = pd.DataFrame({"col": ["abc", "def", "000123", ""]})
        reconstructed = _deserialize_dataframe(_serialize_dataframe(df))
        assert reconstructed["col"].tolist() == df["col"].tolist()
        assert str(reconstructed["col"].dtype) == str(df["col"].dtype)

    def test_int64_dtype_column_round_trips(self):
        df = pd.DataFrame({"col": [1, 2, 3, -5, 0]})
        reconstructed = _deserialize_dataframe(_serialize_dataframe(df))
        assert reconstructed["col"].tolist() == df["col"].tolist()
        assert str(reconstructed["col"].dtype) == "int64"

    def test_float64_dtype_column_round_trips(self):
        df = pd.DataFrame({"col": [1.5, 2.25, -3.75, 0.0]})
        reconstructed = _deserialize_dataframe(_serialize_dataframe(df))
        assert reconstructed["col"].tolist() == df["col"].tolist()
        assert str(reconstructed["col"].dtype) == "float64"

    def test_bool_dtype_column_round_trips(self):
        df = pd.DataFrame({"col": [True, False, True, True]})
        reconstructed = _deserialize_dataframe(_serialize_dataframe(df))
        assert reconstructed["col"].tolist() == df["col"].tolist()
        assert str(reconstructed["col"].dtype) == "bool"

    def test_nan_and_none_preserved(self):
        df = pd.DataFrame(
            {
                "float_col": [1.0, float("nan"), 3.0],
                "obj_col": ["a", None, "c"],
            }
        )
        reconstructed = _deserialize_dataframe(_serialize_dataframe(df))
        assert df["float_col"].isna().tolist() == reconstructed["float_col"].isna().tolist()
        assert df["obj_col"].isna().tolist() == reconstructed["obj_col"].isna().tolist()
        assert reconstructed["obj_col"].tolist() == df["obj_col"].tolist()

    def test_leading_zero_strings_preserved(self):
        """The exact case that silently corrupted under orient="split"
        (leading-zero digit-looking strings auto-converted to int)."""
        df = pd.DataFrame({"id": ["000001", "001234", "0", "0099"]})
        reconstructed = _deserialize_dataframe(_serialize_dataframe(df))
        assert reconstructed["id"].tolist() == ["000001", "001234", "0", "0099"]
        # dtype string varies by pandas version ("object" vs pandas
        # 3.0's default "str" backend) -- what actually matters here is
        # that it round-trips to the SAME dtype as the original, not a
        # numeric one, and the values above already prove no numeric
        # coercion happened.
        assert str(reconstructed["id"].dtype) == str(df["id"].dtype)

    def test_column_order_preserved(self):
        df = pd.DataFrame({"z_col": [1], "a_col": [2], "m_col": [3]})
        reconstructed = _deserialize_dataframe(_serialize_dataframe(df))
        assert list(reconstructed.columns) == ["z_col", "a_col", "m_col"]

    def test_backward_compatible_with_pre_columnar_v1_record(self):
        """Requirement: existing records using the older (pre-columnar_v1)
        compressed orient="table" JSON must remain readable — no
        format marker, dispatched to the pd.read_json(orient="table")
        fallback path automatically."""
        df = _sample_df()
        old_format_json = df.to_json(orient="table")  # what store_dataset() wrote before columnar_v1
        reconstructed = _deserialize_dataframe(old_format_json)
        assert reconstructed["user_id"].tolist() == df["user_id"].tolist()
        assert reconstructed["group"].tolist() == df["group"].tolist()

    def test_backward_compatible_with_pre_columnar_v1_compressed_record(self):
        """Same as above, but also wrapped in the existing gzip+b64:
        compression layer — the realistic shape of a record written by
        the immediately-prior version of store_dataset() (compression
        added, columnar_v1 not yet added)."""
        df = _sample_df()
        old_format_json = df.to_json(orient="table")
        compressed_old_format = _encode_payload(old_format_json)

        dataset_id = str(uuid.uuid4())
        session_factory = _get_session_factory()
        with session_factory() as session:
            session.add(
                DatasetModel(
                    dataset_id=dataset_id,
                    created_at=datetime.now(timezone.utc),
                    data_json=compressed_old_format,
                )
            )
            session.commit()

        fetched = get_dataset(dataset_id)
        assert fetched["user_id"].tolist() == df["user_id"].tolist()

    def test_new_records_use_columnar_v1_format(self, stored_dataset_id):
        """store_dataset() must write the NEW format going forward,
        not silently keep writing the old one."""
        session_factory = _get_session_factory()
        with session_factory() as session:
            row = session.get(DatasetModel, stored_dataset_id)
        decoded = _decode_payload(row.data_json)
        parsed = json.loads(decoded)
        assert parsed["format"] == "columnar_v1"
