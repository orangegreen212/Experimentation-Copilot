"""
One-off migration: re-serializes every row in the `datasets` table into
the gzip-compressed-CSV storage format (`_encode_csv_payload` /
`_CSV_COMPRESSED_PREFIX` in app/core/dataset_store.py), which
`get_dataset()` reads back via `pd.read_csv` — the cheapest of the
three formats that module supports, and the one specifically added
because of the exact memory problem documented right above
`_COLUMNAR_V1_FORMAT` in that file: reconstructing a 294,478-row
dataset peaked at ~1.17 GB via the original `orient="table"` JSON
format, ~208 MB via the newer `columnar_v1` JSON format, and
`pd.read_csv` on raw CSV bytes is cheaper again than both — comfortably
inside a 512 MB Render instance.

Any dataset row written before `columnar_v1`/CSV storage existed is
still stuck on the original expensive format forever (see
`_deserialize_dataframe`'s docstring: old rows are read via a fallback
path, never migrated automatically). This script is the migration that
was never run.

WHY THIS MATTERS FOR EXISTING EXPERIMENTS: rows are updated IN PLACE —
each `dataset_id` is preserved exactly — so every ExperimentDefinition
and Experiment that already points at a dataset_id keeps working
unchanged; only how that row is stored on disk changes.

Usage (run once against whichever DATABASE_URL you want to migrate,
typically production):

    cd backend
    DATABASE_URL=postgresql+psycopg://... python3 scripts/migrate_datasets_to_csv_format.py

Safe to re-run: rows already in CSV-compressed format are detected and
skipped, so interrupting and re-running only does the remaining work.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.dataset_store import (  # noqa: E402
    DatasetModel,
    _CSV_COMPRESSED_PREFIX,
    _decode_payload,
    _deserialize_dataframe,
    _encode_csv_payload,
    _get_session_factory,
)


def migrate() -> None:
    session_factory = _get_session_factory()
    with session_factory() as session:
        rows = session.query(DatasetModel).all()
        print(f"Found {len(rows)} dataset row(s).")

        migrated = 0
        skipped = 0
        failed = 0

        for row in rows:
            if row.data_json.startswith(_CSV_COMPRESSED_PREFIX):
                skipped += 1
                continue

            print(f"Migrating dataset_id={row.dataset_id} ...", end=" ", flush=True)
            t_start = time.perf_counter()
            try:
                # Read via whatever old format this row happens to be in
                # (compressed-JSON, plain-JSON, columnar_v1, or the very
                # original orient="table" — _decode_payload/
                # _deserialize_dataframe already handle all of them).
                json_str = _decode_payload(row.data_json)
                df = _deserialize_dataframe(json_str)

                # Re-encode as gzip-compressed raw CSV — the cheap path.
                raw_csv_bytes = df.to_csv(index=False).encode("utf-8")
                row.data_json = _encode_csv_payload(raw_csv_bytes)
                session.add(row)
                session.commit()

                migrated += 1
                print(f"OK ({len(df)} rows, {time.perf_counter() - t_start:.1f}s)")
            except Exception as exc:  # noqa: BLE001 — report and continue
                session.rollback()
                failed += 1
                print(f"FAILED: {exc!r}")

        print(
            f"\nDone. migrated={migrated} skipped(already migrated)={skipped} failed={failed}"
        )
        if failed:
            print("Re-run this script to retry the failed row(s) after investigating.")


if __name__ == "__main__":
    migrate()
