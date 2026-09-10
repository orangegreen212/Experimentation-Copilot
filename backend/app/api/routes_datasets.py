"""
POST /datasets/classify

Two input modes, matching workspace-view.tsx exactly:
  1. `loadDemo()` -> use_demo=True, simulate_low_quality reflects the toggle
  2. `handleFile(file)` -> real file upload via multipart/form-data — CSV
     or single-sheet .xlsx/.xls (detected by filename extension, falling
     back to sniffing the zip signature in case the extension is missing
     or wrong, since .xlsx is a zip archive under the hood and will
     otherwise fail as a confusing UTF-8 decode error from read_csv)

Both paths converge on the same response shape: ClassifyDatasetResponse.
"""

import io
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.dataset_store import delete_unused_datasets, store_dataset
from app.core.experiment_definition_store import get_experiment_definition_store
from app.core.experiment_store import get_experiment_store
from app.core.rate_limit import rate_limit
from app.schemas.dataset import ClassifyDatasetResponse
from app.stats.dataset_classifier import DatasetClassificationError, classify_dataset

logger = logging.getLogger("api.datasets")
router = APIRouter(prefix="/datasets", tags=["datasets"])

# Hard cap on upload size. Without this, file.file.read() below buffers
# the entire multipart body into memory regardless of size — an
# unauthenticated caller could send an arbitrarily large CSV/Excel file
# and exhaust server memory (DoS). Read in bounded chunks instead of
# calling .read() directly so we bail out before ever holding a huge
# blob in memory.
_MAX_UPLOAD_BYTES = 40 * 1024 * 1024  # 40 MB
_READ_CHUNK_BYTES = 1024 * 1024


def _read_upload_with_limit(upload: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = upload.file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large. Maximum upload size is {max_bytes // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)

# Resolved relative to THIS FILE, not the process's working directory.
# A bare relative path like "data/demo/foo.csv" only works if the
# process happens to be launched with cwd=backend/ (true for local
# `uvicorn app.main:app` run from backend/, NOT guaranteed for a
# serverless function on Vercel, whose cwd is unspecified). This must
# resolve correctly regardless of where the process was started from.
_DEMO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "demo"
_DEMO_PATHS = {
    False: (_DEMO_DIR / "demo_ab_checkout.csv", "demo_ab_checkout.csv"),
    True: (_DEMO_DIR / "demo_ab_checkout_lowq.csv", "demo_ab_checkout_lowq.csv"),
}

# Real, published experiment datasets — an alternative to "Upload CSV" for
# analysts who want to run the Copilot against a genuine randomized
# experiment instead of a synthetic/demo one. Same DataFrame -> classify ->
# store pipeline as everything else; only the source of the DataFrame
# differs. Keyed by a short slug the frontend passes as `dataset_key`.
_REAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "real"
_REAL_DATASETS: dict[str, tuple[Path, str]] = {
    "hillstrom_email": (
        _REAL_DIR / "real_email_campaign.csv",
        "Hillstrom E-Mail Campaign (real A/B/n experiment, 64K customers)",
    ),
    "landing_page_ab": (
        _REAL_DIR / "real_landing_page_ab.csv",
        "Landing Page Redesign (real A/B test, 294K users)",
    ),
    "ecommerce_category": (
        _REAL_DIR / "real_ecommerce_category.csv",
        "E-Commerce Category Experiment (real A/B test, 48K users)",
    ),
}


@router.get("/real")
def list_real_datasets() -> list[dict]:
    """List real, published experiment datasets available as a data source
    alongside file upload and the synthetic demo (see _REAL_DATASETS)."""
    return [
        {"key": key, "label": label}
        for key, (_path, label) in _REAL_DATASETS.items()
    ]


@router.post(
    "/classify",
    response_model=ClassifyDatasetResponse,
    dependencies=[Depends(rate_limit("classify", max_requests=20))],
)
def classify_dataset_route(
    file: UploadFile | None = File(default=None),
    use_demo: bool = Form(default=False),
    simulate_low_quality: bool = Form(default=False),
    dataset_key: str | None = Form(default=None),
) -> ClassifyDatasetResponse:
    """
    Classify an uploaded CSV or load one of the two demo datasets.

    Exactly one of `file` or `use_demo=True` is expected to be set by
    the frontend at a time (mirrors the mutually exclusive UI actions
    "Upload CSV" vs "Load Demo A/B Dataset").
    """
    if file is not None:
        raw_bytes = _read_upload_with_limit(file, _MAX_UPLOAD_BYTES)
        file_name = file.filename or "uploaded.csv"
        is_excel = file_name.lower().endswith((".xlsx", ".xls")) or raw_bytes[:4] == b"PK\x03\x04"
        if is_excel:
            try:
                # engine="calamine" (Rust-based) instead of the default
                # openpyxl: openpyxl took 20+ seconds on a 200K-row sheet
                # in testing, which is enough to trip Render's request
                # timeout and return a 502 before the response ever comes
                # back. calamine reads the same file in ~5 seconds.
                excel = pd.ExcelFile(io.BytesIO(raw_bytes), engine="calamine")
            except Exception as exc:  # openpyxl/calamine raise various parser errors
                logger.warning("Excel parse failed for %r: %s", file_name, exc)
                raise HTTPException(
                    status_code=400,
                    detail="Could not parse this Excel file. Please check that it is a valid, uncorrupted .xlsx/.xls workbook.",
                ) from exc
            if len(excel.sheet_names) > 1:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"This Excel file has {len(excel.sheet_names)} sheets "
                        f"({', '.join(excel.sheet_names)}) — only single-sheet "
                        "workbooks are supported. Please upload the relevant sheet "
                        "as its own file."
                    ),
                )
            df = excel.parse(excel.sheet_names[0])
        else:
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes))
            except Exception as exc:  # pandas raises various parser errors
                logger.warning("CSV parse failed for %r: %s", file_name, exc)
                raise HTTPException(
                    status_code=400,
                    detail="Could not parse this file as CSV. Please check the file format and encoding.",
                ) from exc
    elif use_demo:
        path, file_name = _DEMO_PATHS[simulate_low_quality]
        raw_bytes = path.read_bytes()
        # Demo datasets are loaded from a local path, not an HTTP upload,
        # but the bytes on disk ARE the exact same CSV bytes an uploaded
        # file would have — read them directly so store_dataset() below
        # takes the fast raw-CSV compression path (`_encode_csv_payload`)
        # instead of the much slower columnar_v1 JSON serialize path (see
        # dataset_store.py's module docstring: ~294K rows blew up to
        # ~1.17 GB peak memory even with the columnar format, before
        # compression; the raw-CSV path skips that serialize step
        # entirely). Previously this was hardcoded to `raw_bytes = None`,
        # which meant demo/real bundled datasets ALWAYS paid the slow
        # path even for datasets nearly 300K rows — this only reads local
        # bytes, never an Excel file, so `is_excel` stays False.
        df = pd.read_csv(path)
        is_excel = False
    elif dataset_key:
        if dataset_key not in _REAL_DATASETS:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown dataset_key {dataset_key!r}. Available: {sorted(_REAL_DATASETS)}",
            )
        path, file_name = _REAL_DATASETS[dataset_key]
        # Same reasoning as the use_demo branch above: this is a local
        # bundled file, so its raw bytes are directly available — read
        # them so this hits the same fast raw-CSV storage path an
        # uploaded CSV gets, instead of the slow full-DataFrame JSON
        # serialize path. This matters most here: "Landing Page Redesign"
        # is ~294K rows, exactly the size the columnar_v1 docstring
        # documents as memory/time-expensive to serialize, which is the
        # dataset analysts reported repeatedly failing to select.
        raw_bytes = path.read_bytes()
        df = pd.read_csv(path)
        is_excel = False
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide a file, use_demo=True, or a dataset_key.",
        )

    try:
        dataset_info = classify_dataset(df)
    except DatasetClassificationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    dataset_id = store_dataset(df, raw_csv_bytes=raw_bytes if not is_excel else None)

    return ClassifyDatasetResponse(
        dataset=dataset_info,
        dataset_id=dataset_id,
        file_name=file_name,
    )


@router.post("/cleanup")
def cleanup_unused_datasets() -> dict:
    """Delete every stored dataset no longer referenced by anything.

    Safe by construction, never touches results: an `ExperimentReport`
    (and everything chat can answer about it, including segmentation —
    see chat_generator.py) is fully self-contained once generated, so
    this only ever removes RAW dataset rows, and only the ones nothing
    currently points at — every `experiments.dataset_id` (past and
    in-progress analyses) and every `experiment_definitions`
    `data_source.datasetId` (the dataset a definition would re-run
    against) is kept. A dataset connected to a definition's Data Source,
    or referenced by any saved experiment, is never deleted by this —
    only genuinely orphaned rows (superseded re-selections, old
    reclassifies, abandoned uploads) are.

    Call this manually for now (e.g. a button in Settings, or a cron
    hitting this endpoint) — nothing calls it automatically yet, since
    "referenced" is a snapshot at call time and doing this automatically
    right after every analysis would delete a dataset a definition is
    still configured to re-run against.
    """
    referenced_ids = {row.dataset_id for row in get_experiment_store().list()}
    referenced_ids |= {
        d.data_source.dataset_id
        for d in get_experiment_definition_store().list()
        if d.data_source is not None and d.data_source.dataset_id
    }
    deleted = delete_unused_datasets(referenced_ids=referenced_ids)
    return {"deleted_count": len(deleted), "deleted_ids": deleted}
