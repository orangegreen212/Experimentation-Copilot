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

from app.core.dataset_store import store_dataset
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
        df = pd.read_csv(path)
        # Demo datasets are loaded from a local path, not an upload — there
        # is no raw upload byte stream to persist, and this is never an
        # Excel file. Both must be defined before the shared code below
        # (`raw_bytes if not is_excel else None`) runs, or use_demo=True
        # crashes /datasets/classify outright with an UnboundLocalError.
        raw_bytes = None
        is_excel = False
    elif dataset_key:
        if dataset_key not in _REAL_DATASETS:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown dataset_key {dataset_key!r}. Available: {sorted(_REAL_DATASETS)}",
            )
        path, file_name = _REAL_DATASETS[dataset_key]
        df = pd.read_csv(path)
        # Same reasoning as the use_demo branch above: loaded from a local
        # bundled file, not an upload, so there's no raw byte stream and
        # it's never Excel.
        raw_bytes = None
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
