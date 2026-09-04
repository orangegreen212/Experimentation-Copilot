"""
Vercel serverless entrypoint.

Mounts the EXISTING FastAPI app (backend/app/main.py) under /api,
completely unchanged — this file makes zero modifications to backend
architecture, routing, or logic. It exists purely so that:

  - locally, `uvicorn app.main:app` (run from backend/) still serves
    routes at /datasets/classify, /experiments/analyze, /health — the
    same paths the app has always used.
  - in production on Vercel, the SAME app is reachable at
    /api/datasets/classify, /api/experiments/analyze, /api/health,
    because it's mounted under /api here rather than because its
    routes were rewritten.

vercel.json rewrites every /api/* request to this file; Vercel's
Python runtime then hands the ASGI `app` object below the request.

See backend/README.md's "Deployment (Vercel)" section for the full
picture, including why /api mounting (not editing router prefixes)
was chosen: it keeps local dev and production using byte-identical
backend code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi import FastAPI

from app.main import app as backend_app  # noqa: E402

app = FastAPI()
app.mount("/api", backend_app)
