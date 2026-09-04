"""
Minimal per-IP rate limiting for expensive endpoints.

The API has no authentication (documented, known limitation for this
demo — see main.py / audit notes), which means the *only* signal
available to throttle abuse is the caller's IP. This is deliberately
small and dependency-free (fixed-window counter in an in-process
dict) rather than pulling in slowapi/redis: it's enough to stop a
single client from hammering `/experiments/analyze` (an LLM call,
i.e. real API spend) or `/datasets/classify` (CPU-heavy parsing), and
it doesn't need to survive process restarts or work across multiple
instances for that goal.

NOTE: this is process-local state. It resets on deploy/restart and
does not coordinate across horizontally-scaled instances. That's an
acceptable trade-off for a single-instance demo deployment (Render/
Vercel single function) — call out if this API is ever run with
multiple concurrent instances behind a shared endpoint.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request

# window in seconds, and how many requests one IP may make in that
# window before getting a 429.
_WINDOW_SECONDS = 60.0

# (client_ip, bucket_name) -> (window_start_epoch, count)
_buckets: dict[tuple[str, str], tuple[float, int]] = defaultdict(lambda: (0.0, 0))
_lock = Lock()


def _client_ip(request: Request) -> str:
    # Trust X-Forwarded-For's first hop if present (Render/Vercel sit
    # behind a proxy); fall back to the direct peer address.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(bucket_name: str, max_requests: int):
    """
    FastAPI dependency factory. Usage:

        @router.post("/analyze", dependencies=[Depends(rate_limit("analyze", 10))])

    Raises 429 once `max_requests` have been made by the same client
    IP within the current 60-second window.
    """

    def _dependency(request: Request) -> None:
        ip = _client_ip(request)
        key = (ip, bucket_name)
        now = time.monotonic()

        with _lock:
            window_start, count = _buckets[key]
            if now - window_start >= _WINDOW_SECONDS:
                # New window.
                _buckets[key] = (now, 1)
                return
            if count >= max_requests:
                retry_after = int(_WINDOW_SECONDS - (now - window_start)) + 1
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please slow down and try again shortly.",
                    headers={"Retry-After": str(retry_after)},
                )
            _buckets[key] = (window_start, count + 1)

    return _dependency
