"""
Covers the three security fixes made in response to the code-review
audit:

  1. /datasets/classify now rejects uploads over the size cap (413)
     instead of buffering an unbounded amount into memory.
  2. /debug/db only exists when ENVIRONMENT=development (not asserted
     against the live app_settings singleton here, since flipping that
     at import time is awkward to test in-process — see the docstring
     on the debug_db route itself for the guard).
  3. Rate limiting on /datasets/classify and /experiments/analyze
     returns 429 after the per-IP limit is exceeded within the window.
"""

import io

from fastapi.testclient import TestClient

from app.core.rate_limit import _buckets, rate_limit
from app.main import app

client = TestClient(app)


def setup_function() -> None:
    # Rate-limit state is process-global; reset between tests so one
    # test's requests don't count against another's budget.
    _buckets.clear()


def test_oversized_upload_is_rejected_with_413() -> None:
    oversized = io.BytesIO(b"a" * (41 * 1024 * 1024))  # over the 40 MB cap
    response = client.post(
        "/datasets/classify",
        files={"file": ("huge.csv", oversized, "text/csv")},
    )
    assert response.status_code == 413


def test_upload_within_limit_is_not_rejected_for_size() -> None:
    small = io.BytesIO(b"user_id,value\n1,10\n2,20\n")
    response = client.post(
        "/datasets/classify",
        files={"file": ("small.csv", small, "text/csv")},
    )
    # Not necessarily 200 (depends on classifier rules for a 2-row
    # dataset), but it must not be rejected for size.
    assert response.status_code != 413


def test_rate_limit_dependency_blocks_after_max_requests() -> None:
    dependency = rate_limit("test-bucket", max_requests=3)

    class _FakeClient:
        host = "1.2.3.4"

    class _FakeRequest:
        headers: dict = {}
        client = _FakeClient()

    req = _FakeRequest()
    for _ in range(3):
        dependency(req)  # should not raise

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        dependency(req)
    assert exc_info.value.status_code == 429


def test_rate_limit_tracks_clients_independently() -> None:
    dependency = rate_limit("test-bucket-2", max_requests=1)

    class _FakeRequest:
        def __init__(self, ip: str) -> None:
            self.headers: dict = {}

            class _C:
                host = ip

            self.client = _C()

    dependency(_FakeRequest("1.1.1.1"))  # uses up 1.1.1.1's budget
    dependency(_FakeRequest("2.2.2.2"))  # different IP, own budget — should not raise
