"""
Shared pytest fixtures — test database isolation.

BEFORE this file existed, the test suite had no database override at
all: `ExperimentStore`/`DatasetStore` lazily default to
`app_settings.database_url` (`sqlite:///./data/experiments.db` — see
core/config.py), which is the REAL local development database. Every
test run therefore read and wrote the same on-disk file as `uvicorn
app.main:app`, with no reset between tests. This caused two distinct
problems, both fixed by the fixtures below:

  1. Tests polluted (and were polluted by) the developer's real local
     data — experiment history and uploaded datasets accumulated by
     running the app locally leaked into test assertions, and test
     runs left their own rows behind afterward.
  2. Tests were order-dependent: a test's pass/fail outcome depended
     on what earlier tests (in the same file OR an earlier file in
     the same pytest session) had already written to the shared
     store, since `ExperimentStore`/`DatasetStore` cache a single
     module-level engine/session-factory singleton for the lifetime
     of the process (see `core/experiment_store.py::get_experiment_store`
     and `core/dataset_store.py::_get_session_factory`).

Fix: a fresh SQLite file in a pytest `tmp_path`, one per test, with
the module-level singletons reset before and after so the next
`get_dataset()`/`get_experiment_store()` call rebuilds them against
the new file. This deliberately reuses the exact same lazy-singleton
pattern the app already relies on in production (see both modules'
docstrings for why it's lazy, not eager) — nothing about how the
store is built or used changes, only which file it points at during
tests. The real `backend/data/experiments.db` is never opened by any
test.

SECOND, SEPARATE SOURCE OF CROSS-TEST STATE (found while verifying
the fix above): `core/rate_limit.py`'s `_buckets` dict is also
process-global and is NEVER reset — the DB fix alone did not resolve
the order-dependent failures. FastAPI's `TestClient` sends every
request from the same synthetic client IP, so every test in a given
pytest process that hits a rate-limited route (`/experiments/analyze`
at 10 requests/60s, `/datasets/classify` at 20/60s, the chat-stream
routes at 20/60s — see routes_experiments.py / routes_datasets.py)
shares ONE counter with every other such test in the same process.
Once ~10 tests in a row have called `/experiments/analyze`, every
subsequent one starts receiving 429s instead of the response it
asserts on — exactly the `429 != 422` mismatch this file's fixture
reproduces and fixes. This is cleared the same way as the DB
singletons: directly resetting the module's global dict, no new
dependency needed.

No new test infrastructure/dependencies were introduced — this uses
only pytest's built-in `tmp_path` fixture and the two modules'
existing global/singleton state.
"""

from __future__ import annotations

import pytest

import app.core.dataset_store as dataset_store_module
import app.core.experiment_store as experiment_store_module
import app.core.rate_limit as rate_limit_module
from app.core.config import app_settings


@pytest.fixture(autouse=True)
def isolated_test_database(tmp_path, monkeypatch):
    """
    Point every test at a fresh, empty SQLite file for the duration of
    that single test, then restore the previous state afterward.

    Autouse: applied to every test in the suite without each test file
    needing to opt in, since the pollution/ordering problem this fixes
    was suite-wide, not confined to a handful of tests. A test that
    doesn't touch the store at all pays only the cost of a fixture
    teardown resetting two `None`s — negligible.
    """
    db_path = tmp_path / "test_experiments.db"
    test_database_url = f"sqlite:///{db_path}"

    # `app_settings` is a module-level singleton already constructed at
    # import time (see core/config.py) from real environment variables,
    # so re-setting an env var here would be too late to affect it —
    # the attribute itself must be overridden directly. `monkeypatch`
    # restores the original value automatically at teardown.
    monkeypatch.setattr(app_settings, "database_url", test_database_url)

    # Force both stores to rebuild their engine/session-factory against
    # the new URL on next use, rather than keep serving the previous
    # test's (or the real dev DB's) cached singleton. This mirrors the
    # exact reset each module already performs internally on first use
    # — see `get_experiment_store()` and `_get_session_factory()`.
    monkeypatch.setattr(experiment_store_module, "_store", None)
    monkeypatch.setattr(dataset_store_module, "_engine", None)
    monkeypatch.setattr(dataset_store_module, "_Session", None)

    # Every test gets its own empty rate-limit bucket dict, so a
    # `/experiments/analyze`-heavy test earlier in the run can never
    # cause a later, unrelated test's request to be throttled with a
    # 429 it isn't asserting on. `_buckets` is a plain dict, not an
    # object with its own reset method, so clearing it in place (via
    # a fresh dict of the same `defaultdict` shape) is the direct
    # equivalent of restarting the process between tests, which is
    # the assumption `rate_limit()`'s docstring already documents
    # this state resets on ("resets on deploy/restart").
    from collections import defaultdict

    monkeypatch.setattr(rate_limit_module, "_buckets", defaultdict(lambda: (0.0, 0)))

    yield

    # Reset again after the test so a subsequent test that (for
    # whatever reason) doesn't go through this fixture's setup path
    # can't accidentally inherit a closed/stale engine bound to a
    # tmp_path that pytest has already cleaned up.
    monkeypatch.setattr(experiment_store_module, "_store", None)
    monkeypatch.setattr(dataset_store_module, "_engine", None)
    monkeypatch.setattr(dataset_store_module, "_Session", None)
