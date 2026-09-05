"""
Lazy module proxy — defers importing a heavy dependency until the
first attribute access on it, instead of at module import time.

WHY THIS EXISTS: on a memory/CPU-constrained deploy target (e.g.
Render's free plan), `scipy`/`statsmodels` being imported transitively
by every route module (because `app/main.py` imports the API routers,
which import the LangGraph nodes, which import `app/stats/*`) means
the whole C-extension stack loads before uvicorn ever binds the port —
on a slow/thrashing instance that import chain alone can eat enough of
the boot window that Render's port-scan times out before the process
gets anywhere near `Uvicorn running on ...`. `scipy.stats` /
`statsmodels` are only actually *needed* once the first real
hypothesis test runs, not at import time, so there's no correctness
cost to deferring them — only a startup-latency win.

Usage — replaces `from scipy import stats as scipy_stats`:

    from app.core.lazy_import import LazyModule
    scipy_stats = LazyModule("scipy.stats")

`scipy_stats.norm.ppf(...)` etc. keep working unchanged: the first
attribute access imports the real module once and caches it, so the
cost is paid on first *use*, not on import.
"""

from __future__ import annotations

import importlib
from typing import Any


class LazyModule:
    """Proxies attribute access to a module, importing it on first use."""

    def __init__(self, import_path: str) -> None:
        self._import_path = import_path
        self._module: Any = None

    def _load(self) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._import_path)
        return self._module

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires for names not already found on this
        # instance/class, so this never intercepts _import_path/_module.
        return getattr(self._load(), name)
