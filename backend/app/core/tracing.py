"""
LangSmith tracing — Stage 8.1. Observability only, never business
logic: nothing here reads or writes GraphState, nothing here decides
which node runs. This module's only job is to set (or force-clear) the
environment variables LangChain's tracing reads before the LangGraph
graph is ever invoked.

LangGraph's compiled graphs are LangChain Runnables, so once tracing is
enabled via these env vars, every `.invoke()` call automatically
produces ONE root trace, with each node (classifier, planner,
validation, experiment, decision) automatically appearing as a child
run — no changes to graph_builder.py, the nodes, or any stats module
are needed for that to happen.

FAIL-SAFE GUARANTEE: if `LANGSMITH_API_KEY` is not set, tracing is
force-disabled regardless of what `LANGCHAIN_TRACING_V2` says in
`.env`. The application must run identically with or without a
LangSmith account — see `configure_tracing()` below.
"""

import os

from app.core.config import app_settings
from app.core.logging import get_node_logger

log = get_node_logger("Tracing")


def configure_tracing() -> bool:
    """
    Call once at process startup (see app/main.py). Returns True if
    tracing was actually enabled, False if it was disabled — callers
    can log this, but nothing downstream needs to branch on it: with
    tracing off, LangChain's runnables behave exactly as if the
    tracing callback didn't exist.
    """
    if app_settings.langchain_tracing_v2 and app_settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = app_settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = app_settings.langchain_project
        log.info("[Tracing] LangSmith tracing ENABLED — project=%s", app_settings.langchain_project)
        return True

    # Force-disable, even if LANGCHAIN_TRACING_V2=true is set in .env
    # without a key — never let a half-configured env silently attempt
    # network calls to LangSmith and risk breaking the request path.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ.pop("LANGCHAIN_API_KEY", None)
    log.info("[Tracing] LangSmith tracing disabled (no LANGSMITH_API_KEY set) — running normally.")
    return False
