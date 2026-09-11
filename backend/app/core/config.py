"""
Application configuration.

DECISION: statistical thresholds (SRM alpha, outlier sigma, null %,
normality alpha) live here as fixed constants, NOT as user-configurable
settings exposed through the API/UI. `Settings` (schemas/settings.py)
only exposes `cuped` / `bootstrap` / `model` — methodological toggles
the user is meant to control. Thresholds below are analysis-integrity
constants: changing them changes what "significant"/"passed" MEANS,
which should require a code change + review, not a UI toggle.

Values match the numbers implied by mock-data.ts's reports (e.g. 4σ
outlier detection, 1% null threshold, α=0.05 for SRM and normality).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class StatsThresholds(BaseSettings):
    """Fixed statistical thresholds used throughout app/stats/."""

    model_config = SettingsConfigDict(env_prefix="STATS_")

    # SRM (Sample Ratio Mismatch) — chi-square goodness-of-fit test.
    # p_value < srm_alpha => SRM detected (randomization likely broken).
    srm_alpha: float = 0.05

    # Outlier detection — values beyond this many standard deviations
    # from the arm's mean are flagged.
    outlier_sigma: float = 4.0

    # Null/missing values — fraction of a column that may be missing
    # before the "Null / Missing Values" quality check fails.
    null_threshold_pct: float = 0.01  # 1%

    # Normality (Shapiro-Wilk) — p >= this => normality assumption holds.
    normality_alpha: float = 0.05

    # Hypothesis testing — significance level for the primary test.
    significance_alpha: float = 0.05

    # Equal-variance check (Levene's test) — p >= this => variances
    # considered equal (Student's t-test eligible instead of Welch's).
    equal_variance_alpha: float = 0.05

    # Power analysis defaults.
    target_power: float = 0.80

    # RAG relevance quality gate (app/rag/retriever.py's TF-IDF cosine
    # similarity — higher score = more relevant, range [0, 1]).
    # SINGLE SOURCE OF TRUTH for the app's retrieval quality bar:
    # knowledge_base_node.py passes this explicitly as `min_score` to
    # retriever.retrieve() rather than relying on that method's own
    # (lower, 0.12) default, so this one field is the only place the
    # "is this reference actually relevant enough to show?" question
    # is answered — see knowledge_base_node.py's module docstring for
    # the reasoning.
    #
    # Chosen empirically from this KB's real score distribution: for
    # genuinely on-topic single-concept queries (e.g. "chi-square
    # test" -> 0.449, "CUPED" -> 0.405, "minimum detectable effect" ->
    # 0.275) the correct top chunk scores well above 0.20, while
    # adjacent-but-not-actually-applicable chunks within the same doc
    # cluster around 0.13-0.19 — genuine noise from partial vocabulary
    # overlap, not real topical relevance. 0.20 sits in that gap: it
    # keeps real matches (including the enriched Full-Review query's
    # top hits, which cluster at 0.22-0.30) while excluding the merely
    # tangential ones that used to pad out to `top_k` before this gate
    # existed. This is intentionally NOT the same as retriever.py's
    # own 0.12 default (which stays as that module's generic,
    # test-covered default for direct/unit callers) — this field is
    # the app-level bar applied specifically at the knowledge_base_node
    # call site.
    kb_relevance_threshold: float = 0.20


class AppSettings(BaseSettings):
    """General app/env settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"

    cors_allowed_origins: str = (
        "http://localhost:3000,https://ai-decision-support-system.vercel.app"
    )

    cors_allowed_origin_regex: str = ""

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    llm_provider: str = "openrouter"
    llm_model: str = "z-ai/glm-5.3-flash"

    llm_request_timeout_seconds: float = 30.0
    llm_max_tokens: int = 4096

    # Curated as of 2026-08-29. Paid models come FIRST on purpose: the
    # free-tier OpenRouter models are rate-limited/queued and sometimes
    # simply don't respond, so both the backend default (llm_model,
    # below) and the frontend dropdown's top entries are cheap PAID
    # models that actually load reliably. Free models are still listed
    # further down for anyone who explicitly wants $0 usage.
    available_llm_models: list[dict[str, str]] = [
        # --- Paid, cheap, reliable (checked first) ---
        {
            "id": "z-ai/glm-5.3-flash",
            "label": "GLM-5.3 Flash (paid, cheap)",
        },
        {
            "id": "deepseek/deepseek-v4-flash",
            "label": "DeepSeek V4 Flash (paid, cheap)",
        },
        {
            "id": "qwen/qwen3-30b-a3b-instruct-2507",
            "label": "Qwen3 30B A3B (paid, cheap)",
        },
        # --- Free tier (may queue or fail to respond under load) ---
        {
            "id": "minimax/minimax-m3:free",
            "label": "MiniMax M3 (free)",
        },
        {
            "id": "z-ai/glm-5.2:free",
            "label": "GLM-5.2 (free)",
        },
        {
            "id": "nvidia/nemotron-3-super-120b-a12b:free",
            "label": "Nemotron 3 Super (free)",
        },
    ]

    # LangSmith tracing (Stage 8.1 — see core/tracing.py). Tracing is
    # OFF unless a key is actually present, regardless of what
    # langchain_tracing_v2 says — see configure_tracing()'s docstring
    # for why the key's presence, not this flag alone, is authoritative.
    langsmith_api_key: str = ""
    langchain_project: str = "experiment-review-copilot"
    langchain_tracing_v2: bool = False

    # Which ReportGenerator implementation the Decision node uses
    # (see graph/report_generator.py). "template" needs no API key and
    # is what the graph runs with today; "openrouter" is Stage 8 — the
    # graph itself does not change when this flips, only which
    # ReportGenerator get_report_generator() constructs.
    report_backend: str = "template"

    # Which Planner implementation the Planner node uses (see
    # graph/planner_strategy.py). "keyword" is deterministic, no LLM;
    # "llm" is Stage 8 — same one-line-swap pattern as report_backend.
    planner_backend: str = "keyword"

    # Persistence (Experiment History). SQLAlchemy connection string
    # for the ExperimentStore (see core/experiment_store.py).
    #
    # DEV DEFAULT: a local SQLite file under backend/data/. Fine for a
    # single long-lived local process.
    #
    # PRODUCTION (Vercel/serverless): Vercel's filesystem is read-only
    # outside /tmp and ephemeral between invocations, so SQLite is NOT
    # a valid production store there — set DATABASE_URL to a hosted
    # Postgres-compatible database (Neon/Supabase/Vercel Postgres),
    # e.g. postgresql+psycopg://user:pass@host/db. get_experiment_store()
    # refuses to silently fall back to SQLite when environment ==
    # "production" and DATABASE_URL still points at sqlite — see that
    # function's docstring.
    database_url: str = "sqlite:///./data/experiments.db"

    # Background orphaned-dataset sweep (see main.py's startup hook and
    # core/dataset_store.delete_unused_datasets' `older_than` param).
    # identical re-uploads by content hash, and `/datasets/cleanup`
    # already existed to delete unreferenced rows — but nothing called
    # it automatically, so every upload/reclassify that never went on
    # to a saved experiment or a definition's Data Source (an abandoned
    # upload, a superseded re-selection, a dataset previewed but never
    # analyzed) stayed in the `datasets` table forever. This runs the
    # exact same "unreferenced" check periodically and automatically,
    # restricted to rows older than `dataset_retention_hours` so a
    # dataset mid-upload or awaiting its analyze call is never at risk —
    # no legitimate flow between an upload and either an analyze call or
    # a definition being saved takes anywhere close to that long.
    # Background orphaned-dataset sweep (see main.py's startup hook and
    # core/dataset_store.delete_unused_datasets' `older_than` param).
    # Storage-growth audit finding: `store_dataset()` already dedupes
    # identical re-uploads by content hash, and `/datasets/cleanup`
    # already existed to delete unreferenced rows — but nothing called
    # it automatically, so every upload/reclassify that never went on
    # to a saved experiment or a definition's Data Source (an abandoned
    # upload, a superseded re-selection, a dataset previewed but never
    # analyzed) stayed in the `datasets` table forever. This runs the
    # exact same "unreferenced" check periodically and automatically,
    # restricted to rows older than `dataset_retention_hours` so a
    # dataset mid-upload or awaiting its analyze call is never at risk —
    # no legitimate flow between an upload and either an analyze call or
    # a definition being saved takes anywhere close to that long.
    enable_background_dataset_cleanup: bool = True
    dataset_retention_hours: float = 24.0
    dataset_cleanup_interval_seconds: float = 3600.0

    # Concurrency-audit finding: nothing previously bounded how many
    # FULL LangGraph pipeline executions (`asyncio.to_thread(...)` in
    # routes_experiments.py — real, CPU-bound pandas/statsmodels work,
    # plus a synchronous outbound LLM call) could run at the same time
    # in this one process. Render's free tier is a small/shared-CPU
    # instance; a handful of concurrent /analyze requests (5 users, or
    # one user's browser retry landing alongside a fresh click) would
    # previously all be handed their own worker thread with no ceiling
    # — degrading latency for everyone rather than failing fast for
    # the excess requests. See core/concurrency_limit.py.
    max_concurrent_analyses: int = 4


stats_thresholds = StatsThresholds()
app_settings = AppSettings()

# Startup guardrail: warn loudly (at import time, not silently deep
# inside get_llm()) if `llm_model` is ever set — by code default or by
# an operator's .env override — to something outside the curated
# `available_llm_models` allowlist. Left unchecked, this drift would let
# GET /system/models' "Backend default" point at a model the dropdown
# never actually offered or validated (resolve_model() would silently
# accept it as the fallback target without complaint). Not a hard
# failure, since an operator MAY intentionally run a paid/custom default
# outside the free-tier dropdown — but it must never happen by silent
# accident, so this is logged unmissably at startup.
import logging as _logging

_curated_ids = {m["id"] for m in app_settings.available_llm_models}
if app_settings.llm_model not in _curated_ids:
    _logging.getLogger("uvicorn.error").warning(
        "AppSettings.llm_model=%r is NOT one of the curated available_llm_models ids %r. "
        "GET /system/models will report a 'Backend default' that doesn't match any "
        "selectable dropdown entry. Set LLM_MODEL to one of the curated ids unless this "
        "is an intentional custom deployment.",
        app_settings.llm_model,
        sorted(_curated_ids),
    )
