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

    # RAG relevance quality gate (app/rag/retriever.py's BM25 ranking,
    # scores normalized per-query against that query's own top match —
    # higher score = more relevant, range [0, 1]).
    # SINGLE SOURCE OF TRUTH for the app's retrieval quality bar:
    # knowledge_base_node.py passes this explicitly as `min_score` to
    # retriever.retrieve() rather than relying on that method's own
    # (lower, 0.12) default, so this one field is the only place the
    # "is this reference actually relevant enough to show?" question
    # is answered — see knowledge_base_node.py's module docstring for
    # the reasoning.
    #
    # Recalibrated for BM25's score distribution (retriever.py switched
    # from TF-IDF+cosine to BM25 — see that module's docstring for why).
    # BM25's per-term scoring doesn't punish partial-vocabulary mismatch
    # nearly as harshly as cosine similarity over a full-vocabulary
    # vector does, so its normalized scores run noticeably higher for
    # a merely-adjacent chunk than TF-IDF's did — the old 0.20 bar
    # (tuned for the old algorithm) let too much of that through.
    # Chosen empirically from this KB's real BM25 score distribution:
    # for genuinely on-topic queries (e.g. "chi-square test" -> 1.00,
    # "CUPED" -> 1.00, "minimum detectable effect" -> 1.00 for the top
    # chunk) a SECOND, still genuinely related chunk in the same
    # cluster scores 0.55-0.72 (e.g. "Statistical Significance Is Not
    # Practical Significance" scores 0.72 for the MDE query — legitimately
    # relevant, not noise), while a merely tangential chunk that only
    # shares a little vocabulary (e.g. "Novelty and Primacy Effects" for
    # the same MDE query) scores 0.19-0.35 — genuine noise from partial
    # overlap, not real topical relevance. 0.55 sits in that gap: it
    # keeps real (including secondary-but-genuinely-relevant) matches
    # while excluding the merely tangential ones that would otherwise
    # pad out to `top_k`. This is intentionally NOT the same as
    # retriever.py's own 0.12 default (which stays as that module's
    # generic, test-covered default for direct/unit callers) — this
    # field is the app-level bar applied specifically at the
    # knowledge_base_node call site.
    kb_relevance_threshold: float = 0.55


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

    llm_request_timeout_seconds: float = 30.0

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

    # LangSmith tracing (Stage 8.1 — observability only, never
    # business logic). Tracing is opt-in and fails safe: see
    # core/tracing.py's configure_tracing() — if no API key is
    # present, tracing is forced off regardless of what
    # LANGCHAIN_TRACING_V2 says, so the app always runs without a
    # LangSmith account.
    langsmith_api_key: str = ""
    langchain_project: str = "experiment-review-copilot"
    langchain_tracing_v2: bool = False

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


stats_thresholds = StatsThresholds()
app_settings = AppSettings()

# BUG FIX guardrail: warn loudly (at import time, not silently deep
# inside get_llm()) if `llm_model` is ever set — by code default or by
# an operator's .env override — to something outside the curated
# `available_llm_models` allowlist. This is exactly the drift that
# made GET /system/models' "Backend default" point at a model the
# dropdown never actually offered or validated (resolve_model() would
# silently accept it as the fallback target without complaint). Not a
# hard failure, since an operator MAY intentionally run a paid/custom
# default outside the free-tier dropdown — but it must never happen by
# silent accident, so this is logged unmissably at startup.
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
