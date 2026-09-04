from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import app_settings
from sqlalchemy import create_engine, text
from app.core.tracing import configure_tracing

# Must run before any LangGraph graph is invoked (compiling the graph
# doesn't need it, but the first request does) — env vars set here are
# what LangChain's tracing reads.
configure_tracing()

from app.api.routes_datasets import router as datasets_router
from app.api.routes_experiments import router as experiments_router
from app.api.routes_system import router as system_router

app = FastAPI(
    title="Experiment Review Copilot API",
    description="AI Decision Support System for Product Experimentation",
    version="0.1.0",
)

# Next.js dev server default port; tighten this list for production.
allowed_origins = [
    origin.strip()
    for origin in app_settings.cors_allowed_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # Additionally match any Vercel preview deployment
    # for this project by regex, since each preview build gets a new
    # hashed origin that a static exact-string list (allow_origins /
    # CORS_ALLOWED_ORIGINS) can never contain without a manual update
    # on every deploy — without this, the browser withholds the
    # response from GET /system/info and /system/models for that
    # origin (backend still returns 200; the response just lacks
    # Access-Control-Allow-Origin for it). See
    # AppSettings.cors_allowed_origin_regex.
    allow_origin_regex=app_settings.cors_allowed_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)
app.include_router(experiments_router)
app.include_router(system_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

if app_settings.environment == "development":
    # Diagnostic-only, and deliberately kept out of production: even
    # with the password stripped, a connected/error response still
    # leaks the DB error type and part of the connection string to
    # any anonymous caller. Gated behind ENVIRONMENT=development so it
    # simply doesn't exist as a route once deployed.
    @app.get("/debug/db")
    def debug_db():
        try:
            engine = create_engine(app_settings.database_url)

            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))

            return {
                "status": "ok",
                "database": "connected",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": type(e).__name__,
                "message": str(e).split("password=")[0],
            }
