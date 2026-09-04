"""
Unified LLM client.

This is the ONLY place in the codebase that constructs an LLM client.
`LLMReportGenerator`, `LLMChatResponder`, and `LLMPlanner` (see
graph/report_generator.py, graph/chat_generator.py,
graph/planner_strategy.py) call `get_llm()` — nothing else needs to
know how the client is built or which provider is behind it.

Provider selection is driven entirely by config (`AppSettings.llm_provider`
/ `llm_model`, read from `.env` — see `.env.example`), never hardcoded
here. Today only "openrouter" is implemented (OpenAI-compatible
endpoint via `langchain-openai`'s `ChatOpenAI`, pointed at OpenRouter's
base URL) — adding a second provider means one more branch in
`get_llm()`, not a rewrite of any calling code.

`get_llm(model=...)` additionally accepts a per-request model
override, validated against `AppSettings.available_llm_models`
(see `resolve_model()`) — this is what lets a single deployment
switch between the configured default and a curated set of free
OpenRouter models (see app/api/routes_system.py's GET /system/models)
without an env/redeploy, e.g. when the default model is rate-limited.
"""

from __future__ import annotations

from app.core.config import app_settings


class LLMNotConfiguredError(RuntimeError):
    """Raised when an LLM-backed strategy is selected but no API key is configured."""


def _available_model_ids() -> set[str]:
    return {m["id"] for m in app_settings.available_llm_models}


def resolve_model(requested_model: str | None) -> str:
    """
    Validates a caller-supplied model override against the curated
    allowlist (`AppSettings.available_llm_models` — see
    app/api/routes_system.py's GET /system/models, the only place this
    list is exposed to the frontend). An unknown/empty value falls
    back to `AppSettings.llm_model` rather than being sent to
    OpenRouter unchecked — this keeps the request-scoped model
    selector (Settings.model on the frontend) additive and safe: it
    can only ever pick one of the models the backend explicitly
    curated, never an arbitrary string.
    """
    if requested_model and requested_model in _available_model_ids():
        return requested_model
    return app_settings.llm_model


# Manual cache (not functools.lru_cache) keyed by the resolved model
# string, since a single process may now serve requests for several
# different models (Settings.model, request-scoped) rather than always
# the one globally-configured `llm_model`.
_client_cache: dict[str, object] = {}


def get_llm(model: str | None = None):
    """
    Returns a LangChain chat model client for the configured provider.
    Cached per resolved model — the graph can call this on every node
    invocation without re-constructing the client each time.

    `model` is an OPTIONAL per-request override (see `resolve_model()`
    above for validation) — omitting it preserves the exact previous
    behavior of always using `AppSettings.llm_model`.

    Raises `LLMNotConfiguredError` if the selected provider has no API
    key set, rather than failing with an opaque error deep inside a
    LangChain call.
    """
    resolved = resolve_model(model)
    if resolved in _client_cache:
        return _client_cache[resolved]

    if app_settings.llm_provider == "openrouter":
        client = _build_openrouter_client(resolved)
    else:
        raise NotImplementedError(f"LLM_PROVIDER={app_settings.llm_provider!r} is not implemented yet.")

    _client_cache[resolved] = client
    return client


def _build_openrouter_client(model: str):
    if not app_settings.openrouter_api_key:
        raise LLMNotConfiguredError(
            "OPENROUTER_API_KEY is not set — cannot construct an OpenRouter LLM client. "
            "Set it in .env, or keep PLANNER_BACKEND=keyword / REPORT_BACKEND=template "
            "to run without an LLM."
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=app_settings.openrouter_api_key,
        base_url=app_settings.openrouter_base_url,
        # BUG FIX (402 "requires more credits, or fewer max_tokens"):
        # `AppSettings.llm_max_tokens` existed as a config field but was
        # never actually passed here, so every request left LangChain's/
        # OpenRouter's `max_tokens` unset. Several OpenRouter models then
        # default the request's max_tokens to the MODEL's own context
        # limit (e.g. 65536) rather than anything workload-appropriate —
        # which routinely exceeds what a metered/low-balance API key can
        # afford, even though the response itself (a small structured
        # JSON report) never needed anywhere near that many tokens. This
        # surfaced as a 402 from OpenRouter ("requested up to 65536
        # tokens, but can only afford ~10245") on otherwise-healthy
        # requests, indistinguishable at a glance from actually being
        # out of credits. Explicitly capping the request at the
        # configured budget fixes that class of failure without
        # touching how much the account is topped up.
        max_tokens=app_settings.llm_max_tokens,
        # PRODUCTION AVAILABILITY FIX: without an explicit timeout, this
        # client inherits the OpenAI SDK's default (several minutes),
        # and free/rate-limited OpenRouter models (see
        # AppSettings.available_llm_models — all free-tier) are exactly
        # the kind of backend that hangs or queues rather than failing
        # fast. LLMReportGenerator/LLMChatResponder/LLMPlanner already
        # catch any exception from this client and fall back to the
        # deterministic template/keyword path (see report_generator.py,
        # chat_generator.py, planner_strategy.py) — but that fallback
        # only gets a chance to run if THIS call actually raises within
        # the request's time budget. Render (see routes_experiments.py's
        # dataset-caching fix, added for the same reason) will otherwise
        # kill the whole HTTP request first, which looks externally like
        # "the model is unavailable" even though the in-process fallback
        # logic is sound. `max_retries=0` avoids the SDK's own retry
        # loop silently multiplying this timeout before our fallback can
        # run — LLMReportGenerator/LLMChatResponder/LLMPlanner are the
        # retry/fallback layer, not the SDK.
        request_timeout=app_settings.llm_request_timeout_seconds,
        max_retries=0,
    )


def invoke_structured(llm, messages: list[dict], schema):
    """
    Shared, resilient replacement for
    ``llm.with_structured_output(schema).invoke(messages)`` — used by
    every LLM call site (LLMReportGenerator, LLMChatResponder,
    LLMPlanner) instead of calling `with_structured_output` directly.

    BUG this fixes: `with_structured_output()` is not equally reliable
    across every OpenRouter model this project curates
    (`AppSettings.available_llm_models`, all free-tier). Some don't
    support native tool/function calling, so langchain-openai silently
    degrades to a prompted JSON mode — and a model at that tier will
    sometimes wrap its JSON in a ```json fence, prepend commentary, or
    otherwise return text that fails Pydantic validation on the first
    attempt. That failure used to propagate straight up and discard
    the entire LLM response, silently downgrading a perfectly
    recoverable "valid JSON wrapped in prose/markdown" reply to the
    deterministic fallback — even though the model actually did the
    work.

    Two-tier recovery, cheapest first, no extra network round trip:
      1. Ask langchain for the raw (unparsed) message alongside the
         parse attempt (`include_raw=True`), so a failed parse still
         gives us the actual text instead of only an exception.
      2. If the first parse failed, strip a leading/trailing ```json
         fence and take the outermost {...} span, then validate that
         substring directly against the schema.

    If both fail, this raises `ValueError` — callers keep their
    existing try/except around this call, so the deterministic
    fallback path is unchanged; this only shrinks how often it's
    needed for what is otherwise a recoverable formatting issue.
    """
    return _invoke_structured_full(llm, messages, schema)[0]


def invoke_structured_with_usage(llm, messages: list[dict], schema):
    """
    Same call/recovery behavior as `invoke_structured` above (delegates
    to the identical shared logic — nothing about parsing/fallback
    differs), but additionally returns token/cost accounting for the
    call as an `LLMUsage | None` (see LLMUsage's docstring in
    app/schemas/execution.py). Added for LLMReportGenerator, which is
    the one call in the pipeline worth metering — `invoke_structured`
    itself is left untouched so LLMChatResponder/LLMPlanner (which
    don't need usage) keep their existing (parsed-object-only) return
    shape exactly as before.
    """
    parsed, raw_message = _invoke_structured_full(llm, messages, schema)
    return parsed, _extract_llm_usage(raw_message)


def _invoke_structured_full(llm, messages: list[dict], schema):
    """
    Actual implementation shared by `invoke_structured` (parsed only)
    and `invoke_structured_with_usage` (parsed + usage) above — see
    their docstrings. Returns `(parsed, raw_message)`; `raw_message`
    is the langchain `AIMessage` (or None, in a couple of edge/mocked
    cases) that `_extract_llm_usage` reads token/cost data from.
    """
    import json
    import re

    structured = llm.with_structured_output(schema, include_raw=True)
    result = structured.invoke(messages)
    raw_message = result.get("raw")

    if result.get("parsed") is not None:
        return result["parsed"], raw_message

    raw_text = getattr(raw_message, "content", "") or ""
    if not isinstance(raw_text, str):
        # Some providers return content as a list of blocks.
        raw_text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_text
        )

    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        parsed_dict = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        parse_error = result.get("parsing_error")
        raise ValueError(
            f"LLM structured output was not valid JSON after stripping markdown "
            f"(original parse error: {parse_error}): {exc}"
        ) from exc

    return schema.model_validate(parsed_dict), raw_message


def _extract_llm_usage(raw_message):
    """
    Pulls token/cost accounting off a langchain `AIMessage`, if
    present. Returns None (not a zeroed-out LLMUsage) whenever the
    message carries no usage data at all — e.g. `raw_message` is None
    (some mocked/test call sites), or the provider's response simply
    omitted `usage`.

    - `prompt_tokens`/`completion_tokens`/`total_tokens` come from
      langchain-openai's standardized `usage_metadata` (built from the
      OpenAI-standard `usage` object every provider returns).
    - `cost_usd` comes from `response_metadata["token_usage"]["cost"]`
      — langchain-openai stores the raw provider `usage` dict verbatim
      there, and OpenRouter adds a non-standard `cost` key to that
      same object, so this passes through untouched rather than being
      reconstructed from a per-model price table.
    """
    from app.schemas.execution import LLMUsage

    if raw_message is None:
        return None

    usage_metadata = getattr(raw_message, "usage_metadata", None) or {}
    response_metadata = getattr(raw_message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or {}
    cost = token_usage.get("cost")

    prompt_tokens = usage_metadata.get("input_tokens")
    completion_tokens = usage_metadata.get("output_tokens")
    total_tokens = usage_metadata.get("total_tokens")

    if prompt_tokens is None and completion_tokens is None and total_tokens is None and cost is None:
        return None

    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
    )
