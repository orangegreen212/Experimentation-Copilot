"""
Reliability tests for app/llm/client.py, using REAL `openai` SDK
exception types (not generic `RuntimeError`s) so this actually pins
down behavior for the specific failure modes the storage/reliability
audit asked about: timeout, 429, 402, 5xx, malformed structured
output, and an empty response.

`tests/llm/test_llm_report_generator.py` /
`tests/llm/test_llm_planner.py` already prove the higher-level
generators/planner fall back to their deterministic path on ANY
exception (tested with a plain `RuntimeError`). What those tests don't
individually pin down is that the SPECIFIC exception classes OpenAI's
SDK actually raises for each of these HTTP-level failures are genuine
`Exception` subclasses that the broad `except Exception` in
LLMReportGenerator/LLMChatResponder/LLMPlanner will in fact catch —
that is verified here directly, plus the client construction/JSON-
recovery behavior that lives in app/llm/client.py itself.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai
import pytest

from app.core.config import app_settings
from app.graph.report_generator import LLMReportGenerator, TemplateReportGenerator
from app.llm.client import _build_openrouter_client, _invoke_structured_full, invoke_structured
from tests.llm.test_llm_report_generator import _high_confidence_facts


def _api_error(cls, status_code: int, message: str = "error") -> Exception:
    """Builds a real `openai.APIStatusError` subclass instance the way
    the OpenAI SDK itself does internally (from an httpx.Response),
    rather than a hand-rolled stand-in — so these tests exercise the
    actual exception shape LLMReportGenerator's `except Exception`
    has to catch in production."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": {"message": message}})
    return cls(message, response=response, body={"error": {"message": message}})


class _RaisingStructured:
    def __init__(self, exc: Exception):
        self._exc = exc

    def invoke(self, messages):
        raise self._exc


class _RaisingLLM:
    def __init__(self, exc: Exception):
        self._exc = exc

    def with_structured_output(self, schema, include_raw=True):
        return _RaisingStructured(self._exc)


class TestClientConstructionIsBounded:
    """Verifies the client is built with the bounded-retry / fail-fast
    configuration the reliability fix depends on — not just that the
    docstring claims it."""

    def test_max_retries_is_zero_not_the_sdk_default(self):
        with patch.object(app_settings, "openrouter_api_key", "test-key"):
            client = _build_openrouter_client(app_settings.llm_model)
        assert client.max_retries == 0

    def test_request_timeout_is_explicitly_set(self):
        with patch.object(app_settings, "openrouter_api_key", "test-key"):
            client = _build_openrouter_client(app_settings.llm_model)
        # ChatOpenAI stores this as request_timeout; must not be the
        # SDK's multi-minute default (None means "use the SDK
        # default", which is exactly the bug this config fixes).
        assert client.request_timeout == app_settings.llm_request_timeout_seconds
        assert client.request_timeout is not None

    def test_max_tokens_is_capped_not_left_to_the_model_default(self):
        with patch.object(app_settings, "openrouter_api_key", "test-key"):
            client = _build_openrouter_client(app_settings.llm_model)
        assert client.max_tokens == app_settings.llm_max_tokens


class TestReportGeneratorFallbackForEachRealFailureMode:
    """
    Each of these raises the ACTUAL exception type OpenRouter/OpenAI's
    SDK would raise for that HTTP status, through the exact
    `llm.with_structured_output(...).invoke(...)` call path
    `_invoke_structured_full` uses — then asserts LLMReportGenerator's
    fallback still produces the identical, deterministic template
    report (never a crash, never a hang, never a half-filled report).
    """

    @pytest.mark.parametrize(
        "make_exc",
        [
            pytest.param(lambda: openai.APITimeoutError(request=httpx.Request("POST", "https://openrouter.ai/x")), id="timeout"),
            pytest.param(lambda: _api_error(openai.RateLimitError, 429, "rate limited"), id="429"),
            pytest.param(lambda: _api_error(openai.APIStatusError, 402, "requires more credits"), id="402"),
            pytest.param(lambda: _api_error(openai.InternalServerError, 500, "internal server error"), id="5xx"),
            pytest.param(lambda: _api_error(openai.APIStatusError, 503, "model unavailable"), id="model_unavailable"),
        ],
    )
    def test_falls_back_to_deterministic_template(self, make_exc):
        facts = _high_confidence_facts()
        exc = make_exc()

        with patch("app.llm.client.get_llm", return_value=_RaisingLLM(exc)):
            llm_report = LLMReportGenerator().generate(facts)

        template_report = TemplateReportGenerator().generate(facts)
        assert llm_report.report_fallback_reason is not None
        assert llm_report.model_copy(update={"report_fallback_reason": None}) == template_report

    def test_each_failure_mode_is_a_bounded_single_attempt_not_a_retry_loop(self):
        """The report generator must call the LLM exactly once per
        analyze call, regardless of the failure — no in-process retry
        loop that could turn one flaky call into several (see
        `max_retries=0` on the client itself, verified above; this
        confirms nothing ABOVE the client re-wraps it in its own
        retry)."""
        facts = _high_confidence_facts()
        call_count = 0

        class _CountingLLM(_RaisingLLM):
            def with_structured_output(self, schema, include_raw=True):
                nonlocal call_count
                call_count += 1
                return super().with_structured_output(schema, include_raw=include_raw)

        exc = _api_error(openai.RateLimitError, 429, "rate limited")
        with patch("app.llm.client.get_llm", return_value=_CountingLLM(exc)):
            LLMReportGenerator().generate(facts)

        assert call_count == 1


class TestStructuredOutputRecovery:
    """app/llm/client.py's own JSON-recovery logic — malformed output
    and empty responses, independent of any particular generator."""

    def test_markdown_fenced_json_is_recovered(self):
        from pydantic import BaseModel

        class Schema(BaseModel):
            value: int

        raw_message = SimpleNamespace(content="```json\n{\"value\": 42}\n```", usage_metadata=None, response_metadata=None)
        structured = SimpleNamespace(
            invoke=lambda messages: {"parsed": None, "raw": raw_message, "parsing_error": "did not match schema"}
        )
        llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)

        result = invoke_structured(llm, [{"role": "user", "content": "x"}], Schema)
        assert result.value == 42

    def test_prose_wrapped_json_is_recovered(self):
        from pydantic import BaseModel

        class Schema(BaseModel):
            value: int

        raw_message = SimpleNamespace(
            content='Sure, here is the JSON you asked for: {"value": 7} — let me know if you need anything else.',
            usage_metadata=None,
            response_metadata=None,
        )
        structured = SimpleNamespace(
            invoke=lambda messages: {"parsed": None, "raw": raw_message, "parsing_error": "did not match schema"}
        )
        llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)

        result = invoke_structured(llm, [{"role": "user", "content": "x"}], Schema)
        assert result.value == 7

    def test_genuinely_malformed_output_raises_valueerror_not_silently_swallowed(self):
        """A response that is not recoverable JSON at all must raise
        (so the CALLER's fallback runs) — never return a fabricated
        default or hang."""
        from pydantic import BaseModel

        class Schema(BaseModel):
            value: int

        raw_message = SimpleNamespace(content="I cannot help with that request.", usage_metadata=None, response_metadata=None)
        structured = SimpleNamespace(
            invoke=lambda messages: {"parsed": None, "raw": raw_message, "parsing_error": "did not match schema"}
        )
        llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)

        with pytest.raises(ValueError):
            invoke_structured(llm, [{"role": "user", "content": "x"}], Schema)

    def test_empty_response_raises_cleanly_not_a_crash(self):
        from pydantic import BaseModel

        class Schema(BaseModel):
            value: int

        raw_message = SimpleNamespace(content="", usage_metadata=None, response_metadata=None)
        structured = SimpleNamespace(
            invoke=lambda messages: {"parsed": None, "raw": raw_message, "parsing_error": "empty response"}
        )
        llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)

        with pytest.raises(ValueError):
            invoke_structured(llm, [{"role": "user", "content": "x"}], Schema)

    def test_empty_response_at_report_generator_level_falls_back_to_template(self):
        """The same empty-response case, through the actual
        LLMReportGenerator -> must degrade to the deterministic
        template report, not propagate a raw ValueError to the user."""
        facts = _high_confidence_facts()
        raw_message = SimpleNamespace(content="", usage_metadata=None, response_metadata=None)
        structured = SimpleNamespace(
            invoke=lambda messages: {"parsed": None, "raw": raw_message, "parsing_error": "empty response"}
        )
        llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=True: structured)

        with patch("app.llm.client.get_llm", return_value=llm):
            llm_report = LLMReportGenerator().generate(facts)

        template_report = TemplateReportGenerator().generate(facts)
        assert llm_report.report_fallback_reason is not None
        assert llm_report.model_copy(update={"report_fallback_reason": None}) == template_report


class TestChatMessageLengthIsBounded:
    """Same audit finding as prompt, traced to a second real sink:
    FollowUpChatRequest.message is embedded verbatim into the LLM
    conversation on every chat call (chat_generator.py's
    `_build_chat_conversation`) and had no length limit either — and
    unlike most other free-text fields in this app (objective,
    hypothesis statement, etc.), the frontend's own chat Textarea
    (components/follow-up-chat.tsx) has no maxLength to compensate."""

    def test_oversized_message_is_rejected_at_the_schema_boundary(self):
        from pydantic import ValidationError

        from app.schemas.chat import FollowUpChatRequest

        with pytest.raises(ValidationError):
            FollowUpChatRequest(experiment_id="e1", message="x" * 4001)

    def test_message_at_the_limit_is_accepted(self):
        from app.schemas.chat import FollowUpChatRequest

        req = FollowUpChatRequest(experiment_id="e1", message="x" * 4000)
        assert len(req.message) == 4000


class TestPromptSizeIsBounded:
    """
    Audit finding: `AnalyzeExperimentRequest.prompt` previously had no
    length limit and is embedded verbatim into the LLM report prompt
    on every analyze call — an oversized value would either bloat the
    outbound LLM payload for no benefit or exceed a free-tier model's
    context window. Fixed with a `Field(max_length=4000)` constraint;
    this test pins down the boundary at the schema level (independent
    of any real HTTP call).
    """

    def test_oversized_prompt_is_rejected_at_the_schema_boundary(self):
        from app.api.routes_experiments import AnalyzeExperimentRequest
        from app.schemas.settings import AnalysisSettings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AnalyzeExperimentRequest(
                dataset_id="d1",
                prompt="x" * 4001,
                settings=AnalysisSettings(cuped=False, bootstrap=False),
            )

    def test_prompt_at_the_limit_is_accepted(self):
        from app.api.routes_experiments import AnalyzeExperimentRequest
        from app.schemas.settings import AnalysisSettings

        req = AnalyzeExperimentRequest(
            dataset_id="d1",
            prompt="x" * 4000,
            settings=AnalysisSettings(cuped=False, bootstrap=False),
        )
        assert len(req.prompt) == 4000

    def test_oversized_prompt_is_rejected_over_the_real_http_api(self):
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        resp = client.post(
            "/datasets/classify",
            data={"use_demo": "true"},
        )
        dataset_id = resp.json()["datasetId"]

        resp = client.post(
            "/experiments/analyze",
            json={
                "datasetId": dataset_id,
                "prompt": "x" * 4001,
                "settings": {"cuped": False, "bootstrap": False, "model": "claude-sonnet", "costUsd": 0},
            },
        )
        assert resp.status_code == 422
