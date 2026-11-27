"""Review 2026-11-28 B6: after a 429 the endpoint cools down; a request made while every endpoint
cooled down failed at once with no cause (so the redo list had no reset time, and the error read
like a wrong API key), a usage limit that resets in minutes was tried four times on the same
endpoint, and a short rate limit failed the next pages one after another."""
import asyncio
import logging

import httpx
import openai
import pytest

from manga_translator import api_key_rotation as rotation
from manga_translator.agent.client import _retry_same, quota_reset_seconds, quota_resets

LOG = logging.getLogger("test")


def _endpoint(key="k1"):
    return rotation.APIEndpoint(feature="agent", provider="openai", slot=1, api_key="sk-test", base_url="http://api",
                                model_name="m", status_key=f"test-{key}", label=key)


def _limit(seconds=792, retry_after=None):
    error = {"type": "usage_limit_reached", "message": "The usage limit has been reached", "resets_in_seconds": seconds}
    response = httpx.Response(429, headers={"retry-after": str(retry_after)} if retry_after else {},
                              request=httpx.Request("POST", "http://api/chat/completions"))
    return openai.RateLimitError(f"Error code: 429 - {{'error': {error}}}", response=response, body=error)


@pytest.fixture(autouse=True)
def _clean():
    yield
    for key in ("test-k1", "test-k2"):
        rotation._API_STATUS.pop(key, None)
        rotation._LAST_ERRORS.pop(key, None)


def _run(endpoint, operation, **kwargs):
    return asyncio.run(rotation.run_with_api_candidates(
        endpoints=(endpoint,), strategy="failover", operation=operation, provider_name="agent",
        operation_name="chat completion", logger=LOG, retry_attempts=3, **kwargs))


def test_a_usage_limit_is_not_retried_and_the_next_request_keeps_its_reset_time(monkeypatch):
    monkeypatch.setattr(rotation.asyncio, "sleep", lambda s: asyncio.sleep(0))
    endpoint, calls = _endpoint(), []

    async def operation(_endpoint):
        calls.append(1)
        raise _limit()

    with pytest.raises(rotation.APIRotationExhaustedError):
        _run(endpoint, operation, retry_same=_retry_same, cooldown_for=quota_reset_seconds)
    assert len(calls) == 1  # not four times a few seconds apart
    status = rotation.get_api_status(endpoint)
    assert status["state"] == "cooldown" and status["cooldown_until"] - status["updated_at"] == 600  # capped
    with pytest.raises(rotation.APIRotationExhaustedError) as caught:  # the next page, while it cools down
        _run(endpoint, operation, retry_same=_retry_same, cooldown_for=quota_reset_seconds)
    assert len(calls) == 1 and "429" in str(caught.value)
    assert quota_resets(caught.value).get("resets_at")  # the redo list can say when


def test_a_short_rate_limit_waits_for_the_endpoint(monkeypatch):
    slept = []

    async def sleep(seconds):
        slept.append(seconds)
        rotation._API_STATUS["test-k1"]["cooldown_until"] = 0  # the cooldown is over

    monkeypatch.setattr(rotation.asyncio, "sleep", sleep)
    endpoint = _endpoint()
    rotation.record_api_failure(endpoint, _limit(seconds=None, retry_after=5))

    async def operation(_endpoint):
        return "ok"

    assert _run(endpoint, operation) == "ok" and 4 <= slept[0] <= 6


def test_other_errors_are_still_retried_on_the_same_endpoint():
    assert _retry_same(RuntimeError("connection reset")) and _retry_same(_limit(seconds=5))
    assert not _retry_same(_limit(seconds=792))
    bad = openai.BadRequestError("Error code: 400", body={}, response=httpx.Response(
        400, request=httpx.Request("POST", "http://api/chat/completions")))
    assert not _retry_same(bad)


def test_a_rate_limit_is_not_reported_as_a_key_error(monkeypatch):
    import _bootstrap  # noqa: F401
    import app_logic
    from services.i18n_service import I18nManager

    manager = I18nManager(locale_dir=str(_bootstrap.ROOT / "desktop_qt_ui" / "locales"), fallback_locale="zh_CN",
                          config_language="en_US")
    monkeypatch.setattr(app_logic, "get_i18n_manager", lambda: manager)
    build = app_logic.TranslationWorker._build_friendly_error_message
    cooled = build("agent has no available API candidates for chat completion. Every candidate is cooling down "
                   "after: Error code: 429 - usage_limit_reached", "")
    assert "HTTP 429" in cooled and "API key or candidate unavailable" not in cooled
    keyless = build("agent has no available API candidates for chat completion.", "")
    assert "API key or candidate unavailable" in keyless  # as before
