"""A step that failed on the API's usage limit is done again by hand (``redo``), not waited for: the run
ends as before, the error says when the limit resets, and the user starts the chapter again later.
The API clients are closed in the run's own event loop."""
import asyncio

import httpx
import openai
import pytest

from manga_translator import api_key_rotation
from manga_translator.agent.client import OpenAIChatClient, quota_reset_seconds, quota_resets
from manga_translator.agent.settings import AgentSettings


def _limit(seconds=792, resets_at=None, headers=None, code=429):
    """The error an API proxy sends when its usage limit is used up (ch37 proofread,
    2026-11-22)."""
    error = {"type": "usage_limit_reached", "message": "The usage limit has been reached",
             "limit_window_minutes": 300}
    if seconds is not None:
        error["resets_in_seconds"] = seconds
    if resets_at is not None:
        error["resets_at"] = resets_at
    response = httpx.Response(code, headers=headers or {},
                              request=httpx.Request("POST", "http://api/chat/completions"))
    cls = openai.RateLimitError if code == 429 else openai.InternalServerError
    return cls(f"Error code: {code} - {{'error': {error}}}", response=response, body=error)


def test_the_reset_time_is_read_from_the_error_the_api_sent():
    now = 1_795_357_000.0
    assert quota_reset_seconds(_limit(792), now) == 792
    try:  # as the agent sees it: the key rotation gave up, the 429 is its cause
        try:
            raise _limit(792)
        except openai.RateLimitError as exc:
            raise api_key_rotation.APIRotationExhaustedError("agent chat completion failed") from exc
    except api_key_rotation.APIRotationExhaustedError as wrapped:
        assert quota_reset_seconds(wrapped, now) == 792
        assert set(quota_resets(wrapped, now)) == {"resets"} and len(quota_resets(wrapped, now)["resets"]) == 5
    assert quota_reset_seconds(_limit(None, resets_at=now + 600), now) == 600
    assert quota_reset_seconds(_limit(None, resets_at=(now + 60) * 1000), now) == pytest.approx(60)
    assert quota_reset_seconds(_limit(None, headers={"retry-after": "120"}), now) == 120
    # only the text of the error is left (another client library)
    text = RuntimeError("Error code: 429 - {'error': {'type': 'usage_limit_reached', 'resets_in_seconds': 792}}")
    assert quota_reset_seconds(text, now) == 792
    assert quota_reset_seconds(_limit(None), now) is None  # a 429 that does not say when
    assert quota_reset_seconds(_limit(792, code=500), now) is None
    assert quota_reset_seconds(RuntimeError("the streamed reply stopped before its end"), now) is None
    assert quota_resets(RuntimeError("boom"), now) == {}


def test_the_api_clients_are_closed_in_the_loop_that_opened_them():
    endpoint = api_key_rotation.APIEndpoint(feature="translator", provider="openai", slot=1, api_key="x",
                                            base_url="http://127.0.0.1:9/v1", model_name="m",
                                            status_key="test-redo-close", label="close")
    client = OpenAIChatClient((endpoint,), "failover", AgentSettings())

    async def run():
        api = client._client(endpoint)
        await client.aclose()
        return api

    api = asyncio.run(run())
    assert api.is_closed() and not client._clients
