"""The API's usage limit (``quota_wait_minutes``): a 429 that says when the limit resets makes the agents
wait for the reset and go on, instead of failing the proofread 13 minutes before it (ch37, 2026-11-22:
``usage_limit_reached``, ``resets_in_seconds: 792``). Every conversation of the run waits for the
same reset; the stop button ends the wait. The API clients are closed in the run's own event loop."""
import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from manga_translator import api_key_rotation
from manga_translator.agent import client as agent_client
from manga_translator.agent import events as agent_events
from manga_translator.agent.client import OpenAIChatClient, quota_reset_seconds
from manga_translator.agent.events import EventLog, read_events
from manga_translator.agent.runner import AgentStopped
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.workflow import _watch_quota

from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)

USAGE = SimpleNamespace(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=None)


def _limit(seconds=792, resets_at=None, headers=None, code=429):
    """The error an API proxy sends when its usage limit is used up."""
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
    assert quota_reset_seconds(_limit(None, resets_at=now + 600), now) == 600
    assert quota_reset_seconds(_limit(None, resets_at=(now + 60) * 1000), now) == pytest.approx(60)
    assert quota_reset_seconds(_limit(None, headers={"retry-after": "120"}), now) == 120
    # only the text of the error is left (another client library)
    text = RuntimeError("Error code: 429 - {'error': {'type': 'usage_limit_reached', 'resets_in_seconds': 792}}")
    assert quota_reset_seconds(text, now) == 792
    assert quota_reset_seconds(_limit(None), now) is None  # a 429 that does not say when
    assert quota_reset_seconds(_limit(792, code=500), now) is None
    assert quota_reset_seconds(RuntimeError("the streamed reply stopped before its end"), now) is None


def _endpoint(name):
    return api_key_rotation.APIEndpoint(feature="translator", provider="openai", slot=1, api_key="x",
                                        base_url="http://127.0.0.1:9/v1", model_name="m",
                                        status_key=f"test-quota-{name}-{time.time()}", label=name)


def _client(monkeypatch, name, failures, **settings):
    """``failures``: what the API raises for the first requests, one per request; then it answers.
    The key rotation sends a failed request once more (``retry_attempts`` 1) before the client sees
    the error, so a limit is two failures."""
    sent = []
    failures = list(failures)

    class Completions:
        async def create(self, **kwargs):
            sent.append(time.time())
            if failures:
                raise failures.pop(0)
            call = SimpleNamespace(id="c0", function=SimpleNamespace(name="zoom", arguments="{}"))
            message = SimpleNamespace(content=None, tool_calls=[call], refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=USAGE)

    monkeypatch.setattr(agent_client, "QUOTA_MARGIN_SECONDS", 0)
    endpoint = _endpoint(name)
    client = OpenAIChatClient((endpoint,), "failover", AgentSettings(
        **{"stream_requests": False, "retry_attempts": 1, **settings}))
    api = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(client, "_client", lambda e: api)
    told = []
    client.on_quota_wait = told.append
    return client, sent, told, endpoint


def test_the_agent_waits_for_the_reset_and_sends_the_request_again(monkeypatch):
    client, sent, told, endpoint = _client(monkeypatch, "wait", [_limit(0.3)] * 2)
    reply = asyncio.run(client.complete([], [], "required", cache_key="proofread"))
    assert reply.tool_calls[0].name == "zoom"
    assert len(sent) == 3 and sent[2] - sent[1] >= 0.25  # sent again after the reset, not before
    assert [sorted(t) for t in told] == [["conversation", "error", "seconds", "until"], ["resumed", "until"]]
    assert told[0]["conversation"] == "proofread" and "usage_limit_reached" in told[0]["error"]
    assert (api_key_rotation.get_api_status(endpoint) or {}).get("state") == "available"
    assert AgentSettings().quota_wait_minutes == 300  # on by default: a 300-minute window


def test_every_conversation_waits_for_the_same_reset(monkeypatch):
    """The proofread hits the limit; the inpaint review, running beside it, waits for the same reset
    instead of using up its attempts on it (and only one notice says so)."""
    client, sent, told, _ = _client(monkeypatch, "shared", [_limit(0.4)] * 2)

    async def both():
        first = asyncio.ensure_future(client.complete([], [], "required", cache_key="proofread"))
        while not client.quota_until:
            await asyncio.sleep(0.01)
        second = asyncio.ensure_future(client.complete([], [], "required", cache_key="inpaint-003"))
        return await asyncio.gather(first, second)

    replies = asyncio.run(both())
    assert all(r.tool_calls for r in replies)
    assert len(sent) == 4 and min(sent[2:]) >= client.quota_until - 0.05  # nothing sent during the wait
    assert sum(1 for t in told if not t.get("resumed")) == 1


def test_an_endpoint_cooling_down_after_the_limit_is_tried_again_after_the_reset(monkeypatch):
    client, sent, _, endpoint = _client(monkeypatch, "cooldown", [])
    api_key_rotation.record_api_failure(endpoint, _limit(None))  # 60 s cooldown, no reset time of its own
    client.quota_until = time.time() + 0.2
    assert asyncio.run(client.complete([], [], "required")).tool_calls
    assert len(sent) == 1


def test_a_limit_is_not_waited_for_when_the_setting_is_off_or_the_reset_is_too_far(monkeypatch):
    client, sent, told, _ = _client(monkeypatch, "off", [_limit(0.1)] * 2, quota_wait_minutes=0)
    with pytest.raises(api_key_rotation.APIRotationExhaustedError):
        asyncio.run(client.complete([], [], "required"))
    assert len(sent) == 2 and not told

    client, sent, told, _ = _client(monkeypatch, "far", [_limit(3 * 3600)] * 2, quota_wait_minutes=60)
    with pytest.raises(api_key_rotation.APIRotationExhaustedError):
        asyncio.run(client.complete([], [], "required"))
    assert len(sent) == 2 and not told

    client, sent, told, _ = _client(monkeypatch, "unknown", [_limit(None)] * 2)  # a 429 that does not say when
    with pytest.raises(api_key_rotation.APIRotationExhaustedError):
        asyncio.run(client.complete([], [], "required"))
    assert not told


def test_the_stop_button_ends_the_wait(monkeypatch):
    client, sent, told, _ = _client(monkeypatch, "stop", [_limit(600)] * 2)
    client.should_stop = lambda: bool(told)
    started = time.time()
    with pytest.raises(AgentStopped):
        asyncio.run(client.complete([], [], "required"))
    assert time.time() - started < 6 and len(sent) == 2


def test_the_api_clients_are_closed_in_the_loop_that_opened_them():
    client = OpenAIChatClient((_endpoint("close"),), "failover", AgentSettings())

    async def run():
        api = client._client(client.endpoints[0])
        await client.aclose()
        return api

    api = asyncio.run(run())
    assert api.is_closed() and not client._clients


def test_the_wait_is_written_to_the_run_log_or_heard_before_the_first_chapter(tmp_path):
    client = SimpleNamespace(on_quota_wait=None, should_stop=None, model_label="gpt")
    live = [None]
    logged = []
    stop = lambda: False  # noqa: E731
    _watch_quota(client, stop, live, logged.append)
    assert client.should_stop is stop
    heard = []
    agent_events.add_listener(heard.append)
    try:
        client.on_quota_wait({"until": time.time() + 792, "seconds": 792, "conversation": "translator"})
        client.on_quota_wait({"until": time.time(), "resumed": True})
    finally:
        agent_events.remove_listener(heard.append)
    stages = [e for e in heard if e.get("type") == "workflow_stage"]
    assert [e["stage"] for e in stages] == ["api_wait", "api_check"]
    assert stages[0]["seconds"] == 792 and len(stages[0]["until"]) == 8 and stages[1]["model"] == "gpt"

    events = EventLog(str(tmp_path / "run.jsonl"), "r1")
    live[0] = events
    client.on_quota_wait({"until": time.time() + 60, "seconds": 60, "conversation": "proofread",
                          "error": "Error code: 429"})
    client.on_quota_wait({"until": time.time(), "resumed": True})
    events.close()
    waits = [e for e in read_events(events.path) if e["type"] == "api_wait"]
    assert [(e["agent"], e.get("seconds"), bool(e.get("resumed"))) for e in waits] == [
        ("proofread", 60, False), ("workflow", None, True)]
    assert any("waiting until" in m for m in logged) and any("has reset" in m for m in logged)


def _page():
    from ui.main_page.pages.agents_page import AgentsPage

    return AgentsPage(_t)


def test_the_page_says_until_when_the_agents_wait(app):  # noqa: F811
    page = _page()
    page.replay([{"type": "run_start", "run": "r1", "batch": "b1", "chapter": "第37話", "pages": ["001.jpg"],
                  "model": "m"},
                 {"type": "proofread_start", "run": "r1", "agent": "proofread", "pages": 1, "regions": 3}])
    assert page.status_label.text() == "Agents Proofread Progress"
    page.handle_event({"type": "api_wait", "run": "r1", "agent": "proofread", "until": "23:05:12", "seconds": 792,
                       "error": "Error code: 429"})
    assert page.status_label.text() == "Agents API Wait Detail minutes=13 until=23:05:12"
    rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())]
    assert rows[-1].text(2) == "Agents API Wait" and rows[-1].text(4) == "Error code: 429"
    page.handle_event({"type": "api_wait", "run": "r1", "agent": "workflow", "until": "23:05:12", "resumed": True})
    assert page.status_label.text() == "Agents Proofread Progress"  # back to what it said before

    page = _page()  # before the first chapter: a stage, and the stop button stays on
    page.set_task_running(True)
    page._handle_live({"type": "workflow_stage", "stage": "api_wait", "until": "23:05:12", "seconds": 30})
    assert page.status_label.text() == "Agents Stage API Wait minutes=1 until=23:05:12"
    assert page._agent_task


def test_the_setting_has_its_texts_in_every_language():
    import glob
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in glob.glob(os.path.join(root, "desktop_qt_ui", "locales", "*.json")):
        with open(path, encoding="utf-8") as fh:
            texts = json.load(fh)
        for key in ("label_quota_wait_minutes", "desc_agent_quota_wait_minutes", "Agents API Wait",
                    "Agents API Wait Detail", "Agents API Wait Done", "Agents Stage API Wait"):
            assert texts.get(key), (os.path.basename(path), key)
        assert "{until}" in texts["Agents API Wait Detail"] and "{minutes}" in texts["Agents Stage API Wait"]
