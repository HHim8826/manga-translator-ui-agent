"""Each agent conversation sends its own prompt_cache_key, so the API keeps a conversation on the
machines that hold its cached prefix; an API that refuses the field gets requests without it."""
import asyncio
from types import SimpleNamespace

import httpx
import openai

from manga_translator import api_key_rotation
from manga_translator.agent.client import OpenAIChatClient
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.workflow import run_agent_workflow
from test_agent_core import FakeTranslator, ScriptedClient, _finish_page


def _client(monkeypatch, refuse=()):
    sent = []

    class Completions:
        async def create(self, **kwargs):
            sent.append(kwargs)
            body = kwargs.get("extra_body") or {}
            for field in refuse:
                if field in kwargs or field in body:
                    response = httpx.Response(400, request=httpx.Request("POST", "http://api/chat/completions"))
                    raise openai.BadRequestError(f"Unsupported parameter: '{field}'", response=response, body=None)
            call = SimpleNamespace(id="c0", function=SimpleNamespace(name="zoom", arguments="{}"))
            message = SimpleNamespace(content=None, tool_calls=[call], refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    async def run(*, endpoints, operation, **_):
        return await operation(endpoints[0])

    monkeypatch.setattr(api_key_rotation, "run_with_api_candidates", run)
    endpoint = SimpleNamespace(model_name="m", base_url="u", status_key="k", api_key="x")
    client = OpenAIChatClient((endpoint,), "failover", AgentSettings(stream_requests=False))
    api = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(client, "_client", lambda e: api)
    return client, sent


def _key(kwargs):
    return (kwargs.get("extra_body") or {}).get("prompt_cache_key")


def test_the_client_sends_the_conversation_key_with_the_session(monkeypatch):
    client, sent = _client(monkeypatch)
    asyncio.run(client.complete([], [], "required"))
    asyncio.run(client.complete([], [], "required", cache_key="translator"))
    client.session_id = "manga-translator-r1"
    asyncio.run(client.complete([], [], "required", cache_key="inpaint-003"))
    assert [_key(k) for k in sent] == [None, "translator", "manga-translator-r1:inpaint-003"]
    assert sent[2]["extra_headers"] == {"session_id": "manga-translator-r1"}  # the proxy still groups the run


def test_an_api_that_refuses_the_key_gets_requests_without_it(monkeypatch):
    client, sent = _client(monkeypatch, refuse=("prompt_cache_key",))
    for _ in range(2):
        reply = asyncio.run(client.complete([], [], "required", effort="high", cache_key="translator"))
        assert reply.tool_calls[0].name == "zoom"
    # refused once: sent again without it (effort kept), and the next request goes without it at once
    assert [(_key(k), k.get("reasoning_effort")) for k in sent] == [
        ("translator", "high"), (None, "high"), (None, "high")]


def test_an_api_that_refuses_both_fields_still_answers(monkeypatch):
    client, sent = _client(monkeypatch, refuse=("prompt_cache_key", "reasoning_effort"))
    assert asyncio.run(client.complete([], [], "required", effort="low", cache_key="recall")).tool_calls
    assert [(_key(k), k.get("reasoning_effort")) for k in sent][-1] == (None, None)


def test_each_agent_conversation_has_its_own_key(tmp_path):
    from test_agent_inpaint import FakeBackend

    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png")]

    class Cfg:
        class translator:
            target_lang = "CHT"

    client = ScriptedClient(_finish_page() + _finish_page())
    asyncio.run(run_agent_workflow(FakeTranslator(tmp_path), [(p, Cfg) for p in paths], {}, client=client,
                                   log=lambda m: None, inpaint_backend=FakeBackend(),
                                   agent_settings={"proofread": True}))
    keys = {}
    for tool, key in client.cache_keys:
        keys.setdefault(tool, set()).add(key)
    assert keys["report_unboxed_text"] == {"recall"} and keys["view_page"] == {"proofread"}
    assert keys["view_region"] == {"inpaint-001", "inpaint-002"}  # one inpaint review conversation per page
    helpers = ("report_unboxed_text", "view_region", "view_page")
    assert {k for t, k in client.cache_keys if t not in helpers} == {"translator"}
