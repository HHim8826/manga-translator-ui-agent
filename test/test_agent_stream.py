"""Streamed replies (``stream_requests``): the reply starts within seconds and keeps coming, so the
user's proxy, which cuts a silent connection after 120 s, does not cut a long one. The pieces are put
together as one reply with its token counts; an API that refuses streaming, or streams without token
counts (they drive compaction), gets requests without it."""
import asyncio
from types import SimpleNamespace

import httpx
import openai

from manga_translator import api_key_rotation
from manga_translator.agent.client import OpenAIChatClient
from manga_translator.agent.settings import AgentSettings


def _piece(content=None, calls=(), usage=None, finish=None):
    tool_calls = [SimpleNamespace(index=i, id=cid, function=SimpleNamespace(name=name, arguments=args))
                  for i, cid, name, args in calls]
    delta = SimpleNamespace(content=content, refusal=None, tool_calls=tool_calls or None)
    choices = [] if usage else [SimpleNamespace(delta=delta, finish_reason=finish)]
    return SimpleNamespace(choices=choices, usage=usage)


USAGE = SimpleNamespace(prompt_tokens=5482, completion_tokens=32,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=5376))
PIECES = [_piece(content="I'll look "), _piece(content="at both."),
          _piece(calls=[(0, "c0", "zoom", ""), (1, "c1", "set_text", '{"id": ')]),
          _piece(calls=[(0, None, None, '{"box": [1, 2'), (1, None, None, '3}')]),
          _piece(calls=[(0, None, None, ", 3, 4]}")], finish="tool_calls"),
          _piece(usage=USAGE)]


class Stream:
    def __init__(self, pieces):
        self.pieces = list(pieces)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.pieces:
            raise StopAsyncIteration
        return self.pieces.pop(0)

    async def close(self):
        self.closed = True


def _client(monkeypatch, pieces=PIECES, refuse="", replies=None, attempts=1, **settings):
    """``replies``: the pieces of each stream in turn (otherwise every stream sends ``pieces``);
    ``attempts``: how often the request is sent when it fails, like the API rotation's retries."""
    sent, streams = [], []
    replies = list(replies or [])

    class Completions:
        async def create(self, **kwargs):
            sent.append(kwargs)
            if refuse and kwargs.get("stream"):
                response = httpx.Response(400, request=httpx.Request("POST", "http://api/chat/completions"))
                raise openai.BadRequestError(refuse, response=response, body=None)
            if kwargs.get("stream"):
                streams.append(Stream(replies.pop(0) if replies else pieces))
                return streams[-1]
            call = SimpleNamespace(id="c0", function=SimpleNamespace(name="zoom", arguments="{}"))
            message = SimpleNamespace(content=None, tool_calls=[call], refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=USAGE)

    async def run(*, endpoints, operation, **_):
        for attempt in range(attempts):
            try:
                return await operation(endpoints[0])
            except RuntimeError:
                if attempt == attempts - 1:
                    raise

    monkeypatch.setattr(api_key_rotation, "run_with_api_candidates", run)
    endpoint = SimpleNamespace(model_name="m", base_url="u", status_key="k", api_key="x")
    client = OpenAIChatClient((endpoint,), "failover", AgentSettings(**settings))
    api = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(client, "_client", lambda e: api)
    return client, sent, streams


def test_a_streamed_reply_is_put_together_with_its_token_counts(monkeypatch):
    client, sent, streams = _client(monkeypatch)
    reply = asyncio.run(client.complete([], [], "required", effort="high"))
    assert reply.content == "I'll look at both."
    assert [(c.id, c.name, c.arguments) for c in reply.tool_calls] == [
        ("c0", "zoom", '{"box": [1, 2, 3, 4]}'), ("c1", "set_text", '{"id": 3}')]
    assert (reply.prompt_tokens, reply.cached_tokens, reply.completion_tokens) == (5482, 5376, 32)
    assert sent[0]["stream"] is True and sent[0]["stream_options"] == {"include_usage": True}
    assert sent[0]["reasoning_effort"] == "high" and streams[0].closed
    assert AgentSettings().stream_requests is True  # on by default


def test_an_api_that_refuses_streaming_gets_requests_without_it(monkeypatch):
    client, sent, _ = _client(monkeypatch, refuse="Unsupported parameter: 'stream_options'")
    for _ in range(2):
        assert asyncio.run(client.complete([], [], "required")).tool_calls[0].name == "zoom"
    # refused once: sent again without streaming, and the next request goes without it at once
    assert [k.get("stream", False) for k in sent] == [True, False, False]


def test_an_error_that_only_mentions_upstream_is_not_a_refusal(monkeypatch):
    client, sent, _ = _client(monkeypatch, refuse="upstream error: model overloaded")
    try:
        asyncio.run(client.complete([], [], "required"))
    except openai.BadRequestError:
        pass
    else:
        raise AssertionError("the error was swallowed")
    assert len(sent) == 1 and not client._no_stream


def test_a_stream_without_token_counts_stops_streaming(monkeypatch):
    client, sent, _ = _client(monkeypatch, pieces=PIECES[:-1])
    first = asyncio.run(client.complete([], [], "required"))
    assert first.tool_calls[0].name == "zoom" and first.prompt_tokens == 0
    second = asyncio.run(client.complete([], [], "required"))
    assert second.prompt_tokens == 5482  # compaction needs the counts
    assert [k.get("stream", False) for k in sent] == [True, False]


def test_streaming_can_be_turned_off(monkeypatch):
    client, sent, _ = _client(monkeypatch, stream_requests=False)
    assert asyncio.run(client.complete([], [], "required")).prompt_tokens == 5482
    assert "stream" not in sent[0] and "stream_options" not in sent[0]


def test_an_empty_stream_is_an_error(monkeypatch):
    client, _, _ = _client(monkeypatch, pieces=[_piece(usage=USAGE)])
    try:
        asyncio.run(client.complete([], [], "required"))
    except RuntimeError as err:
        assert "empty reply" in str(err)
    else:
        raise AssertionError("an empty reply was accepted")


DONE_CALL = [_piece(calls=[(0, "c9", "mark_page_done", '{"summary": "ok"}')])]


def test_a_stream_cut_before_its_end_hands_on_no_tool_call(monkeypatch):
    """Review item 12: a complete-looking ``mark_page_done`` and then the stream stops, with no finish
    reason and no token counts. Nothing is handed on, streaming stays on for the next try."""
    client, sent, streams = _client(monkeypatch, pieces=DONE_CALL)
    try:
        asyncio.run(client.complete([], [], "required"))
    except RuntimeError as err:
        assert "stopped before its end" in str(err)
    else:
        raise AssertionError("half a reply was handed on")
    assert streams[0].closed and not client._no_stream

    # sent again (the API rotation retries an error): the whole reply comes this time
    client, sent, _ = _client(monkeypatch, replies=[DONE_CALL, PIECES], attempts=2)
    reply = asyncio.run(client.complete([], [], "required"))
    assert [c.name for c in reply.tool_calls] == ["zoom", "set_text"] and reply.prompt_tokens == 5482
    assert [k.get("stream", False) for k in sent] == [True, True]


def test_the_token_counts_that_close_a_stream_are_its_end_too(monkeypatch):
    """An API that sends no finish reason still ends its stream with the token counts."""
    pieces = [_piece(calls=[(0, "c0", "zoom", '{"box": [1, 2, 3, 4]}')]), _piece(usage=USAGE)]
    client, sent, _ = _client(monkeypatch, pieces=pieces)
    reply = asyncio.run(client.complete([], [], "required"))
    assert reply.tool_calls[0].name == "zoom" and reply.prompt_tokens == 5482 and not client._no_stream
