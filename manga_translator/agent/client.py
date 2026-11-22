"""OpenAI-compatible chat client for the agent, on top of the existing API key
rotation (``OPENAI_API_KEY`` / ``OPENAI_API_BASE`` / ``OPENAI_MODEL`` and their
numbered candidates), so the agent uses the same endpoints as the OpenAI translators."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from .runner import ChatReply, ToolCall
from .settings import AgentSettings

logger = logging.getLogger("manga_translator.agent")

_RESET_TEXT = re.compile(r"""['"]?(resets_in_seconds|resets_at)['"]?\s*[:=]\s*(\d+(?:\.\d+)?)""")


def _reset_from(values: Dict[str, Any], now: float) -> Optional[float]:
    """``resets_in_seconds`` (seconds from the reply) or ``resets_at`` (a Unix time, also in ms)."""
    seconds = values.get("resets_in_seconds")
    if isinstance(seconds, (int, float, str)):
        try:
            return float(seconds)
        except ValueError:
            pass
    at = values.get("resets_at")
    if isinstance(at, (int, float, str)):
        try:
            at = float(at)
        except ValueError:
            return None
        return (at / 1000 if at > 1e12 else at) - now
    return None


def quota_reset_seconds(error: BaseException, now: Optional[float] = None) -> Optional[float]:
    """Seconds until the API's usage limit resets, when a 429 in ``error`` or its causes says so: the
    error body's ``resets_in_seconds`` / ``resets_at`` (for example an API proxy's ``usage_limit_reached``,
    ``resets_in_seconds: 792``) or a ``Retry-After`` header. None for any other error, and for a 429
    that does not say when."""
    now = time.time() if now is None else now
    seen: Set[int] = set()
    exc: Optional[BaseException] = error
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = str(exc)
        if getattr(exc, "status_code", None) == 429 or "429" in text:
            body = getattr(exc, "body", None)
            for values in (body, body.get("error") if isinstance(body, dict) else None):
                found = _reset_from(values, now) if isinstance(values, dict) else None
                if found is not None:
                    return max(0.0, found)
            headers = getattr(getattr(exc, "response", None), "headers", None)
            retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    try:
                        return max(0.0, parsedate_to_datetime(retry_after).timestamp() - now)
                    except (TypeError, ValueError):
                        pass
            found = _reset_from({m.group(1): m.group(2) for m in _RESET_TEXT.finditer(text)}, now)
            if found is not None:
                return max(0.0, found)
        exc = exc.__cause__ or exc.__context__
    return None


def quota_resets(error: BaseException, now: Optional[float] = None) -> Dict[str, Any]:
    """Event fields for a step that failed on the API's usage limit, when the 429 said when the limit
    resets, so the user knows when to do the step again (``redo``): ``resets`` (the local time as the
    user reads it: HH:MM, with the date when it is not today) and ``resets_at`` (an ISO time with its
    time zone, which ``redo.py`` keeps). Empty for any other error."""
    import datetime as _dt

    from .redo import reset_label

    now = time.time() if now is None else now
    seconds = quota_reset_seconds(error, now)
    if seconds is None:
        return {}
    when = _dt.datetime.fromtimestamp(now + seconds + 59).astimezone().replace(second=0, microsecond=0)  # rounded up
    return {"resets": reset_label(when, _dt.datetime.fromtimestamp(now).astimezone()),
            "resets_at": when.isoformat(timespec="minutes")}


class OpenAIChatClient:
    def __init__(self, endpoints, strategy: str, settings: AgentSettings, runtime_config=None):
        self.endpoints = endpoints
        self.strategy = strategy
        self.settings = settings
        self.runtime_config = runtime_config
        self._clients: Dict[str, object] = {}
        self._no_effort: Set[str] = set()
        """Endpoints that rejected ``reasoning_effort``: they get requests without it."""
        self._no_cache_key: Set[str] = set()
        """Endpoints that rejected ``prompt_cache_key``: they get requests without it."""
        self._no_stream: Set[str] = set()
        """Endpoints that rejected streaming or streamed without token counts: they get requests
        without it (the token counts drive compaction)."""
        self.session_id: Optional[str] = None
        """Sent as the ``session_id`` header, so a proxy that groups requests (for example an
        API proxy) shows one chapter run, with its recall and review calls, as one session."""

    @property
    def model_label(self) -> str:
        if self.settings.model:
            return self.settings.model
        return self.endpoints[0].model_name if self.endpoints else ""

    @staticmethod
    def _key(endpoint) -> str:
        return f"{endpoint.base_url}|{endpoint.status_key}"

    def _client(self, endpoint):
        from openai import AsyncOpenAI
        key = self._key(endpoint)
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url,
                                             timeout=self.settings.request_timeout, max_retries=0)
        return self._clients[key]

    async def complete(self, messages: List[dict], tools: List[dict], tool_choice: str,
                       effort: str = "", cache_key: str = "") -> ChatReply:
        """One chat completion. ``effort`` is sent as ``reasoning_effort`` unless empty. ``cache_key``
        names the conversation (translator, recall, inpaint-003, proofread); with the run's session id
        it is sent as ``prompt_cache_key``, so the API keeps each conversation on the machines that
        hold its cached prefix instead of spreading every request of the chapter alike (ch36: 5 of 54
        translator requests found only an older part of their prefix cached or none, about 100k tokens)."""
        from openai import BadRequestError

        from ..api_key_rotation import run_with_api_candidates

        headers = {"session_id": self.session_id} if self.session_id else None
        effort = (effort or "").strip().lower()
        prompt_key = ":".join(p for p in (self.session_id, cache_key) if p) if cache_key else ""

        async def operation(endpoint) -> ChatReply:
            key = self._key(endpoint)
            extra: Dict[str, object] = {}
            if effort and key not in self._no_effort:
                extra["reasoning_effort"] = effort
            if prompt_key and key not in self._no_cache_key:
                extra["extra_body"] = {"prompt_cache_key": prompt_key}
            while True:
                stream = self.settings.stream_requests and key not in self._no_stream
                try:
                    api = self._client(endpoint).chat.completions
                    args = dict(model=self.settings.model or endpoint.model_name, messages=messages, tools=tools,
                                tool_choice=tool_choice, extra_headers=headers, **extra)
                    if stream:
                        message, usage = await self._streamed(api, args, endpoint)
                    else:
                        response = await api.create(**args)
                        if not response.choices:
                            raise RuntimeError("the API returned no choices")
                        message, usage = response.choices[0].message, response.usage
                    break
                except BadRequestError as err:
                    text = str(err).lower()
                    if "extra_body" in extra and "prompt_cache_key" in text:
                        self._no_cache_key.add(key)
                        extra.pop("extra_body")
                        logger.warning("The API at %s rejected prompt_cache_key; its requests go without it: %s",
                                       endpoint.base_url, err)
                    elif "reasoning_effort" in extra and "reasoning" in text:
                        self._no_effort.add(key)
                        extra.pop("reasoning_effort")
                        logger.warning("The API at %s rejected reasoning_effort=%s; its requests go without it: %s",
                                       endpoint.base_url, effort, err)
                    elif stream and re.search(r"\bstream(_options)?\b", text):
                        self._no_stream.add(key)
                        logger.warning("The API at %s rejected streaming; its requests go without it: %s",
                                       endpoint.base_url, err)
                    else:
                        raise
            calls = [ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}")
                     for c in (message.tool_calls or [])]
            if not calls and not (message.content or "").strip():
                refusal = getattr(message, "refusal", None)
                raise RuntimeError(f"empty reply{': ' + refusal if refusal else ''}")
            details = getattr(usage, "prompt_tokens_details", None) if usage else None
            return ChatReply(content=message.content, tool_calls=calls,
                             prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                             completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                             cached_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0)

        return await run_with_api_candidates(
            endpoints=self.endpoints, strategy=self.strategy, operation=operation, provider_name="agent",
            operation_name="chat completion", logger=logger, runtime_config=self.runtime_config,
            retry_attempts=self.settings.retry_attempts)

    async def aclose(self) -> None:
        """Close the HTTP connections in the event loop that opened them. Left to the garbage
        collector, the SDK closes them in whatever loop runs by then, after the run's loop has closed:
        "RuntimeError: Event loop is closed" in the log (2026-11-22)."""
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            try:
                await client.close()
            except Exception:
                logger.debug("agent: closing an API client failed", exc_info=True)

    async def _streamed(self, api, args: dict, endpoint) -> Tuple[SimpleNamespace, object]:
        """The reply sent in pieces (``stream_requests``), put together as one message, with the token
        counts of its last piece. The reply starts within seconds and keeps coming while the model
        writes, so a proxy that cuts a silent connection (the user's: 120 s, Cloudflare 524) does not
        cut a long reply. Without token counts the endpoint gets requests without streaming.

        A stream that stops before its end (neither a finish reason nor the token counts that close it)
        is an error, so the request is sent again: the tool calls it brought may be only the first part
        of the reply, even when their arguments happen to parse (review item 12)."""
        response = await api.create(stream=True, stream_options={"include_usage": True}, **args)
        content: List[str] = []
        refusal: List[str] = []
        calls: Dict[int, Dict[str, str]] = {}
        usage = None
        finish = None
        try:
            async for chunk in response:
                usage = getattr(chunk, "usage", None) or usage
                for choice in getattr(chunk, "choices", None) or []:
                    finish = getattr(choice, "finish_reason", None) or finish
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    content.append(getattr(delta, "content", None) or "")
                    refusal.append(getattr(delta, "refusal", None) or "")
                    for part in getattr(delta, "tool_calls", None) or []:
                        index = part.index if getattr(part, "index", None) is not None else len(calls)
                        slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        slot["id"] = slot["id"] or (getattr(part, "id", None) or "")
                        function = getattr(part, "function", None)
                        if function is not None:
                            slot["name"] = slot["name"] or (getattr(function, "name", None) or "")
                            slot["arguments"] += getattr(function, "arguments", None) or ""
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                await close()
        if finish is None and usage is None:
            raise RuntimeError("the streamed reply stopped before its end (no finish reason, no token counts); "
                               "if this API never ends its streams, turn off stream_requests")
        if usage is None:
            self._no_stream.add(self._key(endpoint))
            logger.warning("The API at %s streams without token counts; its requests go without streaming",
                           endpoint.base_url)
        tool_calls = [SimpleNamespace(id=slot["id"] or f"call_{index}",
                                      function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]))
                      for index, slot in sorted(calls.items())]
        message = SimpleNamespace(content="".join(content) or None, tool_calls=tool_calls,
                                  refusal="".join(refusal) or None)
        return message, usage


def build_client(config, settings: AgentSettings) -> OpenAIChatClient:
    from ..runtime_api_resolver import resolve_runtime_api_config
    from ..utils.dotenv_utils import load_app_dotenv

    load_app_dotenv(override=True)
    resolved = resolve_runtime_api_config(
        config, feature="translator", provider="openai",
        api_key_env="OPENAI_API_KEY", api_base_env="OPENAI_API_BASE", model_env="OPENAI_MODEL",
        fallback_api_key_env=None, fallback_api_base_env=None, fallback_model_env=None,
        default_api_base="https://api.openai.com/v1", default_model="gpt-4o", allow_empty_local_api_key=True)
    if not resolved.candidates:
        raise RuntimeError("The agent workflow needs an OpenAI-compatible API: set OPENAI_API_KEY and "
                           "OPENAI_API_BASE in API Management.")
    return OpenAIChatClient(resolved.candidates, resolved.strategy, settings, config)


class APICheckError(RuntimeError):
    """The API cannot run the agents; the message says what to fix."""


PROBE_TOOL = {"type": "function", "function": {
    "name": "report_shape", "description": "Report the shape in the picture.",
    "parameters": {"type": "object", "properties": {"shape": {"type": "string"}}, "required": ["shape"]}}}


def _probe_image() -> str:
    from PIL import Image, ImageDraw

    from .images import data_url

    image = Image.new("RGB", (96, 64), "white")
    ImageDraw.Draw(image).ellipse((28, 12, 68, 52), fill="black")
    return data_url(image)


async def check_api(client) -> None:
    """One small request the way the agents call the model (an image, a tool, tool_choice
    "required"), so a wrong key, address or model stops the run before the slow preprocess."""
    messages = [{"role": "system", "content": "Answer by calling the tool."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Which shape is in the picture?"},
                    {"type": "image_url", "image_url": {"url": _probe_image(), "detail": "low"}}]}]
    model = getattr(client, "model_label", "") or "?"
    fix = ("The agents need an OpenAI-compatible model that takes images and tool calls: check "
           "OPENAI_API_KEY, OPENAI_API_BASE and OPENAI_MODEL in API Management.")
    try:
        reply = await client.complete(messages, [PROBE_TOOL], "required")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise APICheckError(f"AI Agent API check failed (model {model}): {type(exc).__name__}: "
                            f"{str(exc)[:300]}. {fix}") from exc
    if not reply.tool_calls:
        raise APICheckError(f"AI Agent API check failed (model {model}): the model answered without calling "
                            f"the tool. {fix}")
