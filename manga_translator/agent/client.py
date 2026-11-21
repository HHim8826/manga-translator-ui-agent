"""OpenAI-compatible chat client for the agent, on top of the existing API key
rotation (``OPENAI_API_KEY`` / ``OPENAI_API_BASE`` / ``OPENAI_MODEL`` and their
numbered candidates), so the agent uses the same endpoints as the OpenAI translators."""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Set

from .runner import ChatReply, ToolCall
from .settings import AgentSettings

logger = logging.getLogger("manga_translator.agent")


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
                try:
                    response = await self._client(endpoint).chat.completions.create(
                        model=self.settings.model or endpoint.model_name, messages=messages, tools=tools,
                        tool_choice=tool_choice, extra_headers=headers, **extra)
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
                    else:
                        raise
            if not response.choices:
                raise RuntimeError("the API returned no choices")
            message = response.choices[0].message
            calls = [ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}")
                     for c in (message.tool_calls or [])]
            if not calls and not (message.content or "").strip():
                refusal = getattr(message, "refusal", None)
                raise RuntimeError(f"empty reply{': ' + refusal if refusal else ''}")
            usage = response.usage
            details = getattr(usage, "prompt_tokens_details", None) if usage else None
            return ChatReply(content=message.content, tool_calls=calls,
                             prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                             completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                             cached_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0)

        return await run_with_api_candidates(
            endpoints=self.endpoints, strategy=self.strategy, operation=operation, provider_name="agent",
            operation_name="chat completion", logger=logger, runtime_config=self.runtime_config,
            retry_attempts=self.settings.retry_attempts)


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
