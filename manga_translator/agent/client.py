"""OpenAI-compatible chat client for the agent, on top of the existing API key
rotation (``OPENAI_API_KEY`` / ``OPENAI_API_BASE`` / ``OPENAI_MODEL`` and their
numbered candidates), so the agent uses the same endpoints as the OpenAI translators."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

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
        self.session_id: Optional[str] = None
        """Sent as the ``session_id`` header, so a proxy that groups requests (for example an
        API proxy) shows one chapter run, with its recall and review calls, as one session."""

    @property
    def model_label(self) -> str:
        if self.settings.model:
            return self.settings.model
        return self.endpoints[0].model_name if self.endpoints else ""

    def _client(self, endpoint):
        from openai import AsyncOpenAI
        key = f"{endpoint.base_url}|{endpoint.status_key}"
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url,
                                             timeout=self.settings.request_timeout, max_retries=0)
        return self._clients[key]

    async def complete(self, messages: List[dict], tools: List[dict], tool_choice: str) -> ChatReply:
        from ..api_key_rotation import run_with_api_candidates

        headers = {"session_id": self.session_id} if self.session_id else None

        async def operation(endpoint) -> ChatReply:
            response = await self._client(endpoint).chat.completions.create(
                model=self.settings.model or endpoint.model_name, messages=messages, tools=tools,
                tool_choice=tool_choice, extra_headers=headers)
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
