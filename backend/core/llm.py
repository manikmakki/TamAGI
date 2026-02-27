"""
LLM Client — Universal v1/chat/completions interface.

Supports: OpenAI, Anthropic (via proxy), Ollama, vLLM, LM Studio,
llama.cpp server, and any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from backend.config import LLMConfig

logger = logging.getLogger("tamagi.llm")


class LLMMessage:
    """A single message in a conversation."""

    def __init__(self, role: str, content: str | list[dict], name: str | None = None):
        self.role = role
        self.content = content
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


class ToolCall:
    """Represents a tool/function call from the LLM."""

    def __init__(self, id: str, name: str, arguments: dict[str, Any]):
        self.id = id
        self.name = name
        self.arguments = arguments


class LLMResponse:
    """Response from the LLM."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None,
        raw: dict[str, Any] | None = None,
    ):
        self.content = content or ""
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.raw = raw or {}

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """
    Universal LLM client for any v1/chat/completions endpoint.

    Usage:
        client = LLMClient(config)
        response = await client.chat([
            LLMMessage("system", "You are TamAGI."),
            LLMMessage("user", "Hello!"),
        ])
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=10.0),
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key.lower() != "ollama":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_payload(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        return payload

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request."""
        url = f"{self.base_url}/chat/completions"
        payload = self._build_payload(messages, tools=tools, **kwargs)

        logger.debug(f"LLM request to {url} with model={self.config.model}")

        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM HTTP error: {e.response.status_code} — {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"LLM request error: {e}")
            raise

        return self._parse_response(data)

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response, yielding content deltas."""
        url = f"{self.base_url}/chat/completions"
        payload = self._build_payload(messages, tools=tools, stream=True, **kwargs)

        async with self._client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse a standard v1/chat/completions response."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Parse tool calls if present
        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            raw_args = func.get("arguments", {})
            # Some backends (vLLM, Ollama) return arguments as a dict;
            # OpenAI returns a JSON string. Handle both.
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            else:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=func.get("name", ""),
                arguments=args,
            ))

        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
            raw=data,
        )

    async def close(self):
        await self._client.aclose()

    # Health check
    async def ping(self) -> bool:
        """Check if the LLM endpoint is reachable."""
        try:
            # Try models endpoint first (OpenAI standard)
            resp = await self._client.get(f"{self.base_url}/models")
            return resp.status_code == 200
        except Exception:
            try:
                # Fallback: try a minimal chat request
                test_msg = [LLMMessage("user", "ping")]
                await self.chat(test_msg, max_tokens=1)
                return True
            except Exception:
                return False
