"""
LLM Client — Universal v1/chat/completions interface.

Supports: OpenAI, Anthropic (via proxy), Ollama, vLLM, LM Studio,
llama.cpp server, and any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

import httpx

from backend.config import LLMConfig

logger = logging.getLogger("tamagi.llm")


class LLMMessage:
    """A single message in a conversation."""

    def __init__(
        self,
        role: str,
        content: str | list[dict],
        name: str | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ):
        self.role = role
        self.content = content
        self.name = name
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        # `name` is valid on user/assistant messages only; the OpenAI spec does
        # not define it for tool messages and strict backends (llama.cpp) may reject it.
        if self.name and self.role != "tool":
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
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
        thinking: str | None = None,
    ):
        self.content = content or ""
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.raw = raw or {}
        self.thinking = thinking or ""

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
        # "ollama" is the conventional no-auth sentinel for Ollama deployments.
        # For llama.cpp (or any other backend) set api_key to your actual token,
        # or leave it empty to skip the Authorization header entirely.
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

        # Ollama-specific: tells Ollama how large a KV-cache to allocate.
        # Without it, Ollama may silently truncate long conversations.
        # Ignored by llama.cpp and other backends (context size is a server
        # startup flag for those, not a per-request parameter).
        if self.config.num_ctx is not None:
            payload["num_ctx"] = self.config.num_ctx

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

        logger.info("LLM request → %s  model=%s  messages=%d", url, self.config.model, len(messages))

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

    async def chat_with_retry(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        attempts: int = 1,
        delay: float = 2.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """chat() with automatic retry on server disconnect errors."""
        last_exc: Exception | None = None
        for i in range(attempts + 1):
            try:
                return await self.chat(messages, tools=tools, **kwargs)
            except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
                last_exc = e
                if i < attempts:
                    logger.warning(f"LLM disconnected, retrying ({i + 1}/{attempts}): {e}")
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

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
                id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                name=func.get("name", ""),
                arguments=args,
            ))

        # Extract thinking/reasoning content.
        # Format 1: reasoning_content field (DeepSeek R1, LiteLLM proxy for Anthropic).
        thinking: str | None = message.get("reasoning_content") or None

        # Format 2: content is a list of typed blocks (raw Anthropic-compatible proxy).
        raw_content = message.get("content")
        text_content: str | None = None
        if isinstance(raw_content, list):
            thinking_parts: list[str] = []
            text_parts: list[str] = []
            for block in raw_content:
                btype = block.get("type")
                if btype == "thinking":
                    thinking_parts.append(block.get("thinking") or block.get("text") or "")
                elif btype == "text":
                    text_parts.append(block.get("text") or "")
            if thinking_parts:
                thinking = "\n\n".join(thinking_parts)
            text_content = "\n\n".join(text_parts) if text_parts else None
        else:
            text_content = raw_content

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
            raw=data,
            thinking=thinking,
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
