"""
LLM Client — Universal interface for OpenAI and Anthropic APIs.

Supports:
  - OpenAI and OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, llama.cpp)
  - Anthropic Messages API (native, no proxy required)

Provider is auto-detected from base_url unless explicitly set in config.
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

# Anthropic API version — update when adopting new features
ANTHROPIC_VERSION = "2023-06-01"


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
    Universal LLM client supporting OpenAI and Anthropic APIs.

    Usage:
        client = LLMClient(config)
        response = await client.chat([
            LLMMessage("system", "You are TamAGI."),
            LLMMessage("user", "Hello!"),
        ])

    Provider is auto-detected from base_url unless config.provider is set:
      - URLs containing "anthropic.com" → Anthropic Messages API
      - Everything else → OpenAI-compatible /chat/completions
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.provider = self._detect_provider()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=10.0),
            headers=self._build_headers(),
        )
        logger.info(f"LLMClient initialized: provider={self.provider}, model={config.model}")

    def _detect_provider(self) -> str:
        """Determine provider from config or auto-detect from URL."""
        if self.config.provider:
            return self.config.provider.lower()
        # Auto-detect from URL
        if "anthropic.com" in self.base_url.lower():
            return "anthropic"
        return "openai"

    def _build_headers(self) -> dict[str, str]:
        """Build request headers appropriate for the provider."""
        headers = {"Content-Type": "application/json"}

        if self.provider == "anthropic":
            # Anthropic uses x-api-key header, not Bearer token
            if self.config.api_key:
                headers["x-api-key"] = self.config.api_key
            headers["anthropic-version"] = ANTHROPIC_VERSION
        else:
            # OpenAI-style Bearer token
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
        """Route to provider-specific payload builder."""
        if self.provider == "anthropic":
            return self._build_anthropic_payload(messages, tools, stream, **kwargs)
        return self._build_openai_payload(messages, tools, stream, **kwargs)

    def _build_openai_payload(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build payload for OpenAI-compatible endpoints."""
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

        # Pass num_ctx to Ollama if configured. This tells Ollama exactly how
        # large a KV-cache to allocate. Without it, Ollama uses the model's
        # built-in default (often 2048 or 4096), which may silently truncate
        # long conversations. Harmless/ignored by non-Ollama backends.
        if self.config.num_ctx is not None:
            payload["num_ctx"] = self.config.num_ctx

        return payload

    def _build_anthropic_payload(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build payload for Anthropic Messages API.

        Key differences from OpenAI:
          - System message is a top-level 'system' field, not in messages array
          - Tools use 'input_schema' instead of 'parameters'
          - No 'tool_choice' string — uses object format
          - Tool results are role: "user" with tool_result content blocks
          - Assistant tool calls are content blocks, not a separate field

        Message order in agent history:
          assistant (empty) → tool result → tool result → ...

        Required Anthropic format:
          assistant (with tool_use blocks) → user (with tool_result blocks)
        """
        # First pass: extract system messages and identify structure
        system_parts: list[str] = []
        non_system: list[LLMMessage] = []

        for msg in messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_parts.append(msg.content)
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block.get("text", ""))
            else:
                non_system.append(msg)

        # Second pass: convert message sequence with tool call/result handling
        # Look ahead to find tool results that follow assistant messages
        chat_messages: list[dict[str, Any]] = []
        i = 0
        tool_id_counter = 0

        while i < len(non_system):
            msg = non_system[i]

            if msg.role == "assistant":
                # Look ahead for consecutive tool messages
                tool_results: list[LLMMessage] = []
                j = i + 1
                while j < len(non_system) and non_system[j].role == "tool":
                    tool_results.append(non_system[j])
                    j += 1

                if tool_results:
                    # Build assistant message with tool_use blocks
                    content_blocks: list[dict[str, Any]] = []
                    if msg.content and str(msg.content).strip():
                        content_blocks.append({"type": "text", "text": str(msg.content)})

                    # Build tool_result blocks for the user message
                    result_blocks: list[dict[str, Any]] = []

                    for tr_msg in tool_results:
                        tool_id_counter += 1
                        tool_use_id = f"toolu_{tool_id_counter:024d}"

                        # Add tool_use to assistant
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": tr_msg.name or "tool",
                            "input": {},  # Original input not preserved in history
                        })

                        # Add tool_result to user
                        result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tr_msg.content if isinstance(tr_msg.content, str) else json.dumps(tr_msg.content),
                        })

                    chat_messages.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })
                    chat_messages.append({
                        "role": "user",
                        "content": result_blocks,
                    })

                    # Skip past the tool messages we just processed
                    i = j
                    continue
                else:
                    # Regular assistant message (no following tool results)
                    chat_messages.append(msg.to_dict())

            elif msg.role == "tool":
                # Orphan tool message (shouldn't happen, but handle gracefully)
                tool_id_counter += 1
                tool_use_id = f"toolu_{tool_id_counter:024d}"
                logger.warning(f"Orphan tool message found: {msg.name}")
                # Insert a synthetic assistant + user pair
                chat_messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": msg.name or "tool",
                        "input": {},
                    }],
                })
                chat_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
                    }],
                })

            else:
                # user or other roles
                chat_messages.append(msg.to_dict())

            i += 1

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": chat_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        if stream:
            payload["stream"] = True

        if tools:
            # Convert OpenAI tool format to Anthropic format
            anthropic_tools = []
            for tool in tools:
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            payload["tools"] = anthropic_tools

            # Anthropic tool_choice format
            tool_choice = kwargs.get("tool_choice", "auto")
            if tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                # Anthropic doesn't have "none" — just omit tools
                del payload["tools"]
            elif tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict):
                payload["tool_choice"] = tool_choice

        return payload

    def _get_endpoint(self) -> str:
        """Return the appropriate API endpoint for the provider."""
        if self.provider == "anthropic":
            return f"{self.base_url}/v1/messages"
        return f"{self.base_url}/chat/completions"

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request."""
        url = self._get_endpoint()
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
        url = self._get_endpoint()
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
                    content = self._extract_stream_content(chunk)
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    def _extract_stream_content(self, chunk: dict[str, Any]) -> str | None:
        """Extract content delta from a streaming chunk."""
        if self.provider == "anthropic":
            # Anthropic streaming: content_block_delta events contain text
            event_type = chunk.get("type", "")
            if event_type == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    return delta.get("text")
            return None
        else:
            # OpenAI streaming format
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            return delta.get("content")

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse response from either OpenAI or Anthropic format."""
        if self.provider == "anthropic":
            return self._parse_anthropic_response(data)
        return self._parse_openai_response(data)

    def _parse_openai_response(self, data: dict[str, Any]) -> LLMResponse:
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

    def _parse_anthropic_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse Anthropic Messages API response.

        Anthropic response structure:
        {
            "id": "msg_...",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "tool_use", "id": "toolu_...", "name": "...", "input": {...}}
            ],
            "stop_reason": "end_turn" | "tool_use",
            "usage": {"input_tokens": N, "output_tokens": M}
        }
        """
        content_blocks = data.get("content", [])

        # Extract text content
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            block_type = block.get("type", "")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))

        # Map Anthropic stop_reason to OpenAI-style finish_reason
        stop_reason = data.get("stop_reason", "")
        finish_reason_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
        }
        finish_reason = finish_reason_map.get(stop_reason, stop_reason)

        # Normalize usage format
        usage = data.get("usage", {})
        normalized_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }

        return LLMResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=normalized_usage,
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
