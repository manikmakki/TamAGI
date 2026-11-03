"""
LLM Client — Universal v1/chat/completions interface.

Supports: OpenAI, Anthropic (via proxy), Ollama, vLLM, LM Studio,
llama.cpp server, and any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, AsyncIterator

import httpx

from backend.config import LLMConfig

logger = logging.getLogger("tamagi.llm")


_CHANNEL_HEADER_RE = re.compile(
    r'(?im)(^|\n)[ \t]*(?:\*+)?[ \t]*<\|channel>[ \t]*'
    r'(?P<name>[a-zA-Z_-]+)[ \t]*(?:\|>)?[ \t]*(?:\*+)?[ \t]*'
    r'(?:<channel\|>[ \t]*|'
    r'(?:\r?\n)[ \t]*(?:\*+)?[ \t]*<channel\|>[ \t]*(?:(?:\*+)?[ \t]*(?=\r?\n|$))?)?'
)
_CHANNEL_MARKER_RE = re.compile(
    r'(?im)^[ \t]*(?:\*+)?[ \t]*(?:<\|channel>[ \t]*[a-zA-Z_-]+[ \t]*(?:\|>)?|<channel\|>)'
    r'[ \t]*(?:\*+)?[ \t]*$'
)
_THINK_BLOCK_RE = re.compile(r'(?is)<think>\s*(.*?)\s*</think>')
_TEXT_TOOL_CALL_RE = re.compile(
    r'(?is)(?:\*+)?[ \t]*<\|tool_call>[ \t]*call:'
    r'(?P<name>[a-zA-Z_][\w-]*)[ \t]*\{(?P<args>.*?)\}[ \t]*(?:\*+)?'
)
_REASONING_CHANNELS = {"thought", "thinking", "reasoning", "analysis"}
_VISIBLE_CHANNELS = {"final", "answer", "response", "assistant"}


def _join_parts(parts: list[str]) -> str | None:
    clean = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(clean) if clean else None


def _clean_channel_markers(text: str) -> str:
    """Remove standalone channel marker lines that leaked into a block."""
    lines = [line for line in text.splitlines() if not _CHANNEL_MARKER_RE.match(line)]
    return "\n".join(lines).strip()


def _strip_wrapping_emphasis(text: str) -> str:
    stripped = text.strip()
    for marker in ("***", "**", "*"):
        if stripped.startswith(marker) and stripped.endswith(marker) and len(stripped) > len(marker) * 2:
            return stripped[len(marker):-len(marker)].strip()
    return stripped


def _extract_text_reasoning(text: str | None) -> tuple[str | None, str | None]:
    """Split model-emitted reasoning markup out of plain text content."""
    if not isinstance(text, str) or not text:
        return text, None

    matches = list(_CHANNEL_HEADER_RE.finditer(text))
    if matches:
        visible_parts: list[str] = []
        thinking_parts: list[str] = []
        leading = text[:matches[0].start()].strip()
        if leading:
            visible_parts.append(leading)

        for idx, match in enumerate(matches):
            channel = (match.group("name") or "").strip().lower()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = _clean_channel_markers(text[start:end])
            if not body:
                continue
            if channel in _REASONING_CHANNELS:
                thinking_parts.append(body)
            elif channel in _VISIBLE_CHANNELS:
                visible_parts.append(body)
            else:
                visible_parts.append(body)

        if thinking_parts and not visible_parts:
            return _join_parts(thinking_parts), None
        return _join_parts(visible_parts), _join_parts(thinking_parts)

    thinking_parts: list[str] = []

    def _pull_think(match: re.Match[str]) -> str:
        thinking_parts.append(match.group(1).strip())
        return ""

    visible = _THINK_BLOCK_RE.sub(_pull_think, text).strip()
    return visible or None, _join_parts(thinking_parts)


def _parse_text_tool_args(raw: str) -> dict[str, Any]:
    normalized = raw.replace('<|"|>', '"').strip()
    if not normalized:
        return {}

    jsonish = re.sub(r'(?<!")\b([a-zA-Z_]\w*)\b\s*:', r'"\1":', normalized)
    try:
        parsed = json.loads("{" + jsonish + "}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        args: dict[str, Any] = {}
        for part in normalized.split(","):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.strip().strip('"\'')
            value = value.strip().strip('"\'')
            if key:
                args[key] = value
        return args


def _extract_text_tool_calls(text: str | None) -> tuple[str | None, list[tuple[str, dict[str, Any]]]]:
    """Remove DiffusionGemma/vLLM text tool-call markers and return parsed calls."""
    if not isinstance(text, str) or not text:
        return text, []

    stripped = text.strip()
    wrapping_marker = next(
        (marker for marker in ("***", "**", "*") if stripped.startswith(marker) and stripped.endswith(marker)),
        "",
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    def _pull_call(match: re.Match[str]) -> str:
        calls.append((match.group("name"), _parse_text_tool_args(match.group("args"))))
        return ""

    cleaned = _TEXT_TOOL_CALL_RE.sub(_pull_call, text)
    if calls and wrapping_marker and cleaned.strip().endswith(wrapping_marker):
        cleaned = cleaned.strip()[:-len(wrapping_marker)].strip()
    elif calls and cleaned.strip().endswith("*") and not cleaned.strip().startswith("*"):
        cleaned = cleaned.strip().rstrip("*").strip()
    return _strip_wrapping_emphasis(cleaned) or None, calls


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
        content_is_tool_call_tail: bool = False,
    ):
        self.content = content or ""
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.raw = raw or {}
        self.thinking = thinking or ""
        self.content_is_tool_call_tail = content_is_tool_call_tail

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

        # Format 3: some chat templates emit thinking as plain text channel
        # markers, e.g. DiffusionGemma via vLLM:
        #   <|channel>thought
        #   <channel|>...
        text_content, text_thinking = _extract_text_reasoning(text_content)
        if thinking:
            cleaned_thinking, nested_thinking = _extract_text_reasoning(thinking)
            thinking = _join_parts([nested_thinking or "", cleaned_thinking or ""])
        thinking = _join_parts([thinking or "", text_thinking or ""])

        text_content, text_tool_calls = _extract_text_tool_calls(text_content)
        thinking, thinking_tool_calls = _extract_text_tool_calls(thinking)
        content_is_tool_call_tail = bool(text_tool_calls and text_content)
        if thinking_tool_calls and thinking and not text_content:
            text_content = thinking
            thinking = None
            content_is_tool_call_tail = True
        for name, args in [*text_tool_calls, *thinking_tool_calls]:
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=args,
            ))

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
            raw=data,
            thinking=thinking,
            content_is_tool_call_tail=content_is_tool_call_tail,
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
