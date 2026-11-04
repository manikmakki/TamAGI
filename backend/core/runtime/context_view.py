"""Bound model-facing views while canonical messages and receipts retain their bytes."""

from __future__ import annotations
import copy
import json


class ContextViewTooLarge(ValueError):
    pass


def text_view(value, limit=1200):
    text = str(value)
    if len(text) <= limit:
        return text
    tail = min(200, limit // 4)
    return (
        text[: limit - tail]
        + "\n[context excerpt; full source retained]\n"
        + text[-tail:]
    )


def value_view(value, depth=3):
    if isinstance(value, str):
        return text_view(value)
    if depth <= 0:
        return "[nested source omitted from context]"
    if isinstance(value, dict):
        pairs = list(value.items())
        result = {key: value_view(item, depth - 1) for key, item in pairs[:16]}
        if len(pairs) > 16:
            result["context_omitted_fields"] = len(pairs) - 16
        return result
    if isinstance(value, list):
        result = [value_view(item, depth - 1) for item in value[:12]]
        if len(value) > 12:
            result.append({"context_omitted_items": len(value) - 12})
        return result
    return value


def json_view(value, limit=4096):
    rendered = json.dumps(value_view(value), ensure_ascii=False, default=str)
    if len(rendered.encode()) <= limit:
        return rendered
    # A JSON string can require six bytes per source character for control escapes.
    excerpt = rendered[: max(32, (limit - 200) // 6)]
    return json.dumps(
        {
            "context_truncated": True,
            "preview": excerpt,
            "detail": "Full source remains in canonical storage.",
        },
        ensure_ascii=False,
    )


def receipt_view(receipt):
    result = receipt.get("payload", {}).get("result", {})
    return {
        "id": receipt["id"],
        "actor": receipt["actor"],
        "tool": receipt["tool"],
        "status": receipt["status"],
        "started": receipt["started"],
        "finished": receipt.get("finished"),
        "reported_success": result.get("success"),
        "output_excerpt": json_view(result.get("output", ""), 1600),
        "error_excerpt": json_view(result.get("error"), 600),
        "details": "Use inspect_receipt with this id for paged original output and metadata.",
    }


def tool_result_json(result, limit=4096):
    raw = json.dumps(result, ensure_ascii=False, default=str)
    if len(raw.encode()) <= limit:
        return raw
    source = result if isinstance(result, dict) else {"output": str(result)}
    receipt_id = (source.get("data") or {}).get("receipt_id")
    return json.dumps(
        {
            "success": source.get("success"),
            "receipt_id": receipt_id,
            "context_truncated": True,
            "output_excerpt": json_view(source.get("output", ""), max(256, limit // 2)),
            "error_excerpt": json_view(source.get("error"), max(128, limit // 8)),
            "detail": "Full result is retained. Use inspect_receipt with section output, error, data or arguments.",
        },
        ensure_ascii=False,
    )


def _clip_bytes(text, limit):
    raw = text.encode()
    if len(raw) <= limit:
        return text
    marker = b"\n[context excerpt; full source retained]\n"
    available = max(0, limit - len(marker))
    head = available * 3 // 4
    tail = available - head
    return (
        raw[:head].decode(errors="ignore")
        + marker.decode()
        + (raw[-tail:].decode(errors="ignore") if tail else "")
    )


def bound_messages(messages, tools=None, max_bytes=65536):
    """Keep protocol envelopes intact, shortening only the model's text view.

    This is a conservative byte allowance, not an exact tokenizer or image-token
    estimator. Reserve tool-schema space and template overhead. Images remain intact.
    """
    from backend.core.llm import LLMMessage

    current = copy.deepcopy(messages)
    schema_bytes = len(json.dumps(tools or [], ensure_ascii=False).encode())
    budget = max_bytes - schema_bytes - 2048
    if budget < 4096:
        raise ContextViewTooLarge(
            "Tool schemas exceed the configured model input allowance"
        )

    for message in current:
        if message.role == "tool" and isinstance(message.content, str):
            try:
                content = json.loads(message.content)
            except (ValueError, TypeError):
                content = {"output": message.content}
            message.content = tool_result_json(content)

    def text_size(message):
        size = 0
        if isinstance(message.content, str):
            size += len(message.content.encode())
        elif isinstance(message.content, list):
            size += sum(
                len(str(item.get("text", "")).encode())
                for item in message.content
                if item.get("type") == "text"
            )
        size += len(json.dumps(message.tool_calls or [], ensure_ascii=False).encode())
        return size + 128

    # Remove old whole conversational turns, preserving the current user turn and
    # every assistant tool call plus its matching tool results.
    while sum(map(text_size, current)) > budget:
        users = [
            index for index, message in enumerate(current) if message.role == "user"
        ]
        if len(users) < 2:
            break
        start = users[0]
        end = users[1]
        current[start:end] = []

    if sum(map(text_size, current)) > budget:
        share = max(128, (budget - len(current) * 256) // max(1, len(current)))
        for message in current:
            if isinstance(message.content, str):
                message.content = _clip_bytes(message.content, share)
            elif isinstance(message.content, list):
                for item in message.content:
                    if item.get("type") == "text":
                        item["text"] = _clip_bytes(str(item.get("text", "")), share)
            if message.tool_calls:
                for call in message.tool_calls:
                    function = call.get("function", {})
                    arguments = function.get("arguments", "")
                    if len(str(arguments).encode()) > share:
                        function["arguments"] = json_view(
                            {"argument_excerpt": str(arguments)}, share
                        )
    # Always leave the system context explicit when earlier turns were omitted.
    if current and current[0].role != "system":
        current.insert(
            0,
            LLMMessage(
                "system",
                "Earlier context may be excerpted; full records remain stored.",
            ),
        )
    if sum(map(text_size, current)) > budget:
        raise ContextViewTooLarge(
            "Protocol messages cannot fit the configured model input allowance"
        )
    return current
