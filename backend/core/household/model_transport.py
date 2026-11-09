"""Authenticated native-worker model transport; upstream credentials stay central."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import json

import httpx

from backend.core.llm import LLMResponse, ToolCall
from backend.core.runtime.contracts import BudgetExhausted, current_execution
from .contracts import AccessDenied, Conflict, ReconciliationRequired
from .store import uid

current_household_admission: ContextVar[dict | None] = ContextVar("native_household_admission", default=None)


def response_to_wire(response):
    return {"content": response.content, "thinking": response.thinking,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls],
            "finish_reason": response.finish_reason, "usage": response.usage, "raw": response.raw,
            "content_is_tool_call_tail": response.content_is_tool_call_tail}


def response_from_wire(value):
    if not isinstance(value, dict) or not isinstance(value.get("content"), str):
        raise ReconciliationRequired("Invalid retained native provider response")
    calls = []
    for call in value.get("tool_calls", []):
        if not isinstance(call, dict) or not all(isinstance(call.get(k), str) and call[k] for k in ("id", "name")) or not isinstance(call.get("arguments"), dict):
            raise ReconciliationRequired("Invalid retained native tool call")
        calls.append(ToolCall(call["id"], call["name"], call["arguments"]))
    return LLMResponse(content=value["content"], thinking=value.get("thinking", ""), tool_calls=calls,
                       finish_reason=value.get("finish_reason"), usage=value.get("usage", {}),
                       raw=value.get("raw", {}), content_is_tool_call_tail=value.get("content_is_tool_call_tail", False))


class NativeModelClient:
    """LLM-compatible client and trusted execution hooks for one worker binding.

    ``transport``/``sync_transport`` exist for deterministic tests. Production uses
    private Unix sockets; no generation endpoint or provider key is accepted from
    a prompt, tool argument, model options, or resident configuration.
    """

    def __init__(self, uds_path, token, resident_id, config, *, transport=None, sync_transport=None):
        self.config = config
        self.resident_id = resident_id
        self.runtime = None
        self.bindings = {}
        self._headers = {"X-Tamagi-Worker": token}
        self._client = httpx.AsyncClient(base_url="http://household", headers=self._headers,
            transport=transport or httpx.AsyncHTTPTransport(uds=str(uds_path)),
            timeout=httpx.Timeout(None, connect=10.0))
        self._sync = httpx.Client(base_url="http://household", headers=self._headers,
            transport=sync_transport or httpx.HTTPTransport(uds=str(uds_path)), timeout=10.0)

    def lineage(self, context=None):
        context = context or current_execution.get()
        if context is None or context.execution_id not in self.bindings:
            raise AccessDenied("No native execution binding is active")
        return dict(self.bindings[context.execution_id])

    def _project(self, result):
        if self.runtime:
            for charge in result.get("charges", []):
                self.runtime.project_model_charge(charge)

    def _decode(self, response):
        try:
            result = response.json()
        except (ValueError, TypeError) as error:
            raise ReconciliationRequired("Coordinator response was not recoverable; do not replay generation") from error
        if not isinstance(result, dict):
            raise ReconciliationRequired("Coordinator returned an invalid operation snapshot")
        self._project(result)
        if response.is_error:
            kind, detail = result.get("error"), result.get("detail", "Native coordinator operation failed")
            if kind == "budget_exhausted":
                raise BudgetExhausted(detail)
            if kind == "interrupted":
                raise asyncio.CancelledError(detail)
            if kind == "access_denied":
                raise AccessDenied(detail)
            if kind == "conflict":
                raise Conflict(detail)
            raise ReconciliationRequired(detail)
        return result

    async def _request(self, path, payload):
        try:
            response = await self._client.post("/_native/broker/" + path, json=payload)
        except httpx.HTTPError as error:
            # This is worker-to-coordinator uncertainty, not a new upstream
            # attempt. GovernedLLM's provider retry loop must not duplicate it.
            raise ReconciliationRequired("Coordinator acknowledgment was lost; inspect the retained attempt") from error
        return self._decode(response)

    async def start_execution(self, context, max_calls, *, runtime=None):
        if runtime is not None:
            self.runtime = runtime
        admission = current_household_admission.get()
        grant = (admission.get("grant_id") or admission.get("id")) if isinstance(admission, dict) else None
        result = await self._request("executions/start", {"execution_id": context.execution_id,
            "parent_id": context.parent_id, "actor_id": context.actor_id, "origin": context.origin,
            "max_calls": max_calls, "grant_id": grant})
        self.bindings[context.execution_id] = result["lineage"]
        return result["lineage"]

    async def end_execution(self, context, status):
        if context.execution_id not in self.bindings:
            return
        payload = {"execution_id": context.execution_id, "status": status}
        try:
            await self._request("executions/finish" if status == "completed" else "executions/stop", payload)
        except Conflict:
            # A closed local activation cannot retain a live tool lease when a
            # dropped observer or unfinished child prevents successful finish.
            # Stop revokes effects; a healthy upstream operation still drains.
            await self._request("executions/stop", {**payload, "status": "interrupted"})
            raise

    def assert_active(self, context):
        self.lineage(context)
        try:
            response = self._sync.post("/_native/broker/assert-active", json={"execution_id": context.execution_id})
        except httpx.HTTPError as error:
            raise ReconciliationRequired("Cannot confirm the current native tool lease") from error
        self._decode(response)

    async def resources(self, context=None):
        context = context or current_execution.get()
        result = await self._request("resources", {"execution_id": context.execution_id})
        self.bindings[context.execution_id].update(result["lineage"])
        return result["lineage"]

    async def request_quiet(self, context, expected_segment):
        return await self._request("quiet/request", {"execution_id": context.execution_id, "segment": expected_segment})

    async def advance_quiet(self, context, expected_segment):
        result = await self._request("quiet/advance", {"execution_id": context.execution_id, "segment": expected_segment})
        self.bindings[context.execution_id].update(result["lineage"])
        return result["lineage"]

    async def chat(self, messages, tools=None, **kwargs):
        return await self.chat_for(current_execution.get(), messages, tools=tools, **kwargs)

    async def chat_for(self, context, messages, tools=None, **kwargs):
        self.lineage(context)
        attempt_id = uid("native-attempt")
        options = {k: kwargs[k] for k in ("temperature", "max_tokens", "tool_choice") if k in kwargs}
        if set(kwargs) - set(options):
            raise ValueError("Unsupported native generation options")
        try:
            result = await self._request("generate", {"execution_id": context.execution_id,
                "attempt_id": attempt_id, "messages": [m.to_dict() for m in messages],
                "tools": tools, "options": options, "stream": False})
        except asyncio.CancelledError:
            await asyncio.shield(self._request("executions/stop", {"execution_id": context.execution_id, "status": "interrupted"}))
            raise
        if not result.get("delivery_permitted"):
            raise asyncio.CancelledError("Native execution was stopped before provider result delivery")
        return response_from_wire(result["response"])

    async def chat_stream(self, messages, tools=None, **kwargs):
        context = current_execution.get()
        self.lineage(context)
        options = {k: kwargs[k] for k in ("temperature", "max_tokens", "tool_choice") if k in kwargs}
        if set(kwargs) - set(options):
            raise ValueError("Unsupported native generation options")
        payload = {"execution_id": context.execution_id, "attempt_id": uid("native-attempt"),
                   "messages": [m.to_dict() for m in messages], "tools": tools, "options": options, "stream": True}
        try:
            async with self._client.stream("POST", "/_native/broker/generate", json=payload) as response:
                if response.is_error:
                    await response.aread()
                    self._decode(response)
                ended = False
                projected = ""
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event["type"] == "delta":
                        projected += event["content"]
                        yield event["content"]
                    elif event["type"] == "completed":
                        self._project(event)
                        self.last_stream_response = response_from_wire(event["response"])
                        if not event.get("delivery_permitted"):
                            raise asyncio.CancelledError("Native streaming execution was stopped")
                        if self.last_stream_response.content.startswith(projected):
                            suffix = self.last_stream_response.content[len(projected):]
                            if suffix:
                                yield suffix
                        ended = True
                    elif event["type"] == "error":
                        self._decode(httpx.Response(event.get("status", 503), json=event))
                if not ended:
                    raise ReconciliationRequired("Native stream ended without a retained completion")
        except asyncio.CancelledError:
            await asyncio.shield(self._request("executions/stop", {"execution_id": context.execution_id, "status": "interrupted"}))
            raise
        except (httpx.HTTPError, ValueError, KeyError) as error:
            raise ReconciliationRequired("Native stream transport was interrupted; never replay its generation") from error

    async def count_context_tokens(self, messages, tools=None):
        result = await self._request("tokenize", {"messages": [m.to_dict() for m in messages], "tools": tools})
        return result.get("measurement")

    async def ping(self):
        # Health is observation. Never inherit LLMClient.ping's chat fallback.
        try:
            return bool((await self._request("health", {})).get("available"))
        except ReconciliationRequired:
            return False

    async def close(self):
        await self._client.aclose()
        self._sync.close()
