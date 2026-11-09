"""Native coordinator completion service: real provider, canonical budget authority."""

from __future__ import annotations

import asyncio
from collections import deque
import inspect
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from backend.config import LLMConfig
from backend.core.llm import LLMClient, LLMMessage, LLMResponse
from .broker import InferenceBroker
from .contracts import AccessDenied, BudgetExhausted, Conflict, NotFound, ProtocolError, ReconciliationRequired
from .driver import ProviderFinishedError
from .model_transport import response_to_wire
from .store import encoded, uid


class NativeModelServer:
    def __init__(self, store, provider, worker_auth, grant_lookup=None):
        self.store, self.worker_auth, self.grant_lookup = store, worker_auth, grant_lookup
        self.provider = LLMClient(provider) if isinstance(provider, LLMConfig) else provider
        if isinstance(self.provider, LLMClient):
            # Healthy non-streaming processing has no imposed read/elapsed limit.
            self.provider._client.timeout = httpx.Timeout(None, connect=10.0)
        self.broker = InferenceBroker(store)
        self.router = APIRouter(prefix="/_native/broker")
        self._tasks = {}
        self._driver = None
        self._condition = asyncio.Condition()
        self._buffers = {}
        self._observers = {}
        self._overflowed = set()
        self._closed = False
        self.maintenance = False
        self._tokenizer_gate = asyncio.Semaphore(1)
        self._routes()

    async def _auth(self, request):
        result = self.worker_auth(request)
        result = await result if inspect.isawaitable(result) else result
        resident = result if isinstance(result, str) else result.get("resident_id") if isinstance(result, dict) else getattr(result, "resident_id", None)
        if not resident:
            raise AccessDenied("A current authenticated worker binding is required")
        self.broker.register_resident(resident)
        return resident

    @staticmethod
    async def _body(request, required=(), optional=()):
        try:
            value = await request.json()
            encoded(value)
        except (ValueError, TypeError) as error:
            raise ProtocolError("Native broker requires finite JSON") from error
        if not isinstance(value, dict) or set(required) - set(value) or set(value) - set(required) - set(optional):
            raise ProtocolError("Invalid native broker request fields")
        # An accepted 8 MiB image expands to over 10 MiB in base64, before
        # transcript/tool context. Apply the same ceiling to tokenize and generate.
        if len(encoded(value).encode()) > 16 * 1024 * 1024:
            raise ProtocolError("Native completion request is too large")
        return value

    def _binding(self, resident, local_id):
        record = self.store.get("native_execution", resident + ":" + str(local_id))
        if not record:
            raise AccessDenied("Execution was not admitted for this worker")
        return record["data"]

    def _active(self, resident, local_id):
        binding = self._binding(resident, local_id)
        execution = self.broker.execution(binding["broker_execution_id"])
        exchange = self.broker.exchange(execution["exchange_id"])
        cursor = execution
        while cursor:
            if cursor["status"] != "running":
                raise AccessDenied("Native execution has been stopped or completed")
            cursor = self.broker.execution(cursor["parent_id"]) if cursor["parent_id"] else None
        if exchange["status"] != "running":
            raise AccessDenied("Native exchange is no longer active")
        return binding, execution, exchange

    def _charges(self, resident):
        bindings = {b["data"]["broker_execution_id"]: b["data"]["local_execution_id"]
                    for b in self.store.list("native_execution") if b["data"]["resident_id"] == resident}
        charges = []
        for charge in self.store.list("broker_charge"):
            data = charge["data"]
            if data["execution_id"] not in bindings:
                continue
            calls, cursor = {}, data["execution_id"]
            while cursor:
                execution = self.broker.execution(cursor)
                if cursor in bindings:
                    calls[bindings[cursor]] = execution["calls"]
                cursor = execution["parent_id"]
            charges.append({"attempt_id": charge["id"], "local_execution_id": bindings[data["execution_id"]],
                "local_calls": calls, "bucket": data["bucket"], "charged_at": data["charged_at"],
                "exchange_id": data["exchange_id"], "sponsor_id": data["sponsor_id"]})
        return charges

    def _lineage(self, resident, local_id):
        binding = self._binding(resident, local_id)
        execution = self.broker.execution(binding["broker_execution_id"])
        exchange = self.broker.exchange(execution["exchange_id"])
        try:
            remaining = self.broker._remaining(binding["broker_execution_id"])[0]
        except BudgetExhausted:
            remaining = 0
        return {"execution_id": binding["broker_execution_id"], "local_execution_id": local_id,
                "actor_id": execution["actor_id"], "exchange_id": execution["exchange_id"],
                "root_id": execution["root_id"], "sponsor_id": execution["sponsor_id"],
                "origin": exchange["origin"], "segment": exchange["segment"], "calls_remaining": remaining,
                "first_response": binding.get("first_response", False), "helper": execution["helper"]}

    def completion_lineage_for_grant(self, grant_id, resident):
        admitted = self.store.get("native_grant_execution", grant_id)
        if not admitted or admitted["data"]["resident_id"] != resident:
            raise AccessDenied("No generation was admitted for this worker grant")
        local = admitted["data"]["local_execution_id"]
        return self.completion_lineage(local, resident)

    def completion_lineage(self, local, resident):
        binding, _, _ = self._active(resident, local)
        candidates = [r for r in self.store.list("broker_attempt")
                      if r["data"]["execution_id"] == binding["broker_execution_id"]
                      and self.store.get("native_model_result", r["id"])]
        if not candidates:
            raise AccessDenied("A final response requires retained generation evidence")
        latest = max(candidates, key=lambda r: r["data"]["sequence"])
        lineage = {**self._lineage(resident, local), "completion_attempt_id": latest["id"]}
        return self.validate_completion_lineage(lineage, resident)

    def validate_completion_lineage(self, lineage, author):
        """Final-response authority, separate from budget-gated new contact."""
        if not isinstance(lineage, dict):
            raise AccessDenied("A trusted completion lineage is required")
        source = self.store.get("native_model_input", lineage.get("completion_attempt_id"))
        result = self.store.get("native_model_result", lineage.get("completion_attempt_id"))
        if not source or not result or not result["data"].get("delivery_permitted"):
            raise AccessDenied("No deliverable retained completion supports this response")
        resident, payload = source["data"]["resident_id"], source["data"]["payload"]
        binding, execution, _ = self._active(resident, payload["execution_id"])
        if author != execution["actor_id"] or lineage.get("execution_id") != binding["broker_execution_id"]:
            raise AccessDenied("Completion belongs to another actor or execution")
        canonical = self._lineage(resident, payload["execution_id"])
        if any(lineage.get(k) != canonical[k] for k in ("exchange_id", "root_id")):
            raise AccessDenied("Completion resource lineage does not match")
        return {**canonical, "completion_attempt_id": lineage["completion_attempt_id"]}

    async def _grant(self, identity, resident):
        if self.grant_lookup:
            value = self.grant_lookup(identity, resident)
            value = await value if inspect.isawaitable(value) else value
            if isinstance(value, dict) and "data" in value:
                value = value["data"]
        else:
            record = self.store.get("native_grant", identity)
            value = record["data"] if record else None
        if not isinstance(value, dict) or value.get("resident_id") != resident or value.get("status") in {"stopped", "completed", "failed", "interrupted"}:
            raise AccessDenied("The coordinator grant is unavailable for this worker")
        if value.get("origin") not in {"user", "autonomy", "evaluation"} or not value.get("exchange_id"):
            raise AccessDenied("Coordinator grant lacks canonical resource lineage")
        return value

    async def start_execution(self, resident, body):
        if self._closed:
            raise Conflict("Native coordinator is shutting down")
        if self.maintenance:
            raise Conflict("Household paused briefly for resident recovery")
        local, actor, origin = body["execution_id"], body["actor_id"], body["origin"]
        if not all(isinstance(v, str) and v for v in (local, actor, origin)) or type(body["max_calls"]) is not int or body["max_calls"] < 1:
            raise ProtocolError("Invalid native execution binding")
        grant = await self._grant(body["grant_id"], resident) if body.get("grant_id") and not body.get("parent_id") else None
        if self._closed:
            raise Conflict("Native coordinator is shutting down")
        key = resident + ":" + local
        definition = {k: body.get(k) for k in ("execution_id", "parent_id", "actor_id", "origin", "max_calls", "grant_id")}
        with self.store.transaction():
            old = self.store.get("native_execution", key)
            if old:
                if old["data"]["request"] != definition:
                    raise Conflict("Local execution identity was reused with different lineage")
                return {"lineage": self._lineage(resident, local), "charges": self._charges(resident)}
            parent = self._binding(resident, body["parent_id"]) if body.get("parent_id") else None
            if parent:
                self._active(resident, body["parent_id"])
                exchange_id = parent["exchange_id"]
                first_response = False
            elif grant:
                if actor != resident:
                    raise AccessDenied("Only the bound full resident may consume a root grant")
                used = self.store.get("native_grant_execution", body["grant_id"])
                if used:
                    raise Conflict("Coordinator grant already has an execution")
                exchange_id = grant["exchange_id"]
                self.broker.open_exchange(exchange_id, grant["origin"])
                first_response = grant.get("first_response") is True
                self.store.put("native_grant_execution", body["grant_id"], {"resident_id": resident, "local_execution_id": local})
            else:
                if origin not in {"autonomy", "reflection", "evaluation"} or actor != resident:
                    raise AccessDenied("Actual operator or peer input requires its coordinator-issued grant")
                exchange_id = uid("native-exchange")
                self.broker.open_exchange(exchange_id, "evaluation" if origin == "evaluation" else "autonomy")
                first_response = False
            if not parent:
                # Trusted worker admission establishes current eligibility,
                # including a chosen wake or an actual operator interruption.
                self.broker.set_available(resident, True)
            if actor != resident:
                if not parent:
                    raise AccessDenied("Helpers require a current sponsoring parent")
                self.broker.register_companion(actor, resident)
            broker_id = "native:" + key
            tick = self.store.get("native_world_exchange", exchange_id)
            self.broker.open_execution(broker_id, exchange_id, actor,
                parent_id=parent["broker_execution_id"] if parent else None, commissioned=bool(parent),
                world_sponsor=tick["data"]["sponsor_id"] if tick and not parent else None)
            execution = self.broker.execution(broker_id)
            if grant and grant.get("kind") == "invitation":
                execution["max_calls"] = min(execution["max_calls"], body["max_calls"], 4)
            elif execution["helper"]:
                execution["max_calls"] = min(execution["max_calls"], body["max_calls"], 16)
            elif self.broker.exchange(exchange_id)["origin"] != "user":
                execution["max_calls"] = min(execution["max_calls"], body["max_calls"])
            self.store.put("broker_execution", broker_id, execution)
            self.store.put("native_execution", key, {"request": definition, "resident_id": resident,
                "local_execution_id": local, "broker_execution_id": broker_id, "exchange_id": exchange_id,
                "first_response": first_response})
            return {"lineage": self._lineage(resident, local), "charges": self._charges(resident)}

    @staticmethod
    def _messages(values):
        if not isinstance(values, list) or not values:
            raise ProtocolError("Completion messages must be a nonempty list")
        messages = []
        for value in values:
            if not isinstance(value, dict) or set(value) - {"role", "content", "name", "tool_calls", "tool_call_id"} or value.get("role") not in {"system", "developer", "user", "assistant", "tool"} or not isinstance(value.get("content"), (str, list)):
                raise ProtocolError("Invalid native completion message")
            messages.append(LLMMessage(**value))
        return messages

    async def submit(self, resident, body):
        if self._closed:
            raise Conflict("Native coordinator is shutting down")
        if self.maintenance:
            raise Conflict("Household paused briefly for resident recovery")
        messages = self._messages(body["messages"])
        if type(body["stream"]) is not bool or not isinstance(body["options"], dict) or set(body["options"]) - {"temperature", "max_tokens", "tool_choice"}:
            raise ProtocolError("Unsupported native completion options")
        if body["tools"] is not None and not isinstance(body["tools"], list):
            raise ProtocolError("Tools must be an explicit list")
        identity = body["attempt_id"]
        if not isinstance(identity, str) or not 1 <= len(identity) <= 256:
            raise ProtocolError("Completion requires a stable attempt identity")
        with self.store.transaction():
            binding, _, _ = self._active(resident, body["execution_id"])
            def apply():
                attempt = self.broker.request(identity, binding["broker_execution_id"], first_response=binding["first_response"])
                self.store.put("native_model_input", identity, {"resident_id": resident, "payload": body}, expected=0)
                return {"attempt_id": identity, "status": attempt["status"]}
            self.store.command(resident, identity, "native.generate", body, apply)
        self._buffers.setdefault(identity, deque(maxlen=256))
        if self._driver is None or self._driver.done():
            self._driver = asyncio.create_task(self._drive())
        return identity

    async def _drive(self):
        while not self._closed:
            queued = any(r["data"]["status"] == "queued" and self.store.get("native_model_input", r["id"])
                         for r in self.store.list("broker_attempt"))
            if not queued:
                return
            try:
                selected = self.broker.dispatch()
            except ReconciliationRequired:
                async with self._condition:
                    self._condition.notify_all()
                return
            if selected is None:
                async with self._condition:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=0.05)
                    except asyncio.TimeoutError:
                        pass
                continue
            identity = selected["attempt_id"]
            task = asyncio.create_task(self._perform(identity))
            self._tasks[identity] = task
            await asyncio.shield(task)

    def _emit(self, identity, event):
        buffer = self._buffers.setdefault(identity, deque(maxlen=256))
        event = {"seq": (buffer[-1]["seq"] + 1) if buffer else 1, **event}
        buffer.append(event)
        for queue in self._observers.get(identity, set()):
            if queue in self._overflowed:
                continue
            if queue.full():
                # Keep a contiguous prefix. The terminal retained response
                # supplies the missing suffix to a slow observer.
                self._overflowed.add(queue)
            else:
                queue.put_nowait(event)

    async def _stream_response(self, identity, messages, tools, options):
        if not isinstance(self.provider, LLMClient):
            pieces, final = [], None
            async for item in self.provider.chat_stream(messages, tools=tools, **options):
                if isinstance(item, LLMResponse):
                    final = item
                else:
                    if not isinstance(item, str):
                        raise ProtocolError("Invalid injected provider stream delta")
                    pieces.append(item)
                    self._emit(identity, {"type": "delta", "content": item})
            return final or LLMResponse(content="".join(pieces), finish_reason="stop")
        # Preserve complete tool fragments, reasoning, finish reason, and final
        # usage. LLMClient.chat_stream's legacy public interface exposes only text.
        payload = self.provider._build_payload(messages, tools=tools, stream=True, **options)
        payload["stream_options"] = {"include_usage": True}
        content, thinking, tools_by_index = [], [], {}
        usage, finish, ended = {}, None, False
        async with self.provider._client.stream("POST", self.provider.base_url + "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    ended = True
                    break
                chunk = json.loads(raw)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        if not isinstance(text, str):
                            raise ProtocolError("Unsupported provider stream content")
                        content.append(text)
                        self._emit(identity, {"type": "delta", "content": text})
                    if delta.get("reasoning_content"):
                        thinking.append(delta["reasoning_content"])
                    for tool in delta.get("tool_calls") or []:
                        value = tools_by_index.setdefault(tool.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if tool.get("id"):
                            value["id"] = tool["id"]
                        function = tool.get("function") or {}
                        value["function"]["name"] += function.get("name") or ""
                        arguments = function.get("arguments")
                        if arguments is not None:
                            if not isinstance(arguments, str):
                                raise ProtocolError("Invalid provider tool argument fragment")
                            value["function"]["arguments"] += arguments
                    finish = choice.get("finish_reason") or finish
        if not ended:
            raise RuntimeError("Provider stream ended without its completion marker")
        wire = {"choices": [{"message": {"content": "".join(content), "reasoning_content": "".join(thinking),
            "tool_calls": [tools_by_index[i] for i in sorted(tools_by_index)]}, "finish_reason": finish}], "usage": usage}
        return self.provider._parse_response(wire)

    async def _perform(self, identity):
        source = self.store.get("native_model_input", identity)["data"]
        resident, body = source["resident_id"], source["payload"]
        messages = self._messages(body["messages"])
        try:
            if body["stream"]:
                response = await self._stream_response(identity, messages, body["tools"], body["options"])
            else:
                response = await self.provider.chat(messages, tools=body["tools"], **body["options"])
            wire = response_to_wire(response)
            encoded(wire)
            with self.store.transaction():
                settled = self.broker.complete(identity, usage=response.usage)
                result = {"attempt_id": identity, "status": settled["status"], "response": wire,
                          "delivery_permitted": settled["result_delivery_permitted"], "charges": self._charges(resident)}
                self.store.put("native_model_result", identity, result, expected=0)
            self._emit(identity, {"type": "completed", **result})
        except BaseException as error:
            # Only affirmative pre-send connection failure or a complete HTTP
            # rejection proves that no healthy generation remains upstream.
            known = isinstance(error, (ProviderFinishedError, httpx.ConnectError, httpx.ConnectTimeout, httpx.HTTPStatusError))
            outcome = "failed" if known else "unknown"
            with self.store.transaction():
                self.broker.complete(identity, outcome=outcome, provider_error=True)
                result = {"attempt_id": identity, "status": outcome, "delivery_permitted": False,
                          "error": "provider_failed" if known else "reconciliation_required",
                          "detail": type(error).__name__, "charges": self._charges(resident)}
                self.store.put("native_model_result", identity, result, expected=0)
            self._emit(identity, {"type": "error", **result})
        finally:
            async with self._condition:
                self._condition.notify_all()

    async def wait(self, resident, identity):
        source = self.store.get("native_model_input", identity)
        if not source or source["data"]["resident_id"] != resident:
            raise AccessDenied("Attempt belongs to another worker")
        while True:
            result = self.store.get("native_model_result", identity)
            if result:
                return result["data"]
            attempt = self.broker.attempt(identity)
            if attempt["status"] in {"cancelled", "budget_exhausted", "unknown", "reconciled"}:
                return {"attempt_id": identity, "error": "budget_exhausted" if attempt["status"] == "budget_exhausted" else "interrupted" if attempt["status"] == "cancelled" else "reconciliation_required",
                        "detail": "Native attempt is " + attempt["status"], "charges": self._charges(resident)}
            if self.broker.state()["hold"]:
                raise ReconciliationRequired("Upstream occupancy requires operator reconciliation")
            async with self._condition:
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass

    async def close(self):
        self._closed = True
        # Independent evaluations need not have a native_work controller to
        # stop them. Revoke their authority too, so undispatched generations
        # settle instead of leaving private HTTP handlers waiting forever.
        # stop_execution only requests cancellation of running/unknown attempts;
        # their healthy provider calls still drain and uncertainty stays held.
        for record in self.store.list("broker_execution"):
            if record["data"]["status"] == "running":
                self.broker.stop_execution(record["id"])
        async with self._condition:
            self._condition.notify_all()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        if self._driver:
            await asyncio.gather(self._driver, return_exceptions=True)
        if hasattr(self.provider, "close"):
            await self.provider.close()

    async def _quiet(self, resident, body, advance):
        binding, execution, exchange = self._active(resident, body["execution_id"])
        if execution["helper"] or execution["actor_id"] != resident or exchange["origin"] != "user":
            raise AccessDenied("Only a full participant in a user-origin exchange may request quiet work")
        segment = body["segment"]
        if type(segment) is not int or segment < 0:
            raise ProtocolError("Invalid quiet segment")
        key = execution["exchange_id"] + ":" + str(segment)
        with self.store.transaction():
            if exchange["segment"] == segment + 1 and self.store.get("native_quiet", key):
                return {"lineage": self._lineage(resident, body["execution_id"])}
            if exchange["segment"] != segment:
                raise Conflict("Quiet request has a stale exchange segment")
            if not advance:
                self.broker.request_continuation(execution["exchange_id"], binding["broker_execution_id"])
                self.store.put("native_quiet", key, {"requested": True, "advanced": False})
            else:
                request = self.store.get("native_quiet", key)
                if not request:
                    raise AccessDenied("No explicit quiet request was admitted")
                self.broker.advance_segment(execution["exchange_id"])
                self.store.put("native_quiet", key, {"requested": True, "advanced": True})
        return {"lineage": self._lineage(resident, body["execution_id"])}

    def _routes(self):
        def failure(error):
            kind = "budget_exhausted" if isinstance(error, BudgetExhausted) else "access_denied" if isinstance(error, AccessDenied) else "conflict" if isinstance(error, Conflict) else "reconciliation_required" if isinstance(error, ReconciliationRequired) else "invalid_request"
            code = 403 if kind == "access_denied" else 409 if kind in {"conflict", "budget_exhausted"} else 503 if kind == "reconciliation_required" else 400
            return JSONResponse({"error": kind, "detail": str(error)}, status_code=code)

        async def endpoint(request: Request, operation):
            try:
                resident = await self._auth(request)
                if operation == "executions/start":
                    body = await self._body(request, ("execution_id", "parent_id", "actor_id", "origin", "max_calls", "grant_id"))
                    return await self.start_execution(resident, body)
                if operation in {"executions/finish", "executions/stop"}:
                    body = await self._body(request, ("execution_id", "status"))
                    binding = self._binding(resident, body["execution_id"])
                    if operation.endswith("stop"):
                        self.broker.stop_execution(binding["broker_execution_id"])
                    else:
                        self.broker.finish_execution(binding["broker_execution_id"])
                    async with self._condition:
                        self._condition.notify_all()
                    return {"charges": self._charges(resident)}
                if operation in {"resources", "assert-active"}:
                    body = await self._body(request, ("execution_id",))
                    self._active(resident, body["execution_id"])
                    return {"lineage": self._lineage(resident, body["execution_id"]), "charges": self._charges(resident)}
                if operation.startswith("quiet/"):
                    body = await self._body(request, ("execution_id", "segment"))
                    return await self._quiet(resident, body, operation.endswith("advance"))
                if operation == "tokenize":
                    body = await self._body(request, ("messages", "tools"))
                    messages = self._messages(body["messages"])
                    async with self._tokenizer_gate:
                        counter = getattr(self.provider, "count_context_tokens", None)
                        return {"measurement": await counter(messages, body["tools"]) if counter else None}
                if operation == "health":
                    await self._body(request)
                    return {"available": not self.broker.state()["hold"]}
                body = await self._body(request, ("execution_id", "attempt_id", "messages", "tools", "options", "stream"))
                identity = await self.submit(resident, body)
                if not body["stream"]:
                    result = await asyncio.shield(self.wait(resident, identity))
                    return JSONResponse(result, status_code=503 if result.get("error") else 200)
                async def events():
                    queue = asyncio.Queue(maxsize=256)
                    self._observers.setdefault(identity, set()).add(queue)
                    try:
                        # Retained bounded stream events are a UI projection;
                        # provider consumption never waits on this observer.
                        buffered = list(self._buffers.get(identity, []))
                        if buffered and buffered[0]["seq"] != 1:
                            buffered = []  # A late observer receives the final full response.
                            self._overflowed.add(queue)
                        for event in buffered:
                            if event["type"] == "delta":
                                yield encoded(event) + "\n"
                        while not self.store.get("native_model_result", identity):
                            try:
                                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                                if event["type"] == "delta":
                                    yield encoded(event) + "\n"
                            except asyncio.TimeoutError:
                                attempt = self.broker.attempt(identity)
                                if attempt["status"] in {"cancelled", "unknown", "reconciled", "budget_exhausted"}:
                                    break
                        # Completion can be persisted while an observer is
                        # yielding the preceding delta. Drain the already
                        # delivered projection before its terminal envelope.
                        while not queue.empty():
                            event = queue.get_nowait()
                            if event["type"] == "delta":
                                yield encoded(event) + "\n"
                        result = await self.wait(resident, identity)
                        yield encoded({"type": "error" if result.get("error") else "completed", **result}) + "\n"
                    finally:
                        self._observers.get(identity, set()).discard(queue)
                        self._overflowed.discard(queue)
                return StreamingResponse(events(), media_type="application/x-ndjson")
            except (AccessDenied, BudgetExhausted, Conflict, NotFound, ProtocolError, ReconciliationRequired) as error:
                return failure(error)

        for operation in ("executions/start", "executions/finish", "executions/stop", "resources", "assert-active",
                          "quiet/request", "quiet/advance", "generate", "tokenize", "health"):
            def make_handler(name):
                async def handler(request: Request):
                    return await endpoint(request, name)
                return handler
            self.router.add_api_route("/" + operation, make_handler(operation), methods=["POST"])
