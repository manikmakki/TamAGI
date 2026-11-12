"""Real Agent/provider protocol through the native broker; all traffic is in memory."""

import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
import httpx

from backend.config import TamAGIConfig, OrchestratorConfig, get_config, set_config
from backend.core.agent import TamAGIAgent
from backend.core.identity import IdentityManager
from backend.core.llm import LLMClient, LLMMessage
from backend.core.personality import PersonalityEngine
from backend.core.orchestrator import Orchestrator
from backend.core.runtime import RuntimeService, RuntimeStore, GovernedLLM, set_runtime
from backend.core.runtime.contracts import BudgetExhausted
from backend.core.runtime.memory import CanonicalMemory
from backend.core.runtime.companions import Companions
from backend.core.runtime.chat_turns import run_chat_turns
from backend.core.runtime.live_evaluation import install_evaluator
from backend.core.household.contracts import AccessDenied, BudgetExhausted as HouseholdBudgetExhausted, WorkerBinding, ReconciliationRequired
from backend.core.household.conversations import Conversations
from backend.core.household.model_server import NativeModelServer
from backend.core.household.model_transport import NativeModelClient, current_household_admission
from backend.core.household.store import HouseholdStore
from backend.core.household.worker_bridge import NativeWorkerBridge
from backend.skills.registry import SkillRegistry
from backend.skills.runtime_skill import install_runtime_skills
from backend.skills.write_skill import WriteSkill


class NativeModelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tamagi-native-model-", dir="/tmp")
        self.root = Path(self.temporary.name)
        self.shared = HouseholdStore(self.root / "household.sqlite3")
        self.now = [10000.0]
        self.shared.clock = lambda: self.now[0]
        self.local = RuntimeStore(self.root / "residents/first/runtime.sqlite3")
        self.runtime = RuntimeService(self.local, self.root / "residents/first/workspace", source_root=self.root)
        self.actor = self.runtime.actor_id
        self.config = TamAGIConfig()
        self.config.workspace.path = str(self.runtime.workspace)
        self.config.history.persist_path = str(self.runtime.workspace.parent / "data/history")
        self.config.guardrails.allowed_read_paths = [str(self.runtime.workspace)]
        self.config.guardrails.allowed_write_paths = [str(self.runtime.workspace)]
        self.config.runtime.index_projection = False
        self.runtime.config = self.config.runtime
        self.previous_config = get_config()
        set_config(self.config)
        set_runtime(self.runtime)
        self.responses, self.requests = [], []
        self.gate, self.started = asyncio.Event(), asyncio.Event()
        self.gate.set()
        self.active = self.maximum_active = 0
        self.upstream_error = None
        self.stream_parts = None

        async def upstream(request):
            payload = json.loads(request.content)
            if request.url.path.endswith("/tokenize"):
                return httpx.Response(200, json={"count": 1000, "max_model_len": 32768})
            self.requests.append(payload)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.started.set()
            try:
                await self.gate.wait()
                self.now[0] += 4
                if self.upstream_error:
                    raise self.upstream_error
                if payload.get("stream"):
                    chunks = [
                        {"choices": [{"delta": {"content": "Hello ", "reasoning_content": "A thought. "}, "finish_reason": None}]},
                        {"choices": [{"delta": {"content": "there", "tool_calls": [{"index": 0, "id": "stable-stream-tool", "function": {"name": "write", "arguments": '{"path":"stream.txt",'}}]}, "finish_reason": None}]},
                        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"content":"kept"}'}}]}, "finish_reason": "tool_calls"}]},
                        {"choices": [], "usage": {"completion_tokens": 7, "prompt_tokens": 12}},
                    ]
                    if self.stream_parts is not None:
                        chunks = [{"choices": [{"delta": {"content": piece}, "finish_reason": None}]} for piece in self.stream_parts]
                        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": len(self.stream_parts)}})
                    return httpx.Response(200, text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n")
                response = self.responses.pop(0) if self.responses else {"content": "Provider final response."}
                return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", **response}, "finish_reason": "tool_calls" if response.get("tool_calls") else "stop"}],
                    "usage": {"completion_tokens": 5, "prompt_tokens": 10}})
            finally:
                self.active -= 1

        self.provider = LLMClient(self.config.llm)
        await self.provider._client.aclose()
        self.provider._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        def auth(request):
            token = request.headers.get("X-Tamagi-Worker")
            if token not in {"first-token", "second-token"}:
                raise AccessDenied("Unknown test worker")
            return self.actor if token == "first-token" else "second-resident"
        self.server = NativeModelServer(self.shared, self.provider, auth)
        self.app = FastAPI()
        self.app.include_router(self.server.router)
        self.http = httpx.AsyncClient(base_url="http://household", transport=httpx.ASGITransport(self.app), headers={"X-Tamagi-Worker": "first-token"})
        self.model = NativeModelClient("/unused.sock", "first-token", self.actor, self.config.llm,
            transport=httpx.ASGITransport(self.app), sync_transport=httpx.MockTransport(self.sync_gateway))
        self.runtime.household = self.model
        self.llm = GovernedLLM(self.model, self.runtime)

    def sync_gateway(self, request):
        body = json.loads(request.content)
        if request.url.path == "/_native/notebook" and body["action"] == "inspect":
            return httpx.Response(200, json=[])
        if request.url.path.endswith("/assert-active"):
            self.server._active(self.actor, body["execution_id"])
            return httpx.Response(200, json={"charges": self.server._charges(self.actor)})
        if request.url.path == "/_native/conversations/commit":
            lineage = self.server.completion_lineage_for_grant(body["grant_id"], self.actor)
            message = self.conversations.send(self.actor, self.grant["conversation_id"], body["local_message"]["content"],
                body["command_id"], lineage=lineage)
            return httpx.Response(200, json={"message": message})
        raise AssertionError("Unexpected synchronous native request: " + str(request.url))

    async def asyncTearDown(self):
        self.gate.set()
        await self.server.close()
        await self.model.close()
        await self.http.aclose()
        if isinstance(self.runtime.household, NativeWorkerBridge):
            self.runtime.household.close()
        set_runtime(None)
        set_config(self.previous_config)
        self.local.close()
        self.shared.close()
        self.temporary.cleanup()

    def grant_user(self, identity="human-grant", first_response=True):
        exchange = "exchange:" + identity
        self.server.broker.open_exchange(exchange, "user")
        grant = {"resident_id": self.actor, "exchange_id": exchange, "conversation_id": None,
                 "origin": "user", "status": "pending", "first_response": first_response}
        self.shared.put("native_grant", identity, grant)
        return {"id": identity, **grant}

    async def test_user_roots_require_gateway_grant_and_cannot_forge_other_resident(self):
        with self.assertRaises(AccessDenied):
            async with self.runtime.execution("user"):
                pass
        self.assertEqual(self.shared.list("broker_exchange"), [])
        grant = self.grant_user()
        token = current_household_admission.set(grant)
        try:
            async with self.runtime.execution("user") as execution:
                with self.assertRaises(AccessDenied):
                    self.server._active("second-resident", execution.execution_id)
                result = await self.llm.chat([LLMMessage("user", "A genuine operator input")])
                self.assertEqual(result.content, "Provider final response.")
                attempt = self.shared.list("broker_attempt")[0]["data"]
                self.assertTrue(attempt["priority"])
                self.assertEqual(self.local.run(execution.execution_id)["calls"], 1)
        finally:
            current_household_admission.reset(token)
        self.assertEqual(len(self.shared.list("broker_charge")), 1)

    async def test_helpers_share_parent_budget_and_do_not_get_independent_accounts(self):
        async with self.runtime.execution("autonomy") as root:
            async with self.runtime.execution("companion", actor_id="ephemeral-helper", max_calls=2) as child:
                for _ in range(2):
                    await self.llm.chat([LLMMessage("user", "Help within the existing allowance")])
                with self.assertRaises(BudgetExhausted):
                    await self.llm.chat([LLMMessage("user", "No third helper call")])
                self.assertEqual(self.local.run(child.execution_id)["calls"], 2)
            self.assertEqual(self.local.run(root.execution_id)["calls"], 2)
        accounts = self.server.broker.state()["accounts"]
        self.assertEqual(set(accounts), {self.actor})
        attempts = self.shared.list("broker_attempt")
        self.assertTrue(all(a["data"]["sponsor_id"] == self.actor and not a["data"]["priority"] for a in attempts))

    async def test_resource_projection_includes_tighter_ancestor_limit(self):
        async with self.runtime.execution("autonomy", max_calls=1):
            async with self.runtime.execution("companion", actor_id="small-helper", max_calls=2) as child:
                self.assertEqual(self.model.lineage(child)["calls_remaining"], 1)
                await self.llm.chat([LLMMessage("user", "One available parent call")])
                self.assertEqual((await self.model.resources(child))["calls_remaining"], 0)
                with self.assertRaises(BudgetExhausted):
                    await self.llm.chat([LLMMessage("user", "No extra allowance from the helper")])

    async def test_twentieth_call_can_commit_final_but_new_contact_has_no_allowance(self):
        grant = self.grant_user()
        token = current_household_admission.set(grant)
        try:
            async with self.runtime.execution("user"):
                for _ in range(20):
                    await self.llm.chat([LLMMessage("user", "Count every provider attempt")])
                lineage = self.server.completion_lineage_for_grant(grant["id"], self.actor)
                self.assertEqual(lineage["calls_remaining"], 0)
                with self.assertRaises(HouseholdBudgetExhausted):
                    self.server.broker.validate_message_lineage(lineage, self.actor)
                with self.assertRaises(BudgetExhausted):
                    await self.llm.chat([LLMMessage("user", "Beyond the shared segment")])
        finally:
            current_household_admission.reset(token)
        self.assertEqual(len(self.requests), 20)
        self.assertEqual(len(self.shared.list("broker_charge")), 20)

    async def test_stream_retains_tool_fragments_reasoning_usage_and_stable_ids(self):
        async with self.runtime.execution("autonomy"):
            text = "".join([part async for part in self.llm.chat_stream([LLMMessage("user", "Stream")])])
            self.assertEqual(text, "Hello there")
            response = self.model.last_stream_response
            self.assertEqual(response.thinking, "A thought.")
            self.assertEqual(response.tool_calls[0].id, "stable-stream-tool")
            self.assertEqual(response.tool_calls[0].arguments, {"path": "stream.txt", "content": "kept"})
            self.assertEqual(response.usage["completion_tokens"], 7)
        self.assertEqual(len(self.shared.list("broker_charge")), 1)

    async def test_slow_bounded_stream_observer_recovers_full_retained_suffix(self):
        self.stream_parts = [str(index) + "," for index in range(600)]
        async with self.runtime.execution("autonomy"):
            text = ""
            async for part in self.llm.chat_stream([LLMMessage("user", "A longer stream")]):
                text += part
                await asyncio.sleep(0)
            self.assertEqual(text, "".join(self.stream_parts))
            self.assertEqual(self.model.last_stream_response.content, text)
        self.assertLessEqual(max(map(len, self.server._buffers.values())), 256)

    async def test_peer_quiet_continuation_retains_shared_user_exchange_without_new_human_input(self):
        grant = self.grant_user(first_response=False)
        grant["message"] = {"author": "another-resident"}
        registry = SkillRegistry()
        install_runtime_skills(self.runtime, registry, Companions(self.runtime, self.llm, registry))
        agent = SimpleNamespace(runtime=self.runtime, llm=self.llm, skills=registry, config=self.config, _pending_approvals={})
        self.responses.extend([
            {"tool_calls": [{"id": "quiet-request", "type": "function", "function": {"name": "request_quiet_turn", "arguments": "{}"}}]},
            {"content": "An undelivered working draft."},
            {"content": "A completed household reply."},
        ])
        token = current_household_admission.set(grant)
        try:
            async with self.runtime.execution("peer") as execution:
                result = await run_chat_turns(agent, [LLMMessage("user", "A peer's attributed message")])
                self.assertEqual(result.status, "completed", result.detail)
                self.assertEqual(result.text, "A completed household reply.")
                self.assertEqual(result.interim, ["An undelivered working draft."])
                lineage = self.model.lineage(execution)
                self.assertEqual(lineage["exchange_id"], grant["exchange_id"])
                self.assertEqual(lineage["segment"], 1)
                self.assertEqual(self.local.run(execution.execution_id)["calls"], 3)
        finally:
            current_household_admission.reset(token)
        self.assertEqual(len(self.shared.list("broker_exchange")), 1)
        self.assertTrue(all(not a["data"]["priority"] for a in self.shared.list("broker_attempt")))
        self.assertIn("Turn kind: household reply", self.requests[0]["messages"][0]["content"])
        self.assertEqual(sum(m["role"] == "user" for m in self.requests[-1]["messages"]), 1)

    async def test_exact_attempt_replay_returns_retained_response_and_charges_once(self):
        async with self.runtime.execution("autonomy") as execution:
            body = {"execution_id": execution.execution_id, "attempt_id": "stable-attempt", "messages": [{"role": "user", "content": "one"}], "tools": None, "options": {}, "stream": False}
            first = await self.http.post("/_native/broker/generate", json=body)
            second = await self.http.post("/_native/broker/generate", json=body)
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(first.json(), second.json())
            changed = await self.http.post("/_native/broker/generate", json={**body, "messages": [{"role": "user", "content": "changed"}]})
            self.assertEqual(changed.status_code, 409)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.shared.list("broker_charge")), 1)

    async def test_stop_running_drains_upstream_and_cancelled_queue_is_uncharged(self):
        self.gate.clear()
        async with self.runtime.execution("autonomy"):
            request = asyncio.create_task(self.llm.chat([LLMMessage("user", "A slow actual provider call")]))
            await asyncio.wait_for(self.started.wait(), 2)
            self.server.broker.register_resident("second-resident")
            await self.server.start_execution("second-resident", {"execution_id": "second-run", "actor_id": "second-resident",
                "origin": "autonomy", "parent_id": None, "max_calls": 128, "grant_id": None})
            second = {"execution_id": "second-run", "attempt_id": "queued-second", "messages": [{"role": "user", "content": "Waiting"}],
                "tools": None, "options": {}, "stream": False}
            await self.server.submit("second-resident", second)
            self.server.broker.stop_execution(self.server._binding("second-resident", "second-run")["broker_execution_id"])
            self.assertEqual(self.server.broker.attempt("queued-second")["status"], "cancelled")
            self.assertEqual(len(self.shared.list("broker_charge")), 1)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            running = self.shared.list("broker_attempt")[0]
            self.assertEqual(self.server.broker.attempt(running["id"])["status"], "running")
            self.assertEqual(self.active, 1)
            self.gate.set()
            settled = await asyncio.wait_for(self.server.wait(self.actor, running["id"]), 2)
            self.assertFalse(settled["delivery_permitted"])
            self.assertEqual(self.maximum_active, 1)
        self.assertEqual(len(self.requests), 1)

    async def test_uncertain_provider_transport_holds_all_new_dispatch(self):
        self.upstream_error = httpx.ReadError("The upstream result became unknowable")
        async with self.runtime.execution("autonomy"):
            with self.assertRaises(ReconciliationRequired):
                await self.llm.chat([LLMMessage("user", "A real counted uncertain attempt")])
            self.assertTrue(self.server.broker.state()["hold"])
            self.assertEqual(len(self.shared.list("broker_charge")), 1)
            with self.assertRaises(ReconciliationRequired):
                await self.llm.chat([LLMMessage("user", "No replacement while occupancy is unknown")])
        self.assertEqual(len(self.requests), 1)

    async def test_shutdown_settles_queued_independent_http_generation_and_refuses_new_admission(self):
        start = {"execution_id": "queued-evaluation", "actor_id": self.actor,
                 "origin": "evaluation", "parent_id": None, "max_calls": 2, "grant_id": None}
        admitted = await self.http.post("/_native/broker/executions/start", json=start)
        self.assertEqual(admitted.status_code, 200, admitted.text)
        self.server.broker.set_available(self.actor, False)
        body = {"execution_id": start["execution_id"], "attempt_id": "queued-at-shutdown",
                "messages": [{"role": "user", "content": "Wait without occupying the provider"}],
                "tools": None, "options": {}, "stream": False}
        request = asyncio.create_task(self.http.post("/_native/broker/generate", json=body))
        async with asyncio.timeout(2):
            while not self.shared.get("native_model_input", body["attempt_id"]):
                await asyncio.sleep(0)
        self.assertFalse(request.done())
        await asyncio.wait_for(self.server.close(), 2)
        response = await asyncio.wait_for(request, 2)
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["error"], "interrupted")
        attempt = self.server.broker.attempt(body["attempt_id"])
        self.assertEqual(attempt["status"], "cancelled")
        self.assertFalse(attempt["charged"])
        self.assertEqual(self.requests, [])
        self.assertEqual(self.shared.list("broker_charge"), [])
        binding = self.server._binding(self.actor, start["execution_id"])
        self.assertEqual(self.server.broker.execution(binding["broker_execution_id"])["status"], "stopped")
        for path, payload in (("executions/start", {**start, "execution_id": "too-late"}),
                              ("generate", {**body, "attempt_id": "too-late"})):
            rejected = await self.http.post("/_native/broker/" + path, json=payload)
            self.assertEqual(rejected.status_code, 409, rejected.text)
            self.assertIn("shutting down", rejected.json()["detail"])
        self.assertIsNone(self.shared.get("native_model_input", "too-late"))

    async def test_shutdown_cancels_undispatched_work_but_drains_healthy_inflight_generation(self):
        self.gate.clear()
        start = {"execution_id": "running-at-shutdown", "actor_id": self.actor,
                 "origin": "evaluation", "parent_id": None, "max_calls": 2, "grant_id": None}
        admitted = await self.http.post("/_native/broker/executions/start", json=start)
        self.assertEqual(admitted.status_code, 200, admitted.text)
        body = {"execution_id": start["execution_id"], "attempt_id": "healthy-at-shutdown",
                "messages": [{"role": "user", "content": "A healthy slow call"}],
                "tools": None, "options": {}, "stream": False}
        running = asyncio.create_task(self.http.post("/_native/broker/generate", json=body))
        await asyncio.wait_for(self.started.wait(), 2)
        self.server.broker.register_resident("second-resident")
        await self.server.start_execution("second-resident", {**start,
            "execution_id": "queued-second", "actor_id": "second-resident"})
        queued_body = {**body, "execution_id": "queued-second", "attempt_id": "undispatched-at-shutdown"}
        queued = asyncio.create_task(self.http.post("/_native/broker/generate", json=queued_body,
            headers={"X-Tamagi-Worker": "second-token"}))
        async with asyncio.timeout(2):
            while not self.shared.get("native_model_input", queued_body["attempt_id"]):
                await asyncio.sleep(0)
        closing = asyncio.create_task(self.server.close())
        try:
            response = await asyncio.wait_for(queued, 2)
            self.assertEqual(response.status_code, 503, response.text)
            self.assertEqual(response.json()["error"], "interrupted")
            self.assertFalse(closing.done(), "Shutdown must wait for healthy upstream completion")
            self.assertFalse(running.done())
            self.assertEqual(self.active, 1)
            self.assertFalse(self.provider._client.is_closed)
            self.assertEqual(self.server.broker.attempt(body["attempt_id"])["status"], "running")
            self.assertTrue(self.server.broker.attempt(body["attempt_id"])["cancel_requested"])
            self.assertEqual(len(self.shared.list("broker_charge")), 1)
        finally:
            self.gate.set()
            await asyncio.wait_for(closing, 2)
        result = await asyncio.wait_for(running, 2)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertFalse(result.json()["delivery_permitted"])
        self.assertEqual(self.server.broker.attempt(body["attempt_id"])["status"], "completed")
        self.assertEqual(self.server.broker.attempt(queued_body["attempt_id"])["status"], "cancelled")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.maximum_active, 1)
        self.assertEqual(len(self.shared.list("broker_charge")), 1)
        self.assertTrue(self.provider._client.is_closed)

    async def test_shutdown_preserves_unknown_provider_occupancy_for_reconciliation(self):
        self.upstream_error = httpx.ReadError("Unknown upstream occupancy")
        start = {"execution_id": "unknown-at-shutdown", "actor_id": self.actor,
                 "origin": "evaluation", "parent_id": None, "max_calls": 2, "grant_id": None}
        admitted = await self.http.post("/_native/broker/executions/start", json=start)
        self.assertEqual(admitted.status_code, 200, admitted.text)
        body = {"execution_id": start["execution_id"], "attempt_id": "unknown-at-shutdown",
                "messages": [{"role": "user", "content": "An uncertain provider transport"}],
                "tools": None, "options": {}, "stream": False}
        response = await self.http.post("/_native/broker/generate", json=body)
        self.assertEqual(response.status_code, 503, response.text)
        before = self.server.broker.state()
        self.assertEqual(before["hold"], [body["attempt_id"]])
        await asyncio.wait_for(self.server.close(), 2)
        after = self.server.broker.state()
        self.assertEqual(after["hold"], before["hold"])
        self.assertEqual(after["active_attempt"], before["active_attempt"])
        attempt = self.server.broker.attempt(body["attempt_id"])
        self.assertEqual(attempt["status"], "unknown")
        self.assertTrue(attempt["cancel_requested"])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.shared.list("broker_charge")), 1)

    async def test_unfinished_local_activation_revokes_lease_while_healthy_provider_drains(self):
        self.gate.clear()
        async with self.runtime.execution("autonomy") as execution:
            request = asyncio.create_task(self.llm.chat([LLMMessage("user", "An observer can end before generation")]))
            await asyncio.wait_for(self.started.wait(), 2)
        binding = self.server._binding(self.actor, execution.execution_id)
        self.assertEqual(self.server.broker.execution(binding["broker_execution_id"])["status"], "stopped")
        self.assertEqual(self.active, 1)
        self.gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await request
        self.assertFalse(self.server.broker.state()["accounts"][self.actor]["runnable"])

    async def test_live_evaluation_trial_provider_uses_counted_central_execution(self):
        install_evaluator(self.runtime, self.llm)
        self.local.put("proposal", "proposal", self.actor, {"surface": "world_prompt", "value": "A small test change"})
        queued = self.runtime.submit_evaluation("proposal", ["chosen_rest"])
        await asyncio.wait_for(asyncio.gather(*self.runtime.evaluation_tasks), 5)
        job = self.local.get("evaluation_job", queued["job_id"])["data"]
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(len(self.requests), 6)  # Three matched baseline/candidate pairs.
        self.assertEqual(len(self.shared.list("broker_charge")), 6)
        self.assertEqual(len(self.shared.list("broker_execution")), 1)
        self.assertTrue(all(c["data"]["bucket"] == "evaluation" for c in self.shared.list("broker_charge")))

    async def test_alternate_specialist_endpoint_fails_before_client_creation(self):
        alternate = self.config.llm.model_copy(update={"base_url": "http://forbidden.invalid/v1"})
        with patch("backend.core.orchestrator.LLMClient", side_effect=AssertionError("No direct client is allowed")):
            with self.assertRaisesRegex(ValueError, "centrally selected"):
                Orchestrator(self.llm, SkillRegistry(), OrchestratorConfig(subagent_llm=alternate))

    async def test_real_agent_dispatches_real_tool_and_persists_provider_response(self):
        roster = [{"id": "operator", "name": "Operator", "kind": "human"}, {"id": self.actor, "name": "Native fixture", "kind": "resident"}]
        def validate(lineage, author):
            return self.server.validate_completion_lineage(lineage, author) if lineage.get("completion_attempt_id") else self.server.broker.validate_message_lineage(lineage, author)
        self.conversations = Conversations(self.shared, lambda: roster, validate)
        self.grant = self.grant_user()
        opening = self.conversations.create("operator", [self.actor], "operator-input", opening="Write a small fixture file.",
            lineage={"exchange_id": self.grant["exchange_id"], "root_id": self.grant["exchange_id"], "origin": "user"})
        thread = opening["conversation"]["id"]
        snapshot = self.conversations.snapshot(self.actor, thread)
        self.grant.update(conversation_id=thread, through=snapshot["watermark"], message=opening["message"])
        self.shared.put("native_grant", self.grant["id"], {k: v for k, v in self.grant.items() if k != "id"})
        (self.root / "household.json").write_text(json.dumps({"residents": [{"resident_id": self.actor, "name": "Native fixture"}]}))
        binding = WorkerBinding("household", self.actor, self.local.meta("installation_id"), str(self.runtime.workspace.parent), "test-release")
        bridge = NativeWorkerBridge(self.runtime, binding, "/unused.sock", "first-token", self.model,
                                    transport=httpx.MockTransport(self.sync_gateway))
        self.runtime.household = bridge
        skills = SkillRegistry()
        skills.register(WriteSkill())
        companions = Companions(self.runtime, self.llm, skills)
        install_runtime_skills(self.runtime, skills, companions)
        self.runtime.workspace.joinpath("IDENTITY.md").write_text("A new native test resident.")
        self.runtime.workspace.joinpath("SOUL.md").write_text("A separate fresh identity.")
        identity = IdentityManager(str(self.runtime.workspace.parent / "data"), str(self.runtime.workspace))
        self.responses.extend([{"tool_calls": [{"id": "real-write-call", "type": "function", "function": {"name": "write", "arguments": json.dumps({"path": "result.txt", "content": "A retained effect"})}}]},
                               {"content": "The provider reports that the file is ready."}])
        with patch.object(PersonalityEngine, "STATE_FILE", str(self.runtime.workspace.parent / "data/personality.json")):
            personality = PersonalityEngine("Native fixture")
            agent = TamAGIAgent(self.config, self.llm, CanonicalMemory(self.runtime), personality, skills, identity=identity)
            agent.runtime = self.runtime
            bridge.agent = agent
            bridge.project(agent, snapshot, self.grant)
            token = current_household_admission.set(self.grant)
            try:
                async with self.runtime.execution("user"):
                    result = await agent.chat("Write a small fixture file.", conversation_id=thread,
                        _input_epoch=self.runtime.input_epoch, _admitted_message=bridge.input_record(self.grant))
            finally:
                current_household_admission.reset(token)
        self.assertEqual(result["turn_status"], "completed", result)
        self.assertEqual(result["response"], "The provider reports that the file is ready.")
        self.assertEqual(self.runtime.workspace.joinpath("result.txt").read_text(), "A retained effect")
        self.assertEqual(len(self.local.receipts()), 1)
        self.assertEqual(self.local.receipts()[0]["status"], "succeeded")
        self.assertEqual(len(self.requests), 2)
        history = self.conversations.history("operator", thread)
        self.assertEqual([m["author"] for m in history], ["operator", self.actor])
        self.assertEqual(history[-1]["content"], result["response"])


if __name__ == "__main__":
    unittest.main()
