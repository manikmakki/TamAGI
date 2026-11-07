"""Ordinary browser APIs across actual isolated native worker processes.

The provider is injected in memory. Every worker has a newly provisioned home,
uses private Unix sockets, and has no usable direct provider configuration.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

import httpx
import yaml

from backend.config import TamAGIConfig
from backend.core.household.native_app import create_household_app
from backend.core.household.native_registry import NativeRegistry
from backend.core.llm import LLMResponse, ToolCall


class DeterministicProvider:
    def __init__(self):
        self.calls = []
        self.gate, self.started = asyncio.Event(), asyncio.Event()
        self.gate.set()
        self.closed = False
        self.responses = []

    async def count_context_tokens(self, messages, tools=None):
        return None

    async def chat(self, messages, tools=None, **options):
        self.calls.append(messages)
        ordinal = len(self.calls)
        self.started.set()
        await self.gate.wait()
        if self.responses:
            return self.responses.pop(0)
        # Explicit human address prevents a fixture reply from inviting peer
        # continuations. Attribution and dispatch still use the real Agent.
        return LLMResponse(content=f"@Tester Gateway answer {ordinal}.")

    async def chat_stream(self, messages, tools=None, **options):
        yield await self.chat(messages, tools, **options)

    async def close(self):
        self.closed = True


class NativeAppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hh-app-", dir="/tmp")
        self.root = Path(self.temporary.name) / "h"
        self.registry = NativeRegistry.create(self.root, names=["Cedar", "Willow"],
            operator_name="Tester", release="deterministic-app-tests", offline=True)
        self.first, self.second = list(self.registry.descriptors)
        self.provider = DeterministicProvider()
        config = TamAGIConfig.model_validate(yaml.safe_load((self.root / "config.yaml").read_text()))
        self.app = create_household_app(config, provider=self.provider, registry=self.registry)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url="http://testserver")
        setup = await self.http.post("/api/auth/setup", json={"password": "disposable-gateway-fixture"})
        self.assertEqual(setup.status_code, 200, setup.text)

    async def asyncTearDown(self):
        self.provider.gate.set()
        if hasattr(self, "http"):
            await self.http.aclose()
        try:
            await self.lifespan.__aexit__(None, None, None)
        finally:
            self.temporary.cleanup()

    async def request(self, method, path, *, actor=None, expected=200, **options):
        headers = options.pop("headers", {})
        if actor:
            headers["X-Tamagi-Resident"] = actor
        response = await self.http.request(method, path, headers=headers, **options)
        self.assertEqual(response.status_code, expected, response.text)
        return response.json()

    async def wait_until(self, predicate, timeout=15):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(.02)

    async def completed(self, identity):
        await self.wait_until(lambda: self.app.state.interactions.get(identity)["data"]["status"]
                             not in {"queued", "running"})
        result = await self.request("GET", f"/api/chat/requests/{identity}")
        self.assertEqual(result["work"]["status"], "completed", result)
        return result

    async def test_browser_contract_uses_distinct_real_workers_and_canonical_threads(self):
        hosts = self.app.state.hosts
        self.assertNotEqual(hosts[self.first].process.pid, hosts[self.second].process.pid)
        self.assertNotEqual(hosts[self.first].ready["resident_id"], hosts[self.second].ready["resident_id"])
        for actor, name in [(self.first, "Cedar"), (self.second, "Willow")]:
            state = await self.request("GET", "/api/state", actor=actor)
            scoped = await self.request("GET", f"/r/{actor}/api/state")
            self.assertEqual(state["state"]["name"], name)
            self.assertEqual(scoped["state"]["name"], name)
            onboarding = await self.request("GET", "/api/onboarding/state", actor=actor)
            self.assertFalse(onboarding["needs_onboarding"], "Named fresh homes must open to usable chat")
        default = await self.request("GET", "/api/state")
        self.assertEqual(default["state"]["name"], "Cedar", "Header selection must not become a browser-global default")
        await self.request("GET", "/api/state", actor="unknown-resident", expected=404)
        self.assertEqual(self.provider.calls, [], "Personal browsing must not call the model")

        body = {"request_id": "private-ack-retry", "message": "A private question."}
        admitted = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202, json=body)
        replay = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202, json=body)
        self.assertEqual(admitted["work_id"], replay["work_id"])
        await self.request("POST", "/api/chat/requests", actor=self.first, expected=409,
                           json={**body, "message": "Conflicting retry."})
        private = await self.completed(admitted["work_id"])
        self.assertEqual(len(self.provider.calls), 1)
        thread_id = admitted["conversation_id"]
        history = await self.request("GET", f"/api/conversations/{thread_id}")
        self.assertEqual(history["participants"], ["operator", self.first])
        self.assertEqual([m["metadata"]["author_id"] for m in history["messages"]], ["operator", self.first])
        self.assertEqual([e["type"] for e in private["events"]].count("thread_changed"), 1)
        self.assertEqual([e["type"] for e in private["events"]].count("work_ended"), 1)
        self.assertFalse(any(e["type"] == "message" for e in private["events"]), "Child completion must not end group work")

        first_list = await self.request("GET", "/api/conversations", actor=self.first)
        second_list = await self.request("GET", "/api/conversations", actor=self.second)
        self.assertEqual([t["id"] for t in first_list["conversations"]], [thread_id])
        self.assertEqual(second_list["conversations"], [])
        self.assertEqual((await self.request("GET", "/api/household"))["unread_count"], 1)
        # Fetching any snapshot/list remains observation; only an explicit
        # foreground-visible acknowledgment advances the durable watermark.
        await self.request("GET", f"/api/conversations/{thread_id}")
        self.assertEqual((await self.request("GET", "/api/household/threads?unread=1"))["unread_count"], 1)
        await self.request("POST", f"/api/household/threads/{thread_id}/read",
                           json={"snapshot_id": history["snapshot_id"], "watermark": history["watermark"]})
        self.assertEqual((await self.request("GET", "/api/household"))["unread_count"], 0)

        group = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202,
            json={"request_id": "group-mention", "message": "@Willow Please answer this group question.",
                  "participants": [self.first, self.second], "mentions": [self.second]})
        finished = await self.completed(group["work_id"])
        self.assertEqual(finished["work"]["responders"], [self.second])
        self.assertEqual(len(self.provider.calls), 2)
        group_history = await self.request("GET", "/api/conversations/" + group["conversation_id"])
        self.assertEqual(group_history["participants"], ["operator", self.first, self.second])
        self.assertEqual([m["metadata"]["author_id"] for m in group_history["messages"]], ["operator", self.second])
        self.assertEqual(group_history["messages"][0]["metadata"]["mentions"], [self.second])
        self.assertEqual(group_history["messages"][1]["metadata"]["mentions"], ["operator"])
        colors = {"operator": "#00aa44", self.first: "#aa2211", self.second: "#55bbee"}
        await self.request("PUT", "/api/household/preferences", json={"colors": colors})
        self.assertEqual((await self.request("GET", "/api/household"))["preferences"]["colors"], colors)
        await self.request("PUT", "/api/household/preferences", expected=400, json={"colors": {self.first: "red"}})

    async def test_session_boundary_and_stopping_queued_and_running_provider_work(self):
        cookie = next(cookie for cookie in self.http.cookies.jar if cookie.name.startswith("tamagi_household_"))
        attributes = {key.lower(): value for key, value in cookie._rest.items()}
        self.assertIn("httponly", attributes)
        self.assertEqual(attributes.get("samesite"), "strict")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url="http://testserver") as stranger:
            self.assertEqual((await stranger.get("/api/household")).status_code, 401)
            self.assertEqual((await stranger.get("/")).status_code, 303)
        await self.request("POST", "/_native/conversations/commit", expected=401, json={})
        await self.request("GET", f"/r/{self.first}/_native/ready", expected=404)
        await self.request("POST", f"/r/{self.first}/_native/respond", expected=404, json={})
        await self.request("PUT", "/api/household/preferences", expected=403,
                           headers={"Origin": "https://another.invalid"}, json={"colors": {}})

        self.provider.gate.clear()
        running = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202,
                                    json={"request_id": "running-stop", "message": "Wait for running Stop."})
        await asyncio.wait_for(self.provider.started.wait(), 10)
        queued = await self.request("POST", "/api/chat/requests", actor=self.second, expected=202,
                                   json={"request_id": "queued-stop", "message": "Wait in the model queue."})
        await self.wait_until(lambda: len(self.registry.store.list("broker_attempt")) >= 2)
        self.assertEqual(len(self.provider.calls), 1, "The coordinator permits only one upstream call")
        for work in [queued, running]:
            stopped = await self.request("POST", f"/api/chat/requests/{work['work_id']}/stop")
            self.assertEqual(stopped["status"], "stopped")
        self.provider.gate.set()
        await self.wait_until(lambda: all(task.done() for task in self.app.state.model._tasks.values()))
        self.assertEqual(len(self.provider.calls), 1, "Stopped queued work must never reach the provider")
        for work in [queued, running]:
            snapshot = await self.request("GET", f"/api/chat/requests/{work['work_id']}")
            self.assertEqual(snapshot["work"]["status"], "stopped")
            history = await self.request("GET", "/api/conversations/" + work["conversation_id"])
            self.assertEqual([m["role"] for m in history["messages"]], ["user"], "Stop blocks late canonical publication")
        self.assertEqual((await self.request("GET", "/api/household"))["unread_count"], 0)

    async def test_lost_acceptance_acknowledgment_revokes_authority_before_reporting_failure(self):
        self.provider.gate.clear()
        host = self.app.state.hosts[self.first]
        original = host.json
        async def lose_ack(method, path, body=None):
            result = await original(method, path, body)
            if path == "/_native/respond":
                await self.provider.started.wait()
                raise RuntimeError("Simulated lost acceptance acknowledgment")
            return result
        host.json = lose_ack
        work = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202,
            json={"request_id": "lost-ack", "message": "Keep a durable record of this input."})
        await self.wait_until(lambda: self.app.state.interactions.get(work["work_id"])["data"]["status"] == "failed")
        grants = [r for r in self.registry.store.list("native_grant") if r["data"].get("work_id") == work["work_id"]]
        self.assertEqual([r["data"]["status"] for r in grants], ["stopped"])
        execution = self.registry.store.list("native_execution")[0]["data"]
        self.assertEqual(self.app.state.model.broker.execution(execution["broker_execution_id"])["status"], "stopped")
        self.provider.gate.set()
        await self.wait_until(lambda: bool(self.registry.store.list("native_model_result")))
        self.assertFalse(self.registry.store.list("native_model_result")[0]["data"]["delivery_permitted"])
        history = await self.request("GET", "/api/conversations/" + work["conversation_id"])
        self.assertEqual([m["role"] for m in history["messages"]], ["user"])

    async def test_optional_peer_waits_for_chosen_event_then_uses_the_same_exchange(self):
        self.provider.responses = [
            LLMResponse(tool_calls=[ToolCall("choose-wait", "end_turn", {"status": "waiting", "wake_event": "notebook_changed"})]),
            LLMResponse(content="@Willow Could you look at this when you are ready?"),
            LLMResponse(content="@Tester I have looked at the shared question."),
        ]
        rest = await self.request("POST", "/api/chat/requests", actor=self.second, expected=202,
            json={"request_id": "chosen-wait", "message": "You may choose whether to wait."})
        await self.completed(rest["work_id"])
        sleeping = await self.app.state.hosts[self.second].json("GET", "/_native/ready")
        self.assertEqual(sleeping["activity"]["status"], "waiting")
        self.assertFalse(sleeping["due"])
        silent_history = await self.request("GET", "/api/conversations/" + rest["conversation_id"])
        self.assertEqual([m["role"] for m in silent_history["messages"]], ["user"])
        group = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202,
            json={"request_id": "deferred-peer", "message": "@Cedar Please start this group discussion.",
                  "participants": [self.first, self.second], "mentions": [self.first]})
        await self.completed(group["work_id"])
        self.assertEqual(len(self.provider.calls), 2, "A peer message must not force chosen waiting")
        deferred = [r for r in self.registry.store.list("reply_offer") if r["data"]["status"] == "deferred"]
        self.assertEqual([r["data"]["recipient"] for r in deferred], [self.second])
        await self.request("POST", "/api/runtime/notebook", json={"text": "A shared note and an actual notebook event."})
        await self.wait_until(lambda: len(self.provider.calls) == 3)
        await self.wait_until(lambda: len(self.registry.store.list("native_work")) == 3
            and all(r["data"]["status"] not in {"queued", "running"} for r in self.registry.store.list("native_work")))
        history = await self.request("GET", "/api/conversations/" + group["conversation_id"])
        self.assertEqual([m["metadata"]["author_id"] for m in history["messages"]], ["operator", self.first, self.second])
        resumed = [g["data"] for g in self.registry.store.list("native_grant")
                   if g["data"].get("conversation_id") == group["conversation_id"] and g["data"]["resident_id"] == self.second]
        self.assertEqual(len(resumed), 1)
        self.assertEqual(resumed[0]["exchange_id"], group["exchange_id"])
        self.assertFalse(resumed[0]["first_response"], "A peer message must not gain operator priority")
        snapshot = await self.request("GET", "/api/runtime", actor=self.second)
        self.assertEqual([r["data"]["text"] for r in snapshot["notebook"]], ["A shared note and an actual notebook event."])

    async def test_real_contact_tool_commits_a_new_thread_without_new_user_allowance(self):
        self.provider.responses = [
            LLMResponse(tool_calls=[ToolCall("initiate-contact", "contact", {"participants": ["operator"],
                "message": "I have something I chose to ask you about.", "title": "A resident opening"})]),
            LLMResponse(content="@Tester I created the conversation."),
        ]
        original = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202,
            json={"request_id": "actual-contact", "message": "Try your contact tool if you wish."})
        await self.completed(original["work_id"])
        await self.wait_until(lambda: all(task.done() for task in self.app.state.interactions.tasks.values()))
        threads = await self.request("GET", "/api/conversations", actor=self.first)
        self.assertEqual(len(threads["conversations"]), 2)
        authored = next(t for t in threads["conversations"] if t["id"] != original["conversation_id"])
        history = await self.request("GET", "/api/conversations/" + authored["id"])
        self.assertEqual(history["messages"][0]["metadata"]["author_id"], self.first)
        self.assertEqual(history["messages"][0]["content"], "I have something I chose to ask you about.")
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual({r["data"]["exchange_id"] for r in self.registry.store.list("native_work")}, {original["exchange_id"]})
        self.assertEqual(len(self.registry.store.list("broker_exchange")), 1)


if __name__ == "__main__":
    unittest.main()
