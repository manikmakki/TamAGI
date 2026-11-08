"""Ordinary browser APIs across actual isolated native worker processes.

The provider is injected in memory. Every worker has a newly provisioned home,
uses private Unix sockets, and has no usable direct provider configuration.
"""
from __future__ import annotations

import asyncio
import base64
import struct
import zlib
from pathlib import Path
import tempfile
import unittest

import httpx
import yaml

from backend.config import TamAGIConfig
from backend.core.household.native_app import create_household_app
from backend.core.household.native_registry import NativeRegistry
from backend.core.household.contracts import AccessDenied, NotFound
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
        boundary = next(e for e in private["events"] if e["type"] == "thread_changed")
        self.assertEqual(history["messages"][1]["metadata"]["household_grant_id"], boundary["grant_id"])
        self.assertTrue(boundary["resident_work_id"])
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

    async def test_group_progress_and_publications_keep_each_resident_activation_identity(self):
        self.provider.responses = [
            LLMResponse(content="**Cedar checks** `WORKSPACE.md`.", tool_calls=[ToolCall("cedar-read", "read", {"path": "WORKSPACE.md"})]),
            LLMResponse(content="@Tester Cedar finished."),
            LLMResponse(content="**Willow checks** `WORKSPACE.md`.", tool_calls=[ToolCall("willow-read", "read", {"path": "WORKSPACE.md"})]),
            LLMResponse(content="@Cedar Willow finished; would you check once more?"),
            LLMResponse(content="**Cedar rechecks** `WORKSPACE.md`.", tool_calls=[ToolCall("cedar-read-again", "read", {"path": "WORKSPACE.md"})]),
            LLMResponse(content="@Tester Cedar finished the second activation."),
        ]
        work = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202,
            json={"request_id": "resident-progress", "message": "Both residents may inspect their home.",
                  "participants": [self.first, self.second]})
        done = await self.completed(work["work_id"])
        events = done["events"]
        boundaries = [event for event in events if event["type"] == "thread_changed"]
        self.assertEqual([event["resident_id"] for event in boundaries], [self.first, self.second, self.first])
        self.assertEqual(len({event["grant_id"] for event in boundaries}), 3)
        self.assertEqual(len({event["resident_work_id"] for event in boundaries}), 3)
        for boundary in boundaries:
            group = [event for event in events if event.get("grant_id") == boundary["grant_id"]]
            self.assertEqual({event["resident_id"] for event in group}, {boundary["resident_id"]})
            self.assertEqual({event["resident_work_id"] for event in group}, {boundary["resident_work_id"]})
            self.assertTrue(any(event["type"] == "interim_text" for event in group))
            self.assertEqual(group[-1]["type"], "thread_changed")
        history = await self.request("GET", "/api/conversations/" + work["conversation_id"])
        answers = history["messages"][1:]
        self.assertEqual([message["metadata"]["household_grant_id"] for message in answers], [event["grant_id"] for event in boundaries])
        self.assertEqual([message["metadata"]["interim_messages"] for message in answers],
                         [["**Cedar checks** `WORKSPACE.md`."], ["**Willow checks** `WORKSPACE.md`."], ["**Cedar rechecks** `WORKSPACE.md`."]])

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

    async def test_group_image_survives_peer_trigger_and_later_turns(self):
        # Above the old 8 MiB wire limit after base64 expansion, but within the
        # advertised 8 MiB decoded attachment limit. No upstream model is used.
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        content = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1024, 6200, 8, 0, 0, 0, 0))
                   + chunk(b"IDAT", zlib.compress((b"\x00" * 1025) * 6200, level=0)) + chunk(b"IEND", b""))
        image = base64.b64encode(content).decode()
        self.provider.responses = [LLMResponse(content="@Willow Please inspect the same picture."),
                                   LLMResponse(content="@Tester I received the picture too.")]
        body = {"request_id": "group-image", "message": "What is in this image?",
                "participants": [self.first, self.second], "image_data": image, "image_media_type": "image/png"}
        work = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202, json=body)
        await self.completed(work["work_id"])
        history = await self.request("GET", "/api/conversations/" + work["conversation_id"])
        original = history["messages"][0]
        self.assertEqual(original["metadata"]["image"]["size"], len(content))
        self.assertEqual(len(self.registry.store.list("conversation_attachment")), 1)
        self.assertEqual(len(self.provider.calls), 2)
        for call in self.provider.calls:
            images = [block for m in call if isinstance(m.content, list) for block in m.content if block["type"] == "image_url"]
            self.assertEqual([b["image_url"]["url"] for b in images], ["data:image/png;base64," + image])
        grants = [r["data"] for r in self.registry.store.list("native_grant")]
        self.assertEqual(grants[1]["message"]["author"], self.first, "Second worker is triggered by a peer, not synthetic user input")
        url = f"/api/conversations/{work['conversation_id']}/attachments/{original['id']}"
        response = await self.http.get(url)
        self.assertEqual(response.content, content)
        self.assertEqual(response.headers["content-type"], "image/png")
        with self.assertRaises(AccessDenied):
            self.app.state.interactions.attachment("outsider", work["conversation_id"], original["id"])
        with self.assertRaises(NotFound):
            self.app.state.interactions.attachment(self.first, work["conversation_id"], original["id"], through=0)
        await self.request("GET", f"/_native/conversations/{work['conversation_id']}/attachments/{original['id']}?grant_id=anything", expected=401)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url="http://testserver") as stranger:
            self.assertEqual((await stranger.get(url)).status_code, 401)
        # A later image-free request still sees the original message's bytes.
        follow = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202,
            json={"request_id": "image-followup", "message": "@Cedar Look at that picture again.",
                  "conversation_id": work["conversation_id"], "mentions": [self.first]})
        await self.completed(follow["work_id"])
        self.assertTrue(any(isinstance(m.content, list) for m in self.provider.calls[-1]))

    async def test_invalid_image_is_rejected_before_input_or_exchange_is_created(self):
        for image, media in [("not base64", "image/png"), ("", "image/png"), ("eA==", "text/html")]:
            await self.request("POST", "/api/chat/requests", actor=self.first, expected=400,
                json={"request_id": "invalid-image", "message": "Picture", "image_data": image, "image_media_type": media})
        self.assertFalse(self.registry.store.list("conversation"))
        self.assertFalse(self.registry.store.list("native_work"))
        self.assertFalse(self.registry.store.list("broker_exchange"))

    async def test_real_worker_creations_and_attachment_survive_household_restart(self):
        self.provider.responses = [
            LLMResponse(tool_calls=[ToolCall("save-place", "world_action", {
                "action": "create", "attributes": {"kind": "place", "name": "Stored Garden", "description": "A saved shared place."}}),
                ToolCall("save-file", "write", {"path": "workshop/dice-note.txt", "content": "Saved dice experiment."})]),
            LLMResponse(content="@Tester Saved the garden and my private file."),
        ]
        pixel = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jN2kAAAAASUVORK5CYII=")
        work = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202,
            json={"request_id": "restart-creations", "message": "@Cedar Save a place and a note; here is a picture.",
                  "participants": [self.first, self.second], "mentions": [self.first],
                  "image_data": base64.b64encode(pixel).decode(), "image_media_type": "image/png"})
        await self.completed(work["work_id"])
        world = [r for r in self.registry.store.list("world_object") if r["data"]["name"] == "Stored Garden"]
        self.assertEqual(len(world), 1)
        saved_id, revision = world[0]["id"], world[0]["revision"]
        private = Path(self.registry.descriptors[self.first]["root"]) / "workspace/workshop/dice-note.txt"
        self.assertEqual(private.read_text(), "Saved dice experiment.")
        other = Path(self.registry.descriptors[self.second]["root"]) / "workspace/workshop/dice-note.txt"
        self.assertFalse(other.exists(), "Private files are not silently shared")
        await self.http.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.registry = NativeRegistry(self.root)
        self.provider = DeterministicProvider()
        config = TamAGIConfig.model_validate(yaml.safe_load((self.root / "config.yaml").read_text()))
        self.app = create_household_app(config, provider=self.provider, registry=self.registry)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url="http://testserver")
        await self.request("POST", "/api/auth/login", json={"password": "disposable-gateway-fixture"})
        self.assertEqual(self.registry.store.get("world_object", saved_id)["revision"], revision)
        self.assertEqual(private.read_text(), "Saved dice experiment.")
        history = await self.request("GET", "/api/conversations/" + work["conversation_id"])
        message_id = history["messages"][0]["id"]
        response = await self.http.get(f"/api/conversations/{work['conversation_id']}/attachments/{message_id}")
        self.assertEqual(response.content, pixel)
        self.provider.responses = [LLMResponse(tool_calls=[ToolCall("read-place", "query_world_graph", {
            "node_type": "location", "query": "Stored Garden"})]), LLMResponse(content="@Tester I can read the saved place.")]
        follow = await self.request("POST", "/api/chat/requests", actor=self.second, expected=202,
            json={"request_id": "restart-readback", "conversation_id": work["conversation_id"],
                  "message": "@Willow Read the saved place.", "mentions": [self.second]})
        await self.completed(follow["work_id"])
        tools = [m for m in self.provider.calls[-1] if m.role == "tool"]
        self.assertTrue(any("Stored Garden" in str(m.content) and saved_id in str(m.content) for m in tools))
        self.assertTrue(any(isinstance(m.content, list) for m in self.provider.calls[0]), "Reopened worker receives the original image")

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
