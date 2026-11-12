"""World Thread rest leaves household conversation available without waking it."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest

import test_household_native_app as fixture
from backend.config import RuntimeConfig
from backend.core.llm import LLMResponse, ToolCall
from backend.core.runtime import RuntimeService, RuntimeStore


class ChatReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="chat-rest-")
        self.root = Path(self.temporary.name)
        self.now = 100000.0
        self.store = RuntimeStore(self.root / "runtime.sqlite3", clock=lambda: self.now)
        self.runtime = RuntimeService(self.store, self.root / "workspace", RuntimeConfig(), self.root)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_ordinary_and_legacy_rest_are_ignored_without_changing_the_deadline(self):
        self.runtime.end_turn("resting", wake_at=self.now + 3600)
        for reason in ("chosen_rest", None):
            with self.subTest(wait_reason=reason):
                record = self.store.get("activity", self.runtime.actor_id)
                self.store.put("activity", record["id"], record["actor"], {**record["data"], "wait_reason": reason})
                before = self.store.get("activity", self.runtime.actor_id)
                writes = self.store.db.total_changes
                self.assertFalse(self.runtime.due())
                self.assertTrue(self.runtime.due(ignore_rest=True))
                self.assertEqual(self.store.get("activity", self.runtime.actor_id), before)
                self.assertEqual(self.store.db.total_changes, writes)

    def test_explicit_waits_and_failure_backoff_still_defer_chat(self):
        for status, wake_at, wake_event in (
            ("waiting", self.now + 3600, None),
            ("waiting", None, "notebook_changed"),
            ("waiting", self.now + 3600, "evaluation_completed"),
            ("resting", self.now + 3600, "notebook_changed"),
        ):
            with self.subTest(status=status, wake_event=wake_event):
                self.runtime.end_turn(status, wake_at=wake_at, wake_event=wake_event)
                self.assertFalse(self.runtime.due(ignore_rest=True, ignore_decompression=True))
        for reason in ("context_limit", "request_rejected", "provider_unavailable", "runtime_unavailable", "generation_truncated"):
            with self.subTest(wait_reason=reason):
                record = self.store.get("activity", self.runtime.actor_id)
                self.store.put("activity", record["id"], record["actor"], {
                    **record["data"], "status": "waiting", "wake_event": None,
                    "wake_at": self.now + 3600, "wait_reason": reason,
                })
                self.assertFalse(self.runtime.due(ignore_rest=True, ignore_decompression=True))

    def test_pending_work_and_decompression_keep_their_own_gates(self):
        self.runtime.end_turn("resting", wake_at=self.now + 3600)
        self.store.set_meta("decompression_until", self.now + 100)
        self.assertFalse(self.runtime.due(ignore_rest=True))
        self.assertTrue(self.runtime.due(ignore_rest=True, ignore_decompression=True))
        self.runtime.interactions = SimpleNamespace(has_pending=lambda **options: True)
        self.assertFalse(self.runtime.due(ignore_rest=True, ignore_decompression=True))


class NativeChatRestTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.NativeAppTests.asyncSetUp
    asyncTearDown = fixture.NativeAppTests.asyncTearDown
    request = fixture.NativeAppTests.request
    wait_until = fixture.NativeAppTests.wait_until
    completed = fixture.NativeAppTests.completed

    def private_records(self, actor, kind):
        path = Path(self.registry.descriptors[actor]["root"]) / "runtime.sqlite3"
        with closing(sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)) as database:
            return [json.loads(row[0]) for row in database.execute(
                "SELECT payload FROM records WHERE kind=? ORDER BY created,id", (kind,))]

    async def rest(self, actor):
        self.provider.responses = [LLMResponse(tool_calls=[ToolCall("choose-rest", "end_turn", {
            "status": "resting", "wake_at": time.time() + 3600,
        })])]
        work = await self.request("POST", "/api/chat/requests", actor=actor, expected=202,
            json={"request_id": "choose-rest", "message": "Choose whether to rest."})
        await self.completed(work["work_id"])
        ready = await self.app.state.hosts[actor].json("GET", "/_native/ready")
        self.assertEqual(ready["activity"]["wait_reason"], "chosen_rest")
        self.assertTrue(ready["due"], "Chosen rest affects World Thread, not household chat")
        self.assertFalse(ready["world_due"])
        return ready["activity"]

    async def test_five_peer_replies_share_one_exchange_and_preserve_world_rest(self):
        resting = await self.rest(self.first)
        self.provider.responses = [
            LLMResponse(content="@Cedar What would you like to build?"),
            LLMResponse(content="@Willow A small garden. What should we plant?"),
            LLMResponse(content="@Cedar How about herbs?"),
            LLMResponse(content="@Willow Which herbs would you pick?"),
            LLMResponse(content="@Tester We have chosen rosemary together."),
        ]
        group = await self.request("POST", "/api/chat/requests", actor=self.second, expected=202,
            json={"request_id": "peer-rest-discussion", "message": "@Willow Please open this discussion.",
                  "participants": [self.first, self.second], "mentions": [self.second]})
        await self.completed(group["work_id"])
        history = await self.request("GET", "/api/conversations/" + group["conversation_id"])
        self.assertEqual([message["metadata"]["author_id"] for message in history["messages"]],
                         ["operator", self.second, self.first, self.second, self.first, self.second])
        grants = [record["data"] for record in self.registry.store.list("native_grant")
                  if record["data"]["conversation_id"] == group["conversation_id"]]
        self.assertEqual(len(grants), 5)
        self.assertEqual({grant["exchange_id"] for grant in grants}, {group["exchange_id"]})
        self.assertEqual(sum(grant["first_response"] for grant in grants), 1)
        exchange = self.app.state.model.broker.exchange(group["exchange_id"])
        self.assertEqual((exchange["calls"], exchange["segment_calls"], exchange["continuations_used"]), (5, 5, 0))
        self.assertEqual(len(self.registry.store.list("broker_exchange")), 2)
        self.assertEqual(len(self.provider.calls), 6)
        self.assertEqual(self.provider.responses, [])
        ready = await self.app.state.hosts[self.first].json("GET", "/_native/ready")
        self.assertEqual(ready["activity"], resting, "Peer replies must neither clear nor extend the World Thread rest")
        self.assertTrue(ready["due"])
        self.assertFalse(ready["world_due"])
        self.assertEqual([record["input_epoch"] for record in self.private_records(self.first, "interaction")], [1, 1, 1])

    async def test_invitation_and_joined_reply_do_not_consume_world_rest(self):
        resting = await self.rest(self.first)
        opened = await self.request("POST", "/api/chat/requests", actor=self.second, expected=202,
            json={"request_id": "private-before-invitation", "message": "A conversation with Willow."})
        await self.completed(opened["work_id"])
        self.provider.responses = [
            LLMResponse(tool_calls=[ToolCall("invite-cedar", "contact", {
                "action": "invite", "participants": [self.first], "message": "Cedar, would you join us?",
            })]),
            LLMResponse(content="@Tester I sent the invitation."),
            LLMResponse(tool_calls=[ToolCall("accept-invitation", "contact", {"action": "respond", "decision": "accept"})]),
            LLMResponse(content="@Tester I joined the conversation."),
        ]
        invited = await self.request("POST", "/api/chat/requests", actor=self.second, expected=202,
            json={"request_id": "invite-resting-resident", "conversation_id": opened["conversation_id"],
                  "message": "Please invite Cedar here."})
        await self.completed(invited["work_id"])
        history = await self.request("GET", "/api/conversations/" + opened["conversation_id"])
        self.assertIn(self.first, history["participants"])
        self.assertEqual(history["invitations"][0]["status"], "joined")
        self.assertEqual(history["messages"][-1]["metadata"]["author_id"], self.first)
        self.assertEqual(self.provider.responses, [])
        ready = await self.app.state.hosts[self.first].json("GET", "/_native/ready")
        self.assertEqual(ready["activity"], resting)
        self.assertFalse(ready["world_due"])
        grants = [record["data"] for record in self.registry.store.list("native_grant")
                  if record["data"]["resident_id"] == self.first and record["data"]["exchange_id"] == invited["exchange_id"]]
        self.assertEqual(len(grants), 2)
        self.assertTrue(any(grant.get("kind") == "invitation" for grant in grants))
        self.assertTrue(all(not grant["first_response"] for grant in grants))
