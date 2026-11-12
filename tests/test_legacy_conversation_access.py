"""Exact private transcript access must not create current household chat state."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from backend.core.agent import Conversation, Message, TamAGIAgent
from backend.core.household.contracts import AccessDenied, ProtocolError, WorkerBinding
from backend.core.household.worker_bridge import NativeWorkerBridge, admission_context
from backend.core.memory import MemoryEntry, MemoryType
from backend.core.runtime import ExecutionContext, RuntimeService, RuntimeStore, current_execution, current_runtime
from backend.core.runtime.memory import CanonicalMemory


class LegacyConversationAccessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="legacy-memory-access-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name) / "h"
        self.root = self.home / "residents/r0"
        self.root.mkdir(parents=True)
        self.store = RuntimeStore(self.root / "runtime.sqlite3", clock=lambda: 1700000000)
        self.addCleanup(self.store.close)
        self.actor = self.store.resident_id
        self.runtime = SimpleNamespace(store=self.store, actor_id=self.actor, workspace=self.root / "workspace")
        self.runtime.scope = MethodType(RuntimeService.scope, self.runtime)
        (self.home / "household.json").write_text(json.dumps({
            "operator": {"name": "Operator"},
            "residents": [{"resident_id": self.actor, "name": "Ember"}],
        }))
        self.requests = []
        self.response_status = 404
        self.response_body = {"detail": "Unknown conversation"}
        self.transport_error = None
        binding = WorkerBinding("household-test", self.actor, "installation-test", str(self.root), "test-release")
        self.bridge = NativeWorkerBridge(self.runtime, binding, self.home / "ipc/c.sock", "test-worker-token",
            SimpleNamespace(), transport=httpx.MockTransport(self.respond))
        self.addCleanup(self.bridge.close)
        self.runtime.household = self.bridge
        token = current_runtime.set(self.runtime)
        self.addCleanup(current_runtime.reset, token)
        token = current_execution.set(None)
        self.addCleanup(current_execution.reset, token)
        token = admission_context().set(None)
        self.addCleanup(admission_context().reset, token)
        self.conversation = Conversation(id="old-thread", title="Earlier conversation", created_at=100, updated_at=200,
            world_summarized=True, memory_archived_through=2, world_context_initialized=True,
            world_context_summary="Retained summary", world_context_turn_count=3, world_context_latest_turn_id="turn-3",
            messages=[Message(role="user" if index % 2 == 0 else "assistant", content=f"Earlier message {index}",
                id=f"old-message-{index}", timestamp=100 + index, metadata={"original": index}) for index in range(6)])
        self.agent = TamAGIAgent.__new__(TamAGIAgent)
        self.agent.runtime = self.runtime
        self.agent.memory = CanonicalMemory(self.runtime)
        self.agent.personality = SimpleNamespace(name="Ember")
        self.agent.relational_consolidator = Mock()
        active = Conversation(id="current-thread", messages=[Message("user", "Current conversation", timestamp=300)])
        self.agent.conversations = {active.id: active}

    def respond(self, request):
        self.requests.append(request)
        if self.transport_error is not None:
            raise self.transport_error
        return httpx.Response(self.response_status, json=self.response_body)

    def seed_conversation(self, *, actor=None, payload=None):
        self.store.put("conversation", self.conversation.id, actor or self.actor,
            payload if payload is not None else self.conversation.to_dict())

    def seed_memory(self, *, actor=None, conversation_id="old-thread", anchor=2):
        entry = MemoryEntry(id="old-memory", content="Retained memory window", memory_type=MemoryType.CONVERSATION,
            timestamp=102, metadata={"conversation_id": conversation_id,
                "conversation_title": "Earlier conversation", "anchor_message_index": anchor})
        self.store.put("memory", entry.id, actor or self.actor, entry.to_dict())
        return entry

    def database_dump(self):
        return tuple(self.store.db.iterdump())

    def canonical_response(self):
        self.response_status = 200
        self.response_body = {"conversation": {"id": self.conversation.id, "title": "Canonical conversation"},
            "messages": [{"id": "canonical-message", "conversation_id": self.conversation.id,
                "author": self.actor, "content": "Canonical response", "timestamp": 150, "sequence": 1,
                "participants": ["operator", self.actor], "lineage": {}, "metadata": {}}], "watermark": 1}

    def test_private_lookup_preserves_original_fields_without_mutation(self):
        self.seed_conversation()
        before = self.database_dump()
        changes = self.store.db.total_changes

        result = self.bridge.fetch_memory_conversation(self.conversation.id)

        self.assertEqual(result.to_dict(), self.conversation.to_dict())
        self.assertEqual(self.database_dump(), before)
        self.assertEqual(self.store.db.total_changes, changes)
        self.assertEqual(len(self.requests), 1)

    def test_expansion_uses_anchor_and_does_not_change_current_dialogue(self):
        self.seed_conversation()
        self.seed_memory()
        before = self.database_dump()
        cache = dict(self.agent.conversations)
        dialogue = self.agent.recent_dialogue_text()
        self.agent.get_conversation = Mock(side_effect=AssertionError("Historical expansion must use its own lookup"))

        result = asyncio.run(self.agent.expand_conversation_memory("old-memory", messages_before=1, messages_after=1))

        self.assertEqual(result["messages"], [message.to_dict() for message in self.conversation.messages[1:4]])
        self.assertEqual(result["stored_window"], "Retained memory window")
        self.assertEqual(result["conversation_id"], self.conversation.id)
        self.assertEqual(self.agent.conversations, cache)
        self.assertEqual(self.agent.recent_dialogue_text(), dialogue)
        self.assertEqual(self.agent.relational_consolidator.mock_calls, [])
        self.assertEqual(self.database_dump(), before)

    def test_canonical_transcript_wins_and_preserves_admission_watermark(self):
        self.seed_conversation()
        self.canonical_response()
        token = admission_context().set({"id": "current-grant", "conversation_id": self.conversation.id, "through": 1})
        try:
            with patch.object(self.store, "get", side_effect=AssertionError("No private fallback for canonical history")):
                result = self.bridge.fetch_memory_conversation(self.conversation.id)
        finally:
            admission_context().reset(token)

        self.assertEqual([message.id for message in result.messages], ["canonical-message"])
        self.assertTrue(result.messages[0].metadata["canonical"])
        self.assertEqual(dict(self.requests[0].url.params), {"grant_id": "current-grant", "through": "1"})

    def test_agent_expansion_also_uses_canonical_history_before_legacy(self):
        self.seed_conversation()
        self.seed_memory(anchor=0)
        self.canonical_response()

        result = asyncio.run(self.agent.expand_conversation_memory("old-memory"))

        self.assertEqual([message["id"] for message in result["messages"]], ["canonical-message"])
        self.assertNotIn(self.conversation.id, self.agent.conversations)

    def test_missing_transcript_retains_stored_memory_window(self):
        self.seed_memory()
        before = self.database_dump()

        result = asyncio.run(self.agent.expand_conversation_memory("old-memory"))

        self.assertEqual(result["messages"], [])
        self.assertEqual(result["stored_window"], "Retained memory window")
        self.assertEqual(self.database_dump(), before)

    def test_memory_without_conversation_id_needs_no_transport(self):
        self.seed_memory(conversation_id="")

        result = asyncio.run(self.agent.expand_conversation_memory("old-memory"))

        self.assertEqual(result["messages"], [])
        self.assertEqual(self.requests, [])

    def test_other_actor_memory_is_not_exposed(self):
        self.seed_conversation()
        self.seed_memory(actor="companion-private")

        self.assertIsNone(asyncio.run(self.agent.expand_conversation_memory("old-memory")))
        self.assertEqual(self.requests, [])

    def test_other_actor_transcript_is_not_exposed(self):
        self.seed_conversation(actor="companion-private")

        with self.assertRaises(AccessDenied):
            self.bridge.fetch_memory_conversation(self.conversation.id)

    def test_companion_scope_is_denied_before_canonical_or_private_lookup(self):
        self.seed_conversation()
        self.canonical_response()
        token = current_execution.set(ExecutionContext("companion-execution", "companion-private", "user", depth=1))
        try:
            with patch.object(self.store, "get", side_effect=AssertionError("No private lookup for companions")):
                with self.assertRaises(AccessDenied):
                    self.bridge.fetch_memory_conversation(self.conversation.id)
        finally:
            current_execution.reset(token)
        self.assertEqual(self.requests, [])

    def test_companion_thread_is_not_a_legacy_conversation(self):
        self.store.put("companion_thread", self.conversation.id, self.actor,
            {"participants": [self.actor, "companion-private"], "messages": self.conversation.to_dict()["messages"]})

        self.assertIsNone(self.bridge.fetch_memory_conversation(self.conversation.id))

    def test_access_denial_protocol_failure_and_transport_error_never_fall_back(self):
        self.seed_conversation()
        failures = [(403, AccessDenied), (400, ProtocolError), (409, ValueError)]
        for status, exception in failures:
            with self.subTest(status=status):
                self.response_status = status
                self.response_body = {"detail": "Canonical access failed"}
                with patch.object(self.store, "get", side_effect=AssertionError("Canonical errors must not fall back")):
                    with self.assertRaises(exception):
                        self.bridge.fetch_memory_conversation(self.conversation.id)
        self.transport_error = httpx.ConnectError("Coordinator unavailable")
        with patch.object(self.store, "get", side_effect=AssertionError("Transport errors must not fall back")):
            with self.assertRaises(httpx.ConnectError):
                self.bridge.fetch_memory_conversation(self.conversation.id)

    def test_payload_identity_mismatch_is_rejected(self):
        self.seed_conversation(payload={**self.conversation.to_dict(), "id": "another-thread"})

        with self.assertRaises(ProtocolError):
            self.bridge.fetch_memory_conversation(self.conversation.id)

    def test_malformed_transcripts_fail_without_mutating_evidence(self):
        original = self.conversation.to_dict()
        message = original["messages"][0]
        malformed = [
            {**original, "messages": {}},
            {**original, "messages": [{key: value for key, value in message.items() if key != "id"}]},
            {**original, "messages": [{**message, "role": 1}]},
            {**original, "messages": [{**message, "content": {"unexpected": "object"}}]},
            {**original, "messages": [{**message, "timestamp": "yesterday"}]},
            {**original, "messages": [{**message, "timestamp": float("nan")}]},
            {**original, "messages": [{**message, "timestamp": float("inf")}]},
            {**original, "messages": [{**message, "metadata": []}]},
        ]
        for payload in malformed:
            with self.subTest(payload=payload):
                self.seed_conversation(payload=payload)
                before = self.database_dump()
                changes = self.store.db.total_changes
                with self.assertRaises(ProtocolError):
                    self.bridge.fetch_memory_conversation(self.conversation.id)
                self.assertEqual(self.database_dump(), before)
                self.assertEqual(self.store.db.total_changes, changes)

    def test_standalone_memory_expansion_keeps_existing_conversation_behavior(self):
        self.runtime.household = None
        self.seed_memory()
        self.agent.conversations[self.conversation.id] = self.conversation

        result = asyncio.run(self.agent.expand_conversation_memory("old-memory", messages_before=0, messages_after=0))

        self.assertEqual(result["messages"], [self.conversation.messages[2].to_dict()])
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
