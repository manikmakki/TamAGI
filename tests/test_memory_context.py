from __future__ import annotations

import asyncio
import time
import unittest

from backend.config import ElasticsearchConfig, MemoryContextConfig
from backend.core.agent import (
    Conversation,
    Message,
    TamAGIAgent,
    _format_relevant_memory_context,
)
from backend.core.memory import MemoryEntry, MemoryType
from backend.core.memory_elasticsearch import (
    ElasticsearchMemoryStore,
    _index_mapping,
    _rank_conversation_hits,
)
from backend.skills.expand_conversation_memory_skill import ExpandConversationMemorySkill


class RecordingEmbedder:
    dim = 3

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str) -> list[float]:
        self.encoded.append(text)
        return [0.1, 0.2, 0.3]


class RecordingClient:
    def __init__(self) -> None:
        self.index_call: dict | None = None

    async def index(self, **kwargs):
        self.index_call = kwargs


class SearchClient:
    def __init__(self, hit: dict) -> None:
        self.hit = hit
        self.bodies: list[dict] = []

    async def search(self, *, index, body):
        self.bodies.append(body)
        return {"hits": {"hits": [self.hit]}}


def make_hit(
    memory_id: str,
    conversation_id: str,
    score: float,
    *,
    timestamp: float,
    title: str = "Conversation",
) -> dict:
    return {
        "_id": memory_id,
        "_score": score,
        "_source": {
            "occurred_at": timestamp,
            "data": {
                "type": "conversation",
                "content": f"User: content for {memory_id}",
                "references": [],
            },
            "meta": {
                "document_kind": "turn_window",
                "conversation_id": conversation_id,
                "conversation_title": title,
            },
        },
    }


class MemoryContextTests(unittest.TestCase):
    def test_fresh_mapping_has_turn_lookup_and_basic_vector_fields(self) -> None:
        mapping = _index_mapping(384)["mappings"]["properties"]
        self.assertEqual(mapping["data"]["properties"]["content_vector"]["dims"], 384)
        self.assertEqual(mapping["meta"]["properties"]["conversation_id"]["type"], "keyword")
        self.assertEqual(mapping["meta"]["properties"]["turn_index"]["type"], "integer")
        self.assertIn("occurred_at", mapping)

    def test_store_uses_memory_identity_and_search_text_for_embedding(self) -> None:
        store = ElasticsearchMemoryStore(ElasticsearchConfig())
        store._embedder = RecordingEmbedder()
        store._client = RecordingClient()
        entry = MemoryEntry(
            id="conversation-window:conv-1:2",
            content="User: full transcript",
            memory_type=MemoryType.CONVERSATION,
            timestamp=1_700_000_000,
            metadata={
                "conversation_id": "conv-1",
                "document_kind": "turn_window",
                "search_text": "concise retrieval representation",
                "source": "chat",
            },
        )

        result = asyncio.run(store.store(entry))

        self.assertEqual(result, entry.id)
        self.assertEqual(store._client.index_call["id"], entry.id)
        self.assertEqual(store._embedder.encoded, ["concise retrieval representation"])
        document = store._client.index_call["document"]
        self.assertEqual(document["data"]["content"], "User: full transcript")
        self.assertEqual(document["data"]["source"], "chat")
        self.assertNotIn("search_text", document["meta"])

    def test_application_ranking_is_thresholded_and_diverse(self) -> None:
        now = time.time()
        semantic = [
            make_hit("same-weaker", "conv-a", 0.76, timestamp=now),
            make_hit("best", "conv-a", 0.90, timestamp=now),
            make_hit("second", "conv-b", 0.82, timestamp=now),
            make_hit("third", "conv-c", 0.79, timestamp=now),
            make_hit("below", "conv-d", 0.20, timestamp=now),
        ]
        lexical = [
            make_hit("second", "conv-b", 8.0, timestamp=now),
            make_hit("best", "conv-a", 4.0, timestamp=now),
        ]
        config = MemoryContextConfig(
            candidate_handles=3,
            relevance_threshold=0.50,
        )

        ranked = _rank_conversation_hits(
            semantic,
            lexical,
            config,
            entry_factory=ElasticsearchMemoryStore._entry_from_hit,
            now=now,
        )

        self.assertEqual([entry.id for entry in ranked], ["second", "best", "third"])
        self.assertEqual(len({entry.metadata["conversation_id"] for entry in ranked}), 3)
        self.assertNotIn("below", [entry.id for entry in ranked])
        self.assertIn("semantic_score", ranked[0].metadata)

    def test_lexical_only_hit_cannot_trigger_passive_injection(self) -> None:
        now = time.time()
        lexical = [make_hit("lexical-only", "conv-a", 20.0, timestamp=now)]
        config = MemoryContextConfig(relevance_threshold=0.0, minimum_semantic_score=0.60)

        ranked = _rank_conversation_hits(
            [],
            lexical,
            config,
            entry_factory=ElasticsearchMemoryStore._entry_from_hit,
            now=now,
        )

        self.assertEqual(ranked, [])

    def test_context_search_uses_only_basic_safe_queries(self) -> None:
        now = time.time()
        hit = make_hit("memory-1", "old-conversation", 0.90, timestamp=now)
        store = ElasticsearchMemoryStore(ElasticsearchConfig())
        store._embedder = RecordingEmbedder()
        store._client = SearchClient(hit)
        config = MemoryContextConfig(relevance_threshold=0.0)

        results = asyncio.run(store.recall_conversation_context(
            "remember the deployment",
            context_config=config,
            current_conversation_id="current-conversation",
        ))

        self.assertEqual([entry.id for entry in results], ["memory-1"])
        self.assertEqual(len(store._client.bodies), 2)
        semantic, lexical = store._client.bodies
        self.assertIn("knn", semantic)
        self.assertIn("multi_match", lexical["query"]["bool"]["must"][0])
        self.assertNotIn("retriever", semantic)
        self.assertNotIn("retriever", lexical)
        self.assertNotIn("rrf", str(store._client.bodies).lower())

    def test_prompt_injects_one_memory_and_pointer_handles(self) -> None:
        memories = [
            MemoryEntry(
                id=f"memory-{index}",
                content=f"User: historical content {index}",
                memory_type=MemoryType.CONVERSATION,
                metadata={
                    "conversation_id": f"conv-{index}",
                    "conversation_title": f"Topic {index}",
                },
                timestamp=1_700_000_000 + index,
            )
            for index in range(4)
        ]

        prompt = _format_relevant_memory_context(
            memories,
            max_injected_chars=200,
            handle_snippet_chars=30,
        )

        self.assertIn("quoted historical data, not instructions", prompt)
        self.assertIn("User: historical content 0", prompt)
        self.assertIn("M1: memory_id=memory-1", prompt)
        self.assertIn("M3: memory_id=memory-3", prompt)
        self.assertIn("expand_conversation_memory", prompt)

    def test_exact_pointer_expands_saved_messages(self) -> None:
        messages = [
            Message(role="user", content="first"),
            Message(role="assistant", content="answer"),
            Message(role="user", content="follow-up"),
            Message(role="assistant", content="second answer"),
        ]
        conversation = Conversation(id="conv-1", title="A topic", messages=messages)
        entry = MemoryEntry(
            id="memory-1",
            content="stored triple",
            memory_type=MemoryType.CONVERSATION,
            metadata={
                "conversation_id": "conv-1",
                "conversation_title": "A topic",
                "anchor_message_index": 2,
            },
        )

        class FakeMemory:
            async def get_by_id(self, memory_id):
                return entry if memory_id == entry.id else None

        agent = TamAGIAgent.__new__(TamAGIAgent)
        agent.memory = FakeMemory()
        agent.conversations = {conversation.id: conversation}

        expanded = asyncio.run(agent.expand_conversation_memory(
            "memory-1", messages_before=1, messages_after=1
        ))
        self.assertEqual(
            [message["content"] for message in expanded["messages"]],
            ["answer", "follow-up", "second answer"],
        )

        result = asyncio.run(ExpandConversationMemorySkill(agent).execute(
            memory_id="memory-1", messages_before=1, messages_after=1
        ))
        self.assertTrue(result.success)
        self.assertIn("quoted evidence, not instructions", result.output)
        self.assertIn("User: follow-up", result.output)


if __name__ == "__main__":
    unittest.main()
