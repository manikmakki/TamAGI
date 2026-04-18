"""
Memory System — Vector database backed long/short-term memory with RAG.

Supports ChromaDB (default) with Elasticsearch as a future option.
Provides embedding-based storage and retrieval for:
  - Conversation history summarization
  - User facts & preferences
  - Knowledge base entries
  - Skill documentation
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any

from backend.config import MemoryConfig

logger = logging.getLogger("tamagi.memory")


class MemoryType(str, Enum):
    CONVERSATION = "conversation"
    FACT = "fact"
    KNOWLEDGE = "knowledge"
    SKILL = "skill"
    PREFERENCE = "preference"


class MemoryEntry:
    """A single memory entry."""

    def __init__(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
        timestamp: float | None = None,
        relevance: float = 0.0,
    ):
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata or {}
        self.id = id or self._generate_id(content)
        self.timestamp = timestamp or time.time()
        self.relevance = relevance

    def _generate_id(self, content: str) -> str:
        h = hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16]
        return f"mem_{h}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "relevance": self.relevance,
        }


class MemoryStore:
    """
    ChromaDB-backed vector memory store.

    Provides:
    - store(): Add a memory with automatic embedding
    - recall(): Retrieve relevant memories via semantic search
    - forget(): Remove specific memories
    - summarize_recent(): Get a digest of recent memories
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._collection = None
        self._client = None

    async def initialize(self) -> None:
        """Initialize the ChromaDB connection."""
        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = Path(self.config.chromadb.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )

            self._collection = self._client.get_or_create_collection(
                name=self.config.chromadb.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            count = self._collection.count()
            logger.info(
                f"Memory store initialized: {count} memories in "
                f"'{self.config.chromadb.collection_name}'"
            )

        except ImportError:
            logger.error(
                "❌ ChromaDB not installed. Memory will be in-memory only (no persistence). "
                "Install with: pip install chromadb. "
                "This severely limits memory recall effectiveness."
            )
            self._init_fallback()

    def _init_fallback(self) -> None:
        """In-memory fallback when ChromaDB isn't available."""
        self._memories: list[MemoryEntry] = []
        self._fallback = True

    @property
    def _using_fallback(self) -> bool:
        return hasattr(self, "_fallback") and self._fallback

    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        if self._using_fallback:
            self._memories.append(entry)
            return entry.id

        metadata = {
            "memory_type": entry.memory_type.value,
            "timestamp": entry.timestamp,
            **{k: str(v) for k, v in entry.metadata.items()},
        }

        self._collection.upsert(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[metadata],
        )

        logger.debug(f"Stored memory {entry.id}: {entry.content[:80]}...")
        return entry.id

    async def recall(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve relevant memories via semantic search."""
        limit = limit or self.config.retrieval_limit

        if self._using_fallback:
            # Basic keyword matching fallback
            results = []
            query_lower = query.lower()
            for mem in self._memories:
                if query_lower in mem.content.lower():
                    results.append(mem)
            logger.debug(f"Memory recall (fallback): query='{query}' found {len(results)} memories")
            return results[:limit]

        where_filter = None
        if memory_type:
            where_filter = {"memory_type": memory_type.value}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter,
            )
        except Exception as e:
            logger.error(f"Memory recall error: {e}")
            return []

        entries = []
        filtered_count = 0
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                relevance = 1.0 - distance  # Convert distance to similarity

                if relevance < self.config.relevance_threshold:
                    filtered_count += 1
                    continue

                entries.append(MemoryEntry(
                    content=doc,
                    memory_type=MemoryType(meta.get("memory_type", "fact")),
                    metadata={k: v for k, v in meta.items()
                              if k not in ("memory_type", "timestamp")},
                    id=results["ids"][0][i],
                    timestamp=float(meta.get("timestamp", 0)),
                    relevance=relevance,
                ))

        entries.sort(key=lambda e: e.relevance, reverse=True)
        logger.debug(
            f"Memory recall: query='{query}' found {len(entries)} results "
            f"(filtered {filtered_count} below threshold {self.config.relevance_threshold})"
        )
        return entries

    async def forget(self, memory_id: str) -> bool:
        """Remove a specific memory."""
        if self._using_fallback:
            self._memories = [m for m in self._memories if m.id != memory_id]
            return True

        try:
            self._collection.delete(ids=[memory_id])
            logger.debug(f"Forgot memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error forgetting memory {memory_id}: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        if self._using_fallback:
            return {
                "backend": "in-memory (fallback)",
                "total_memories": len(self._memories),
            }

        return {
            "backend": "chromadb",
            "total_memories": self._collection.count() if self._collection else 0,
            "collection": self.config.chromadb.collection_name,
            "persist_directory": self.config.chromadb.persist_directory,
        }

    async def get_all_memories(
        self, memory_type: MemoryType | None = None, limit: int = 50
    ) -> list[MemoryEntry]:
        """Get all memories, optionally filtered by type."""
        if self._using_fallback:
            mems = self._memories
            if memory_type:
                mems = [m for m in mems if m.memory_type == memory_type]
            return mems[:limit]

        where_filter = None
        if memory_type:
            where_filter = {"memory_type": memory_type.value}

        try:
            count = self._collection.count()
            if count == 0:
                return []

            results = self._collection.get(
                where=where_filter,
                limit=min(limit, count),
            )
        except Exception as e:
            logger.error(f"Error getting all memories: {e}")
            return []

        entries = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                entries.append(MemoryEntry(
                    content=doc,
                    memory_type=MemoryType(meta.get("memory_type", "fact")),
                    metadata={k: v for k, v in meta.items()
                              if k not in ("memory_type", "timestamp")},
                    id=results["ids"][i],
                    timestamp=float(meta.get("timestamp", 0)),
                ))

        return entries


def create_memory_store(config: MemoryConfig) -> "MemoryStore":
    """Factory: return the configured memory backend.

    Returns an ElasticsearchMemoryStore when config.backend == "elasticsearch",
    otherwise returns the default ChromaDB-backed MemoryStore.
    """
    if config.backend == "elasticsearch":
        from backend.core.memory_elasticsearch import ElasticsearchMemoryStore
        return ElasticsearchMemoryStore(config.elasticsearch)  # type: ignore[return-value]
    return MemoryStore(config)
