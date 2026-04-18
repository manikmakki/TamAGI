"""
Elasticsearch Memory Store

First-class vector memory backend using Elasticsearch with the Basic license.
Replaces ChromaDB when memory.backend is set to "elasticsearch".

Document schema (one index, generic):
  @timestamp        — ISO-8601 creation time (Kibana-compatible)
  _id               — SHA-256 hash of content (deduplication via upsert)
  data.type         — free-form keyword: "memory", "dream", "discovery", etc.
  data.content      — plain-text content (also full-text indexed)
  data.content_vector — dense_vector for kNN semantic search
  data.references   — keyword array: graph node IDs, doc IDs, URLs, etc.
  data.source       — optional origin tag (e.g. "chat", "dream", "agent")
  meta.*            — any extra metadata passed by the caller

Index mapping uses cosine similarity kNN (HNSW) — available in Basic license
since Elasticsearch 8.0.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any

from backend.config import ElasticsearchConfig
from backend.core.embeddings import get_engine
from backend.core.memory import MemoryEntry, MemoryType

logger = logging.getLogger("tamagi.memory.elasticsearch")

# Index mapping — created once on first connect if the index doesn't exist.
# Adjust embedding_dim to match your chosen model (default: all-MiniLM-L6-v2 = 384).
def _index_mapping(embedding_dim: int) -> dict:
    return {
        "mappings": {
            "properties": {
                "@timestamp": {"type": "date"},
                "data": {
                    "properties": {
                        "type":    {"type": "keyword"},
                        "content": {"type": "text"},
                        "content_vector": {
                            "type":       "dense_vector",
                            "dims":       embedding_dim,
                            "index":      True,
                            "similarity": "cosine",
                        },
                        "references": {"type": "keyword"},
                        "source":     {"type": "keyword"},
                    }
                },
                "meta": {"type": "object", "dynamic": True},
            }
        },
        "settings": {
            "number_of_shards":     1,
            "auto_expand_replicas": "0-1",
        },
    }


def _content_id(content: str) -> str:
    """Deterministic document ID — SHA-256 of content text.

    Same content stored twice → same ID → ES upsert is a no-op.
    """
    return hashlib.sha256(content.encode()).hexdigest()


class ElasticsearchMemoryStore:
    """Elasticsearch-backed vector memory store.

    Implements the same interface as the ChromaDB MemoryStore so it can
    be used as a drop-in replacement via create_memory_store().
    """

    def __init__(self, config: ElasticsearchConfig) -> None:
        self.config = config
        self._client = None
        self._embedder = None

    async def initialize(self) -> None:
        """Connect to ES and ensure the index + mapping exist."""
        try:
            from elasticsearch import AsyncElasticsearch, BadRequestError
        except ImportError:
            raise RuntimeError(
                "elasticsearch package is not installed. "
                "Run: pip install elasticsearch"
            )

        # Build auth kwargs
        auth_kwargs: dict[str, Any] = {}
        if self.config.api_key:
            auth_kwargs["api_key"] = self.config.api_key
        elif self.config.username and self.config.password:
            auth_kwargs["basic_auth"] = (self.config.username, self.config.password)

        self._client = AsyncElasticsearch(self.config.url, **auth_kwargs)

        # Warm up the embedding engine (downloads model if not cached)
        self._embedder = get_engine(self.config.embedding_model)
        self._embedder._load()

        # Ensure index exists with correct mapping
        index = self.config.index
        exists = await self._client.indices.exists(index=index)
        if not exists:
            try:
                await self._client.indices.create(
                    index=index,
                    body=_index_mapping(self._embedder.dim),
                )
                logger.info("Created Elasticsearch index '%s'.", index)
            except BadRequestError as e:
                if "resource_already_exists_exception" not in str(e):
                    raise

        count_resp = await self._client.count(index=index)
        count = count_resp.get("count", 0)
        logger.info(
            "Memory store initialized: %d documents in ES index '%s'.",
            count, index,
        )

    async def store(self, entry: MemoryEntry) -> str:
        """Index a memory entry. Returns the document _id."""
        doc_id = _content_id(entry.content)

        # data.type: caller can override via metadata["data_type"]
        data_type = entry.metadata.get("data_type") or entry.memory_type.value

        # data.references: caller can pass a list via metadata["references"]
        references = entry.metadata.get("references") or []
        if isinstance(references, str):
            references = [references]

        # Strip internal-use keys from meta before storing
        meta = {
            k: v for k, v in entry.metadata.items()
            if k not in ("data_type", "references")
        }

        vector = self._embedder.encode(entry.content)

        doc = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "type":           data_type,
                "content":        entry.content,
                "content_vector": vector,
                "references":     references,
                "source":         meta.pop("source", ""),
            },
            "meta": meta,
        }

        await self._client.index(
            index=self.config.index,
            id=doc_id,
            document=doc,
        )

        logger.debug("Stored document %s (type=%s).", doc_id, data_type)
        return doc_id

    async def recall(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int | None = None,
        data_type: str | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve documents via kNN semantic search."""
        limit = limit or 5
        vector = self._embedder.encode(query)

        knn: dict[str, Any] = {
            "field":         "data.content_vector",
            "query_vector":  vector,
            "k":             limit,
            "num_candidates": limit * 10,
        }

        # Optional type filter
        filter_type = data_type or (memory_type.value if memory_type else None)
        if filter_type:
            knn["filter"] = {"term": {"data.type": filter_type}}

        body: dict[str, Any] = {
            "knn": knn,
            "_source": True,
            "size": limit,
        }

        try:
            resp = await self._client.search(index=self.config.index, body=body)
        except Exception as e:
            logger.error("ES recall error: %s", e)
            return []

        entries = []
        for hit in resp["hits"]["hits"]:
            score = hit.get("_score", 0.0)
            src = hit["_source"]
            data = src.get("data", {})
            raw_type = data.get("type", "fact")

            # Map back to MemoryType for interface compatibility
            try:
                mem_type = MemoryType(raw_type)
            except ValueError:
                mem_type = MemoryType.KNOWLEDGE

            meta = dict(src.get("meta", {}))
            meta["data_type"] = raw_type
            if data.get("references"):
                meta["references"] = data["references"]

            entries.append(MemoryEntry(
                content=data.get("content", ""),
                memory_type=mem_type,
                metadata=meta,
                id=hit["_id"],
                timestamp=_parse_timestamp(src.get("@timestamp")),
                relevance=float(score),
            ))

        logger.debug(
            "ES recall: query=%r found %d results.", query[:40], len(entries)
        )
        return entries

    async def forget(self, memory_id: str) -> bool:
        """Delete a document by ID."""
        try:
            await self._client.delete(index=self.config.index, id=memory_id)
            logger.debug("Deleted document %s.", memory_id)
            return True
        except Exception as e:
            logger.error("ES forget error: %s", e)
            return False

    async def get_stats(self) -> dict[str, Any]:
        resp = await self._client.count(index=self.config.index)
        return {
            "backend":         "elasticsearch",
            "total_documents": resp.get("count", 0),
            "index":           self.config.index,
            "url":             self.config.url,
            "embedding_model": self.config.embedding_model,
        }

    async def get_all_memories(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """Fetch documents by type (no semantic ranking)."""
        query: dict[str, Any] = {"match_all": {}}
        if memory_type:
            query = {"term": {"data.type": memory_type.value}}

        try:
            resp = await self._client.search(
                index=self.config.index,
                body={"query": query, "size": limit, "_source": True},
            )
        except Exception as e:
            logger.error("ES get_all error: %s", e)
            return []

        entries = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            data = src.get("data", {})
            raw_type = data.get("type", "fact")
            try:
                mem_type = MemoryType(raw_type)
            except ValueError:
                mem_type = MemoryType.KNOWLEDGE

            meta = dict(src.get("meta", {}))
            meta["data_type"] = raw_type
            entries.append(MemoryEntry(
                content=data.get("content", ""),
                memory_type=mem_type,
                metadata=meta,
                id=hit["_id"],
                timestamp=_parse_timestamp(src.get("@timestamp")),
            ))

        return entries


def _parse_timestamp(iso: str | None) -> float:
    if not iso:
        return time.time()
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError):
        return time.time()
