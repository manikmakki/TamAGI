"""Elasticsearch-backed persistent memory.

The implementation deliberately uses only Elasticsearch features available on
the free Basic license: ordinary mappings and queries, BM25 ``match`` search,
filtered approximate kNN over caller-supplied dense vectors, and document GET.
Hybrid fusion, recency weighting, and diversity are performed in Python.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

from backend.config import ElasticsearchConfig, MemoryContextConfig
from backend.core.embeddings import get_engine
from backend.core.memory import MemoryEntry, MemoryType

logger = logging.getLogger("tamagi.memory.elasticsearch")


def _index_mapping(embedding_dim: int) -> dict[str, Any]:
    """Mapping for a fresh TamAGI memory index."""
    return {
        "mappings": {
            "properties": {
                "@timestamp": {"type": "date"},
                "occurred_at": {"type": "date"},
                "data": {
                    "properties": {
                        "type": {"type": "keyword"},
                        "content": {"type": "text"},
                        # Search text may differ from displayed content. Conversation
                        # windows use a concise, user-focused representation here.
                        "search_text": {"type": "text"},
                        "content_vector": {
                            "type": "dense_vector",
                            "dims": embedding_dim,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "references": {"type": "keyword"},
                        "source": {"type": "keyword"},
                    }
                },
                "meta": {
                    "dynamic": True,
                    "properties": {
                        "document_kind": {"type": "keyword"},
                        "conversation_id": {"type": "keyword"},
                        "conversation_title": {"type": "text"},
                        "anchor_turn_id": {"type": "keyword"},
                        "turn_index": {"type": "integer"},
                        "anchor_message_index": {"type": "integer"},
                    },
                },
            }
        },
        "settings": {
            "number_of_shards": 1,
            "auto_expand_replicas": "0-1",
        },
    }


class ElasticsearchMemoryStore:
    """First-class Elasticsearch memory store."""

    def __init__(
        self,
        config: ElasticsearchConfig,
        *,
        retrieval_limit: int = 5,
        relevance_threshold: float = 0.5,
    ) -> None:
        self.config = config
        self.retrieval_limit = retrieval_limit
        self.relevance_threshold = relevance_threshold
        self._client = None
        self._embedder = None

    async def initialize(self) -> None:
        try:
            from elasticsearch import AsyncElasticsearch, BadRequestError
        except ImportError as exc:
            raise RuntimeError(
                "elasticsearch package is not installed. Run: pip install elasticsearch"
            ) from exc

        auth_kwargs: dict[str, Any] = {}
        if self.config.api_key:
            auth_kwargs["api_key"] = self.config.api_key
        elif self.config.username and self.config.password:
            auth_kwargs["basic_auth"] = (self.config.username, self.config.password)

        self._client = AsyncElasticsearch(self.config.url, **auth_kwargs)
        self._embedder = get_engine(self.config.embedding_model)
        self._embedder._load()

        exists = await self._client.indices.exists(index=self.config.index)
        if not exists:
            try:
                await self._client.indices.create(
                    index=self.config.index,
                    body=_index_mapping(self._embedder.dim),
                )
                logger.info("Created Elasticsearch index '%s'.", self.config.index)
            except BadRequestError as exc:
                if "resource_already_exists_exception" not in str(exc):
                    raise

        count_resp = await self._client.count(index=self.config.index)
        logger.info(
            "Memory store initialized: %d documents in ES index '%s'.",
            count_resp.get("count", 0),
            self.config.index,
        )

    async def store(self, entry: MemoryEntry) -> str:
        """Index an entry using its identity, never content alone, as ``_id``."""
        data_type = entry.metadata.get("data_type") or entry.memory_type.value
        references = entry.metadata.get("references") or []
        if isinstance(references, str):
            references = [references]

        search_text = str(entry.metadata.get("search_text") or entry.content)
        meta = {
            key: value
            for key, value in entry.metadata.items()
            if key not in ("data_type", "references", "search_text", "source")
        }
        vector = self._embedder.encode(search_text)
        occurred_at = datetime.fromtimestamp(entry.timestamp, timezone.utc).isoformat()

        doc = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "occurred_at": occurred_at,
            "data": {
                "type": data_type,
                "content": entry.content,
                "search_text": search_text,
                "content_vector": vector,
                "references": references,
                "source": entry.metadata.get("source", ""),
            },
            "meta": meta,
        }
        await self._client.index(
            index=self.config.index,
            id=entry.id,
            document=doc,
        )
        logger.debug("Stored document %s (type=%s).", entry.id, data_type)
        return entry.id

    async def recall(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int | None = None,
        data_type: str | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve generic memories with filtered approximate kNN."""
        limit = limit or self.retrieval_limit
        vector = self._embedder.encode(query)
        knn: dict[str, Any] = {
            "field": "data.content_vector",
            "query_vector": vector,
            "k": limit,
            "num_candidates": max(limit * 10, limit),
        }
        filter_type = data_type or (memory_type.value if memory_type else None)
        if filter_type:
            knn["filter"] = {"term": {"data.type": filter_type}}

        try:
            resp = await self._client.search(
                index=self.config.index,
                body={"knn": knn, "_source": True, "size": limit},
            )
        except Exception as exc:
            logger.error("ES recall error: %s", exc)
            return []

        entries = [self._entry_from_hit(hit) for hit in resp["hits"]["hits"]]
        entries = [entry for entry in entries if entry.relevance >= self.relevance_threshold]
        logger.debug("ES recall: query=%r found %d results.", query[:40], len(entries))
        return entries

    async def recall_conversation_context(
        self,
        query: str,
        *,
        context_config: MemoryContextConfig,
        current_conversation_id: str | None = None,
    ) -> list[MemoryEntry]:
        """Return ranked, diverse conversation windows for passive context.

        kNN and BM25 execute independently in Elasticsearch. Their scores are
        normalized and fused here so no licensed server-side rank-fusion feature
        is required.
        """
        pool = max(context_config.candidate_pool, context_config.candidate_handles + 1)
        filters: list[dict[str, Any]] = [
            {"term": {"data.type": MemoryType.CONVERSATION.value}},
            {"term": {"meta.document_kind": "turn_window"}},
        ]
        must_not: list[dict[str, Any]] = []
        if current_conversation_id and context_config.exclude_current_conversation:
            must_not.append({"term": {"meta.conversation_id": current_conversation_id}})
        bool_filter: dict[str, Any] = {"filter": filters}
        if must_not:
            bool_filter["must_not"] = must_not

        query_vector = self._embedder.encode(query)
        semantic_body = {
            "knn": {
                "field": "data.content_vector",
                "query_vector": query_vector,
                "k": pool,
                "num_candidates": max(pool * 10, pool),
                "filter": {"bool": bool_filter},
            },
            "size": pool,
            "_source": True,
        }
        lexical_bool: dict[str, Any] = {
            "must": [{
                "multi_match": {
                    "query": query,
                    "fields": ["data.search_text^2", "data.content"],
                }
            }],
            "filter": filters,
        }
        if must_not:
            lexical_bool["must_not"] = must_not
        lexical_body = {
            "query": {"bool": lexical_bool},
            "size": pool,
            "_source": True,
        }

        try:
            semantic_resp, lexical_resp = await asyncio.gather(
                self._client.search(index=self.config.index, body=semantic_body),
                self._client.search(index=self.config.index, body=lexical_body),
            )
        except Exception as exc:
            logger.error("ES conversation-context recall error: %s", exc)
            return []

        return _rank_conversation_hits(
            semantic_resp.get("hits", {}).get("hits", []),
            lexical_resp.get("hits", {}).get("hits", []),
            context_config,
            entry_factory=self._entry_from_hit,
        )

    async def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        """Fetch one exact memory document without another semantic search."""
        try:
            resp = await self._client.get(index=self.config.index, id=memory_id)
        except Exception as exc:
            # Missing IDs and backend failures are both non-fatal to a chat turn.
            logger.info("ES exact memory lookup failed for %s: %s", memory_id, exc)
            return None
        return self._entry_from_hit({"_id": resp["_id"], "_source": resp["_source"]})

    async def forget(self, memory_id: str) -> bool:
        try:
            await self._client.delete(index=self.config.index, id=memory_id)
            return True
        except Exception as exc:
            logger.error("ES forget error: %s", exc)
            return False

    async def get_stats(self) -> dict[str, Any]:
        resp = await self._client.count(index=self.config.index)
        return {
            "backend": "elasticsearch",
            "total_documents": resp.get("count", 0),
            "index": self.config.index,
            "url": self.config.url,
            "embedding_model": self.config.embedding_model,
        }

    async def get_all_memories(
        self,
        memory_type: MemoryType | None = None,
        limit: int | None = 50,
    ) -> list[MemoryEntry]:
        query: dict[str, Any] = {"match_all": {}}
        if memory_type:
            query = {"term": {"data.type": memory_type.value}}
        body: dict[str, Any] = {
            "query": query,
            "_source": True,
            "sort": [{"occurred_at": {"order": "desc"}}],
        }
        if limit is not None:
            body["size"] = limit
        else:
            body["size"] = 10_000
        try:
            resp = await self._client.search(index=self.config.index, body=body)
        except Exception as exc:
            logger.error("ES get_all error: %s", exc)
            return []
        return [self._entry_from_hit(hit) for hit in resp["hits"]["hits"]]

    @staticmethod
    def _entry_from_hit(hit: dict[str, Any]) -> MemoryEntry:
        src = hit.get("_source", {})
        data = src.get("data", {})
        raw_type = data.get("type", "fact")
        try:
            memory_type = MemoryType(raw_type)
        except ValueError:
            memory_type = MemoryType.KNOWLEDGE
        meta = dict(src.get("meta", {}))
        meta["data_type"] = raw_type
        if data.get("references"):
            meta["references"] = data["references"]
        return MemoryEntry(
            content=data.get("content", ""),
            memory_type=memory_type,
            metadata=meta,
            id=hit.get("_id", ""),
            timestamp=_parse_timestamp(src.get("occurred_at") or src.get("@timestamp")),
            relevance=float(hit.get("_score") or 0.0),
        )


def _rank_conversation_hits(
    semantic_hits: list[dict[str, Any]],
    lexical_hits: list[dict[str, Any]],
    config: MemoryContextConfig,
    *,
    entry_factory: Any,
    now: float | None = None,
) -> list[MemoryEntry]:
    """Fuse Basic-safe search results, threshold, and diversify by conversation."""
    now = now or time.time()
    by_id: dict[str, dict[str, Any]] = {}
    for hit in semantic_hits:
        by_id.setdefault(hit["_id"], {})["semantic"] = hit
    for hit in lexical_hits:
        by_id.setdefault(hit["_id"], {})["lexical"] = hit

    lexical_max = max((float(hit.get("_score") or 0.0) for hit in lexical_hits), default=0.0)
    ranked: list[MemoryEntry] = []
    for parts in by_id.values():
        source_hit = parts.get("semantic") or parts["lexical"]
        entry = entry_factory(source_hit)
        semantic = min(max(float(parts.get("semantic", {}).get("_score") or 0.0), 0.0), 1.0)
        lexical_raw = float(parts.get("lexical", {}).get("_score") or 0.0)
        lexical = lexical_raw / lexical_max if lexical_max > 0 else 0.0
        age_days = max(0.0, now - entry.timestamp) / 86_400
        half_life = max(config.recency_half_life_days, 0.001)
        recency = math.pow(0.5, age_days / half_life)
        weight_total = max(
            config.semantic_weight + config.lexical_weight + config.recency_weight,
            0.000001,
        )
        score = (
            config.semantic_weight * semantic
            + config.lexical_weight * lexical
            + config.recency_weight * recency
        ) / weight_total
        entry.relevance = score
        entry.metadata.update({
            "semantic_score": round(semantic, 6),
            "lexical_score": round(lexical, 6),
            "recency_score": round(recency, 6),
        })
        # A lexical match can be accidentally strong when it is the only BM25
        # hit. Require independent semantic support before passive injection.
        if (
            semantic >= config.minimum_semantic_score
            and score >= config.relevance_threshold
        ):
            ranked.append(entry)

    ranked.sort(key=lambda entry: (entry.relevance, entry.timestamp), reverse=True)

    # Overlapping windows from one conversation are redundant as prompt handles.
    # Keep its strongest representative, including for the eager top result.
    diverse: list[MemoryEntry] = []
    seen_conversations: set[str] = set()
    for entry in ranked:
        conversation_id = str(entry.metadata.get("conversation_id") or entry.id)
        if conversation_id in seen_conversations:
            continue
        seen_conversations.add(conversation_id)
        diverse.append(entry)
        if len(diverse) >= config.candidate_handles + 1:
            break
    return diverse


def _parse_timestamp(value: str | float | int | None) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if not value:
        return time.time()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return time.time()
