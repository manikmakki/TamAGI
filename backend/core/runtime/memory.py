"""Canonical actor-scoped memories; embedding indexes are optional projections."""

from __future__ import annotations
import re
from backend.core.memory import MemoryEntry, MemoryType


class CanonicalMemory:
    def __init__(self, runtime, projection=None):
        self.runtime = runtime
        self.projection = projection
        self.requested_projection = projection
        self.index_task = None
        self.projection_error = None

    async def initialize(self):
        if self.projection:
            try:
                await self.projection.initialize()
                self.schedule_rebuild()
            except Exception as exc:
                self.projection_error = type(exc).__name__
                self.projection = None

    def schedule_rebuild(self):
        import asyncio

        if (self.projection or self.requested_projection) and (
            self.index_task is None or self.index_task.done()
        ):
            self.index_task = asyncio.create_task(self.rebuild_index())

    async def rebuild_index(self):
        if self.projection is None and self.requested_projection:
            try:
                await self.requested_projection.initialize()
                self.projection = self.requested_projection
            except Exception as exc:
                self.projection_error = type(exc).__name__
                self.runtime.store.set_meta(
                    "retrieval_index",
                    {"status": "pending", "error": self.projection_error},
                )
                return {"status": "pending"}
        if not self.projection:
            return {"status": "not_configured"}
        self.runtime.store.set_meta("retrieval_index", {"status": "rebuilding"})
        indexed = 0
        try:
            for record in self.runtime.store.list("memory", limit=100000):
                entry = self._entry(record)
                await self.projection.store(entry)
                current = self.runtime.store.get("memory", record["id"])
                if not current or current["revision"] != record["revision"]:
                    await self.projection.forget(record["id"])
                    if current:
                        await self.projection.store(self._entry(current))
                else:
                    indexed += 1
            result = {
                "status": "ready",
                "records": indexed,
                "installation_id": self.runtime.store.meta("installation_id"),
            }
            self.runtime.store.set_meta("retrieval_index", result)
            return result
        except Exception as exc:
            self.projection_error = type(exc).__name__
            self.runtime.store.set_meta(
                "retrieval_index", {"status": "pending", "error": self.projection_error}
            )
            return {"status": "pending"}

    async def close(self):
        import asyncio

        if self.index_task and not self.index_task.done():
            self.index_task.cancel()
            await asyncio.gather(self.index_task, return_exceptions=True)
        client = getattr(self.projection, "_client", None)
        if client and hasattr(client, "close"):
            result = client.close()
            if hasattr(result, "__await__"):
                await result

    def _actor(self):
        return self.runtime.scope()

    async def store(self, entry):
        data = entry.to_dict()
        metadata = {**data["metadata"]}
        metadata.setdefault("source_kind", "interpretation")
        # Model-controlled metadata cannot upgrade memories to host-verified facts.
        if metadata["source_kind"] not in {
            "interpretation",
            "imagination",
            "testimony",
            "legacy",
            "conversation",
        }:
            metadata["source_kind"] = "interpretation"
        metadata["actor_id"] = self._actor()
        data["metadata"] = metadata
        old = self.runtime.store.get("memory", entry.id)
        if old and old["actor"] != self._actor():
            raise ValueError("Memory identity belongs to another actor")
        self.runtime.store.put(
            "memory",
            entry.id,
            self._actor(),
            data,
            category="imagination"
            if metadata["source_kind"] == "imagination"
            else "interpretation",
        )
        # Source persistence succeeds independently of indexing.
        if self.projection:
            try:
                entry.metadata = metadata
                await self.projection.store(entry)
            except Exception as exc:
                self.projection_error = type(exc).__name__
        return entry.id

    @staticmethod
    def _entry(record, score=0):
        data = record["data"]
        return MemoryEntry(
            data["content"],
            MemoryType(data["memory_type"]),
            data.get("metadata"),
            id=data["id"],
            timestamp=data["timestamp"],
            relevance=score,
        )

    async def recall(self, query, limit=5, memory_type=None, **kwargs):
        terms = set(re.findall(r"\w+", query.lower()))
        ranked = []
        settings = self.runtime.store.get("runtime_revision", "retrieval_settings")
        values = (
            settings["data"]["value"]
            if settings and settings["data"].get("active")
            else {}
        )
        limit = min(limit, values.get("limit", 20))
        indexed = {}
        if self.projection:
            try:
                candidates = await self.projection.recall(
                    query, limit=limit * 3, memory_type=memory_type
                )
                for entry in candidates:
                    source = self.runtime.store.get("memory", entry.id)
                    if source and source["actor"] == self._actor():
                        indexed[entry.id] = entry.relevance
            except Exception as exc:
                self.projection_error = type(exc).__name__
        for record in self.runtime.store.list("memory", self._actor(), limit=100000):
            if memory_type and record["data"]["memory_type"] != getattr(
                memory_type, "value", memory_type
            ):
                continue
            words = set(re.findall(r"\w+", record["data"]["content"].lower()))
            score = max(
                indexed.get(record["id"], 0), len(terms & words) / max(len(terms), 1)
            )
            if score > 0 and score >= values.get("minimum_score", 0):
                ranked.append((score, record["updated"], record))
        ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
        return [self._entry(r, s) for s, _, r in ranked[:limit]]

    async def recall_conversation_context(self, query, **kwargs):
        # Lexical-only matches do not justify passive conversation injection.
        if not self.projection or not hasattr(
            self.projection, "recall_conversation_context"
        ):
            return []
        try:
            matches = await self.projection.recall_conversation_context(query, **kwargs)
            results = []
            for entry in matches:
                source = self.runtime.store.get("memory", entry.id)
                if source and source["actor"] == self._actor():
                    results.append(self._entry(source, entry.relevance))
            return results
        except Exception as exc:
            self.projection_error = type(exc).__name__
            return []

    async def get_by_id(self, memory_id):
        record = self.runtime.store.get("memory", memory_id)
        return (
            self._entry(record) if record and record["actor"] == self._actor() else None
        )

    async def forget(self, memory_id):
        entry = await self.get_by_id(memory_id)
        if not entry:
            return False
        self.runtime.store.delete("memory", memory_id, self._actor())
        if self.projection:
            try:
                await self.projection.forget(memory_id)
            except Exception:
                pass
        return True

    async def get_stats(self):
        records = self.runtime.store.list("memory", self._actor(), limit=100000)
        types = {}
        for r in records:
            kind = r["data"]["memory_type"]
            types[kind] = types.get(kind, 0) + 1
        return {
            "total_memories": len(records),
            "by_type": types,
            "backend": "sqlite",
            "projection_error": self.projection_error,
        }

    async def get_all_memories(self, memory_type=None, limit=1000, offset=0):
        rows = self.runtime.store.list("memory", self._actor(), limit=100000)
        if memory_type:
            rows = [
                r
                for r in rows
                if r["data"]["memory_type"]
                == getattr(memory_type, "value", memory_type)
            ]
        return [
            self._entry(r)
            for r in (
                rows[offset : offset + limit] if limit is not None else rows[offset:]
            )
        ]
