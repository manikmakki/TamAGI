"""
Recall Memory Skill — lets TamAGI explicitly search its vector memory store.

Complements the passive auto-injection (top-2 docs added to every system prompt)
with intentional, targeted retrieval. TamAGI can call this when the question at
hand requires deeper context than what was automatically surfaced.

Works with both ChromaDB and Elasticsearch backends. The optional `data_type`
filter maps to ES's free-form `data.type` keyword; on ChromaDB it is mapped to
the closest `MemoryType` enum value where possible.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, TYPE_CHECKING

from backend.skills.base import Skill, SkillResult

if TYPE_CHECKING:
    from backend.core.agent import TamAGIAgent


def _format_entry(entry: Any, index: int) -> str:
    """Render a single MemoryEntry as a readable block."""
    ts = ""
    try:
        ts = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    relevance = f"{entry.relevance:.2f}" if entry.relevance else "—"
    data_type = entry.metadata.get("data_type") or entry.memory_type.value

    lines = [
        f"[{index}] type={data_type}  relevance={relevance}  {ts}",
        f"    {entry.content[:300]}{'…' if len(entry.content) > 300 else ''}",
    ]
    return "\n".join(lines)


class RecallMemorySkill(Skill):
    """
    Search your persistent vector memory for stored knowledge, facts,
    past conversations, and dream insights.

    Use this when the question requires specific information you may have
    stored — things the user told you, knowledge you fed yourself, or
    content from your autonomous sessions — that isn't already visible
    in your current context.
    """

    name = "recall_memory"
    description = (
        "Search your persistent memory store for relevant stored knowledge, facts, "
        "user preferences, past conversation context, or dream insights. "
        "Use this when you need to recall something specific that isn't already "
        "visible in your current context — e.g. 'what do I know about X?', "
        "'what has the user told me about Y?', or 'what did I discover during my explorations?'. "
        "Supports optional filtering by memory type."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "What to search for. Be specific — this is used for semantic similarity search.",
            "required": True,
        },
        "data_type": {
            "type": "string",
            "description": (
                "Optional: filter by memory type. "
                "Examples: 'knowledge', 'fact', 'conversation', 'conversation_history', "
                "'dream_memory', 'dream_exploration', 'dream_experiment', 'dream_journal', "
                "'dream_wandering'. Leave empty to search all types."
            ),
            "default": "",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (1–10). Default: 5.",
            "default": 5,
        },
    }

    def __init__(self, agent: "TamAGIAgent") -> None:
        self._agent = agent

    async def execute(self, **kwargs: Any) -> SkillResult:
        query = str(kwargs.get("query", "")).strip()
        data_type = str(kwargs.get("data_type", "")).strip().lower()
        limit = min(max(int(kwargs.get("limit", 5)), 1), 10)

        if not query:
            return SkillResult(
                success=False,
                error="query is required",
                output="I need a search query to look up memories.",
            )

        try:
            entries = await _recall(self._agent.memory, query, data_type, limit)
        except Exception as exc:
            return SkillResult(
                success=False,
                error=str(exc),
                output=f"Memory search failed: {exc}",
            )

        if not entries:
            type_note = f" of type '{data_type}'" if data_type else ""
            return SkillResult(
                success=True,
                output=f"No memories{type_note} found matching '{query}'.",
                data={"memories": [], "count": 0, "query": query},
            )

        formatted = "\n\n".join(_format_entry(e, i + 1) for i, e in enumerate(entries))
        header = f"Found {len(entries)} memor{'y' if len(entries) == 1 else 'ies'} for '{query}':"
        if data_type:
            header += f"  (type={data_type})"

        return SkillResult(
            success=True,
            output=f"{header}\n\n{formatted}",
            data={
                "memories": [e.to_dict() for e in entries],
                "count": len(entries),
                "query": query,
                "data_type": data_type,
            },
        )


async def _recall(memory: Any, query: str, data_type: str, limit: int) -> list:
    """
    Route the recall call to the appropriate backend method.

    ElasticsearchMemoryStore.recall() accepts a `data_type` kwarg for free-form
    type filtering. ChromaDB MemoryStore.recall() accepts `memory_type: MemoryType`.
    We detect which backend is in use and route accordingly.
    """
    from backend.core.memory_elasticsearch import ElasticsearchMemoryStore

    if isinstance(memory, ElasticsearchMemoryStore):
        return await memory.recall(
            query,
            limit=limit,
            data_type=data_type or None,
        )

    # ChromaDB path — attempt to map data_type to a MemoryType enum value
    from backend.core.memory import MemoryType
    mem_type = None
    if data_type:
        try:
            mem_type = MemoryType(data_type)
        except ValueError:
            pass  # unrecognised type — search without filter

    return await memory.recall(query, memory_type=mem_type, limit=limit)
