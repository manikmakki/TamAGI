"""
TamAGI Monologue Log

Persistent, append-only event stream capturing TamAGI's inner life:
goals set (by user, self-exploration, or autonomous execution), actions
taken during dream cycles, and reflections.

Events are written to data/monologue.jsonl one JSON object per line.
The active goal queue is checkpointed to data/goals.json after every
Phase 2 so crash-recovery restores pending work.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger("tamagi.monologue")

# Event type taxonomy
MonologueEventType = Literal[
    "goal_added",       # New exploration/task goal queued
    "goal_completed",   # Goal resolved successfully
    "goal_abandoned",   # Goal expired or failed
    "action_started",   # Dream activity began
    "action_completed", # Dream activity finished
    "reflection",       # Journal or post-cycle reflection
    "user_task",        # Goal/task surfaced from user conversation
]

# Who originated this event
MonologueSource = Literal[
    "user",       # Prompted by a user message
    "self",       # Self-generated from uncertainty / motivation engine
    "autonomous", # Background execution without external prompt
]


@dataclass
class MonologueEvent:
    id: str
    type: str
    source: str
    title: str
    content: str
    timestamp: float
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MonologueEvent":
        return cls(**d)


class MonologueLog:
    """Persistent append-only inner-monologue event stream.

    Thread-safe for append; not designed for concurrent writers.
    """

    def __init__(self, log_path: str | Path = "data/monologue.jsonl"):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []
        self._load()

    # ── Public API ────────────────────────────────────────────

    def append(
        self,
        type: str,
        source: str,
        title: str,
        content: str = "",
        metadata: dict | None = None,
    ) -> MonologueEvent:
        """Append an event and flush it to disk immediately."""
        event = MonologueEvent(
            id=f"ml-{uuid.uuid4().hex[:8]}",
            type=type,
            source=source,
            title=title,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        d = event.to_dict()
        self._events.append(d)
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(d) + "\n")
        except OSError as exc:
            logger.warning("Could not write monologue event: %s", exc)
        return event

    def recent(
        self,
        limit: int = 50,
        source: str | None = None,
        type: str | None = None,
    ) -> list[dict]:
        """Return the most recent events, optionally filtered."""
        events = self._events
        if source:
            events = [e for e in events if e.get("source") == source]
        if type:
            events = [e for e in events if e.get("type") == type]
        return events[-limit:]

    def __len__(self) -> int:
        return len(self._events)

    # ── Internal ──────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            logger.info("Monologue log loaded: %d events", len(self._events))
        except OSError as exc:
            logger.warning("Could not load monologue log: %s", exc)
