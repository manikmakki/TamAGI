"""
TamAGI Self-Model Schemas — World-Native Edition

Node and edge types for the world-native self-model graph.
The graph is the TamAGI's living world map: locations explored,
quests pursued, events witnessed, skills developed, mysteries pondered,
things known, and the lore that makes up identity.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Node Types ────────────────────────────────────────────────

class NodeType(enum.Enum):
    LOCATION = "location"   # A place in the world
    QUEST    = "quest"      # Something being pursued
    EVENT    = "event"      # Something that happened
    SKILL    = "skill"      # A developed capability
    PERK     = "perk"       # An unlocked trait / achievement
    KNOWN    = "known"      # Something understood or believed
    MYSTERY  = "mystery"    # Something not yet understood
    LORE     = "lore"       # Identity, world-lore, narrative fabric


class QuestStatus(enum.Enum):
    ACTIVE     = "active"
    COMPLETE   = "complete"
    ABANDONED  = "abandoned"
    DISCOVERED = "discovered"


class SkillProficiency(enum.Enum):
    NOVICE    = "novice"
    PRACTICED = "practiced"
    FLUENT    = "fluent"


# ── Node Dataclasses ──────────────────────────────────────────

@dataclass
class LocationNode:
    id: str
    node_type: str = field(default=NodeType.LOCATION.value, init=False)
    name: str = ""
    description: str = ""
    atmosphere: str = ""
    last_visited: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "name": self.name,
            "description": self.description,
            "atmosphere": self.atmosphere,
            "last_visited": self.last_visited,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LocationNode":
        node = cls(
            id=data["id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            atmosphere=data.get("atmosphere", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
        node.last_visited = data.get("last_visited")
        return node


@dataclass
class QuestNode:
    id: str
    node_type: str = field(default=NodeType.QUEST.value, init=False)
    title: str = ""
    description: str = ""
    status: str = QuestStatus.ACTIVE.value
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuestNode":
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            status=data.get("status", QuestStatus.ACTIVE.value),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class EventNode:
    id: str
    node_type: str = field(default=NodeType.EVENT.value, init=False)
    description: str = ""
    location_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "description": self.description,
            "location_id": self.location_id,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EventNode":
        now = datetime.now(timezone.utc).isoformat()
        node = cls(
            id=data["id"],
            description=data.get("description", ""),
            location_id=data.get("location_id", ""),
            created_at=data.get("created_at", now),
        )
        node.timestamp = data.get("timestamp", now)
        return node


@dataclass
class SkillNode:
    id: str
    node_type: str = field(default=NodeType.SKILL.value, init=False)
    name: str = ""
    description: str = ""
    proficiency: str = SkillProficiency.NOVICE.value
    usage_count: int = 0
    success_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "name": self.name,
            "description": self.description,
            "proficiency": self.proficiency,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillNode":
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            proficiency=data.get("proficiency", SkillProficiency.NOVICE.value),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class PerkNode:
    id: str
    node_type: str = field(default=NodeType.PERK.value, init=False)
    name: str = ""
    description: str = ""
    source_event_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "name": self.name,
            "description": self.description,
            "source_event_id": self.source_event_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerkNode":
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            source_event_id=data.get("source_event_id", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class KnownNode:
    id: str
    node_type: str = field(default=NodeType.KNOWN.value, init=False)
    description: str = ""
    confidence: float = 0.8
    evidence_count: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "description": self.description,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnownNode":
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.8),
            evidence_count=data.get("evidence_count", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class MysteryNode:
    id: str
    node_type: str = field(default=NodeType.MYSTERY.value, init=False)
    description: str = ""
    domain: str = ""
    entropy_score: float = 1.0
    # Lazily set by QA pipeline: "domain"|"preference"|"capability"|"consequence"
    subtype: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "description": self.description,
            "domain": self.domain,
            "entropy_score": self.entropy_score,
            "subtype": self.subtype,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MysteryNode":
        node = cls(
            id=data["id"],
            description=data.get("description", ""),
            domain=data.get("domain", ""),
            entropy_score=data.get("entropy_score", 1.0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
        node.subtype = data.get("subtype", "")
        return node


@dataclass
class LoreNode:
    id: str
    node_type: str = field(default=NodeType.LORE.value, init=False)
    description: str = ""
    context: str = ""   # world_genre | identity | preference | narrative
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "description": self.description,
            "context": self.context,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoreNode":
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            context=data.get("context", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


# ── Node factory ──────────────────────────────────────────────

NODE_TYPE_MAP: dict[str, type] = {
    NodeType.LOCATION.value: LocationNode,
    NodeType.QUEST.value:    QuestNode,
    NodeType.EVENT.value:    EventNode,
    NodeType.SKILL.value:    SkillNode,
    NodeType.PERK.value:     PerkNode,
    NodeType.KNOWN.value:    KnownNode,
    NodeType.MYSTERY.value:  MysteryNode,
    NodeType.LORE.value:     LoreNode,
}

AnyNode = (
    "LocationNode | QuestNode | EventNode | SkillNode | "
    "PerkNode | KnownNode | MysteryNode | LoreNode"
)


def node_from_dict(data: dict):
    """Reconstruct a typed node from a dict (used during deserialization)."""
    node_type = data.get("node_type", "")
    cls = NODE_TYPE_MAP.get(node_type)
    if cls is None:
        raise ValueError(f"Unknown node_type: {node_type!r}")
    return cls.from_dict(data)


# ── Edge Types ────────────────────────────────────────────────

class EdgeType(enum.Enum):
    LOCATED_AT = "located_at"  # event → location
    ADVANCES   = "advances"    # event / known → quest
    REQUIRES   = "requires"    # quest → skill needed
    LEADS_TO   = "leads_to"    # location → location
    UNLOCKS    = "unlocks"     # quest / event → skill / perk
    RESOLVES   = "resolves"    # known → mystery
    RELATES_TO = "relates_to"  # general connection (anything → anything)


_WORLD_TYPES: set[str] = {t.value for t in NodeType}

# World edges are narrative, not rigidly constrained by type pairs.
EDGE_CONSTRAINTS: dict[str, list[tuple[str, str]]] = {
    edge.value: [(s, t) for s in _WORLD_TYPES for t in _WORLD_TYPES]
    for edge in EdgeType
}


def validate_edge(
    source_type: str, target_type: str, edge_type: str
) -> tuple[bool, str]:
    """Check if an edge is valid given source/target node types."""
    if edge_type not in EDGE_CONSTRAINTS:
        return False, f"Unknown edge type: {edge_type!r}"
    if source_type in _WORLD_TYPES and target_type in _WORLD_TYPES:
        return True, "Valid"
    return False, f"Unknown node types: {source_type!r} → {target_type!r}"
