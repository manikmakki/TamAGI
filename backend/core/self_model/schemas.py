"""
TamAGI Self-Model Schemas

Typed definitions for self-model graph nodes and edges.
Ported from AURA's self_model/schemas.py — all node/edge type logic is
identical; only the module path changes.

Design note: Node schemas are mutable dataclasses because the self-model is
a living structure that changes through reflection. Mutations happen only
through the store's internal _apply_* methods.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Node Types ────────────────────────────────────────────────


class NodeType(enum.Enum):
    GOAL = "goal"
    CAPABILITY = "capability"
    STRATEGY = "strategy"
    BELIEF = "belief"
    PREFERENCE = "preference"
    UNCERTAINTY = "uncertainty"
    SIGNAL = "signal"


class GoalStatus(enum.Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


@dataclass
class GoalNode:
    """A goal the system is pursuing or has pursued."""

    id: str
    node_type: str = field(default=NodeType.GOAL.value, init=False)
    description: str = ""
    priority: float = 0.5          # 0.0–1.0
    status: str = GoalStatus.ACTIVE.value
    parent_goal_id: str | None = None
    domain: str = ""               # Exploration domain (for eg-* goals)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "node_type": self.node_type,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "parent_goal_id": self.parent_goal_id,
            "created_at": self.created_at,
        }
        if self.domain:
            d["domain"] = self.domain
        return d

    @classmethod
    def from_dict(cls, data: dict) -> GoalNode:
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            priority=data.get("priority", 0.5),
            status=data.get("status", GoalStatus.ACTIVE.value),
            parent_goal_id=data.get("parent_goal_id"),
            domain=data.get("domain", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class CapabilityNode:
    """Something the system believes it can do, with a confidence level."""

    id: str
    node_type: str = field(default=NodeType.CAPABILITY.value, init=False)
    description: str = ""
    confidence: float = 0.5        # 0.0–1.0
    last_tested: str | None = None
    test_count: int = 0
    success_rate: float = 0.0      # 0.0–1.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "description": self.description,
            "confidence": self.confidence,
            "last_tested": self.last_tested,
            "test_count": self.test_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CapabilityNode:
        node = cls(
            id=data["id"],
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.5),
            test_count=data.get("test_count", 0),
            success_rate=data.get("success_rate", 0.0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
        node.last_tested = data.get("last_tested")
        return node


@dataclass
class StrategyNode:
    """An approach the system can take to achieve a goal."""

    id: str
    node_type: str = field(default=NodeType.STRATEGY.value, init=False)
    description: str = ""
    applicable_contexts: list[str] = field(default_factory=list)
    success_history: list[bool] = field(default_factory=list)
    preference_weight: float = 0.5  # 0.0–1.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "description": self.description,
            "applicable_contexts": self.applicable_contexts,
            "success_history": self.success_history,
            "preference_weight": self.preference_weight,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StrategyNode:
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            applicable_contexts=data.get("applicable_contexts", []),
            success_history=data.get("success_history", []),
            preference_weight=data.get("preference_weight", 0.5),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class BeliefNode:
    """Something the system believes to be true, with confidence."""

    id: str
    node_type: str = field(default=NodeType.BELIEF.value, init=False)
    description: str = ""
    confidence: float = 0.5        # 0.0–1.0
    evidence_count: int = 0
    last_updated: str = field(
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
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "last_updated": self.last_updated,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BeliefNode:
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.5),
            evidence_count=data.get("evidence_count", 0),
            last_updated=data.get("last_updated", now),
            created_at=data.get("created_at", now),
        )


@dataclass
class PreferenceNode:
    """A preference the system has developed."""

    id: str
    node_type: str = field(default=NodeType.PREFERENCE.value, init=False)
    description: str = ""
    strength: float = 0.5          # 0.0–1.0
    context: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "description": self.description,
            "strength": self.strength,
            "context": self.context,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PreferenceNode:
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            strength=data.get("strength", 0.5),
            context=data.get("context", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class UncertaintyNode:
    """A domain where the system has significant uncertainty."""

    id: str
    node_type: str = field(default=NodeType.UNCERTAINTY.value, init=False)
    domain: str = ""
    entropy_score: float = 1.0     # 0.0–1.0 (1.0 = maximally uncertain)
    last_explored: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "domain": self.domain,
            "entropy_score": self.entropy_score,
            "last_explored": self.last_explored,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UncertaintyNode:
        node = cls(
            id=data["id"],
            domain=data.get("domain", ""),
            entropy_score=data.get("entropy_score", data.get("entropy", 1.0)),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
        node.last_explored = data.get("last_explored")
        return node


class SignalStatus(enum.Enum):
    PENDING = "pending"         # Detected, awaiting exploration
    EXPLORING = "exploring"     # Currently being processed
    COMMITTED = "committed"     # Exploration succeeded, results in self-model
    DISCARDED = "discarded"     # Exploration failed quality gates
    ARCHIVED = "archived"       # Committed and no longer active


@dataclass
class SignalNode:
    """A detected signal from conversation that warrants exploration.

    Signals are first-class graph nodes. They represent the system's
    awareness that something meaningful was said and needs to be
    understood, explored, and potentially integrated into identity.

    Lifecycle: pending → exploring → committed/discarded → archived
    """

    id: str
    node_type: str = field(default=NodeType.SIGNAL.value, init=False)
    raw_text: str = ""                  # The original user input
    trigger_type: str = ""              # "preference", "feedback", "goal", "entity", "periodic"
    signal_type: str = ""               # Strategy pattern: "simple", "adversarial", "collaborative"
    source: str = "user"                # "user", "system", "reflection"
    status: str = field(default=SignalStatus.PENDING.value)
    weight: float = 1.0                 # Priority weight (recency + source boost)
    keywords: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    sentiment: float = 0.0
    # Exploration results
    produced_nodes: list = field(default_factory=list)  # IDs of nodes created from this signal
    exploration_attempts: int = 0
    last_explored: str | None = None
    exploration_notes: str = ""          # Why committed/discarded
    # Timestamps
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "raw_text": self.raw_text,
            "trigger_type": self.trigger_type,
            "signal_type": self.signal_type,
            "source": self.source,
            "status": self.status,
            "weight": self.weight,
            "keywords": self.keywords,
            "entities": self.entities,
            "sentiment": self.sentiment,
            "produced_nodes": self.produced_nodes,
            "exploration_attempts": self.exploration_attempts,
            "last_explored": self.last_explored,
            "exploration_notes": self.exploration_notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SignalNode:
        node = cls(id=data["id"])
        node.raw_text = data.get("raw_text", "")
        node.trigger_type = data.get("trigger_type", "")
        node.signal_type = data.get("signal_type", "")
        node.source = data.get("source", "user")
        node.status = data.get("status", SignalStatus.PENDING.value)
        node.weight = data.get("weight", 1.0)
        node.keywords = data.get("keywords", [])
        node.entities = data.get("entities", [])
        node.sentiment = data.get("sentiment", 0.0)
        node.produced_nodes = data.get("produced_nodes", [])
        node.exploration_attempts = data.get("exploration_attempts", 0)
        node.last_explored = data.get("last_explored")
        node.exploration_notes = data.get("exploration_notes", "")
        node.created_at = data.get("created_at", datetime.now(timezone.utc).isoformat())
        return node


# ── Node factory ──────────────────────────────────────────────

# Map node_type string → class for deserialization
NODE_TYPE_MAP: dict[str, type] = {
    NodeType.GOAL.value: GoalNode,
    NodeType.CAPABILITY.value: CapabilityNode,
    NodeType.STRATEGY.value: StrategyNode,
    NodeType.BELIEF.value: BeliefNode,
    NodeType.PREFERENCE.value: PreferenceNode,
    NodeType.UNCERTAINTY.value: UncertaintyNode,
    NodeType.SIGNAL.value: SignalNode,
}


def node_from_dict(
    data: dict,
) -> GoalNode | CapabilityNode | StrategyNode | BeliefNode | PreferenceNode | UncertaintyNode | SignalNode:
    """Reconstruct a typed node from a dict (used during deserialization)."""
    node_type = data.get("node_type", "")
    cls = NODE_TYPE_MAP.get(node_type)
    if cls is None:
        raise ValueError(f"Unknown node_type: {node_type!r}")
    return cls.from_dict(data)


# ── Edge Types ────────────────────────────────────────────────


class EdgeType(enum.Enum):
    SUPPORTS = "supports"              # strategy → goal
    REQUIRES = "requires"              # goal → capability
    INFORMS = "informs"                # belief → strategy
    CONFLICTS_WITH = "conflicts_with"  # strategy ↔ strategy
    DERIVED_FROM = "derived_from"      # belief → belief
    EXPLORED_BY = "explored_by"        # uncertainty → goal
    RELATES_TO = "relates_to"          # signal → any (detected relevance)


# Valid (source_type, target_type) pairs for each edge type.
EDGE_CONSTRAINTS: dict[str, list[tuple[str, str]]] = {
    EdgeType.SUPPORTS.value: [(NodeType.STRATEGY.value, NodeType.GOAL.value)],
    EdgeType.REQUIRES.value: [(NodeType.GOAL.value, NodeType.CAPABILITY.value)],
    EdgeType.INFORMS.value: [(NodeType.BELIEF.value, NodeType.STRATEGY.value)],
    EdgeType.CONFLICTS_WITH.value: [
        (NodeType.STRATEGY.value, NodeType.STRATEGY.value),
    ],
    EdgeType.DERIVED_FROM.value: [(NodeType.BELIEF.value, NodeType.BELIEF.value)],
    EdgeType.EXPLORED_BY.value: [(NodeType.UNCERTAINTY.value, NodeType.GOAL.value)],
    EdgeType.RELATES_TO.value: [
        # Signal can relate to any node type
        (NodeType.SIGNAL.value, NodeType.GOAL.value),
        (NodeType.SIGNAL.value, NodeType.CAPABILITY.value),
        (NodeType.SIGNAL.value, NodeType.STRATEGY.value),
        (NodeType.SIGNAL.value, NodeType.BELIEF.value),
        (NodeType.SIGNAL.value, NodeType.PREFERENCE.value),
        (NodeType.SIGNAL.value, NodeType.UNCERTAINTY.value),
        # Belief can relate to goals, capabilities, uncertainties
        (NodeType.BELIEF.value, NodeType.GOAL.value),
        (NodeType.BELIEF.value, NodeType.CAPABILITY.value),
        (NodeType.BELIEF.value, NodeType.UNCERTAINTY.value),
        # Preference can relate to goals
        (NodeType.PREFERENCE.value, NodeType.GOAL.value),
        (NodeType.PREFERENCE.value, NodeType.CAPABILITY.value),
    ],
}


def validate_edge(
    source_type: str, target_type: str, edge_type: str
) -> tuple[bool, str]:
    """Check if an edge is valid given source/target node types.

    Returns:
        (is_valid, reason)
    """
    constraints = EDGE_CONSTRAINTS.get(edge_type)
    if constraints is None:
        return False, f"Unknown edge type: {edge_type!r}"

    for valid_source, valid_target in constraints:
        if source_type == valid_source and target_type == valid_target:
            return True, "Valid"

    expected = ", ".join(f"{s}→{t}" for s, t in constraints)
    return False, (
        f"Edge type {edge_type!r} requires {expected}, "
        f"got {source_type}→{target_type}"
    )
