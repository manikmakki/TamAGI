"""TamAGI Self-Model — public re-exports."""

from .schemas import (
    EdgeType,
    EventNode,
    KnownNode,
    LocationNode,
    LoreNode,
    MysteryNode,
    NodeType,
    PerkNode,
    QuestNode,
    QuestStatus,
    SkillNode,
    SkillProficiency,
    node_from_dict,
    validate_edge,
)
from .store import ModificationProposal, SelfModel
from .seed import seed_self_model

__all__ = [
    "SelfModel",
    "ModificationProposal",
    "seed_self_model",
    # Types
    "NodeType",
    "EdgeType",
    "QuestStatus",
    "SkillProficiency",
    # Node classes
    "LocationNode",
    "QuestNode",
    "EventNode",
    "SkillNode",
    "PerkNode",
    "KnownNode",
    "MysteryNode",
    "LoreNode",
    # Utilities
    "node_from_dict",
    "validate_edge",
]
