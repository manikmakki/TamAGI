"""TamAGI Self-Model — public re-exports."""

from .schemas import (
    BeliefNode,
    CapabilityNode,
    EdgeType,
    GoalNode,
    GoalStatus,
    NodeType,
    PreferenceNode,
    SignalNode,
    SignalStatus,
    StrategyNode,
    UncertaintyNode,
    node_from_dict,
    validate_edge,
)
from .store import ModificationProposal, SelfModel
from .seed import seed_self_model

__all__ = [
    "SelfModel",
    "ModificationProposal",
    "seed_self_model",
    # Node types
    "NodeType",
    "EdgeType",
    "GoalStatus",
    "SignalStatus",
    # Node classes
    "GoalNode",
    "CapabilityNode",
    "StrategyNode",
    "BeliefNode",
    "PreferenceNode",
    "UncertaintyNode",
    "SignalNode",
    # Utilities
    "node_from_dict",
    "validate_edge",
]
