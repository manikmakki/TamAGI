"""
TamAGI Self-Model Store

The system's persistent, evolving self-representation. A NetworkX directed
graph where nodes are typed entities (goals, capabilities, beliefs, etc.)
and edges are typed relationships (supports, requires, informs, etc.).

Ported from AURA's self_model/store.py. The AuditLedger and Modification
Pipeline are removed — mutations apply directly, no proposal pipeline needed.

Architecture:
  - Public API (propose_*) still returns ModificationProposal objects for
    any callers that want them, but mutations can also be applied directly
    via the internal _apply_* methods.
  - Internal API (_apply_add_node, _apply_update_node, etc.) performs the
    actual graph mutations. Used during seed bootstrap and by the reflection
    engine after outcomes.
  - State is periodically serialized to disk as JSON.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from .schemas import (
    EDGE_CONSTRAINTS,
    NODE_TYPE_MAP,
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

logger = logging.getLogger("tamagi.self_model")


# ── Inline ModificationProposal (stripped from aura.schemas) ──


@dataclass(frozen=True)
class ModificationProposal:
    """Lightweight record of a proposed self-model change."""

    id: str = field(default_factory=lambda: f"mod-{uuid.uuid4().hex[:12]}")
    source_component: str = ""
    source_reflection_id: str = ""
    modification_type: str = ""  # 'weight_update', 'node_add', 'edge_add', etc.
    target: str = ""
    current_state: tuple = ()
    proposed_state: tuple = ()
    rationale: str = ""


class SelfModel:
    """The system's persistent, evolving self-representation.

    Backed by a NetworkX DiGraph. Nodes carry typed attribute dicts,
    edges carry an 'edge_type' attribute for relationship semantics.
    """

    def __init__(
        self,
        data_path: str | Path = "data/self_model.json",
    ) -> None:
        self._graph = nx.DiGraph()
        self._data_path = Path(data_path)
        logger.info("Self-Model Store initialized.")

    # ══════════════════════════════════════════════════════════
    # Public Query API — read-only, no mutations
    # ══════════════════════════════════════════════════════════

    def get_node(self, node_id: str) -> dict | None:
        """Get a node's attributes by ID, or None if not found."""
        if node_id not in self._graph:
            return None
        return dict(self._graph.nodes[node_id])

    def get_typed_node(self, node_id: str):
        """Get a node as its typed dataclass, or None if not found."""
        data = self.get_node(node_id)
        if data is None:
            return None
        try:
            return node_from_dict(data)
        except (ValueError, KeyError):
            return None

    def get_all_nodes(self, node_type: str | None = None) -> list[dict]:
        """Get all nodes, optionally filtered by type."""
        nodes = []
        for nid, attrs in self._graph.nodes(data=True):
            if node_type is None or attrs.get("node_type") == node_type:
                nodes.append(dict(attrs))
        return nodes

    def search_nodes(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across all node description, name, and ID fields.

        Case-insensitive substring match. Returns up to `limit` results sorted
        by how many query terms match (descending).
        """
        terms = [t.lower() for t in query.split() if t]
        scored: list[tuple[int, dict]] = []
        for _, attrs in self._graph.nodes(data=True):
            searchable = " ".join(
                str(attrs.get(f, "")).lower()
                for f in ("id", "description", "name", "capability_name", "belief", "domain")
            )
            score = sum(1 for t in terms if t in searchable)
            if score > 0:
                scored.append((score, dict(attrs)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:limit]]

    def auto_wire_node(self, node_id: str) -> int:
        """Stitch a newly created node into the graph via semantically appropriate edges.

        Searches for related nodes by description and creates typed edges. For most
        node types, edges are created FROM the new node TO related nodes. For
        capabilities (which can only appear as edge targets in the schema), edges are
        created FROM related goals/uncertainties TO the new capability node instead.

        Caps at 3 new edges per call. Returns the number of edges created.
        """
        node = self.get_node(node_id)
        if not node:
            return 0

        node_type = node.get("node_type", "")
        description = (
            node.get("description") or node.get("domain") or node.get("belief") or ""
        )
        if not description:
            return 0

        # Forward rules: edge FROM new node TO related node
        _FWD_RULES: dict[tuple[str, str], str] = {
            ("belief",      "strategy"):    EdgeType.INFORMS.value,
            ("belief",      "goal"):        EdgeType.RELATES_TO.value,
            ("belief",      "capability"):  EdgeType.RELATES_TO.value,
            ("belief",      "uncertainty"): EdgeType.RELATES_TO.value,
            ("preference",  "goal"):        EdgeType.RELATES_TO.value,
            ("preference",  "capability"):  EdgeType.RELATES_TO.value,
            ("signal",      "goal"):        EdgeType.RELATES_TO.value,
            ("signal",      "belief"):      EdgeType.RELATES_TO.value,
            ("signal",      "capability"):  EdgeType.RELATES_TO.value,
            ("signal",      "uncertainty"): EdgeType.RELATES_TO.value,
            ("goal",        "capability"):  EdgeType.REQUIRES.value,
            ("uncertainty", "goal"):        EdgeType.EXPLORED_BY.value,
        }

        # Reverse rules: edge FROM related node TO new node
        # Used for node types the schema only allows as edge targets (e.g. capability).
        _REV_RULES: dict[tuple[str, str], str] = {
            # (related_type, new_node_type) → edge_type  [creates: related → new_node]
            ("goal",        "capability"):  EdgeType.REQUIRES.value,
            ("belief",      "capability"):  EdgeType.RELATES_TO.value,
            ("preference",  "capability"):  EdgeType.RELATES_TO.value,
            ("signal",      "capability"):  EdgeType.RELATES_TO.value,
            ("belief",      "strategy"):    EdgeType.INFORMS.value,
        }

        related = self.search_nodes(description, limit=8)
        edges_created = 0

        for related_node in related:
            related_id = related_node.get("id", "")
            if not related_id or related_id == node_id:
                continue
            related_type = related_node.get("node_type", "")

            # Try forward rule first
            edge_type = _FWD_RULES.get((node_type, related_type))
            if edge_type and not self._graph.has_edge(node_id, related_id):
                try:
                    self._apply_add_edge(node_id, related_id, edge_type)
                    edges_created += 1
                except (KeyError, ValueError):
                    pass
            else:
                # Try reverse rule
                edge_type = _REV_RULES.get((related_type, node_type))
                if edge_type and not self._graph.has_edge(related_id, node_id):
                    try:
                        self._apply_add_edge(related_id, node_id, edge_type)
                        edges_created += 1
                    except (KeyError, ValueError):
                        pass

            if edges_created >= 3:
                break

        # Fallback: user-preference beliefs wire to g-001 ("engage genuinely with humans")
        # since knowing user context always serves that goal, even without text overlap.
        if edges_created == 0 and node_type == "belief" and description.startswith("User preference:"):
            g001 = self.get_node("g-001")
            if g001 and not self._graph.has_edge(node_id, "g-001"):
                try:
                    self._apply_add_edge(node_id, "g-001", EdgeType.RELATES_TO.value)
                    edges_created += 1
                except (KeyError, ValueError):
                    pass

        if edges_created:
            logger.debug("Auto-wired %s (%s): %d edge(s) created", node_id, node_type, edges_created)
        return edges_created

    def wire_orphaned_nodes(self) -> int:
        """Call auto_wire_node for every node that currently has no edges.

        Run once after seed or load to catch nodes that predate auto-wiring.
        Returns the total number of edges created.
        """
        total = 0
        orphans = [
            nid for nid in self._graph.nodes
            if self._graph.degree(nid) == 0
        ]
        for nid in orphans:
            total += self.auto_wire_node(nid)
        if total:
            logger.info("wire_orphaned_nodes: created %d edge(s) across %d orphan(s)", total, len(orphans))
        return total

    def query_capabilities(self, context: dict | None = None) -> list[CapabilityNode]:
        """Return capability nodes sorted by confidence (highest first)."""
        caps = []
        for nid, attrs in self._graph.nodes(data=True):
            if attrs.get("node_type") == NodeType.CAPABILITY.value:
                caps.append(CapabilityNode.from_dict(attrs))
        caps.sort(key=lambda c: c.confidence, reverse=True)
        return caps

    def query_strategies(
        self, goal_id: str | None = None, context: dict | None = None
    ) -> list[StrategyNode]:
        """Return strategy nodes sorted by preference_weight (highest first).

        If goal_id is provided, returns only strategies connected to that goal
        via 'supports' edges. Otherwise returns all strategies.
        """
        if goal_id is not None and goal_id in self._graph:
            strategy_ids = set()
            for pred in self._graph.predecessors(goal_id):
                edge_data = self._graph.edges[pred, goal_id]
                if edge_data.get("edge_type") == EdgeType.SUPPORTS.value:
                    if self._graph.nodes[pred].get("node_type") == NodeType.STRATEGY.value:
                        strategy_ids.add(pred)
            strategies = [
                StrategyNode.from_dict(dict(self._graph.nodes[sid]))
                for sid in strategy_ids
            ]
        else:
            strategies = [
                StrategyNode.from_dict(dict(attrs))
                for _, attrs in self._graph.nodes(data=True)
                if attrs.get("node_type") == NodeType.STRATEGY.value
            ]
        strategies.sort(key=lambda s: s.preference_weight, reverse=True)
        return strategies

    def get_uncertainty_map(self) -> list[UncertaintyNode]:
        """Return all uncertainty nodes sorted by entropy (highest first)."""
        uncertainties = [
            UncertaintyNode.from_dict(dict(attrs))
            for _, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == NodeType.UNCERTAINTY.value
        ]
        uncertainties.sort(key=lambda u: u.entropy_score, reverse=True)
        return uncertainties

    def get_goals(self, status: str | None = None) -> list[GoalNode]:
        """Return goals, optionally filtered by status."""
        goals = []
        for _, attrs in self._graph.nodes(data=True):
            if attrs.get("node_type") == NodeType.GOAL.value:
                if status is None or attrs.get("status") == status:
                    goals.append(GoalNode.from_dict(dict(attrs)))
        goals.sort(key=lambda g: g.priority, reverse=True)
        return goals

    def get_beliefs(self) -> list[BeliefNode]:
        """Return all beliefs sorted by confidence (highest first)."""
        beliefs = [
            BeliefNode.from_dict(dict(attrs))
            for _, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == NodeType.BELIEF.value
        ]
        beliefs.sort(key=lambda b: b.confidence, reverse=True)
        return beliefs

    def get_signals(self, status: str | None = None) -> list[SignalNode]:
        """Return signal nodes, optionally filtered by status."""
        signals = []
        for _, attrs in self._graph.nodes(data=True):
            if attrs.get("node_type") != NodeType.SIGNAL.value:
                continue
            if status is not None and attrs.get("status") != status:
                continue
            signals.append(SignalNode.from_dict(dict(attrs)))
        signals.sort(key=lambda s: s.weight, reverse=True)
        return signals

    def get_edges(
        self, source: str | None = None, target: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict]:
        """Query edges with optional filters."""
        results = []
        for u, v, data in self._graph.edges(data=True):
            if source is not None and u != source:
                continue
            if target is not None and v != target:
                continue
            if edge_type is not None and data.get("edge_type") != edge_type:
                continue
            results.append({"source": u, "target": v, **data})
        return results

    def get_neighbors(self, node_id: str, direction: str = "both") -> list[str]:
        """Get neighbor node IDs.

        Args:
            direction: 'outgoing', 'incoming', or 'both'
        """
        if node_id not in self._graph:
            return []
        if direction == "outgoing":
            return list(self._graph.successors(node_id))
        elif direction == "incoming":
            return list(self._graph.predecessors(node_id))
        else:
            return list(set(self._graph.successors(node_id)) |
                        set(self._graph.predecessors(node_id)))

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ══════════════════════════════════════════════════════════
    # Public Mutation API — returns ModificationProposals
    # ══════════════════════════════════════════════════════════

    def propose_add_node(
        self, node_type: str, attributes: dict, source_component: str = "self_model"
    ) -> ModificationProposal:
        """Propose adding a new node."""
        node_id = attributes.get("id", f"{node_type[0]}-{uuid.uuid4().hex[:8]}")
        attrs_with_id = {**attributes, "id": node_id, "node_type": node_type}

        return ModificationProposal(
            source_component=source_component,
            modification_type="node_add",
            target=node_id,
            current_state=(),
            proposed_state=tuple(sorted(attrs_with_id.items())),
            rationale=f"Add {node_type} node: {attributes.get('description', node_id)}",
        )

    def propose_update_node(
        self, node_id: str, updates: dict, source_component: str = "self_model"
    ) -> ModificationProposal:
        """Propose updating a node's attributes."""
        current = self.get_node(node_id)
        current_state = tuple(sorted(current.items())) if current else ()

        return ModificationProposal(
            source_component=source_component,
            modification_type="weight_update" if _is_weight_update(updates) else "node_add",
            target=node_id,
            current_state=current_state,
            proposed_state=tuple(sorted(updates.items())),
            rationale=f"Update node {node_id}: {list(updates.keys())}",
        )

    def propose_add_edge(
        self, source: str, target: str, edge_type: str,
        source_component: str = "self_model",
    ) -> ModificationProposal:
        """Propose adding a new edge."""
        return ModificationProposal(
            source_component=source_component,
            modification_type="edge_add",
            target=f"{source}->{target}",
            proposed_state=(("source", source), ("target", target), ("edge_type", edge_type)),
            rationale=f"Add {edge_type} edge: {source} → {target}",
        )

    def propose_remove_node(
        self, node_id: str, source_component: str = "self_model"
    ) -> ModificationProposal:
        """Propose removing a node (and all its edges)."""
        current = self.get_node(node_id)
        return ModificationProposal(
            source_component=source_component,
            modification_type="node_remove",
            target=node_id,
            current_state=tuple(sorted(current.items())) if current else (),
            rationale=f"Remove node {node_id}",
        )

    def propose_remove_edge(
        self, source: str, target: str, source_component: str = "self_model"
    ) -> ModificationProposal:
        """Propose removing an edge."""
        return ModificationProposal(
            source_component=source_component,
            modification_type="edge_remove",
            target=f"{source}->{target}",
            current_state=(("source", source), ("target", target)),
            rationale=f"Remove edge: {source} → {target}",
        )

    # ══════════════════════════════════════════════════════════
    # Internal Mutation API — direct graph operations
    # ══════════════════════════════════════════════════════════
    #
    # Called by:
    #   1. seed.py during bootstrap
    #   2. The reflection engine after proposal approval
    #
    # All mutations are logged at DEBUG level.

    def _apply_add_node(self, node_type: str, attributes: dict) -> str:
        """Add a node to the graph. Returns the node ID.

        Raises:
            ValueError: If node_type is unknown or ID already exists.
        """
        if node_type not in NODE_TYPE_MAP:
            raise ValueError(f"Unknown node_type: {node_type!r}")

        node_id = attributes.get("id")
        if node_id is None:
            raise ValueError("Node attributes must include 'id'.")
        if node_id in self._graph:
            raise ValueError(f"Node {node_id!r} already exists.")

        full_attrs = {**attributes, "node_type": node_type}
        typed_node = node_from_dict(full_attrs)
        store_attrs = typed_node.to_dict()

        self._graph.add_node(node_id, **store_attrs)
        logger.debug("Added node %s (type=%s)", node_id, node_type)
        return node_id

    def _apply_update_node(self, node_id: str, updates: dict) -> None:
        """Update attributes on an existing node.

        Raises:
            KeyError: If node_id doesn't exist.
        """
        if node_id not in self._graph:
            raise KeyError(f"Node {node_id!r} does not exist.")

        for key, value in updates.items():
            if key in ("id", "node_type"):
                continue  # Never allow changing identity fields
            self._graph.nodes[node_id][key] = value

        logger.debug("Updated node %s: %s", node_id, list(updates.keys()))

    def _apply_remove_node(self, node_id: str) -> dict:
        """Remove a node and all its edges. Returns the removed node's attributes.

        Raises:
            KeyError: If node_id doesn't exist.
        """
        if node_id not in self._graph:
            raise KeyError(f"Node {node_id!r} does not exist.")

        attrs = dict(self._graph.nodes[node_id])
        self._graph.remove_node(node_id)
        logger.debug("Removed node %s", node_id)
        return attrs

    def _apply_add_edge(
        self, source: str, target: str, edge_type: str,
        extra_attrs: dict | None = None,
    ) -> None:
        """Add a typed edge between two nodes.

        Validates edge type constraints (source/target node types).

        Raises:
            KeyError: If source or target node doesn't exist.
            ValueError: If the edge type constraint is violated.
        """
        if source not in self._graph:
            raise KeyError(f"Source node {source!r} does not exist.")
        if target not in self._graph:
            raise KeyError(f"Target node {target!r} does not exist.")

        source_type = self._graph.nodes[source].get("node_type", "")
        target_type = self._graph.nodes[target].get("node_type", "")

        valid, reason = validate_edge(source_type, target_type, edge_type)
        if not valid:
            raise ValueError(reason)

        edge_attrs = {"edge_type": edge_type}
        if extra_attrs:
            edge_attrs.update(extra_attrs)

        self._graph.add_edge(source, target, **edge_attrs)
        logger.debug("Added edge %s -[%s]-> %s", source, edge_type, target)

    def _apply_remove_edge(self, source: str, target: str) -> dict:
        """Remove an edge. Returns the removed edge's attributes.

        Raises:
            KeyError: If the edge doesn't exist.
        """
        if not self._graph.has_edge(source, target):
            raise KeyError(f"Edge {source!r} → {target!r} does not exist.")

        attrs = dict(self._graph.edges[source, target])
        self._graph.remove_edge(source, target)
        logger.debug("Removed edge %s -> %s", source, target)
        return attrs

    # ══════════════════════════════════════════════════════════
    # Serialization
    # ══════════════════════════════════════════════════════════

    def snapshot(self) -> dict:
        """Serialize the current self-model state to a dict."""
        nodes = [dict(attrs) for _, attrs in self._graph.nodes(data=True)]
        edges = [
            {"source": u, "target": v, **data}
            for u, v, data in self._graph.edges(data=True)
        ]
        return {
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "nodes": nodes,
            "edges": edges,
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Serialize the self-model to disk as JSON."""
        out_path = Path(path) if path else self._data_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.snapshot()
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(
            "Self-model saved to %s (%d nodes, %d edges)",
            out_path, data["node_count"], data["edge_count"],
        )
        return out_path

    def load(self, path: str | Path | None = None) -> None:
        """Load self-model state from a JSON file. Replaces the current graph.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file contains invalid data.
        """
        in_path = Path(path) if path else self._data_path
        with open(in_path, "r") as f:
            data = json.load(f)

        if data.get("version") != 1:
            raise ValueError(
                f"Unsupported self-model version: {data.get('version')}"
            )

        self._graph.clear()

        for node_data in data.get("nodes", []):
            node_id = node_data["id"]
            self._graph.add_node(node_id, **node_data)

        for edge_data in data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            self._graph.add_edge(source, target, **edge_data)

        logger.info(
            "Self-model loaded from %s (%d nodes, %d edges)",
            in_path, self._graph.number_of_nodes(), self._graph.number_of_edges(),
        )


# ── Helpers ───────────────────────────────────────────────────

def _is_weight_update(updates: dict) -> bool:
    """Check if an update dict only touches weight/score/confidence fields."""
    weight_keys = {
        "priority", "confidence", "preference_weight", "strength",
        "entropy_score", "success_rate", "evidence_count",
    }
    return all(k in weight_keys for k in updates.keys())
