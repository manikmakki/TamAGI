"""
TamAGI Self-Model Store — World-Native Edition

The world graph: a NetworkX DiGraph where nodes are typed world entities
(locations, quests, events, skills, perks, known things, mysteries, lore)
and edges are typed world relationships.

Mutations apply directly via _apply_* methods. The proposal layer is kept
for call-site compatibility but proposals are lightweight records only.
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

logger = logging.getLogger("tamagi.self_model")


# ── ModificationProposal (kept for call-site compat) ──────────

@dataclass(frozen=True)
class ModificationProposal:
    id: str = field(default_factory=lambda: f"mod-{uuid.uuid4().hex[:12]}")
    source_component: str = ""
    source_reflection_id: str = ""
    modification_type: str = ""
    target: str = ""
    current_state: tuple = ()
    proposed_state: tuple = ()
    rationale: str = ""


class SelfModel:
    """The world graph: TamAGI's persistent, evolving world-native self-model."""

    def __init__(self, data_path: str | Path = "data/self_model.json") -> None:
        self._graph = nx.DiGraph()
        self._data_path = Path(data_path)
        logger.info("Self-Model Store initialized.")

    # ══════════════════════════════════════════════════════════
    # Public Query API
    # ══════════════════════════════════════════════════════════

    def get_node(self, node_id: str) -> dict | None:
        if node_id not in self._graph:
            return None
        return dict(self._graph.nodes[node_id])

    def get_typed_node(self, node_id: str):
        data = self.get_node(node_id)
        if data is None:
            return None
        try:
            return node_from_dict(data)
        except (ValueError, KeyError):
            return None

    def get_all_nodes(self, node_type: str | None = None) -> list[dict]:
        nodes = []
        for _, attrs in self._graph.nodes(data=True):
            if node_type is None or attrs.get("node_type") == node_type:
                nodes.append(dict(attrs))
        return nodes

    def search_nodes(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across node fields. Returns up to `limit` results."""
        terms = [t.lower() for t in query.split() if t]
        scored: list[tuple[int, dict]] = []
        for _, attrs in self._graph.nodes(data=True):
            searchable = " ".join(
                str(attrs.get(f, "")).lower()
                for f in ("id", "name", "title", "description", "domain", "context", "atmosphere")
            )
            score = sum(1 for t in terms if t in searchable)
            if score > 0:
                scored.append((score, dict(attrs)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:limit]]

    # ── World-native query methods ─────────────────────────────

    def get_locations(self) -> list[LocationNode]:
        return [
            LocationNode.from_dict(dict(attrs))
            for _, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == NodeType.LOCATION.value
        ]

    def get_quests(self, status: str | None = None) -> list[QuestNode]:
        quests = [
            QuestNode.from_dict(dict(attrs))
            for _, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == NodeType.QUEST.value
            and (status is None or attrs.get("status") == status)
        ]
        return sorted(quests, key=lambda q: q.created_at, reverse=True)

    def get_skills(self) -> list[SkillNode]:
        order = {SkillProficiency.FLUENT.value: 0, SkillProficiency.PRACTICED.value: 1, SkillProficiency.NOVICE.value: 2}
        skills = [
            SkillNode.from_dict(dict(attrs))
            for _, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == NodeType.SKILL.value
        ]
        return sorted(skills, key=lambda s: (order.get(s.proficiency, 9), -s.usage_count))

    def get_mysteries(self) -> list[MysteryNode]:
        mysteries = [
            MysteryNode.from_dict(dict(attrs))
            for _, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == NodeType.MYSTERY.value
        ]
        return sorted(mysteries, key=lambda m: m.entropy_score, reverse=True)

    def get_lore(self) -> list[LoreNode]:
        return [
            LoreNode.from_dict(dict(attrs))
            for _, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == NodeType.LORE.value
        ]

    def get_events(self, limit: int = 20) -> list[EventNode]:
        events = [
            EventNode.from_dict(dict(attrs))
            for _, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == NodeType.EVENT.value
        ]
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_known(self) -> list[KnownNode]:
        known = [
            KnownNode.from_dict(dict(attrs))
            for _, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == NodeType.KNOWN.value
        ]
        return sorted(known, key=lambda k: k.confidence, reverse=True)

    # Alias kept for QA pipeline compatibility
    def get_uncertainty_map(self) -> list[MysteryNode]:
        return self.get_mysteries()

    def auto_wire_node(self, node_id: str) -> int:
        """Stitch a new node into the graph via semantically appropriate edges.

        Returns the number of edges created (capped at 3).
        """
        node = self.get_node(node_id)
        if not node:
            return 0

        node_type = node.get("node_type", "")
        description = " ".join(filter(None, [
            node.get("description", ""),
            node.get("name", ""),
            node.get("title", ""),
            node.get("domain", ""),
        ]))
        if not description.strip():
            return 0

        # Semantic wiring rules: (source_type, target_type) → edge_type
        # Direction is always: new_node → related_node (forward) or
        # related_node → new_node (reverse).
        _FWD: dict[tuple[str, str], str] = {
            ("event",   "location"): EdgeType.LOCATED_AT.value,
            ("event",   "quest"):    EdgeType.ADVANCES.value,
            ("known",   "mystery"):  EdgeType.RESOLVES.value,
            ("known",   "quest"):    EdgeType.ADVANCES.value,
            ("quest",   "skill"):    EdgeType.REQUIRES.value,
            ("quest",   "location"): EdgeType.RELATES_TO.value,
            ("skill",   "quest"):    EdgeType.RELATES_TO.value,
            ("perk",    "quest"):    EdgeType.RELATES_TO.value,
            ("mystery", "quest"):    EdgeType.RELATES_TO.value,
            ("lore",    "quest"):    EdgeType.RELATES_TO.value,
        }
        _REV: dict[tuple[str, str], str] = {
            ("quest",   "skill"):   EdgeType.REQUIRES.value,
            ("quest",   "event"):   EdgeType.ADVANCES.value,
            ("mystery", "known"):   EdgeType.RESOLVES.value,
        }

        related = self.search_nodes(description, limit=8)
        edges_created = 0

        for rn in related:
            rid = rn.get("id", "")
            if not rid or rid == node_id:
                continue
            rtype = rn.get("node_type", "")

            edge_type = _FWD.get((node_type, rtype))
            if edge_type and not self._graph.has_edge(node_id, rid):
                try:
                    self._apply_add_edge(node_id, rid, edge_type)
                    edges_created += 1
                except (KeyError, ValueError):
                    pass
            else:
                edge_type = _REV.get((rtype, node_type))
                if edge_type and not self._graph.has_edge(rid, node_id):
                    try:
                        self._apply_add_edge(rid, node_id, edge_type)
                        edges_created += 1
                    except (KeyError, ValueError):
                        pass

            if edges_created >= 3:
                break

        if edges_created:
            logger.debug("Auto-wired %s (%s): %d edge(s)", node_id, node_type, edges_created)
        return edges_created

    def wire_orphaned_nodes(self) -> int:
        total = 0
        orphans = [n for n in self._graph.nodes if self._graph.degree(n) == 0]
        for nid in orphans:
            total += self.auto_wire_node(nid)
        if total:
            logger.info("wire_orphaned_nodes: %d edge(s) across %d orphan(s)", total, len(orphans))
        return total

    def get_edges(
        self,
        source: str | None = None,
        target: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict]:
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
        if node_id not in self._graph:
            return []
        if direction == "outgoing":
            return list(self._graph.successors(node_id))
        elif direction == "incoming":
            return list(self._graph.predecessors(node_id))
        return list(set(self._graph.successors(node_id)) | set(self._graph.predecessors(node_id)))

    def ego_subgraph(self, node_id: str, radius: int = 1) -> nx.DiGraph:
        """Return the ego-graph (node + neighbors within radius hops)."""
        if node_id not in self._graph:
            return nx.DiGraph()
        return nx.ego_graph(self._graph.to_undirected(), node_id, radius=radius).to_directed()

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ══════════════════════════════════════════════════════════
    # Public Mutation API (proposal wrappers for call-site compat)
    # ══════════════════════════════════════════════════════════

    def propose_add_node(self, node_type: str, attributes: dict, source_component: str = "self_model") -> ModificationProposal:
        node_id = attributes.get("id", f"{node_type[0]}-{uuid.uuid4().hex[:8]}")
        return ModificationProposal(
            source_component=source_component,
            modification_type="node_add",
            target=node_id,
            proposed_state=tuple(sorted({**attributes, "node_type": node_type}.items())),
            rationale=f"Add {node_type}: {attributes.get('description', node_id)}",
        )

    def propose_update_node(self, node_id: str, updates: dict, source_component: str = "self_model") -> ModificationProposal:
        current = self.get_node(node_id)
        return ModificationProposal(
            source_component=source_component,
            modification_type="weight_update",
            target=node_id,
            current_state=tuple(sorted(current.items())) if current else (),
            proposed_state=tuple(sorted(updates.items())),
            rationale=f"Update {node_id}: {list(updates.keys())}",
        )

    def propose_add_edge(self, source: str, target: str, edge_type: str, source_component: str = "self_model") -> ModificationProposal:
        return ModificationProposal(
            source_component=source_component,
            modification_type="edge_add",
            target=f"{source}->{target}",
            proposed_state=(("source", source), ("target", target), ("edge_type", edge_type)),
            rationale=f"Add {edge_type}: {source} → {target}",
        )

    def propose_remove_node(self, node_id: str, source_component: str = "self_model") -> ModificationProposal:
        current = self.get_node(node_id)
        return ModificationProposal(
            source_component=source_component,
            modification_type="node_remove",
            target=node_id,
            current_state=tuple(sorted(current.items())) if current else (),
            rationale=f"Remove node {node_id}",
        )

    def propose_remove_edge(self, source: str, target: str, source_component: str = "self_model") -> ModificationProposal:
        return ModificationProposal(
            source_component=source_component,
            modification_type="edge_remove",
            target=f"{source}->{target}",
            current_state=(("source", source), ("target", target)),
            rationale=f"Remove edge: {source} → {target}",
        )

    # ══════════════════════════════════════════════════════════
    # Internal Mutation API
    # ══════════════════════════════════════════════════════════

    def _apply_add_node(self, node_type: str, attributes: dict) -> str:
        if node_type not in NODE_TYPE_MAP:
            raise ValueError(f"Unknown node_type: {node_type!r}")
        node_id = attributes.get("id")
        if node_id is None:
            raise ValueError("Node attributes must include 'id'.")
        if node_id in self._graph:
            raise ValueError(f"Node {node_id!r} already exists.")
        full_attrs = {**attributes, "node_type": node_type}
        typed_node = node_from_dict(full_attrs)
        self._graph.add_node(node_id, **typed_node.to_dict())
        logger.debug("Added node %s (type=%s)", node_id, node_type)
        return node_id

    def _apply_update_node(self, node_id: str, updates: dict) -> None:
        if node_id not in self._graph:
            raise KeyError(f"Node {node_id!r} does not exist.")
        for key, value in updates.items():
            if key in ("id", "node_type"):
                continue
            self._graph.nodes[node_id][key] = value
        logger.debug("Updated node %s: %s", node_id, list(updates.keys()))

    def _apply_remove_node(self, node_id: str) -> dict:
        if node_id not in self._graph:
            raise KeyError(f"Node {node_id!r} does not exist.")
        attrs = dict(self._graph.nodes[node_id])
        self._graph.remove_node(node_id)
        logger.debug("Removed node %s", node_id)
        return attrs

    def _apply_add_edge(self, source: str, target: str, edge_type: str, extra_attrs: dict | None = None) -> None:
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
        nodes = [dict(attrs) for _, attrs in self._graph.nodes(data=True)]
        edges = [{"source": u, "target": v, **data} for u, v, data in self._graph.edges(data=True)]
        return {
            "version": 2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "nodes": nodes,
            "edges": edges,
        }

    def save(self, path: str | Path | None = None) -> Path:
        out_path = Path(path) if path else self._data_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.snapshot()
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Self-model saved (%d nodes, %d edges)", data["node_count"], data["edge_count"])
        return out_path

    def load(self, path: str | Path | None = None) -> None:
        in_path = Path(path) if path else self._data_path
        with open(in_path, "r") as f:
            data = json.load(f)

        version = data.get("version", 1)
        if version < 2:
            raise ValueError(
                f"Self-model version {version} is incompatible with the world-native schema. "
                "Delete data/self_model.json to start fresh."
            )

        self._graph.clear()
        for node_data in data.get("nodes", []):
            node_id = node_data["id"]
            self._graph.add_node(node_id, **node_data)
        for edge_data in data.get("edges", []):
            edge_data = dict(edge_data)
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            self._graph.add_edge(source, target, **edge_data)

        logger.info(
            "Self-model loaded (%d nodes, %d edges)",
            self._graph.number_of_nodes(), self._graph.number_of_edges(),
        )
