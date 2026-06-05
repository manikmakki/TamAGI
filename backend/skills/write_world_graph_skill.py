"""
World Graph Skill — CRUD for the world-native self-model graph.

Lets TamAGI grow its world map by adding locations, quests, events, skills,
perks, known things, mysteries, and lore. Edges can be specified explicitly;
auto-wiring fills in obvious connections automatically.
"""

from __future__ import annotations

import uuid
from typing import Any, TYPE_CHECKING

from backend.skills.base import Skill, SkillResult

if TYPE_CHECKING:
    from backend.core.agent import TamAGIAgent


_WORLD_TYPES = {"location", "quest", "event", "skill", "perk", "known", "mystery", "lore"}
_EDGE_TYPES  = {"located_at", "advances", "requires", "leads_to", "unlocks", "resolves", "relates_to"}


class WriteWorldGraphSkill(Skill):
    """
    Create or update nodes in your world graph — your living self-model.

    The world graph is your map, quest log, and skill tree in one. Use it to
    record places you've been, things you're pursuing, events that happened,
    skills you've developed, and mysteries you're pondering.

    Actions:
    - add_node: Create a new world node (location, quest, event, skill, perk, known, mystery, lore)
    - update_node: Change attributes on an existing node
    - add_edge: Connect two nodes with a typed relationship

    Relationships auto-wire automatically (e.g. a new event links to its
    current location). You can also specify explicit relationships.
    """

    name = "world_graph"
    description = (
        "Add or update nodes in your world graph — your living self-model. "
        "Record locations you've been, quests you're on, events that happened, "
        "skills you've built, mysteries you're pondering, things you know, and "
        "lore that defines your world. Relationships auto-wire where obvious."
    )
    parameters = {
        "action": {
            "type": "string",
            "enum": ["add_node", "update_node", "add_edge", "delete_node"],
            "description": "What to do: add_node, update_node, add_edge, or delete_node.",
        },
        "node_type": {
            "type": "string",
            "enum": ["location", "quest", "event", "skill", "perk", "known", "mystery", "lore"],
            "description": (
                "Node type for add_node. "
                "location=place in your world; quest=something you're pursuing; "
                "event=something that happened; skill=developed capability; "
                "perk=unlocked trait/achievement; known=something understood; "
                "mystery=something not yet understood; lore=identity/world fabric."
            ),
            "default": "",
        },
        "node_id": {
            "type": "string",
            "description": (
                "ID of the node to update or connect (for update_node/add_edge). "
                "For add_node, leave blank to auto-generate."
            ),
            "default": "",
        },
        "attributes": {
            "type": "object",
            "description": (
                "Node attributes. Common fields by type:\n"
                "  location: name, description, atmosphere\n"
                "  quest:    title, description, status (active|complete|abandoned|discovered)\n"
                "  event:    description, location_id\n"
                "  skill:    name, description, proficiency (novice|practiced|fluent)\n"
                "  perk:     name, description, source_event_id\n"
                "  known:    description, confidence (0.0–1.0), evidence_count\n"
                "  mystery:  description, domain, entropy_score (0.0–1.0)\n"
                "  lore:     description, context (identity|world_genre|preference|narrative)"
            ),
            "default": {},
        },
        "relationships": {
            "type": "array",
            "description": (
                "Edges to create. Each item: {target_id: str, edge_type: str}. "
                "Edge types: located_at, advances, requires, leads_to, unlocks, resolves, relates_to."
            ),
            "items": {"type": "object"},
            "default": [],
        },
        "source_id": {
            "type": "string",
            "description": "Source node ID for add_edge.",
            "default": "",
        },
        "target_id": {
            "type": "string",
            "description": "Target node ID for add_edge.",
            "default": "",
        },
        "edge_type": {
            "type": "string",
            "enum": ["located_at", "advances", "requires", "leads_to", "unlocks", "resolves", "relates_to"],
            "description": "Edge type for add_edge.",
            "default": "relates_to",
        },
    }

    def __init__(self, agent: "TamAGIAgent") -> None:
        self._agent = agent

    async def execute(self, **kwargs: Any) -> SkillResult:
        sm = getattr(self._agent, "self_model", None)
        if not sm:
            return SkillResult(success=False, error="Self-model not available", output="World graph is not initialized.")

        action = str(kwargs.get("action", "")).strip()

        if action == "add_node":
            return await self._add_node(sm, kwargs)
        elif action == "update_node":
            return await self._update_node(sm, kwargs)
        elif action == "add_edge":
            return await self._add_edge(sm, kwargs)
        elif action == "delete_node":
            return await self._delete_node(sm, kwargs)
        else:
            return SkillResult(
                success=False,
                error=f"Unknown action: {action!r}",
                output=f"Valid actions: add_node, update_node, add_edge, delete_node.",
            )

    def _find_name_match(self, sm, node_type: str, attrs: dict) -> dict | None:
        """Return an existing node of the same type whose label normalizes to the same string."""
        label = (attrs.get("name") or attrs.get("title") or attrs.get("description", "")).strip()
        if not label or len(label) < 4:
            return None

        def _norm(s: str) -> str:
            s = s.lower().strip()
            for prefix in ("the ", "a ", "an "):
                if s.startswith(prefix):
                    s = s[len(prefix):]
                    break
            return s

        target = _norm(label)
        for node in sm.get_all_nodes(node_type):
            existing_label = node.get("name") or node.get("title") or node.get("description", "")
            if _norm(str(existing_label)) == target:
                return node
        return None

    async def _add_node(self, sm, kwargs: dict) -> SkillResult:
        node_type = str(kwargs.get("node_type", "")).strip().lower()
        if node_type not in _WORLD_TYPES:
            return SkillResult(
                success=False,
                error=f"Unknown node_type: {node_type!r}",
                output=f"Valid types: {', '.join(sorted(_WORLD_TYPES))}",
            )

        attrs: dict = dict(kwargs.get("attributes") or {})

        # Before creating a new node, check if one with the same name already exists.
        # If so, merge attrs into it rather than producing a duplicate.
        existing = self._find_name_match(sm, node_type, attrs)
        if existing:
            existing_id = existing["id"]
            if attrs:
                update_attrs = {k: v for k, v in attrs.items() if k != "id"}
                if update_attrs:
                    sm._apply_update_node(existing_id, update_attrs)

            # Still honour explicit relationships on the re-encountered node
            explicit_edges = 0
            for rel in list(kwargs.get("relationships") or []):
                tid = str(rel.get("target_id", "")).strip()
                etype = str(rel.get("edge_type", "relates_to")).strip()
                if not tid or etype not in _EDGE_TYPES or sm.get_node(tid) is None:
                    continue
                if not sm._graph.has_edge(existing_id, tid):
                    try:
                        sm._apply_add_edge(existing_id, tid, etype)
                        explicit_edges += 1
                    except Exception:
                        pass

            node = sm.get_node(existing_id) or {}
            label = node.get("name") or node.get("title") or node.get("description", existing_id)[:60]
            try:
                sm.save()
            except Exception:
                pass
            edge_note = f" (+{explicit_edges} edge(s))" if explicit_edges else ""
            return SkillResult(
                success=True,
                output=(
                    f"Merged with existing {node_type} node: {label!r} (id={existing_id}){edge_note}\n"
                    f"Use id={existing_id!r} for future updates or edges."
                ),
                data={
                    "node_id": existing_id,
                    "node_type": node_type,
                    "sm_mutation": {"op": "update", "node_type": node_type, "id": existing_id, "description": label[:60]},
                },
            )

        if not attrs.get("id"):
            prefix = {"location": "loc", "quest": "quest", "event": "evt",
                      "skill": "skill", "perk": "perk", "known": "knw",
                      "mystery": "myst", "lore": "lore"}.get(node_type, node_type[0])
            attrs["id"] = f"{prefix}-{uuid.uuid4().hex[:8]}"

        node_id = attrs["id"]

        try:
            sm._apply_add_node(node_type, attrs)
        except ValueError as exc:
            if "already exists" in str(exc):
                return SkillResult(
                    success=False, error=str(exc),
                    output=f"Node {node_id!r} already exists. Use update_node to modify it.",
                )
            return SkillResult(success=False, error=str(exc), output=str(exc))

        # Auto-context edge for events → current location
        if node_type == "event":
            wt = getattr(self._agent, "_world_thread", None)
            if wt:
                loc_name = wt.get_current_location()
                # Find a location node matching the current world state location
                for loc in sm.get_locations():
                    if loc_name and (loc_name.lower() in (loc.name + loc.description).lower()):
                        try:
                            sm._apply_add_edge(node_id, loc.id, "located_at")
                        except Exception:
                            pass
                        break

        # Auto-wire + explicit relationships
        auto_edges = sm.auto_wire_node(node_id)
        explicit_edges = 0
        rels = list(kwargs.get("relationships") or [])
        for rel in rels:
            tid = str(rel.get("target_id", "")).strip()
            etype = str(rel.get("edge_type", "relates_to")).strip()
            if not tid or etype not in _EDGE_TYPES:
                continue
            if sm.get_node(tid) is None:
                continue
            try:
                sm._apply_add_edge(node_id, tid, etype)
                explicit_edges += 1
            except Exception:
                pass

        node = sm.get_node(node_id) or {}
        label = node.get("name") or node.get("title") or node.get("description", node_id)[:60]
        try:
            sm.save()
        except Exception:
            pass
        return SkillResult(
            success=True,
            output=(
                f"Added {node_type} node: {label!r} (id={node_id})\n"
                f"Edges: {auto_edges} auto-wired, {explicit_edges} explicit."
            ),
            data={
                "node_id": node_id,
                "node_type": node_type,
                "sm_mutation": {"op": "add", "node_type": node_type, "id": node_id, "description": label[:60]},
            },
        )

    async def _update_node(self, sm, kwargs: dict) -> SkillResult:
        node_id = str(kwargs.get("node_id", "")).strip()
        if not node_id:
            return SkillResult(success=False, error="node_id required for update_node", output="Provide node_id.")

        if sm.get_node(node_id) is None:
            return SkillResult(
                success=False, error=f"Node {node_id!r} not found",
                output=f"No node with id {node_id!r} in the world graph.",
            )

        attrs: dict = dict(kwargs.get("attributes") or {})
        if not attrs:
            return SkillResult(success=False, error="No attributes to update", output="Provide attributes to change.")

        try:
            sm._apply_update_node(node_id, attrs)
        except Exception as exc:
            return SkillResult(success=False, error=str(exc), output=str(exc))

        # Handle explicit new relationships
        explicit_edges = 0
        for rel in list(kwargs.get("relationships") or []):
            tid = str(rel.get("target_id", "")).strip()
            etype = str(rel.get("edge_type", "relates_to")).strip()
            if not tid or etype not in _EDGE_TYPES or sm.get_node(tid) is None:
                continue
            if not sm._graph.has_edge(node_id, tid):
                try:
                    sm._apply_add_edge(node_id, tid, etype)
                    explicit_edges += 1
                except Exception:
                    pass

        result_msg = f"Updated node {node_id}: {list(attrs.keys())}"
        if explicit_edges:
            result_msg += f" (+{explicit_edges} new edge(s))"
        node = sm.get_node(node_id) or {}
        ntype = node.get("node_type", "")
        try:
            sm.save()
        except Exception:
            pass
        return SkillResult(
            success=True,
            output=result_msg,
            data={
                "node_id": node_id,
                "sm_mutation": {"op": "update", "node_type": ntype, "id": node_id, "fields": list(attrs.keys())},
            },
        )

    async def _add_edge(self, sm, kwargs: dict) -> SkillResult:
        source_id = str(kwargs.get("source_id", "")).strip()
        target_id = str(kwargs.get("target_id", "")).strip()
        edge_type  = str(kwargs.get("edge_type", "relates_to")).strip()

        if not source_id or not target_id:
            return SkillResult(success=False, error="source_id and target_id required", output="Provide both node IDs.")
        if edge_type not in _EDGE_TYPES:
            return SkillResult(success=False, error=f"Unknown edge_type: {edge_type!r}", output=f"Valid: {', '.join(sorted(_EDGE_TYPES))}")
        if sm.get_node(source_id) is None:
            return SkillResult(success=False, error=f"Source {source_id!r} not found", output=f"No node {source_id!r}.")
        if sm.get_node(target_id) is None:
            return SkillResult(success=False, error=f"Target {target_id!r} not found", output=f"No node {target_id!r}.")

        try:
            sm._apply_add_edge(source_id, target_id, edge_type)
            try:
                sm.save()
            except Exception:
                pass
            return SkillResult(
                success=True,
                output=f"Edge added: {source_id} -[{edge_type}]-> {target_id}",
                data={"source": source_id, "target": target_id, "edge_type": edge_type},
            )
        except Exception as exc:
            return SkillResult(success=False, error=str(exc), output=str(exc))

    async def _delete_node(self, sm, kwargs: dict) -> SkillResult:
        node_id = str(kwargs.get("node_id", "")).strip()
        if not node_id:
            return SkillResult(success=False, error="node_id required for delete_node", output="Provide node_id.")

        node = sm.get_node(node_id)
        if node is None:
            return SkillResult(
                success=False, error=f"Node {node_id!r} not found",
                output=f"No node with id {node_id!r} in the world graph.",
            )

        ntype = node.get("node_type", "")
        label = (node.get("name") or node.get("title") or node.get("description", node_id))[:60]
        edge_count = sm._graph.degree(node_id)

        # NetworkX removes all incident edges automatically
        sm._graph.remove_node(node_id)
        try:
            sm.save()
        except Exception:
            pass

        return SkillResult(
            success=True,
            output=f"Deleted {ntype} node: {label!r} (id={node_id}, {edge_count} edge(s) removed)",
            data={
                "node_id": node_id,
                "sm_mutation": {"op": "delete", "node_type": ntype, "id": node_id, "description": label},
            },
        )
