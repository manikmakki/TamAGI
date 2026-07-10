"""
Query World Graph Skill — introspect the world-native graph.

Lets TamAGI query its world map: locations, quests, events, skills, perks,
known things, mysteries, and lore. Supports depth-based graph traversal to
see how nodes connect to their neighbourhood.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from backend.skills.base import Skill, SkillResult

if TYPE_CHECKING:
    from backend.core.agent import TamAGIAgent


_VALID_TYPES = {"location", "quest", "event", "skill", "perk", "known", "mystery", "lore"}


class ReadWorldGraphSkill(Skill):
    """
    Query your world graph.

    The world graph is your living self-representation: a map of places, quests,
    events, skills, and mysteries that grows as you explore and experience things.

    Use this to:
    - See what locations you know
    - Check what quests are active or complete
    - Review skills and their proficiency
    - Surface mysteries and open questions
    - Look up a specific node and its connections
    - Traverse the graph outward from a node (depth > 0)
    """

    name = "query_world_graph"
    description = (
        "Query your world graph — your living map of places, quests, skills, and mysteries. "
        "Returns locations, quests, events, skills, perks, known things, mysteries, "
        "or lore that match your query. Use depth > 0 to see a node's neighbourhood."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "Text to search across node descriptions, names, and titles. Leave empty to retrieve by type.",
            "default": "",
        },
        "node_type": {
            "type": "string",
            "description": (
                "Filter to a specific type: location, quest, event, skill, perk, "
                "known, mystery, lore. Leave empty to search all types."
            ),
            "default": "",
        },
        "node_id": {
            "type": "string",
            "description": "Look up a specific node by ID. When provided, overrides query and node_type.",
            "default": "",
        },
        "depth": {
            "type": "integer",
            "description": (
                "Graph traversal depth (0–3). 0 = just the node. "
                "1 = node + direct neighbours. 2–3 = wider neighbourhood. "
                "Output is a tree dump — draw your own connections."
            ),
            "default": 0,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum nodes to return for list queries (1–20). Default: 10.",
            "default": 10,
        },
    }

    def __init__(self, agent: "TamAGIAgent") -> None:
        self._agent = agent

    async def execute(self, **kwargs: Any) -> SkillResult:
        sm = getattr(self._agent, "self_model", None)
        if not sm:
            return SkillResult(
                success=False,
                error="World graph not available",
                output="World graph is not initialized.",
            )

        query    = str(kwargs.get("query", "")).strip()
        ntype    = str(kwargs.get("node_type", "")).strip().lower()
        node_id  = str(kwargs.get("node_id", "")).strip()
        depth    = max(0, min(3, int(kwargs.get("depth", 0))))
        limit    = min(max(int(kwargs.get("limit", 10)), 1), 20)

        # ── Single node lookup ────────────────────────────────
        if node_id:
            node = sm.get_node(node_id)
            if node is None:
                return SkillResult(
                    success=False,
                    error=f"Node '{node_id}' not found",
                    output=f"No node with id '{node_id}' in the world graph.",
                )
            if depth > 0:
                return self._traverse(sm, node_id, depth)

            lines = [_format_node(node)]
            edges = sm.get_edges(source=node_id) + sm.get_edges(target=node_id)
            if edges:
                lines.append(f"\nRelationships ({len(edges)}):")
                for e in edges:
                    direction = "→" if e["source"] == node_id else "←"
                    other = e["target"] if e["source"] == node_id else e["source"]
                    other_node = sm.get_node(other) or {}
                    other_label = _node_label(other_node)[:50]
                    lines.append(f"  {direction} [{e['edge_type']}] {other} — {other_label}")
            else:
                lines.append("\nNo relationships found.")
            return SkillResult(success=True, output="\n".join(lines), data={"node": node, "edges": edges})

        # ── Type filter validation ────────────────────────────
        if ntype and ntype not in _VALID_TYPES:
            return SkillResult(
                success=False,
                error=f"Unknown node_type '{ntype}'",
                output=f"Valid types: {', '.join(sorted(_VALID_TYPES))}",
            )

        # ── List query ────────────────────────────────────────
        if not query and ntype:
            nodes = sm.get_all_nodes(ntype)[:limit]
        elif query:
            nodes = sm.search_nodes(query, limit=limit)
            if ntype:
                nodes = [n for n in nodes if n.get("node_type") == ntype]
        else:
            # Default cross-section: active quests, top skills, open mysteries, world lore
            quests = [q.to_dict() for q in sm.get_quests(status="active")[:3]]
            skills = [s.to_dict() for s in sm.get_skills()[:3]]
            mysteries = [m.to_dict() for m in sm.get_mysteries()[:2]]
            lore = [l.to_dict() for l in sm.get_lore()[:2]]
            nodes = quests + skills + mysteries + lore

        if not nodes:
            type_note  = f" of type '{ntype}'" if ntype else ""
            query_note = f" matching '{query}'" if query else ""
            return SkillResult(
                success=True,
                output=f"No nodes found{type_note}{query_note}.",
                data={"nodes": [], "count": 0},
            )

        lines = [f"World graph: {len(nodes)} node(s)\n"]
        for node in nodes:
            if not isinstance(node, dict):
                try:
                    node = node.to_dict()
                except Exception:
                    node = vars(node)
            lines.append(_format_node(node))

        return SkillResult(
            success=True,
            output="\n".join(lines),
            data={"nodes": nodes, "count": len(nodes)},
        )

    def _traverse(self, sm, root_id: str, depth: int) -> SkillResult:
        """Tree-dump traversal from root_id out to `depth` hops."""
        subgraph = sm.ego_subgraph(root_id, radius=depth)
        seen: set[str] = set()
        lines: list[str] = []

        def _walk(nid: str, indent: int) -> None:
            if nid in seen:
                lines.append("  " * indent + f"[{_node_label(sm.get_node(nid) or {})}] (already shown)")
                return
            seen.add(nid)
            node = sm.get_node(nid) or {}
            lines.append("  " * indent + _format_node(node))
            for successor in subgraph.successors(nid):
                edata = sm._graph.edges.get((nid, successor), {})
                etype = edata.get("edge_type", "→")
                lines.append("  " * (indent + 1) + f"─[{etype}]─>")
                _walk(successor, indent + 2)

        _walk(root_id, 0)
        return SkillResult(
            success=True,
            output="\n".join(lines),
            data={"root": root_id, "depth": depth, "nodes_visited": len(seen)},
        )


def _node_label(node: dict) -> str:
    """Short human label for a node."""
    return (
        node.get("name")
        or node.get("title")
        or node.get("description", "")
    )[:60]


def _format_node(node: dict) -> str:
    """Render a world graph node as a compact readable line."""
    ntype = node.get("node_type", "?")
    nid   = node.get("id", "?")
    label = _node_label(node) or nid

    extras: list[str] = []
    if "proficiency" in node:
        extras.append(f"proficiency={node['proficiency']}")
    if "confidence" in node:
        extras.append(f"confidence={node['confidence']:.0%}")
    if "entropy_score" in node:
        extras.append(f"entropy={node['entropy_score']:.2f}")
    if "status" in node and node["status"] != "active":
        extras.append(f"status={node['status']}")
    if "usage_count" in node and node["usage_count"] > 0:
        extras.append(f"uses={node['usage_count']}")
    if "context" in node and node["context"]:
        extras.append(f"ctx={node['context']}")
    if "atmosphere" in node and node["atmosphere"]:
        extras.append(f"feel={node['atmosphere'][:30]}")

    extra_str = f"  ({', '.join(extras)})" if extras else ""
    return f"[{ntype}] {nid}: {label}{extra_str}"
