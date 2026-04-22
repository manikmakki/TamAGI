"""
Query Self-Model Skill — lets TamAGI actively introspect its own AURA graph.

The self-model graph captures TamAGI's goals, capabilities, beliefs, preferences,
strategies, uncertainties, and the typed relationships between them. A compact
summary is injected into every system prompt, but this skill gives TamAGI the
ability to query the graph directly when it needs deeper self-understanding:
"What capabilities do I have?", "What am I uncertain about?", "What strategies
have worked for goals like this?"
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from backend.skills.base import Skill, SkillResult

if TYPE_CHECKING:
    from backend.core.agent import TamAGIAgent


_VALID_TYPES = {"goal", "capability", "strategy", "belief", "preference", "uncertainty", "signal"}


class QuerySelfModelSkill(Skill):
    """
    Query your internal AURA self-model graph to understand yourself better.

    The self-model is a living knowledge graph that tracks your goals, capabilities,
    beliefs, preferences, strategies, uncertainties, and the relationships between them.
    Use this when you want to:
    - Check what capabilities you have and how confident you are in them
    - See what you're currently uncertain about (high entropy = strong curiosity)
    - Review your active goals and the strategies available for achieving them
    - Look up a specific belief, preference, or relationship
    - Understand how a specific node relates to the rest of your identity

    You can search by text query, filter by node type, look up a specific node ID,
    or request the edges (relationships) for a node.
    """

    name = "query_self_model"
    description = (
        "Query your internal AURA self-model graph — your living self-representation. "
        "Returns goals, capabilities, beliefs, preferences, strategies, or uncertainties "
        "that match your query. Use this to understand your own capabilities before "
        "attempting a task, check what you're uncertain about, or review your active goals. "
        "Optionally include relationship edges to see how nodes connect."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": (
                "Text to search across node descriptions. Leave empty to retrieve all nodes "
                "of the specified type. Examples: 'web search', 'planning', 'python code'."
            ),
            "default": "",
        },
        "node_type": {
            "type": "string",
            "description": (
                "Filter results to a specific node type. "
                "Options: goal, capability, strategy, belief, preference, uncertainty, signal. "
                "Leave empty to search all types."
            ),
            "default": "",
        },
        "node_id": {
            "type": "string",
            "description": (
                "Look up a specific node by its ID (e.g. 'g-001', 'cap-web-search'). "
                "When provided, returns the node and all its edges. "
                "Overrides query and node_type."
            ),
            "default": "",
        },
        "include_edges": {
            "type": "boolean",
            "description": "Include relationship edges in the output. Default: false.",
            "default": False,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of nodes to return (1–20). Default: 10.",
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
                error="Self-model not available",
                output="Self-model graph is not initialized.",
            )

        query = str(kwargs.get("query", "")).strip()
        node_type = str(kwargs.get("node_type", "")).strip().lower()
        node_id = str(kwargs.get("node_id", "")).strip()
        include_edges = bool(kwargs.get("include_edges", False))
        limit = min(max(int(kwargs.get("limit", 10)), 1), 20)

        # ── Single node lookup by ID ──────────────────────────────────
        if node_id:
            node = sm.get_node(node_id)
            if node is None:
                return SkillResult(
                    success=False,
                    error=f"Node '{node_id}' not found",
                    output=f"No node with ID '{node_id}' exists in the self-model.",
                )
            lines = [_format_node(node)]
            edges = sm.get_edges(source=node_id) + sm.get_edges(target=node_id)
            if edges:
                lines.append(f"\nRelationships ({len(edges)}):")
                for e in edges:
                    direction = "→" if e["source"] == node_id else "←"
                    other = e["target"] if e["source"] == node_id else e["source"]
                    other_node = sm.get_node(other) or {}
                    other_desc = (
                        other_node.get("description") or other_node.get("domain") or other
                    )[:50]
                    lines.append(f"  {direction} [{e['edge_type']}] {other} — {other_desc}")
            else:
                lines.append("\nNo relationships found for this node.")
            return SkillResult(
                success=True,
                output="\n".join(lines),
                data={"node": node, "edges": edges},
            )

        # ── Type filter without query → return all of that type ───────
        if node_type and node_type not in _VALID_TYPES:
            return SkillResult(
                success=False,
                error=f"Unknown node_type '{node_type}'",
                output=f"Valid types: {', '.join(sorted(_VALID_TYPES))}",
            )

        if not query and node_type:
            nodes = sm.get_all_nodes(node_type)[:limit]
        elif query:
            nodes = sm.search_nodes(query, limit=limit)
            if node_type:
                nodes = [n for n in nodes if n.get("node_type") == node_type]
        else:
            # No query, no type — return a cross-section summary
            nodes = (
                sm.get_goals(status="active")[:3]
                + [c.__dict__ if hasattr(c, '__dict__') else vars(c)
                   for c in sm.query_capabilities()[:4]]
                + sm.get_all_nodes("uncertainty")[:3]
            )
            # normalize — some may be dataclasses
            nodes = [n if isinstance(n, dict) else n.to_dict() for n in nodes]

        if not nodes:
            type_note = f" of type '{node_type}'" if node_type else ""
            query_note = f" matching '{query}'" if query else ""
            return SkillResult(
                success=True,
                output=f"No nodes found{type_note}{query_note}.",
                data={"nodes": [], "count": 0},
            )

        lines = [f"Self-model: {len(nodes)} node(s) found\n"]
        for node in nodes:
            if not isinstance(node, dict):
                try:
                    node = node.to_dict()
                except Exception:
                    node = vars(node)
            lines.append(_format_node(node))
            if include_edges:
                nid = node.get("id", "")
                edges = sm.get_edges(source=nid) + sm.get_edges(target=nid)
                for e in edges:
                    direction = "→" if e["source"] == nid else "←"
                    other = e["target"] if e["source"] == nid else e["source"]
                    lines.append(f"    {direction} [{e['edge_type']}] {other}")

        return SkillResult(
            success=True,
            output="\n".join(lines),
            data={"nodes": nodes, "count": len(nodes)},
        )


def _format_node(node: dict) -> str:
    """Render a self-model node as a compact readable line."""
    ntype = node.get("node_type", "?")
    nid = node.get("id", "?")

    desc = (
        node.get("description")
        or node.get("domain")
        or node.get("belief")
        or node.get("raw_text", "")
    )[:100]

    extras: list[str] = []
    if "confidence" in node:
        extras.append(f"confidence={node['confidence']:.0%}")
    if "priority" in node:
        extras.append(f"priority={node['priority']:.1f}")
    if "entropy_score" in node:
        extras.append(f"entropy={node['entropy_score']:.2f}")
    if "preference_weight" in node:
        extras.append(f"weight={node['preference_weight']:.2f}")
    if "strength" in node:
        extras.append(f"strength={node['strength']:.1f}")
    if "status" in node and node["status"] not in ("active", "pending"):
        extras.append(f"status={node['status']}")
    if "success_rate" in node and node["success_rate"] > 0:
        extras.append(f"success={node['success_rate']:.0%}")

    extra_str = f"  ({', '.join(extras)})" if extras else ""
    return f"[{ntype}] {nid}: {desc}{extra_str}"
