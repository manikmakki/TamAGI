"""
Self-Model API — read-only endpoints for inspecting TamAGI's world graph.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.chat import get_agent

router = APIRouter(prefix="/api/self-model", tags=["self-model"])


@router.get("")
async def get_self_model_summary():
    """Summary of TamAGI's world graph: node/edge counts and world-native collections."""
    agent = get_agent()
    sm = agent.self_model
    if sm is None:
        raise HTTPException(status_code=503, detail="Self-model not initialized")

    nodes = sm.get_all_nodes()
    by_type: dict[str, int] = {}
    for node in nodes:
        nt = node.get("node_type", "unknown")
        by_type[nt] = by_type.get(nt, 0) + 1

    return {
        "node_count": sm.node_count,
        "edge_count": sm.edge_count,
        "nodes_by_type": by_type,
        "quests":    [q.to_dict() for q in sm.get_quests()],
        "skills":    [s.to_dict() for s in sm.get_skills()],
        "mysteries": [m.to_dict() for m in sm.get_mysteries()],
        "lore":      [l.to_dict() for l in sm.get_lore()],
        "locations": [l.to_dict() for l in sm.get_locations()],
    }


@router.get("/graph")
async def get_self_model_graph():
    """Full world graph as flat nodes + edges arrays (for the built-in visualizer)."""
    agent = get_agent()
    sm = agent.self_model
    if sm is None:
        raise HTTPException(status_code=503, detail="Self-model not initialized")
    return sm.snapshot()


@router.get("/node/{node_id}")
async def get_self_model_node(node_id: str):
    """Single node with all attributes and connected edges."""
    agent = get_agent()
    sm = agent.self_model
    if sm is None:
        raise HTTPException(status_code=503, detail="Self-model not initialized")
    node = sm.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id!r} not found")
    edges = sm.get_edges(source=node_id) + sm.get_edges(target=node_id)
    return {"node": node, "edges": edges}
