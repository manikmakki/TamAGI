"""
Self-Model API — read-only endpoints for inspecting TamAGI's self-model graph.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.chat import get_agent

router = APIRouter(prefix="/api/self-model", tags=["self-model"])


@router.get("")
async def get_self_model_summary():
    """Summary of TamAGI's self-model: node/edge counts, top goals, capabilities, etc."""
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
        "goals": [g.to_dict() for g in sm.get_goals()],
        "capabilities": [c.to_dict() for c in sm.query_capabilities()],
        "beliefs": [b.to_dict() for b in sm.get_beliefs()],
        "uncertainties": [u.to_dict() for u in sm.get_uncertainty_map()],
    }


@router.get("/graph")
async def get_self_model_graph():
    """Full self-model graph as flat nodes + edges arrays (for the built-in visualizer)."""
    agent = get_agent()
    sm = agent.self_model
    if sm is None:
        raise HTTPException(status_code=503, detail="Self-model not initialized")
    return sm.snapshot()


@router.get("/node/{node_id}")
async def get_self_model_node(node_id: str):
    """Single node with all attributes and its connected edges (for the inspector)."""
    agent = get_agent()
    sm = agent.self_model
    if sm is None:
        raise HTTPException(status_code=503, detail="Self-model not initialized")
    node = sm.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id!r} not found")
    edges = sm.get_edges(source=node_id) + sm.get_edges(target=node_id)
    return {"node": node, "edges": edges}
