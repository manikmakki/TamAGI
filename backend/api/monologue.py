"""
Monologue API — TamAGI's inner-monologue event stream.

Exposes the persistent audit log of goals, actions, and reflections so
the frontend can render the "Mind" tab without polling the dream engine.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/monologue", tags=["monologue"])

_monologue_log = None
_motivation_engine = None


def set_monologue_log(log) -> None:
    global _monologue_log
    _monologue_log = log


def set_motivation_engine(engine) -> None:
    global _motivation_engine
    _motivation_engine = engine


@router.get("/log")
async def get_monologue_log(
    limit: int = Query(default=50, le=200),
    source: str | None = Query(default=None),
    type: str | None = Query(default=None),
):
    """Return the most recent monologue events, newest last."""
    if _monologue_log is None:
        return {"events": [], "total": 0}
    events = _monologue_log.recent(limit=limit, source=source or None, type=type or None)
    return {"events": events, "total": len(_monologue_log)}


@router.get("/goals")
async def get_pending_goals():
    """Return the current pending goal queue from the motivation engine."""
    if _motivation_engine is None:
        return {"goals": []}
    goals = [
        {
            "id": g.id,
            "domain": g.domain,
            "description": g.description,
            "priority": g.priority,
            "estimated_voi": g.estimated_voi,
            "timestamp": g.timestamp,
        }
        for g in _motivation_engine.pending_goals
    ]
    # Highest priority first
    goals.sort(key=lambda g: g["priority"], reverse=True)
    return {"goals": goals}
