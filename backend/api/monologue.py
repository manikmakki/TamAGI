"""
Monologue API — TamAGI's inner-monologue event stream.

Exposes the persistent audit log of goals, actions, and reflections so
the frontend can render the "Mind" tab without polling the dream engine.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/monologue", tags=["monologue"])

_monologue_log = None


def set_monologue_log(log) -> None:
    global _monologue_log
    _monologue_log = log


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


