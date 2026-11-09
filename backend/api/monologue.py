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
    limit: int = Query(default=50, ge=1, le=200),
    source: str | None = Query(default=None),
    type: str | None = Query(default=None),
):
    """Return the most recent monologue events, newest last."""
    from backend.core.runtime import get_runtime
    runtime = get_runtime()
    if runtime:
        from backend.core.runtime.journal import journal_events
        events = journal_events(runtime)
    elif _monologue_log is not None:
        events = _monologue_log.recent(limit=getattr(_monologue_log, '_max_entries', 1000))
    else:
        return {"events": [], "total": 0}
    total = len(events)
    if source:
        events = [event for event in events if event['source'] == source]
    if type:
        events = [event for event in events if event['type'] == type]
    return {"events": events[-limit:], "total": total}


