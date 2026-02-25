"""
Dream API — Endpoints for TamAGI's autonomous behavior / dream engine.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.chat import get_agent

router = APIRouter(prefix="/api/dreams", tags=["dreams"])

# The dream engine reference is set from main.py
_dream_engine = None


def set_dream_engine(engine):
    global _dream_engine
    _dream_engine = engine


def get_dream_engine():
    return _dream_engine


@router.get("/state")
async def dream_state():
    """Get dream engine status and recent activity."""
    engine = get_dream_engine()
    if not engine:
        return {"enabled": False, "running": False, "dream_count": 0, "recent_dreams": []}
    return engine.get_state()


@router.post("/trigger")
async def trigger_dream():
    """Manually trigger a dream cycle right now."""
    engine = get_dream_engine()
    if not engine:
        return {"error": "Dream engine not initialized"}
    if not engine.enabled:
        return {"error": "Dream engine is disabled in config"}

    result = await engine.dream_now()
    if result is None:
        return {"error": "No activities available"}

    return {
        "type": result.get("type", "unknown"),
        "summary": result.get("summary", ""),
        "content": result.get("content", ""),
        "mood_delta": result.get("mood_delta", {}),
    }


@router.get("/log")
async def dream_log(limit: int = 20):
    """Get the dream activity log."""
    engine = get_dream_engine()
    if not engine:
        return {"dreams": []}
    return {"dreams": engine.get_dream_log(limit=min(limit, 50))}
