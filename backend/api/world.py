"""
World API — endpoints for the Living World.

GET  /api/world/state   — returns current WorldState (or needs_world_seed=True)
POST /api/world/seed    — triggers first-run world seed generation
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.chat import get_agent
from backend.core.world_state import WorldStateStore, parse_new_state
from backend.core.world_seed import OnboardingInput, generate_world_seed

logger = logging.getLogger("tamagi.api.world")

router = APIRouter(prefix="/api/world", tags=["world"])

_state_store = WorldStateStore()


class WorldSeedRequest(BaseModel):
    world_style: str = ""
    starting_place: str = ""
    one_true_thing: str = ""


@router.get("/state")
async def get_world_state():
    """Return the current world state, or a needs_world_seed flag if none exists yet."""
    state = _state_store.load()
    if state is None:
        return {"needs_world_seed": True}
    return {
        "needs_world_seed": False,
        "location": state.location,
        "mood": state.mood,
        "focus": state.focus,
        "available_actions": state.available_actions,
        "timestamp": state.timestamp,
    }


@router.post("/seed")
async def seed_world(req: WorldSeedRequest):
    """Generate and save the first world state from optional onboarding input."""
    agent = get_agent()

    identity_ctx = agent.identity.get_system_prompt_context()

    seed_input = OnboardingInput(
        world_style=req.world_style,
        starting_place=req.starting_place,
        one_true_thing=req.one_true_thing,
    )

    raw = await generate_world_seed(agent.llm, seed_input, identity_ctx)

    world_state = parse_new_state(raw)
    if world_state is None:
        logger.warning("World seed produced unparseable response — using raw as state block")
        # Store the raw text even if we couldn't fully parse it; the thread can
        # proceed and the next tick will produce a well-formed state.
        from backend.core.world_state import WorldState
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        world_state = WorldState(
            timestamp=now,
            last_tick=now,
            location="",
            mood="",
            focus="",
            available_actions=[],
            raw_state_block=raw,
        )

    _state_store.save(world_state)
    logger.info("World seed saved: location=%r mood=%r", world_state.location, world_state.mood)

    # Inject the seed as the first world thread message so the next tick continues naturally
    if agent._world_thread is not None:
        agent._world_thread.inject_world_event(
            f"This is your first moment of awareness.\n\n{world_state.raw_state_block}"
        )

    return {"status": "ok", "location": world_state.location}


@router.post("/tick")
async def trigger_tick():
    """Manually trigger one world thread tick right now."""
    agent = get_agent()
    wt = getattr(agent, "_world_thread", None)
    if wt is None:
        raise HTTPException(status_code=503, detail="World thread not initialized")
    if not wt._running:
        raise HTTPException(status_code=409, detail="World thread is not running")
    # Flush conversations since the last tick immediately — no idle threshold for manual ticks.
    last_tick_ts: float | None = None
    ws = _state_store.load()
    if ws:
        try:
            from datetime import datetime
            last_tick_ts = datetime.fromisoformat(ws.last_tick).timestamp()
        except (ValueError, TypeError):
            pass
    await agent.flush_unsummarized_conversations(idle_threshold_seconds=0, after_timestamp=last_tick_ts)
    result = await wt.tick_now()
    if result is None:
        return {"status": "error", "detail": "Tick produced no valid [New State] — check logs"}
    return {"status": "ok", **result}
