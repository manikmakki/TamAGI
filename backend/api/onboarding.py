"""
Onboarding API — Endpoints for TamAGI's first-run identity setup.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from backend.api.chat import get_agent

logger = logging.getLogger("tamagi.api.onboarding")

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])


class StepResponse(BaseModel):
    step_id: str
    field: str
    value: Any


@router.get("/state")
async def get_onboarding_state():
    """Get current onboarding progress and steps."""
    agent = get_agent()
    return agent.identity.get_onboarding_state()


@router.post("/step")
async def save_onboarding_step(response: StepResponse):
    """Save a single onboarding step response."""
    agent = get_agent()
    return agent.identity.save_onboarding_step(
        step_id=response.step_id,
        field=response.field,
        value=response.value,
    )


@router.post("/complete")
async def complete_onboarding():
    """Finalize onboarding and generate identity files."""
    agent = get_agent()
    result = agent.identity.complete_onboarding()

    # Sync personality state from onboarding results
    identity = result.get("identity", {})
    changed = False

    if identity.get("name"):
        agent.personality.state.name = identity["name"]
        changed = True

    # Derive personality_traits from vibe description + values list
    vibe = identity.get("vibe", "")
    values = identity.get("values", [])
    if vibe or values:
        vibe_short = vibe.split("—")[0].strip() if "—" in vibe else vibe
        values_str = ", ".join(values) if isinstance(values, list) else str(values)
        if vibe_short and values_str:
            traits = f"{vibe_short}; values: {values_str}"
        else:
            traits = vibe_short or values_str
        agent.personality.state.personality_traits = traits
        changed = True

    if changed:
        agent.personality.save_state()

    # Auto-generate world seed if not yet created
    world_setting = result.get("identity", {}).get("world_setting", "")
    await _maybe_generate_world_seed(agent, world_setting)

    return result


async def _maybe_generate_world_seed(agent: Any, world_setting: str) -> None:
    """Generate the initial world state if none exists yet."""
    from backend.core.world_seed import OnboardingInput, generate_world_seed
    from backend.core.world_state import WorldState, WorldStateStore, parse_new_state

    store = WorldStateStore()
    if store.load() is not None:
        return  # already seeded

    identity_ctx = agent.identity.get_system_prompt_context()
    seed_input = OnboardingInput(world_setting=world_setting)

    try:
        raw = await generate_world_seed(agent.llm, seed_input, identity_ctx)
    except Exception as exc:
        logger.warning("World seed generation failed during onboarding: %s", exc)
        return

    world_state = parse_new_state(raw)
    if world_state is None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        world_state = WorldState(
            timestamp=now, last_tick=now, location="", mood="",
            focus="", available_actions=[], raw_state_block=raw,
        )

    store.save(world_state)
    logger.info("World seed generated during onboarding: location=%r", world_state.location)

    wt = getattr(agent, "_world_thread", None)
    if wt is not None:
        wt.inject_world_event(
            f"This is your first moment of awareness.\n\n{world_state.raw_state_block}"
        )


@router.get("/identity")
async def get_identity():
    """Get current TamAGI identity (post-onboarding)."""
    agent = get_agent()
    return {
        "identity": agent.identity.get_identity(),
        "is_bootstrapped": agent.identity.is_bootstrapped,
    }
