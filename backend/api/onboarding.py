"""
Onboarding API — Endpoints for TamAGI's first-run identity setup.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.chat import get_agent

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

    return result


@router.get("/identity")
async def get_identity():
    """Get current TamAGI identity (post-onboarding)."""
    agent = get_agent()
    return {
        "identity": agent.identity.get_identity(),
        "is_bootstrapped": agent.identity.is_bootstrapped,
    }
