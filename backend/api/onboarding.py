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

    # Update personality engine with new name
    identity = result.get("identity", {})
    if identity.get("name"):
        agent.personality.state.name = identity["name"]
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
