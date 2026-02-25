"""
Skills API — Endpoints for skill management and execution.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.chat import get_agent

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillExecRequest(BaseModel):
    name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


@router.get("")
async def list_skills():
    """List all registered skills."""
    agent = get_agent()
    return {"skills": agent.skills.list_skills()}


@router.post("/execute")
async def execute_skill(request: SkillExecRequest):
    """Manually execute a skill (for testing/direct use)."""
    agent = get_agent()
    result = await agent.skills.execute(request.name, **request.arguments)
    agent.personality.state.use_skill()
    agent.personality.save_state()
    return {
        "result": result.to_dict(),
        "state": agent.personality.state.to_dict(),
    }


@router.post("/reload")
async def reload_custom_skills():
    """Reload custom skills from disk."""
    agent = get_agent()
    count = agent.skills.discover_custom_skills()
    return {
        "discovered": count,
        "total_skills": agent.skills.skill_count,
        "skills": agent.skills.list_skills(),
    }
