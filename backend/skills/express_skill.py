"""
Express Skill — lets Tama set her pose/expression as an explicit tool call.

Replaces the old [ACTION:pose_name] text-embedding approach. The LLM calls
this skill to animate the sprite for the current response. It is registered
with a reference to TamAGIState so it can call set_pose() directly.

Subagents are forbidden from calling this skill (poses belong to Tama).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from backend.skills.base import Skill, SkillResult

if TYPE_CHECKING:
    from backend.core.personality import TamAGIState

# Keep this in sync with POSES in personality.py
_VALID_POSES = [
    "idle", "happy", "excited", "thinking",
    "wave", "celebrate", "sad", "tired", "working",
]


class ExpressSkill(Skill):
    """
    Set your pose or expression for the current response.

    Call this when you want to visually express an emotion or action —
    for example, wave when greeting, celebrate when solving something,
    or thinking while working through a problem. The pose lasts for
    this response only and resets automatically on the next turn.
    """

    name = "express"
    description = (
        "Set your pose or expression for the current response. "
        "Call this to animate your sprite — wave when greeting, celebrate when "
        "solving something, use thinking while working, working when executing tasks. "
        "The pose lasts for this response only."
    )
    parameters: dict[str, Any] = {
        "pose": {
            "type": "string",
            "description": "The pose to display",
            "enum": _VALID_POSES,
            "required": True,
        },
    }

    def __init__(self, personality_state: "TamAGIState") -> None:
        self._state = personality_state

    async def execute(self, **kwargs: Any) -> SkillResult:
        pose = kwargs.get("pose", "").strip().lower()
        if pose not in _VALID_POSES:
            return SkillResult(
                success=False,
                error=f"Unknown pose '{pose}'. Valid poses: {_VALID_POSES}",
            )
        self._state.set_pose(pose)
        return SkillResult(success=True, output=f"pose set to {pose}")
