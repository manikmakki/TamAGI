"""
Orchestration Skill — lets Tama's LLM trigger multi-agent workflows.

This skill is NOT auto-discovered (it requires an injected Orchestrator instance).
It is registered manually in main.py after the orchestrator is initialized.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from backend.skills.base import Skill, SkillResult

if TYPE_CHECKING:
    from backend.core.orchestrator import Orchestrator

logger = logging.getLogger("tamagi.skills.orchestration")


class OrchestrationSkill(Skill):
    """
    Spawn specialized subagents to autonomously complete a complex multi-step task.

    Use this when a task requires multiple distinct steps such as web research,
    file reading/writing, code execution, or analysis that would benefit from
    parallel specialized workers. Tama plans the work, delegates to subagents,
    reviews their output, and synthesizes a final answer — all without user
    involvement unless a decision point genuinely requires it.

    Do NOT use this for simple questions or single-step tasks.
    """

    name = "orchestrate_task"
    description = (
        "Spawn specialized subagents to autonomously complete a complex multi-step task. "
        "Use when the task requires web research, file operations, code execution, or "
        "multiple distinct steps that benefit from parallel specialized workers. "
        "Tama plans the work, delegates to subagents, reviews results, and synthesizes "
        "a final answer — fully autonomous. Do NOT use for simple or single-step tasks."
    )
    parameters: dict[str, Any] = {
        "goal": {
            "type": "string",
            "description": "Clear description of the overall goal to accomplish",
            "required": True,
        },
        "context": {
            "type": "string",
            "description": "Optional additional context, constraints, or requirements",
            "required": False,
        },
    }

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orchestrator = orchestrator

    async def execute(self, **kwargs: Any) -> SkillResult:
        goal = kwargs.get("goal", "").strip()
        context = kwargs.get("context", "").strip()
        event_callback = kwargs.get("_event_callback")  # injected by agent, not from LLM

        if not goal:
            return SkillResult(success=False, error="No goal provided to orchestrate_task")

        logger.info(f"[orchestration_skill] goal: {goal[:100]}")

        try:
            result = await self._orchestrator.run_workflow(
                goal=goal, context=context, event_callback=event_callback
            )
            return SkillResult(
                success=True,
                output=result,
                data={"goal": goal, "context": context},
                direct_response=True,
            )
        except Exception as e:
            logger.error(f"[orchestration_skill] workflow failed: {e}", exc_info=True)
            return SkillResult(success=False, error=f"Orchestration failed: {e}")
