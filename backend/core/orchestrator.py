"""
Orchestrator — Multi-agent workflow engine for TamAGI.

Tama can invoke orchestration via the `orchestrate_task` skill. The orchestrator:
  1. Plans: asks the LLM to decompose a goal into subtasks (JSON with dependency graph)
  2. Executes: runs independent subtasks in parallel (asyncio.gather), dependent ones in sequence
  3. Reviews: quickly validates each subagent result, retries once on failure
  4. Synthesizes: merges all results into a single coherent answer for Tama to surface
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from backend.core.llm import LLMClient, LLMMessage, LLMResponse
from backend.skills.registry import SkillRegistry

logger = logging.getLogger("tamagi.orchestrator")

# Skill names that subagents are never allowed to call.
# orchestrate_task: prevents recursion
# express: poses belong to Tama, not subagents
_FORBIDDEN_SUBAGENT_SKILLS = {"orchestrate_task", "express"}


class OrchestratorDepthError(RuntimeError):
    """Raised when orchestration is attempted from within a subagent."""


@dataclass
class SubAgentConfig:
    name: str
    role: str
    allowed_skills: list[str] | None  # None = all skills (minus forbidden)
    max_tool_rounds: int = 3


@dataclass
class SubAgentResult:
    agent_name: str
    task: str
    success: bool
    output: str
    skills_used: list[str] = field(default_factory=list)
    rounds_taken: int = 0
    error: str | None = None


class SubAgent:
    """
    Lightweight agent that executes a single task using Tama's LLM and skills.

    Borrows llm and skills references from the parent TamAGIAgent — no new
    HTTP clients or registries are created. State (memory, personality,
    conversations) is entirely Tama's concern; subagents are stateless.
    """

    def __init__(
        self,
        config: SubAgentConfig,
        llm: LLMClient,
        skills: SkillRegistry,
    ) -> None:
        self.config = config
        self.llm = llm
        self.skills = skills

    async def run(self, task: str) -> SubAgentResult:
        """Execute task autonomously. Returns a structured result."""
        messages: list[LLMMessage] = [
            LLMMessage("system", self._build_system_prompt()),
            LLMMessage("user", task),
        ]
        tools = self._get_tools()
        skills_used: list[str] = []
        response = LLMResponse()
        rounds_taken = 0

        try:
            for round_num in range(self.config.max_tool_rounds):
                rounds_taken = round_num + 1
                response = await self.llm.chat(messages, tools=tools)

                if not response.tool_calls:
                    break

                # Append assistant message once before processing tool calls.
                # Omit intermediate content — subagents have no display channel
                # for it anyway, and feeding it back causes redundant generation.
                messages.append(LLMMessage("assistant", ""))

                for tc in response.tool_calls:
                    logger.debug(f"[{self.config.name}] tool call: {tc.name}({tc.arguments})")
                    skills_used.append(tc.name)
                    result = await self.skills.execute(name=tc.name, **tc.arguments)
                    messages.append(LLMMessage(
                        "tool",
                        json.dumps(result.to_dict() if hasattr(result, "to_dict") else result),
                        name=tc.name,
                    ))

        except Exception as e:
            logger.error(f"[{self.config.name}] error during execution: {e}", exc_info=True)
            return SubAgentResult(
                agent_name=self.config.name,
                task=task,
                success=False,
                output="",
                skills_used=skills_used,
                rounds_taken=rounds_taken,
                error=str(e),
            )

        return SubAgentResult(
            agent_name=self.config.name,
            task=task,
            success=True,
            output=response.content or "",
            skills_used=skills_used,
            rounds_taken=rounds_taken,
        )

    def _build_system_prompt(self) -> str:
        return (
            f"You are {self.config.name}, a specialized subagent. "
            f"Your role: {self.config.role}. "
            "Complete your assigned task thoroughly and report findings clearly. "
            "Use available tools to accomplish the task. "
            "Be precise, complete, and structured in your output. "
            "Do not ask for clarification — make reasonable decisions and finish the task."
        )

    def _get_tools(self) -> list[dict[str, Any]] | None:
        allowed = self.config.allowed_skills
        filtered = [
            skill.to_openai_tool()
            for name, skill in self.skills._skills.items()
            if name not in _FORBIDDEN_SUBAGENT_SKILLS
            and (allowed is None or name in allowed)
        ]
        return filtered or None


class Orchestrator:
    """
    Drives multi-agent workflows on behalf of TamAGIAgent.

    Entry point: run_workflow(goal, context) → str

    If config.subagent_llm is set, subagents use a dedicated LLMClient
    (different model/endpoint/temperature). Otherwise they share Tama's client.
    Call close() at application shutdown to release the subagent client if one
    was created.
    """

    def __init__(
        self,
        llm: LLMClient,
        skills: SkillRegistry,
        config: Any,  # OrchestratorConfig (imported via config.py)
    ) -> None:
        self.llm = llm          # Tama's LLM — used for plan/review/synthesize
        self.skills = skills
        self.config = config

        # Subagent LLM: dedicated client if configured, otherwise share Tama's
        if config.subagent_llm is not None:
            self._subagent_llm = LLMClient(config.subagent_llm)
            self._owns_subagent_llm = True
            logger.info(
                f"[orchestrator] subagent LLM: {config.subagent_llm.base_url} "
                f"model={config.subagent_llm.model}"
            )
        else:
            self._subagent_llm = llm
            self._owns_subagent_llm = False

    async def close(self) -> None:
        """Release the subagent LLM client if it was separately created."""
        if self._owns_subagent_llm:
            await self._subagent_llm.close()

    async def run_workflow(self, goal: str, context: str = "", event_callback: Any = None) -> str:
        """Full orchestration cycle. Returns synthesized final text."""
        logger.info(f"[orchestrator] starting workflow: {goal[:80]}")

        # Phase 1: Plan
        await _emit(event_callback, f"Planning workflow…")
        task_plan = await self._plan(goal, context)
        task_plan = task_plan[: self.config.max_subagents]
        logger.info(f"[orchestrator] plan: {[t['name'] for t in task_plan]}")
        plan_names = ", ".join(f"**{t['name']}**" for t in task_plan)
        await _emit(event_callback, f"Subtasks: {plan_names}")

        # Phase 2: Execute (adaptive — parallel where possible, serial on deps)
        results = await self._execute_adaptive(task_plan, event_callback)

        # Phase 3: Synthesize
        n_ok = sum(1 for r in results if r.success and r.output.strip())
        await _emit(event_callback, f"Synthesizing results ({n_ok}/{len(results)} agents succeeded)…")
        final = await self._synthesize(goal, results)
        logger.info(f"[orchestrator] workflow complete ({len(results)} agents)")
        return final

    # ── Phase 1: Plan ─────────────────────────────────────────

    async def _plan(self, goal: str, context: str) -> list[dict]:
        available = [s["name"] for s in self.skills.list_skills()
                     if s["name"] not in _FORBIDDEN_SUBAGENT_SKILLS]

        prompt = (
            f"You are planning a multi-agent workflow to accomplish:\n{goal}\n\n"
            + (f"Context / constraints:\n{context}\n\n" if context else "")
            + f"Available skills that agents can use: {available}\n\n"
            "Break this goal into 2-4 specific subtasks. "
            "For each subtask output a JSON object with these fields:\n"
            '  "name": short unique identifier (snake_case)\n'
            '  "agent": descriptive agent name (e.g. "researcher", "writer")\n'
            '  "role": one sentence describing the agent\'s specialty\n'
            '  "task": complete, self-contained task description (include all needed context)\n'
            '  "skills": list of skill names this agent needs, or null for all\n'
            '  "depends_on": list of "name" values that must complete first, or []\n\n'
            "Respond ONLY with a valid JSON array. No markdown, no commentary.\n"
            "Example:\n"
            '[\n'
            '  {"name":"research","agent":"researcher","role":"Web research specialist",'
            '"task":"Search for recent ...","skills":["web_search"],"depends_on":[]},\n'
            '  {"name":"write","agent":"writer","role":"Technical writer",'
            '"task":"Write a summary of: [paste research results here]",'
            '"skills":["write"],"depends_on":["research"]}\n'
            ']'
        )

        try:
            response = await self.llm.chat(
                [
                    LLMMessage("system", "You are a precise task planner. Output only valid JSON."),
                    LLMMessage("user", prompt),
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            content = _strip_fences(response.content or "")
            plan = json.loads(content)
            if not isinstance(plan, list) or not plan:
                raise ValueError("Plan is not a non-empty list")
            return plan
        except Exception as e:
            logger.warning(f"[orchestrator] planning failed ({e}), using single-task fallback")
            return [{"name": "task", "agent": "assistant",
                     "role": "General-purpose assistant",
                     "task": goal + (f"\n\nContext: {context}" if context else ""),
                     "skills": None, "depends_on": []}]

    # ── Phase 2: Execute (adaptive) ───────────────────────────

    async def _execute_adaptive(self, tasks: list[dict], event_callback: Any = None) -> list[SubAgentResult]:
        """
        Topological execution:
        - Tasks with no unsatisfied deps run in parallel via asyncio.gather()
        - Each wave injects upstream outputs into dependent task descriptions
        """
        completed: dict[str, SubAgentResult] = {}
        results: list[SubAgentResult] = []
        remaining = list(tasks)

        while remaining:
            ready = [
                t for t in remaining
                if all(d in completed for d in t.get("depends_on", []))
            ]
            if not ready:
                logger.warning("[orchestrator] dependency deadlock — running remaining tasks sequentially")
                ready = remaining[:1]

            # Enrich dependent tasks with upstream context
            for t in ready:
                deps = t.get("depends_on", [])
                if deps:
                    dep_ctx = "\n".join(
                        f"[{d}]: {completed[d].output[:600]}"
                        for d in deps if d in completed
                    )
                    if dep_ctx:
                        t["task"] = t["task"] + f"\n\nContext from prior agents:\n{dep_ctx}"

            logger.info(f"[orchestrator] running batch: {[t['name'] for t in ready]}")
            batch: list[SubAgentResult] = list(
                await asyncio.gather(*[self._run_single_task(t, event_callback) for t in ready])
            )

            for t, r in zip(ready, batch):
                completed[t["name"]] = r
                results.append(r)
                remaining.remove(t)

        return results

    async def _run_single_task(self, task_def: dict, event_callback: Any = None) -> SubAgentResult:
        """Spawn a SubAgent for one task, with one retry on failure."""
        agent_name = task_def.get("agent", task_def["name"])
        task_preview = task_def["task"][:150].replace("\n", " ")
        if len(task_def["task"]) > 150:
            task_preview += "…"
        await _emit(event_callback, f"▶ **{agent_name}**: {task_preview}")

        subagent = SubAgent(
            config=SubAgentConfig(
                name=agent_name,
                role=task_def.get("role", "General assistant"),
                allowed_skills=task_def.get("skills"),
                max_tool_rounds=self.config.subagent_max_rounds,
            ),
            llm=self._subagent_llm,
            skills=self.skills,
        )

        result = await subagent.run(task_def["task"])

        # One retry on failure
        if not result.success and result.error:
            logger.info(f"[orchestrator] retrying {task_def['name']}: {result.error[:100]}")
            await _emit(event_callback, f"↻ **{agent_name}** failed, retrying: {result.error[:120]}")
            retry_task = (
                task_def["task"]
                + f"\n\nNote: A previous attempt failed with: {result.error}. "
                "Please try a different approach."
            )
            result = await subagent.run(retry_task)

        verdict = await self._review(task_def, result)
        if verdict == "retry" and result.success:
            logger.info(f"[orchestrator] quality retry for {task_def['name']}")
            await _emit(event_callback, f"↻ **{agent_name}** improving response…")
            result = await subagent.run(
                task_def["task"] + "\n\nPlease provide a more complete and detailed response."
            )

        if result.success and result.output.strip():
            preview = result.output[:200].replace("\n", " ")
            if len(result.output) > 200:
                preview += "…"
            await _emit(event_callback, f"✓ **{agent_name}**: {preview}")
        else:
            error_detail = result.error or "returned empty output"
            await _emit(event_callback, f"✗ **{agent_name}** failed: {error_detail}")

        logger.info(
            f"[orchestrator] {task_def['name']} done "
            f"(success={result.success}, rounds={result.rounds_taken}, "
            f"skills={result.skills_used})"
        )
        return result

    # ── Phase 3: Review ───────────────────────────────────────

    async def _review(self, task_def: dict, result: SubAgentResult) -> str:
        """
        Returns "accept", "retry", or "skip".
        Fast heuristic first — only invoke LLM for suspiciously short outputs.
        """
        if not result.success:
            return "skip"  # already retried in _run_single_task
        if len(result.output.strip()) > 80:
            return "accept"

        try:
            response = await self.llm.chat(
                [
                    LLMMessage("system", "You are a quality reviewer. Reply with exactly one word."),
                    LLMMessage(
                        "user",
                        f"Task: {task_def['task'][:300]}\n"
                        f"Agent output: {result.output}\n\n"
                        "Is this output sufficient? Reply: accept, retry, or skip.",
                    ),
                ],
                temperature=0.1,
                max_tokens=10,
            )
            verdict = (response.content or "accept").strip().lower().split()[0]
            return verdict if verdict in ("accept", "retry", "skip") else "accept"
        except Exception:
            return "accept"

    # ── Phase 4: Synthesize ───────────────────────────────────

    async def _synthesize(self, goal: str, results: list[SubAgentResult]) -> str:
        successful = [r for r in results if r.success and r.output.strip()]
        if not successful:
            lines = []
            for r in results:
                if r.success:
                    lines.append(f"- **{r.agent_name}**: completed but returned empty output")
                else:
                    lines.append(f"- **{r.agent_name}**: {r.error or 'failed with no error message'}")
            return "The workflow could not be completed:\n\n" + "\n".join(lines)

        agent_reports = "\n\n".join(
            f"### {r.agent_name}\n{r.output}" for r in successful
        )

        try:
            response = await self.llm.chat(
                [
                    LLMMessage(
                        "system",
                        "You are synthesizing the work of several specialized agents into "
                        "a clear, cohesive final answer. Preserve important details; "
                        "eliminate redundancy. Write in first person as if you did the work.",
                    ),
                    LLMMessage(
                        "user",
                        f"Goal: {goal}\n\nAgent reports:\n{agent_reports}\n\n"
                        "Provide a cohesive final answer that fully addresses the goal.",
                    ),
                ],
                max_tokens=2048,
            )
            return response.content or "Workflow completed but synthesis produced no output."
        except Exception as e:
            logger.error(f"[orchestrator] synthesis failed: {e}")
            # Fallback: concatenate outputs
            return "\n\n".join(f"**{r.agent_name}:** {r.output}" for r in successful)


# ── Helpers ───────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


async def _emit(callback: Any, message: str) -> None:
    """Fire an interim_text event if a callback is registered."""
    if callback:
        try:
            await callback({"type": "interim_text", "content": message})
        except Exception:
            pass  # never let a callback error break the workflow
