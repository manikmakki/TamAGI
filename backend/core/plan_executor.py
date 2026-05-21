"""
Plan Executor — Steps through an ActionPlan, observing real results.

The gap PlanExecutor fills: the PlanningEngine generates structured plans,
but previously they were only formatted as prose injected into the system
prompt. The LLM improvised from there. PlanExecutor actually steps through
the plan, executes each ActionStep via the appropriate skill, emits live
interim_text events so the user can see progress, and returns an ActualOutcome
for the ReflectionEngine to learn from.

Usage (from agent.py):
    executor = PlanExecutor(agent=self, event_callback=cb, plan=active_plan)
    outcome = await executor.execute()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from backend.core.planning_engine import ActionPlan, ActionStep, ActionStepType
from backend.core.reflection import ActualOutcome

if TYPE_CHECKING:
    from backend.core.agent import TamAGIAgent

logger = logging.getLogger("tamagi.plan_executor")


@dataclass
class StepResult:
    step_id: str
    success: bool
    output: str
    error: str | None = None
    skipped: bool = False


_CAPABILITY_GAP_TAG = "capability_gap:approval_required_in_autonomous_context"

_STEP_CAPABILITY_TERMS: dict[str, list[str]] = {
    ActionStepType.BASH.value:          ["bash", "exec", "command", "shell"],
    ActionStepType.EXECUTE_CODE.value:  ["code", "execution", "python"],
    ActionStepType.GENERATE_CODE.value: ["code", "generation", "python"],
    ActionStepType.WEB_SEARCH.value:    ["web", "search"],
    ActionStepType.WEB_FETCH.value:     ["web", "fetch", "retrieval"],
    ActionStepType.CREATE_TOOL.value:   ["tool", "creation"],
}


class PlanExecutor:
    """Execute an ActionPlan step-by-step with live WebSocket feedback.

    When is_autonomous=True (called from the dream engine), there is no user
    present to approve commands. Approve-tier steps are auto-denied and tagged
    as capability gaps. These flow back through ActualOutcome to the
    ReflectionEngine, which Bayesian-updates the strategy's success_rate
    downward — teaching AURA to avoid those strategies when running alone.
    """

    def __init__(
        self,
        agent: "TamAGIAgent",
        event_callback,
        plan: ActionPlan,
        depth: int = 0,
        is_autonomous: bool = False,
    ) -> None:
        self._agent = agent
        self._cb = event_callback
        self._plan = plan
        self._depth = depth  # max 1 level of recursive SUB_GOAL
        self._is_autonomous = is_autonomous

    # ── Public entry point ────────────────────────────────────

    async def execute(self) -> ActualOutcome:
        start = time.time()
        steps = self._plan.steps
        n = len(steps)
        goal_summary = self._plan.predicted_outcome.get("summary", f"plan {self._plan.id}")
        estimated = self._plan.predicted_outcome.get("estimated_steps", n)

        await self._emit(f"Starting plan: {goal_summary} ({n} step{'s' if n != 1 else ''})")

        # Build dependency order — topological sort
        ordered = self._topo_sort(steps)

        upstream_outputs: dict[str, str] = {}  # step_id → output text
        step_results: list[StepResult] = []

        for i, step in enumerate(ordered, start=1):
            await self._emit(f"Step {i}/{n}: {step.description}")
            sr = await self._execute_step(step, upstream_outputs)
            step_results.append(sr)

            if sr.skipped:
                await self._emit(f"⏭ Skipped step {i}: {sr.output}")
            elif sr.success:
                preview = sr.output[:120].replace("\n", " ")
                await self._emit(f"✓ Step {i}: {preview}{'…' if len(sr.output) > 120 else ''}")
                upstream_outputs[step.id] = sr.output
            else:
                await self._emit(f"✗ Step {i} failed: {sr.error or sr.output}")
                # Continue best-effort; downstream steps will lack context

        elapsed = time.time() - start
        successes = sum(1 for r in step_results if r.success)
        gap_count = sum(1 for r in step_results if r.error == _CAPABILITY_GAP_TAG)
        success_rate = successes / len(step_results) if step_results else 1.0

        # Capability gaps in autonomous mode reduce success score — this is the
        # learning signal: the ReflectionEngine Bayesian-updates the strategy's
        # success_rate downward, so AURA avoids these strategies when running alone.
        if gap_count:
            gap_penalty = gap_count / len(step_results)
            success_rate = max(0.0, success_rate - gap_penalty)
            await self._emit(
                f"⚠ {gap_count} step(s) needed user approval — unavailable in autonomous mode. "
                "Flagged as capability gaps for AURA to learn from."
            )

        await self._emit(
            f"Plan complete — {successes}/{n} steps succeeded in {elapsed:.1f}s"
        )

        # Collect gap capability descriptions for ReflectionEngine uncertainty nodes
        gap_capabilities = [
            sr.step_id for sr in step_results if sr.error == _CAPABILITY_GAP_TAG
        ]

        return ActualOutcome(
            plan_id=self._plan.id,
            success=success_rate,
            time_taken=elapsed,
            predicted_time=float(estimated) * 5.0,  # rough 5 s/step estimate
            side_effects=[sr.step_id for sr in step_results if sr.success],
            step_outcomes=[
                {
                    "step_id": sr.step_id,
                    "success": sr.success,
                    "output": sr.output[:200],
                    **({"capability_gap": True} if sr.error == _CAPABILITY_GAP_TAG else {}),
                }
                for sr in step_results
            ],
            context={
                "autonomous": self._is_autonomous,
                "gap_count": gap_count,
                "gap_capabilities": gap_capabilities,
            },
        )

    def summary_text(self) -> str:
        """Human-readable summary of what the executor produced (for final LLM response)."""
        return f"I executed the plan ({len(self._plan.steps)} steps). See the reasoning panel for details."

    # ── Step dispatcher ───────────────────────────────────────

    async def _execute_step(
        self, step: ActionStep, upstream: dict[str, str]
    ) -> StepResult:
        step_type = step.step_type
        ctx = self._build_context(step, upstream)

        confidence, cap_desc = self._capability_confidence(step_type)
        if confidence < 0.25 and cap_desc:
            await self._emit(
                f"⚠ Low confidence in '{cap_desc}' ({confidence:.0%}) — proceeding carefully"
            )

        _TOOL_LOOP_STEPS = {
            ActionStepType.BASH.value,
            ActionStepType.WEB_SEARCH.value,
            ActionStepType.WEB_FETCH.value,
            ActionStepType.TOOL_USE.value,
            ActionStepType.GENERATE_CODE.value,
            ActionStepType.EXECUTE_CODE.value,
            ActionStepType.EXPLORE.value,
        }

        try:
            if step_type in _TOOL_LOOP_STEPS:
                return await self._run_with_tool_loop(step, ctx)
            elif step_type == ActionStepType.COMMUNICATE.value:
                return await self._run_communicate(step)
            elif step_type == ActionStepType.QUERY_SELF_MODEL.value:
                return await self._run_query_self_model(step)
            elif step_type == ActionStepType.MODIFY_SELF.value:
                return await self._run_modify_self(step)
            elif step_type == ActionStepType.SUB_GOAL.value:
                return await self._run_sub_goal(step, ctx)
            elif step_type == ActionStepType.CREATE_TOOL.value:
                return await self._run_create_tool(step, ctx)
            else:
                return StepResult(
                    step_id=step.id, success=False,
                    output="", error=f"Unknown step type: {step_type}"
                )
        except Exception as exc:
            logger.exception("Step %s raised: %s", step.id, exc)
            return StepResult(step_id=step.id, success=False, output="", error=str(exc))

    # ── Step implementations ──────────────────────────────────

    async def _run_with_tool_loop(self, step: ActionStep, ctx: str) -> StepResult:
        import json
        from backend.core.llm import LLMMessage
        from backend.core.tool_loop import run_tool_loop

        system = (
            "You are executing a step in an action plan. "
            "Use the available tools to complete the task described. "
            "Be concise and focus on the objective."
        )
        prompt = step.description
        if step.spec:
            relevant = {k: v for k, v in step.spec.items() if v}
            if relevant:
                prompt += f"\n\nStep parameters: {json.dumps(relevant)}"
        if ctx:
            prompt += f"\n\nContext from previous steps:\n{ctx}"

        try:
            content, _ = await run_tool_loop(
                self._agent.llm,
                self._agent.skills,
                [
                    LLMMessage("system", system),
                    LLMMessage("user", prompt),
                ],
                event_callback=self._cb,
                pending_approvals=self._agent._pending_approvals,
                is_autonomous=self._is_autonomous,
            )
            return StepResult(step_id=step.id, success=True, output=content)
        except Exception as exc:
            return StepResult(step_id=step.id, success=False, output="", error=str(exc))

    async def _run_communicate(self, step: ActionStep) -> StepResult:
        message = step.spec.get("message") or step.description
        await self._emit(f"💬 {message}")
        return StepResult(step_id=step.id, success=True, output=message)

    async def _run_query_self_model(self, step: ActionStep) -> StepResult:
        if not self._agent.self_model:
            return StepResult(step_id=step.id, success=False, output="", error="Self-model not available")
        query = step.spec.get("query") or step.description
        nodes = self._agent.self_model.search_nodes(query=query, limit=5)
        output = "\n".join(
            f"[{n.get('node_type','?')}] {n.get('id','?')}: {n.get('description', n.get('belief',''))}"
            for n in nodes
        )
        return StepResult(step_id=step.id, success=True, output=output or "(no matching nodes)")

    async def _run_modify_self(self, step: ActionStep) -> StepResult:
        observation = step.spec.get("observation")
        if not observation:
            return StepResult(step_id=step.id, success=True, output="(no observation to record)")
        if not self._agent.self_model:
            return StepResult(step_id=step.id, success=False, output="", error="Self-model not available")

        import uuid as _uuid
        from datetime import datetime, timezone

        # Write the observation as a BeliefNode — durable, queryable, Bayesian-updatable
        belief_id = f"b-auto-{_uuid.uuid4().hex[:8]}"
        confidence = float(step.spec.get("confidence", 0.6))
        try:
            self._agent.self_model._apply_add_node("belief", {
                "id": belief_id,
                "description": observation[:200],
                "confidence": confidence,
                "evidence_count": 1,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            if not self._agent.self_model.auto_wire_node(belief_id):
                self._agent.self_model._apply_remove_node(belief_id)
                return StepResult(
                    step_id=step.id,
                    success=True,
                    output="Self-model: observation discarded (no matching nodes to wire).",
                )
            if hasattr(self._agent, "_detect_belief_conflicts"):
                self._agent._detect_belief_conflicts(belief_id, observation[:200])
            return StepResult(
                step_id=step.id,
                success=True,
                output=f"Self-model updated: belief '{belief_id}' added — {observation[:100]}",
            )
        except Exception as exc:
            return StepResult(step_id=step.id, success=False, output="", error=str(exc))

    async def _run_sub_goal(self, step: ActionStep, ctx: str) -> StepResult:
        if self._depth >= 1:
            return StepResult(
                step_id=step.id, success=False, output="",
                error="Sub-goal depth limit reached (max 1 level)",
            )
        goal_desc = step.spec.get("goal") or step.description
        if not self._agent.planning_engine:
            return StepResult(step_id=step.id, success=False, output="", error="Planning engine not available")
        try:
            goal_id = self._agent._create_transient_goal(goal_desc)
            sub_plan = await self._agent.planning_engine.create_plan(goal_id)
            sub_executor = PlanExecutor(
                agent=self._agent,
                event_callback=self._cb,
                plan=sub_plan,
                depth=self._depth + 1,
                is_autonomous=self._is_autonomous,
            )
            outcome = await sub_executor.execute()
            return StepResult(
                step_id=step.id,
                success=outcome.success >= 0.5,
                output=f"Sub-goal outcome: {outcome.success:.0%} success",
            )
        except Exception as exc:
            return StepResult(step_id=step.id, success=False, output="", error=str(exc))

    async def _run_create_tool(self, step: ActionStep, ctx: str) -> StepResult:
        result = await self._run_with_tool_loop(step, ctx)
        # Trigger rediscovery so any newly written skill is available immediately
        try:
            from pathlib import Path
            self._agent.skills.discover_custom_skills(Path("workspace/skills"))
        except Exception:
            pass
        return result

    # ── Helpers ───────────────────────────────────────────────

    def _capability_confidence(self, step_type: str) -> tuple[float, str]:
        """Return (confidence, description) for the capability most relevant to step_type.

        Returns (1.0, '') when no self-model is available or no capability matches,
        so the caller's threshold check is a no-op.
        """
        sm = getattr(self._agent, "self_model", None)
        if not sm:
            return 1.0, ""
        terms = _STEP_CAPABILITY_TERMS.get(step_type, [])
        if not terms:
            return 1.0, ""
        best_cap = None
        best_score = 0
        for cap in sm.query_capabilities():
            desc = cap.description.lower()
            score = sum(1 for t in terms if t in desc)
            if score > best_score:
                best_score = score
                best_cap = cap
        if best_cap and best_score > 0:
            return best_cap.confidence, best_cap.description
        return 1.0, ""

    async def _skill(self, name: str, **kwargs) -> Any:
        return await self._agent.skills.execute(
            name=name,
            _event_callback=self._cb,
            _pending_approvals=self._agent._pending_approvals,
            _is_autonomous=self._is_autonomous,
            **kwargs,
        )

    async def _emit(self, text: str) -> None:
        if self._cb:
            try:
                await self._cb({"type": "interim_text", "content": text})
            except Exception:
                pass

    def _build_context(self, step: ActionStep, upstream: dict[str, str]) -> str:
        if not step.depends_on:
            return ""
        parts = [upstream[dep] for dep in step.depends_on if dep in upstream]
        return "\n\n".join(parts)

    def _topo_sort(self, steps: list) -> list:
        """Return steps in dependency-safe execution order."""
        id_to_step = {s.id: s for s in steps}
        visited: set[str] = set()
        order: list = []

        def visit(sid: str) -> None:
            if sid in visited:
                return
            visited.add(sid)
            step = id_to_step.get(sid)
            if step:
                for dep in step.depends_on:
                    visit(dep)
                order.append(step)

        for s in steps:
            visit(s.id)
        return order


