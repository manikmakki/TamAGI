"""
TamAGI Planning Engine

Goal-decomposition via graph traversal + strategy selection via weighted
scoring. Produces structured ActionPlan objects — not natural language.

Ported from AURA's reasoning/planning_engine.py. ActionPlan, ActionStep,
and ActionStepType are defined inline (stripped from aura.schemas).

The Planning Engine does NOT invoke the LLM directly. It operates on the
self-model graph to assess capability readiness, score strategies, and
decompose plans. The LLM is optionally wired for query/code generation
at plan time — set via set_llm().

LLM interface expected: any object with a method
    chat(messages: list[dict], temperature: float) → object with .content: str
"""

from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .self_model.schemas import (
    CapabilityNode,
    EdgeType,
    GoalNode,
    NodeType,
    StrategyNode,
    UncertaintyNode,
)

if TYPE_CHECKING:
    from .self_model.store import SelfModel

logger = logging.getLogger("tamagi.reasoning.planning")


# ── Inline Schema Types (from aura.schemas) ───────────────────


class ActionStepType(enum.Enum):
    GENERATE_CODE = "generate_code"
    EXECUTE_CODE = "execute_code"
    QUERY_SELF_MODEL = "query_self_model"
    COMMUNICATE = "communicate"
    EXPLORE = "explore"
    MODIFY_SELF = "modify_self"
    TOOL_USE = "tool_use"
    SUB_GOAL = "sub_goal"
    BASH = "bash"
    WEB_FETCH = "web_fetch"
    WEB_SEARCH = "web_search"
    CREATE_TOOL = "create_tool"


@dataclass
class ActionStep:
    id: str = field(default_factory=lambda: f"step-{uuid.uuid4().hex[:8]}")
    step_type: str = ActionStepType.EXECUTE_CODE.value
    description: str = ""
    spec: dict = field(default_factory=dict)
    required_capabilities: list = field(default_factory=list)
    predicted_outcome: dict = field(default_factory=dict)
    rollback: str = ""
    depends_on: list = field(default_factory=list)


@dataclass
class ActionPlan:
    id: str = field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}")
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    goal_id: str = ""
    strategy_id: str = ""
    steps: list = field(default_factory=list)
    predicted_outcome: dict = field(default_factory=dict)
    confidence: float = 0.5
    capability_risks: list = field(default_factory=list)
    uncertainty_flags: list = field(default_factory=list)


# ── Configuration ─────────────────────────────────────────────

DEFAULT_CAPABILITY_THRESHOLD = 0.4

SCORING_WEIGHTS = {
    "success_rate": 0.30,
    "preference_weight": 0.25,
    "capability_alignment": 0.25,
    "uncertainty_cost": 0.20,
}


# ── Planning Engine ───────────────────────────────────────────


class PlanningEngine:
    """Goal decomposition and strategy selection over the self-model.

    Reads from the self-model to assess capabilities, score strategies,
    and produce structured action plans. Never mutates the self-model.
    """

    def __init__(
        self,
        model: "SelfModel",
        capability_threshold: float = DEFAULT_CAPABILITY_THRESHOLD,
        llm=None,
    ) -> None:
        self._model = model
        self._capability_threshold = capability_threshold
        self._llm = llm

    def set_llm(self, llm) -> None:
        """Wire in an LLM for optional query/code generation at plan time.

        Expected interface:
            llm.chat(messages: list[dict], temperature: float) → obj with .content: str
        """
        self._llm = llm

    # ══════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════

    async def create_plan(self, goal_id: str) -> ActionPlan:
        """Create an action plan for the given goal.

        Args:
            goal_id: ID of the goal to plan for.

        Returns:
            A complete ActionPlan.

        Raises:
            KeyError: If goal_id doesn't exist in the self-model.
        """
        goal_node = self._model.get_typed_node(goal_id)
        if goal_node is None or not isinstance(goal_node, GoalNode):
            raise KeyError(f"Goal {goal_id!r} not found in self-model.")

        logger.info("Planning for goal: %s (%s)", goal_id, goal_node.description)

        required_caps = self._get_required_capabilities(goal_id)
        cap_assessment = self._assess_capabilities(required_caps)

        strategies = self._model.query_strategies(goal_id=goal_id)
        if not strategies:
            strategies = self._model.query_strategies()

        if not strategies:
            logger.warning("No strategies found for goal %s. Creating exploratory plan.", goal_id)
            return await self._create_exploratory_plan(goal_id, goal_node, cap_assessment)

        scored = self._score_strategies(strategies, cap_assessment)
        best_strategy, best_score = scored[0]

        logger.info("Selected strategy: %s (score=%.3f)", best_strategy.id, best_score)

        steps = await self._decompose_strategy(best_strategy, goal_node, cap_assessment)
        predicted_outcome = self._predict_outcome(
            goal_node, best_strategy, cap_assessment, steps,
        )

        plan = ActionPlan(
            goal_id=goal_id,
            strategy_id=best_strategy.id,
            steps=steps,
            predicted_outcome=predicted_outcome,
            confidence=best_score,
            capability_risks=[
                {"capability_id": cid, "confidence": conf}
                for cid, conf, available in cap_assessment
                if not available
            ],
            uncertainty_flags=self._get_relevant_uncertainties(goal_id),
        )

        logger.info(
            "Plan %s created: %d steps, confidence=%.3f, %d capability risks",
            plan.id, len(steps), plan.confidence, len(plan.capability_risks),
        )
        return plan

    # ══════════════════════════════════════════════════════════
    # Capability Assessment
    # ══════════════════════════════════════════════════════════

    def _get_required_capabilities(self, goal_id: str) -> list[CapabilityNode]:
        caps = []
        edges = self._model.get_edges(source=goal_id, edge_type=EdgeType.REQUIRES.value)
        for edge in edges:
            typed = self._model.get_typed_node(edge["target"])
            if isinstance(typed, CapabilityNode):
                caps.append(typed)
        return caps

    def _assess_capabilities(
        self, capabilities: list[CapabilityNode]
    ) -> list[tuple[str, float, bool]]:
        assessment = []
        for cap in capabilities:
            available = cap.confidence >= self._capability_threshold
            assessment.append((cap.id, cap.confidence, available))
            if not available:
                logger.debug(
                    "Capability %s below threshold: %.2f < %.2f",
                    cap.id, cap.confidence, self._capability_threshold,
                )
        return assessment

    # ══════════════════════════════════════════════════════════
    # Strategy Scoring
    # ══════════════════════════════════════════════════════════

    def _score_strategies(
        self,
        strategies: list[StrategyNode],
        cap_assessment: list[tuple[str, float, bool]],
    ) -> list[tuple[StrategyNode, float]]:
        available_cap_ids = {cid for cid, _, avail in cap_assessment if avail}
        avg_cap_confidence = (
            sum(conf for _, conf, _ in cap_assessment) / len(cap_assessment)
            if cap_assessment else 0.5
        )

        scored = []
        for strategy in strategies:
            if strategy.success_history:
                success_rate = sum(strategy.success_history) / len(strategy.success_history)
            else:
                success_rate = 0.5

            pref_weight = strategy.preference_weight

            if strategy.applicable_contexts:
                matched = sum(
                    1 for ctx in strategy.applicable_contexts
                    if ctx in available_cap_ids
                )
                cap_alignment = matched / len(strategy.applicable_contexts)
            else:
                cap_alignment = avg_cap_confidence

            uncertainty_cost = 1.0 - self._estimate_strategy_uncertainty(strategy)

            score = (
                SCORING_WEIGHTS["success_rate"] * success_rate
                + SCORING_WEIGHTS["preference_weight"] * pref_weight
                + SCORING_WEIGHTS["capability_alignment"] * cap_alignment
                + SCORING_WEIGHTS["uncertainty_cost"] * uncertainty_cost
            )
            scored.append((strategy, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _estimate_strategy_uncertainty(self, strategy: StrategyNode) -> float:
        uncertainties = self._model.get_uncertainty_map()
        if not uncertainties or not strategy.applicable_contexts:
            return 0.0

        related_entropy = []
        for u in uncertainties:
            domain_lower = u.domain.lower()
            for ctx in strategy.applicable_contexts:
                if ctx.lower() in domain_lower or domain_lower in ctx.lower():
                    related_entropy.append(u.entropy_score)
                    break

        if not related_entropy:
            return 0.0
        return sum(related_entropy) / len(related_entropy)

    # ══════════════════════════════════════════════════════════
    # Plan Decomposition
    # ══════════════════════════════════════════════════════════

    async def _decompose_strategy(
        self,
        strategy: StrategyNode,
        goal: GoalNode,
        cap_assessment: list[tuple[str, float, bool]],
    ) -> list[ActionStep]:
        steps: list[ActionStep] = []
        unavailable_caps = [cid for cid, _, avail in cap_assessment if not avail]

        if unavailable_caps:
            steps.extend(await self._decompose_explore(goal, unavailable_caps))

        required_cap_ids = [cid for cid, _, _ in cap_assessment]
        steps.append(ActionStep(
            step_type=ActionStepType.EXECUTE_CODE.value,
            description=f"Execute strategy '{strategy.description}' for goal '{goal.description}'",
            spec={
                "strategy_id": strategy.id,
                "goal_id": goal.id,
                "strategy_description": strategy.description,
            },
            required_capabilities=required_cap_ids,
            predicted_outcome={"goal_progress": True, "success_probability": strategy.preference_weight},
            depends_on=[s.id for s in steps],
        ))

        steps.append(ActionStep(
            step_type=ActionStepType.MODIFY_SELF.value,
            description="Reflect on execution results and update self-model",
            spec={"action": "reflect"},
            required_capabilities=[],
            predicted_outcome={"self_model_updated": True},
            depends_on=[steps[-1].id],
        ))

        return steps

    async def _decompose_explore(
        self,
        goal: GoalNode,
        missing_capabilities: list[str],
    ) -> list[ActionStep]:
        """Decompose an exploration intent into real executable steps."""
        steps: list[ActionStep] = []

        query = await self._generate_search_query(goal.description, missing_capabilities)

        search_step = ActionStep(
            step_type=ActionStepType.WEB_SEARCH.value,
            description=f"Search for information to address capability gaps: {missing_capabilities}",
            spec={"query": query, "max_results": 5},
            predicted_outcome={"search_results_obtained": True},
        )
        steps.append(search_step)

        fetch_step = ActionStep(
            step_type=ActionStepType.WEB_FETCH.value,
            description="Fetch and summarise the most relevant search result",
            spec={
                "url": "",
                "context": goal.description,
                "from_search_step": search_step.id,
            },
            predicted_outcome={"content_retrieved": True},
            depends_on=[search_step.id],
        )
        steps.append(fetch_step)

        if any("code" in cap.lower() or "bash" in cap.lower() or "exec" in cap.lower()
               for cap in missing_capabilities):
            bash_step = ActionStep(
                step_type=ActionStepType.BASH.value,
                description="Probe local environment for relevant tools",
                spec={"command": "python3 --version && pip list --format=columns 2>/dev/null | head -20"},
                predicted_outcome={"environment_probed": True},
                depends_on=[fetch_step.id],
            )
            steps.append(bash_step)

        return steps

    async def _generate_search_query(
        self, goal_description: str, missing_caps: list[str],
    ) -> str:
        """Generate a search query for an exploration goal.

        Uses the LLM if available; otherwise constructs a keyword query
        from the goal description.
        """
        if self._llm is not None:
            try:
                from backend.core.llm import LLMMessage
                prompt = (
                    f"Generate a concise web search query (10 words max) to help with: "
                    f"{goal_description}. "
                    f"Focus on: {', '.join(missing_caps)}. "
                    f"Reply with only the query string."
                )
                response = await self._llm.chat(
                    [LLMMessage("user", prompt)],
                    temperature=0.1,
                )
                query = response.content.strip().strip('"').strip("'")
                if query:
                    return query
            except Exception as exc:
                logger.debug("LLM query generation failed: %s", exc)

        tokens = goal_description.lower().split()[:6]
        return " ".join(tokens)

    async def _create_exploratory_plan(
        self,
        goal_id: str,
        goal: GoalNode,
        cap_assessment: list[tuple[str, float, bool]],
    ) -> ActionPlan:
        """Create a plan when no strategies are available."""
        missing_caps = [cid for cid, _, avail in cap_assessment if not avail]

        explore_steps = await self._decompose_explore(goal, missing_caps or [goal_id])

        reflect_step = ActionStep(
            step_type=ActionStepType.MODIFY_SELF.value,
            description="Reflect on exploratory results",
            spec={"action": "reflect"},
            predicted_outcome={"self_model_updated": True},
            depends_on=[explore_steps[-1].id] if explore_steps else [],
        )

        steps = explore_steps + [reflect_step]

        return ActionPlan(
            goal_id=goal_id,
            strategy_id="",
            steps=steps,
            predicted_outcome={"success_probability": 0.3, "exploratory": True},
            confidence=0.3,
            capability_risks=[
                {"capability_id": cid, "confidence": conf}
                for cid, conf, available in cap_assessment
                if not available
            ],
        )

    # ══════════════════════════════════════════════════════════
    # Outcome Prediction
    # ══════════════════════════════════════════════════════════

    def _predict_outcome(
        self,
        goal: GoalNode,
        strategy: StrategyNode,
        cap_assessment: list[tuple[str, float, bool]],
        steps: list[ActionStep],
    ) -> dict:
        if strategy.success_history:
            base_success = sum(strategy.success_history) / len(strategy.success_history)
        else:
            base_success = 0.5

        if cap_assessment:
            avg_confidence = sum(c for _, c, _ in cap_assessment) / len(cap_assessment)
            cap_factor = avg_confidence
        else:
            cap_factor = 0.5

        success_probability = base_success * 0.6 + cap_factor * 0.4
        success_probability = max(0.05, min(0.95, success_probability))

        return {
            "success_probability": round(success_probability, 3),
            "estimated_steps": len(steps),
            "capability_readiness": round(cap_factor, 3),
            "strategy_track_record": round(base_success, 3),
        }

    # ══════════════════════════════════════════════════════════
    # Uncertainty Scanning
    # ══════════════════════════════════════════════════════════

    def _get_relevant_uncertainties(self, goal_id: str) -> list[str]:
        results = []
        edges = self._model.get_edges(target=goal_id, edge_type=EdgeType.EXPLORED_BY.value)
        for edge in edges:
            typed = self._model.get_typed_node(edge["source"])
            if isinstance(typed, UncertaintyNode):
                results.append(f"{typed.id}:{typed.domain}(entropy={typed.entropy_score})")
        return results
