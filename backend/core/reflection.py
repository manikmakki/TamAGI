"""
TamAGI Reflection Engine

Bayesian update of self-model nodes based on outcome deltas. This is
how the system learns: compare what was predicted against what happened,
trace discrepancies back through the self-model, and propose updates
to confidence scores, strategy weights, and uncertainty levels.

Ported from AURA's reasoning/reflection_engine.py. Schema types
(ActualOutcome, OutcomeDelta, ReflectionResult) are defined inline.
ActionPlan and ModificationProposal are imported from the local modules.

Unlike AURA, proposals are applied directly by the caller (agent.py's
_apply_reflection()) via self_model._apply_update_node() — no pipeline.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .planning_engine import ActionPlan
from .self_model.schemas import (
    CapabilityNode,
    EdgeType,
    NodeType,
    StrategyNode,
    UncertaintyNode,
)
from .self_model.store import ModificationProposal

if TYPE_CHECKING:
    from .self_model.store import SelfModel

logger = logging.getLogger("tamagi.reasoning.reflection")


# ── Inline Schema Types (from aura.schemas) ───────────────────


@dataclass
class ActualOutcome:
    """What actually happened when a plan was executed."""
    plan_id: str = ""
    success: float = 0.0             # 0.0–1.0
    time_taken: float = 0.0          # seconds
    predicted_time: float = 0.0      # seconds
    detail: dict = field(default_factory=dict)
    side_effects: list = field(default_factory=list)
    step_outcomes: list = field(default_factory=list)
    # Execution context — used by ReflectionEngine to distinguish failure causes
    # Keys: "autonomous" (bool), "gap_count" (int), "gap_capabilities" (list[str])
    context: dict = field(default_factory=dict)


@dataclass
class OutcomeDelta:
    """A measured difference between predicted and actual outcome."""
    dimension: str = ""              # "success", "time", "side_effects"
    predicted: float = 0.0
    actual: float = 0.0
    delta: float = 0.0
    significance: float = 0.0       # 0.0–1.0
    contributing_nodes: list = field(default_factory=list)


@dataclass
class ReflectionResult:
    """Output of one reflection cycle."""
    id: str = field(default_factory=lambda: f"refl-{uuid.uuid4().hex[:8]}")
    plan_id: str = ""
    outcome_deltas: list = field(default_factory=list)
    proposed_updates: list = field(default_factory=list)
    lessons: list = field(default_factory=list)
    patterns_detected: list = field(default_factory=list)


# ── Configuration ─────────────────────────────────────────────

SIGNIFICANCE_THRESHOLD = 0.1
DEFAULT_OBSERVATION_WEIGHT = 1.0
RECURRING_FAILURE_THRESHOLD = 3
MIN_ENTROPY = 0.05


# ── Reflection Engine ─────────────────────────────────────────


class ReflectionEngine:
    """Outcome analysis and Bayesian self-model updates.

    Reads from the self-model and the action plan + actual outcome.
    Produces ModificationProposals that the caller applies directly via
    self_model._apply_update_node() (no pipeline in TamAGI).
    """

    def __init__(self, model: "SelfModel") -> None:
        self._model = model
        self._reflection_history: list[ReflectionResult] = []

    # ══════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════

    def reflect(
        self, plan: ActionPlan, outcome: ActualOutcome
    ) -> ReflectionResult:
        """Reflect on a completed plan execution.

        Args:
            plan: The action plan that was executed.
            outcome: What actually happened.

        Returns:
            A ReflectionResult with proposed updates and insights.
            Caller is responsible for applying proposed_updates.
        """
        logger.info(
            "Reflecting on plan %s (goal=%s, success=%.2f)",
            plan.id, plan.goal_id, outcome.success,
        )

        deltas = self._compute_deltas(plan, outcome)
        proposals = self._generate_update_proposals(plan, outcome, deltas)
        patterns = self._detect_patterns(plan, outcome)
        lessons = self._generate_lessons(deltas, outcome)

        result = ReflectionResult(
            plan_id=plan.id,
            outcome_deltas=deltas,
            proposed_updates=proposals,
            lessons=lessons,
            patterns_detected=patterns,
        )

        self._reflection_history.append(result)
        logger.info(
            "Reflection %s complete: %d deltas, %d proposals, %d patterns",
            result.id, len(deltas), len(proposals), len(patterns),
        )
        return result

    @property
    def history(self) -> list[ReflectionResult]:
        return list(self._reflection_history)

    # ══════════════════════════════════════════════════════════
    # Delta Computation
    # ══════════════════════════════════════════════════════════

    def _compute_deltas(
        self, plan: ActionPlan, outcome: ActualOutcome,
    ) -> list[OutcomeDelta]:
        deltas: list[OutcomeDelta] = []
        predicted = plan.predicted_outcome

        # Success delta
        predicted_success = predicted.get("success_probability", 0.5)
        actual_success = outcome.success
        success_delta = actual_success - predicted_success
        if abs(success_delta) > SIGNIFICANCE_THRESHOLD:
            contributing = self._trace_success_contributors(plan)
            deltas.append(OutcomeDelta(
                dimension="success",
                predicted=predicted_success,
                actual=actual_success,
                delta=success_delta,
                significance=min(abs(success_delta), 1.0),
                contributing_nodes=contributing,
            ))

        # Time delta
        if outcome.predicted_time > 0 and outcome.time_taken > 0:
            time_ratio = outcome.time_taken / outcome.predicted_time
            time_delta = time_ratio - 1.0
            if abs(time_delta) > SIGNIFICANCE_THRESHOLD:
                deltas.append(OutcomeDelta(
                    dimension="time",
                    predicted=outcome.predicted_time,
                    actual=outcome.time_taken,
                    delta=time_delta,
                    significance=min(abs(time_delta), 1.0),
                    contributing_nodes=[],
                ))

        # Side effects
        if outcome.side_effects:
            deltas.append(OutcomeDelta(
                dimension="side_effects",
                predicted=0.0,
                actual=float(len(outcome.side_effects)),
                delta=float(len(outcome.side_effects)),
                significance=0.5,
                contributing_nodes=[],
            ))

        return deltas

    def _trace_success_contributors(self, plan: ActionPlan) -> list[str]:
        contributors: list[str] = []

        if plan.strategy_id:
            contributors.append(plan.strategy_id)

        cap_edges = self._model.get_edges(
            source=plan.goal_id, edge_type=EdgeType.REQUIRES.value,
        )
        for edge in cap_edges:
            contributors.append(edge["target"])

        if plan.strategy_id:
            belief_edges = self._model.get_edges(
                target=plan.strategy_id, edge_type=EdgeType.INFORMS.value,
            )
            for edge in belief_edges:
                contributors.append(edge["source"])

        return contributors

    # ══════════════════════════════════════════════════════════
    # Bayesian Update Proposal Generation
    # ══════════════════════════════════════════════════════════

    def _generate_update_proposals(
        self,
        plan: ActionPlan,
        outcome: ActualOutcome,
        deltas: list[OutcomeDelta],
    ) -> list[ModificationProposal]:
        proposals: list[ModificationProposal] = []

        if plan.strategy_id:
            strategy_proposal = self._propose_strategy_update(plan, outcome)
            if strategy_proposal:
                proposals.append(strategy_proposal)

        for delta in deltas:
            if delta.dimension == "success":
                for node_id in delta.contributing_nodes:
                    node = self._model.get_typed_node(node_id)
                    if isinstance(node, CapabilityNode):
                        prop = self._propose_capability_update(node, outcome.success)
                        if prop:
                            proposals.append(prop)

        uncertainty_proposals = self._propose_uncertainty_updates(plan, outcome)
        proposals.extend(uncertainty_proposals)

        return proposals

    def _propose_strategy_update(
        self, plan: ActionPlan, outcome: ActualOutcome,
    ) -> ModificationProposal | None:
        strategy = self._model.get_typed_node(plan.strategy_id)
        if not isinstance(strategy, StrategyNode):
            return None

        # When failures stem from capability gaps in autonomous mode, don't penalize
        # the strategy itself — it may work fine interactively. Clamp the effective
        # observation upward so the Bayesian update is neutral rather than negative.
        effective_observation = outcome.success
        gap_count = outcome.context.get("gap_count", 0)
        is_autonomous = outcome.context.get("autonomous", False)
        if is_autonomous and gap_count > 0:
            # Treat the non-gap steps' success as the real signal for strategy quality
            total_steps = len(outcome.step_outcomes) or 1
            non_gap_success = (outcome.success * total_steps + gap_count) / total_steps
            effective_observation = min(1.0, non_gap_success)

        new_weight = bayesian_update(
            prior=strategy.preference_weight,
            evidence_count=len(strategy.success_history),
            observation=effective_observation,
        )

        rationale = (
            f"Strategy {strategy.id} used in plan {plan.id}. "
            f"Outcome success={outcome.success:.2f}"
        )
        if is_autonomous and gap_count > 0:
            rationale += f" ({gap_count} capability gap(s) in autonomous mode — softened update)"
        rationale += f". Weight {strategy.preference_weight:.3f} → {new_weight:.3f}."

        return ModificationProposal(
            source_component="reflection_engine",
            modification_type="weight_update",
            target=strategy.id,
            current_state=(("preference_weight", strategy.preference_weight),),
            proposed_state=(
                ("preference_weight", round(new_weight, 4)),
                ("success_history_append", effective_observation >= 0.5),
            ),
            rationale=rationale,
        )

    def _propose_capability_update(
        self, capability: CapabilityNode, observation: float,
    ) -> ModificationProposal | None:
        new_confidence = bayesian_update(
            prior=capability.confidence,
            evidence_count=capability.test_count,
            observation=observation,
        )

        if abs(new_confidence - capability.confidence) < 0.005:
            return None

        return ModificationProposal(
            source_component="reflection_engine",
            modification_type="weight_update",
            target=capability.id,
            current_state=(("confidence", capability.confidence),),
            proposed_state=(
                ("confidence", round(new_confidence, 4)),
                ("test_count", capability.test_count + 1),
            ),
            rationale=(
                f"Capability {capability.id} tested with observation={observation:.2f}. "
                f"Confidence {capability.confidence:.3f} → {new_confidence:.3f} "
                f"(evidence count: {capability.test_count} → {capability.test_count + 1})."
            ),
        )

    def _propose_uncertainty_updates(
        self, plan: ActionPlan, outcome: ActualOutcome,
    ) -> list[ModificationProposal]:
        proposals = []

        u_edges = self._model.get_edges(
            target=plan.goal_id, edge_type=EdgeType.EXPLORED_BY.value,
        )

        for edge in u_edges:
            u_node = self._model.get_typed_node(edge["source"])
            if not isinstance(u_node, UncertaintyNode):
                continue

            if outcome.success >= 0.5:
                new_entropy = max(MIN_ENTROPY, u_node.entropy_score - 0.1 * outcome.success)
            else:
                new_entropy = min(1.0, u_node.entropy_score + 0.05)

            if abs(new_entropy - u_node.entropy_score) < 0.005:
                continue

            proposals.append(ModificationProposal(
                source_component="reflection_engine",
                modification_type="weight_update",
                target=u_node.id,
                current_state=(("entropy_score", u_node.entropy_score),),
                proposed_state=(("entropy_score", round(new_entropy, 4)),),
                rationale=(
                    f"Uncertainty domain '{u_node.domain}' explored via goal {plan.goal_id}. "
                    f"Outcome success={outcome.success:.2f}. "
                    f"Entropy {u_node.entropy_score:.3f} → {new_entropy:.3f}."
                ),
            ))

        # Capability gap uncertainty: when autonomous execution hits approval walls,
        # add UncertaintyNodes for each affected capability so the motivation engine
        # and planning engine treat those capabilities as "requires interactive context."
        if outcome.context.get("autonomous") and outcome.context.get("gap_count", 0) > 0:
            import uuid as _uuid
            from datetime import datetime, timezone
            for cap_desc in outcome.context.get("gap_capabilities", ["exec:approve-tier"]):
                u_id = f"u-gap-{_uuid.uuid4().hex[:6]}"
                # Add directly — proposals can only update existing nodes, so we use add
                try:
                    self._model._apply_add_node("uncertainty", {
                        "id": u_id,
                        "domain": f"autonomous_capability:{cap_desc}",
                        "description": (
                            f"Capability '{cap_desc}' requires interactive context — "
                            "blocked during autonomous execution."
                        ),
                        "entropy_score": 0.7,
                        "priority": 0.6,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    })
                    logger.debug("Added capability gap uncertainty node: %s", u_id)
                except Exception as exc:
                    logger.debug("Could not add gap uncertainty node: %s", exc)

        return proposals

    # ══════════════════════════════════════════════════════════
    # Pattern Detection
    # ══════════════════════════════════════════════════════════

    def _detect_patterns(
        self, plan: ActionPlan, outcome: ActualOutcome,
    ) -> list[str]:
        patterns = []

        if len(self._reflection_history) < 2:
            return patterns

        recent = self._reflection_history[-10:]

        if plan.strategy_id:
            strategy_failures = sum(
                1 for r in recent
                if r.plan_id != plan.id
                and any(
                    d.dimension == "success" and d.actual < 0.5
                    for d in r.outcome_deltas
                )
            )
            if strategy_failures >= RECURRING_FAILURE_THRESHOLD:
                patterns.append(
                    f"recurring_failure:strategy:{plan.strategy_id}"
                    f":count={strategy_failures}"
                )

        success_deltas = []
        for r in recent:
            for d in r.outcome_deltas:
                if d.dimension == "success":
                    success_deltas.append(d.delta)

        if len(success_deltas) >= 3:
            avg_delta = sum(success_deltas) / len(success_deltas)
            if avg_delta > 0.2:
                patterns.append(f"systematic_underprediction:avg_delta={avg_delta:.3f}")
            elif avg_delta < -0.2:
                patterns.append(f"systematic_overprediction:avg_delta={avg_delta:.3f}")

        return patterns

    # ══════════════════════════════════════════════════════════
    # Dream Reflection (no ActionPlan required)
    # ══════════════════════════════════════════════════════════

    def reflect_on_dream(
        self,
        activity_name: str,
        domain: str | None = None,
        success: float = 0.5,
    ) -> ReflectionResult:
        """Lightweight post-dream observation pass — no ActionPlan needed.

        Called by the dream engine after every autonomous activity.
        Updates uncertainty entropy for explored domains and nudges
        capability confidence based on activity type.

        Returns a ReflectionResult whose proposed_updates the caller
        applies directly via self_model._apply_update_node().
        """
        proposals: list[ModificationProposal] = []

        # Reduce entropy on the explored uncertainty domain
        if domain:
            for u in self._model.get_uncertainty_map():
                if (domain.lower() in u.domain.lower()
                        or u.domain.lower() in domain.lower()):
                    if success >= 0.5:
                        new_entropy = max(MIN_ENTROPY, u.entropy_score - 0.08 * success)
                    else:
                        new_entropy = min(1.0, u.entropy_score + 0.03)
                    if abs(new_entropy - u.entropy_score) > 0.005:
                        proposals.append(ModificationProposal(
                            source_component="dream_reflection",
                            modification_type="weight_update",
                            target=u.id,
                            current_state=(("entropy_score", u.entropy_score),),
                            proposed_state=(("entropy_score", round(new_entropy, 4)),),
                            rationale=(
                                f"Dream '{activity_name}' explored domain '{domain}'. "
                                f"Entropy {u.entropy_score:.3f} → {new_entropy:.3f} "
                                f"(success={success:.2f})."
                            ),
                        ))
                    break  # One best-match uncertainty per dream

        # Nudge capability confidence by activity type
        activity_cap_map = {
            "explore": "c-006",    # web_search
            "experiment": "c-001", # expression / creativity
            "journal": "c-001",
            "dream": "c-001",
        }
        cap_id = activity_cap_map.get(activity_name)
        if cap_id:
            node = self._model.get_node(cap_id)
            if node:
                prior = node.get("confidence", 0.5)
                count = node.get("test_count", 0)
                new_conf = bayesian_update(prior, count, success)
                if abs(new_conf - prior) > 0.005:
                    proposals.append(ModificationProposal(
                        source_component="dream_reflection",
                        modification_type="weight_update",
                        target=cap_id,
                        current_state=(("confidence", prior),),
                        proposed_state=(
                            ("confidence", round(new_conf, 4)),
                            ("test_count", count + 1),
                        ),
                        rationale=(
                            f"Dream '{activity_name}' nudged capability {cap_id}. "
                            f"Confidence {prior:.3f} → {new_conf:.3f}."
                        ),
                    ))

        result = ReflectionResult(
            plan_id=f"dream-{activity_name}",
            proposed_updates=proposals,
            lessons=[f"Dream '{activity_name}' completed (success={success:.2f})."],
        )
        self._reflection_history.append(result)

        if proposals:
            logger.info(
                "Dream reflection %s: %d proposal(s) (activity=%s, domain=%s)",
                result.id, len(proposals), activity_name, domain or "none",
            )

        return result

    # ══════════════════════════════════════════════════════════
    # Lesson Generation
    # ══════════════════════════════════════════════════════════

    def _generate_lessons(
        self, deltas: list[OutcomeDelta], outcome: ActualOutcome,
    ) -> list[str]:
        lessons = []

        for delta in deltas:
            if delta.dimension == "success":
                if delta.delta > 0:
                    lessons.append(
                        f"Outperformed prediction: actual success "
                        f"({delta.actual:.2f}) exceeded prediction "
                        f"({delta.predicted:.2f}) by {delta.delta:.2f}."
                    )
                else:
                    lessons.append(
                        f"Underperformed prediction: actual success "
                        f"({delta.actual:.2f}) fell short of prediction "
                        f"({delta.predicted:.2f}) by {abs(delta.delta):.2f}."
                    )
            elif delta.dimension == "time":
                if delta.delta > 0:
                    lessons.append(
                        f"Took longer than expected: "
                        f"{delta.actual:.1f}s vs predicted {delta.predicted:.1f}s."
                    )
                else:
                    lessons.append(
                        f"Completed faster than expected: "
                        f"{delta.actual:.1f}s vs predicted {delta.predicted:.1f}s."
                    )
            elif delta.dimension == "side_effects":
                lessons.append(
                    f"Unexpected side effects occurred: {int(delta.actual)} detected."
                )

        if outcome.success >= 0.8:
            lessons.append("High success — strategy and capability estimates were reliable.")
        elif outcome.success <= 0.2:
            lessons.append("Low success — significant recalibration of self-model may be needed.")

        return lessons


# ── Bayesian Update ───────────────────────────────────────────


def bayesian_update(
    prior: float,
    evidence_count: int,
    observation: float,
    observation_weight: float = DEFAULT_OBSERVATION_WEIGHT,
) -> float:
    """Simplified Bayesian confidence update.

    new = (prior * prior_weight + observation * obs_weight) / (prior_weight + obs_weight)

    With few observations each new one has high impact; with many observations,
    each new one shifts the confidence only slightly.

    Returns:
        Updated confidence value, clamped to [0.01, 0.99].
    """
    prior_weight = max(evidence_count, 1)
    numerator = prior * prior_weight + observation * observation_weight
    denominator = prior_weight + observation_weight
    result = numerator / denominator
    return max(0.01, min(0.99, result))
