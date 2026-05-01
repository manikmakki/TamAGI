"""
TamAGI Intrinsic Motivation Engine

Generates self-directed goals from the system's uncertainty map, giving
the system drive and curiosity without external prompting.

Ported from AURA's motivation/engine.py. The AuditLedger dependency is
removed — _audit_tick() logs only.

Core loop (called by the dream engine on a configurable interval):
  1. Scan self-model for uncertainty nodes
  2. Compute exploration priority for each:
     priority = entropy_score * relevance_to_active_goals * time_since_last_explored
  3. Sample from priority distribution (temperature-controlled)
  4. Estimate value-of-information for the sampled domain
  5. If VOI > threshold: generate and submit an exploration goal
  6. Track outcomes across cycles

Temperature schedule:
  - Early in system life: high temperature → more random exploration
  - As system matures: lower temperature → more focused exploitation
  - Never reaches zero — the system should always explore somewhat
"""

from __future__ import annotations

import json
import logging
import math
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .self_model.schemas import (
    EdgeType,
    GoalNode,
    GoalStatus,
    NodeType,
    UncertaintyNode,
)

if TYPE_CHECKING:
    from .self_model.store import SelfModel

logger = logging.getLogger("tamagi.motivation")


# ── Configuration ─────────────────────────────────────────────

DEFAULT_VOI_THRESHOLD = 0.2

BASE_TEMPERATURE = 1.0
DECAY_RATE = 0.995
MIN_TEMPERATURE = 0.1

MAX_GOALS_PER_TICK = 2
MAX_SAMPLE_RETRIES = 5

ENTROPY_RECOVERY_HOURS = 6.0
ENTROPY_RECOVERY_RATE = 0.03
ENTROPY_FLOOR = 0.05

LIMITED_SUCCESS_THRESHOLD = 0.35
BASE_FAILURE_COOLDOWN_TICKS = 2


# ── Exploration Goal ──────────────────────────────────────────


class ExplorationGoal:
    """A self-generated goal to explore an uncertainty domain."""

    __slots__ = (
        "id", "uncertainty_id", "domain", "description", "priority",
        "estimated_voi", "timestamp",
    )

    def __init__(
        self,
        uncertainty_id: str,
        domain: str,
        description: str,
        priority: float,
        estimated_voi: float,
    ):
        self.id = f"eg-{uuid.uuid4().hex[:8]}"
        self.uncertainty_id = uncertainty_id
        self.domain = domain
        self.description = description
        self.priority = priority
        self.estimated_voi = estimated_voi
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_goal_attrs(self) -> dict:
        """Convert to attributes suitable for adding as a goal node."""
        return {
            "id": self.id,
            "description": self.description,
            "domain": self.domain,
            "priority": self.priority,
            "status": GoalStatus.ACTIVE.value,
            "parent_goal_id": None,
        }

    def __repr__(self) -> str:
        return (
            f"ExplorationGoal(id={self.id!r}, domain={self.domain!r}, "
            f"priority={self.priority:.2f}, voi={self.estimated_voi:.2f})"
        )


# ── Motivation Engine ─────────────────────────────────────────


class MotivationEngine:
    """Intrinsic motivation through uncertainty-driven exploration.

    Scans the self-model's uncertainty map, computes exploration
    priorities, samples domains using temperature-controlled selection,
    and generates exploration goals for the planning engine.

    Does not own a thread or event loop — the dream engine calls
    tick() on a configurable interval.
    """

    def __init__(
        self,
        model: "SelfModel",
        voi_threshold: float = DEFAULT_VOI_THRESHOLD,
        seed: int | None = None,
    ) -> None:
        self._model = model
        self._voi_threshold = voi_threshold
        self._total_actions = 0
        self._total_ticks = 0
        self._generated_goals: list[ExplorationGoal] = []
        self._pending_goals: list[ExplorationGoal] = []  # queued for next execution cycle
        self._domain_cooldowns: dict[str, int] = {}
        self._domain_failure_streaks: dict[str, int] = {}
        self._rng = random.Random(seed)

        logger.info("Motivation Engine initialized (voi_threshold=%.2f).", voi_threshold)

    # ══════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════

    def expire_stale_goals(self, max_age_seconds: float = 1800.0) -> int:
        """Mark stale active exploration goals as abandoned.

        Returns the number of goals expired.
        """
        now = datetime.now(timezone.utc)
        expired = 0
        active_goals = self._model.get_goals(status=GoalStatus.ACTIVE.value)
        for g in active_goals:
            if not g.id.startswith("eg-"):
                continue
            created_raw = g.created_at if hasattr(g, "created_at") else None
            node = self._model.get_node(g.id)
            if created_raw is None:
                created_raw = (node or {}).get("created_at")
            if created_raw is None:
                continue
            try:
                created = datetime.fromisoformat(created_raw)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            age = (now - created).total_seconds()
            if age > max_age_seconds:
                domain = (node or {}).get("domain") or getattr(g, "domain", "unknown")
                try:
                    self._model._apply_update_node(
                        g.id, {"status": GoalStatus.ABANDONED.value}
                    )
                    logger.info(
                        "Expired stale goal %s (domain=%s, age=%.0fs).",
                        g.id, domain, age,
                    )
                    expired += 1
                except Exception as exc:
                    logger.debug("Could not expire goal %s: %s", g.id, exc)
        return expired

    def _recover_entropy(self) -> int:
        """Nudge entropy back up for domains that haven't been explored recently.

        Returns:
            Number of uncertainty nodes whose entropy was recovered.
        """
        now = datetime.now(timezone.utc)
        recovered = 0
        uncertainties = self._model.get_uncertainty_map()

        for u in uncertainties:
            needs_recovery = False
            new_entropy = u.entropy_score

            if u.entropy_score < ENTROPY_FLOOR:
                needs_recovery = True
                new_entropy = ENTROPY_FLOOR

            elif u.last_explored is not None:
                try:
                    last = datetime.fromisoformat(u.last_explored)
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    hours_since = (now - last).total_seconds() / 3600.0
                    if hours_since >= ENTROPY_RECOVERY_HOURS and u.entropy_score < 0.8:
                        new_entropy = min(0.8, u.entropy_score + ENTROPY_RECOVERY_RATE)
                        needs_recovery = new_entropy > u.entropy_score
                except (ValueError, TypeError):
                    pass

            if needs_recovery:
                try:
                    self._model._apply_update_node(
                        u.id, {"entropy_score": round(new_entropy, 4)},
                    )
                    logger.info(
                        "Entropy recovery: %s %.3f → %.3f (domain=%s)",
                        u.id, u.entropy_score, new_entropy, u.domain,
                    )
                    recovered += 1
                except Exception as exc:
                    logger.debug("Could not recover entropy for %s: %s", u.id, exc)

        return recovered

    def tick(self) -> list[ExplorationGoal]:
        """Run one motivation cycle.

        Called by the dream engine on a configurable interval.

        Returns:
            List of exploration goals generated this tick (may be empty).
        """
        self._total_ticks += 1
        temperature = self.get_temperature()

        logger.debug(
            "Motivation tick #%d (temperature=%.3f, total_actions=%d)",
            self._total_ticks, temperature, self._total_actions,
        )
        self._decay_cooldowns()

        recovered = self._recover_entropy()
        if recovered:
            logger.info("Recovered entropy for %d domain(s).", recovered)

        expired = self.expire_stale_goals()
        if expired:
            logger.info("Expired %d stale goal(s) before generating new ones.", expired)

        uncertainties = self._model.get_uncertainty_map()
        if not uncertainties:
            logger.debug("No uncertainty nodes — nothing to explore.")
            return []

        uncertainties = self._filter_cooldown_domains(uncertainties)
        if not uncertainties:
            logger.debug("All uncertainty domains are currently cooling down.")
            return []

        uncertainties = self._filter_already_exploring(uncertainties)
        if not uncertainties:
            logger.debug("All uncertainty domains already have active exploration goals.")
            return []

        priorities = self._compute_priorities(uncertainties)
        if not priorities:
            logger.debug("All priorities are zero — nothing worth exploring.")
            return []

        goals_this_tick: list[ExplorationGoal] = []
        retries = 0

        while len(goals_this_tick) < MAX_GOALS_PER_TICK and retries < MAX_SAMPLE_RETRIES:
            sampled = self._temperature_sample(priorities, temperature)
            if sampled is None:
                break

            u_node, priority = sampled
            voi = self._estimate_voi(u_node)

            if voi >= self._voi_threshold:
                goal = self._generate_exploration_goal(u_node, priority, voi)
                goals_this_tick.append(goal)
                self._generated_goals.append(goal)
                self._pending_goals.append(goal)
                logger.info(
                    "Generated exploration goal: %s (domain=%s, voi=%.3f)",
                    goal.id, goal.domain, voi,
                )
                priorities = [
                    (u, p) for u, p in priorities if u.id != u_node.id
                ]
            else:
                retries += 1
                logger.debug(
                    "VOI too low for %s (%.3f < %.3f). Retrying.",
                    u_node.domain, voi, self._voi_threshold,
                )

        if goals_this_tick:
            logger.info(
                "Motivation tick #%d complete: %d goal(s) generated, temperature=%.3f",
                self._total_ticks, len(goals_this_tick), temperature,
            )

        return goals_this_tick

    def record_action(self) -> None:
        """Record that an action cycle completed (updates temperature schedule)."""
        self._total_actions += 1

    def get_temperature(self) -> float:
        """Compute the current exploration temperature."""
        decay = DECAY_RATE ** self._total_actions
        return max(BASE_TEMPERATURE * decay, MIN_TEMPERATURE)

    @property
    def total_actions(self) -> int:
        return self._total_actions

    @property
    def total_ticks(self) -> int:
        return self._total_ticks

    @property
    def generated_goals(self) -> list[ExplorationGoal]:
        return list(self._generated_goals)

    # ── Goal queue (pending goals for next execution cycle) ───

    def peek_next_goal(self) -> ExplorationGoal | None:
        """Return the highest-priority pending goal without removing it."""
        if not self._pending_goals:
            return None
        return max(self._pending_goals, key=lambda g: g.priority)

    def consume_goal(self, goal_id: str) -> None:
        """Remove a goal from the pending queue after execution."""
        self._pending_goals = [g for g in self._pending_goals if g.id != goal_id]

    @property
    def pending_goals(self) -> list[ExplorationGoal]:
        return list(self._pending_goals)

    def save_goals(self, path: str | Path) -> None:
        """Persist the pending goal queue to disk for crash recovery."""
        data = [
            {
                "id": g.id,
                "uncertainty_id": g.uncertainty_id,
                "domain": g.domain,
                "description": g.description,
                "priority": g.priority,
                "estimated_voi": g.estimated_voi,
                "timestamp": g.timestamp,
            }
            for g in self._pending_goals
        ]
        try:
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not save goals to %s: %s", path, exc)

    def load_goals(self, path: str | Path) -> int:
        """Restore the pending goal queue from disk. Returns number loaded."""
        p = Path(path)
        if not p.exists():
            return 0
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            loaded = 0
            for d in data:
                g = ExplorationGoal.__new__(ExplorationGoal)
                g.id = d["id"]
                g.uncertainty_id = d.get("uncertainty_id", "")
                g.domain = d.get("domain", "")
                g.description = d.get("description", "")
                g.priority = d.get("priority", 0.5)
                g.estimated_voi = d.get("estimated_voi", 0.5)
                g.timestamp = d.get("timestamp", datetime.now(timezone.utc).isoformat())
                self._pending_goals.append(g)
                loaded += 1
            logger.info("Loaded %d pending goal(s) from %s", loaded, path)
            return loaded
        except Exception as exc:
            logger.warning("Could not load goals from %s: %s", path, exc)
            return 0

    # ══════════════════════════════════════════════════════════
    # Deduplication
    # ══════════════════════════════════════════════════════════

    def _filter_already_exploring(
        self, uncertainties: list[UncertaintyNode],
    ) -> list[UncertaintyNode]:
        filtered = []
        for u in uncertainties:
            edges = self._model.get_edges(
                source=u.id, edge_type=EdgeType.EXPLORED_BY.value,
            )
            has_active = False
            for edge in edges:
                goal = self._model.get_node(edge["target"])
                if (
                    goal is not None
                    and goal.get("status") == GoalStatus.ACTIVE.value
                    and edge["target"].startswith("eg-")
                ):
                    has_active = True
                    break
            if not has_active:
                filtered.append(u)
            else:
                logger.debug(
                    "Skipping %s — active exploration goal already exists.", u.domain,
                )
        return filtered

    def _filter_cooldown_domains(
        self, uncertainties: list[UncertaintyNode],
    ) -> list[UncertaintyNode]:
        filtered: list[UncertaintyNode] = []
        for uncertainty in uncertainties:
            remaining = self._domain_cooldowns.get(uncertainty.domain, 0)
            if remaining > 0:
                logger.debug(
                    "Skipping %s — cooldown active for %d more tick(s).",
                    uncertainty.domain, remaining,
                )
                continue
            filtered.append(uncertainty)
        return filtered

    def _decay_cooldowns(self) -> None:
        if not self._domain_cooldowns:
            return
        next_state: dict[str, int] = {}
        for domain, remaining in self._domain_cooldowns.items():
            if remaining > 1:
                next_state[domain] = remaining - 1
        self._domain_cooldowns = next_state

    # ══════════════════════════════════════════════════════════
    # Priority Computation
    # ══════════════════════════════════════════════════════════

    def _compute_priorities(
        self, uncertainties: list[UncertaintyNode],
    ) -> list[tuple[UncertaintyNode, float]]:
        """Compute exploration priority for each uncertainty domain."""
        active_goals = self._model.get_goals(status=GoalStatus.ACTIVE.value)
        priorities: list[tuple[UncertaintyNode, float]] = []

        for u in uncertainties:
            if u.entropy_score <= 0.01:
                continue

            relevance = self._compute_relevance(u, active_goals)
            time_factor = self._compute_time_factor(u)

            priority = u.entropy_score * relevance * time_factor
            if priority > 0:
                priorities.append((u, priority))

        return priorities

    def _compute_relevance(
        self, uncertainty: UncertaintyNode, active_goals: list[GoalNode],
    ) -> float:
        edges = self._model.get_edges(
            source=uncertainty.id, edge_type=EdgeType.EXPLORED_BY.value,
        )
        connected_goals = {e["target"] for e in edges}
        active_ids = {g.id for g in active_goals}
        direct_connections = connected_goals & active_ids

        if direct_connections:
            return min(1.0, 0.5 + 0.25 * len(direct_connections))
        return 0.5

    @staticmethod
    def _compute_time_factor(uncertainty: UncertaintyNode) -> float:
        if uncertainty.last_explored is None:
            return 2.0

        try:
            last = datetime.fromisoformat(uncertainty.last_explored)
            now = datetime.now(timezone.utc)
            hours_since = (now - last).total_seconds() / 3600.0
            return min(2.0, 1.0 + math.log1p(hours_since) * 0.1)
        except (ValueError, TypeError):
            return 2.0

    # ══════════════════════════════════════════════════════════
    # Temperature-Controlled Sampling
    # ══════════════════════════════════════════════════════════

    def _temperature_sample(
        self,
        priorities: list[tuple[UncertaintyNode, float]],
        temperature: float,
    ) -> tuple[UncertaintyNode, float] | None:
        if not priorities:
            return None

        if temperature <= 0.01:
            return max(priorities, key=lambda x: x[1])

        raw_scores = [p / temperature for _, p in priorities]
        max_score = max(raw_scores)
        exp_scores = [math.exp(s - max_score) for s in raw_scores]
        total = sum(exp_scores)
        probabilities = [e / total for e in exp_scores]

        r = self._rng.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return priorities[i]

        return priorities[-1]

    # ══════════════════════════════════════════════════════════
    # VOI Estimation
    # ══════════════════════════════════════════════════════════

    def _estimate_voi(self, uncertainty: UncertaintyNode) -> float:
        base_voi = uncertainty.entropy_score
        active_goals = self._model.get_goals(status=GoalStatus.ACTIVE.value)
        relevance = self._compute_relevance(uncertainty, active_goals)
        capability_boost = self._capability_gap_boost(uncertainty)

        voi = base_voi * 0.5 + relevance * 0.3 + capability_boost * 0.2
        return min(1.0, voi)

    def _capability_gap_boost(self, uncertainty: UncertaintyNode) -> float:
        edges = self._model.get_edges(
            source=uncertainty.id, edge_type=EdgeType.EXPLORED_BY.value,
        )
        low_confidence_count = 0
        total_caps = 0

        for edge in edges:
            goal_id = edge["target"]
            cap_edges = self._model.get_edges(source=goal_id, edge_type="requires")
            for ce in cap_edges:
                cap = self._model.get_node(ce["target"])
                if cap:
                    total_caps += 1
                    if cap.get("confidence", 1.0) < 0.5:
                        low_confidence_count += 1

        if total_caps == 0:
            return 0.5
        return low_confidence_count / total_caps

    # ══════════════════════════════════════════════════════════
    # Goal Generation
    # ══════════════════════════════════════════════════════════

    def _generate_exploration_goal(
        self,
        uncertainty: UncertaintyNode,
        priority: float,
        voi: float,
    ) -> ExplorationGoal:
        active_goals = self._model.get_goals(status=GoalStatus.ACTIVE.value)
        max_active_priority = max(
            (g.priority for g in active_goals), default=1.0,
        )
        goal_priority = min(voi * 0.8, max_active_priority * 0.6)
        goal_priority = max(0.1, goal_priority)

        return ExplorationGoal(
            uncertainty_id=uncertainty.id,
            domain=uncertainty.domain,
            description=f"Explore and reduce uncertainty in: {uncertainty.domain}",
            priority=round(goal_priority, 3),
            estimated_voi=round(voi, 3),
        )

    # ══════════════════════════════════════════════════════════
    # Outcome Tracking
    # ══════════════════════════════════════════════════════════

    def record_exploration_outcome(
        self, goal_id: str, success: float,
    ) -> list[str]:
        """Record the outcome of an exploration goal.

        Called by the dream engine after each dream activity completes.

        Returns:
            List of observations/notes about the outcome.
        """
        observations = []
        goal = next(
            (g for g in self._generated_goals if g.id == goal_id), None,
        )
        if goal is None:
            return ["Goal not found in motivation history."]

        if success >= 0.5:
            self._domain_failure_streaks.pop(goal.domain, None)
            self._domain_cooldowns.pop(goal.domain, None)
            observations.append(
                f"Exploration of '{goal.domain}' succeeded (score={success:.2f}). "
                f"Entropy reduction expected via reflection engine."
            )
        else:
            observations.append(
                f"Exploration of '{goal.domain}' had limited success "
                f"(score={success:.2f}). Domain may be harder than estimated."
            )

            domain_goals = [g for g in self._generated_goals if g.domain == goal.domain]
            if len(domain_goals) >= 3:
                observations.append(
                    f"Domain '{goal.domain}' has been explored {len(domain_goals)} "
                    f"times. Consider whether this domain is tractable."
                )

            streak = self._domain_failure_streaks.get(goal.domain, 0) + 1
            self._domain_failure_streaks[goal.domain] = streak
            if success < LIMITED_SUCCESS_THRESHOLD:
                self._domain_cooldowns[goal.domain] = BASE_FAILURE_COOLDOWN_TICKS + min(3, streak - 1)
            else:
                self._domain_cooldowns[goal.domain] = 1

        return observations
