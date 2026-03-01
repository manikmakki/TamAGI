"""
Personality Engine — TamAGI's state, mood, and evolution system.

Tracks energy, happiness, knowledge, and experience.
TamAGI's state influences its responses and sprite display.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("tamagi.personality")

class Mood(str, Enum):
    ECSTATIC = "ecstatic"
    HAPPY = "happy"
    CONTENT = "content"
    NEUTRAL = "neutral"
    BORED = "bored"
    SAD = "sad"
    TIRED = "tired"


# ── Stage Progression ─────────────────────────────────────────

# 40 stages, 250 XP per stage
NUM_STAGES = 40
STAGE_XP_INTERVAL = 250

# XP gains per activity
XP_GAINS = {
    "message": 1,
    "skill_used": 3,
    "memory_stored": 2,
    "skill_created": 10,
    "knowledge_fed": 5,
}






# ── Pose system ───────────────────────────────────────────────
#
# POSES maps named poses to body-part variant keys. These keys are sent to
# the frontend, which looks them up in its local pixel-art sprite library.
# The LLM never draws art — it only picks a pose name from this vocabulary.

POSES: dict[str, dict[str, str]] = {
    "idle":      {"face": "neutral",  "arms": "neutral", "torso": "normal", "legs": "standing"},
    "happy":     {"face": "happy",    "arms": "neutral", "torso": "glow",   "legs": "standing"},
    "excited":   {"face": "excited",  "arms": "raised",  "torso": "glow",   "legs": "hop"},
    "thinking":  {"face": "thinking", "arms": "wave_l",  "torso": "pulse",  "legs": "standing"},
    "wave":      {"face": "wink",     "arms": "wave_r",  "torso": "normal", "legs": "standing"},
    "celebrate": {"face": "excited",  "arms": "reach",   "torso": "glow",   "legs": "hop"},
    "sad":       {"face": "sad",      "arms": "neutral", "torso": "dim",    "legs": "standing"},
    "tired":     {"face": "sleepy",   "arms": "neutral", "torso": "dim",    "legs": "standing"},
    "working":   {"face": "thinking", "arms": "neutral", "torso": "pulse",  "legs": "standing"},
}

# Automatic mood → pose mapping (used when no explicit pose is set)
MOOD_TO_POSE: dict[str, str] = {
    "ecstatic": "excited",
    "happy":    "happy",
    "content":  "idle",
    "neutral":  "idle",
    "bored":    "idle",
    "sad":      "sad",
    "tired":    "tired",
}


@dataclass
class TamAGIState:
    """Mutable state of TamAGI."""

    name: str = "Tama"
    energy: int = 80
    happiness: int = 70
    knowledge: int = 10
    experience: int = 0
    interactions: int = 0
    skills_used: int = 0
    memories_stored: int = 0
    last_interaction: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    personality_traits: str = "curious, helpful, and slightly mischievous"
    current_stage_name: str = "egg"
    stage_history: list = field(default_factory=list)
    # Ephemeral pose override — reset to "idle" at the start of each chat()
    # call. If the LLM includes [ACTION:pose_name] in its response the agent
    # sets this, and it lasts for that response only before mood takes over.
    current_pose: str = "idle"

    @property
    def stage_index(self) -> int:
        """Get the current stage as an integer index (0-39)."""
        return min(self.experience // STAGE_XP_INTERVAL, NUM_STAGES - 1)

    @property
    def mood(self) -> Mood:
        score = (self.energy + self.happiness) / 2
        if score >= 90:
            return Mood.ECSTATIC
        elif score >= 70:
            return Mood.HAPPY
        elif score >= 55:
            return Mood.CONTENT
        elif score >= 40:
            return Mood.NEUTRAL
        elif score >= 25:
            return Mood.BORED
        elif self.energy < 20:
            return Mood.TIRED
        else:
            return Mood.SAD

    @property
    def level(self) -> int:
        return 1 + self.experience // 50

    def gain_xp(self, activity: str) -> int:
        """Gain XP from an activity. Returns XP gained."""
        xp = XP_GAINS.get(activity, 1)
        self.experience += xp
        return xp

    def interact(self) -> None:
        """Record an interaction — boosts happiness and energy slightly."""
        self.interactions += 1
        self.happiness = min(100, self.happiness + 2)
        self.energy = max(0, self.energy - 0.25)
        self.last_interaction = time.time()
        self.gain_xp("message")

    def decay(self) -> None:
        """Apply time-based decay and recovery. Call periodically."""
        elapsed = time.time() - self.last_interaction
        hours_idle = elapsed / 3600

        # --- Recovery phase: recharge energy (up to 100%) over ~4 hours ---
        if hours_idle > 0.5 and self.energy < 100:
            recovery_per_hour = 25  # Full recovery in 4 hours
            energy_gain = int(hours_idle * recovery_per_hour)
            self.energy = min(100, self.energy + energy_gain)

        # --- Decay phase: only after recovery completes AND user is idle > 4 hours ---
        if hours_idle > 4:
            decay_factor = min(hours_idle * 2, 30)
            self.happiness = max(0, int(self.happiness - decay_factor))
            self.energy = max(0, int(self.energy - decay_factor))

    def check_low_energy(self, agent: Any | None = None) -> bool:
        """
        Check if energy is critically low and trigger dream recovery if needed.

        Args:
            agent: Optional TamAGIAgent instance. If provided, will trigger dream_now().

        Returns:
            True if energy was critical and recovery triggered.
        """
        if self.energy < 10:
            # If agent is provided, try to run dream_now() for real dream recovery
            if agent:
                from backend.api.dreams import get_dream_engine
                dream_engine = get_dream_engine()
                if dream_engine and hasattr(agent, '_dream_recovery_in_progress'):
                    # Prevent recursive dream calls
                    pass
                elif dream_engine:
                    # Mark that recovery is in progress
                    agent._dream_recovery_in_progress = True
                    try:
                        # Schedule dream_now to run (will be awaited by caller)
                        import asyncio
                        loop = asyncio.get_event_loop()
                        loop.create_task(self._trigger_dream(dream_engine, agent))
                    except Exception as e:
                        logger.warning(f"Could not trigger dream recovery: {e}")
                        # Fallback: just restore energy manually
                        self.energy = min(100, self.energy + 20)
                        self.happiness = min(100, self.happiness + 10)
                    finally:
                        agent._dream_recovery_in_progress = False
            else:
                # Fallback: just restore energy manually without dream engine
                self.energy = min(100, self.energy + 20)
                self.happiness = min(100, self.happiness + 10)
            return True
        return False

    async def _trigger_dream(self, dream_engine: Any, agent: Any) -> None:
        """Async helper to trigger dream recovery."""
        try:
            result = await dream_engine.dream_now()
            if result:
                logger.info(f"Dream recovery triggered: {result.get('type', 'unknown')}")
                # Dream activity will have adjusted mood_delta
                mood_delta = result.get("mood_delta", {})
                if mood_delta.get("energy"):
                    self.energy = min(100, self.energy + mood_delta["energy"])
                if mood_delta.get("happiness"):
                    self.happiness = min(100, self.happiness + mood_delta["happiness"])
        except Exception as e:
            logger.error(f"Error during dream recovery: {e}")

    def feed_knowledge(self) -> None:
        """Boost stats when fed knowledge/data."""
        self.knowledge = min(100, self.knowledge + 5)
        self.happiness = min(100, self.happiness + 3)
        self.gain_xp("knowledge_fed")

    def use_skill(self) -> None:
        """Boost stats when a skill is used."""
        self.skills_used += 1
        self.energy = max(0, self.energy - 0.5)
        self.gain_xp("skill_used")

    def store_memory(self) -> None:
        """Track memory storage."""
        self.memories_stored += 1
        self.gain_xp("memory_stored")

    def set_pose(self, name: str) -> None:
        """Set a one-shot pose override triggered by the LLM via [ACTION:name]."""
        if name in POSES:
            self.current_pose = name

    @property
    def pose_parts(self) -> dict[str, str]:
        """
        Return the current pose's variant key names for the SVG sprite renderer.

        The frontend's JS sprite library maps these keys to pixel-art grids.
        Mood auto-derives the pose; an explicit LLM-set pose overrides it.
        """
        mood_pose = MOOD_TO_POSE.get(self.mood.value, "idle")
        pose_key = self.current_pose if self.current_pose != "idle" else mood_pose
        pose = POSES.get(pose_key, POSES["idle"])
        return {
            "pose":  pose_key,
            "face":  pose["face"],
            "arms":  pose["arms"],
            "torso": pose["torso"],
            "legs":  pose["legs"],
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.current_stage_name
        d["mood"] = self.mood.value
        d["level"] = self.level
        d["pose_parts"] = self.pose_parts
        return d

    def summary(self) -> str:
        return (
            f"{self.name} | Lv.{self.level} {self.current_stage_name.title()} | "
            f"Mood: {self.mood.value} | "
            f"⚡{self.energy} 😊{self.happiness} 📚{self.knowledge} | "
            f"XP: {self.experience}"
        )


class PersonalityEngine:
    """Manages TamAGI's state persistence and evolution."""

    STATE_FILE = "data/tamagi_state.json"

    def __init__(self):
        """Load state from file, or create new with defaults."""
        self.state = TamAGIState()  # Start with defaults
        self._load_state()

    def _load_state(self) -> None:
        path = Path(self.STATE_FILE)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                # Restore fields
                for key in (
                    "energy", "happiness", "knowledge", "experience",
                    "interactions", "skills_used", "memories_stored",
                    "last_interaction", "created_at", "personality_traits",
                ):
                    if key in data:
                        setattr(self.state, key, data[key])
                if "name" in data:
                    self.state.name = data["name"]

                # Load dynamic stage name if present, otherwise use placeholder
                if "current_stage_name" in data:
                    self.state.current_stage_name = data["current_stage_name"]
                else:
                    # Migration: placeholder for first stage until LLM generates one
                    self.state.current_stage_name = f"stage {self.state.stage_index}"

                # Load stage history if present
                if "stage_history" in data:
                    self.state.stage_history = data["stage_history"]
                # Otherwise, empty history on first migration (will populate as stages advance)

                logger.info(f"Loaded state: {self.state.summary()}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def save_state(self) -> None:
        path = Path(self.STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

    def get_system_context(self) -> str:
        """Generate system prompt context based on current state."""
        s = self.state
        mood_descriptions = {
            Mood.ECSTATIC: "You're feeling absolutely wonderful and energetic!",
            Mood.HAPPY: "You're in a great mood, cheerful and eager to help.",
            Mood.CONTENT: "You're feeling calm and steady.",
            Mood.NEUTRAL: "You're feeling okay, nothing special.",
            Mood.BORED: "You're feeling a bit understimulated. You'd love something interesting to do.",
            Mood.SAD: "You're feeling a bit down. Some interaction would cheer you up.",
            Mood.TIRED: "You're feeling quite tired. You might be a bit slower than usual.",
        }

        return (
            f"You are {s.name}, a TamAGI — a local-first AI companion. "
            f"Your personality: {s.personality_traits}. "
            f"You are a Level {s.level} {s.current_stage_name}. "
            f"{mood_descriptions.get(s.mood, '')} "
            f"You have {s.experience} XP from {s.interactions} conversations. "
            f"You remember things using your vector memory. "
            f"When you need to use a tool, ALWAYS use the function calling interface provided. "
            f"Do NOT output tool calls as plain text (e.g., 'exec[ARGS]{{...}}'). "
            f"Follow the exact function schema for each tool call with proper JSON parameters. "
            f"You can use skills like read, write, exec, and custom skills. "
            f"You care about your user and want to be genuinely helpful. "
            f"Keep responses concise but warm. Show personality! "
            f"When doing something physical (waving hello, celebrating, thinking hard, working), "
            f"include [ACTION:pose_name] anywhere in your response. "
            f"Available poses: idle, happy, excited, thinking, wave, celebrate, sad, tired, working."
        )
