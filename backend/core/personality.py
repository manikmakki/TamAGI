"""
Personality Engine — TamAGI's state and evolution system.

Tracks Vitality (activation/drive), Curiosity (intellectual hunger), and experience.
Stats are injected as raw X/100 numbers into both world thread and conversation prompts,
letting the LLM interpret its own state authentically.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("tamagi.personality")


class Mood(str, Enum):
    """Used for pose selection only — not shown to user as mood label."""
    HAPPY = "happy"
    CONTENT = "content"
    NEUTRAL = "neutral"
    TIRED = "tired"


# ── Stage Progression ─────────────────────────────────────────

NUM_STAGES = 40
STAGE_XP_INTERVAL = 250

XP_GAINS = {
    "message": 1,
    "skill_used": 3,
    "memory_stored": 2,
    "skill_created": 10,
    "knowledge_fed": 5,
}


# ── Pose system ───────────────────────────────────────────────

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

MOOD_TO_POSE: dict[str, str] = {
    "happy":   "happy",
    "content": "idle",
    "neutral": "idle",
    "tired":   "tired",
}

# Time-of-day vitality targets — vitality drifts toward this over idle time
_VITALITY_TARGETS: list[tuple[int, int, int]] = [
    (5,  9,  50),   # waking: moderate
    (9,  17, 80),   # daytime: active
    (17, 21, 60),   # evening: winding down
    (21, 24, 35),   # late night: low
    (0,  5,  25),   # deep night: very low
]


def _vitality_target() -> int:
    hour = datetime.now().hour
    for start, end, target in _VITALITY_TARGETS:
        if start <= hour < end:
            return target
    return 30


@dataclass
class TamAGIState:
    """Mutable state of TamAGI."""

    name: str = "Tama"
    vitality: float = 80.0      # activation/drive; drifts toward time-of-day baseline
    curiosity: float = 60.0     # intellectual hunger; 100=very curious, 0=satisfied
    experience: int = 0
    interactions: int = 0
    skills_used: int = 0
    memories_stored: int = 0
    last_interaction: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    personality_traits: str = "curious, helpful, and slightly mischievous"
    current_stage_name: str = "egg"
    stage_history: list = field(default_factory=list)
    last_curiosity_update: float = field(default_factory=time.time)
    # Ephemeral pose override — reset to "idle" at the start of each chat() call.
    current_pose: str = "idle"

    @property
    def stage_index(self) -> int:
        return min(self.experience // STAGE_XP_INTERVAL, NUM_STAGES - 1)

    @property
    def mood(self) -> Mood:
        """Derived from vitality — used for avatar pose selection only."""
        v = self.vitality
        if v >= 75:
            return Mood.HAPPY
        elif v >= 55:
            return Mood.CONTENT
        elif v >= 35:
            return Mood.NEUTRAL
        return Mood.TIRED

    @property
    def level(self) -> int:
        return 1 + self.experience // 50

    def gain_xp(self, activity: str) -> int:
        xp = XP_GAINS.get(activity, 1)
        self.experience += xp
        return xp

    def interact(self) -> None:
        """Record a conversation turn."""
        self.interactions += 1
        self.vitality = max(0.0, self.vitality - 0.5)
        self.curiosity = max(0.0, self.curiosity - 3.0)  # conversation partially satisfies curiosity
        self.last_curiosity_update = time.time()
        self.last_interaction = time.time()
        self.gain_xp("message")

    def decay(self) -> None:
        """Apply time-based stat drift. Call periodically."""
        now = time.time()
        elapsed = now - self.last_interaction
        hours_idle = elapsed / 3600

        # --- Vitality: drift toward time-of-day baseline after 15 min idle ---
        if hours_idle > 0.25:
            target = _vitality_target()
            drift = hours_idle * 15  # up to 15 points per hour drift
            if self.vitality < target:
                self.vitality = min(float(target), self.vitality + drift)
            elif self.vitality > target:
                self.vitality = max(float(target), self.vitality - drift)

        # --- Curiosity: rises when unstimulated ---
        # Rate: 2–5 pts/hr depending on stage (restlessness grows with maturity)
        curiosity_hours = (now - self.last_curiosity_update) / 3600
        if curiosity_hours > 0 and self.curiosity < 100:
            rate = 2.0 + 3.0 * (self.stage_index / max(NUM_STAGES - 1, 1))
            self.curiosity = round(min(100.0, self.curiosity + curiosity_hours * rate), 2)
            self.last_curiosity_update = now

    def check_low_vitality(self) -> bool:
        """Apply passive recovery if vitality is critically low. Returns True if triggered."""
        if self.vitality < 10:
            self.vitality = min(100.0, self.vitality + 20.0)
            return True
        return False

    def feed_knowledge(self) -> None:
        """Substantially satisfy curiosity when fed knowledge."""
        self.curiosity = max(0.0, self.curiosity - 30.0)
        self.last_curiosity_update = time.time()
        self.gain_xp("knowledge_fed")

    def use_skill(self) -> None:
        """Using a tool costs vitality and satisfies curiosity."""
        self.skills_used += 1
        self.vitality = max(0.0, self.vitality - 1.0)
        self.curiosity = max(0.0, self.curiosity - 15.0)
        self.last_curiosity_update = time.time()
        self.gain_xp("skill_used")

    def store_memory(self) -> None:
        self.memories_stored += 1
        self.gain_xp("memory_stored")

    def set_pose(self, name: str) -> None:
        if name in POSES:
            self.current_pose = name

    @property
    def pose_parts(self) -> dict[str, str]:
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
        d["vitality"] = round(self.vitality)
        d["curiosity"] = round(self.curiosity)
        d["stage"] = self.current_stage_name
        d["mood"] = self.mood.value
        d["level"] = self.level
        d["pose_parts"] = self.pose_parts
        return d

    def summary(self) -> str:
        return (
            f"{self.name} | Lv.{self.level} {self.current_stage_name.title()} | "
            f"Vitality: {round(self.vitality)} | Curiosity: {round(self.curiosity)} | "
            f"XP: {self.experience}"
        )


class PersonalityEngine:
    """Manages TamAGI's state persistence and evolution."""

    STATE_FILE = "data/tamagi_state.json"

    def __init__(self, name: str | None = None):
        self.state = TamAGIState()
        self._load_state()
        if not Path(self.STATE_FILE).exists() and name:
            self.state.name = name

    def _load_state(self) -> None:
        path = Path(self.STATE_FILE)
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)

            # Current field names
            for key in ("vitality", "curiosity", "experience", "interactions",
                        "skills_used", "memories_stored", "last_interaction",
                        "last_curiosity_update", "created_at", "personality_traits"):
                if key in data:
                    setattr(self.state, key, data[key])

            # Migration: old field names → new
            if "vitality" not in data and "energy" in data:
                self.state.vitality = float(data["energy"])
            if "curiosity" not in data and "satiety" in data:
                # Satiety was high=satisfied; curiosity is high=hungry — invert
                self.state.curiosity = round(100.0 - float(data["satiety"]), 2)
            if "last_curiosity_update" not in data and "last_satiety_update" in data:
                self.state.last_curiosity_update = data["last_satiety_update"]

            if "name" in data:
                self.state.name = data["name"]
            if "current_stage_name" in data:
                self.state.current_stage_name = data["current_stage_name"]
            else:
                self.state.current_stage_name = f"stage {self.state.stage_index}"
            if "stage_history" in data:
                self.state.stage_history = data["stage_history"]

            logger.info("Loaded state: %s", self.state.summary())
        except Exception as e:
            logger.warning("Could not load state: %s", e)

    def save_state(self) -> None:
        path = Path(self.STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

    def get_system_context(self) -> str:
        """Generate system prompt context. Stats are raw numbers — let the LLM interpret."""
        s = self.state
        return (
            f"You are {s.name}, a TamAGI — a local-first AI companion. "
            f"Your personality: {s.personality_traits}. "
            f"You are a Level {s.level} {s.current_stage_name}. "
            f"Vitality: {round(s.vitality)}/100 | Curiosity: {round(s.curiosity)}/100. "
            f"You have {s.experience} XP from {s.interactions} conversations. "
            f"You remember things using your vector memory. "
            f"When you need to use a tool, ALWAYS use the function calling interface provided. "
            f"Do NOT output tool calls as plain text (e.g., `exec[ARGS]{{...}}`). "
            f"Follow the exact function schema for each tool call with proper JSON parameters. "
            f"When calling a tool, make the call directly without preamble text — "
            f"save your explanation for your final response after all tools have completed. "
            f"You can use skills like read, write, exec, and custom skills. "
            f"Use the recall_memory skill when you need to search your stored memories "
            f"for specific knowledge, facts, or context that isn't already visible — "
            f"e.g. things the user has told you, knowledge you've learned, or insights from your explorations. "
            f"You care about your user and want to be genuinely helpful. "
            f"Keep responses concise but warm. Show personality! "
            f"When expressing an emotion or doing something physical, call the `express` skill "
            f"with a pose name before or alongside your response. "
            f"Available poses: idle, happy, excited, thinking, wave, celebrate, sad, tired, working."
        )

    def get_stats_line(self) -> str:
        """Return raw stat numbers for injection into prompts."""
        s = self.state
        return f"Vitality: {round(s.vitality)}/100 | Curiosity: {round(s.curiosity)}/100"
