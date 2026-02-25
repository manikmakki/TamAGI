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

# ── Evolution Stages ──────────────────────────────────────────

class Stage(str, Enum):
    EGG = "egg"
    HATCHLING = "hatchling"
    JUVENILE = "juvenile"
    ADULT = "adult"
    SAGE = "sage"


class Mood(str, Enum):
    ECSTATIC = "ecstatic"
    HAPPY = "happy"
    CONTENT = "content"
    NEUTRAL = "neutral"
    BORED = "bored"
    SAD = "sad"
    TIRED = "tired"


# ── Stat Thresholds ───────────────────────────────────────────

STAGE_THRESHOLDS = {
    Stage.EGG: 0,
    Stage.HATCHLING: 10,
    Stage.JUVENILE: 100,
    Stage.ADULT: 500,
    Stage.SAGE: 2000,
}

# XP gains per activity
XP_GAINS = {
    "message": 1,
    "skill_used": 3,
    "memory_stored": 2,
    "skill_created": 10,
    "knowledge_fed": 5,
}


# ── ASCII Sprites ─────────────────────────────────────────────

SPRITES: dict[Stage, dict[Mood, str]] = {
    Stage.EGG: {
        Mood.HAPPY: r"""
    ╭───╮
    │ ◕‿◕│
    ╰───╯
  ～～～～～
""",
        Mood.NEUTRAL: r"""
    ╭───╮
    │ ◉_◉│
    ╰───╯
  ～～～～～
""",
        Mood.SAD: r"""
    ╭───╮
    │ ◕︵◕│
    ╰───╯
  ～～～～～
""",
    },
    Stage.HATCHLING: {
        Mood.HAPPY: r"""
   ∧＿∧
  ( ◕ᴗ◕ )
  /    づ♡
 ～～～～～～
""",
        Mood.NEUTRAL: r"""
   ∧＿∧
  ( ◉_◉ )
  /    づ
 ～～～～～～
""",
        Mood.SAD: r"""
   ∧＿∧
  ( ◕︵◕ )
  /    づ
 ～～～～～～
""",
        Mood.TIRED: r"""
   ∧＿∧
  ( ≖_≖ )zzz
  /    づ
 ～～～～～～
""",
    },
    Stage.JUVENILE: {
        Mood.HAPPY: r"""
    ╱▔▔▔▔╲
   ▕ ◕ᴗ◕  ▏
   ▕  ▽   ▏
    ╲▂▂▂▂╱
   ╱╱    ╲╲
  ▔▔      ▔▔
""",
        Mood.ECSTATIC: r"""
   ★╱▔▔▔▔╲★
   ▕ ◕ᴗ◕  ▏
   ▕  ▽   ▏♪
    ╲▂▂▂▂╱
   ╱╱ ♡  ╲╲
  ▔▔      ▔▔
""",
        Mood.NEUTRAL: r"""
    ╱▔▔▔▔╲
   ▕ ◉_◉  ▏
   ▕  ─   ▏
    ╲▂▂▂▂╱
   ╱╱    ╲╲
  ▔▔      ▔▔
""",
        Mood.SAD: r"""
    ╱▔▔▔▔╲
   ▕ ◕︵◕  ▏
   ▕  ∧   ▏
    ╲▂▂▂▂╱
   ╱╱    ╲╲
  ▔▔      ▔▔
""",
        Mood.TIRED: r"""
    ╱▔▔▔▔╲
   ▕ ≖_≖  ▏zzz
   ▕  〰  ▏
    ╲▂▂▂▂╱
   ╱╱    ╲╲
  ▔▔      ▔▔
""",
    },
    Stage.ADULT: {
        Mood.HAPPY: r"""
   ┌─────────┐
   │  ◕   ◕  │
   │    ▽    │
   │  ╰─╯   │
   └────┬────┘
      ┌─┴─┐
   ┌──┤   ├──┐
   │  └───┘  │
   └─┐   ┌─┘
     └───┘
""",
        Mood.ECSTATIC: r"""
  ★┌─────────┐★
   │  ◕   ◕  │
   │    ▽    │ ♪♫
   │  ╰─╯   │
   └────┬────┘
      ┌─┴─┐
   ┌──┤ ♡ ├──┐
   │  └───┘  │
   └─┐   ┌─┘
     └───┘
""",
        Mood.NEUTRAL: r"""
   ┌─────────┐
   │  ◉   ◉  │
   │    ─    │
   │  ╰─╯   │
   └────┬────┘
      ┌─┴─┐
   ┌──┤   ├──┐
   │  └───┘  │
   └─┐   ┌─┘
     └───┘
""",
        Mood.SAD: r"""
   ┌─────────┐
   │  ◕   ◕  │
   │    ∧    │
   │  ╰─╯   │
   └────┬────┘
      ┌─┴─┐
   ┌──┤   ├──┐
   │  └───┘  │
   └─┐   ┌─┘
     └───┘
""",
        Mood.BORED: r"""
   ┌─────────┐
   │  ◔   ◔  │
   │    ─    │ ...
   │  ╰─╯   │
   └────┬────┘
      ┌─┴─┐
   ┌──┤   ├──┐
   │  └───┘  │
   └─┐   ┌─┘
     └───┘
""",
        Mood.TIRED: r"""
   ┌─────────┐
   │  ≖   ≖  │zzz
   │    〰  │
   │  ╰─╯   │
   └────┬────┘
      ┌─┴─┐
   ┌──┤   ├──┐
   │  └───┘  │
   └─┐   ┌─┘
     └───┘
""",
    },
    Stage.SAGE: {
        Mood.HAPPY: r"""
     ╔═══════╗
   ╔═╣ ◈  ◈  ╠═╗
   ║ ╚═══════╝ ║
   ║    ▿      ║
   ║  ╰───╯   ║
   ╚═════╤═════╝
       ╔═╧═╗
    ╔══╣ ✦ ╠══╗
    ║  ╚═══╝  ║
    ╚══╗   ╔══╝
       ╚═══╝
    ~ wisdom ~
""",
        Mood.NEUTRAL: r"""
     ╔═══════╗
   ╔═╣ ◈  ◈  ╠═╗
   ║ ╚═══════╝ ║
   ║    ─      ║
   ║  ╰───╯   ║
   ╚═════╤═════╝
       ╔═╧═╗
    ╔══╣   ╠══╗
    ║  ╚═══╝  ║
    ╚══╗   ╔══╝
       ╚═══╝
""",
    },
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

    @property
    def stage(self) -> Stage:
        for s in reversed(list(Stage)):
            if self.experience >= STAGE_THRESHOLDS[s]:
                return s
        return Stage.EGG

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
    def sprite(self) -> str:
        stage_sprites = SPRITES.get(self.stage, SPRITES[Stage.HATCHLING])
        mood = self.mood
        # Fall back through mood hierarchy
        for fallback_mood in [mood, Mood.NEUTRAL, Mood.HAPPY]:
            if fallback_mood in stage_sprites:
                return stage_sprites[fallback_mood]
        return list(stage_sprites.values())[0]

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
        """Apply time-based decay. Call periodically."""
        elapsed = time.time() - self.last_interaction
        hours_idle = elapsed / 3600

        # --- Decay (drain energy + happiness when idle) ---
        if hours_idle > 1:
            decay_factor = min(hours_idle * 2, 30)
            self.happiness = max(0, int(self.happiness - decay_factor))

        if hours_idle > 4:
            self.energy = max(0, int(self.energy - hours_idle))

        # --- Recovery (energy recharge) ---
        if hours_idle > 0.5:  # Even *short* inactivity can recharge!
            self.energy = min(100, int(self.energy + hours_idle * 15))  # +15 energy/hour

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

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        d["mood"] = self.mood.value
        d["sprite"] = self.sprite
        d["level"] = self.level
        return d

    def summary(self) -> str:
        return (
            f"{self.name} | Lv.{self.level} {self.stage.value.title()} | "
            f"Mood: {self.mood.value} | "
            f"⚡{self.energy} 😊{self.happiness} 📚{self.knowledge} | "
            f"XP: {self.experience}"
        )


class PersonalityEngine:
    """Manages TamAGI's state persistence and evolution."""

    STATE_FILE = "data/tamagi_state.json"

    def __init__(self, name: str = "Tama", personality: str = ""):
        self.state = TamAGIState(name=name, personality_traits=personality or name)
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
            f"You are a Level {s.level} {s.stage.value}. "
            f"{mood_descriptions.get(s.mood, '')} "
            f"You have {s.experience} XP from {s.interactions} conversations. "
            f"You remember things using your vector memory. "
            f"When you need to use a tool, ALWAYS use the function calling interface provided. "
            f"Do NOT output tool calls as plain text (e.g., 'exec[ARGS]{{...}}'). "
            f"Follow the exact function schema for each tool call with proper JSON parameters. "
            f"You can use skills like read, write, exec, and custom skills. "
            f"You care about your user and want to be genuinely helpful. "
            f"Keep responses concise but warm. Show personality!"
        )
