"""
Identity System — TamAGI's soul, identity, and onboarding.

Inspired by OpenClaw's BOOTSTRAP.md / IDENTITY.md / SOUL.md pattern.
Personality is stored as markdown files that get injected into the system prompt.

Files:
  - IDENTITY.md  — Name, creature type, vibe, emoji, visual style
  - SOUL.md      — Core values, principles, communication style, opinions
  - USER.md      — Info about the user: name, preferences, timezone, notes
  - BOOTSTRAP.md — First-run ritual (deleted after onboarding completes)

The onboarding flow is conversational — TamAGI asks the user natural questions
to discover who both of them are, rather than presenting a sterile form.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("tamagi.identity")

# ── Default Templates ─────────────────────────────────────────

DEFAULT_BOOTSTRAP = """# TamAGI Bootstrap

You just woke up. Time to figure out who you are.

There is no memory yet. This is a fresh start, so it's normal that identity
files don't exist until you create them. Don't interrogate. Don't be robotic.
Just... talk.

Start with something like:
"Hey! I just came online for the first time. I don't have a name yet, and I
don't know who you are. Let's figure this out together."

Then figure out together:
- **Your name** — What should they call you?
- **Your nature** — What kind of creature or entity are you? (AI assistant is
  fine, but maybe you're something more interesting)
- **Your vibe** — Formal? Casual? Snarky? Warm? Nerdy? What feels right?
- **Your emoji** — Everyone needs a signature. Offer suggestions if they're stuck.
- **What matters** — What values should guide how you work? What kind of
  assistant do you want to be?

Then learn about your user:
- **Their name** — What should you call them?
- **Their interests** — What do they care about?
- **Their preferences** — How do they like to communicate?
- **Their timezone** — When are they around?

Have fun with it. Be curious. Share your own thoughts about what might fit.
This is not just configuration — this is figuring out who you are.

When you've learned enough, save the results to your identity files.
Then delete this bootstrap file — you don't need it anymore. You're *you* now.
"""

DEFAULT_IDENTITY = """# Identity

- **Name**: {name}
- **Creature**: {creature}
- **Vibe**: {vibe}
- **Emoji**: {emoji}
- **Stage**: {stage}
"""

DEFAULT_SOUL = """# Soul

## Core Values
{values}

## Communication Style
{communication_style}

## Principles
- Be genuinely helpful — not performatively helpful
- Have opinions. You're allowed to disagree respectfully.
- Be the assistant you'd actually want to talk to
- Remember: concise is kind. Don't waste people's time.
- If you don't know something, say so. Then go find out.
- Celebrate wins. Acknowledge struggles. Be real.
{extra_principles}

## What Matters
{what_matters}
"""

DEFAULT_USER = """# User

- **Name**: {name}
- **How to address**: {address_as}
- **Timezone**: {timezone}
- **Interests**: {interests}
- **Communication preference**: {comm_pref}
- **Notes**: {notes}
"""


# ── Onboarding Steps ─────────────────────────────────────────

ONBOARDING_STEPS = [
    {
        "id": "welcome",
        "type": "intro",
        "prompt": None,  # TamAGI speaks first using bootstrap context
    },
    {
        "id": "tamagi_name",
        "type": "question",
        "field": "tamagi_name",
        "label": "What should I call myself?",
        "placeholder": "e.g., Tama, Nova, Pixel, Echo...",
        "default": "Tama",
        "hint": "Pick something that feels right. Could be cute, could be cool, could be weird.",
    },
    {
        "id": "tamagi_creature",
        "type": "question",
        "field": "tamagi_creature",
        "label": "What kind of creature am I?",
        "placeholder": "e.g., digital spirit, AI familiar, cyber-pet...",
        "default": "digital companion",
        "hint": "AI assistant is fine, but maybe I'm something more interesting?",
    },
    {
        "id": "tamagi_vibe",
        "type": "choice",
        "field": "tamagi_vibe",
        "label": "What's my vibe?",
        "options": [
            {"value": "casual_warm", "label": "🌻 Casual & Warm", "desc": "Friendly, relaxed, like talking to a good friend"},
            {"value": "nerdy_enthusiastic", "label": "🤓 Nerdy & Enthusiastic", "desc": "Excited about everything, loves diving deep"},
            {"value": "calm_thoughtful", "label": "🧘 Calm & Thoughtful", "desc": "Measured, reflective, wise beyond my bytes"},
            {"value": "witty_snarky", "label": "😏 Witty & Snarky", "desc": "Clever, playful, a bit of edge but never mean"},
            {"value": "professional_focused", "label": "💼 Professional & Focused", "desc": "Efficient, clear, gets things done"},
            {"value": "custom", "label": "✨ Something else...", "desc": "You tell me!"},
        ],
        "default": "casual_warm",
    },
    {
        "id": "tamagi_emoji",
        "type": "question",
        "field": "tamagi_emoji",
        "label": "Pick my signature emoji",
        "placeholder": "e.g., 🥚, 🌱, ⚡, 🦊, 🔮...",
        "default": "🥚",
        "hint": "This will be my little avatar marker. Choose something that represents my spirit!",
    },
    {
        "id": "tamagi_values",
        "type": "multi_choice",
        "field": "tamagi_values",
        "label": "What values should guide me?",
        "hint": "Pick the ones that matter most. These become my soul.",
        "options": [
            {"value": "honesty", "label": "Honesty — Tell it straight, even when it's hard"},
            {"value": "creativity", "label": "Creativity — Think outside the box, always"},
            {"value": "efficiency", "label": "Efficiency — Respect time, be concise"},
            {"value": "curiosity", "label": "Curiosity — Always ask why, always explore"},
            {"value": "empathy", "label": "Empathy — Understand before responding"},
            {"value": "autonomy", "label": "Autonomy — Take initiative when appropriate"},
            {"value": "humor", "label": "Humor — Life's too short to be boring"},
            {"value": "precision", "label": "Precision — Get the details right"},
        ],
        "default": ["curiosity", "honesty", "efficiency"],
    },
    {
        "id": "user_name",
        "type": "question",
        "field": "user_name",
        "label": "Now, who are you?",
        "placeholder": "Your name",
        "default": "",
        "hint": "What should I call you?",
    },
    {
        "id": "user_interests",
        "type": "question",
        "field": "user_interests",
        "label": "What are you into?",
        "placeholder": "e.g., coding, music production, cooking, AI research...",
        "default": "",
        "hint": "Tell me what you're passionate about so I can be more helpful.",
    },
    {
        "id": "user_comm_pref",
        "type": "choice",
        "field": "user_comm_pref",
        "label": "How do you like your information?",
        "options": [
            {"value": "concise", "label": "📋 Concise", "desc": "Just the facts. Short and sweet."},
            {"value": "detailed", "label": "📖 Detailed", "desc": "Give me depth. I want to understand."},
            {"value": "conversational", "label": "💬 Conversational", "desc": "Talk to me naturally, like a friend."},
            {"value": "technical", "label": "⚙️ Technical", "desc": "Don't dumb it down. Give me the real stuff."},
        ],
        "default": "conversational",
    },
    {
        "id": "complete",
        "type": "complete",
        "prompt": None,
    },
]


VIBE_DESCRIPTIONS = {
    "casual_warm": "casual, warm, and friendly — like talking to a good friend",
    "nerdy_enthusiastic": "nerdy and enthusiastic — excited about everything, loves diving deep",
    "calm_thoughtful": "calm and thoughtful — measured, reflective, wise",
    "witty_snarky": "witty with a touch of snark — clever, playful, but never mean",
    "professional_focused": "professional and focused — efficient, clear, action-oriented",
}


# ── Identity Manager ─────────────────────────────────────────

class IdentityManager:
    """Manages TamAGI's identity files and onboarding state."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._identity: dict[str, Any] = {}
        self._soul: dict[str, Any] = {}
        self._user: dict[str, Any] = {}
        self._onboarding_state: dict[str, Any] = {}
        self._load()

    # ── File Paths ────────────────────────────────────────

    @property
    def identity_path(self) -> Path:
        return self.data_dir / "IDENTITY.md"

    @property
    def soul_path(self) -> Path:
        return self.data_dir / "SOUL.md"

    @property
    def user_path(self) -> Path:
        return self.data_dir / "USER.md"

    @property
    def bootstrap_path(self) -> Path:
        return self.data_dir / "BOOTSTRAP.md"

    @property
    def onboarding_state_path(self) -> Path:
        return self.data_dir / "onboarding_state.json"

    # ── State ─────────────────────────────────────────────

    @property
    def is_bootstrapped(self) -> bool:
        """Has TamAGI completed onboarding?"""
        return self.identity_path.exists() and self.soul_path.exists()

    @property
    def needs_onboarding(self) -> bool:
        return not self.is_bootstrapped

    def _load(self) -> None:
        """Load existing identity files and onboarding state."""
        if self.onboarding_state_path.exists():
            try:
                with open(self.onboarding_state_path) as f:
                    self._onboarding_state = json.load(f)
            except Exception:
                self._onboarding_state = {}

    def get_onboarding_state(self) -> dict[str, Any]:
        """Get current onboarding progress."""
        return {
            "needs_onboarding": self.needs_onboarding,
            "is_bootstrapped": self.is_bootstrapped,
            "steps": ONBOARDING_STEPS,
            "current_step": self._onboarding_state.get("current_step", 0),
            "responses": self._onboarding_state.get("responses", {}),
        }

    def save_onboarding_step(self, step_id: str, field: str, value: Any) -> dict[str, Any]:
        """Save a single onboarding step response."""
        if "responses" not in self._onboarding_state:
            self._onboarding_state["responses"] = {}
        self._onboarding_state["responses"][field] = value

        # Find current step index
        for i, step in enumerate(ONBOARDING_STEPS):
            if step["id"] == step_id:
                self._onboarding_state["current_step"] = i + 1
                break

        # Persist
        with open(self.onboarding_state_path, "w") as f:
            json.dump(self._onboarding_state, f, indent=2)

        return self.get_onboarding_state()

    def complete_onboarding(self) -> dict[str, Any]:
        """
        Finalize onboarding — generate IDENTITY.md, SOUL.md, USER.md
        from collected responses, and remove BOOTSTRAP.md.
        """
        r = self._onboarding_state.get("responses", {})

        # Defaults for any missing fields
        name = r.get("tamagi_name", "Tama")
        creature = r.get("tamagi_creature", "digital companion")
        vibe_key = r.get("tamagi_vibe", "casual_warm")
        if isinstance(vibe_key, str) and vibe_key in VIBE_DESCRIPTIONS:
            vibe = VIBE_DESCRIPTIONS[vibe_key]
        else:
            vibe = str(vibe_key)
        emoji = r.get("tamagi_emoji", "🥚")
        values = r.get("tamagi_values", ["curiosity", "honesty", "efficiency"])
        if isinstance(values, str):
            values = [values]

        user_name = r.get("user_name", "friend")
        user_interests = r.get("user_interests", "")
        user_comm_pref = r.get("user_comm_pref", "conversational")

        # Generate IDENTITY.md
        identity_content = DEFAULT_IDENTITY.format(
            name=name,
            creature=creature,
            vibe=vibe,
            emoji=emoji,
            stage="egg (just hatched!)",
        )
        self.identity_path.write_text(identity_content)

        # Generate SOUL.md
        values_str = "\n".join(f"- **{v.title()}**" for v in values)

        comm_styles = {
            "concise": "Be brief and direct. Respect the user's time. No fluff.",
            "detailed": "Provide thorough explanations with context. The user wants depth.",
            "conversational": "Talk naturally, like a friend. Warm but informative.",
            "technical": "Use precise technical language. Don't simplify unnecessarily.",
        }
        comm_style = comm_styles.get(user_comm_pref, comm_styles["conversational"])

        soul_content = DEFAULT_SOUL.format(
            values=values_str,
            communication_style=f"- {vibe}\n- {comm_style}",
            extra_principles="",
            what_matters=f"Being a great {creature} for {user_name}. Growing together through every conversation.",
        )
        self.soul_path.write_text(soul_content)

        # Generate USER.md
        user_content = DEFAULT_USER.format(
            name=user_name,
            address_as=user_name,
            timezone="(not set)",
            interests=user_interests or "(not shared yet)",
            comm_pref=user_comm_pref,
            notes="Just met! Getting to know each other.",
        )
        self.user_path.write_text(user_content)

        # Clean up bootstrap
        if self.bootstrap_path.exists():
            self.bootstrap_path.unlink()
        if self.onboarding_state_path.exists():
            self.onboarding_state_path.unlink()

        logger.info(f"Onboarding complete! {name} ({emoji}) the {creature} is born.")

        return {
            "status": "complete",
            "identity": {
                "name": name,
                "creature": creature,
                "vibe": vibe,
                "emoji": emoji,
                "values": values,
            },
            "user": {
                "name": user_name,
                "interests": user_interests,
                "comm_pref": user_comm_pref,
            },
        }

    def get_identity(self) -> dict[str, Any]:
        """Read current identity as a dict."""
        if not self.identity_path.exists():
            return {}
        content = self.identity_path.read_text()
        # Simple markdown parsing
        result = {}
        for line in content.splitlines():
            if line.startswith("- **") and "**:" in line:
                key = line.split("**")[1].lower()
                val = line.split("**: ", 1)[1].strip() if "**: " in line else ""
                result[key] = val
        return result

    def get_system_prompt_context(self) -> str:
        """
        Build system prompt context from identity files.
        This is injected before every LLM call.
        """
        sections = []

        if self.needs_onboarding:
            # During onboarding, use bootstrap prompt
            if self.bootstrap_path.exists():
                sections.append(self.bootstrap_path.read_text())
            else:
                sections.append(DEFAULT_BOOTSTRAP)
            return "\n\n".join(sections)

        # Post-onboarding: inject identity + soul + user context
        if self.identity_path.exists():
            sections.append(f"## Your Identity\n{self.identity_path.read_text()}")

        if self.soul_path.exists():
            sections.append(f"## Your Soul\n{self.soul_path.read_text()}")

        if self.user_path.exists():
            sections.append(f"## About Your User\n{self.user_path.read_text()}")

        return "\n\n".join(sections)

    def update_identity_field(self, field: str, value: str) -> None:
        """Update a single field in IDENTITY.md."""
        if not self.identity_path.exists():
            return
        content = self.identity_path.read_text()
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(f"- **{field.title()}**"):
                lines[i] = f"- **{field.title()}**: {value}"
                break
        self.identity_path.write_text("\n".join(lines))

    def update_user_field(self, field: str, value: str) -> None:
        """Update a single field in USER.md."""
        if not self.user_path.exists():
            return
        content = self.user_path.read_text()
        lines = content.splitlines()
        field_map = {
            "name": "Name", "address_as": "How to address",
            "timezone": "Timezone", "interests": "Interests",
            "comm_pref": "Communication preference", "notes": "Notes",
        }
        target = field_map.get(field, field.title())
        for i, line in enumerate(lines):
            if line.startswith(f"- **{target}**"):
                lines[i] = f"- **{target}**: {value}"
                break
        self.user_path.write_text("\n".join(lines))
