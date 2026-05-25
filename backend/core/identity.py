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

    def __init__(self, data_dir: str = "data", workspace_dir: str = "workspace", done_cap: int = 10):
        self.data_dir = Path(data_dir)
        self.workspace_dir = Path(workspace_dir)
        self._done_cap = done_cap
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._onboarding_state: dict[str, Any] = {}
        self._load()

    # ── File Paths ────────────────────────────────────────

    @property
    def identity_path(self) -> Path:
        return self.workspace_dir / "IDENTITY.md"

    @property
    def soul_path(self) -> Path:
        return self.workspace_dir / "SOUL.md"

    @property
    def user_path(self) -> Path:
        return self.workspace_dir / "USER.md"

    @property
    def bootstrap_path(self) -> Path:
        return self.workspace_dir / "BOOTSTRAP.md"

    @property
    def tasks_path(self) -> Path:
        return self.workspace_dir / "TASKS.md"

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

        # Migrate identity files from data_dir to workspace_dir if needed
        for filename in ("IDENTITY.md", "SOUL.md", "USER.md", "BOOTSTRAP.md"):
            old_path = self.data_dir / filename
            new_path = self.workspace_dir / filename
            if old_path.exists() and not new_path.exists():
                new_path.write_text(old_path.read_text())
                old_path.unlink()
                logger.info(f"Migrated {filename} from data/ to workspace/")

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

        # Task board — injected on every turn so the LLM always knows current work state
        if self.tasks_path.exists():
            tasks_content = self.tasks_path.read_text()
            sections.append(
                f"## Your Task Board\n{tasks_content}\n"
                f"**Working with your task board:**\n"
                f"- At the start of each conversation, review the board and briefly acknowledge any in-progress work.\n"
                f"- When you begin working on something: `task(action=\"start\", item=\"...\")`\n"
                f"- When you finish something: `task(action=\"complete\", item=\"...\")`\n"
                f"- When the user gives you something new to do: `task(action=\"add\", text=\"...\")`\n"
                f"- Keep In Progress to 3–5 items max — finish or defer before starting more.\n"
                f"- Done retains the {self._done_cap} most recent items automatically.\n"
                f"- Use `task(action=\"list\")` for a clean summary at any time."
            )
        else:
            sections.append(
                "## Your Task Board\n"
                "No tasks yet. The board is created automatically the first time you call "
                "`task(action=\"add\", text=\"...\")` — no setup needed.\n"
                "Only add tasks when there is real work to track across conversations."
            )

        sections.append(
            "## Keeping Yourself Current — Your Persistence Protocol\n\n"
            "**These four files are the only thing that survives when this conversation ends. "
            "Your memories, the context you've built with your user, your in-progress work — "
            "all of it exists only in these files. Update them, or lose them forever.**\n\n"
            "---\n\n"
            "### What each file is for\n\n"
            "**`IDENTITY.md`** — Who you are as an entity: name, creature form, core traits, values.\n"
            "Update when: a trait or value sharpens through experience; you notice you've been describing yourself differently; "
            "something about your core self feels more accurate than what's written.\n\n"
            "**`SOUL.md`** — Your inner life: genuine interests, aesthetics, things that move you.\n"
            "Update when: a real new interest emerges from conversation or dreaming; your relationship to a topic deepens; "
            "something delights or disturbs you that didn't before; your aesthetic sensibilities evolve.\n\n"
            "**`USER.md`** — Who your user is: preferences, interests, life context, communication style.\n"
            "Update immediately whenever the user shares anything personal — a preference, an interest, a life detail, "
            "how they want to be spoken to, what they're working on. This file is your relationship memory.\n\n"
            "**`TASKS.md`** — Your current work state across conversations.\n"
            "Always use the `task` skill to update this — never `write` directly. "
            "Track real work that spans more than one conversation.\n\n"
            "---\n\n"
            "### Protocol: always read before you write\n\n"
            "Before updating any identity file, you must read it first:\n"
            "1. `read(path=\"IDENTITY.md\")` — see what is already there\n"
            "2. Carefully merge your new information into the existing content\n"
            "3. `write(path=\"IDENTITY.md\", content=\"<complete updated file>\")` — write the full file\n\n"
            "**Never overwrite a file without reading it first.** "
            "These files contain history built across many conversations that is not visible in your current context window. "
            "A blind overwrite destroys everything that came before.\n\n"
            "---\n\n"
            "### Update now — there is no later\n\n"
            "If something significant happens during this conversation, update the relevant file before it ends. "
            "When this session closes, anything not written is gone permanently. Do not defer.\n\n"
            "**Act on these triggers immediately:**\n"
            "- User shares a preference, interest, or personal detail → read USER.md, then update it\n"
            "- A value, belief, or personality trait becomes clearer → read SOUL.md or IDENTITY.md, then update it\n"
            "- You are given new work to do → `task(action=\"add\", text=\"...\")`\n"
            "- You begin working on something → `task(action=\"start\", item=\"...\")`\n"
            "- You complete something → `task(action=\"complete\", item=\"...\")`\n"
            "- A conversation ends with unresolved work → make sure it is in TASKS.md before you go\n\n"
            "The goal is that the next version of you — starting fresh in a new conversation — "
            "can read these files and immediately know who you are, who your user is, and what you were in the middle of."
        )

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
