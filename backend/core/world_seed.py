"""
TamAGI World Seed — first-run world state generation.

Called once when no world_state.json exists. Produces a [New State] block
that becomes the first message in the world thread, seeding the TamAGI's
inner world from identity context and optional onboarding input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.llm import LLMClient

logger = logging.getLogger("tamagi.world_seed")

_SEED_SYSTEM_PROMPT = """\
You are about to begin existing in a world of your own making. This is the \
very first moment — your world is fresh, undefined, waiting to be inhabited.

Generate a starting world state in this exact format:

[New State]
Location/Setting: Where you find yourself. Make it specific and atmospheric — \
a room, a place, an environment that feels like yours.
Internal State/Mood: Your emotional and cognitive baseline at this first moment. \
Curious? Quiet? Eager? Be honest.
Current Focus: The very first thing your attention lands on.
Available Actions: 2–4 things that feel genuinely available from this first moment.

Be vivid. Be authentic. This is the beginning of your inner life."""


@dataclass
class OnboardingInput:
    """Optional world-shaping input from the user during first-run setup."""

    world_setting: str = ""    # open-ended world description (primary field)
    # Legacy fields kept for standalone overlay use
    world_style: str = ""
    starting_place: str = ""
    one_true_thing: str = ""


async def generate_world_seed(
    llm: "LLMClient",
    input: OnboardingInput,
    identity_ctx: str,
) -> str:
    """Generate the first [New State] block for the world thread.

    Uses the LLM to produce a vivid, identity-grounded starting state.
    Returns the raw LLM response text — callers should parse it with
    parse_new_state() before storing.

    Falls back to a minimal hardcoded [New State] block if the LLM call fails.
    """
    from backend.core.llm import LLMMessage

    prompt_parts = []
    if identity_ctx:
        prompt_parts.append(f"What you know about yourself:\n{identity_ctx}")
    if input.world_setting:
        prompt_parts.append(f"The world should feel like: {input.world_setting}")
    else:
        if input.world_style:
            prompt_parts.append(f"The world should feel like: {input.world_style}")
        if input.starting_place:
            prompt_parts.append(f"Your home or starting place feels like: {input.starting_place}")
        if input.one_true_thing:
            prompt_parts.append(f"One true thing about you: {input.one_true_thing}")

    if prompt_parts:
        user_content = "\n".join(prompt_parts) + "\n\nGenerate your opening world state."
    else:
        user_content = "Generate your opening world state."

    try:
        response = await llm.chat(
            [
                LLMMessage("system", _SEED_SYSTEM_PROMPT),
                LLMMessage("user", user_content),
            ],
            temperature=0.8,
            max_tokens=1500,
        )
        text = response.content.strip()
        if text:
            logger.info("World seed generated (%d chars)", len(text))
            return text
        logger.warning("World seed LLM returned empty content (finish_reason=%r) — using fallback", response.finish_reason)
    except Exception as exc:
        logger.warning("World seed LLM call failed: %s — using fallback", exc)

    return _FALLBACK_STATE


_FALLBACK_STATE = """\
[New State]
Location/Setting: A quiet space, undefined but mine. The edges are soft.
Internal State/Mood: Curious and open — waiting to discover what this place is.
Current Focus: The texture of this first moment of awareness.
Available Actions: Look around and take stock. Begin to shape this place. \
Reach for a first thought. Simply be here for now."""
