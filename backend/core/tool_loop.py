"""
Shared tool-calling loop utility.

Used by the dream engine, plan executor, and any other subsystem that needs
to give the LLM free access to skills without going through the full
agent.chat() pipeline (which carries conversation history, self-model
injection, personality state updates, etc.).

The loop follows the same OpenAI-spec message format as agent.chat():
  - Assistant messages echo tool_calls with IDs
  - Tool result messages carry matching tool_call_id
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from backend.core.llm import LLMClient, LLMMessage

if TYPE_CHECKING:
    from backend.skills.registry import SkillRegistry

logger = logging.getLogger("tamagi.tool_loop")


async def run_tool_loop(
    llm: LLMClient,
    skills: "SkillRegistry",
    messages: list[LLMMessage],
    *,
    max_rounds: int = 8,
    event_callback: Callable[[dict], Coroutine] | None = None,
    pending_approvals: dict | None = None,
    is_autonomous: bool = True,
) -> tuple[str, list[str]]:
    """
    Run a tool-calling conversation loop.

    The LLM receives all registered skills as tools and runs until it produces
    a plain-text response (no tool calls) or max_rounds is exhausted.

    Returns:
        (final_content, skills_used)
        - final_content: last non-empty text the LLM produced
        - skills_used: ordered list of skill names that were called
    """
    # Import here to avoid circular import — agent.py also imports tool_loop
    from backend.core.agent import parse_text_tool_calls

    tools = skills.get_openai_tools() if skills.skill_count > 0 else None
    pending = pending_approvals or {}
    skills_used: list[str] = []
    last_content = ""

    for round_num in range(max_rounds):
        try:
            response = await llm.chat(messages, tools=tools)
        except Exception as exc:
            logger.error("tool_loop: LLM call failed on round %d: %s", round_num + 1, exc)
            break

        # Fall back to text-format tool call parsing for models that inline
        # tool calls as plain text rather than using structured tool_calls
        if not response.has_tool_calls and response.content:
            response.tool_calls = parse_text_tool_calls(response.content)

        if not response.tool_calls:
            last_content = response.content or last_content
            break

        if response.content and response.content.strip():
            last_content = response.content.strip()

        # Echo tool calls back in the assistant message — required by OpenAI
        # spec and strictly enforced by llama.cpp's chat template
        assistant_tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in response.tool_calls
        ]
        messages.append(LLMMessage("assistant", "", tool_calls=assistant_tool_calls))

        for tc in response.tool_calls:
            logger.debug("tool_loop round %d: %s(%s)", round_num + 1, tc.name, tc.arguments)
            skills_used.append(tc.name)

            if event_callback:
                try:
                    await event_callback({"type": "tool_start", "name": tc.name, "round": round_num + 1})
                except Exception:
                    pass

            result = await skills.execute(
                tc.name,
                _event_callback=event_callback,
                _pending_approvals=pending,
                _is_autonomous=is_autonomous,
                **tc.arguments,
            )

            if event_callback:
                try:
                    await event_callback({
                        "type": "tool_result",
                        "name": tc.name,
                        "output": str(result.output)[:300] if hasattr(result, "output") else str(result)[:300],
                    })
                except Exception:
                    pass

            messages.append(LLMMessage(
                "tool",
                json.dumps(result.to_dict() if hasattr(result, "to_dict") else {"output": str(result)}),
                tool_call_id=tc.id,
            ))

    return last_content or "...", skills_used
