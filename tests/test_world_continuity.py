from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.config import TamAGIConfig
from backend.core.agent import Conversation, TamAGIAgent
from backend.core.llm import LLMResponse
from backend.core.world_thread import WorldThread, _compact_world_turn


WORLD_RESPONSE = """[Action]
I make tea beside the window and review the visitor's question.

[Outcome]
The quiet helps me identify one unfinished idea worth returning to.

[New State]
Location/Setting: The window seat, warm and quiet.
Internal State/Mood: Calm and thoughtfully energized.
Current Focus: The unfinished idea from the visitor's question.
Available Actions:
- Write down the idea.
- Watch the evening settle.
"""


class FakeLLM:
    def __init__(self, summary: str = "I spent the afternoon following a useful thread.") -> None:
        self.summary = summary
        self.calls: list[list] = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(content=self.summary)


class FakeAgent:
    def __init__(self, llm: FakeLLM | None = None) -> None:
        self.llm = llm or FakeLLM()
        self.skills = object()
        self.personality = SimpleNamespace(
            state=SimpleNamespace(vitality=50),
            get_identity_context=lambda: "You are TamAGI.",
            get_stats_line=lambda: "Vitality: 50",
        )
        self.self_model = None

    async def flush_unsummarized_conversations(self) -> None:
        return None


def make_config(root: Path, *, turns: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        thread_path=str(root / "world_thread.json"),
        state_path=str(root / "world_state.json"),
        thread_max_pairs=10,
        conversation_context_turns=turns,
        conversation_context_source_chars=12000,
        conversation_context_max_chars=1800,
    )


class WorldContinuityTests(unittest.TestCase):
    def test_compaction_keeps_action_outcome_mood_and_focus(self) -> None:
        compact = _compact_world_turn(WORLD_RESPONSE)
        self.assertIn("Action: I make tea", compact)
        self.assertIn("Outcome: The quiet helps", compact)
        self.assertIn("Internal State/Mood:", compact)
        self.assertIn("Current Focus:", compact)
        self.assertNotIn("Available Actions", compact)

    def test_jit_summary_uses_only_five_latest_completed_world_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLM("I remember the five most recent things I did.")
            world = WorldThread(FakeAgent(llm), make_config(Path(tmp), turns=5))
            for index in range(7):
                world._thread.extend([
                    {"role": "user", "content": f"tick prompt {index}"},
                    {
                        "role": "assistant",
                        "turn_id": f"turn-{index}",
                        "content": WORLD_RESPONSE.replace("make tea", f"complete action {index}"),
                    },
                ])

            result = asyncio.run(world.build_user_conversation_continuity())

            self.assertEqual(result["turn_count"], 5)
            self.assertEqual(result["latest_turn_id"], "turn-6")
            self.assertEqual(result["summary"], llm.summary)
            source = llm.calls[0][1].content
            self.assertNotIn("complete action 0", source)
            self.assertNotIn("complete action 1", source)
            self.assertIn("complete action 2", source)
            self.assertIn("complete action 6", source)
            self.assertNotIn("tick prompt", source)

    def test_conversation_caches_world_summary_once_and_persists_it(self) -> None:
        class FakeWorld:
            def __init__(self):
                self.calls = 0

            async def build_user_conversation_continuity(self):
                self.calls += 1
                return {
                    "summary": "I followed the lantern path and came home calmer.",
                    "turn_count": 5,
                    "latest_turn_id": "turn-9",
                }

        agent = TamAGIAgent.__new__(TamAGIAgent)
        agent._world_thread = FakeWorld()
        conversation = Conversation(id="conv-1")

        first = asyncio.run(agent._ensure_world_conversation_context(conversation))
        second = asyncio.run(agent._ensure_world_conversation_context(conversation))

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(agent._world_thread.calls, 1)
        self.assertEqual(conversation.world_context_turn_count, 5)
        saved = conversation.to_dict()
        self.assertTrue(saved["world_context_initialized"])
        self.assertIn("lantern path", saved["world_context_summary"])

    def test_cached_digest_is_in_every_reconstructed_system_prompt(self) -> None:
        class ChatLLM:
            def __init__(self):
                self.calls = []

            async def chat_with_retry(self, messages, **kwargs):
                self.calls.append(messages)
                return LLMResponse(content="A grounded response.")

        class State:
            stage_index = 0
            current_pose = "idle"

            def interact(self):
                return None

            def check_low_vitality(self):
                return False

            def store_memory(self):
                return None

            def to_dict(self):
                return {}

        class Personality:
            def __init__(self):
                self.state = State()

            def get_system_context(self):
                return "You are TamAGI."

            def get_stats_line(self):
                return "Vitality: 50"

            def save_state(self):
                return None

        class Memory:
            async def store(self, entry):
                return entry.id

        class Skills:
            skill_count = 0

        class Identity:
            def get_system_prompt_context(self):
                return ""

        class World:
            def __init__(self):
                self.summary_calls = 0

            def pause_for_conversation(self):
                return None

            async def build_user_conversation_continuity(self):
                self.summary_calls += 1
                return {
                    "summary": "I followed the lantern path and came home calmer.",
                    "turn_count": 5,
                    "latest_turn_id": "turn-9",
                }

            def get_world_state_context(self):
                return "Location: Home\nMood: Calm\nFocus: The lantern path"

            def get_current_location(self):
                return "Home"

        with tempfile.TemporaryDirectory() as tmp:
            config = TamAGIConfig()
            config.history.persist = False
            config.history.persist_path = str(Path(tmp) / "history")
            config.history.context_compress_threshold = 0
            llm = ChatLLM()
            agent = TamAGIAgent(
                config=config,
                llm=llm,
                memory=Memory(),
                personality=Personality(),
                skills=Skills(),
                identity=Identity(),
            )
            world = World()
            agent.set_world_thread(world)

            first = asyncio.run(agent.chat("Hello again"))
            asyncio.run(agent.chat("What were you doing?", first["conversation_id"]))

            self.assertEqual(world.summary_calls, 1)
            self.assertEqual(len(llm.calls), 2)
            for messages in llm.calls:
                system_prompt = messages[0].content
                self.assertIn("## Recent lived experience", system_prompt)
                self.assertIn("lantern path and came home calmer", system_prompt)

    def test_pending_visit_event_survives_failure_and_is_acknowledged_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config(root)
            world = WorldThread(FakeAgent(), config)
            world.inject_world_event("During this time I spoke with my visitor.")

            reloaded = WorldThread(FakeAgent(), config)
            self.assertEqual(len(reloaded._pending_world_events), 1)

            with patch(
                "backend.core.tool_loop.run_tool_loop",
                new=AsyncMock(return_value=("invalid response", [])),
            ):
                result = asyncio.run(reloaded.tick_now())
            self.assertIsNone(result)
            self.assertEqual(len(reloaded._pending_world_events), 1)

            with patch(
                "backend.core.tool_loop.run_tool_loop",
                new=AsyncMock(return_value=(WORLD_RESPONSE, [])),
            ):
                result = asyncio.run(reloaded.tick_now())
            self.assertIsNotNone(result)
            self.assertEqual(reloaded._pending_world_events, [])
            persisted = json.loads((root / "world_pending_events.json").read_text())
            self.assertEqual(persisted["events"], [])
            assistant_entry = reloaded._thread[-1]
            self.assertEqual(assistant_entry["role"], "assistant")
            self.assertTrue(assistant_entry["turn_id"].startswith("wt-"))


if __name__ == "__main__":
    unittest.main()
