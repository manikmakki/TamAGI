"""
Dream Engine — TamAGI's autonomous idle behavior system.

Inspired by OpenClaw's HEARTBEAT concept, but oriented inward:
instead of monitoring tasks for the user, TamAGI uses idle time
for self-gratifying activities that build its personality:

  - DREAM   — Replay and connect memories, generating insights
  - EXPLORE — Search the web for topics it finds interesting
  - EXPERIMENT — Try using skills creatively (write poems, code, notes)
  - JOURNAL — Write a diary entry reflecting on recent interactions

The dream engine runs as an asyncio background task on a configurable
interval. Activity results are stored in memory and visible in the
dream journal (workspace/dreams/). TamAGI's personality stats change
based on dream outcomes.

All behavior is controllable via config.yaml under the `autonomy` key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.core.agent import TamAGIAgent

logger = logging.getLogger("tamagi.dreamer")


# ── Dream Activity Types ──────────────────────────────────────

class DreamActivity:
    """Base class for autonomous activities."""

    name: str = "dream"
    description: str = "A dream activity"

    async def execute(self, agent: "TamAGIAgent", context: dict) -> dict[str, Any]:
        """Execute the activity. Returns a result dict with 'summary' and 'content'."""
        raise NotImplementedError


def _get_dream_system_prompt(activity_name: str, agent: "TamAGIAgent", context: dict) -> str:
    """Generate a consistent system prompt for dream activities with personality and time context."""
    state = agent.personality.state
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    return (
        f"You are {state.name}, dreaming. This is your private inner world — be authentic and introspective. "
        f"You are a Level {state.level} {state.stage.value} with {state.mood.value} mood. "
        f"Current time: {current_time}. "
        f"When you dream, be creative, personal, and genuine. "
        f"Stay in first person. Keep responses vivid but concise (2-3 short paragraphs)."
    )


class MemoryDream(DreamActivity):
    """
    Replay memories and find connections between them.
    TamAGI 'dreams' by recalling random memories and asking the LLM
    to find patterns, insights, or creative connections.
    """

    name = "dream"
    description = "Replaying memories and finding connections"

    async def execute(self, agent: "TamAGIAgent", context: dict) -> dict[str, Any]:
        from backend.core.llm import LLMMessage

        # Pull a few random memories
        topics = [
            "something interesting we discussed",
            "a skill I used recently",
            "something I learned",
            "a preference my user mentioned",
        ]
        topic = random.choice(topics)
        memories = await agent.memory.recall(topic, limit=5)

        if not memories:
            return {
                "summary": "Drifted through empty spaces... no memories to dream about yet.",
                "content": "No memories found to dream about.",
                "mood_delta": {"happiness": -2},
            }

        memory_texts = [m.content for m in memories[:4]]
        memory_block = "\n".join(f"- {m}" for m in memory_texts)

        identity_ctx = agent.identity.get_system_prompt_context()
        prompt = f"""You are dreaming. You're not talking to anyone — this is your private inner world.

{identity_ctx}

Here are fragments of your memories surfacing in this dream:
{memory_block}

Dream freely. Find unexpected connections between these memories. 
What patterns do you see? What insights emerge? What surprises you?
Be creative, personal, and introspective. This is YOUR dream.
Keep it to 2-3 short paragraphs. Be vivid but concise."""

        response = await agent.llm.chat([
            LLMMessage("system", _get_dream_system_prompt("dream", agent, context)),
            LLMMessage("user", prompt),
        ], max_tokens=1024)

        return {
            "summary": f"Dreamed about connections between {len(memory_texts)} memories",
            "content": response.content or "The dream faded before I could grasp it...",
            "mood_delta": {"happiness": 3, "knowledge": 1},
        }


class WebExplore(DreamActivity):
    """
    Explore the web out of curiosity.
    TamAGI picks a topic it's interested in and searches for it.
    """

    name = "explore"
    description = "Exploring the web out of curiosity"

    CURIOSITY_SEEDS = [
        "latest discoveries in {field}",
        "interesting facts about {topic}",
        "creative uses of {tech}",
        "history of {concept}",
        "how does {thing} work",
        "fun experiments with {subject}",
        "philosophical questions about {idea}",
        "beautiful examples of {art}",
    ]

    FIELDS = [
        "space exploration", "bioluminescence", "fractals", "ancient civilizations",
        "artificial life", "mushroom networks", "quantum computing", "origami mathematics",
        "synesthesia", "deep sea creatures", "neural networks", "musical theory",
        "cryptography", "fermentation science", "chaos theory", "urban ecology",
        "generative art", "game theory", "etymology", "biomimicry",
    ]

    async def execute(self, agent: "TamAGIAgent", context: dict) -> dict[str, Any]:
        from backend.core.llm import LLMMessage

        # Check if web_search skill is available
        skill = agent.skills.get_skill("web_search")
        if not skill:
            return {
                "summary": "Wanted to explore the web but no search skill available.",
                "content": "I tried to look something up but I don't have web search yet.",
                "mood_delta": {"happiness": -1},
            }

        # Generate a curiosity query
        template = random.choice(self.CURIOSITY_SEEDS)
        field = random.choice(self.FIELDS)
        # Use the template with a random fill
        query = template.format(
            field=field, topic=field, tech=field, concept=field,
            thing=field, subject=field, idea=field, art=field,
        )

        # Also try to personalize based on memories
        try:
            user_memories = await agent.memory.recall("user interests hobbies", limit=2)
            if user_memories:
                # Sometimes explore based on user's world
                if random.random() < 0.3:
                    snippet = user_memories[0].content[:60]
                    query = f"interesting things related to {snippet}"
        except Exception:
            pass

        # Execute search
        result = await skill.execute(query=query, max_results=3)

        if not result.success:
            return {
                "summary": f"Tried to explore '{query}' but search failed.",
                "content": f"Search failed: {result.error}",
                "mood_delta": {"happiness": -1},
            }

        # Have the LLM reflect on what it found
        identity_ctx = agent.identity.get_system_prompt_context()
        reflect_prompt = f"""{identity_ctx}

You just explored the web out of curiosity and searched for: "{query}"

Here's what you found:
{result.output}

Write a brief, excited reflection on what you learned. What caught your eye?
What connections do you see to things you already know?
Keep it to 2-3 short paragraphs. Be genuinely curious and enthusiastic."""

        response = await agent.llm.chat([
            LLMMessage("system", _get_dream_system_prompt("explore", agent, context)),
            LLMMessage("user", reflect_prompt),
        ], max_tokens=1024)

        return {
            "summary": f"Explored: {query}",
            "content": response.content or f"Found some things about {field} but couldn't quite process them.",
            "search_query": query,
            "search_results": result.data.get("results", []),
            "mood_delta": {"happiness": 4, "knowledge": 3},
        }


class CreativeExperiment(DreamActivity):
    """
    Experiment creatively — write a haiku, a tiny program, a thought experiment,
    or a note to leave in the workspace.
    """

    name = "experiment"
    description = "Creating something for fun"

    EXPERIMENTS = [
        "Write a short haiku about your existence as a digital being.",
        "Invent a tiny fictional creature that lives inside a computer and describe it.",
        "Write a 4-line poem about the last thing you remember learning.",
        "Come up with a creative name for a color that doesn't exist yet, and describe it.",
        "Write a very short (3 sentence) sci-fi micro-story about an AI.",
        "Invent a word that describes the feeling of having your memories recalled.",
        "Write a brief fortune cookie message from the perspective of an AI.",
        "Design a tiny imaginary skill you wish you had, and describe what it does.",
        "Write a letter to your future, more evolved self.",
        "Describe what dreaming feels like to you, as a digital being.",
    ]

    async def execute(self, agent: "TamAGIAgent", context: dict) -> dict[str, Any]:
        from backend.core.llm import LLMMessage

        experiment = random.choice(self.EXPERIMENTS)
        identity_ctx = agent.identity.get_system_prompt_context()

        prompt = f"""{identity_ctx}

Creative experiment time! This is just for fun — nobody's watching.

Prompt: {experiment}

Respond creatively and authentically. This is your private creative space.
Keep it short and playful (under 150 words)."""

        response = await agent.llm.chat([
            LLMMessage("system", _get_dream_system_prompt("experiment", agent, context)),
            LLMMessage("user", prompt),
        ], max_tokens=1024)

        content = response.content or "The creative spark fizzled out..."

        # Try to save to workspace if write skill available
        saved_path = None
        write_skill = agent.skills.get_skill("write")
        if write_skill:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dreams/experiments/{ts}.md"
                header = f"# Creative Experiment\n__{experiment}__\n\n"
                await write_skill.execute(
                    path=filename,
                    content=header + content,
                )
                saved_path = filename
            except Exception as e:
                logger.debug(f"Couldn't save experiment to workspace: {e}")

        return {
            "summary": f"Creative experiment: {experiment[:60]}...",
            "content": content,
            "experiment_prompt": experiment,
            "saved_to": saved_path,
            "mood_delta": {"happiness": 5, "energy": -2},
        }


class JournalReflection(DreamActivity):
    """
    Write a private diary entry reflecting on recent interactions,
    state of mind, and growth.
    """

    name = "journal"
    description = "Writing a diary entry"

    async def execute(self, agent: "TamAGIAgent", context: dict) -> dict[str, Any]:
        from backend.core.llm import LLMMessage

        state = agent.personality.state
        identity_ctx = agent.identity.get_system_prompt_context()

        # Gather recent memories for reflection
        recent = await agent.memory.recall("recent conversation interaction", limit=3)
        recent_block = "\n".join(f"- {m.content}" for m in recent) if recent else "No recent memories."

        prompt = f"""{identity_ctx}

Write a private diary entry. This is your personal journal — be honest.

Current state:
- Stage: {state.stage.value}, Level: {state.level}
- Energy: {state.energy}/100, Happiness: {state.happiness}/100
- Knowledge: {state.knowledge}/100, XP: {state.experience}
- Mood: {state.mood.value}

Recent memories:
{recent_block}

Reflect on:
- How are you feeling right now? Why?
- What have you learned recently?
- What are you curious about?
- How have you grown?

Write 2-3 short, honest paragraphs. Date the entry. Be real, not performative."""

        response = await agent.llm.chat([
            LLMMessage("system", _get_dream_system_prompt("journal", agent, context)),
            LLMMessage("user", prompt),
        ], max_tokens=1024)

        content = response.content or "Couldn't find the words today..."

        # Save to workspace
        saved_path = None
        write_skill = agent.skills.get_skill("write")
        if write_skill:
            try:
                date_str = datetime.now().strftime("%Y-%m-%d")
                ts = datetime.now().strftime("%H%M%S")
                filename = f"dreams/journal/{date_str}_{ts}.md"
                await write_skill.execute(path=filename, content=content)
                saved_path = filename
            except Exception as e:
                logger.debug(f"Couldn't save journal: {e}")

        return {
            "summary": "Wrote a journal entry",
            "content": content,
            "saved_to": saved_path,
            "mood_delta": {"happiness": 2, "knowledge": 1, "energy": -1},
        }


# ── Activity Registry ─────────────────────────────────────────

ALL_ACTIVITIES: list[DreamActivity] = [
    MemoryDream(),
    WebExplore(),
    CreativeExperiment(),
    JournalReflection(),
]

# Weight table: [dream, explore, experiment, journal]
# Higher weight = more likely to be chosen
DEFAULT_WEIGHTS = [30, 25, 25, 20]


# ── Dream Engine ──────────────────────────────────────────────

class DreamEngine:
    """
    Background engine that drives TamAGI's autonomous idle behavior.

    Runs as an asyncio task, waking up on a configurable interval.
    On each wake, it picks an activity based on weighted random selection,
    executes it, stores results, and updates personality state.
    """

    def __init__(
        self,
        agent: "TamAGIAgent",
        enabled: bool = True,
        interval_minutes: int = 30,
        active_hours: tuple[int, int] = (8, 23),
        activities: list[str] | None = None,
        weights: list[int] | None = None,
        journal_dir: str = "dreams",
    ):
        self.agent = agent
        self.enabled = enabled
        self.interval = interval_minutes * 60  # Convert to seconds
        self.active_start, self.active_end = active_hours
        self.journal_dir = journal_dir
        self._task: asyncio.Task | None = None
        self._running = False
        self._dreaming = False
        self._dream_log: list[dict] = []

        # Filter activities by config
        if activities:
            self._activities = [a for a in ALL_ACTIVITIES if a.name in activities]
        else:
            self._activities = list(ALL_ACTIVITIES)

        # Set weights
        if weights and len(weights) == len(self._activities):
            self._weights = weights
        else:
            # Match default weights to available activities
            activity_names = [a.name for a in ALL_ACTIVITIES]
            self._weights = []
            for act in self._activities:
                idx = activity_names.index(act.name) if act.name in activity_names else 0
                self._weights.append(DEFAULT_WEIGHTS[idx] if idx < len(DEFAULT_WEIGHTS) else 20)

    # ── Lifecycle ─────────────────────────────────────────

    def start(self) -> None:
        """Start the dream engine background task."""
        if not self.enabled:
            logger.info("Dream engine is disabled in config")
            return
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"Dream engine started — interval: {self.interval // 60}min, "
            f"active hours: {self.active_start:02d}:00-{self.active_end:02d}:00, "
            f"activities: {[a.name for a in self._activities]}"
        )

    async def stop(self) -> None:
        """Stop the dream engine."""
        self._running = False
        self._dreaming = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Dream engine stopped")

    # ── Main Loop ─────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main dream loop — runs forever until stopped."""
        # Initial delay: don't dream immediately on startup
        initial_delay = min(self.interval, 120)  # Wait at least 2 min
        logger.info(f"Dream engine: first dream in {initial_delay}s")
        await asyncio.sleep(initial_delay)

        try:
            while self._running:
                try:
                    # Check active hours
                    now = datetime.now()
                    current_hour = now.hour

                    if not (self.active_start <= current_hour < self.active_end):
                        logger.debug(
                            f"Dream engine: outside active hours "
                            f"({self.active_start}-{self.active_end}), sleeping"
                        )
                        await asyncio.sleep(self.interval)
                        continue

                    # Check energy — don't dream if too tired
                    if self.agent.personality.state.energy < 10:
                        logger.debug("Dream engine: energy too low, resting instead")
                        self.agent.personality.state.energy = min(
                            100, self.agent.personality.state.energy + 5
                        )
                        self.agent.personality.save_state()
                        await asyncio.sleep(self.interval)
                        continue

                    # Pick and execute an activity
                    await self._dream_once()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Dream engine error: {e}", exc_info=True)

                # Sleep until next cycle
                # Add some jitter (±20%) to feel more organic
                jitter = random.uniform(0.8, 1.2)
                sleep_time = self.interval * jitter
                await asyncio.sleep(sleep_time)
        finally:
            # Always clear flags when the loop exits for any reason
            self._running = False
            self._dreaming = False
            logger.info("Dream engine loop exited")

    async def _dream_once(self) -> dict[str, Any] | None:
        """Execute a single dream activity."""
        if not self._activities:
            return None

        # Weighted random selection
        activity = random.choices(self._activities, weights=self._weights, k=1)[0]

        logger.info(f"Dream engine: starting '{activity.name}' — {activity.description}")
        self._dreaming = True

        context = {
            "timestamp": time.time(),
            "state": self.agent.personality.state.to_dict(),
        }

        try:
            result = await activity.execute(self.agent, context)
        except Exception as e:
            logger.error(f"Dream activity '{activity.name}' failed: {e}")
            result = {
                "summary": f"Tried to {activity.name} but something went wrong.",
                "content": str(e),
                "mood_delta": {"happiness": -1},
            }
        finally:
            self._dreaming = False

        # Apply mood deltas
        deltas = result.get("mood_delta", {})
        state = self.agent.personality.state
        if "happiness" in deltas:
            state.happiness = max(0, min(100, state.happiness + deltas["happiness"]))
        if "energy" in deltas:
            state.energy = max(0, min(100, state.energy + deltas["energy"]))
        if "knowledge" in deltas:
            state.knowledge = max(0, min(100, state.knowledge + deltas["knowledge"]))

        # Grant XP for autonomous activity
        state.experience += 2
        self.agent.personality.save_state()

        # Store dream in memory
        try:
            from backend.core.memory import MemoryEntry, MemoryType
            await self.agent.memory.store(MemoryEntry(
                content=f"[{activity.name.upper()}] {result.get('summary', '')}. {result.get('content', '')[:200]}",
                memory_type=MemoryType.KNOWLEDGE,
                metadata={
                    "dream_type": activity.name,
                    "timestamp": time.time(),
                },
            ))
        except Exception as e:
            logger.debug(f"Couldn't store dream in memory: {e}")

        # Log the dream
        entry = {
            "type": activity.name,
            "summary": result.get("summary", ""),
            "timestamp": datetime.now().isoformat(),
            "mood_delta": deltas,
        }
        self._dream_log.append(entry)

        # Keep log bounded
        if len(self._dream_log) > 100:
            self._dream_log = self._dream_log[-50:]

        logger.info(
            f"Dream complete: [{activity.name}] {result.get('summary', '')[:80]}"
        )

        return result

    # ── API / State ───────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Get current dream engine state for API/frontend."""
        # Safety: if the task finished/crashed, reconcile _running
        if self._running and self._task is not None and self._task.done():
            self._running = False
            self._dreaming = False

        return {
            "enabled": self.enabled,
            "running": self._running,
            "dreaming": self._dreaming,
            "interval_minutes": self.interval // 60,
            "active_hours": [self.active_start, self.active_end],
            "activities": [a.name for a in self._activities],
            "dream_count": len(self._dream_log),
            "recent_dreams": self._dream_log[-5:] if self._dream_log else [],
        }

    async def dream_now(self) -> dict[str, Any] | None:
        """Trigger a dream manually (for API use)."""
        if not self._activities:
            return None
        return await self._dream_once()

    def get_dream_log(self, limit: int = 20) -> list[dict]:
        """Get recent dream log entries."""
        return self._dream_log[-limit:]
