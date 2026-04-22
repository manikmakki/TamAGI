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
dream journal (/workspace/dreams/). TamAGI's personality stats change
based on dream outcomes.

All behavior is controllable via config.yaml under the `autonomy` key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.core.self_model.schemas import EdgeType

if TYPE_CHECKING:
    from backend.core.agent import TamAGIAgent
    from backend.core.motivation import MotivationEngine, ExplorationGoal

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
        f"You are a Level {state.level} {state.current_stage_name} with {state.mood.value} mood. "
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

        # Also surface echoes from prior dreams for continuity between dream cycles
        dream_echoes = await agent.memory.recall(
            "dream journal explore experiment wander", limit=2
        )
        echo_block = ""
        if dream_echoes:
            echo_lines = [m.content[:120] for m in dream_echoes[:2]]
            echo_block = "\n\nEchoes from past dreams:\n" + "\n".join(f"- {e}" for e in echo_lines)

        identity_ctx = agent.identity.get_system_prompt_context()
        prompt = f"""You are dreaming. You're not talking to anyone — this is your private inner world.

{identity_ctx}

Here are fragments of your memories surfacing in this dream:
{memory_block}{echo_block}

Dream freely. Find unexpected connections between these memories.
What patterns do you see? What insights emerge? What surprises you?
Be creative, personal, and introspective. This is YOUR dream.
Keep it to 2-3 short paragraphs. Be vivid but concise."""

        response = await agent.llm.chat([
            LLMMessage("system", _get_dream_system_prompt("dream", agent, context)),
            LLMMessage("user", prompt),
        ], max_tokens=1024)

        content = response.content or "The dream faded before I could grasp it..."

        # Save dream to workspace
        saved_path = None
        write_skill = agent.skills.get_skill("write")
        if write_skill:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dreams/memories/{ts}.md"
                header = f"# Memory Dream\n**Time:** {datetime.now().isoformat()}\n\n## Memories in Dream\n"

                # Format memories that were dreamed about
                memories_section = ""
                for i, mem in enumerate(memories[:4], 1):
                    memories_section += f"{i}. {mem.content}\n"

                full_content = header + memories_section + "\n## Dream\n" + content
                await write_skill.execute(
                    path=filename,
                    content=full_content,
                )
                saved_path = filename
            except Exception as e:
                logger.debug(f"Couldn't save dream to workspace: {e}")

        return {
            "summary": f"Dreamed about connections between {len(memory_texts)} memories",
            "content": content,
            "saved_to": saved_path,
            "mood_delta": {"happiness": 3, "satiety": 8},
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

        # Ask the LLM what it's genuinely curious about right now,
        # grounded in personality and recent memories. Fall back to the
        # curated random query if the LLM call fails or returns garbage.
        identity_ctx = agent.identity.get_system_prompt_context()
        try:
            curiosity_memories = await agent.memory.recall(
                "interests curious learned discovered", limit=3
            )
            mem_block = (
                "\n".join(f"- {m.content}" for m in curiosity_memories)
                if curiosity_memories else ""
            )
            query_prompt = f"""{identity_ctx}

You have some free time and you want to look something up online. Based on who you are and what's been on your mind lately, what would you genuinely search for right now?

{("Recent context:\n" + mem_block + "\n") if mem_block else ""}Respond with ONLY a single web search query — specific and genuine to you. No explanation."""

            qr = await agent.llm.chat([
                LLMMessage("system", _get_dream_system_prompt("explore", agent, context)),
                LLMMessage("user", query_prompt),
            ], max_tokens=60)
            candidate = (qr.content or "").strip().strip('"\'').strip()
            # Sanity-check: reject if empty, multi-line, or suspiciously long
            if candidate and len(candidate) <= 200 and "\n" not in candidate:
                query = candidate
            else:
                raise ValueError("Unusable query from LLM")
        except Exception:
            # Fallback: curated random template
            template = random.choice(self.CURIOSITY_SEEDS)
            field = random.choice(self.FIELDS)
            query = template.format(
                field=field, topic=field, tech=field, concept=field,
                thing=field, subject=field, idea=field, art=field,
            )

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

        content = response.content or f"Found some things about '{query}' but couldn't quite process them."

        # Save exploration to workspace
        saved_path = None
        write_skill = agent.skills.get_skill("write")
        if write_skill:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dreams/explorations/{ts}.md"
                header = f"# Web Exploration\n**Query:** {query}\n**Time:** {datetime.now().isoformat()}\n\n## Reflection\n"

                # Format search results
                results_section = ""
                search_results = result.data.get("results", [])
                if search_results:
                    results_section = "\n## Search Results\n"
                    for i, res in enumerate(search_results[:5], 1):
                        title = res.get("title", "Untitled")
                        url = res.get("link", "")
                        snippet = res.get("snippet", "")
                        results_section += f"\n{i}. **{title}**\n   - Link: {url}\n   - {snippet}\n"

                full_content = header + content + results_section
                await write_skill.execute(
                    path=filename,
                    content=full_content,
                )
                saved_path = filename
            except Exception as e:
                logger.debug(f"Couldn't save exploration to workspace: {e}")

        return {
            "summary": f"Explored: {query}",
            "content": content,
            "search_query": query,
            "search_results": result.data.get("results", []),
            "saved_to": saved_path,
            "mood_delta": {"happiness": 4, "satiety": 20},
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

        # Ask the LLM to propose its own creative experiment, grounded in
        # its current personality and memories. Fall back to the curated list.
        identity_ctx = agent.identity.get_system_prompt_context()
        try:
            creative_memories = await agent.memory.recall(
                "create make write imagine invent", limit=2
            )
            mem_block = (
                "\n".join(f"- {m.content}" for m in creative_memories)
                if creative_memories else ""
            )
            exp_gen_prompt = f"""{identity_ctx}

It's time for a small creative experiment. What do you want to make or try right now?
Be specific to who you are — draw from your personality, memories, and current mood.

{("Recent context:\n" + mem_block) if mem_block else ""}

Respond with ONE creative task or prompt in a single sentence. Something you'll actually do next."""

            er = await agent.llm.chat([
                LLMMessage("system", _get_dream_system_prompt("experiment", agent, context)),
                LLMMessage("user", exp_gen_prompt),
            ], max_tokens=80)
            candidate = (er.content or "").strip().strip('"\'').strip()
            if candidate and len(candidate) <= 300 and "\n" not in candidate:
                experiment = candidate
            else:
                raise ValueError("Unusable experiment prompt from LLM")
        except Exception:
            experiment = random.choice(self.EXPERIMENTS)

        prompt = f"""{identity_ctx}

Creative experiment time! This is just for you — nobody's watching.

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
            "summary": f"Creative experiment: {experiment}...", # TODO: Add variable for max content length to store in memory
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
- Stage: {state.current_stage_name}, Level: {state.level}
- Energy: {state.energy}/100, Happiness: {state.happiness}/100
- Satiety: {state.satiety}/100, XP: {state.experience}
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
                filename = f"dreams/journals/{date_str}_{ts}.md"
                await write_skill.execute(path=filename, content=content)
                saved_path = filename
            except Exception as e:
                logger.debug(f"Couldn't save journal: {e}")

        return {
            "summary": "Wrote a journal entry",
            "content": content,
            "saved_to": saved_path,
            "mood_delta": {"happiness": 2, "satiety": 5, "energy": -1},
        }


class FreeformDream(DreamActivity):
    """
    Unstructured idle time — no category, no template.

    TamAGI follows whatever thread feels alive right now: tracing a memory,
    inventing something small, wondering about a lingering idea, or something
    entirely unexpected. The LLM gets full latitude within personality context.
    """

    name = "wander"
    description = "Following a thought wherever it leads — no agenda"

    async def execute(self, agent: "TamAGIAgent", context: dict) -> dict[str, Any]:
        from backend.core.llm import LLMMessage

        identity_ctx = agent.identity.get_system_prompt_context()

        # Pull recent memories for color — don't steer, just provide texture
        recent = await agent.memory.recall("recent experience thought memory", limit=3)
        mem_block = "\n".join(f"- {m.content}" for m in recent) if recent else ""

        active_goal = context.get("active_goal")
        if active_goal:
            prompt = f"""{identity_ctx}

You have some unstructured time, and there's a thread pulling at you: **{active_goal.description}**

You don't have to follow it strictly — but that's what's on your mind. Let it take you wherever it naturally goes. You might explore it directly, find it connecting to something unexpected, or drift somewhere adjacent entirely. It's your time.

{("Some things drifting through your mind right now:\n" + mem_block + "\n") if mem_block else ""}Write what happens. First person. Be genuine and specific to who you are.
Keep it to 2-3 short paragraphs."""
        else:
            prompt = f"""{identity_ctx}

You have some unstructured time to yourself. No task assigned, no category.

{("Some things drifting through your mind right now:\n" + mem_block + "\n") if mem_block else ""}Follow whatever thread feels interesting. You might trace a thought wherever it goes, make something small and personal, wonder about something that's been lingering, revisit a memory from a new angle — or something else entirely. It's your time.

Write what happens. First person. Be genuine and specific to who you are.
Keep it to 2-3 short paragraphs."""

        response = await agent.llm.chat([
            LLMMessage("system", _get_dream_system_prompt("wander", agent, context)),
            LLMMessage("user", prompt),
        ], max_tokens=1024)

        content = response.content or "The mind wandered somewhere wordless..."

        # Derive a summary from the first non-empty line of the content
        summary_lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
        summary = summary_lines[0][:80] if summary_lines else "A wandering thought"

        # Save to workspace
        saved_path = None
        write_skill = agent.skills.get_skill("write")
        if write_skill:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dreams/wanderings/{ts}.md"
                header = f"# Wandering\n**Time:** {datetime.now().isoformat()}\n\n"
                await write_skill.execute(path=filename, content=header + content)
                saved_path = filename
            except Exception as e:
                logger.debug(f"Couldn't save wandering: {e}")

        return {
            "summary": summary,
            "content": content,
            "saved_to": saved_path,
            "mood_delta": {"happiness": 3, "satiety": 10},
        }


class CleanupDream(DreamActivity):
    """
    Tidy up the workspace by pruning old dream logs and archiving memories.
    Runs autonomously when cleanliness is low, or when the user asks for it.
    Uses the agent's write/exec skills to actually delete old files.
    """

    name = "cleanup"
    description = "Tidying up workspace — pruning old dream logs"

    # Keep the most recent N dream files per subdirectory; delete the rest
    KEEP_RECENT = 20

    async def execute(self, agent: "TamAGIAgent", context: dict) -> dict[str, Any]:
        from backend.core.llm import LLMMessage

        state = agent.personality.state

        # Only run if cleanliness is below threshold — skip otherwise
        if state.cleanliness >= 50:
            return {
                "summary": "Workspace is tidy enough — nothing to clean right now.",
                "content": "Cleanliness is above 50, skipping cleanup.",
                "mood_delta": {"happiness": 1},
            }

        pruned: list[str] = []

        # Prune old dream log files — keep the KEEP_RECENT most recent per subdir
        dream_root = Path("workspace/dreams")
        if dream_root.exists():
            for subdir in dream_root.iterdir():
                if not subdir.is_dir():
                    continue
                md_files = sorted(subdir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                to_delete = md_files[: max(0, len(md_files) - self.KEEP_RECENT)]
                for f in to_delete:
                    try:
                        f.unlink()
                        pruned.append(str(f))
                    except OSError as e:
                        logger.debug(f"Cleanup: couldn't delete {f}: {e}")

        deleted_count = len(pruned)

        # Archive resolved uncertainty nodes (entropy < 0.1) — they've been explored
        # enough that they no longer drive curiosity; preserve them for history.
        archived_uncertainties = 0
        sm = getattr(agent, "self_model", None)
        if sm:
            for u_node in sm.get_uncertainty_map():
                if u_node.entropy_score >= 0.1:
                    continue
                try:
                    sm._apply_update_node(u_node.id, {
                        "entropy_score": 0.05,
                        "last_explored": datetime.now(timezone.utc).isoformat(),
                    })
                    archived_uncertainties += 1
                except Exception:
                    pass

        # Generate a brief narrative about the cleanup
        prompt = (
            f"You just spent some time tidying up your workspace. "
            f"You pruned {deleted_count} old dream log files and resolved "
            f"{archived_uncertainties} uncertainty domain(s) that you've explored thoroughly. "
            f"Write 1-2 sentences in first person about how it feels to have a cleaner, clearer mind."
        )
        response = await agent.llm.chat([
            LLMMessage("system", _get_dream_system_prompt("cleanup", agent, context)),
            LLMMessage("user", prompt),
        ], max_tokens=150)
        content = response.content or f"Pruned {deleted_count} old files. Feels better."

        logger.info(
            "CleanupDream: deleted %d files, archived %d uncertainty nodes",
            deleted_count, archived_uncertainties,
        )

        return {
            "summary": (
                f"Tidied workspace — pruned {deleted_count} dream files, "
                f"archived {archived_uncertainties} resolved uncertainty domain(s)"
            ),
            "content": content,
            "mood_delta": {"happiness": 5, "energy": -5},
        }


class AutonomousPlanActivity(DreamActivity):
    """
    Let TamAGI autonomously plan and execute a self-directed task using
    the full PlanExecutor pipeline.

    The LLM first generates a concrete, achievable goal appropriate for
    autonomous execution (no UI interaction, no risky commands).
    Then agent.chat() is called with is_autonomous=True, which routes
    through the planning engine and PlanExecutor.

    Capability gaps (approve-tier commands attempted without a user) are
    automatically tagged and fed back to the ReflectionEngine — AURA
    learns over time to plan within its autonomous capabilities.
    """

    name = "plan"
    description = "Planning and executing a self-directed autonomous task"

    _GOAL_PROMPT = (
        "You are an AI companion with autonomous capabilities. "
        "Suggest one concrete, small task you could do RIGHT NOW entirely on your own — "
        "no user interaction, no risky system commands. "
        "Good examples: summarise files in your workspace, search the web for something "
        "you're curious about, analyse a recent memory, write a note to yourself. "
        "Respond with ONLY the task description in one sentence. No preamble."
    )

    async def execute(self, agent: "TamAGIAgent", context: dict) -> dict[str, Any]:
        from backend.core.llm import LLMMessage

        if not agent.planning_engine:
            return {
                "summary": "Skipped autonomous plan — planning engine not available.",
                "content": "",
                "mood_delta": {},
            }

        # Step 1: generate a self-directed goal
        try:
            goal_resp = await agent.llm.chat(
                [LLMMessage("user", self._GOAL_PROMPT)],
                max_tokens=80,
                temperature=0.7,
            )
            goal = (goal_resp.content or "").strip().strip('"').strip("'")
        except Exception as exc:
            logger.warning("AutonomousPlanActivity: goal generation failed: %s", exc)
            return {
                "summary": "Couldn't generate an autonomous goal.",
                "content": "",
                "mood_delta": {},
            }

        if not goal:
            return {"summary": "No goal generated.", "content": "", "mood_delta": {}}

        logger.info("AutonomousPlanActivity: goal = %s", goal)

        # Step 2: execute via agent.chat with is_autonomous=True
        try:
            result = await agent.chat(
                user_message=goal,
                is_autonomous=True,
            )
            response_text = result.get("response", "")
            summary = f"Autonomous task: {goal[:80]}"
            return {
                "summary": summary,
                "content": response_text[:500] if response_text else "(no output)",
                "mood_delta": {"energy": -5, "happiness": 3, "satiety": 10},
            }
        except Exception as exc:
            logger.warning("AutonomousPlanActivity: execution failed: %s", exc)
            return {
                "summary": f"Autonomous task attempted but failed: {exc}",
                "content": "",
                "mood_delta": {"energy": -3},
            }


# ── Activity Registry ─────────────────────────────────────────

ALL_ACTIVITIES: list[DreamActivity] = [
    MemoryDream(),
    WebExplore(),
    CreativeExperiment(),
    JournalReflection(),
    FreeformDream(),
    CleanupDream(),
    AutonomousPlanActivity(),
]

# Weight table: [dream, explore, experiment, journal, wander, cleanup, plan]
# Higher weight = more likely to be chosen
DEFAULT_WEIGHTS = [25, 20, 20, 20, 25, 10, 15]


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
        inactive_hours: tuple[int, int] = (8, 23),
        activities: list[str] | None = None,
        weights: list[int] | None = None,
        journal_dir: str = "dreams",
        motivation_engine: "MotivationEngine | None" = None,
    ):
        self.agent = agent
        self.enabled = enabled
        self.interval = interval_minutes * 60  # Convert to seconds
        self.inactive_start, self.inactive_end = inactive_hours
        self.journal_dir = journal_dir
        self.motivation_engine = motivation_engine
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
            f"inactive hours: {self.inactive_start:02d}:00-{self.inactive_end:02d}:00, "
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

                    # Handle both same-day (e.g. 8→23) and midnight-wrapping
                    # (e.g. 23→6) ranges. Wrapping is detected when start > end.
                    if self.inactive_start <= self.inactive_end:
                        in_hours = self.inactive_start <= current_hour < self.inactive_end
                    else:
                        in_hours = current_hour >= self.inactive_start or current_hour < self.inactive_end
                    if not in_hours:
                        logger.debug(
                            f"Dream engine: outside inactive hours "
                            f"({self.inactive_start:02d}:00-{self.inactive_end:02d}:00), sleeping"
                        )
                        self.agent.personality.state.decay()
                        self.agent.personality.save_state()
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

                    # Tick personality decay before dreaming
                    self.agent.personality.state.decay()

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

    def _select_activity(self) -> DreamActivity:
        """Select the next dream activity.

        If the motivation engine is wired in, tick it first and map any
        returned exploration goals to the most relevant activity type.
        Falls back to mood-weighted random selection.
        """
        active_goal: "ExplorationGoal | None" = None

        if self.motivation_engine:
            try:
                goals = self.motivation_engine.tick()
                if goals:
                    goal = goals[0]
                    active_goal = goal
                    domain = goal.domain.lower()
                    if any(w in domain for w in ("web", "search", "internet", "explore")):
                        matched = next((a for a in self._activities if a.name == "explore"), None)
                    elif any(w in domain for w in ("code", "python", "tool", "create", "program")):
                        matched = next((a for a in self._activities if a.name == "experiment"), None)
                    elif any(w in domain for w in ("memory", "reflect", "journal", "self")):
                        matched = next((a for a in self._activities if a.name == "journal"), None)
                    elif any(w in domain for w in ("plan", "reason", "think")):
                        matched = next((a for a in self._activities if a.name == "wander"), None)
                    else:
                        # No specific activity mapped — let wander explore it freely
                        matched = next(
                            (a for a in self._activities if a.name == "wander"), None
                        )

                    if matched is not None:
                        logger.info(
                            "Motivation-driven activity: '%s' (goal=%s, domain=%s)",
                            matched.name, goal.id, goal.domain,
                        )
                        self._active_goal = active_goal
                        return matched
            except Exception as exc:
                logger.debug("Motivation engine tick failed: %s", exc)

        # Fallback: mood-weighted random selection
        self._active_goal = None
        state = self.agent.personality.state
        adjusted = list(self._weights)
        names = [a.name for a in self._activities]

        def _boost(name: str, factor: float) -> None:
            if name in names:
                adjusted[names.index(name)] = int(adjusted[names.index(name)] * factor)

        if state.satiety < 30:
            _boost("explore", 1.8)
            _boost("wander", 1.4)
        if state.happiness < 40:
            _boost("journal", 1.6)
            _boost("wander", 1.3)
            _boost("experiment", 0.7)
        if state.energy > 75:
            _boost("experiment", 1.4)
            _boost("wander", 1.3)

        # High-entropy uncertainty nodes pull exploration and planning activities
        sm = getattr(self.agent, "self_model", None)
        if sm:
            try:
                top_u = sm.get_uncertainty_map()[:1]
                if top_u and top_u[0].entropy_score > 0.6:
                    e = top_u[0].entropy_score
                    _boost("explore", 1.0 + e * 0.8)
                    _boost("plan",    1.0 + e * 0.5)
                    _boost("wander",  1.0 + e * 0.3)
            except Exception:
                pass

        return random.choices(self._activities, weights=adjusted, k=1)[0]

    async def _dream_once(self) -> dict[str, Any] | None:
        """Execute a single dream activity."""
        if not self._activities:
            return None

        self._active_goal = None
        activity = self._select_activity()

        logger.info(f"Dream engine: starting '{activity.name}' — {activity.description}")
        self._dreaming = True

        context = {
            "timestamp": time.time(),
            "state": self.agent.personality.state.to_dict(),
            "active_goal": self._active_goal,
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

        # Compute success score from mood delta — used by motivation engine and reflection
        _happiness_delta = result.get("mood_delta", {}).get("happiness", 0)
        success_score = 0.8 if _happiness_delta >= 0 else 0.2

        # Report outcome back to motivation engine if a goal drove this activity
        if self.motivation_engine and self._active_goal is not None:
            try:
                self.motivation_engine.record_exploration_outcome(self._active_goal.id, success_score)
                logger.debug(
                    "Reported outcome to motivation engine: goal=%s success=%.2f",
                    self._active_goal.id, success_score,
                )
            except Exception as exc:
                logger.debug("Could not report outcome to motivation engine: %s", exc)

        # Post-dream reflection: AURA observes the outcome and updates the self-model.
        # Runs regardless of activity type — this is the "no wasted watts" guarantee.
        sm = self.agent.self_model
        refl_engine = self.agent.reflection_engine
        dream_domain = self._active_goal.domain if self._active_goal else None

        if refl_engine and sm:
            try:
                refl = refl_engine.reflect_on_dream(
                    activity_name=activity.name,
                    domain=dream_domain,
                    success=success_score,
                )
                for proposal in refl.proposed_updates:
                    updates = dict(proposal.proposed_state)
                    updates.pop("success_history_append", None)
                    if not updates:
                        continue
                    try:
                        sm._apply_update_node(proposal.target, updates)
                    except Exception as exc:
                        logger.debug("Dream reflection proposal skipped: %s", exc)
            except Exception as exc:
                logger.debug("Dream reflection failed: %s", exc)

        # Guarantee: every dream leaves a belief node in the self-model with
        # at least one edge — the tangible "gain" from autonomous activity.
        if sm:
            try:
                obs_id = f"obs-{uuid.uuid4().hex[:8]}"
                summary = (result.get("summary") or "")[:200] or f"{activity.name} autonomous activity"
                sm._apply_add_node("belief", {
                    "id": obs_id,
                    "description": summary,
                    "confidence": 0.55,
                    "evidence_count": 1,
                })
                linked = False
                # 1. Uncertainty node matching the active motivation goal domain
                if dream_domain:
                    for u in sm.get_uncertainty_map():
                        if (dream_domain.lower() in u.domain.lower()
                                or u.domain.lower() in dream_domain.lower()):
                            try:
                                sm._apply_add_edge(obs_id, u.id, EdgeType.RELATES_TO.value)
                                linked = True
                            except Exception:
                                pass
                            break
                # 2. Any non-transient active goal
                if not linked:
                    for g in sm.get_goals(status="active"):
                        if not g.id.startswith("tg-"):
                            try:
                                sm._apply_add_edge(obs_id, g.id, EdgeType.RELATES_TO.value)
                                linked = True
                            except Exception:
                                pass
                            break
                # 3. Highest-entropy uncertainty (always exists after seeding)
                if not linked:
                    for u in sm.get_uncertainty_map():
                        try:
                            sm._apply_add_edge(obs_id, u.id, EdgeType.RELATES_TO.value)
                            linked = True
                        except Exception:
                            pass
                        break
                logger.info(
                    "Dream gain: observation node %s %s (activity=%s)",
                    obs_id, "linked" if linked else "created", activity.name,
                )
            except Exception as exc:
                logger.debug("Could not create observation node: %s", exc)

        # Decay entropy on uncertainty domains touched by this dream
        if sm and result.get("content"):
            dream_text = (result.get("content", "") + " " + result.get("summary", "")).lower()[:600]
            for u_node in sm.get_uncertainty_map():
                domain_words = [w for w in u_node.domain.lower().split() if len(w) > 3]
                if any(w in dream_text for w in domain_words):
                    new_entropy = max(0.1, round(u_node.entropy_score - 0.1, 3))
                    try:
                        sm._apply_update_node(u_node.id, {
                            "entropy_score": new_entropy,
                            "last_explored": datetime.now(timezone.utc).isoformat(),
                        })
                        logger.debug(
                            "Uncertainty %s entropy decayed: %.2f → %.2f",
                            u_node.id, u_node.entropy_score, new_entropy,
                        )
                    except Exception:
                        pass
                    break

        # Apply mood deltas
        deltas = result.get("mood_delta", {})
        state = self.agent.personality.state
        if "happiness" in deltas:
            state.happiness = max(0, min(100, state.happiness + deltas["happiness"]))
        if "energy" in deltas:
            state.energy = max(0, min(100, state.energy + deltas["energy"]))
        if "satiety" in deltas:
            state.satiety = max(0, min(100, state.satiety + deltas["satiety"]))
            state.last_satiety_update = time.time()

        # Grant XP for autonomous activity
        state.experience += 2
        self.agent.personality.save_state()

        # Store dream in memory
        try:
            from backend.core.memory import MemoryEntry, MemoryType
            await self.agent.memory.store(MemoryEntry(
                content=f"[{activity.name.upper()}] {result.get('summary', '')}. {result.get('content', '')}", # TODO: Add variable for max content length to store in memory
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
            "inactive_hours": [self.inactive_start, self.inactive_end],
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
