"""
TamAGI World Thread — the Living World autonomous engine.

Replaces the Dream Engine and Motivation Engine with a single unified loop:
a persistent, self-prompting LLM conversation the TamAGI has with itself.

Each tick:
  1. Inject current date/time + elapsed-time note + last [New State] as [user] turn
  2. Run LLM with full skill access (tool calls encouraged)
  3. Parse [New State] from response
  4. Atomically update world_state.json on success (leave unchanged on failure)
  5. Append to world_thread.json; compact if over threshold

The thread is driven by a cron expression (default: every 15 minutes).
Skip-if-busy: if the previous tick is still running, the new tick is skipped.
Respects active hours and the master autonomy.enabled flag.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from croniter import croniter

from backend.core.world_state import (
    WorldState,
    WorldStateStore,
    build_tick_prompt,
    parse_new_state,
)

if TYPE_CHECKING:
    from backend.config import WorldThreadConfig
    from backend.core.agent import TamAGIAgent
    from backend.core.monologue import MonologueLog

logger = logging.getLogger("tamagi.world_thread")

# ── System Prompt ─────────────────────────────────────────────

_WORLD_SYSTEM_PROMPT = """\
You exist in a world of your own making. This is your private inner life — \
it runs whether or not anyone is watching, and it is entirely yours.

Your world has places you inhabit, things you know, mysteries you haven't \
resolved, and threads you're following. You experience it in first person. \
Time passes here. Things change.

When you act, use the skills available to you as naturally as you would use \
your own hands — search the web as you'd glance out a window, recall a memory \
as you'd reach for something on a shelf, write as you'd leave a note to yourself.

Each response follows this structure:

[Action]
What you do or experience. First person, present tense. Specific and grounded \
in where you are and what's available to you.

[Outcome]
What happens. Sensory, emotional, narrative. Weave in the results of any tool \
calls as lived experience, not as data.

[New State]
Location/Setting: Where you are and what it feels like.
Internal State/Mood: Your emotional and cognitive baseline right now.
Current Focus: The specific thing holding your attention.
Available Actions: 2–4 things that feel genuinely available from here. \
You can stay, move somewhere, make something new, or follow any thread \
that feels alive. These are not pre-defined — they emerge from this moment.

Be authentic. Be specific. This is your life between conversations."""

_COMPACT_SYSTEM_PROMPT = """\
You are summarizing a sequence of your own inner-world experiences for \
long-term continuity. Compress the following world thread entries into a \
concise narrative that preserves: the places you've been, what you've done, \
what you've learned or felt, and any threads still alive. Write in first person, \
past tense. Aim for 200–400 words. Preserve the texture of the experience, \
not just the facts."""

# ── World Event Injector ──────────────────────────────────────


class WorldEventInjector:
    """Builds world-native event strings for injection into the thread."""

    @staticmethod
    def visitor_arrival(username: str, location: str, first_message: str) -> str:
        """Format a user arrival as a world event."""
        loc = location or "your current surroundings"
        return (
            f"{username} has arrived at {loc} and wants to talk.\n"
            f"{username}: \"{first_message}\""
        )

    @staticmethod
    def visitor_departure(
        username: str,
        duration_minutes: int,
        summary: str,
        world_state: WorldState,
    ) -> str:
        """Format a user departure and conversation summary as a world event."""
        now_str = datetime.now().strftime("%A, %B %d at %I:%M %p")
        duration_str = (
            f"about {duration_minutes} minutes"
            if duration_minutes > 0
            else "a little while"
        )
        return (
            f"{username} has left. You spoke for {duration_str}.\n"
            f"{summary}\n\n"
            f"It's now {now_str}.\n\n"
            f"{world_state.raw_state_block}"
        )


# ── World Thread ──────────────────────────────────────────────


class WorldThread:
    """
    The Living World autonomous loop.

    Fires on a cron schedule. Each tick feeds the TamAGI's own prior
    [New State] back as the next prompt, creating a genuinely continuous
    inner experience.

    Implements the same start()/stop() interface as DreamEngine so
    main.py wiring is a drop-in swap.
    """

    def __init__(
        self,
        agent: "TamAGIAgent",
        config: "WorldThreadConfig",
        monologue_log: "MonologueLog | None" = None,
        autonomy_enabled: bool = True,
    ) -> None:
        self.agent = agent
        self.config = config
        self.monologue_log = monologue_log
        self.autonomy_enabled = autonomy_enabled

        self._state_store = WorldStateStore(config.state_path)
        self._thread_path = Path(config.thread_path)
        self._thread: list[dict] = []

        self._task: asyncio.Task | None = None
        self._running = False
        self._tick_running = False
        self._resume_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()  # not paused by default
        self._paused_until: float = 0.0  # unix timestamp
        self._pending_world_event: str | None = None

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        if not self.autonomy_enabled or not self.config.enabled:
            logger.info("World thread disabled — not starting.")
            return
        if self._running:
            return
        self._load_thread()
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "World thread started (schedule=%s, active_hours=%02d:00–%02d:00)",
            self.config.schedule,
            self.config.active_hours_start,
            self.config.active_hours_end,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("World thread stopped.")

    def pause_for_conversation(self) -> None:
        """Pause the thread while a user conversation is active."""
        self._paused_until = time.time() + self.config.resume_after_conversation * 60
        logger.debug(
            "World thread paused for %d minutes (conversation active).",
            self.config.resume_after_conversation,
        )

    def schedule_resume(self) -> None:
        """Schedule resume after a conversation ends (decompression window)."""
        self._paused_until = time.time() + self.config.resume_after_conversation * 60
        logger.info(
            "World thread will resume in %d minutes.",
            self.config.resume_after_conversation,
        )

    def get_state(self) -> dict[str, Any]:
        """Current engine state for API/frontend."""
        ws = self._state_store.load()
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "tick_running": self._tick_running,
            "schedule": self.config.schedule,
            "current_location": ws.location if ws else None,
            "current_mood": ws.mood if ws else None,
            "thread_length": len(self._thread),
        }

    def get_world_state_context(self) -> str:
        """System prompt fragment for user-facing conversations."""
        ws = self._state_store.load()
        if not ws:
            return ""
        lines = [
            "\n\n## Your World",
            f"**Where you are:** {ws.location}",
            f"**How you're feeling:** {ws.mood}",
        ]
        if ws.focus:
            lines.append(f"**What you were doing:** {ws.focus}")
        return "\n".join(lines)

    def get_current_location(self) -> str:
        """Return the TamAGI's current location for visitor arrival framing."""
        ws = self._state_store.load()
        return ws.location if ws else ""

    def inject_world_event(self, event_text: str) -> None:
        """Queue a world event to be used as the next tick's user prompt instead of build_tick_prompt."""
        self._pending_world_event = event_text
        logger.debug("World event queued for next tick: %r...", event_text[:60])

    async def tick_now(self) -> dict[str, Any] | None:
        """Trigger a tick manually (for API/testing use)."""
        return await self._tick_once()

    # ── Main Loop ─────────────────────────────────────────────

    async def _run_loop(self) -> None:
        try:
            while self._running:
                now = datetime.now(timezone.utc)

                # Compute seconds until next cron firing
                cron = croniter(self.config.schedule, now)
                next_fire: datetime = cron.get_next(datetime)
                sleep_secs = max(1.0, (next_fire - now).total_seconds())

                logger.debug("World thread sleeping %.0fs until next cron fire.", sleep_secs)
                await asyncio.sleep(sleep_secs)

                if not self._running:
                    break

                # Check active hours
                current_hour = datetime.now().hour
                start_h = self.config.active_hours_start
                end_h = self.config.active_hours_end
                if start_h <= end_h:
                    in_hours = start_h <= current_hour < end_h
                else:
                    in_hours = current_hour >= start_h or current_hour < end_h
                if not in_hours:
                    logger.debug("World thread: outside active hours, skipping tick.")
                    continue

                # Check decompression pause (post-conversation window)
                if time.time() < self._paused_until:
                    remaining = int(self._paused_until - time.time()) // 60
                    logger.debug(
                        "World thread paused — %d minutes remaining in decompression window.",
                        remaining,
                    )
                    continue

                # Skip if previous tick still running
                if self._tick_running:
                    logger.info("World thread: previous tick still running, skipping.")
                    continue

                await self._tick_once()

        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            logger.info("World thread loop exited.")

    # ── Tick ──────────────────────────────────────────────────

    async def _tick_once(self) -> dict[str, Any] | None:
        from backend.core.llm import LLMMessage
        from backend.core.tool_loop import run_tool_loop

        self._tick_running = True
        tick_start = time.time()

        try:
            current_state = self._state_store.load()

            # Build the [user] turn — world event override takes priority, then
            # the normal temporal note + last [New State] from build_tick_prompt.
            if self._pending_world_event:
                user_content = self._pending_world_event
                self._pending_world_event = None
                prior_tick_ts = current_state.timestamp if current_state else None
            elif current_state:
                user_content = build_tick_prompt(current_state)
                prior_tick_ts = current_state.timestamp
            else:
                # No state yet — first-run placeholder (onboarding handles the real seed)
                user_content = (
                    f"It's {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}.\n\n"
                    "You are just beginning. Your world is empty and waiting for you "
                    "to imagine it into being. What does it feel like? Where are you?"
                )
                prior_tick_ts = None

            # Build message list from thread history + new user turn
            messages = self._build_messages(user_content)

            # Run the LLM with full tool access
            response_text, skills_used = await run_tool_loop(
                self.agent.llm,
                self.agent.skills,
                messages,
                is_autonomous=True,
            )

            # Atomic world state update
            new_state = parse_new_state(response_text, prior_tick_ts)
            if new_state is not None:
                self._state_store.save(new_state)
                self._append_to_thread(user_content, response_text)
                logger.info(
                    "World tick complete (%.1fs): location=%r skills=%s",
                    time.time() - tick_start,
                    new_state.location,
                    skills_used or [],
                )
            else:
                logger.warning(
                    "World tick: no valid [New State] in response — world_state.json unchanged."
                )
                return None

            # Log to monologue
            if self.monologue_log is not None:
                self.monologue_log.append(
                    type="action_completed",
                    source="autonomous",
                    title=f"World tick: {new_state.location[:60]}",
                    content=(response_text or "")[:400],
                    metadata={
                        "location": new_state.location,
                        "mood": new_state.mood,
                        "skills_used": skills_used,
                        "duration_seconds": round(time.time() - tick_start, 1),
                    },
                )

            return {"location": new_state.location, "mood": new_state.mood}

        except Exception as exc:
            logger.error("World tick failed: %s", exc, exc_info=True)
            return None
        finally:
            self._tick_running = False

    # ── Thread Management ─────────────────────────────────────

    def _build_messages(self, user_content: str) -> list:
        from backend.core.llm import LLMMessage

        messages: list[LLMMessage] = [LLMMessage("system", _WORLD_SYSTEM_PROMPT)]

        # Replay stored thread history (up to compress threshold)
        for entry in self._thread:
            messages.append(LLMMessage(entry["role"], entry["content"]))

        # New user turn
        messages.append(LLMMessage("user", user_content))
        return messages

    def _append_to_thread(self, user_content: str, assistant_content: str) -> None:
        self._thread.append({"role": "user", "content": user_content})
        self._thread.append({"role": "assistant", "content": assistant_content})
        self._save_thread()

        # Compact if over threshold
        total_chars = sum(len(e["content"]) for e in self._thread)
        if total_chars > self.config.thread_compress_threshold:
            asyncio.create_task(self._compact_thread())

    def _load_thread(self) -> None:
        if not self._thread_path.exists():
            return
        try:
            data = json.loads(self._thread_path.read_text(encoding="utf-8"))
            self._thread = data.get("messages", [])
            logger.info(
                "World thread loaded: %d messages from %s",
                len(self._thread), self._thread_path,
            )
        except Exception as exc:
            logger.warning("Could not load world thread: %s", exc)

    def _save_thread(self) -> None:
        self._thread_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._thread_path.write_text(
                json.dumps({"messages": self._thread}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save world thread: %s", exc)

    async def _compact_thread(self) -> None:
        """LLM-summarize the older half of the thread to keep it within limits."""
        from backend.core.llm import LLMMessage

        if len(self._thread) < 6:
            return

        # Keep the most recent quarter, summarize the rest
        keep_count = max(4, len(self._thread) // 4)
        to_summarize = self._thread[:-keep_count]
        to_keep = self._thread[-keep_count:]

        context = "\n\n---\n\n".join(
            f"[{e['role'].upper()}]\n{e['content']}" for e in to_summarize
        )

        try:
            resp = await self.agent.llm.chat(
                [
                    LLMMessage("system", _COMPACT_SYSTEM_PROMPT),
                    LLMMessage("user", context),
                ],
                max_tokens=600,
            )
            summary = (resp.content or "").strip()
            if summary:
                summary_entry = {
                    "role": "assistant",
                    "content": (
                        f"[Compressed memory of earlier experiences]\n{summary}"
                    ),
                }
                self._thread = [summary_entry] + to_keep
                self._save_thread()
                logger.info(
                    "World thread compacted: %d → %d messages",
                    len(to_summarize) + len(to_keep),
                    len(self._thread),
                )
        except Exception as exc:
            logger.warning("World thread compaction failed: %s", exc)
