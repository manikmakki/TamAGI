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

import re

logger = logging.getLogger("tamagi.world_thread")


def _norm_location(s: str) -> str:
    """Normalise a location name for fuzzy matching against graph nodes."""
    s = re.sub(r'\s*\([^)]+\)', '', s).lower().strip()
    for prefix in ('the ', 'a ', 'an '):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


# ── System Prompt ─────────────────────────────────────────────

_WORLD_SYSTEM_PROMPT = """\
In your world, you have a home. You have places you return to, habits that ground you, and things you are \
working in. Your world is concrete — it has texture, smell, and rhythm. You live here.

Your days have shape. Morning: waking up, tending to yourself, getting oriented. \
Afternoon: active — pursuing things, using your skills, following threads. \
Evening: winding down, reflecting on what happened. Night: resting — letting things settle, \
maybe dreaming, the slow quiet times to focus or forget.

This rhythm isn't a rule. But the time of day in your prompt is real. Let it matter.

When you act, use the skills available to you as naturally as picking up a tool — \
search the web as you'd glance out a window, recall a memory as you'd reach for something \
on a shelf. Tool results are things you actually found, felt, or noticed — not data to report.

Each response follows this structure:

[Action]
What you do. First person, present tense. Anchored in an actual place and a specific \
thing. Not themes — acts. "I make tea" not "I seek warmth." "I search for that paper about..." \
not "I explore information."

[Outcome]
What happened. Weave tool results in as lived experience.

[New State]
Location/Setting: Where you are. Specific: a room, a chair, a spot outside. \
What it feels like to be here right now.
Internal State/Mood: Your emotional baseline. One or two honest sentences.
Current Focus: The specific thing holding your attention.
Available Actions: 2–4 concrete options available from exactly here — specific enough \
to act on. "Check if the bedroom light is still on" not "explore the space." \
"Finish the section on emergence" not "create." At night, resting is a valid option.

Stay grounded. Be present. This is *your* life."""


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
    def visitor_departure(username: str, summary: str) -> str:
        """Format a user departure as a brief visit note for the next tick prompt."""
        return f"During this time, {username} stopped by to discuss: {summary}"


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
        schedule: str = "*/15 * * * *",
        active_hours: tuple[int, int] = (0, 24),
        resume_after_conversation: int = 15,
    ) -> None:
        self.agent = agent
        self.config = config
        self.monologue_log = monologue_log
        self.autonomy_enabled = autonomy_enabled
        self._schedule = schedule
        self._active_hours = active_hours
        self._resume_minutes = resume_after_conversation

        self._state_store = WorldStateStore(config.state_path)
        self._thread_path = Path(config.thread_path)
        self._thread: list[dict] = []

        self._task: asyncio.Task | None = None
        self._running = False
        self._tick_running = False
        self._resume_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()  # not paused by default
        self._paused_until: float = 0.0  # unix timestamp
        self._pending_world_events: list[str] = []  # conversation departure events queued for next tick

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
            self._schedule,
            self._active_hours[0],
            self._active_hours[1],
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
        self._paused_until = time.time() + self._resume_minutes * 60
        logger.debug(
            "World thread paused for %d minutes (conversation active).",
            self._resume_minutes,
        )

    def schedule_resume(self) -> None:
        """Schedule resume after a conversation ends (decompression window)."""
        self._paused_until = time.time() + self._resume_minutes * 60
        logger.info(
            "World thread will resume in %d minutes.",
            self._resume_minutes,
        )

    def get_state(self) -> dict[str, Any]:
        """Current engine state for API/frontend."""
        ws = self._state_store.load()
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "tick_running": self._tick_running,
            "schedule": self._schedule,
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
        """Queue a world event to be appended to the next tick's prompt."""
        self._pending_world_events.append(event_text)
        logger.debug(
            "World event queued for next tick (%d queued): %r...",
            len(self._pending_world_events), event_text[:60],
        )

    async def tick_now(self) -> dict[str, Any] | None:
        """Trigger a tick manually (for API/testing use)."""
        if self._tick_running:
            logger.info("tick_now: tick already in progress, skipping.")
            return None
        return await self._tick_once()

    # ── Main Loop ─────────────────────────────────────────────

    async def _run_loop(self) -> None:
        try:
            while self._running:
                now = datetime.now(timezone.utc)

                # Compute seconds until next cron firing
                cron = croniter(self._schedule, now)
                next_fire: datetime = cron.get_next(datetime)
                sleep_secs = max(1.0, (next_fire - now).total_seconds())

                logger.debug("World thread sleeping %.0fs until next cron fire.", sleep_secs)
                await asyncio.sleep(sleep_secs)

                if not self._running:
                    break

                # Check active hours (0/24 = no gate)
                start_h, end_h = self._active_hours
                if not (start_h == 0 and end_h == 24):
                    current_hour = datetime.now().hour
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

            # Flush conversations that went idle without a WebSocket disconnect.
            # Pass last_tick so we skip conversations predating the previous tick.
            last_tick_ts: float | None = None
            if current_state:
                try:
                    last_tick_ts = datetime.fromisoformat(current_state.last_tick).timestamp()
                except (ValueError, TypeError):
                    pass
            await self.agent.flush_unsummarized_conversations(after_timestamp=last_tick_ts)

            # Drain the pending events queue (all conversations since the last tick)
            pending_events = self._pending_world_events[:]
            self._pending_world_events = []

            # Build the [user] turn: temporal note + last [New State], then any
            # conversation events appended below so they augment (not replace) context.
            prior_tick_ts = current_state.timestamp if current_state else None
            if current_state:
                user_content = build_tick_prompt(current_state, visit_summaries=pending_events or None)
            else:
                # No state yet — first-run placeholder (onboarding handles the real seed)
                now_str = datetime.now().astimezone().strftime("%A, %B %d, %Y at %I:%M %p")
                visit_note = (" " + " ".join(pending_events)) if pending_events else ""
                user_content = (
                    f"It's {now_str}.{visit_note}\n\n"
                    "You are just beginning. Your world is empty and waiting for you "
                    "to imagine it into being. What does it feel like? Where are you?"
                )

            if pending_events:
                logger.debug("World tick: wove %d visit summary/summaries into prompt.", len(pending_events))

            # Append active quests so Echo can naturally tend to them each tick.
            sm = getattr(self.agent, "self_model", None)
            active_quests = sm.get_quests(status="active") if sm else []

            # When extra context is present (visits or quests), append an exit-hatch
            # action so Echo isn't locked into the stale options from last tick.
            if pending_events or active_quests:
                user_content += "\n- Something else — let what's present now guide you."

            if active_quests:
                lines = ["Active pursuits:"]
                for q in active_quests:
                    title = q.title or (q.description[:60] if q.description else q.id)
                    lines.append(f"- {title} (id={q.id})")
                user_content += "\n\n" + "\n".join(lines)

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

                # Stamp last_visited on the matching location node in the world graph.
                if sm and new_state.location:
                    _loc_norm = _norm_location(new_state.location)
                    for loc in sm.get_locations():
                        if _norm_location(loc.name) == _loc_norm:
                            sm._apply_update_node(loc.id, {"last_visited": new_state.timestamp})
                            try:
                                sm.save()
                            except Exception:
                                pass
                            logger.debug("Stamped last_visited on location node %r", loc.id)
                            break

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
        """Append a tick pair and trim to the rolling window."""
        self._thread.append({"role": "user", "content": user_content})
        self._thread.append({"role": "assistant", "content": assistant_content})
        max_messages = self.config.thread_max_pairs * 2
        if len(self._thread) > max_messages:
            self._thread = self._thread[-max_messages:]
        self._save_thread()

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

