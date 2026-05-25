"""
TamAGI World State — persistence layer for the Living World.

WorldState is the single record of where the TamAGI currently is, what they
are experiencing, and what feels available to them next. It is written after
every successful world thread tick and read at:
  - User arrival (to inject location-aware greeting context)
  - World thread resume (to seed the next tick prompt)
  - System prompt injection (to surface world context in user conversations)

The atomic update contract: WorldStateStore.save() is only called when
parse_new_state() returns a non-None result. A failed or malformed tick
leaves the record untouched; the last good state remains current.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("tamagi.world_state")

# ── Data Model ────────────────────────────────────────────────


@dataclass
class WorldState:
    """The TamAGI's current world position and awareness."""

    timestamp: str                   # ISO8601 — when this state was written
    last_tick: str                   # ISO8601 — when the previous tick completed
    location: str                    # Current location name/description
    mood: str                        # Current emotional/cognitive baseline
    focus: str                       # What the TamAGI is attending to right now
    available_actions: list[str]     # LLM-generated options from this tick
    raw_state_block: str             # Full [New State] text block as extracted

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "last_tick": self.last_tick,
            "location": self.location,
            "mood": self.mood,
            "focus": self.focus,
            "available_actions": self.available_actions,
            "raw_state_block": self.raw_state_block,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorldState":
        return cls(
            timestamp=data.get("timestamp", ""),
            last_tick=data.get("last_tick", ""),
            location=data.get("location", ""),
            mood=data.get("mood", ""),
            focus=data.get("focus", ""),
            available_actions=data.get("available_actions", []),
            raw_state_block=data.get("raw_state_block", ""),
        )


# ── Store ─────────────────────────────────────────────────────


class WorldStateStore:
    """Reads and writes the world state record atomically."""

    def __init__(self, path: str | Path = "data/world_state.json") -> None:
        self._path = Path(path)

    def load(self) -> WorldState | None:
        """Return the current world state, or None if no state exists yet."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return WorldState.from_dict(data)
        except Exception as exc:
            logger.warning("Could not load world state from %s: %s", self._path, exc)
            return None

    def save(self, state: WorldState) -> None:
        """Write world state atomically via temp file + rename."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self._path.parent, prefix=".world_state_", suffix=".json"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(state.to_dict(), f, indent=2)
                os.replace(tmp_path, self._path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            logger.debug("World state saved: location=%r mood=%r", state.location, state.mood)
        except Exception as exc:
            logger.error("Failed to save world state: %s", exc)
            raise


# ── Parser ────────────────────────────────────────────────────

# Field label patterns — flexible matching for LLM formatting variation
_LOCATION_RE = re.compile(
    r"(?:location(?:/setting)?|setting)\s*:\s*(.+?)(?=\n(?:internal|mood|current focus|focus|available)|$)",
    re.IGNORECASE | re.DOTALL,
)
_MOOD_RE = re.compile(
    r"(?:internal state(?:/mood)?|mood)\s*:\s*(.+?)(?=\n(?:current focus|focus|available)|$)",
    re.IGNORECASE | re.DOTALL,
)
_FOCUS_RE = re.compile(
    r"(?:current focus(?:/object of attention)?|focus)\s*:\s*(.+?)(?=\n(?:available)|$)",
    re.IGNORECASE | re.DOTALL,
)
_ACTIONS_RE = re.compile(
    r"(?:available actions(?:/next steps)?)\s*:\s*(.+?)$",
    re.IGNORECASE | re.DOTALL,
)
_NEW_STATE_BLOCK_RE = re.compile(
    r"\[New State\](.*?)(?=\[|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_new_state(llm_response: str, previous_tick_ts: str | None = None) -> WorldState | None:
    """Extract a WorldState from the [New State] block in an LLM tick response.

    Returns None if the block is absent or the required fields cannot be parsed.
    Callers must treat None as a signal to leave the stored world state untouched.
    """
    if not llm_response:
        return None

    # Extract the [New State] block
    block_match = _NEW_STATE_BLOCK_RE.search(llm_response)
    if not block_match:
        logger.debug("parse_new_state: no [New State] block found in response")
        return None

    block = block_match.group(1).strip()
    if not block:
        logger.debug("parse_new_state: [New State] block is empty")
        return None

    # Extract individual fields
    location = _extract_field(_LOCATION_RE, block)
    mood = _extract_field(_MOOD_RE, block)
    focus = _extract_field(_FOCUS_RE, block)
    actions_raw = _extract_field(_ACTIONS_RE, block)

    # Require at least location and mood — without these the state is unusable
    if not location or not mood:
        logger.debug(
            "parse_new_state: missing required fields (location=%r, mood=%r)",
            location, mood,
        )
        return None

    available_actions = _parse_action_list(actions_raw) if actions_raw else []

    now = datetime.now(timezone.utc).isoformat()
    return WorldState(
        timestamp=now,
        last_tick=previous_tick_ts or now,
        location=location,
        mood=mood,
        focus=focus or "",
        available_actions=available_actions,
        raw_state_block=block,
    )


def build_tick_prompt(state: WorldState, visit_summaries: list[str] | None = None) -> str:
    """Build the [user] message for each world thread tick.

    Structure:
      It's {datetime}. {elapsed note}. {visit summaries if any}

      {raw_state_block from previous tick}
    """
    now = datetime.now().astimezone()  # local wall-clock time, timezone-aware
    date_str = now.strftime("%A, %B %d, %Y at %I:%M %p")

    hour = now.hour
    if 5 <= hour < 12:
        time_of_day = "Morning"
    elif 12 <= hour < 17:
        time_of_day = "Afternoon"
    elif 17 <= hour < 21:
        time_of_day = "Evening"
    else:
        time_of_day = "Night"

    elapsed_note = _elapsed_note(state.timestamp, now)
    temporal_line = f"It's {time_of_day} — {date_str}."
    if elapsed_note:
        temporal_line += f" {elapsed_note}"

    if visit_summaries:
        visits_text = " ".join(visit_summaries)
        temporal_line += f" {visits_text}"

    return f"{temporal_line}\n\n{state.raw_state_block}"


# ── Helpers ───────────────────────────────────────────────────

def _extract_field(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    if not m:
        return ""
    return m.group(1).strip().rstrip(".")


def _parse_action_list(raw: str) -> list[str]:
    """Parse a bullet/numbered list of actions into clean strings."""
    actions = []
    for line in raw.splitlines():
        line = line.strip()
        # Strip leading bullets, numbers, dashes
        line = re.sub(r"^[\d]+[.)]\s*|^[-*•]\s*", "", line).strip()
        # Strip bold/italic markers
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"\*(.+?)\*", r"\1", line)
        if line and len(line) > 3:
            actions.append(line)
    return actions[:6]  # cap at 6


def _elapsed_note(last_tick_iso: str, now: datetime) -> str:
    """Human-readable elapsed time since last tick, for temporal grounding."""
    if not last_tick_iso:
        return ""
    try:
        last = datetime.fromisoformat(last_tick_iso)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        minutes = (now - last).total_seconds() / 60.0
        if minutes < 20:
            return ""
        if minutes < 180:
            return f"About {int(minutes)} minutes have passed since your last activity."
        if minutes < 720:
            hours = int(minutes / 60)
            return f"A few hours have passed — about {hours}."
        return "You've been resting for a while — the light has shifted."
    except (ValueError, TypeError):
        return ""
