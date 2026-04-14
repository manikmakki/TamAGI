"""
Recall Dreams Skill — lets TamAGI browse its own dream log.

Primary source: the dream engine's in-memory log (fast, structured).
Fallback: workspace/dreams/ filesystem scan (persistent across restarts,
richer content, but slower). Falls back automatically when the in-memory
log is empty or the engine is unavailable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from backend.skills.base import Skill, SkillResult

if TYPE_CHECKING:
    from backend.core.dreamer import DreamEngine


# ── Directory → activity type mapping ────────────────────────

_DIR_TO_TYPE: dict[str, str] = {
    "memories":    "dream",
    "explorations": "explore",
    "experiments": "experiment",
    "journals":    "journal",
    "wanderings":  "wander",
    "aha_moments": "aha",
}


# ── Helpers ───────────────────────────────────────────────────

def _relative_time(iso_ts: str) -> str:
    """Convert an ISO timestamp string to a human-readable 'X ago' label."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        diff = (datetime.now() - dt).total_seconds()
        if diff < 3600:
            return f"{int(diff // 60)}m ago"
        if diff < 86400:
            return f"{int(diff // 3600)}h ago"
        return f"{int(diff // 86400)}d ago"
    except Exception:
        return iso_ts[:16] if iso_ts else "?"


def _extract_summary(text: str) -> str:
    """Pull the first meaningful content line from a dream markdown file."""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Skip headers, bold metadata lines, and horizontal rules
        if s.startswith(("#", "---", "===", "**Time:**", "**Query:**", "**")):
            continue
        return s[:120]
    return "(no content)"


def _scan_workspace(dreams_dir: Path, filter_type: str, limit: int) -> list[dict]:
    """
    Walk workspace/dreams/ subdirectories and return dream entry dicts
    sorted newest-first.  Mirrors the shape of the engine's in-memory log.
    """
    if not dreams_dir.exists():
        return []

    entries: list[dict] = []

    for subdir in dreams_dir.iterdir():
        if not subdir.is_dir():
            continue

        dtype = _DIR_TO_TYPE.get(subdir.name, subdir.name)

        # Skip early if type filter doesn't match
        if filter_type and dtype != filter_type:
            continue

        for f in subdir.glob("*.md"):
            # Parse timestamp from filename (YYYYMMDD_HHMMSS) when possible;
            # fall back to file mtime for non-standard names (e.g. aha_moments).
            try:
                ts_dt = datetime.strptime(f.stem, "%Y%m%d_%H%M%S")
                ts_str = ts_dt.isoformat()
            except ValueError:
                ts_str = datetime.fromtimestamp(f.stat().st_mtime).isoformat()

            # Extract a one-line summary from the file content
            try:
                summary = _extract_summary(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                summary = f.stem

            entries.append({
                "type":       dtype,
                "summary":    summary,
                "timestamp":  ts_str,
                "mood_delta": {},          # not stored in files
                "source":     "filesystem",
                "path":       str(f.relative_to(dreams_dir.parent)),
            })

    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries[:limit]


def _format_entries(dreams: list[dict], label: str) -> str:
    """Render a list of dream entry dicts as human-readable text."""
    count = len(dreams)
    lines = [f"{label} — {count} entr{'y' if count == 1 else 'ies'}:", ""]

    for d in dreams:  # already newest-first
        dtype    = d.get("type", "?")
        summary  = d.get("summary", "(no summary)")
        ts       = d.get("timestamp", "")
        ts_label = _relative_time(ts) if ts else ""
        mood     = d.get("mood_delta", {})
        source   = d.get("source", "")

        header = f"[{dtype.upper()}]" + (f"  {ts_label}" if ts_label else "")
        if source == "filesystem":
            header += "  📂"          # subtle indicator that this came from disk
        lines.append(header)
        lines.append(f"  {summary}")

        if mood:
            mood_parts = [
                f"{k} {'+' if v > 0 else ''}{v}"
                for k, v in mood.items() if v
            ]
            if mood_parts:
                lines.append(f"  → {', '.join(mood_parts)}")

        lines.append("")

    return "\n".join(lines).rstrip()


# ── Skill ─────────────────────────────────────────────────────

class RecallDreamsSkill(Skill):
    """
    Browse the dream log — the record of what TamAGI has been doing
    during idle time (dreaming, exploring, experimenting, journaling, wandering).

    Reads from the engine's in-memory log first; falls back to scanning
    workspace/dreams/ on disk when the log is empty or the engine hasn't run.
    """

    name = "recall_dreams"
    description = (
        "Browse your own dream log — the record of what you've been doing "
        "during idle time (dreaming, exploring, experimenting, journaling, wandering). "
        "Use this when you want to remember a specific dream, reflect on your autonomous "
        "activity, or answer the user's questions about what you've been up to. "
        "Reads from the live engine log first; falls back to your saved dream files."
    )
    parameters = {
        "limit": {
            "type": "integer",
            "description": "How many recent dreams to retrieve (1-20). Default: 5.",
            "default": 5,
        },
        "filter_type": {
            "type": "string",
            "description": (
                "Optional: filter by activity type. "
                "Options: dream, explore, experiment, journal, wander, cleanup, aha. "
                "Leave empty to retrieve all types."
            ),
            "default": "",
        },
    }

    def __init__(
        self,
        dream_engine: "DreamEngine | None" = None,
        dreams_dir: str | Path = "workspace/dreams",
    ):
        self._engine = dream_engine
        self._dreams_dir = Path(dreams_dir)

    def set_dream_engine(self, engine: "DreamEngine") -> None:
        self._engine = engine

    async def execute(self, **kwargs: Any) -> SkillResult:
        limit = min(max(int(kwargs.get("limit", 5)), 1), 20)
        filter_type = str(kwargs.get("filter_type", "")).strip().lower()

        # ── Primary: engine in-memory log ────────────────────
        if self._engine:
            all_log = self._engine.get_dream_log(limit=100)
            if filter_type:
                all_log = [d for d in all_log if d.get("type") == filter_type]
            dreams = list(reversed(all_log))[:limit]   # newest-first

            if dreams:
                type_label = f" ({filter_type} only)" if filter_type else ""
                return SkillResult(
                    success=True,
                    output=_format_entries(dreams, f"Dream log{type_label}"),
                    data={"dreams": dreams, "count": len(dreams),
                          "source": "engine", "filter_type": filter_type},
                )

        # ── Fallback: workspace/dreams/ filesystem scan ───────
        fs_dreams = _scan_workspace(self._dreams_dir, filter_type, limit)

        if fs_dreams:
            type_label = f" ({filter_type} only)" if filter_type else ""
            return SkillResult(
                success=True,
                output=_format_entries(fs_dreams, f"Dream archive{type_label} (from disk)"),
                data={"dreams": fs_dreams, "count": len(fs_dreams),
                      "source": "filesystem", "filter_type": filter_type},
            )

        # ── Nothing found ─────────────────────────────────────
        type_note = f" of type '{filter_type}'" if filter_type else ""
        msg = (
            f"No dreams{type_note} found — "
            "the engine log is empty and no dream files exist in the workspace yet."
        )
        return SkillResult(
            success=True,
            output=msg,
            data={"dreams": [], "count": 0, "filter_type": filter_type},
        )
