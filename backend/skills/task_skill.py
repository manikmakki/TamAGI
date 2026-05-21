"""
Task Skill — Structured CRUD for workspace/TASKS.md.

Manages a Kanban-style task board (Todo / In Progress / Done) that persists
across conversations. Handles parsing, mutation, and Done-column capping so
the LLM never has to manually rewrite the markdown file.

Matching strategy for item references:
  - A plain integer string (e.g. "2") is treated as a 1-based index within
    the relevant source column.
  - Any other string is matched case-insensitively as a substring of item text
    (first match wins).
  - The matched text is always echoed in the result so the LLM can verify.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from backend.skills.base import Skill, SkillResult

# Canonical column names used in the serialized file
_HEADERS = {
    "todo": "Todo",
    "in_progress": "In Progress",
    "done": "Done",
}

# Board type alias
# todo/in_progress: list of str
# done: list of (text, date_str) tuples, newest appended last
_Board = dict[str, list]


class TaskSkill(Skill):
    name = "task"
    description = (
        "Manage your persistent task board (Todo / In Progress / Done). "
        "Use this to add new tasks, move them between columns, mark them complete, "
        "remove them, or list the current board. "
        "Prefer this over editing TASKS.md directly — it handles parsing and capping."
    )
    parameters = {
        "action": {
            "type": "string",
            "description": (
                "What to do: "
                "'add' — add a new item to Todo; "
                "'start' — move an item from Todo to In Progress; "
                "'complete' — move an item from In Progress to Done (with today's date); "
                "'remove' — delete an item from any column; "
                "'list' — return a formatted summary of the full board."
            ),
            "enum": ["add", "start", "complete", "remove", "list"],
            "required": True,
        },
        "text": {
            "type": "string",
            "description": "Text of the new task. Required for 'add'.",
            "default": "",
        },
        "item": {
            "type": "string",
            "description": (
                "Which item to act on. Required for 'start', 'complete', 'remove'. "
                "Use a 1-based integer index (e.g. '1' = first item in the source column) "
                "or a substring of the item text (case-insensitive, first match wins)."
            ),
            "default": "",
        },
        "column": {
            "type": "string",
            "description": (
                "For 'remove' only: which column to search. "
                "Default 'any' searches all columns."
            ),
            "enum": ["todo", "in_progress", "done", "any"],
            "default": "any",
        },
    }

    def __init__(self, workspace_path: str = "./workspace", done_cap: int = 10) -> None:
        self._workspace = Path(workspace_path)
        self._done_cap = done_cap

    @property
    def _tasks_path(self) -> Path:
        return self._workspace / "TASKS.md"

    # ── Public execute ────────────────────────────────────────

    async def execute(self, **kwargs: Any) -> SkillResult:
        action = kwargs.get("action", "").strip().lower()
        text = kwargs.get("text", "") or ""
        item = kwargs.get("item", "") or ""
        column = kwargs.get("column", "any") or "any"

        if action == "add":
            return self._add(text.strip())
        if action == "start":
            return self._start(item.strip())
        if action == "complete":
            return self._complete(item.strip())
        if action == "remove":
            return self._remove(item.strip(), column.strip().lower())
        if action == "list":
            return self._list()

        return SkillResult(
            success=False,
            output=f"Unknown action '{action}'. Valid: add, start, complete, remove, list.",
            error="invalid_action",
        )

    # ── Actions ───────────────────────────────────────────────

    def _add(self, text: str) -> SkillResult:
        if not text:
            return SkillResult(success=False, output="'text' is required for add.", error="missing_text")
        # Sanitize: strip potential comment-breaking sequences
        text = text.replace("-->", "—>")
        board = self._read()
        board["todo"].append(text)
        self._write(board)
        return SkillResult(success=True, output=f"Added to Todo: {text}")

    def _start(self, item: str) -> SkillResult:
        if not item:
            return SkillResult(success=False, output="'item' is required for start.", error="missing_item")
        board = self._read()
        idx = _find_item(board["todo"], item)
        if idx is None:
            numbered = _numbered_list(board["todo"])
            return SkillResult(
                success=False,
                output=f"No item matching '{item}' found in Todo.\nCurrent Todo:\n{numbered}",
                error="item_not_found",
            )
        text = board["todo"].pop(idx)
        board["in_progress"].append(text)
        self._write(board)
        return SkillResult(success=True, output=f"Moved to In Progress: {text}")

    def _complete(self, item: str) -> SkillResult:
        if not item:
            return SkillResult(success=False, output="'item' is required for complete.", error="missing_item")
        board = self._read()
        # Search In Progress first, fall back to Todo so items can skip straight to Done
        source = None
        idx = _find_item(board["in_progress"], item)
        if idx is not None:
            source = "in_progress"
        else:
            idx = _find_item(board["todo"], item)
            if idx is not None:
                source = "todo"
        if source is None:
            all_items = board["in_progress"] + board["todo"]
            numbered = _numbered_list(all_items)
            return SkillResult(
                success=False,
                output=f"No item matching '{item}' found in In Progress or Todo.\nCurrent items:\n{numbered}",
                error="item_not_found",
            )
        text = board[source].pop(idx)
        today = date.today().isoformat()
        board["done"].append((text, today))
        # Cap Done column — keep the most recent N items
        if len(board["done"]) > self._done_cap:
            board["done"] = board["done"][-self._done_cap:]
        self._write(board)
        return SkillResult(success=True, output=f"Completed: {text}")

    def _remove(self, item: str, column: str) -> SkillResult:
        if not item:
            return SkillResult(success=False, output="'item' is required for remove.", error="missing_item")
        board = self._read()

        search_cols: list[str]
        if column == "any":
            search_cols = ["todo", "in_progress", "done"]
        elif column in ("todo", "in_progress", "done"):
            search_cols = [column]
        else:
            search_cols = ["todo", "in_progress", "done"]

        for col in search_cols:
            items = board[col] if col != "done" else [t for t, _ in board["done"]]
            idx = _find_item(items, item)
            if idx is not None:
                if col == "done":
                    removed_text = board["done"].pop(idx)[0]
                else:
                    removed_text = board[col].pop(idx)
                self._write(board)
                return SkillResult(success=True, output=f"Removed from {_HEADERS[col]}: {removed_text}")

        return SkillResult(
            success=False,
            output=f"No item matching '{item}' found in {column}.",
            error="item_not_found",
        )

    def _list(self) -> SkillResult:
        board = self._read()
        return SkillResult(success=True, output=_serialize_board(board))

    # ── File I/O ──────────────────────────────────────────────

    def _read(self) -> _Board:
        if not self._tasks_path.exists():
            return {"todo": [], "in_progress": [], "done": []}
        return _parse_board(self._tasks_path.read_text())

    def _write(self, board: _Board) -> None:
        self._tasks_path.parent.mkdir(parents=True, exist_ok=True)
        self._tasks_path.write_text(_serialize_board(board))


# ── Parsing & Serialization ───────────────────────────────────

def _parse_board(text: str) -> _Board:
    board: _Board = {"todo": [], "in_progress": [], "done": []}
    current: str | None = None

    # Map lowercase header text → board key
    header_map = {
        "todo": "todo",
        "in progress": "in_progress",
        "done": "done",
    }

    for line in text.splitlines():
        # Detect section headers: ## Todo / ## In Progress / ## Done
        if line.startswith("## "):
            key = line[3:].strip().lower()
            current = header_map.get(key)
            continue

        if current is None:
            continue

        # Open item: - [ ] text
        open_match = re.match(r"^- \[ \] (.+)$", line)
        if open_match and current in ("todo", "in_progress"):
            board[current].append(open_match.group(1).strip())
            continue

        # Done item: - [x] text  <!-- YYYY-MM-DD -->
        done_match = re.match(r"^- \[x\] (.+?)(?:\s+<!--\s*([\d-]+)\s*-->)?$", line)
        if done_match and current == "done":
            item_text = done_match.group(1).strip()
            item_date = done_match.group(2) or date.today().isoformat()
            board["done"].append((item_text, item_date))

    return board


def _serialize_board(board: _Board) -> str:
    lines = ["# Tasks", ""]

    lines.append("## Todo")
    for text in board["todo"]:
        lines.append(f"- [ ] {text}")
    lines.append("")

    lines.append("## In Progress")
    for text in board["in_progress"]:
        lines.append(f"- [ ] {text}")
    lines.append("")

    lines.append("## Done")
    for text, done_date in board["done"]:
        lines.append(f"- [x] {text}  <!-- {done_date} -->")
    lines.append("")

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────

def _find_item(items: list, ref: str) -> int | None:
    """Return the 0-based index of a matching item, or None."""
    # 1-based integer index
    try:
        idx = int(ref) - 1
        if 0 <= idx < len(items):
            return idx
    except ValueError:
        pass
    # Substring match (case-insensitive)
    ref_lower = ref.lower()
    for i, item in enumerate(items):
        text = item if isinstance(item, str) else item[0]
        if ref_lower in text.lower():
            return i
    return None


def _numbered_list(items: list) -> str:
    if not items:
        return "  (empty)"
    return "\n".join(
        f"  {i + 1}. {item if isinstance(item, str) else item[0]}"
        for i, item in enumerate(items)
    )
