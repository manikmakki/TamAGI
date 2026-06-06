"""
Write Skill — Write files to allowed paths.

Guardrails:
  - Only writes to explicitly allowed directories
  - Path traversal prevention
  - Maximum file size enforcement
  - Creates parent directories as needed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import get_config
from backend.core.identity import CORE_IDENTITY_FILES, CORE_FILE_ROLES
from backend.skills.base import Skill, SkillResult


class WriteSkill(Skill):
    name = "write"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does. Use 'append' mode to add to existing files. "
        "Writes to allowed directories only (configured workspace by default)."
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "Path to write to (relative to configured workspace, or absolute if in allowed paths)",
            "required": True,
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file",
            "required": True,
        },
        "mode": {
            "type": "string",
            "description": "Write mode: 'overwrite' or 'append'",
            "enum": ["overwrite", "append"],
            "default": "overwrite",
        },
        "encoding": {
            "type": "string",
            "description": "File encoding (default: utf-8)",
            "default": "utf-8",
        },
    }

    def to_openai_tool(self) -> dict[str, Any]:
        tool = super().to_openai_tool()
        workspace = get_config().workspace.path
        tool["function"]["description"] = (
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Use 'append' mode to add to existing files. "
            f"Writes to allowed directories only ({workspace} by default)."
        )
        tool["function"]["parameters"]["properties"]["path"]["description"] = (
            f"Filename or relative path within {workspace} (e.g. 'SOUL.md' or 'notes/foo.md'). "
            f"Do NOT prepend '{workspace}/' — it is already the base. "
            f"Absolute paths are allowed if they fall under an allowed directory."
        )
        return tool

    async def execute(self, **kwargs: Any) -> SkillResult:
        config = get_config()
        file_path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        mode = kwargs.get("mode", "overwrite")
        encoding = kwargs.get("encoding", "utf-8")

        if not file_path:
            return SkillResult(success=False, error="No file path provided")
        if content is None:
            return SkillResult(success=False, error="No content provided")

        # Check file size
        content_size = len(content.encode(encoding))
        if content_size > config.guardrails.max_write_size:
            max_mb = config.guardrails.max_write_size / (1024 * 1024)
            return SkillResult(
                success=False,
                error=f"Content too large ({content_size} bytes). Max: {max_mb:.1f}MB",
            )

        # Resolve path
        resolved = self._resolve_path(file_path, config.workspace.path)

        # TASKS.md is managed exclusively by the task skill — direct writes would
        # corrupt its format and bypass Done-capping. Redirect the LLM.
        if resolved.name == "TASKS.md":
            return SkillResult(
                success=False,
                error=(
                    "TASKS.md is managed by the task skill. "
                    "Use task(action='add'/'start'/'complete'/'remove') to update it."
                ),
            )

        # The three core identity files are injected into the system prompt in full,
        # every turn, so they must stay brief and high-signal. Enforce a hard char
        # cap and steer overflow into supplemental files / memory.
        if resolved.name in CORE_IDENTITY_FILES:
            limit = config.identity_files.file_char_limit
            projected = content
            if mode == "append" and resolved.exists():
                try:
                    projected = resolved.read_text(encoding=encoding) + content
                except OSError:
                    projected = content
            char_count = len(projected)
            if char_count > limit:
                role = CORE_FILE_ROLES.get(resolved.name, "")
                return SkillResult(
                    success=False,
                    error=(
                        f"{resolved.name} must stay under {limit} characters — it is injected into "
                        f"your mind in full, every time you think, so it has to stay brief and "
                        f"grounding. This version is {char_count} characters "
                        f"({char_count - limit} over). {resolved.name} is {role}. "
                        f"Read it, distill to the essentials, and move detail into a supplemental "
                        f"file (named in USER.md) or memory — then write the trimmed version."
                    ),
                )

        # Check if path is allowed
        if not self._is_allowed(resolved, config.guardrails.allowed_write_paths):
            return SkillResult(
                success=False,
                error=f"Access denied: '{file_path}' is not in allowed write paths",
            )

        try:
            # Create parent directories
            resolved.parent.mkdir(parents=True, exist_ok=True)

            if mode == "append":
                with open(resolved, "a", encoding=encoding) as f:
                    f.write(content)
            else:
                resolved.write_text(content, encoding=encoding)

            return SkillResult(
                success=True,
                output=f"Successfully wrote {content_size} bytes to {file_path}",
                data={
                    "path": str(resolved),
                    "size": content_size,
                    "mode": mode,
                },
            )

        except Exception as e:
            return SkillResult(success=False, error=f"Error writing file: {e}")

    def _resolve_path(self, file_path: str, workspace: str) -> Path:
        p = Path(file_path)
        if not p.is_absolute():
            p = Path(workspace) / p
        return p.resolve()

    def _is_allowed(self, resolved: Path, allowed_paths: list[str]) -> bool:
        for allowed in allowed_paths:
            allowed_resolved = Path(allowed).resolve()
            try:
                resolved.relative_to(allowed_resolved)
                return True
            except ValueError:
                continue
        return False
