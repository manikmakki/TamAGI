"""
Read Skill — Read files from allowed paths.

Guardrails:
  - Only reads from explicitly allowed directories
  - Path traversal prevention
  - Maximum file size limit
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend.config import get_config
from backend.skills.base import Skill, SkillResult


class ReadSkill(Skill):
    name = "read"
    description = (
        "Read the contents of a file. Can read text files from allowed directories. "
        "Use this to examine files, check configurations, read logs, etc."
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "Path to the file to read (relative to workspace, or absolute if in allowed paths)",
            "required": True,
        },
        "encoding": {
            "type": "string",
            "description": "File encoding (default: utf-8)",
            "default": "utf-8",
        },
        "max_lines": {
            "type": "integer",
            "description": "Maximum number of lines to read (0 = all)",
            "default": 0,
        },
    }

    async def execute(self, **kwargs: Any) -> SkillResult:
        config = get_config()
        file_path = kwargs.get("path", "")
        encoding = kwargs.get("encoding", "utf-8")
        max_lines = kwargs.get("max_lines", 0)

        if not file_path:
            return SkillResult(success=False, error="No file path provided")

        # Resolve path
        resolved = self._resolve_path(file_path, config.workspace.path)

        # Check if path is allowed
        if not self._is_allowed(resolved, config.guardrails.allowed_read_paths):
            return SkillResult(
                success=False,
                error=f"Access denied: '{file_path}' is not in allowed read paths",
            )

        if not resolved.exists():
            return SkillResult(success=False, error=f"File not found: {file_path}")

        if not resolved.is_file():
            # If it's a directory, list contents
            if resolved.is_dir():
                entries = sorted(resolved.iterdir())
                listing = "\n".join(
                    f"{'📁' if e.is_dir() else '📄'} {e.name}" for e in entries
                )
                return SkillResult(
                    success=True,
                    output=f"Directory listing of {file_path}:\n{listing}",
                    data={"type": "directory", "entries": [e.name for e in entries]},
                )
            return SkillResult(success=False, error=f"Not a readable file: {file_path}")

        try:
            content = resolved.read_text(encoding=encoding)

            if max_lines and max_lines > 0:
                lines = content.splitlines()
                if len(lines) > max_lines:
                    content = "\n".join(lines[:max_lines])
                    content += f"\n... ({len(lines) - max_lines} more lines)"

            return SkillResult(
                success=True,
                output=content,
                data={
                    "path": str(resolved),
                    "size": resolved.stat().st_size,
                    "lines": content.count("\n") + 1,
                },
            )

        except UnicodeDecodeError:
            return SkillResult(
                success=False,
                error=f"Cannot read '{file_path}' as text (encoding: {encoding}). Binary file?",
            )
        except Exception as e:
            return SkillResult(success=False, error=f"Error reading file: {e}")

    def _resolve_path(self, file_path: str, workspace: str) -> Path:
        """Resolve a file path, treating relative paths as workspace-relative."""
        p = Path(file_path)
        if not p.is_absolute():
            p = Path(workspace) / p
        return p.resolve()

    def _is_allowed(self, resolved: Path, allowed_paths: list[str]) -> bool:
        """Check if a resolved path falls under any allowed directory."""
        for allowed in allowed_paths:
            allowed_resolved = Path(allowed).resolve()
            try:
                resolved.relative_to(allowed_resolved)
                return True
            except ValueError:
                continue
        return False
