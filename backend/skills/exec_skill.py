"""
Exec Skill — Execute allowlisted shell commands.

Guardrails:
  - ONLY executes commands from the explicit allowlist
  - Timeout enforcement
  - Working directory set to workspace
  - Captures stdout, stderr, and return code
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any

from backend.config import get_config
from backend.skills.base import Skill, SkillResult

logger = logging.getLogger("tamagi.skills.exec")


class ExecSkill(Skill):
    name = "exec"
    description = (
        "Execute a shell command. Only allowlisted commands can be run. "
        "Commands execute in the workspace directory. "
        "Use this for running scripts, checking system info, processing data, etc."
    )
    parameters = {
        "command": {
            "type": "string",
            "description": "The shell command to execute (must start with an allowlisted command)",
            "required": True,
        },
        "working_dir": {
            "type": "string",
            "description": "Working directory (defaults to workspace)",
            "default": "",
        },
    }

    async def execute(self, **kwargs: Any) -> SkillResult:
        config = get_config()
        command = kwargs.get("command", "").strip()
        working_dir = kwargs.get("working_dir", "") or config.workspace.path

        if not command:
            return SkillResult(success=False, error="No command provided")

        # Parse and validate the command
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return SkillResult(success=False, error=f"Invalid command syntax: {e}")

        if not parts:
            return SkillResult(success=False, error="Empty command")

        base_command = parts[0]

        # Strip path from command (e.g., /usr/bin/python -> python)
        base_command_name = base_command.split("/")[-1]

        # Check allowlist
        if base_command_name not in config.guardrails.exec_allowlist:
            return SkillResult(
                success=False,
                error=(
                    f"Command '{base_command_name}' is not in the allowlist. "
                    f"Allowed: {config.guardrails.exec_allowlist}"
                ),
            )

        # Check for shell operators that could bypass allowlist
        dangerous_operators = ["|", "&&", "||", ";", "`", "$(", ">", ">>", "<"]
        for op in dangerous_operators:
            if op in command:
                return SkillResult(
                    success=False,
                    error=f"Shell operator '{op}' is not allowed for security. Use separate exec calls instead.",
                )

        logger.info(f"Executing: {command}")

        try:
            process = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.guardrails.exec_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return SkillResult(
                    success=False,
                    error=f"Command timed out after {config.guardrails.exec_timeout}s",
                )

            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()

            success = process.returncode == 0

            output_parts = []
            if stdout_text:
                output_parts.append(stdout_text)
            if stderr_text:
                output_parts.append(f"[stderr] {stderr_text}")
            if not output_parts:
                output_parts.append("(no output)")

            return SkillResult(
                success=success,
                output="\n".join(output_parts),
                error=stderr_text if not success else None,
                data={
                    "command": command,
                    "return_code": process.returncode,
                    "working_dir": working_dir,
                },
            )

        except FileNotFoundError:
            return SkillResult(
                success=False,
                error=f"Command not found: {base_command}",
            )
        except Exception as e:
            return SkillResult(success=False, error=f"Execution error: {e}")
