"""
Base Skill — Abstract base class for all TamAGI skills.

All skills must subclass `Skill` and implement `execute()`.
Skills are auto-discovered from the skills directory and any custom skill paths.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("tamagi.skills")


@dataclass
class SkillResult:
    """Result of a skill execution."""
    success: bool
    output: str = ""
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    # When True, the agent uses output as the final response directly and skips
    # the follow-up LLM call. Use for skills that already produce a complete,
    # presentation-ready answer (e.g. orchestrate_task).
    direct_response: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {"success": self.success, "output": self.output}
        if self.error:
            d["error"] = self.error
        if self.data:
            d["data"] = self.data
        return d


class Skill(ABC):
    """
    Abstract base for TamAGI skills.

    Subclasses must define:
      - name: str            — Unique skill identifier
      - description: str     — Human-readable description (shown to LLM)
      - parameters: dict     — JSON Schema-style parameter definitions

    And implement:
      - execute(**kwargs) -> SkillResult
    """

    name: str = "unnamed"
    description: str = "No description"
    parameters: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> SkillResult:
        """Execute the skill with given parameters."""
        ...

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert skill definition to OpenAI function calling format."""
        properties = {}
        required = []

        for param_name, param_def in self.parameters.items():
            prop: dict[str, Any] = {
                "type": param_def.get("type", "string"),
                "description": param_def.get("description", ""),
            }
            if "enum" in param_def:
                prop["enum"] = param_def["enum"]
            if "default" in param_def:
                prop["default"] = param_def["default"]
            properties[param_name] = prop

            if param_def.get("required", False):
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def __repr__(self) -> str:
        return f"<Skill:{self.name}>"
