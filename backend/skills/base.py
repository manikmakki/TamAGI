"""
Base Skill — Abstract base class for all TamAGI skills.

Registry adapters subclass `Skill` and implement `execute()`. The home platform
registers built-ins and versioned subprocess tools; resident code is not imported
into the backend. Directory discovery is retained only for legacy non-platform
custom skills. See docs/workspace-layout.md for the supported tool lifecycle.
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
      - parameters: dict     — Mapping of argument names to definitions, with
                               required: True on each required argument; this
                               is not a complete top-level JSON Schema

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
            for schema_key in ('items', 'properties', 'additionalProperties'):
                if schema_key in param_def:
                    prop[schema_key] = param_def[schema_key]
            if prop['type'] == 'array' and 'items' not in prop:
                prop['items'] = {'type':'string'}
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
