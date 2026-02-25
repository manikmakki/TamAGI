"""
Skill Registry — Discovery, registration, and management of skills.

Auto-discovers skills from:
  1. Built-in skills (read, write, exec)
  2. Custom skills in backend/skills/custom/
  3. Dynamically registered skills
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

from backend.skills.base import Skill, SkillResult

logger = logging.getLogger("tamagi.skills.registry")


class SkillRegistry:
    """Central registry for all TamAGI skills."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill instance."""
        if skill.name in self._skills:
            logger.warning(f"Overwriting existing skill: {skill.name}")
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name}")

    def unregister(self, name: str) -> bool:
        """Remove a skill by name."""
        if name in self._skills:
            del self._skills[name]
            logger.info(f"Unregistered skill: {name}")
            return True
        return False

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    # Alias for cleaner API
    get_skill = get

    def list_skills(self) -> list[dict[str, Any]]:
        """List all registered skills with their definitions."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            }
            for s in self._skills.values()
        ]

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get all skills as OpenAI function calling tool definitions."""
        return [s.to_openai_tool() for s in self._skills.values()]

    async def execute(self, name: str, **kwargs: Any) -> SkillResult:
        """Execute a skill by name."""
        skill = self._skills.get(name)
        if not skill:
            return SkillResult(
                success=False,
                error=f"Unknown skill: {name}. Available: {list(self._skills.keys())}",
            )

        try:
            logger.info(f"Executing skill: {name} with args: {list(kwargs.keys())}")
            result = await skill.execute(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Skill execution error ({name}): {e}", exc_info=True)
            return SkillResult(success=False, error=str(e))

    def discover_custom_skills(self, custom_dir: str | Path = "backend/skills/custom") -> int:
        """
        Auto-discover and register custom skills from a directory.
        Returns the number of skills discovered.
        """
        custom_path = Path(custom_dir)
        if not custom_path.exists():
            custom_path.mkdir(parents=True, exist_ok=True)
            return 0

        count = 0
        for py_file in custom_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                module_name = f"custom_skill_{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find Skill subclasses in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, Skill)
                            and attr is not Skill
                            and hasattr(attr, "name")
                        ):
                            instance = attr()
                            self.register(instance)
                            count += 1

            except Exception as e:
                logger.error(f"Error loading custom skill from {py_file}: {e}")

        if count:
            logger.info(f"Discovered {count} custom skill(s)")
        return count

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __repr__(self) -> str:
        return f"<SkillRegistry skills={list(self._skills.keys())}>"
