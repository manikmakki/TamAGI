"""
Skill Registry — Discovery, registration, and management of skills.

Registers built-ins, MCP adapters, and home-platform versioned subprocess tools.
Directory discovery of Python Skill subclasses is a legacy non-platform feature;
it is deliberately disabled when a home-platform runtime is active.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import asyncio
import hashlib
from pathlib import Path
from typing import Any

from backend.config import get_config
from backend.skills.base import Skill, SkillResult

logger = logging.getLogger("tamagi.skills.registry")


class SkillRegistry:
    """Central registry for all TamAGI skills."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._custom_dir: Path | None = None

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
        tools = [s.to_openai_tool() for s in self._skills.values()]
        from backend.core.runtime import get_runtime

        runtime = get_runtime()
        from backend.core.runtime import current_execution

        context = current_execution.get()
        if not (
            runtime
            and context
            and not context.depth
            and context.origin == "user"
            and context.root_id in runtime.chat_controls
        ):
            tools = [
                tool
                for tool in tools
                if tool["function"]["name"] != "request_quiet_turn"
            ]
        if runtime and runtime.authority == "unrestricted_host":
            for tool in tools:
                if tool["function"]["name"] in {"read", "write", "exec"}:
                    tool["function"]["description"] = {
                        "read": "Read a host file as root. Relative paths use the active workspace; absolute host paths are supported.",
                        "write": "Write a host file as root. Relative paths use the active workspace. Core identity and task documents have structured revision tools.",
                        "exec": "Execute a command using non-login Bash with pipefail on the host as UID 0. Pipes, redirects, &&, variables and cd work normally. Use curl -sS to retain download errors. The active workspace is the default directory. Local host operations have no command-tier approval. Publishing, messaging and purchases still need operator authorization.",
                    }[tool["function"]["name"]]
                    for field in ("path", "working_dir"):
                        if field in tool["function"]["parameters"]["properties"]:
                            tool["function"]["parameters"]["properties"][field][
                                "description"
                            ] = "Relative to the active workspace, or an absolute host path."
        if runtime:
            for tool in tools:
                tool["function"]["parameters"]["properties"]["repair_of_receipt_id"] = {
                    "type": "string",
                    "description": "Optional reference to your earlier failed receipt when making a corrected attempt with this same tool and changed arguments. The platform validates the reference; it grants no extra authority.",
                }
        return tools

    async def execute(self, name: str, /, **kwargs: Any) -> SkillResult:
        """Execute a skill by name."""
        from backend.core.runtime import get_runtime, current_execution

        runtime = get_runtime()
        context = current_execution.get()
        if runtime and context is None:
            async with runtime.execution("user"):
                return await self.execute(name, **kwargs)
        if runtime and context:
            if context.allowed_tools is not None and name not in context.allowed_tools:
                return SkillResult(
                    False, error="Tool is not available in this activation"
                )
            if context.depth and name in {"orchestrate_task", "consult_companion"}:
                return SkillResult(
                    False, error="Descendant orchestration is unavailable"
                )
        skill = self._skills.get(name)
        if not skill:
            return SkillResult(False, error=f"Unknown skill: {name}")
        if runtime and getattr(skill, "external_effect", False):
            scope = skill.external_scope
            grant = runtime.store.get("external_grant", scope)
            if (
                not grant
                or not grant["data"].get("enabled")
                or grant["data"].get("expires_at", 0) <= runtime.store.clock()
            ):
                return SkillResult(
                    False,
                    error="Operator authorization is needed for this external service action",
                    data={"authorization_scope": scope},
                )
        call_key = kwargs.pop("_call_id", None)
        repair_reference = kwargs.pop("repair_of_receipt_id", None) if runtime else None
        receipt = None
        if runtime and context:
            receipt, fresh = runtime.store.begin_receipt(
                context,
                name,
                call_key,
                {k: v for k, v in kwargs.items() if not k.startswith("_")},
                repair_of_receipt_id=repair_reference,
            )
            if not fresh:
                prior = receipt["payload"].get("result")
                if prior and receipt["status"] in {"succeeded", "failed"}:
                    return SkillResult(
                        bool(prior.get("success")),
                        prior.get("output", ""),
                        prior.get("error"),
                        {**prior.get("data", {}), "receipt_id": receipt["id"]},
                    )
                return SkillResult(
                    False,
                    error="Previous dispatch outcome is unknown; inspect before retrying",
                )
        if runtime and receipt and repair_reference is not None:
            from backend.core.runtime.store import encoded

            earlier = (
                runtime.store.receipt(repair_reference)
                if isinstance(repair_reference, str)
                else None
            )
            arguments_hash = hashlib.sha256(
                encoded(
                    {k: v for k, v in kwargs.items() if not k.startswith("_")}
                ).encode()
            ).hexdigest()
            if (
                not earlier
                or earlier["actor"] != context.actor_id
                or earlier["tool"] != name
                or earlier["status"] != "failed"
                or earlier["payload"]
                .get("result", {})
                .get("data", {})
                .get("dispatch_rejected")
                or earlier["payload"].get("arguments_hash") == arguments_hash
            ):
                result = SkillResult(
                    False,
                    error="A repair reference must identify your earlier failed attempt with this same tool and changed arguments",
                    data={"dispatch_rejected": True},
                )
                runtime.store.finish_receipt(receipt["id"], "failed", result.to_dict())
                runtime.growth.observe_receipt(runtime.store.receipt(receipt["id"]))
                result.data["receipt_id"] = receipt["id"]
                return result
        try:
            result = await skill.execute(**kwargs)
        except asyncio.CancelledError:
            if receipt:
                runtime.store.finish_receipt(
                    receipt["id"],
                    "outcome_unknown",
                    {
                        "success": False,
                        "error": "interrupted; effects may have occurred",
                    },
                )
            raise
        except Exception as exc:
            logger.error("Skill execution failed: %s (%s)", name, type(exc).__name__)
            result = SkillResult(False, error=str(exc))
        if receipt:
            runtime.store.finish_receipt(
                receipt["id"],
                "succeeded" if result.success else "failed",
                result.to_dict(),
            )
            runtime.growth.observe_receipt(runtime.store.receipt(receipt["id"]))
            result.data = {**result.data, "receipt_id": receipt["id"]}
        return result

    def discover_custom_skills(self, custom_dir: str | Path | None = None) -> int:
        """
        Discover Python Skill subclasses only in legacy non-platform mode.

        A home-platform runtime skips in-process imports and activates versioned
        subprocess definitions through runtime_change instead.
        If custom_dir is provided, it is stored for subsequent reload calls.
        Returns the number of skills discovered.
        """
        from backend.core.runtime import get_runtime

        if get_runtime() is not None:
            logger.info(
                "Platform tools use versioned subprocess definitions; skipping in-process custom imports"
            )
            return 0
        if custom_dir is not None:
            self._custom_dir = Path(custom_dir)
        custom_path = self._custom_dir or Path(get_config().workspace.path) / "skills"
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
