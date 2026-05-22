"""
MCP (Model Context Protocol) adapter.

Connects to MCP servers via stdio or HTTP+SSE transport, discovers their
tools, and wraps each one as a native TamAGI Skill.  From the perspective
of the tool loop and the skill registry, MCP tools are indistinguishable
from built-in skills.

Lifecycle
---------
Connections are opened once at startup and held for the application lifetime
via AsyncExitStack.  All MCPSkill instances for a given server share one
persistent ClientSession.  MCPManager.close() tears everything down cleanly
on shutdown.

Secret injection
----------------
API keys referenced in config.mcp.servers[*].env are resolved through
SecretStore and injected as subprocess env vars at spawn time.  The resolved
values are never written into any LLM message or conversation history.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any

from backend.skills.base import Skill, SkillResult

if TYPE_CHECKING:
    from backend.config import MCPServerConfig
    from backend.skills.registry import SkillRegistry

logger = logging.getLogger("tamagi.mcp")

# ── Global manager reference (set during lifespan startup) ─────────────────
_manager: "MCPManager | None" = None


def set_mcp_manager(m: "MCPManager | None") -> None:
    global _manager
    _manager = m


def get_mcp_manager() -> "MCPManager | None":
    return _manager


class MCPSkill(Skill):
    """Wraps a single MCP tool as a TamAGI Skill."""

    def __init__(self, tool: Any, session: Any) -> None:
        schema: dict[str, Any] = tool.inputSchema or {}
        required_set: set[str] = set(schema.get("required", []))

        self.name = tool.name
        self.description = tool.description or f"MCP tool: {tool.name}"
        self.parameters = {
            k: {**v, "required": k in required_set}
            for k, v in schema.get("properties", {}).items()
        }
        self._session = session

    async def execute(self, **kwargs: Any) -> SkillResult:
        try:
            result = await self._session.call_tool(self.name, arguments=kwargs)
            text = "\n".join(
                block.text for block in result.content if hasattr(block, "text")
            )
            return SkillResult(
                success=not result.isError,
                output=text or "(no output)",
            )
        except Exception as exc:
            logger.error("MCP tool '%s' failed: %s", self.name, exc)
            return SkillResult(success=False, error=str(exc))


class MCPServerAdapter:
    """
    Manages a persistent connection to one MCP server.

    Uses AsyncExitStack so the stdio subprocess / SSE connection and the
    ClientSession remain open for the application lifetime.  Call close()
    during shutdown to release resources cleanly.
    """

    def __init__(self, config: "MCPServerConfig") -> None:
        self._config = config
        self._exit_stack = contextlib.AsyncExitStack()
        self._session: Any = None
        self.skills: list[MCPSkill] = []

    async def connect(self) -> list[MCPSkill]:
        """Open the server connection and return MCPSkill instances for all its tools."""
        from backend.core.secrets import get_secret_store

        extra_env = (
            get_secret_store().resolve_env(self._config.env)
            if self._config.env
            else {}
        )
        child_env = {**os.environ, **extra_env}

        if self._config.transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=self._config.command,
                args=self._config.args or [],
                env=child_env,
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )

        elif self._config.transport == "sse":
            from mcp.client.sse import sse_client

            # For SSE transport, extra_env is passed as headers if the server
            # expects auth there; plain env vars don't apply to remote processes.
            read, write = await self._exit_stack.enter_async_context(
                sse_client(self._config.url)
            )

        else:
            raise ValueError(
                f"MCP server '{self._config.name}': unknown transport "
                f"'{self._config.transport}'. Use 'stdio' or 'sse'."
            )

        from mcp import ClientSession

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

        tools_response = await self._session.list_tools()
        self.skills = [MCPSkill(t, self._session) for t in tools_response.tools]
        logger.info(
            "MCP '%s' connected — %d tool(s): %s",
            self._config.name,
            len(self.skills),
            [s.name for s in self.skills],
        )
        return self.skills

    async def close(self) -> None:
        await self._exit_stack.aclose()
        self._session = None
        logger.info("MCP server '%s' disconnected", self._config.name)


class MCPManager:
    """
    Manages all MCP server connections for TamAGI.

    Usage in main.py lifespan:

        mcp_manager = MCPManager(config.mcp.servers)
        n = await mcp_manager.connect_all(skills)
        logger.info(f"MCP tools registered: {n}")
        yield
        await mcp_manager.close()
    """

    def __init__(self, server_configs: list["MCPServerConfig"]) -> None:
        self._configs = server_configs
        self._adapters: list[MCPServerAdapter] = []

    def status(self) -> list[dict]:
        """Return a summary of all server connections and their registered tools."""
        return [
            {
                "name": a._config.name,
                "transport": a._config.transport,
                "connected": a._session is not None,
                "tools": [s.name for s in a.skills],
            }
            for a in self._adapters
        ]

    async def connect_all(self, registry: "SkillRegistry") -> int:
        """
        Connect to all configured MCP servers and register their tools.
        Returns the total number of tools registered.
        Failures for individual servers are logged and skipped.
        """
        total = 0
        for cfg in self._configs:
            adapter = MCPServerAdapter(cfg)
            self._adapters.append(adapter)
            try:
                skills = await adapter.connect()
                for skill in skills:
                    registry.register(skill)
                total += len(skills)
            except Exception as exc:
                logger.error(
                    "Failed to connect MCP server '%s': %s", cfg.name, exc
                )
        return total

    async def close(self) -> None:
        for adapter in self._adapters:
            try:
                await adapter.close()
            except Exception as exc:
                logger.warning("Error closing MCP adapter: %s", exc)
        self._adapters.clear()
