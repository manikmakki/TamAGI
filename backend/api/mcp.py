"""
MCP status API.

Exposes a read-only view of the currently connected MCP servers and the
tools they have registered.  Used by the Settings UI to show server health
without requiring a config file reload.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.skills.mcp_adapter import get_mcp_manager

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class MCPToolInfo(BaseModel):
    name: str


class MCPServerStatus(BaseModel):
    name: str
    transport: str
    connected: bool
    tools: list[str]


class MCPStatusResponse(BaseModel):
    servers: list[MCPServerStatus]


@router.get("/status", response_model=MCPStatusResponse)
async def mcp_status():
    """Return the connection status and registered tools for all MCP servers."""
    manager = get_mcp_manager()
    if manager is None:
        return MCPStatusResponse(servers=[])
    return MCPStatusResponse(
        servers=[MCPServerStatus(**s) for s in manager.status()]
    )
