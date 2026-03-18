"""
Chat API — REST and WebSocket endpoints for TamAGI conversations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.config import get_config

logger = logging.getLogger("tamagi.api.chat")

router = APIRouter(prefix="/api", tags=["chat"])


# ── Request/Response Models ───────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    conversation_id: str | None = None
    image_data: str | None = None        # base64-encoded image
    image_media_type: str = "image/jpeg"


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    state: dict[str, Any]
    skills_used: list[str]
    memories_recalled: int


class KnowledgeRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=100000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


# ── Agent accessor (set by main.py) ──────────────────────────

_agent = None


def set_agent(agent) -> None:
    global _agent
    _agent = agent


def get_agent():
    if _agent is None:
        raise HTTPException(status_code=503, detail="TamAGI agent not initialized")
    return _agent


# ── Push notification helpers ─────────────────────────────────

_MIN_PUSH_LENGTH = 50   # skip push for very short responses like "Done!" or "OK"


def _should_notify(text: str) -> bool:
    return len(text.strip()) >= _MIN_PUSH_LENGTH


def _make_notification(result: dict, agent) -> tuple[str, str]:
    name = getattr(agent.personality.state, "name", "TamAGI") or "TamAGI"
    title = f"{name} replied"
    clean = re.sub(r"[*_`#\[\]()\n]", " ", result.get("response", ""))
    clean = re.sub(r"\s+", " ", clean).strip()
    body = (clean[:100] + "...") if len(clean) > 100 else clean
    return title, body


# ── Chat Endpoints ────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to TamAGI and get a response."""
    agent = get_agent()
    result = await agent.chat(
        user_message=request.message,
        conversation_id=request.conversation_id,
        image_data=request.image_data,
        image_media_type=request.image_media_type,
    )
    return ChatResponse(**result)


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    subdirectory: str = Form(default="uploads"),
):
    """Upload a file to the workspace directory."""
    config = get_config()
    workspace = Path(config.workspace.path).resolve()
    dest_dir = (workspace / subdirectory).resolve()

    if not str(dest_dir).startswith(str(workspace)):
        raise HTTPException(status_code=400, detail="Invalid subdirectory")

    dest_dir.mkdir(parents=True, exist_ok=True)
    content = await file.read()

    if len(content) > config.guardrails.max_write_size:
        raise HTTPException(status_code=413, detail="File too large")

    dest = dest_dir / file.filename
    dest.write_bytes(content)

    return {
        "path": str(dest.relative_to(workspace)),
        "filename": file.filename,
        "size": len(content),
        "media_type": file.content_type,
    }


@router.get("/conversations")
async def list_conversations():
    """List all conversations."""
    agent = get_agent()
    return {"conversations": agent.list_conversations()}


@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Get a specific conversation with full message history."""
    agent = get_agent()
    conv = agent.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv.to_dict()


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation."""
    agent = get_agent()
    if agent.delete_conversation(conv_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Conversation not found")


# ── State & Memory ────────────────────────────────────────────

@router.get("/state")
async def get_state():
    """Get TamAGI's current state (mood, energy, level, sprite, etc.)."""
    agent = get_agent()
    state = agent.personality.state.to_dict()
    memory_stats = await agent.memory.get_stats()
    return {
        "state": state,
        "memory": memory_stats,
        "skills": agent.skills.list_skills(),
    }


@router.post("/knowledge")
async def feed_knowledge(request: KnowledgeRequest):
    """Feed knowledge to TamAGI's long-term memory."""
    agent = get_agent()
    mem_id = await agent.store_knowledge(request.content, request.metadata)
    return {
        "status": "stored",
        "memory_id": mem_id,
        "state": agent.personality.state.to_dict(),
    }


@router.post("/memory/recall")
async def recall_memories(request: MemoryQueryRequest):
    """Query TamAGI's memories."""
    agent = get_agent()
    memories = await agent.recall_memories(request.query, limit=request.limit)
    return {"memories": memories, "count": len(memories)}


# ── WebSocket for streaming ───────────────────────────────────

@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    agent = get_agent()

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                user_message = msg.get("message", "")
                conv_id = msg.get("conversation_id")

                if not user_message:
                    await websocket.send_json({"error": "Empty message"})
                    continue

                # Send typing indicator
                await websocket.send_json({"type": "typing", "status": True})

                async def send_event(event: dict) -> None:
                    await websocket.send_json(event)

                result = await agent.chat(
                    user_message=user_message,
                    conversation_id=conv_id,
                    event_callback=send_event,
                )

                # Fire a pose_change event immediately so the sprite animates
                # reactively on the LLM's chosen pose before the full message lands.
                pose_parts = result.get("state", {}).get("pose_parts")
                if pose_parts:
                    await websocket.send_json({"type": "pose_change", "pose_parts": pose_parts})

                await websocket.send_json({
                    "type": "message",
                    **result,
                })

                # Web Push — fire-and-forget, does not block the WS loop
                try:
                    from backend.api.push import get_push_service as _gpush
                    _psvc = _gpush()
                    if _psvc.has_subscription and _should_notify(result.get("response", "")):
                        _t, _b = _make_notification(result, agent)
                        asyncio.ensure_future(
                            _psvc.send_notification(title=_t, body=_b, url="/")
                        )
                except Exception:
                    pass   # push is best-effort; never affect the WS response

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
