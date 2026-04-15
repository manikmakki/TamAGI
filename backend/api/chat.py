"""
Chat API — REST and WebSocket endpoints for TamAGI conversations.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx
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


class DeliberationRespondRequest(BaseModel):
    proposal_id: str
    choice: str   # accept_system | accept_original | modify | force_override | reject
    justification: str = ""


# ── Agent accessor (set by main.py) ──────────────────────────

_agent = None


def set_agent(agent) -> None:
    global _agent
    _agent = agent


def get_agent():
    if _agent is None:
        raise HTTPException(status_code=503, detail="TamAGI agent not initialized")
    return _agent


# ── AURA client accessor (set by main.py when brain.mode == "aura") ──────────

_aura_client = None


def set_aura_client(client) -> None:
    global _aura_client
    _aura_client = client


def get_aura_client():
    if _aura_client is None:
        raise HTTPException(status_code=503, detail="AURA client not initialized — check brain.mode config")
    return _aura_client


# ── Helpers ───────────────────────────────────────────────────

def _aura_to_chat_response(aura_result: dict[str, Any], fallback_state: dict[str, Any]) -> dict[str, Any]:
    """Map an AURA /api/input response to TamAGI's ChatResponse dict."""
    response_text = (
        aura_result.get("response")
        or aura_result.get("llm_error")
        or "I'm having trouble connecting to my reasoning engine right now."
    )
    conv_id = aura_result.get("conversation_id") or str(uuid.uuid4())
    return {
        "response": response_text,
        "conversation_id": conv_id,
        "state": fallback_state,
        "skills_used": [],
        "memories_recalled": 0,
    }


# ── Persona context builder ───────────────────────────────────

async def _build_persona_context(agent, user_message: str) -> str:
    """Build the persona_context string appended to AURA's system prompt.

    Pure factual grounding: name, personality, current state, identity
    values, and relevant memories. No directive language about roles or
    architecture — AURA integrates this as self-knowledge, the same way
    it integrates its own self-model nodes.
    """
    config = get_config()
    name = config.tamagi.name
    personality = config.tamagi.personality

    # Current mood/energy/happiness/level from the personality engine
    state = agent.personality.state.to_dict()
    mood = state.get("mood", "neutral")
    energy = state.get("energy", 0)
    happiness = state.get("happiness", 0)
    level = state.get("level", 1)

    # Onboarding identity values (goals, values, communication style, etc.)
    identity = agent.identity.get_identity()
    identity_lines = "\n".join(
        f"  {k}: {v}" for k, v in identity.items() if v
    ) if identity else ""

    # Relevant memories for this message (best-effort; silently skip on error)
    memory_lines = ""
    try:
        memories = await agent.recall_memories(user_message, limit=2)
        if memories:
            memory_lines = "\n".join(f"  - {m.get('content', '')[:200]}" for m in memories)
    except Exception:
        pass

    lines = []
    if identity_lines:
        lines += ["", "Identity:", identity_lines]
    if memory_lines:
        lines += ["", "Recent context:", memory_lines]

    return "\n".join(lines)


# ── Chat Endpoints ────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to TamAGI and get a response."""
    config = get_config()

    if config.brain.mode == "aura":
        aura = get_aura_client()
        agent = get_agent()
        persona_context = await _build_persona_context(agent, request.message)
        try:
            aura_result = await aura.input(
                user_message=request.message,
                conversation_id=request.conversation_id,
                persona_context=persona_context,
            )
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            raise HTTPException(status_code=502, detail=f"AURA unreachable: {exc}")
        result = _aura_to_chat_response(aura_result, agent.personality.state.to_dict())
        return ChatResponse(**result)

    # local mode — native agent loop
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


# ── AURA proxy endpoints (brain.mode == "aura") ───────────────

@router.get("/aura/status")
async def aura_status():
    """Proxy GET /api/status from AURA — system snapshot."""
    aura = get_aura_client()
    try:
        return await aura.get_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise HTTPException(status_code=502, detail=f"AURA unreachable: {exc}")


@router.get("/aura/deliberation/pending")
async def aura_deliberation_pending():
    """Proxy GET /api/deliberation/pending from AURA — Tier-2 approvals."""
    aura = get_aura_client()
    try:
        return await aura.get_pending_approvals()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise HTTPException(status_code=502, detail=f"AURA unreachable: {exc}")


@router.post("/aura/deliberation/respond")
async def aura_deliberation_respond(request: DeliberationRespondRequest):
    """Proxy POST /api/deliberation/respond to AURA — resolve a pending approval."""
    aura = get_aura_client()
    try:
        return await aura.respond_to_deliberation(
            proposal_id=request.proposal_id,
            choice=request.choice,
            justification=request.justification,
        )
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise HTTPException(status_code=502, detail=f"AURA unreachable: {exc}")


# ── WebSocket for streaming ───────────────────────────────────

@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    agent = get_agent()
    config = get_config()

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

                if config.brain.mode == "aura":
                    aura = get_aura_client()
                    try:
                        persona_context = await _build_persona_context(agent, user_message)
                        aura_result = await aura.input(
                            user_message=user_message,
                            conversation_id=conv_id,
                            persona_context=persona_context,
                        )
                        result = _aura_to_chat_response(
                            aura_result, agent.personality.state.to_dict()
                        )
                    except Exception as exc:
                        logger.error("AURA input failed over WebSocket: %s", exc)
                        result = {
                            "response": "I'm having trouble connecting to my reasoning engine right now.",
                            "conversation_id": conv_id or str(uuid.uuid4()),
                            "state": agent.personality.state.to_dict(),
                            "skills_used": [],
                            "memories_recalled": 0,
                        }
                    await websocket.send_json({"type": "message", **result})
                    continue

                # local mode — streaming with event_callback
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

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
