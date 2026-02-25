"""
Agent Core — The brain of TamAGI.

Orchestrates:
  - LLM conversations with tool calling
  - Memory retrieval (RAG) and storage
  - Skill execution
  - Personality state updates
  - Conversation history management
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.config import TamAGIConfig
from backend.core.llm import LLMClient, LLMMessage, LLMResponse
from backend.core.memory import MemoryEntry, MemoryStore, MemoryType
from backend.core.personality import PersonalityEngine
from backend.core.identity import IdentityManager
from backend.skills.registry import SkillRegistry

logger = logging.getLogger("tamagi.agent")

MAX_TOOL_ROUNDS = 5  # Max tool-call loops per user message


def parse_text_tool_calls(content: str) -> list:
    """
    Parse tool calls that appear as plain text in LLM response.
    Handles format: toolname[ARGS]{...json...}
    Returns list of objects with 'name' and 'arguments' attributes.
    """
    if not content:
        return []

    # Pattern: word[ARGS]{json}
    pattern = r'(\w+)\[ARGS\](\{[^}]*\})'
    matches = re.findall(pattern, content)

    tool_calls = []
    for tool_name, args_str in matches:
        try:
            args = json.loads(args_str)
            # Create simple object matching ToolCall interface
            tc = type('ToolCall', (), {
                'name': tool_name,
                'arguments': args
            })()
            tool_calls.append(tc)
            logger.debug(f"Parsed text tool call: {tool_name}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call args: {args_str} - {e}")

    return tool_calls


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Conversation"
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
            "preview": self.messages[-1].content[:100] if self.messages else "",
        }


class TamAGIAgent:
    """
    The core TamAGI agent.

    Handles the full conversation loop:
    1. User sends message
    2. Retrieve relevant memories (RAG)
    3. Build context (system prompt + memories + history + tools)
    4. Send to LLM
    5. If LLM wants to use tools, execute them and loop
    6. Return final response
    7. Update personality state & store memories
    """

    def __init__(
        self,
        config: TamAGIConfig,
        llm: LLMClient,
        memory: MemoryStore,
        personality: PersonalityEngine,
        skills: SkillRegistry,
        identity: IdentityManager | None = None,
    ):
        self.config = config
        self.llm = llm
        self.memory = memory
        self.personality = personality
        self.skills = skills
        self.identity = identity or IdentityManager()
        self.conversations: dict[str, Conversation] = {}
        self._history_dir = Path(config.history.persist_path)
        self._load_conversations()

    # ── Conversation Management ───────────────────────────────

    def create_conversation(self, title: str = "New Conversation") -> Conversation:
        conv = Conversation(title=title)
        self.conversations[conv.id] = conv
        return conv

    def get_conversation(self, conv_id: str) -> Conversation | None:
        return self.conversations.get(conv_id)

    def list_conversations(self) -> list[dict[str, Any]]:
        convs = sorted(
            self.conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True,
        )
        return [c.to_summary() for c in convs[: self.config.history.max_conversations]]

    def delete_conversation(self, conv_id: str) -> bool:
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            # Remove persisted file
            path = self._history_dir / f"{conv_id}.json"
            if path.exists():
                path.unlink()
            return True
        return False

    # ── Main Chat Loop ────────────────────────────────────────

    async def chat(
        self,
        user_message: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a user message and return TamAGI's response.

        Returns:
            {
                "response": str,
                "conversation_id": str,
                "state": dict,
                "skills_used": list[str],
                "memories_recalled": int,
            }
        """
        # Get or create conversation
        if conversation_id and conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
        else:
            conv = self.create_conversation()
            # Auto-title from first message
            conv.title = user_message[:60] + ("..." if len(user_message) > 60 else "")

        # Record user message
        conv.messages.append(Message(role="user", content=user_message))
        conv.updated_at = time.time()

        # Update personality
        self.personality.state.interact()

        # Check if rapid interactions drained energy too much
        if self.personality.state.check_low_energy(agent=self):
            logger.info(f"Energy critical ({self.personality.state.energy}%), dream recovery triggered from interactions")

        # 1. Retrieve relevant memories
        memories = await self.memory.recall(user_message, limit=self.config.memory.retrieval_limit)
        memory_context = ""
        if memories:
            memory_lines = [f"- {m.content}" for m in memories]
            memory_context = (
                "\n\nRelevant memories:\n" + "\n".join(memory_lines)
            )

        # 2. Build messages for LLM
        # Layer: personality base + identity/soul context + memory
        identity_context = self.identity.get_system_prompt_context()
        system_prompt = self.personality.get_system_context()
        if identity_context:
            system_prompt += "\n\n" + identity_context
        if memory_context:
            system_prompt += memory_context
        llm_messages = [LLMMessage("system", system_prompt)]

        # Add conversation history (last N messages)
        history_window = conv.messages[-(self.config.history.max_messages_per_conversation):]
        for msg in history_window:
            llm_messages.append(LLMMessage(msg.role, msg.content))

        # 3. Get tool definitions
        tools = self.skills.get_openai_tools() if self.skills.skill_count > 0 else None

        # 4. LLM loop with tool calling
        skills_used = []
        for round_num in range(MAX_TOOL_ROUNDS):
            response = await self.llm.chat(llm_messages, tools=tools)

            # Try to parse text tool calls if structured ones are missing
            if not response.has_tool_calls and response.content:
                response.tool_calls = parse_text_tool_calls(response.content)

            if not response.tool_calls:
                break

            # Execute each tool call
            for tc in response.tool_calls:
                logger.info(f"Tool call: {tc.name}({tc.arguments})")
                skills_used.append(tc.name)

                result = await self.skills.execute(tc.name, **tc.arguments)
                self.personality.state.use_skill()

                # Check if energy dropped critically low; trigger dream recovery if needed
                if self.personality.state.check_low_energy(agent=self):
                    logger.info(f"Energy critical ({self.personality.state.energy}%), dream recovery triggered")

                # Add assistant message with tool call
                llm_messages.append(LLMMessage(
                    "assistant",
                    response.content or "",
                ))

                # Add tool result
                llm_messages.append(LLMMessage(
                    "tool",
                    json.dumps(result.to_dict()),
                    name=tc.name,
                ))

        # 5. Extract final response
        final_text = response.content or "..."

        # 6. Record assistant message
        conv.messages.append(Message(
            role="assistant",
            content=final_text,
            metadata={"skills_used": skills_used},
        ))

        # 7. Store conversation summary in memory (every 3 messages for better recall)
        if len(conv.messages) % 3 == 0 and len(conv.messages) > 0:
            summary = f"Conversation about: {conv.title}. Latest exchange: User asked '{user_message}', TamAGI responded about {final_text}" # TODO: Add variable for max content length to store in memory
            await self.memory.store(MemoryEntry(
                content=summary,
                memory_type=MemoryType.CONVERSATION,
                metadata={"conversation_id": conv.id},
            ))
            self.personality.state.store_memory()

        # 8. Save state
        self.personality.save_state()
        self._save_conversation(conv)

        return {
            "response": final_text,
            "conversation_id": conv.id,
            "state": self.personality.state.to_dict(),
            "skills_used": skills_used,
            "memories_recalled": len(memories),
        }

    # ── Memory Operations ─────────────────────────────────────

    async def store_knowledge(self, content: str, metadata: dict | None = None) -> str:
        """Store a knowledge entry in memory."""
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.KNOWLEDGE,
            metadata=metadata or {},
        )
        mem_id = await self.memory.store(entry)
        self.personality.state.feed_knowledge()
        self.personality.save_state()
        return mem_id

    async def recall_memories(self, query: str, limit: int = 5) -> list[dict]:
        memories = await self.memory.recall(query, limit=limit)
        return [m.to_dict() for m in memories]

    async def get_memory_stats(self) -> dict:
        """Get memory system statistics for debugging."""
        stats = await self.memory.get_stats()
        all_memories = await self.memory.get_all_memories(limit=None)
        stats["total_memories_stored"] = len(all_memories)
        stats["memory_types"] = {}
        for mem in all_memories:
            mem_type = mem.memory_type.value
            stats["memory_types"][mem_type] = stats["memory_types"].get(mem_type, 0) + 1
        stats["recent_memories"] = [m.to_dict() for m in all_memories[:5]]
        logger.info(f"Memory stats: {stats}")
        return stats

    # ── Persistence ───────────────────────────────────────────

    def _save_conversation(self, conv: Conversation) -> None:
        if not self.config.history.persist:
            return
        self._history_dir.mkdir(parents=True, exist_ok=True)
        path = self._history_dir / f"{conv.id}.json"
        with open(path, "w") as f:
            json.dump(conv.to_dict(), f, indent=2)

    def _load_conversations(self) -> None:
        if not self.config.history.persist:
            return
        if not self._history_dir.exists():
            return

        count = 0
        for path in sorted(
            self._history_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[: self.config.history.max_conversations]:
            try:
                with open(path) as f:
                    data = json.load(f)
                conv = Conversation(
                    id=data["id"],
                    title=data.get("title", "Untitled"),
                    created_at=data.get("created_at", 0),
                    updated_at=data.get("updated_at", 0),
                    messages=[
                        Message(
                            role=m["role"],
                            content=m["content"],
                            timestamp=m.get("timestamp", 0),
                            metadata=m.get("metadata", {}),
                        )
                        for m in data.get("messages", [])
                    ],
                )
                self.conversations[conv.id] = conv
                count += 1
            except Exception as e:
                logger.warning(f"Error loading conversation {path}: {e}")

        if count:
            logger.info(f"Loaded {count} conversation(s) from history")
