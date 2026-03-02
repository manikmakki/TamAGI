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
from typing import Any, Callable, Coroutine

import httpx

from backend.config import TamAGIConfig
from backend.core.llm import LLMClient, LLMMessage, LLMResponse
from backend.core.memory import MemoryEntry, MemoryStore, MemoryType
from backend.core.personality import PersonalityEngine
from backend.core.identity import IdentityManager
from backend.skills.registry import SkillRegistry

logger = logging.getLogger("tamagi.agent")

# Matches [ACTION:pose_name] directives the LLM can embed in its response
# to trigger an explicit sprite pose (e.g. [ACTION:wave], [ACTION:celebrate]).
_ACTION_RE = re.compile(r'\[ACTION:((\s)?\w+)\]', re.IGNORECASE)


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
        image_data: str | None = None,
        image_media_type: str = "image/jpeg",
        event_callback: Callable[[dict], Coroutine] | None = None,
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

        # Build user content — image block for LLM, plain text for history
        if image_data:
            user_llm_content = [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:{image_media_type};base64,{image_data}"}},
            ]
            history_content = f"[image: {image_media_type}] {user_message}"
        else:
            user_llm_content = user_message
            history_content = user_message

        # Record user message (text-only placeholder for persistence)
        conv.messages.append(Message(role="user", content=history_content))
        conv.updated_at = time.time()

        # Capture current stage before interaction
        prev_stage_index = self.personality.state.stage_index

        # Update personality
        # Reset any LLM-set pose so mood drives the default for this turn.
        # If the LLM includes [ACTION:xxx] in its response it will set a new pose below.
        self.personality.state.current_pose = "idle"

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

        # ── Build conversation history ─────────────────────────────────────────
        # Start from the last N messages, excluding the current (just-appended) turn.
        history_messages = list(
            conv.messages[-(self.config.history.max_messages_per_conversation):-1]
        )

        # Context compression: if the total character count of the history exceeds
        # the configured threshold, archive the oldest messages into ChromaDB and
        # remove them from the active context window.
        #
        # Archived messages are NOT lost — they're embedded and stored in the
        # vector DB, so memory.recall() can still surface relevant content from
        # them when the user asks about something from earlier in the conversation.
        #
        # Set context_compress_threshold: 0 in config to disable this entirely.
        archived_count = 0
        threshold = self.config.history.context_compress_threshold
        if threshold > 0:
            total_chars = sum(len(str(m.content)) for m in history_messages)
            if total_chars > threshold:
                to_archive: list[Message] = []
                chars_removed = 0
                excess = total_chars - threshold

                # Pop oldest messages until we've freed enough character budget.
                while history_messages and chars_removed < excess:
                    m = history_messages.pop(0)
                    to_archive.append(m)
                    chars_removed += len(str(m.content))

                if to_archive:
                    # Concatenate the archived messages into a single transcript
                    # and store it in ChromaDB. The embedding allows semantic
                    # retrieval later — if the user references something from
                    # this segment, recall() will pull it back into context.
                    transcript = "\n".join(
                        f"{m.role}: {m.content}" for m in to_archive
                    )
                    await self.memory.store(MemoryEntry(
                        content=transcript,
                        memory_type=MemoryType.CONVERSATION,
                        metadata={
                            "conversation_id": conv.id,
                            "archived_count": len(to_archive),
                            "chars_archived": chars_removed,
                        },
                    ))
                    archived_count = len(to_archive)
                    logger.info(
                        f"Context compressed: archived {archived_count} messages "
                        f"({chars_removed} chars) to memory store"
                    )

        # If compression ran, prepend a brief system note so the model knows the
        # context window was trimmed and earlier content is still in memory.
        if archived_count:
            llm_messages.append(LLMMessage(
                "system",
                f"[Note: {archived_count} earlier messages from this conversation "
                f"were archived to long-term memory to fit the context window. "
                f"They remain searchable via memory recall.]",
            ))

        for msg in history_messages:
            llm_messages.append(LLMMessage(msg.role, msg.content))
        # Append current user message with image block if provided
        llm_messages.append(LLMMessage("user", user_llm_content))

        # 3. Get tool definitions
        tools = self.skills.get_openai_tools() if self.skills.skill_count > 0 else None

        # 4. LLM loop with tool calling
        skills_used = []
        llm_error: str | None = None
        for round_num in range(self.config.agent.max_tool_rounds):
            try:
                response = await self.llm.chat_with_retry(
                    llm_messages,
                    tools=tools,
                    attempts=self.config.agent.llm_retry_attempts,
                    delay=self.config.agent.llm_retry_delay,
                )
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.TimeoutException) as e:
                logger.error(f"LLM call failed after retries on round {round_num + 1}: {e}")
                if event_callback:
                    await event_callback({"type": "error", "message": "LLM connection lost"})
                llm_error = "I ran into a connection issue while working on your request. Please try again."
                break
            except httpx.HTTPStatusError as e:
                # The LLM backend returned a 4xx/5xx — often caused by a malformed
                # context (e.g. oversized payload, bad tool-result structure).
                # Log it and surface a graceful message rather than crashing.
                logger.error(f"LLM returned HTTP {e.response.status_code} on round {round_num + 1}: {e.response.text[:300]}")
                if event_callback:
                    await event_callback({"type": "error", "message": f"LLM error {e.response.status_code}"})
                llm_error = "I hit an error processing that request. The context may be too complex — please try rephrasing or starting a new conversation."
                break

            # Try to parse text tool calls if structured ones are missing
            if not response.has_tool_calls and response.content:
                response.tool_calls = parse_text_tool_calls(response.content)

            if not response.tool_calls:
                break

            # If the LLM included commentary alongside its tool calls (e.g.
            # "Let me check that file first..." or "I see the issue, trying..."),
            # surface it immediately so the user can follow the agent's reasoning.
            if response.content and response.content.strip() and event_callback:
                await event_callback({
                    "type": "interim_text",
                    "content": response.content.strip(),
                })

            # The assistant message for this round is appended ONCE before the
            # tool-result loop. Appending it inside the loop would duplicate it
            # for every tool call in the same round, which corrupts the context
            # and causes Ollama to return 500 on the next request.
            llm_messages.append(LLMMessage("assistant", response.content or ""))

            # Execute each tool call and append its result
            for tc in response.tool_calls:
                logger.info(f"Tool call: {tc.name}({tc.arguments})")
                skills_used.append(tc.name)

                if event_callback:
                    await event_callback({"type": "tool_start", "name": tc.name, "round": round_num + 1})

                result = await self.skills.execute(name=tc.name, **tc.arguments)
                self.personality.state.use_skill()

                if event_callback:
                    await event_callback({
                        "type": "tool_result",
                        "name": tc.name,
                        "output": str(result.output)[:500] if hasattr(result, "output") else str(result.to_dict())[:500],
                    })

                # Check if energy dropped critically low; trigger dream recovery if needed
                if self.personality.state.check_low_energy(agent=self):
                    logger.info(f"Energy critical ({self.personality.state.energy}%), dream recovery triggered")

                # Append the tool result so the LLM sees what the skill returned
                llm_messages.append(LLMMessage(
                    "tool",
                    json.dumps(result.to_dict()),
                    name=tc.name,
                ))

        # 5. Extract final response
        final_text = llm_error or response.content or "..."

        # Scan for [ACTION:pose_name] directives the LLM may embed in its response.
        # Strip them from the displayed text and apply the pose so the sprite
        # animates reactively rather than waiting for the next state poll.
        if not llm_error:
            action_match = _ACTION_RE.search(final_text)
            if action_match:
                self.personality.state.set_pose(action_match.group(1).lower())
                final_text = _ACTION_RE.sub("", final_text).strip()

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

        # 8. Check for stage advancement and generate name if needed
        await self._maybe_advance_stage(prev_stage_index)

        # 9. Save state
        self.personality.save_state()
        self._save_conversation(conv)

        return {
            "response": final_text,
            "conversation_id": conv.id,
            "state": self.personality.state.to_dict(),
            "skills_used": skills_used,
            "memories_recalled": len(memories),
        }

    # ── Stage Advancement ─────────────────────────────────────

    async def _maybe_advance_stage(self, prev_stage_index: int) -> None:
        """Check if stage advanced and generate a new stage name if needed."""
        state = self.personality.state
        new_index = state.stage_index
        if new_index <= prev_stage_index:
            return

        # Call LLM for new stage name
        name = await self._generate_stage_name(new_index)
        state.current_stage_name = name
        state.stage_history.append({
            "name": name,
            "stage_index": new_index,
            "experience_at": state.experience,
            "timestamp": time.time(),
        })
        logger.info(f"Stage advanced to index {new_index}: '{name}'")

    async def _generate_stage_name(self, stage_index: int) -> str:
        """Generate a 1-3 word stage name using the LLM."""
        try:
            system_prompt = self.personality.get_system_context()
            identity_ctx = self.identity.get_system_prompt_context()
            if identity_ctx:
                system_prompt += "\n\n" + identity_ctx

            # Include stage history for context
            state = self.personality.state
            history_context = ""
            if state.stage_history:
                history_lines = [f"- Stage {i+1}: {entry['name']}" for i, entry in enumerate(state.stage_history)]
                history_context = "\n\nYour evolution so far:\n" + "\n".join(history_lines)

            response = await self.llm.chat([
                LLMMessage("system", system_prompt + history_context),
                LLMMessage("user",
                    "You've just grown and evolved into a new phase of your existence. "
                    "Reflect on your current state, your experiences, and your journey so far. "
                    "Give this new stage of your being a title: ONLY 1 to 3 words, evocative and personal to you. "
                    "Respond with ONLY the title — no punctuation, no explanation."
                ),
            ], max_tokens=50)

            raw = (response.content or "").strip().strip('"\'').strip()
            words = raw.split()
            if 1 <= len(words) <= 3:
                return raw
            # Too long/short — fall through to fallback
        except Exception as e:
            logger.warning(f"Stage name LLM call failed: {e}")

        return f"stage {stage_index}"

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
