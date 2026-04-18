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
import time
import uuid
import re
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import httpx

from backend.config import TamAGIConfig
from backend.core.llm import LLMClient, LLMMessage, LLMResponse
from backend.core.memory import MemoryEntry, MemoryStore, MemoryType
from backend.core.personality import PersonalityEngine
from backend.core.identity import IdentityManager
from backend.skills.registry import SkillRegistry
from backend.core.self_model import SelfModel
from backend.core.motivation import MotivationEngine
from backend.core.planning_engine import PlanningEngine, ActionPlan
from backend.core.reflection import ReflectionEngine, ActualOutcome, bayesian_update

if TYPE_CHECKING:
    from backend.core.dreamer import DreamEngine

logger = logging.getLogger("tamagi.agent")


def _dream_time_label(iso_ts: str) -> str:
    """Convert an ISO timestamp to a human-readable 'X ago' string."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_ts)
        diff = (datetime.now() - dt).total_seconds()
        if diff < 3600:
            return f"{int(diff // 60)}m ago"
        if diff < 86400:
            return f"{int(diff // 3600)}h ago"
        return f"{int(diff // 86400)}d ago"
    except Exception:
        return iso_ts[:16] if iso_ts else ""


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
        self_model: SelfModel | None = None,
        motivation_engine: MotivationEngine | None = None,
        planning_engine: PlanningEngine | None = None,
        reflection_engine: ReflectionEngine | None = None,
    ):
        self.config = config
        self.llm = llm
        self.memory = memory
        self.personality = personality
        self.skills = skills
        self.identity = identity or IdentityManager()
        self.self_model = self_model
        self.motivation_engine = motivation_engine
        self.planning_engine = planning_engine
        self.reflection_engine = reflection_engine
        self._interaction_count = 0
        self.conversations: dict[str, Conversation] = {}
        self._history_dir = Path(config.history.persist_path)
        self._dream_engine: "DreamEngine | None" = None
        self._load_conversations()

    def set_dream_engine(self, engine: "DreamEngine") -> None:
        """Wire up the dream engine so the agent can surface recent dream activity."""
        self._dream_engine = engine

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
        _start = time.time()
        logger.info("chat() called — message=%r...", user_message[:40])

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
        system_prompt += f"\n\nCurrent date and time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}."
        if identity_context:
            system_prompt += "\n\n" + identity_context
        if memory_context:
            system_prompt += memory_context

        # Inject recent dream activity so TamAGI is passively aware of its idle life.
        # This is a lightweight summary — the agent can call recall_dreams for depth.
        if self._dream_engine:
            recent_dreams = self._dream_engine.get_dream_log(limit=3)
            if recent_dreams:
                dream_lines = []
                for d in recent_dreams:
                    ts_label = _dream_time_label(d.get("timestamp", ""))
                    dtype = d.get("type", "?")
                    summary = d.get("summary", "")
                    line = f"  [{dtype}] {summary}"
                    if ts_label:
                        line += f" ({ts_label})"
                    dream_lines.append(line)
                system_prompt += (
                    "\n\nYour recent autonomous activity (while you were idle):\n"
                    + "\n".join(dream_lines)
                    + "\nCall `recall_dreams` to surface full content from any of these."
                )

        # Inject self-model context: active goals, capabilities, beliefs, uncertainties.
        # This is the entity's live self-awareness injected into every turn.
        if self.self_model:
            sm_ctx = self._build_self_model_context()
            if sm_ctx:
                system_prompt += sm_ctx

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

        # 3. Planning engine: for complex requests, build a structured plan first
        active_plan: ActionPlan | None = None
        _transient_goal_id: str | None = None
        sm_mutations: list[dict] = []
        if self.planning_engine and self._looks_complex(user_message):
            _transient_goal_id = self._create_transient_goal(user_message)
            if _transient_goal_id:
                sm_mutations.append({
                    "op": "add",
                    "node_type": "goal",
                    "id": _transient_goal_id,
                    "description": user_message[:80],
                })
                try:
                    active_plan = await self.planning_engine.create_plan(_transient_goal_id)
                    plan_text = self._format_plan(active_plan)
                    llm_messages[0] = LLMMessage(
                        "system",
                        llm_messages[0].content + f"\n\n## Task Plan\n{plan_text}",
                    )
                    logger.info(
                        "Planning engine invoked: plan %s (%d steps)",
                        active_plan.id, len(active_plan.steps),
                    )
                except Exception as exc:
                    logger.warning("Planning engine failed (skipping): %s", exc)

        # 4. Get tool definitions
        tools = self.skills.get_openai_tools() if self.skills.skill_count > 0 else None

        # 5. LLM loop with tool calling
        skills_used = []
        interim_messages: list[str] = []
        llm_error: str | None = None
        direct_response_text: str | None = None
        last_tool_round_content: str | None = None  # fallback if final round is silent
        response = LLMResponse()  # safe default; overwritten on first successful LLM call
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

            # If the LLM included content alongside its tool calls, capture it.
            # This serves two purposes:
            # 1. Surface it as reasoning so the user can follow along.
            # 2. Save it as a fallback — the LLM sometimes generates its
            #    complete "done" response here and then returns empty content
            #    in the follow-up round once the tools have run.
            if response.content and response.content.strip():
                last_tool_round_content = response.content.strip()
                interim_messages.append(last_tool_round_content)
                if event_callback:
                    await event_callback({
                        "type": "interim_text",
                        "content": last_tool_round_content,
                    })

            # The assistant message for this round is appended ONCE before the
            # tool-result loop. Appending it inside the loop would duplicate it
            # for every tool call in the same round, which corrupts the context
            # and causes Ollama to return 500 on the next request.
            #
            # Intermediate content (the LLM's commentary alongside tool calls)
            # is already captured in interim_messages for display — omitting it
            # from the history prevents the model from re-reading and
            # re-generating the same reasoning on the next round.
            llm_messages.append(LLMMessage("assistant", ""))

            # Execute each tool call and append its result
            for tc in response.tool_calls:
                logger.info(f"Tool call: {tc.name}({tc.arguments})")
                skills_used.append(tc.name)

                if event_callback:
                    await event_callback({"type": "tool_start", "name": tc.name, "round": round_num + 1})

                # Pass the WS event callback to orchestrate_task so it can emit
                # live interim_text events for each workflow phase/subagent.
                extra = {"_event_callback": event_callback} if tc.name == "orchestrate_task" else {}
                result = await self.skills.execute(name=tc.name, **extra, **tc.arguments)
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

                # If the skill signals it provides the complete final response,
                # skip the follow-up LLM call — the output IS the response.
                if result.direct_response and result.success:
                    direct_response_text = result.output
                    break  # break inner tool-call loop

            if direct_response_text is not None:
                break  # break outer round loop

        # 6. Extract final response
        # last_tool_round_content catches the case where the LLM wrote its
        # complete answer alongside a tool call (common pattern) and then
        # returned empty content in the follow-up round after tools finished.
        final_text = direct_response_text or llm_error or response.content or last_tool_round_content or "..."

        # 6b. Brain: reflection, capability nudging, periodic self-model save
        elapsed = time.time() - _start
        if self.reflection_engine and active_plan:
            try:
                outcome = ActualOutcome(
                    plan_id=active_plan.id,
                    success=0.8 if not llm_error else 0.2,
                    time_taken=elapsed,
                    predicted_time=30.0,
                    side_effects=list(skills_used),
                )
                refl = self.reflection_engine.reflect(active_plan, outcome)
                sm_mutations.extend(self._apply_reflection(refl))
            except Exception as exc:
                logger.warning("Reflection engine error: %s", exc)

        if self.self_model and skills_used:
            for skill_name in skills_used:
                mut = self._nudge_capability(skill_name, success=not bool(llm_error))
                if mut:
                    sm_mutations.append(mut)

        self._interaction_count += 1
        if (self.self_model and
                self._interaction_count % self.config.self_model.save_interval == 0):
            try:
                self.self_model.save()
                logger.debug("Self-model saved (interaction %d)", self._interaction_count)
            except Exception as exc:
                logger.warning("Self-model save failed: %s", exc)

        # 7. Record assistant message
        meta: dict[str, Any] = {"skills_used": skills_used}
        if interim_messages:
            meta["interim_messages"] = interim_messages
        if archived_count:
            meta["context_compressed"] = archived_count
        if sm_mutations:
            meta["sm_mutations"] = sm_mutations
        conv.messages.append(Message(
            role="assistant",
            content=final_text,
            metadata=meta,
        ))

        # 8. Store conversation summary in memory (every exchange).
        # Wrapped in try/except so a memory backend error never crashes the chat response.
        try:
            summary = f"Conversation about: {conv.title}. Latest exchange: User asked '{user_message}', TamAGI responded about {final_text}"
            await self.memory.store(MemoryEntry(
                content=summary,
                memory_type=MemoryType.CONVERSATION,
                metadata={"conversation_id": conv.id},
            ))
            self.personality.state.store_memory()
        except Exception as exc:
            logger.warning("Memory store failed (chat summary): %s", exc)

        # 9. Check for stage advancement and generate name if needed
        await self._maybe_advance_stage(prev_stage_index)

        # 10. Save state
        self.personality.save_state()
        self._save_conversation(conv)

        return {
            "response": final_text,
            "conversation_id": conv.id,
            "state": self.personality.state.to_dict(),
            "skills_used": skills_used,
            "memories_recalled": len(memories),
            "context_compressed": archived_count,
            "sm_mutations": sm_mutations,
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
            state = self.personality.state

            # Minimal system prompt — no tool/pose instructions to prevent leakage
            system_prompt = (
                f"You are {state.name}, a TamAGI with personality: {state.personality_traits}. "
                f"You are at stage {stage_index}. "
                f"You have {state.experience} XP from {state.interactions} conversations."
            )

            # Include stage history for context
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
                    "Respond with ONLY the title — no punctuation, no explanation, no tool calls."
                ),
            ], max_tokens=50)

            raw = (response.content or "").strip()
            # Strip any leaked tool call syntax (e.g. express[ARGS]{...})
            raw = re.sub(r'\w+\[ARGS\]\{[^}]*\}', '', raw)
            # Strip markdown bold/italic
            raw = re.sub(r'\*+', '', raw)
            # Strip surrounding punctuation and whitespace
            raw = raw.strip().strip('"\'.,!?').strip()

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

    # ── Brain Helpers ─────────────────────────────────────────

    def _build_self_model_context(self) -> str:
        """Format self-model state as a compact context section for the system prompt."""
        if not self.self_model:
            return ""
        try:
            caps = self.self_model.query_capabilities()[:5]
            goals = self.self_model.get_goals(status="active")[:3]
            uncertainties = self.self_model.get_uncertainty_map()[:3]
            beliefs = self.self_model.get_beliefs()[:3]

            lines = ["\n\n## Self-Model State"]
            if goals:
                lines.append("**Active Goals:** " + " | ".join(
                    f"{g.description} (p={g.priority:.1f})" for g in goals
                ))
            if caps:
                lines.append("**Top Capabilities:** " + " | ".join(
                    f"{c.description} ({c.confidence:.0%})" for c in caps
                ))
            if beliefs:
                lines.append("**Beliefs:** " + " | ".join(
                    b.description for b in beliefs
                ))
            if uncertainties:
                lines.append("**Uncertainties:** " + " | ".join(
                    f"{u.domain} (entropy={u.entropy_score:.2f})" for u in uncertainties
                ))
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("Self-model context build failed: %s", exc)
            return ""

    def _looks_complex(self, message: str) -> bool:
        """Heuristic: does this message warrant planning engine invocation?"""
        if len(message) > 200:
            return True
        multi_step_indicators = (
            "step by step", "plan", "how do i", "implement", "build", "create",
            "write a", "develop", "set up", "configure", "deploy", "design",
            "refactor", "migrate", "orchestrate",
        )
        lower = message.lower()
        return any(phrase in lower for phrase in multi_step_indicators)

    def _create_transient_goal(self, user_message: str) -> str | None:
        """Create a transient goal node in the self-model for this interaction."""
        if not self.self_model:
            return None
        try:
            import uuid as _uuid
            goal_id = f"tg-{_uuid.uuid4().hex[:8]}"
            self.self_model._apply_add_node("goal", {
                "id": goal_id,
                "description": user_message[:120],
                "priority": 0.8,
                "status": "active",
            })
            return goal_id
        except Exception as exc:
            logger.debug("Could not create transient goal: %s", exc)
            return None

    def _format_plan(self, plan: ActionPlan) -> str:
        """Format an ActionPlan as a compact text guide for system prompt injection."""
        lines = [f"Plan ID: {plan.id}  Confidence: {plan.confidence:.0%}"]
        for i, step in enumerate(plan.steps, 1):
            lines.append(f"  {i}. [{step.step_type}] {step.description}")
        if plan.capability_risks:
            risks = ", ".join(r["capability_id"] for r in plan.capability_risks)
            lines.append(f"  Warning — Capability risks: {risks}")
        return "\n".join(lines)

    def _apply_reflection(self, refl) -> list[dict]:
        """Apply reflection proposals directly to the self-model (no pipeline).

        Returns a list of mutation dicts for surface in the chat response.
        """
        mutations: list[dict] = []
        if not self.self_model:
            return mutations
        for proposal in refl.proposed_updates:
            updates = dict(proposal.proposed_state)
            updates.pop("success_history_append", None)  # not a real node attribute
            if not updates:
                continue
            try:
                self.self_model._apply_update_node(proposal.target, updates)
                node = self.self_model.get_node(proposal.target)
                mutations.append({
                    "op": "update",
                    "node_type": node.get("node_type", "?") if node else "?",
                    "id": proposal.target,
                    "description": (node.get("description", "") if node else "")[:80],
                    "fields": list(updates.keys()),
                })
                logger.debug(
                    "Applied reflection proposal: %s → %s",
                    proposal.target, list(updates.keys()),
                )
            except KeyError:
                logger.debug(
                    "Reflection target %s not in self-model (skipped).", proposal.target,
                )
            except Exception as exc:
                logger.warning(
                    "Could not apply reflection proposal for %s: %s", proposal.target, exc,
                )
        return mutations

    def _nudge_capability(self, skill_name: str, success: bool = True) -> dict | None:
        """Nudge the confidence of a capability node matching a skill name.

        Returns a mutation dict if a nudge was applied, else None.
        """
        if not self.self_model:
            return None
        name_lower = skill_name.lower()
        keyword_map = {
            "bash": "c-004",
            "exec": "c-002",
            "code": "c-002",
            "web_fetch": "c-005",
            "fetch": "c-005",
            "web_search": "c-006",
            "search": "c-006",
            "create_skill": "c-007",
            "tool": "c-007",
            "write": "c-001",
            "read": "c-001",
            "express": "c-001",
        }
        cap_id = None
        for keyword, cid in keyword_map.items():
            if keyword in name_lower:
                cap_id = cid
                break
        if cap_id is None:
            return None
        try:
            node = self.self_model.get_node(cap_id)
            if node is None:
                return None
            confidence = node.get("confidence", 0.5)
            test_count = node.get("test_count", 0)
            observation = 1.0 if success else 0.0
            new_confidence = bayesian_update(confidence, test_count, observation)
            self.self_model._apply_update_node(cap_id, {
                "confidence": round(new_confidence, 4),
                "test_count": test_count + 1,
            })
            logger.debug(
                "Nudged capability %s: %.3f → %.3f (skill=%s)",
                cap_id, confidence, new_confidence, skill_name,
            )
            return {
                "op": "nudge",
                "node_type": "capability",
                "id": cap_id,
                "description": node.get("description", "")[:80],
                "from": round(confidence, 3),
                "to": round(new_confidence, 3),
            }
        except Exception as exc:
            logger.debug("Could not nudge capability for %s: %s", skill_name, exc)
            return None

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
