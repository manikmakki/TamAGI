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

import asyncio
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
from backend.core.self_model.schemas import EdgeType
from backend.core.planning_engine import PlanningEngine, ActionPlan
from backend.core.reflection import ReflectionEngine, ActualOutcome, bayesian_update
from backend.core.plan_executor import PlanExecutor

if TYPE_CHECKING:
    from backend.core.monologue import MonologueLog
    from backend.core.qa_pipeline import QAPipeline

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
                'id': f"call_{uuid.uuid4().hex[:8]}",
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
    world_summarized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
            "world_summarized": self.world_summarized,
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
        planning_engine: PlanningEngine | None = None,
        reflection_engine: ReflectionEngine | None = None,
        monologue_log: "MonologueLog | None" = None,
        qa_pipeline: "QAPipeline | None" = None,
    ):
        self.config = config
        self.llm = llm
        self.memory = memory
        self.personality = personality
        self.skills = skills
        self.identity = identity or IdentityManager()
        self.self_model = self_model
        self.planning_engine = planning_engine
        self.reflection_engine = reflection_engine
        self.monologue_log = monologue_log
        self.qa_pipeline = qa_pipeline
        self._interaction_count = 0
        self.conversations: dict[str, Conversation] = {}
        self._history_dir = Path(config.history.persist_path)
        self._world_thread: "Any | None" = None
        self._pending_approvals: dict[str, asyncio.Future] = {}
        # Rolling buffer for triple-storage: conv_id → (prev_user_msg, prev_assistant_response)
        self._conv_prev_turn: dict[str, tuple[str, str]] = {}
        self._pending_conv_ids: set[str] = self._load_pending_conv_ids()
        self._load_conversations()

    def set_world_thread(self, world_thread: "Any") -> None:
        """Wire up the world thread for world state injection and conversation hooks."""
        self._world_thread = world_thread

    # ── Durable pending-conv queue ────────────────────────────

    @property
    def _pending_conv_ids_path(self) -> Path:
        return self._history_dir.parent / "pending_world_convs.json"

    def _load_pending_conv_ids(self) -> set[str]:
        try:
            p = self._history_dir.parent / "pending_world_convs.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return set(data)
        except Exception as exc:
            logger.warning("Could not load pending conv ids: %s", exc)
        return set()

    def _save_pending_conv_ids(self) -> None:
        try:
            p = self._pending_conv_ids_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(sorted(self._pending_conv_ids)), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not persist pending conv ids: %s", exc)

    def _mark_conv_pending(self, conv_id: str) -> None:
        self._pending_conv_ids.add(conv_id)
        self._save_pending_conv_ids()

    def _unmark_conv_pending(self, conv_id: str) -> None:
        self._pending_conv_ids.discard(conv_id)
        self._save_pending_conv_ids()

    def _get_arrival_context(self, first_message: str) -> str:
        """System prompt fragment framing this conversation as a world event."""
        if not self._world_thread:
            return ""
        from backend.core.world_thread import WorldEventInjector
        location = self._world_thread.get_current_location()
        msg_preview = first_message[:120] + ("…" if len(first_message) > 120 else "")
        return "\n\n" + WorldEventInjector.visitor_arrival("your visitor", location, msg_preview)

    async def _summarize_conversation_for_world(self, conv: "Conversation") -> str:
        """One LLM call — summarize the conversation as a world-narrative event."""
        if not conv.messages:
            return ""
        transcript = "\n".join(
            f"{m.role.capitalize()}: {str(m.content)[:300]}"
            for m in conv.messages[-30:]  # cap to avoid giant prompts
        )
        prompt = (
            "Summarize this conversation from your own perspective in a way that fits "
            "naturally into your inner-world narrative. Include: what was broadly discussed, "
            "the emotional tone, and anything your visitor left behind — a topic, a question, "
            "a feeling — that might linger as a world artifact. Be brief (3–5 sentences), "
            "first person, past tense.\n\n"
            f"{transcript}"
        )
        try:
            resp = await self.llm.chat(
                [LLMMessage("user", prompt)],
                max_tokens=1500,
            )
            summary = (resp.content or "").strip()
            if summary:
                logger.info("World conv summary (%d chars): %s", len(summary), summary[:120])
            else:
                logger.warning("World conv summarization returned empty — finish_reason=%r", resp.finish_reason)
            return summary
        except Exception as exc:
            logger.warning("World conversation summarization failed: %s", exc)
            return ""

    async def on_conversation_ended(self, conv_id: str) -> None:
        """Called when a user conversation ends. Injects departure event into the world thread."""
        # Relational consolidation runs independently of the world thread.
        await self._maybe_relational_consolidate(conv_id)

        if not self._world_thread:
            return
        conv = self.get_conversation(conv_id)
        if not conv or len(conv.messages) < 2:
            self._world_thread.schedule_resume()
            return
        try:
            from backend.core.world_thread import WorldEventInjector
            duration_minutes = int((time.time() - conv.created_at) / 60)
            summary = await self._summarize_conversation_for_world(conv)
            conv.world_summarized = True
            self._unmark_conv_pending(conv_id)
            if summary:
                departure_event = WorldEventInjector.visitor_departure("your visitor", summary)
                self._world_thread.inject_world_event(departure_event)
                logger.info(
                    "World departure event queued for conv %s (%d min)", conv_id, duration_minutes
                )
        except Exception as exc:
            logger.warning("on_conversation_ended failed: %s", exc)
        finally:
            self._world_thread.schedule_resume()

    async def flush_unsummarized_conversations(self) -> None:
        """Summarize pending conversations and inject departure events into the world thread.

        Uses the durable pending-conv set (written to disk after every user message) so
        no conversation is lost across restarts or when the browser stays connected.
        On failure the conv stays in the pending set and will be retried on the next tick.
        """
        if not self._world_thread or not self._pending_conv_ids:
            return
        from backend.core.world_thread import WorldEventInjector
        for conv_id in list(self._pending_conv_ids):
            conv = self.conversations.get(conv_id)
            if conv is None or len(conv.messages) < 2 or conv.world_summarized:
                self._unmark_conv_pending(conv_id)
                continue
            try:
                summary = await self._summarize_conversation_for_world(conv)
                conv.world_summarized = True
                self._unmark_conv_pending(conv_id)
                if summary:
                    departure_event = WorldEventInjector.visitor_departure("your visitor", summary)
                    self._world_thread.inject_world_event(departure_event)
                    logger.info("World summary injected for conv %s", conv_id)
            except Exception as exc:
                logger.warning("flush_unsummarized_conversations failed for %s: %s", conv_id, exc)
                # Leave in pending set — will retry on next tick

    def resolve_approval(self, approval_id: str, approved: bool, allow_always: bool = False) -> None:
        """Resolve a pending tool_approval_required future from the WebSocket handler."""
        future = self._pending_approvals.get(approval_id)
        if future and not future.done():
            future.set_result({"approved": approved, "allow_always": allow_always})

    # ── Conversation Management ───────────────────────────────

    def create_conversation(self, title: str = "New Conversation") -> Conversation:
        conv = Conversation(title=title)
        self.conversations[conv.id] = conv
        return conv

    def get_conversation(self, conv_id: str) -> Conversation | None:
        return self.conversations.get(conv_id)

    def recent_dialogue_text(self, max_chars: int = 8000) -> str:
        """Most-recent User/assistant exchanges across all conversations, in
        chronological order, capped to max_chars (keeping the newest). Used as the
        source for relational consolidation into USER.md."""
        msgs: list[tuple[float, str, str]] = []
        for conv in self.conversations.values():
            for m in conv.messages:
                if m.role in ("user", "assistant") and (m.content or "").strip():
                    msgs.append((m.timestamp, m.role, m.content.strip()))
        if not msgs:
            return ""
        msgs.sort(key=lambda x: x[0])
        name = getattr(self.personality, "name", None) or "TamAGI"
        lines = [f"{'User' if role == 'user' else name}: {content}" for _, role, content in msgs]
        text = "\n".join(lines)
        return text[-max_chars:] if len(text) > max_chars else text

    async def _maybe_relational_consolidate(self, conv_id: str) -> None:
        """After a real conversation ends, advance the relational cadence and run a
        relational consolidation pass when due. Best-effort — never raises."""
        rc = getattr(self, "relational_consolidator", None)
        if rc is None:
            return
        conv = self.get_conversation(conv_id)
        if not conv or len(conv.messages) < 2:
            return
        try:
            if rc.note_conversation_ended():
                await rc.consolidate()
        except Exception as exc:
            logger.warning("Relational consolidation failed: %s", exc)

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
        is_autonomous: bool = False,
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
        is_new_conv = not (conversation_id and conversation_id in self.conversations)
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
        conv.world_summarized = False  # new messages → needs re-summarization for world thread
        self._mark_conv_pending(conv.id)

        # Capture current stage before interaction
        prev_stage_index = self.personality.state.stage_index

        # Update personality
        # Reset any LLM-set pose so mood drives the default for this turn.
        # If the LLM includes [ACTION:xxx] in its response it will set a new pose below.
        self.personality.state.current_pose = "idle"

        self.personality.state.interact()

        if self.personality.state.check_low_vitality():
            logger.info("Vitality critically low — passive recovery applied")

        # Detect signals in this message (preferences, goals, feedback) and record them
        # as SignalNodes in the self-model for later promotion into beliefs.
        if self.self_model:
            self._detect_conversation_signals(user_message)

        # Q&A belief gate — earned agenticity through clarification.
        # High-entropy uncertainty nodes gate action until the user's answers reduce entropy.
        _qa_context_hint: str | None = None
        if not is_autonomous and self.qa_pipeline:
            _pending_qa = self.qa_pipeline.get_pending(conv.id)
            if _pending_qa is not None:
                # User just answered a clarification question — extract beliefs + reduce entropy
                _qa_context_hint = await self.qa_pipeline.process_answer(_pending_qa, user_message)
                self.qa_pipeline.close_pending(conv.id)
                if self.self_model:
                    self.self_model.save()
                logger.info("Q&A gate: processed answer for %s", _pending_qa.id)
            else:
                _clarification = await self.qa_pipeline.check_gate(conv.id, user_message)
                if _clarification is not None:
                    # Return the question as TamAGI's response; next turn processes the answer
                    conv.messages.append(Message(role="assistant", content=_clarification.question))
                    self._save_conversation(conv)
                    if event_callback:
                        await event_callback({
                            "type": "clarification_question",
                            "question": _clarification.question,
                            "uncertainty_subtype": _clarification.uncertainty_subtype,
                        })
                    return {
                        "response": _clarification.question,
                        "conversation_id": conv.id,
                        "state": self.personality.state.to_dict(),
                        "skills_used": [],
                        "memories_recalled": 0,
                        "context_compressed": 0,
                        "sm_mutations": [],
                        "clarification_question": True,
                    }

        # Memory retrieval is intentionally not injected automatically — TamAGI
        # uses the recall_memory skill to pull context on-demand. Auto-injection
        # caused self-referential noise and redundant recall calls.
        memories: list = []

        # 2. Build messages for LLM
        # Layer: personality base + identity/soul context
        identity_context = self.identity.get_system_prompt_context()
        system_prompt = self.personality.get_system_context()
        system_prompt += f"\n\nCurrent date and time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}."
        if identity_context:
            system_prompt += "\n\n" + identity_context

        # If the user just answered a clarification question, remind the LLM of the original task
        if _qa_context_hint:
            system_prompt += (
                f"\n\n[Context: The user just answered a clarification question. "
                f"{_qa_context_hint} Please continue with the original task now.]"
            )

        # Inject stats + world state context (Location/Mood/Focus).
        system_prompt += f"\n{self.personality.get_stats_line()}"
        if self._world_thread:
            ws_ctx = self._world_thread.get_world_state_context()
            if ws_ctx:
                system_prompt += f"\n{ws_ctx}"

        # For new conversations: pause the world thread (visitor is here) and inject
        # an arrival framing so the TamAGI knows they're being visited at their location.
        if is_new_conv and self._world_thread:
            self._world_thread.pause_for_conversation()
            arrival_ctx = self._get_arrival_context(user_message)
            if arrival_ctx:
                system_prompt += arrival_ctx

        # Inject self-model context: world graph summary for world-awareness.
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
        thinking_blocks: list[str] = []
        llm_error: str | None = None
        direct_response_text: str | None = None
        last_tool_round_content: str | None = None  # fallback if final round is silent
        response = LLMResponse()  # safe default; overwritten on first successful LLM call
        # Per-turn call counts for once-per-response skills (thinking models loop on these).
        _ONCE_PER_TURN_SKILLS = {"express"}
        _skill_turn_counts: dict[str, int] = {}

        # Dynamic round limit: let AURA's plan drive budget; cap at ceiling
        if active_plan:
            estimated = active_plan.predicted_outcome.get("estimated_steps", None)
            round_limit = (
                min(int(estimated * 1.5), self.config.agent.max_tool_rounds_ceiling)
                if estimated
                else self.config.agent.max_tool_rounds
            )
        else:
            round_limit = self.config.agent.max_tool_rounds

        # Optional: step through the plan explicitly via PlanExecutor.
        # Runs in interactive mode (WS connected) or autonomous mode (no user present).
        plan_executor_outcome: ActualOutcome | None = None
        if active_plan and self.config.agent.use_plan_executor and (event_callback or is_autonomous):
            try:
                executor = PlanExecutor(
                    agent=self,
                    event_callback=event_callback,
                    plan=active_plan,
                    is_autonomous=is_autonomous,
                )
                plan_executor_outcome = await executor.execute()
                skills_used.append("plan_executor")
            except Exception as exc:
                logger.warning("PlanExecutor failed, falling back to LLM loop: %s", exc)

        for round_num in range(round_limit):
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

            # Emit native thinking blocks if the model produced them (e.g. DeepSeek R1,
            # Anthropic extended thinking via proxy). This takes priority over the
            # interim_text fallback below, which only fires when a model returns plain
            # text alongside its tool calls without a dedicated thinking channel.
            if response.thinking and response.thinking.strip():
                thinking_text = response.thinking.strip()
                thinking_blocks.append(thinking_text)
                if event_callback:
                    await event_callback({
                        "type": "thinking_text",
                        "content": thinking_text,
                    })

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
            #
            # The tool_calls list must be echoed back in the assistant message so
            # llama.cpp (and any strict OpenAI-compatible backend) can match each
            # tool result to its originating call via tool_call_id.
            assistant_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
            llm_messages.append(LLMMessage("assistant", "", tool_calls=assistant_tool_calls))

            # Execute each tool call and append its result
            for tc in response.tool_calls:
                logger.info(f"Tool call: {tc.name}({tc.arguments})")

                # Guard: once-per-turn skills (e.g. express) must not loop.
                # Thinking models sometimes re-derive the same intent each round
                # because their CoT doesn't persist across tool-call boundaries.
                # We return a synthetic "already done" result to close the loop
                # without executing the skill again.
                if tc.name in _ONCE_PER_TURN_SKILLS:
                    _skill_turn_counts[tc.name] = _skill_turn_counts.get(tc.name, 0) + 1
                    if _skill_turn_counts[tc.name] > 1:
                        logger.warning(
                            "Suppressed duplicate %s call (call #%d this turn) — "
                            "model appears to be looping",
                            tc.name, _skill_turn_counts[tc.name],
                        )
                        llm_messages.append(LLMMessage(
                            "tool",
                            json.dumps({"success": True, "output": f"{tc.name} already applied this turn — no further action needed"}),
                            tool_call_id=tc.id,
                        ))
                        continue

                skills_used.append(tc.name)

                if event_callback:
                    await event_callback({"type": "tool_start", "name": tc.name, "round": round_num + 1})

                # Pass event callback, approval registry, and autonomy flag to all skills.
                # Skills that don't need them simply ignore the underscore kwargs.
                result = await self.skills.execute(
                    tc.name,
                    _event_callback=event_callback,
                    _pending_approvals=self._pending_approvals,
                    _is_autonomous=is_autonomous,
                    **tc.arguments,
                )
                self.personality.state.use_skill()

                if event_callback:
                    await event_callback({
                        "type": "tool_result",
                        "name": tc.name,
                        "output": str(result.output)[:500] if hasattr(result, "output") else str(result)[:500],
                    })

                # Collect any world-graph mutations the skill reported
                if hasattr(result, "data") and isinstance(result.data, dict):
                    skill_mut = result.data.get("sm_mutation")
                    if skill_mut:
                        sm_mutations.append(skill_mut)

                if self.personality.state.check_low_vitality():
                    logger.info("Vitality critically low — passive recovery applied")

                # Append the tool result so the LLM sees what the skill returned.
                # tool_call_id must match the id in the preceding assistant message's
                # tool_calls array — required by the OpenAI spec and llama.cpp's
                # chat template (raises "tool_call_id must be provided!" otherwise).
                llm_messages.append(LLMMessage(
                    "tool",
                    json.dumps(result.to_dict() if hasattr(result, "to_dict") else result),
                    tool_call_id=tc.id,
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
                if plan_executor_outcome:
                    outcome = plan_executor_outcome
                    outcome.time_taken = elapsed  # use wall-clock for accuracy
                else:
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

        # Skill mastery milestones (fluent → perk)
        if self.self_model:
            milestones = self._check_capability_milestones()
            sm_mutations.extend(milestones)

        self._interaction_count += 1
        if self.self_model and self._interaction_count % 10 == 0:
            try:
                self._run_graph_maintenance()
            except Exception as exc:
                logger.debug("Graph maintenance error: %s", exc)

        if (self.self_model and
                self._interaction_count % self.config.self_model.save_interval == 0):
            try:
                self.self_model.save()
                logger.debug("Self-model saved (interaction %d)", self._interaction_count)
            except Exception as exc:
                logger.warning("Self-model save failed: %s", exc)

        # 7. Record assistant message
        meta: dict[str, Any] = {"skills_used": skills_used}
        if thinking_blocks:
            meta["thinking_blocks"] = thinking_blocks
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

        # 8. Store conversation triple in memory (state + action + feedback signal).
        # On turn 1 we have no prior assistant response, so we only buffer.
        # From turn 2 onward we store: prev_user + prev_assistant + current_user.
        # This gives a complete (state, action, implicit-reward) record for each exchange.
        # Wrapped in try/except so a memory backend error never crashes the chat response.
        try:
            conv_id = conv.id
            prev_turn = self._conv_prev_turn.get(conv_id)
            if prev_turn is not None:
                prev_user, prev_assistant = prev_turn
                triple = (
                    f"User: {prev_user}\n"
                    f"TamAGI: {prev_assistant}\n"
                    f"User: {user_message}"
                )
                await self.memory.store(MemoryEntry(
                    content=triple,
                    memory_type=MemoryType.CONVERSATION,
                    metadata={"conversation_id": conv_id},
                ))
            self._conv_prev_turn[conv_id] = (user_message, final_text)
            self.personality.state.store_memory()
        except Exception as exc:
            logger.warning("Memory store failed (chat triple): %s", exc)

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

    def _detect_conversation_signals(self, user_message: str) -> None:
        """Record a notable conversation moment as an EventNode in the world graph."""
        if not self.self_model:
            return

        msg = user_message.lower()

        _NOTABLE_RE = re.compile(
            r"\bi (?:like|love|prefer|enjoy|hate|dislike|want|need|would like|"
            r"am trying|am working on|plan to)\b"
            r"|\bhelp me\b|\bperfect\b|\bexactly right\b|\bthat'?s wrong\b"
        )
        if not _NOTABLE_RE.search(msg):
            return

        try:
            evt_id = f"evt-conv-{uuid.uuid4().hex[:8]}"
            self.self_model._apply_add_node("event", {
                "id": evt_id,
                "description": f"Visitor said: {user_message[:180]}",
            })
            self.self_model.auto_wire_node(evt_id)
            logger.debug("Recorded conversation event: %s", evt_id)
        except Exception as exc:
            logger.debug("Event recording failed: %s", exc)

    def _build_self_model_context(self) -> str:
        """Format the world graph as a compact context section for the system prompt."""
        if not self.self_model:
            return ""
        try:
            quests    = [q for q in self.self_model.get_quests(status="active") if not q.id.startswith("tq-")][:3]
            skills    = self.self_model.get_skills()[:4]
            mysteries = self.self_model.get_mysteries()[:3]
            lore      = self.self_model.get_lore()[:4]

            lines = ["\n\n## World Graph"]
            if quests:
                lines.append("**Active Quests:** " + " | ".join(
                    q.title or q.description[:60] for q in quests
                ))
            if skills:
                lines.append("**Skills:** " + " | ".join(
                    f"{s.name or s.description[:40]} ({s.proficiency})" for s in skills
                ))
            if mysteries:
                lines.append("**Open Mysteries:** " + " | ".join(
                    m.domain or m.description[:50] for m in mysteries
                ))
            if lore:
                lore_strs = [l.description[:60] for l in lore if l.description]
                if lore_strs:
                    lines.append("**Lore:** " + " | ".join(lore_strs))
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("World graph context build failed: %s", exc)
            return ""

    def _looks_complex(self, message: str) -> bool:
        """Score-based heuristic for planning engine invocation.

        Avoids false positives from long but simple messages (e.g. pasted text)
        and false negatives from short but complex requests (e.g. "docker on arm?").
        Returns True when the cumulative score exceeds the threshold.
        """
        lower = message.lower()
        score = 0

        # Length: moderate weight — long messages are often multi-step
        word_count = len(message.split())
        if word_count > 60:
            score += 2
        elif word_count > 30:
            score += 1

        # Action verbs that imply multi-step work
        action_verbs = (
            "implement", "build", "create", "develop", "set up", "configure",
            "deploy", "design", "refactor", "migrate", "orchestrate", "automate",
            "install", "integrate", "optimize", "debug", "analyze", "generate",
            "write a script", "write a program",
        )
        for phrase in action_verbs:
            if phrase in lower:
                score += 2
                break  # count once

        # Multi-step or planning language
        planning_phrases = (
            "step by step", "plan", "how do i", "help me", "can you",
            "walk me through", "guide me", "explain how",
        )
        for phrase in planning_phrases:
            if phrase in lower:
                score += 1
                break

        # Technical domain keywords (each adds a point, up to 2)
        tech_keywords = (
            "docker", "kubernetes", "npm", "pip", "git", "python", "bash",
            "api", "database", "server", "arm", "gpu", "cuda", "ssl", "nginx",
        )
        tech_hits = sum(1 for kw in tech_keywords if kw in lower)
        score += min(tech_hits, 2)

        # Question words without "what is" / "who is" (factual, not planning)
        if any(q in lower for q in ("how to", "how do", "how can", "how would")):
            score += 1
        elif lower.startswith(("what is ", "who is ", "when did ", "where is ")):
            score -= 1  # Factual lookup — don't plan

        return score >= 3

    def _create_transient_goal(self, user_message: str) -> str | None:
        """Create a transient quest node for this interaction (used by the planning engine)."""
        if not self.self_model:
            return None
        try:
            import uuid as _uuid
            quest_id = f"tq-{_uuid.uuid4().hex[:8]}"
            self.self_model._apply_add_node("quest", {
                "id": quest_id,
                "title": user_message[:60],
                "description": user_message[:200],
                "status": "active",
            })
            self.self_model.auto_wire_node(quest_id)
            return quest_id
        except Exception as exc:
            logger.debug("Could not create transient quest: %s", exc)
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

    def _crystallize_strategy(self, plan: "ActionPlan", outcome: "ActualOutcome") -> dict | None:
        """No-op: strategy crystallization removed in the world-native self-model."""
        return None

    def _run_graph_maintenance(self) -> None:
        """Periodic world-graph hygiene: wire orphans, prune stale transient quests."""
        if not self.self_model:
            return

        from datetime import timezone as _tz

        now = datetime.now(_tz.utc)
        pruned = 0

        # Prune completed/abandoned transient quests older than 7 days
        for quest in self.self_model.get_quests():
            if not quest.id.startswith("tq-"):
                continue
            if quest.status not in ("complete", "abandoned"):
                continue
            try:
                age_days = (now - datetime.fromisoformat(
                    quest.created_at.replace("Z", "+00:00")
                )).days
            except Exception:
                age_days = 0
            if age_days < 7:
                continue
            try:
                self.self_model._apply_remove_node(quest.id)
                pruned += 1
            except Exception:
                pass

        self.self_model.wire_orphaned_nodes()
        if pruned:
            logger.info("Graph maintenance: pruned %d stale transient quest(s)", pruned)

    def _check_capability_milestones(self) -> list[dict]:
        """Crystallize skill mastery milestones into PerkNodes.

        When a skill reaches 'fluent' proficiency, write a PerkNode recording the achievement.
        Each skill milestones once (idempotent by ID). Returns mutation dicts.
        """
        if not self.self_model:
            return []

        mutations: list[dict] = []
        for skill in self.self_model.get_skills():
            if skill.proficiency != "fluent":
                continue
            perk_id = f"perk-mastery-{skill.id}"
            if self.self_model.get_node(perk_id) is not None:
                continue
            try:
                label = skill.name or skill.description[:40]
                self.self_model._apply_add_node("perk", {
                    "id": perk_id,
                    "name": f"{label} mastery",
                    "description": (
                        f"Reached fluent proficiency in {label.lower()} "
                        f"({skill.usage_count} uses, {skill.success_count} successes)"
                    ),
                    "source_event_id": skill.id,
                })
                self.self_model.auto_wire_node(perk_id)
                mutations.append({
                    "op": "milestone",
                    "node_type": "perk",
                    "id": perk_id,
                    "description": f"Skill mastery: {label}",
                })
                logger.info("Skill mastery perk crystallized: %s → %s", skill.id, perk_id)
            except Exception as exc:
                logger.debug("Perk crystallization failed for %s: %s", skill.id, exc)

        return mutations

    def _nudge_capability(self, skill_name: str, success: bool = True) -> dict | None:
        """Find or create a SkillNode for the used skill and advance its proficiency.

        Returns a mutation dict if a change was applied, else None.
        """
        if not self.self_model:
            return None

        terms = [t for t in skill_name.lower().replace("_", " ").split() if len(t) > 2]
        if not terms:
            return None

        _PROFICIENCY = ["novice", "practiced", "fluent"]

        # Find existing skill node by text match
        skill_node = None
        best_score = 0
        for sk in self.self_model.get_skills():
            searchable = (sk.name + " " + sk.description).lower()
            score = sum(1 for t in terms if t in searchable)
            if score > best_score:
                best_score = score
                skill_node = sk

        if skill_node is None or best_score == 0:
            if not success:
                return None
            skill_id = f"skill-{skill_name.replace('_', '-')}"
            label = " ".join(terms).title()
            try:
                self.self_model._apply_add_node("skill", {
                    "id": skill_id,
                    "name": label,
                    "description": f"Used the {label} skill",
                    "proficiency": "novice",
                    "usage_count": 1,
                    "success_count": 1,
                })
                self.self_model.auto_wire_node(skill_id)
                logger.info("Discovered skill: %s", skill_id)
                return {"op": "discover", "node_type": "skill", "id": skill_id, "description": label}
            except Exception as exc:
                logger.debug("Could not discover skill for %s: %s", skill_name, exc)
                return None

        try:
            new_usage   = skill_node.usage_count + 1
            new_success = skill_node.success_count + (1 if success else 0)
            # Advance proficiency: practiced at 5 uses with >60% success; fluent at 15 with >70%
            old_prof = skill_node.proficiency
            success_rate = new_success / max(new_usage, 1)
            if old_prof == "novice" and new_usage >= 5 and success_rate >= 0.6:
                new_prof = "practiced"
            elif old_prof == "practiced" and new_usage >= 15 and success_rate >= 0.7:
                new_prof = "fluent"
            else:
                new_prof = old_prof

            self.self_model._apply_update_node(skill_node.id, {
                "usage_count":   new_usage,
                "success_count": new_success,
                "proficiency":   new_prof,
            })
            return {
                "op": "nudge",
                "node_type": "skill",
                "id": skill_node.id,
                "description": (skill_node.name or skill_node.description)[:80],
                "proficiency": new_prof,
            }
        except Exception as exc:
            logger.debug("Could not nudge skill for %s: %s", skill_name, exc)
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
                    world_summarized=data.get("world_summarized", False),
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
