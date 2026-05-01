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
from backend.core.motivation import MotivationEngine
from backend.core.planning_engine import PlanningEngine, ActionPlan
from backend.core.reflection import ReflectionEngine, ActualOutcome, bayesian_update
from backend.core.plan_executor import PlanExecutor

if TYPE_CHECKING:
    from backend.core.dreamer import DreamEngine
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
        self.motivation_engine = motivation_engine
        self.planning_engine = planning_engine
        self.reflection_engine = reflection_engine
        self.monologue_log = monologue_log
        self.qa_pipeline = qa_pipeline
        self._interaction_count = 0
        self.conversations: dict[str, Conversation] = {}
        self._history_dir = Path(config.history.persist_path)
        self._dream_engine: "DreamEngine | None" = None
        self._pending_approvals: dict[str, asyncio.Future] = {}
        self._load_conversations()

    def set_dream_engine(self, engine: "DreamEngine") -> None:
        """Wire up the dream engine so the agent can surface recent dream activity."""
        self._dream_engine = engine

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

                # Pass event callback, approval registry, and autonomy flag to all skills.
                # Skills that don't need them simply ignore the underscore kwargs.
                result = await self.skills.execute(
                    name=tc.name,
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

                # Check if energy dropped critically low; trigger dream recovery if needed
                if self.personality.state.check_low_energy(agent=self):
                    logger.info(f"Energy critical ({self.personality.state.energy}%), dream recovery triggered")

                # Append the tool result so the LLM sees what the skill returned
                llm_messages.append(LLMMessage(
                    "tool",
                    json.dumps(result.to_dict() if hasattr(result, "to_dict") else result),
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

        # Signal→Belief promotion + milestone crystallization + strategy crystallization
        if self.self_model:
            self._promote_signals()
            milestones = self._check_capability_milestones()
            sm_mutations.extend(milestones)

        if self.self_model and active_plan and plan_executor_outcome:
            strat_mut = self._crystallize_strategy(active_plan, plan_executor_outcome)
            if strat_mut:
                sm_mutations.append(strat_mut)

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

    def _detect_conversation_signals(self, user_message: str) -> None:
        """Scan a user message for preference/goal/feedback signals and record them.

        Matching messages create SignalNodes. Repeated matches on an existing pending
        signal increment its weight (first detection=1.0, second=1.5 → promotes to belief).
        """
        if not self.self_model:
            return

        msg = user_message.lower()

        _PREF_RE = re.compile(
            r"\bi (?:like|love|prefer|enjoy|really like|really enjoy|"
            r"hate|dislike|don't like|do not like|can't stand|cannot stand)\b"
        )
        _GOAL_RE = re.compile(
            r"\bi (?:want|need|would like|am trying|am working on|plan to|am going to)\b"
            r"|\bhelp me\b|\bcan you please\b|\bplease help\b"
        )
        _FEEDBACK_POS = ("perfect", "exactly right", "great job", "well done",
                         "that's right", "that is right", "spot on", "nice work")
        _FEEDBACK_NEG = ("that's wrong", "that is wrong", "not right", "not quite right",
                         "incorrect", "you missed", "that's incorrect")

        trigger_type: str | None = None
        sentiment = 0.0

        if _PREF_RE.search(msg):
            trigger_type = "preference"
            sentiment = -0.5 if any(w in msg for w in ("hate", "dislike", "don't like", "can't stand")) else 0.5
        elif _GOAL_RE.search(msg):
            trigger_type = "goal"
        elif any(p in msg for p in _FEEDBACK_POS):
            trigger_type = "feedback"
            sentiment = 0.8
        elif any(p in msg for p in _FEEDBACK_NEG):
            trigger_type = "feedback"
            sentiment = -0.8

        if not trigger_type:
            return

        try:
            pending = self.self_model.get_signals(status="pending")
            # Find an existing signal of the same type with similar text to reinforce
            fingerprint = user_message[:60].lower()
            existing = next(
                (s for s in pending
                 if s.trigger_type == trigger_type and s.raw_text[:60].lower() == fingerprint),
                None,
            )
            if existing:
                self.self_model._apply_update_node(existing.id, {
                    "weight": min(3.0, existing.weight + 0.5),
                })
                logger.debug("Reinforced signal %s (weight→%.1f)", existing.id, existing.weight + 0.5)
            else:
                sig_id = f"sig-{uuid.uuid4().hex[:8]}"
                self.self_model._apply_add_node("signal", {
                    "id": sig_id,
                    "raw_text": user_message[:200],
                    "trigger_type": trigger_type,
                    "source": "user",
                    "status": "pending",
                    "weight": 1.0,
                    "sentiment": sentiment,
                    "keywords": [],
                    "entities": [],
                })
                self.self_model.auto_wire_node(sig_id)
                if trigger_type == "goal" and self.monologue_log is not None:
                    self.monologue_log.append(
                        type="user_task",
                        source="user",
                        title=f"User goal signal: {user_message[:60]}",
                        content=user_message[:200],
                        metadata={
                            "signal_node_id": sig_id,
                            "trigger_type": trigger_type,
                        },
                    )
        except Exception as exc:
            logger.debug("Signal detection failed: %s", exc)

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
            prefs = sorted(
                self.self_model.get_all_nodes("preference"),
                key=lambda p: p.get("strength", 0.5), reverse=True
            )[:5]
            pref_strs = [p.get("description", "") for p in prefs if p.get("description")]
            if pref_strs:
                lines.append("**Preferences:** " + " | ".join(pref_strs))
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("Self-model context build failed: %s", exc)
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
            self.self_model.auto_wire_node(goal_id)
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

    def _promote_signals(self) -> None:
        """Promote pending SignalNodes that meet the confidence threshold to BeliefNodes.

        A signal becomes a belief when:
        - status == "pending" (not yet promoted)
        - weight >= 1.5 (recurring or strongly weighted)
        - trigger_type in ("preference", "feedback", "goal") — user-directed signals

        The original signal is marked "committed" so it isn't promoted twice.
        """
        if not self.self_model:
            return
        from datetime import datetime, timezone
        import uuid as _uuid

        PROMOTE_WEIGHT_THRESHOLD = 1.5
        PROMOTABLE_TYPES = {"preference", "feedback", "goal"}

        try:
            pending = self.self_model.get_signals(status="pending")
        except Exception:
            return

        for signal in pending:
            if signal.weight < PROMOTE_WEIGHT_THRESHOLD:
                continue
            if signal.trigger_type not in PROMOTABLE_TYPES:
                continue

            belief_id = f"b-sig-{_uuid.uuid4().hex[:8]}"
            description = signal.raw_text[:200] or f"Signal: {signal.trigger_type}"
            confidence = min(0.9, 0.5 + signal.weight * 0.1 + abs(signal.sentiment) * 0.2)

            try:
                self.self_model._apply_add_node("belief", {
                    "id": belief_id,
                    "description": description,
                    "confidence": round(confidence, 3),
                    "evidence_count": 1,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                self.self_model.auto_wire_node(belief_id)
                # Mark original signal as committed
                self.self_model._apply_update_node(signal.id, {
                    "status": "committed",
                    "produced_nodes": [belief_id],
                    "exploration_notes": f"Promoted to belief {belief_id}",
                })
                logger.debug(
                    "Promoted signal %s → belief %s (confidence=%.2f)",
                    signal.id, belief_id, confidence,
                )
                self._detect_belief_conflicts(belief_id, description)
            except Exception as exc:
                logger.debug("Signal promotion failed for %s: %s", signal.id, exc)

    def _detect_belief_conflicts(self, new_belief_id: str, new_belief_text: str) -> None:
        """Check a new belief for semantic conflicts with existing beliefs.

        Conflict = high term overlap AND opposite negation polarity. On detection,
        creates (or finds) an UncertaintyNode for the conflicted domain and links
        both beliefs to it via RELATES_TO edges. More conflict evidence on an existing
        uncertainty node bumps its entropy_score by 0.1.
        """
        if not self.self_model:
            return

        _STOPWORDS = {"a", "an", "the", "is", "are", "i", "my", "it", "in", "of", "to", "and", "or"}
        _NEGATIONS = {"not", "never", "cannot", "can't", "won't", "no", "without", "fail", "fails", "failed"}

        def _words(text: str) -> set[str]:
            return set(text.lower().split()) - _STOPWORDS

        new_words = _words(new_belief_text)
        new_negated = bool(new_words & _NEGATIONS)

        try:
            existing_beliefs = self.self_model.get_beliefs()
        except Exception:
            return

        from datetime import datetime, timezone

        for existing in existing_beliefs:
            if existing.id == new_belief_id:
                continue
            old_words = _words(existing.description)
            old_negated = bool(old_words & _NEGATIONS)

            shared = new_words & old_words
            overlap = len(shared) / max(len(new_words | old_words), 1)

            if overlap > 0.3 and (new_negated != old_negated):
                domain = " ".join(sorted(shared - _NEGATIONS - _STOPWORDS)[:4])
                if not domain:
                    domain = new_belief_text[:40]

                try:
                    existing_u = next(
                        (u for u in self.self_model.get_uncertainty_map()
                         if any(w in u.domain.lower() for w in domain.split() if len(w) > 3)),
                        None,
                    )
                    if existing_u is None:
                        uid = f"u-conflict-{uuid.uuid4().hex[:6]}"
                        self.self_model._apply_add_node("uncertainty", {
                            "id": uid,
                            "domain": f"Conflicting beliefs: {domain}",
                            "entropy_score": 0.8,
                        })
                        target_uid = uid
                    else:
                        target_uid = existing_u.id
                        new_entropy = min(1.0, round(existing_u.entropy_score + 0.1, 3))
                        self.self_model._apply_update_node(target_uid, {"entropy_score": new_entropy})

                    for bid in (new_belief_id, existing.id):
                        try:
                            self.self_model._apply_add_edge(bid, target_uid, EdgeType.RELATES_TO.value)
                        except Exception:
                            pass

                    logger.debug(
                        "Belief conflict: %s ↔ %s → uncertainty %s",
                        new_belief_id, existing.id, target_uid,
                    )
                except Exception as exc:
                    logger.debug("Conflict detection failed: %s", exc)

    def _crystallize_strategy(self, plan: "ActionPlan", outcome: "ActualOutcome") -> dict | None:
        """Crystallize a successful plan execution into a StrategyNode.

        Derives a stable strategy ID from the set of step types used. Creates the node on
        first success, then Bayesian-updates preference_weight on subsequent executions.
        """
        if not self.self_model or outcome.success < 0.7:
            return None

        step_types = sorted({s.step_type for s in plan.steps})
        if not step_types:
            return None

        type_key = "_".join(t.split(".")[-1] for t in step_types)
        strat_id = f"strat-{type_key}"[:48]

        description = (
            plan.predicted_outcome.get("summary")
            or f"Approach using: {', '.join(t.split('.')[-1] for t in step_types)}"
        )

        existing = self.self_model.get_node(strat_id)
        try:
            if existing is None:
                self.self_model._apply_add_node("strategy", {
                    "id": strat_id,
                    "description": str(description)[:200],
                    "applicable_contexts": step_types,
                    "success_history": [True],
                    "preference_weight": 0.6,
                })
                if plan.goal_id and self.self_model.get_node(plan.goal_id):
                    try:
                        from backend.core.self_model.schemas import EdgeType as _ET
                        self.self_model._apply_add_edge(strat_id, plan.goal_id, _ET.SUPPORTS.value)
                    except Exception:
                        pass
                self.self_model.auto_wire_node(strat_id)
            else:
                history = existing.get("success_history", [])
                new_history = (history + [outcome.success >= 0.7])[-20:]
                new_weight = bayesian_update(
                    existing.get("preference_weight", 0.5),
                    len(history),
                    1.0 if outcome.success >= 0.7 else 0.0,
                )
                self.self_model._apply_update_node(strat_id, {
                    "preference_weight": round(new_weight, 4),
                    "success_history": new_history,
                })
            logger.debug("Strategy crystallized: %s (success=%.2f)", strat_id, outcome.success)
            return {
                "op": "strategy",
                "node_type": "strategy",
                "id": strat_id,
                "description": str(description)[:80],
                "fields": ["preference_weight", "success_history"],
            }
        except Exception as exc:
            logger.debug("Strategy crystallization failed: %s", exc)
            return None

    def _run_graph_maintenance(self) -> None:
        """Periodic self-model hygiene: prune finished transient goals, discard stale
        signals, and — every 50 interactions — apply a confidence decay to beliefs that
        haven't been reinforced recently.
        """
        if not self.self_model:
            return

        from datetime import timezone as _tz, timedelta

        now = datetime.now(_tz.utc)
        stats: dict[str, int] = {"pruned_goals": 0, "discarded_signals": 0, "decayed_beliefs": 0}

        # Prune completed transient goals older than 7 days
        for goal in self.self_model.get_goals():
            if not goal.id.startswith("tg-"):
                continue
            if goal.status not in ("achieved", "abandoned"):
                continue
            try:
                age_days = (now - datetime.fromisoformat(
                    goal.created_at.replace("Z", "+00:00")
                )).days
            except Exception:
                age_days = 0
            if age_days < 7:
                continue
            try:
                self.self_model._apply_remove_node(goal.id)
                stats["pruned_goals"] += 1
            except Exception:
                pass

        # Discard weak signals that have sat pending for more than 3 days
        for signal in self.self_model.get_signals(status="pending"):
            if signal.weight >= 1.0:
                continue
            try:
                age_days = (now - datetime.fromisoformat(
                    signal.created_at.replace("Z", "+00:00")
                )).days
            except Exception:
                continue
            if age_days < 3:
                continue
            try:
                self.self_model._apply_update_node(signal.id, {"status": "discarded"})
                stats["discarded_signals"] += 1
            except Exception:
                pass

        # Heavy pass every 50 interactions: decay belief confidence (forgetting curve)
        if self._interaction_count % 50 == 0:
            for belief in self.self_model.get_beliefs():
                try:
                    last_updated = datetime.fromisoformat(
                        belief.last_updated.replace("Z", "+00:00")
                    )
                    weeks = max(0.0, (now - last_updated).days / 7.0)
                    if weeks < 1.0:
                        continue
                    # High-evidence beliefs decay more slowly
                    decay_per_week = 0.02 / max(1.0, belief.evidence_count / 5.0)
                    new_conf = max(0.1, belief.confidence * ((1.0 - decay_per_week) ** weeks))
                    if abs(new_conf - belief.confidence) < 0.005:
                        continue
                    self.self_model._apply_update_node(belief.id, {
                        "confidence": round(new_conf, 4),
                    })
                    stats["decayed_beliefs"] += 1
                except Exception:
                    pass

        if any(v > 0 for v in stats.values()):
            logger.info("Graph maintenance: %s", stats)

    def _check_capability_milestones(self) -> list[dict]:
        """Crystallize capability milestones into belief nodes.

        When a capability's success_rate >= 0.7 with test_count >= 5, write a BeliefNode
        recording the achievement. Each capability milestones once (idempotent by ID).
        Returns mutation dicts for surface in the chat response meta.
        """
        if not self.self_model:
            return []

        MILESTONE_SUCCESS_RATE = 0.7
        MILESTONE_TEST_COUNT = 5
        mutations: list[dict] = []

        from datetime import datetime, timezone

        for cap in self.self_model.query_capabilities():
            if cap.success_rate < MILESTONE_SUCCESS_RATE or cap.test_count < MILESTONE_TEST_COUNT:
                continue
            milestone_id = f"b-milestone-{cap.id}"
            if self.self_model.get_node(milestone_id) is not None:
                continue
            try:
                self.self_model._apply_add_node("belief", {
                    "id": milestone_id,
                    "description": (
                        f"I have demonstrated reliable {cap.description.lower()} "
                        f"({cap.success_rate:.0%} success across {cap.test_count} attempts)"
                    ),
                    "confidence": min(0.95, cap.success_rate),
                    "evidence_count": cap.test_count,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                self.self_model.auto_wire_node(milestone_id)
                mutations.append({
                    "op": "milestone",
                    "node_type": "belief",
                    "id": milestone_id,
                    "description": f"Capability milestone: {cap.description}",
                    "fields": ["confidence", "evidence_count"],
                })
                logger.info("Capability milestone crystallized: %s → %s", cap.id, milestone_id)
            except Exception as exc:
                logger.debug("Milestone crystallization failed for %s: %s", cap.id, exc)

        return mutations

    def _nudge_capability(self, skill_name: str, success: bool = True) -> dict | None:
        """Nudge the confidence of a capability node whose description matches the skill.

        Uses query_capabilities() + description substring match rather than hardcoded IDs,
        so new capabilities seeded into the self-model are automatically picked up.

        Returns a mutation dict if a nudge was applied, else None.
        """
        if not self.self_model:
            return None

        # Build search terms from the skill name (e.g. "web_search" → ["web", "search"])
        terms = [t for t in skill_name.lower().replace("_", " ").split() if len(t) > 2]
        if not terms:
            return None

        cap_node = None
        best_score = 0
        for cap in self.self_model.query_capabilities():
            desc = cap.description.lower()
            cap_name = getattr(cap, "capability_name", cap.id).lower()
            score = sum(1 for t in terms if t in desc or t in cap_name)
            if score > best_score:
                best_score = score
                cap_node = cap

        if cap_node is None or best_score == 0:
            # No existing capability node — discover it from first use
            if not success:
                return None  # Don't record a capability that immediately failed
            cap_id = f"cap-{skill_name.replace('_', '-')}"
            description = " ".join(terms).title()
            try:
                from datetime import timezone as _tz
                self.self_model._apply_add_node("capability", {
                    "id": cap_id,
                    "description": description,
                    "confidence": 0.4,
                    "test_count": 1,
                    "success_rate": 1.0,
                    "last_tested": datetime.now(_tz.utc).isoformat(),
                    "created_at": datetime.now(_tz.utc).isoformat(),
                })
                self.self_model.auto_wire_node(cap_id)
                logger.info("Discovered new capability: %s (%s)", cap_id, description)
                return {
                    "op": "discover",
                    "node_type": "capability",
                    "id": cap_id,
                    "description": description,
                }
            except Exception as exc:
                logger.debug("Could not discover capability for %s: %s", skill_name, exc)
                return None

        try:
            confidence = cap_node.confidence
            test_count = cap_node.test_count
            observation = 1.0 if success else 0.0
            new_confidence = bayesian_update(confidence, test_count, observation)
            new_success_rate = (
                (cap_node.success_rate * test_count + observation) / (test_count + 1)
                if test_count > 0 else observation
            )
            self.self_model._apply_update_node(cap_node.id, {
                "confidence": round(new_confidence, 4),
                "test_count": test_count + 1,
                "success_rate": round(new_success_rate, 4),
                "last_tested": datetime.now().isoformat(),
            })
            logger.debug(
                "Nudged capability %s: %.3f → %.3f (skill=%s, score=%d)",
                cap_node.id, confidence, new_confidence, skill_name, best_score,
            )
            return {
                "op": "nudge",
                "node_type": "capability",
                "id": cap_node.id,
                "description": cap_node.description[:80],
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
