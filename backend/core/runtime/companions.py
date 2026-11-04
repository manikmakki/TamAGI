"""Durable individual context, bounded activations, explicit shared conversation."""

from __future__ import annotations
import asyncio
import json
from .contracts import current_execution, current_terminal
from .store import uid
from backend.core.llm import LLMMessage

FORBIDDEN = {"orchestrate_task", "consult_companion"}


class Companions:
    def __init__(self, runtime, llm, skills):
        self.runtime = runtime
        self.llm = llm
        self.skills = skills

    def create(self, name, seed, interests=None):
        context = current_execution.get()
        if context and context.depth:
            raise ValueError("Only the primary resident may create companions")
        if not name.strip() or not seed.strip():
            raise ValueError("A companion needs a name and founding seed")
        actor = self.runtime.scope()
        return self.runtime.store.put(
            "actor",
            uid("companion"),
            actor,
            {
                "name": name,
                "seed": seed,
                "self_description": seed,
                "interests": interests or [],
                "primary": False,
                "status": "active",
                "created_by": actor,
            },
        )

    def list(self):
        return [
            r for r in self.runtime.store.list("actor") if not r["data"].get("primary")
        ]

    def revise(self, actor_id, changes, expected=None):
        old = self.runtime.store.get("actor", actor_id)
        if not old or old["data"].get("primary"):
            raise ValueError("Unknown companion")
        actor = self.runtime.scope()
        # Founding seed and creator are historical. A parent can propose a change in conversation.
        allowed = (
            {"status"}
            if actor != actor_id
            else {"self_description", "interests", "name", "status"}
        )
        if any(k not in allowed for k in changes):
            raise ValueError("Only a companion may revise its current self-description")
        if changes.get("status", old["data"]["status"]) not in {"active", "archived"}:
            raise ValueError("Invalid lifecycle status")
        return self.runtime.store.put(
            "actor",
            actor_id,
            old["actor"],
            {**old["data"], **changes},
            expected=expected,
        )

    async def consult(
        self,
        actor_id,
        content,
        mode="conversation",
        thread_id=None,
        allowed_tools=None,
        *,
        _caller_actor_id=None,
    ):
        from backend.core.tool_loop import run_tool_loop

        if mode not in {"conversation", "task"}:
            raise ValueError("Choose conversation or task")
        context = current_execution.get()
        if context is None:
            async with self.runtime.execution("user"):
                return await self.consult(
                    actor_id,
                    content,
                    mode,
                    thread_id,
                    allowed_tools,
                    _caller_actor_id=_caller_actor_id,
                )
        caller_actor_id = _caller_actor_id or context.actor_id
        if context.depth:
            raise ValueError("Companions cannot activate descendants")
        actor = self.runtime.store.get("actor", actor_id)
        if (
            not actor
            or actor["data"].get("primary")
            or actor["data"]["status"] != "active"
        ):
            raise ValueError("Companion is unavailable")
        lock = self.runtime.companion_locks.setdefault(actor_id, asyncio.Lock())
        async with lock:
            thread = (
                self.runtime.store.get("companion_thread", thread_id)
                if thread_id
                else None
            )
            participants = [caller_actor_id, actor_id]
            visible = thread and (
                thread["data"]["participants"] == participants
                or caller_actor_id == "operator"
                and thread["data"]["participants"] == [self.runtime.actor_id, actor_id]
            )
            if thread_id and not visible:
                raise ValueError("Conversation is not visible to these participants")
            if thread:
                participants = thread["data"]["participants"]
            thread_id = thread_id or uid("thread")
            history = thread["data"]["messages"] if thread else []
            started_at = self.runtime.store.clock()
            request_message = {
                "id": uid("message"),
                "role": "user",
                "actor": caller_actor_id,
                "content": content,
                "timestamp": started_at,
            }
            parent_run = self.runtime.store.run(context.execution_id)
            parent_provenance = json.loads(parent_run["payload"]) if parent_run else {}
            if (
                mode == "conversation"
                and caller_actor_id == self.runtime.actor_id
                and not parent_provenance.get("occasion_id")
            ):
                self.runtime.growth.bind_conversation(
                    "companion:" + thread_id, history + [request_message]
                )
            tools = set(
                allowed_tools
                if allowed_tools is not None
                else (
                    ["remember", "share_note", "companion", "revise_identity"]
                    if mode == "conversation"
                    else [
                        "read",
                        "write",
                        "web_search",
                        "world_action",
                        "pursuit",
                        "remember",
                        "share_note",
                        "register_artifact",
                        "inspect_receipt",
                        "end_turn",
                    ]
                )
            )
            tools -= FORBIDDEN
            tools.add("end_turn")
            async with self.runtime.execution(
                "companion",
                actor_id=actor_id,
                allowed_tools=tools,
                max_calls=getattr(self.runtime.config, "companion_calls", 3),
            ) as invocation:
                seed = actor["data"]
                messages = [
                    LLMMessage(
                        "system",
                        f"You are {seed['name']}. Founding disposition (historical): {seed['seed']}\nCurrent self-description: {seed.get('self_description', seed['seed'])}\n"
                        + self.runtime.context(actor_id)
                        + f"\nThis is a {mode} invitation. You may disagree, decline, renegotiate, or ask for help. Never adopt another actor’s memories as your own.",
                    )
                ]
                messages += [LLMMessage(m["role"], m["content"]) for m in history[-12:]]
                messages.append(LLMMessage("user", content))
                text, used = await run_tool_loop(
                    self.llm,
                    self.skills,
                    messages,
                    max_rounds=getattr(self.runtime.config, "companion_calls", 3),
                    is_autonomous=True,
                )
                terminal = current_terminal.get()
                receipts = self.runtime.store.receipts(invocation.execution_id)
                status = (
                    terminal["status"]
                    if terminal
                    else ("completed" if text else "budget_exhausted")
                )
                completed_at = self.runtime.store.clock()
                run = self.runtime.store.run(invocation.execution_id)
                provenance = json.loads(run["payload"]) if run else {}
                result = {
                    "mode": mode,
                    "caller_actor_id": caller_actor_id,
                    "content": content,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "finished_at": completed_at,
                    "conversation_id": provenance.get("conversation_id"),
                    "occasion_id": provenance.get("occasion_id"),
                    "root_id": invocation.root_id,
                    "growth_eligible": bool(
                        provenance.get("eligible_live")
                        and not provenance.get("evaluation")
                        and caller_actor_id == self.runtime.actor_id
                    ),
                    "growth_completed": bool(status == "completed" and text.strip()),
                    "actor_id": actor_id,
                    "speaker": seed["name"],
                    "thread_id": thread_id,
                    "invocation_id": invocation.execution_id,
                    "status": status,
                    "text": text,
                    "receipts": [r["id"] for r in receipts],
                    "skills_used": used,
                }
                self.runtime.store.put(
                    "companion_thread",
                    thread_id,
                    caller_actor_id,
                    {
                        "participants": participants,
                        "messages": history
                        + [
                            request_message,
                            {
                                "id": uid("message"),
                                "role": "assistant",
                                "actor": actor_id,
                                "content": text,
                                "timestamp": completed_at,
                                "metadata": {
                                    "growth_eligible": result["growth_eligible"],
                                    "growth_completed": result["growth_completed"],
                                    "occasion_id": result["occasion_id"],
                                },
                            },
                        ],
                        "last_result": result,
                    },
                )
                self.runtime.store.put(
                    "invocation",
                    invocation.execution_id,
                    actor_id,
                    result,
                    category="platform",
                )
                self.runtime.growth.observe_companion(
                    thread_id, result, caller_actor_id
                )
                return result
