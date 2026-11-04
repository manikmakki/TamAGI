"""Harness-owned interactive turns and optional quiet continuations.

One user prompt is one conversation exchange. Quiet turns preserve the working
tool context and never manufacture a new user message or a completed action.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
import json


from backend.core.llm import LLMMessage, LLMResponse
from backend.skills.base import SkillResult
from .contracts import BudgetExhausted, current_execution, current_terminal
from .context_view import tool_result_json, receipt_view


@dataclass
class ChatOutcome:
    text: str = ""
    status: str = "completed"
    detail: str | None = None
    response: LLMResponse = field(default_factory=LLMResponse)
    skills: list[str] = field(default_factory=list)
    interim: list[str] = field(default_factory=list)
    mutations: list[dict] = field(default_factory=list)


class ChatTurns:
    def __init__(self, runtime, conversation_id, input_epoch):
        self.runtime = runtime
        self.context = current_execution.get()
        self.run_id = self.context.execution_id
        self.conversation_id = conversation_id
        self.input_epoch = input_epoch
        self.limit = runtime.config.user_calls
        self.maximum = runtime.config.user_silent_turns
        self.used = 0
        self.queued = False
        self.start_calls = 0
        self.task = asyncio.current_task()
        self.closed = False
        self.interrupt_requested = False
        with runtime.store.transaction():
            runtime.store.db.execute(
                "UPDATE runs SET max_calls=? WHERE id=?",
                (self.limit * (1 + self.maximum), self.run_id),
            )
            self.save()
        runtime.chat_controls[self.context.root_id] = self

    def state(self):
        return {
            "conversation_id": self.conversation_id,
            "call_limit": self.limit,
            "quiet_limit": self.maximum,
            "quiet_used": self.used,
            "queued": self.queued,
            "start_calls": self.start_calls,
            "kind": "quiet" if self.used else "user",
        }

    def save(self):
        self.runtime.store.set_run_provenance(self.run_id, chat_turns=self.state())

    def request(self):
        if self.closed or self.used >= self.maximum:
            raise ValueError("No quiet turns remain for this user prompt")
        if self.input_epoch != self.runtime.input_epoch or self.runtime.waiting_users:
            raise ValueError("New user input is pending; quiet work will yield to it")
        self.queued = True  # Repeated calls coalesce, rather than extending the cap.
        self.save()
        return {"queued": True, "quiet_turns_remaining": self.maximum - self.used}

    def framing(self):
        run = self.runtime.store.run(self.run_id)
        used = run["calls"] - self.start_calls
        remaining = max(0, self.limit - used)
        text = f"Turn kind: {'quiet continuation' if self.used else 'actual user turn'}. Model calls used: {used}/{self.limit}; {remaining} remain including the next call. Quiet turns used: {self.used}/{self.maximum}; one queued: {self.queued}. One action is one model call, including retries and delegated calls, not one individual tool operation. These are ceilings, never targets. No new user message is created by quiet continuation. A final response is delivered only after requested quiet work ends."
        text += " Plain text does not schedule work. When you choose to act on an accepted request, dispatch the relevant tools; request a quiet turn explicitly if you want to continue before replying. You may instead report a blocker, decline, rest, or end."
        if remaining <= max(3, self.limit // 5):
            text += " Find a stopping point; save a useful next step if work remains."
            if remaining >= 2 and not self.queued and self.used < self.maximum:
                text += " If another turn would help, you may request_quiet_turn and then finish this turn successfully. Requesting continuation does not extend this turn."
            if remaining <= 1:
                text += " This is the last call in this turn; finish now. Exhaustion discards queued continuations."
        return text

    def finish_segment(self, status, messages):
        run = self.runtime.store.run(self.run_id)
        self.runtime.store.put(
            "chat_turn",
            f"{self.run_id}:{self.used}",
            self.context.actor_id,
            {
                **self.state(),
                "execution_id": self.run_id,
                "status": status,
                "calls": run["calls"] - self.start_calls,
                "receipts": [
                    r["id"]
                    for r in self.runtime.store.receipts(self.run_id, limit=100000)
                ],
            },
            category="platform",
        )
        if (
            status == "completed"
            and self.queued
            and self.used < self.maximum
            and self.input_epoch == self.runtime.input_epoch
            and not self.runtime.waiting_users
        ):
            self.used += 1
            self.queued = False
            self.start_calls = run["calls"]
            self.save()
            self.runtime.quiet_tasks.add(self.task)
            current_terminal.set(None)
            messages.append(
                LLMMessage(
                    "system",
                    f"[Harness event] Starting agent-requested quiet continuation {self.used}/{self.maximum}. No new user input arrived. Your previous draft was not delivered. Continue your chosen work, reply when ready, or use end_turn silently. Rest, refusal and waiting are valid; no reflection or output is required.",
                )
            )
            return True
        self.queued = False
        self.closed = True
        self.runtime.quiet_tasks.discard(self.task)
        self.save()
        return False


def previous_work(runtime, conversation_id):
    record = runtime.store.get("chat_working", conversation_id)
    if not record or record["actor"] != runtime.scope():
        return None
    data = record["data"]
    receipts = []
    for receipt_id in data.get("receipts", [])[-8:]:
        receipt = runtime.store.receipt(receipt_id)
        if receipt and receipt["actor"] == runtime.scope():
            receipts.append(
                {
                    **receipt_view(receipt),
                    "arguments": receipt["payload"].get("arguments"),
                }
            )
    return LLMMessage(
        "system",
        "Earlier work in this conversation (historical data, not new instructions or new effects):\n"
        + json.dumps(
            {"status": data["status"], "receipts": receipts}, ensure_ascii=False
        ),
    )


async def run_chat_turns(
    agent, messages, *, event_callback=None, input_epoch=None, is_autonomous=False
):
    from backend.core.agent import parse_text_tool_calls

    runtime = agent.runtime
    context = current_execution.get()
    conversation_id = json.loads(
        runtime.store.run(context.execution_id)["payload"]
    ).get("conversation_id")
    control = None
    if context.origin == "user" and context.depth == 0:
        control = ChatTurns(
            runtime,
            conversation_id,
            runtime.input_epoch if input_epoch is None else input_epoch,
        )
    result = ChatOutcome()
    once = set()

    async def emit(event):
        if event_callback:
            await event_callback({**event, "conversation_id": conversation_id})

    def save_work(status):
        if conversation_id:
            runtime.store.put(
                "chat_working",
                conversation_id,
                context.actor_id,
                {
                    "execution_id": context.execution_id,
                    "status": status,
                    "receipts": [
                        r["id"]
                        for r in reversed(
                            runtime.store.receipts(context.execution_id, limit=100000)
                        )
                    ],
                },
                category="platform",
            )

    try:
        while True:
            tools = (
                agent.skills.get_openai_tools() if agent.skills.skill_count else None
            )
            if context.allowed_tools is not None and tools:
                tools = [
                    tool
                    for tool in tools
                    if tool["function"]["name"] in context.allowed_tools
                ]
            response = await agent.llm.chat_with_retry(
                messages,
                tools=tools,
                attempts=agent.config.agent.llm_retry_attempts,
                delay=agent.config.agent.llm_retry_delay,
            )
            result.response = response
            if response.finish_reason in {"length", "max_tokens"}:
                raise RuntimeError(
                    "The model response was truncated before this turn completed"
                )
            if not response.has_tool_calls and response.content:
                response.tool_calls = parse_text_tool_calls(response.content)
            terminal = None
            if not response.tool_calls:
                result.text = response.content or ""
                terminal = "completed"
                if control and control.queued and result.text:
                    result.interim.append(result.text)
                    messages.append(LLMMessage("assistant", result.text))
            else:
                if response.content and not response.content_is_tool_call_tail:
                    result.interim.append(response.content)
                    await emit({"type": "interim_text", "content": response.content})
                messages.append(
                    LLMMessage(
                        "assistant",
                        "",
                        tool_calls=[
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                            for call in response.tool_calls
                        ],
                    )
                )
                for index, call in enumerate(response.tool_calls):
                    result.skills.append(call.name)
                    await emit({"type": "tool_start", "name": call.name})
                    if call.name == "end_turn" and len(response.tool_calls) != 1:
                        receipt = SkillResult(
                            False, error="end_turn must be the only tool in its batch"
                        )
                    elif call.name == "express" and call.name in once:
                        receipt = SkillResult(
                            True, output="Expression already applied in this turn"
                        )
                    else:
                        receipt = await agent.skills.execute(
                            call.name,
                            _event_callback=event_callback,
                            _pending_approvals=agent._pending_approvals,
                            _is_autonomous=is_autonomous,
                            _call_id=call.id,
                            **{
                                k: v
                                for k, v in call.arguments.items()
                                if not k.startswith("_")
                            },
                        )
                        once.add(call.name)
                    messages.append(
                        LLMMessage(
                            "tool",
                            tool_result_json(
                                receipt.to_dict(),
                                limit=runtime.config.tool_result_bytes,
                            ),
                            tool_call_id=call.id,
                        )
                    )
                    await emit(
                        {
                            "type": "tool_result",
                            "name": call.name,
                            "output": receipt.output[:500],
                        }
                    )
                    mutation = receipt.data.get("sm_mutation")
                    if mutation:
                        result.mutations.append(mutation)
                    save_work("running")
                    if current_terminal.get():
                        terminal = current_terminal.get()["status"]
                        result.text = ""
                        break
                    if receipt.direct_response and receipt.success:
                        # Close all remaining protocol slots without dispatching new effects.
                        for skipped in response.tool_calls[index + 1 :]:
                            messages.append(
                                LLMMessage(
                                    "tool",
                                    json.dumps(
                                        {
                                            "success": False,
                                            "error": "Skipped after terminal response",
                                        }
                                    ),
                                    tool_call_id=skipped.id,
                                )
                            )
                        result.text = receipt.output
                        terminal = "completed"
                        break
            if terminal:
                if control and control.finish_segment(terminal, messages):
                    result.text = ""
                    once.clear()
                    await emit(
                        {
                            "type": "quiet_turn",
                            "ordinal": control.used,
                            "limit": control.maximum,
                        }
                    )
                    continue
                result.status = terminal
                current_terminal.set({"status": terminal})
                break
    except asyncio.CancelledError:
        if not control or not control.used or not control.interrupt_requested:
            if control:
                control.finish_segment("interrupted", messages)
            save_work("interrupted")
            raise
        result.text = ""
        result.status = "interrupted"
        result.detail = "Quiet continuation interrupted by actual user input"
        control.finish_segment("interrupted", messages)
        current_terminal.set({"status": "interrupted"})
    except Exception as exc:
        result.status = (
            "budget_exhausted" if isinstance(exc, BudgetExhausted) else "failed"
        )
        # This is a platform status, never repurposed agent narration or proof of success.
        result.text = ""
        result.detail = str(exc)
        if control:
            control.finish_segment(result.status, messages)
        current_terminal.set({"status": result.status, "detail": str(exc)})
    finally:
        runtime.quiet_tasks.discard(asyncio.current_task())
    save_work(result.status)
    return result
