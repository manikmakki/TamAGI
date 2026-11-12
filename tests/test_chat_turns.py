"""Exercise twenty-call user turns and five agent-requested continuations."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from backend.config import RuntimeConfig, TamAGIConfig, get_config, set_config
from backend.core.llm import LLMResponse, LLMMessage, ToolCall
from backend.core.runtime import RuntimeStore, RuntimeService, GovernedLLM, set_runtime
from backend.core.runtime.chat_turns import run_chat_turns, previous_work
from backend.skills.base import Skill, SkillResult
from backend.skills.registry import SkillRegistry
from backend.skills.runtime_skill import install_runtime_skills
from backend.core.runtime.companions import Companions


def call(name="probe", **arguments):
    return LLMResponse(tool_calls=[ToolCall("temporary", name, arguments)])


class Provider:
    config = SimpleNamespace(model="fixture", max_tokens=128)

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    async def chat(self, messages, **kwargs):
        self.prompts.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        for index, tool in enumerate(response.tool_calls):
            tool.id = f"call-{len(self.prompts)}-{index}"
        return response


class Probe(Skill):
    name = "probe"
    description = "A concrete write"
    parameters = {}

    def __init__(self, path):
        self.path = path
        self.calls = 0
        self.block = None

    async def execute(self, **kwargs):
        self.calls += 1
        self.path.write_text(str(self.calls))
        if self.block:
            self.block.set()
            await asyncio.Event().wait()
        return SkillResult(True, output=f"effect {self.calls}")


class ChatTurnTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.previous_config = get_config()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = RuntimeStore(self.root / "runtime.sqlite3")
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.root
        )
        self.config = TamAGIConfig()
        self.config.workspace.path = str(self.runtime.workspace)
        set_config(self.config)
        set_runtime(self.runtime)
        self.registry = SkillRegistry()
        self.probe = Probe(self.runtime.workspace / "effect.txt")
        self.registry.register(self.probe)
        companions = Companions(self.runtime, None, self.registry)
        install_runtime_skills(self.runtime, self.registry, companions)

    def tearDown(self):
        set_runtime(None)
        set_config(self.previous_config)
        self.store.close()
        self.temporary.cleanup()

    async def run_chat(self, responses, conversation="thread", event_callback=None):
        provider = Provider(responses)
        agent = SimpleNamespace(
            runtime=self.runtime,
            llm=GovernedLLM(provider, self.runtime),
            skills=self.registry,
            config=self.config,
            _pending_approvals={},
        )
        epoch = self.runtime.notice_user_input()
        async with self.runtime.execution("user") as context:
            self.store.set_run_provenance(
                context.execution_id, conversation_id=conversation
            )
            result = await run_chat_turns(
                agent,
                [LLMMessage("user", "Please investigate")],
                input_epoch=epoch,
                event_callback=event_callback,
            )
        return result, provider, self.store.run(context.execution_id)

    async def test_more_than_five_calls_without_another_user_prompt(self):
        result, provider, run = await self.run_chat(
            [call() for _ in range(9)]
            + [LLMResponse(content="Finished with evidence.")]
        )
        self.assertEqual(result.text, "Finished with evidence.")
        self.assertEqual(run["calls"], 10)
        self.assertEqual(self.probe.calls, 9)
        self.assertEqual(run["status"], "completed")

    async def test_optional_quiet_work_finishes_before_returning_reply(self):
        events = []

        async def event(value):
            events.append(value)

        result, provider, run = await self.run_chat(
            [
                call("request_quiet_turn"),
                LLMResponse(content="Let me do that."),
                call(),
                LLMResponse(content="I wrote the result."),
            ],
            event_callback=event,
        )
        self.assertEqual(result.text, "I wrote the result.")
        self.assertEqual(result.interim, ["Let me do that."])
        self.assertEqual(run["calls"], 4)
        self.assertEqual(json.loads(run["payload"])["chat_turns"]["quiet_used"], 1)
        self.assertEqual(sum(event["type"] == "quiet_turn" for event in events), 1)
        self.assertEqual(
            sum(message.role == "user" for message in provider.prompts[-1]), 1
        )
        self.assertTrue(
            any(
                "Starting agent-requested quiet" in str(message.content)
                for message in provider.prompts[-1]
            )
        )

    async def test_twenty_calls_per_turn_and_five_quiet_turns_allow_at_most_120(self):
        script = []
        for _ in range(5):
            script.extend([call() for _ in range(18)])
            script.extend([call("request_quiet_turn"), call("end_turn")])
        script.extend([call() for _ in range(19)])
        script.append(LLMResponse(content="Done."))
        result, provider, run = await self.run_chat(script)
        self.assertEqual(result.text, "Done.", run["payload"])
        self.assertEqual(run["calls"], 120)
        self.assertEqual(len(provider.prompts), 120)
        turns = self.store.list("chat_turn", limit=100)
        self.assertEqual(len(turns), 6)
        self.assertTrue(all(turn["data"]["calls"] == 20 for turn in turns))

    async def test_duplicate_requests_coalesce_and_sixth_is_rejected(self):
        requests = [
            ToolCall("a", "request_quiet_turn", {}),
            ToolCall("b", "request_quiet_turn", {}),
        ]
        script = [LLMResponse(tool_calls=requests), call("end_turn")]
        for _ in range(4):
            script.extend([call("request_quiet_turn"), call("end_turn")])
        script.extend(
            [call("request_quiet_turn"), LLMResponse(content="Stopping here.")]
        )
        result, _, run = await self.run_chat(script)
        controls = json.loads(run["payload"])["chat_turns"]
        self.assertEqual(controls["quiet_used"], 5)
        self.assertFalse(controls["queued"])
        self.assertEqual(result.text, "Stopping here.")
        rejected = [
            r
            for r in self.store.receipts(run["id"], limit=100)
            if r["tool"] == "request_quiet_turn" and r["status"] == "failed"
        ]
        self.assertEqual(len(rejected), 1)

    async def test_exhaustion_does_not_promote_last_commentary_to_success(self):
        script = [call() for _ in range(20)]
        script[-1].content = "Let me finish that next."
        result, provider, run = await self.run_chat(script)
        self.assertEqual(result.text, "")
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(run["status"], "budget_exhausted")
        self.assertEqual(len(provider.prompts), 20)
        self.assertEqual(self.probe.path.read_text(), "20")

    async def test_failed_or_exhausted_turn_discards_queued_work(self):
        for failure in [RuntimeError("provider truncated"), None]:
            script = [call("request_quiet_turn")]
            script += [failure] if failure else [call() for _ in range(19)]
            result, _, run = await self.run_chat(script)
            controls = json.loads(run["payload"])["chat_turns"]
            self.assertEqual(controls["quiet_used"], 0)
            self.assertFalse(controls["queued"])
            self.assertIn(result.status, {"failed", "budget_exhausted"})

    async def test_rest_clears_queue_without_extra_inference(self):
        result, provider, run = await self.run_chat(
            [call("request_quiet_turn"), call("end_turn", status="resting")]
        )
        self.assertEqual(result.status, "resting")
        self.assertEqual(len(provider.prompts), 2)
        self.assertEqual(json.loads(run["payload"])["chat_turns"]["quiet_used"], 0)

    async def test_no_request_means_no_automatic_continuation(self):
        result, provider, _ = await self.run_chat(
            [LLMResponse(content="A normal conversation reply.")]
        )
        self.assertEqual(len(provider.prompts), 1)
        self.assertEqual(result.text, "A normal conversation reply.")

    async def test_actual_input_interrupts_quiet_tool_and_preserves_unknown_effect(
        self,
    ):
        entered = asyncio.Event()
        self.probe.block = entered
        task = asyncio.create_task(
            self.run_chat([call("request_quiet_turn"), call("end_turn"), call()])
        )
        await asyncio.wait_for(entered.wait(), 3)
        self.runtime.notice_user_input()
        result, _, run = await asyncio.wait_for(task, 3)
        self.assertEqual(result.status, "interrupted")
        self.assertEqual(run["status"], "interrupted")
        self.assertTrue(self.probe.path.is_file())
        self.assertEqual(self.store.receipts(run["id"])[0]["status"], "outcome_unknown")

    async def test_pending_input_prevents_queued_quiet_turn_from_starting(self):
        async def event(value):
            if value["type"] == "tool_result" and value["name"] == "request_quiet_turn":
                self.runtime.notice_user_input()

        result, provider, run = await self.run_chat(
            [
                call("request_quiet_turn"),
                LLMResponse(content="Yielding to the next message."),
            ],
            event_callback=event,
        )
        self.assertEqual(len(provider.prompts), 2)
        self.assertEqual(json.loads(run["payload"])["chat_turns"]["quiet_used"], 0)

    async def test_child_dispatch_cannot_request_quiet_turn(self):
        async with self.runtime.execution("autonomy"):
            denied = await self.registry.execute("request_quiet_turn")
            self.assertFalse(denied.success)
            names = [
                tool["function"]["name"] for tool in self.registry.get_openai_tools()
            ]
            self.assertNotIn("request_quiet_turn", names)

    async def test_later_user_turn_can_recover_receipt_context(self):
        result, _, run = await self.run_chat(
            [call(), LLMResponse(content="I made a note.")]
        )
        async with self.runtime.execution("user"):
            working = previous_work(self.runtime, "thread")
            self.assertIn(self.store.receipts(run["id"])[0]["id"], working.content)
            self.assertIn("effect 1", working.content)
            self.assertEqual(working.role, "system")

    async def test_restart_discards_queue_without_resetting_recorded_call_counts(self):
        from backend.core.runtime.contracts import ExecutionContext, current_execution
        from backend.core.runtime.chat_turns import ChatTurns

        context = ExecutionContext(
            "crashed-chat", self.runtime.actor_id, "user", root_id="crashed-chat"
        )
        self.store.start_run(context, 20)
        token = current_execution.set(context)
        try:
            control = ChatTurns(self.runtime, "thread", self.runtime.input_epoch)
            self.store.charge(context)
            control.request()
        finally:
            current_execution.reset(token)
        self.store.recover()
        run = self.store.run(context.execution_id)
        self.assertEqual(run["status"], "interrupted")
        self.assertEqual(run["calls"], 1)
        self.assertFalse(json.loads(run["payload"])["chat_turns"]["queued"])
        restored = RuntimeService(
            self.store, self.runtime.workspace, RuntimeConfig(), self.root
        )
        self.assertEqual(restored.chat_controls, {})
