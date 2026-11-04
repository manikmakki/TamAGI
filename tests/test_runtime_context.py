"""Large retained evidence must not poison every future autonomous opportunity."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import httpx

from backend.config import RuntimeConfig, WorldThreadConfig
from backend.core.llm import LLMMessage, LLMResponse, ToolCall
from backend.core.runtime import RuntimeService, RuntimeStore, GovernedLLM, set_runtime
from backend.core.runtime.context_view import bound_messages, ContextViewTooLarge
from backend.core.world_thread import WorldThread
from backend.core.tool_loop import run_tool_loop
from backend.skills.base import Skill, SkillResult
from backend.skills.registry import SkillRegistry

BIG_OUTPUT = (
    "BEGIN full diagnostic\n"
    + ("binary-like \\u0000 data\x01" * 24000)
    + "\nEND full diagnostic"
)


class LargeFailure(Skill):
    name = "large_failure"
    description = "Return a large diagnostic failure"
    parameters = {}

    async def execute(self, **kwargs):
        return SkillResult(False, output=BIG_OUTPUT, error="The operation failed")


class Provider:
    config = SimpleNamespace(model="fixture", temperature=0, max_tokens=64)

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def chat(self, messages, **kwargs):
        self.prompts.append(messages)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class ContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.now = [100000.0]
        self.store = RuntimeStore(
            self.root / "runtime.sqlite3", clock=lambda: self.now[0]
        )
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.root
        )
        self.registry = SkillRegistry()
        self.registry.register(LargeFailure())
        set_runtime(self.runtime)

    def tearDown(self):
        set_runtime(None)
        self.store.close()
        self.temporary.cleanup()

    def world(self, provider):
        async def flush():
            pass

        agent = SimpleNamespace(
            runtime=self.runtime,
            llm=GovernedLLM(provider, self.runtime),
            skills=self.registry,
            identity=SimpleNamespace(
                needs_onboarding=False, get_system_prompt_context=lambda: ""
            ),
            personality=SimpleNamespace(
                state=SimpleNamespace(decay=lambda: None), save_state=lambda: None
            ),
            consolidation=None,
            flush_unsummarized_conversations=flush,
        )
        return WorldThread(
            agent,
            WorldThreadConfig(
                thread_path=str(self.root / "thread.json"),
                state_path=str(self.root / "world.json"),
            ),
        )

    async def test_next_world_opportunity_uses_excerpt_and_preserves_full_receipt(self):
        async with self.runtime.execution("autonomy"):
            failed = await self.registry.execute("large_failure")
        reference = failed.data["receipt_id"]
        provider = Provider(
            [LLMResponse(content="I can consider the failure and continue.")]
        )
        result = await self.world(provider).tick_now()
        self.assertEqual(result["status"], "completed")
        user = json.loads(provider.prompts[0][1].content.split("\n", 1)[1])
        report = user["recent_receipts"][0]
        self.assertEqual(report["id"], reference)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["reported_success"])
        self.assertLess(
            len(json.dumps([m.to_dict() for m in provider.prompts[0]])), 65536
        )
        self.assertEqual(
            self.store.receipt(reference)["payload"]["result"]["output"], BIG_OUTPUT
        )
        self.assertEqual(
            self.runtime.inspect_receipt(reference, offset=100, limit=512)["text"],
            BIG_OUTPUT[100:612],
        )
        async with self.runtime.execution("companion", actor_id="peer"):
            with self.assertRaises(ValueError):
                self.runtime.inspect_receipt(reference)

    async def test_large_output_within_tool_loop_is_bounded_without_losing_protocol(
        self,
    ):
        provider = Provider(
            [
                LLMResponse(tool_calls=[ToolCall("call-one", "large_failure", {})]),
                LLMResponse(
                    content="The operation failed; its complete receipt is retained."
                ),
            ]
        )
        async with self.runtime.execution("autonomy"):
            text, _ = await run_tool_loop(
                GovernedLLM(provider, self.runtime),
                self.registry,
                [LLMMessage("user", "Try this operation")],
            )
        self.assertIn("failed", text)
        second = provider.prompts[1]
        self.assertEqual(second[-1].tool_call_id, "call-one")
        self.assertEqual(second[-2].tool_calls[0]["id"], "call-one")
        view = json.loads(second[-1].content)
        self.assertFalse(view["success"])
        self.assertTrue(view["receipt_id"])
        self.assertTrue(view["context_truncated"])
        self.assertLess(
            len(second[-1].content), self.runtime.config.tool_result_bytes + 512
        )
        self.assertEqual(
            self.store.receipts()[0]["payload"]["result"]["output"], BIG_OUTPUT
        )

    async def test_http_context_failure_is_visible_and_manual_retry_respects_rest(self):
        request = httpx.Request("POST", "http://model.test/chat/completions")
        response = httpx.Response(
            400,
            request=request,
            json={"error": {"message": "maximum context length exceeded"}},
        )
        error = httpx.HTTPStatusError("bad request", request=request, response=response)
        provider = Provider(
            [error, LLMResponse(content="The bounded request succeeded.")]
        )
        world = self.world(provider)
        failed = await world.tick_now()
        self.assertEqual(failed["reason"], "context_limit")
        self.assertEqual(self.runtime.activity_state()["wait_reason"], "context_limit")
        self.assertEqual(
            self.store.meta("model_availability")["status"], "request_rejected"
        )
        waiting = await world._tick_once()
        self.assertFalse(waiting["activation_started"])
        self.assertEqual(waiting["reason"], "context_limit")
        self.assertEqual(len(provider.prompts), 1)
        recovered = await world.tick_now()
        self.assertEqual(recovered["status"], "completed")
        self.assertEqual(self.store.meta("provider_failures"), 0)
        self.assertIsNone(self.runtime.activity_state()["wait_reason"])
        self.runtime.end_turn("resting", wake_at=self.now[0] + 3600)
        resting = await world.tick_now()
        self.assertFalse(resting["activation_started"])
        self.assertEqual(resting["reason"], "sleep")
        self.assertEqual(len(provider.prompts), 2)

    async def test_legacy_backoff_is_distinguished_by_event_order_not_clock(self):
        prior = self.store.get("activity", self.runtime.actor_id)
        self.store.put(
            "activity",
            self.runtime.actor_id,
            self.runtime.actor_id,
            {**prior["data"], "status": "waiting", "wake_at": self.now[0] + 3600},
        )
        self.store.event(
            self.runtime.actor_id,
            "autonomy_unavailable",
            {"error_class": "HTTPStatusError"},
        )
        self.assertEqual(
            self.runtime.activity_state()["wait_reason"], "runtime_unavailable"
        )
        self.assertFalse(self.runtime.due())
        self.assertTrue(self.runtime.due(ignore_error_backoff=True))
        self.runtime.end_turn("waiting", wake_at=self.now[0] + 3600)
        self.assertEqual(self.runtime.activity_state()["wait_reason"], "chosen_wait")
        self.assertFalse(self.runtime.due(ignore_error_backoff=True))

    async def test_boundary_keeps_latest_turn_call_ids_images_and_original_objects(
        self,
    ):
        image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "a" * 100000},
        }
        messages = [
            LLMMessage("system", "Identity and operating rules."),
            LLMMessage("user", "old" * 100000),
            LLMMessage("assistant", "prior" * 100000),
            LLMMessage("user", [{"type": "text", "text": "Current request"}, image]),
            LLMMessage(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": "same-id",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            ),
            LLMMessage(
                "tool",
                json.dumps(
                    {
                        "success": False,
                        "output": BIG_OUTPUT,
                        "data": {"receipt_id": "receipt-x"},
                    }
                ),
                tool_call_id="same-id",
            ),
        ]
        current = bound_messages(messages)
        self.assertEqual(current[0].content, "Identity and operating rules.")
        self.assertEqual(current[1].content[0]["text"], "Current request")
        self.assertEqual(current[1].content[1], image)
        self.assertEqual(current[-1].tool_call_id, "same-id")
        self.assertEqual(current[-2].tool_calls[0]["id"], "same-id")
        self.assertEqual(messages[1].content, "old" * 100000)
        self.assertEqual(json.loads(messages[-1].content)["output"], BIG_OUTPUT)

    async def test_oversized_schema_is_local_context_error_without_provider_charge(
        self,
    ):
        provider = Provider([])
        self.runtime.config.model_input_bytes = 8192
        async with self.runtime.execution("autonomy") as execution:
            with self.assertRaises(ContextViewTooLarge):
                await GovernedLLM(provider, self.runtime).chat(
                    [LLMMessage("user", "Hello")], tools=[{"description": "x" * 20000}]
                )
            self.assertEqual(self.store.run(execution.execution_id)["calls"], 0)
        self.assertEqual(provider.prompts, [])

    async def test_budget_stop_after_recovery_clears_the_old_error_backoff(self):
        activity = self.store.get("activity", self.runtime.actor_id)
        self.store.put(
            "activity",
            self.runtime.actor_id,
            self.runtime.actor_id,
            {**activity["data"], "status": "waiting", "wake_at": self.now[0] + 3600},
        )
        self.store.event(
            self.runtime.actor_id,
            "autonomy_unavailable",
            {"error_class": "HTTPStatusError"},
        )
        self.store.set_meta("provider_failures", 12)
        self.runtime.config.background_calls_per_activation = 1
        provider = Provider(
            [LLMResponse(tool_calls=[ToolCall("attempt", "large_failure", {})])]
        )
        result = await self.world(provider).tick_now()
        self.assertEqual(result["status"], "budget_exhausted")
        state = self.runtime.activity_state()
        self.assertEqual(state["status"], "budget_exhausted")
        self.assertIsNone(state["wake_at"])
        self.assertIsNone(state["wait_reason"])
        self.assertEqual(self.store.meta("provider_failures"), 0)
        self.assertTrue(self.runtime.due())

    async def test_restart_repairs_a_retained_budget_stop_without_repeating_work(self):
        from backend.core.runtime import current_terminal

        activity = self.store.get("activity", self.runtime.actor_id)
        self.store.put(
            "activity",
            self.runtime.actor_id,
            self.runtime.actor_id,
            {
                **activity["data"],
                "status": "waiting",
                "wake_at": self.now[0] + 3600,
                "wait_reason": "context_limit",
            },
        )
        self.store.event(
            self.runtime.actor_id,
            "autonomy_unavailable",
            {"error_class": "HTTPStatusError"},
        )
        async with self.runtime.execution("autonomy") as execution:
            self.store.event(
                self.runtime.actor_id,
                "world_turn_completed",
                {"execution_id": execution.execution_id, "status": "budget_exhausted"},
            )
            current_terminal.set({"status": "budget_exhausted"})
        restarted = RuntimeService(
            self.store, self.runtime.workspace, RuntimeConfig(), self.root
        )
        self.assertEqual(restarted.activity_state()["status"], "budget_exhausted")
        self.assertIsNone(restarted.activity_state()["wake_at"])
        self.assertTrue(restarted.due())
        events = self.store.events(limit=100)
        self.assertEqual(
            len([e for e in events if e["kind"] == "activity_backoff_reconciled"]), 1
        )
        restarted.reconcile_activity()
        self.assertEqual(
            len(
                [
                    e
                    for e in self.store.events(limit=100)
                    if e["kind"] == "activity_backoff_reconciled"
                ]
            ),
            1,
        )
