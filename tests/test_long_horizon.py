"""Room to choose sustained work, without requiring sustained activity."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import httpx

from backend.config import RuntimeConfig, LLMConfig
from backend.core.llm import LLMClient, LLMMessage, LLMResponse, ToolCall
from backend.core.runtime import RuntimeService, RuntimeStore, GovernedLLM, set_runtime
from backend.core.runtime.contracts import BudgetExhausted
from backend.core.runtime.context_view import ContextViewTooLarge, tool_result_json
from backend.core.tool_loop import run_tool_loop
from backend.skills.base import Skill, SkillResult
from backend.skills.registry import SkillRegistry


class Provider:
    config = SimpleNamespace(model="fixture", max_tokens=4096)

    def __init__(self, window=131072, tokenizer=True):
        self.window = window
        self.tokenizer = tokenizer
        self.prompts = []
        self.token_probes = []

    async def count_context_tokens(self, messages, tools=None):
        if not self.tokenizer:
            return None
        tokens = (
            len(json.dumps([m.to_dict() for m in messages]))
            + len(json.dumps(tools or []))
        ) // 4
        self.token_probes.append(tokens)
        return {"count": tokens, "max_model_len": self.window}

    async def chat(self, messages, **kwargs):
        self.prompts.append(messages)
        return LLMResponse(content="I can stop here.")

    async def chat_stream(self, messages, **kwargs):
        self.prompts.append(messages)
        yield "A response"


class EndRest(Skill):
    name = "end_turn"
    description = "Choose rest"
    parameters = {}

    def __init__(self, runtime):
        self.runtime = runtime

    async def execute(self, **kwargs):
        self.runtime.end_turn("resting", wake_at=self.runtime.store.clock() + 3600)
        return SkillResult(True)


class LongHorizonTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = RuntimeStore(self.root / "runtime.sqlite3")
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.root
        )
        set_runtime(self.runtime)

    def tearDown(self):
        set_runtime(None)
        self.store.close()
        self.temporary.cleanup()

    async def test_no_daily_cap_preserves_charges_across_restart(self):
        for _ in range(2):
            async with self.runtime.execution("autonomy") as context:
                for _ in range(100):
                    self.store.charge(
                        context, self.runtime.config.daily_background_calls
                    )
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM charges").fetchone()[0], 200
        )
        self.store.close()
        self.store = RuntimeStore(self.root / "runtime.sqlite3")
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.root
        )
        async with self.runtime.execution("autonomy") as context:
            self.store.charge(context, self.runtime.config.daily_background_calls)
            self.assertEqual(self.store.run(context.execution_id)["calls"], 1)
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM charges").fetchone()[0], 201
        )

    async def test_configured_daily_cap_still_works(self):
        async with self.runtime.execution("autonomy") as context:
            self.store.charge(context, background_limit=1)
            with self.assertRaises(BudgetExhausted):
                self.store.charge(context, background_limit=1)

    async def test_children_do_not_escape_activation_boundary_without_daily_cap(self):
        async with self.runtime.execution("autonomy", max_calls=2) as parent:
            async with self.runtime.execution(
                "companion", actor_id="peer", max_calls=16
            ) as child:
                self.store.charge(child, background_limit=None)
                self.store.charge(child, background_limit=None)
                with self.assertRaises(BudgetExhausted):
                    self.store.charge(child, background_limit=None)
                self.assertEqual(self.store.run(parent.execution_id)["calls"], 2)

    async def test_chosen_rest_stops_after_one_call_with_unused_allowance(self):
        provider = Provider()

        async def rest(messages, **kwargs):
            provider.prompts.append(messages)
            return LLMResponse(tool_calls=[ToolCall("rest", "end_turn", {})])

        provider.chat = rest
        registry = SkillRegistry()
        registry.register(EndRest(self.runtime))
        async with self.runtime.execution("autonomy") as context:
            text, _ = await run_tool_loop(
                GovernedLLM(provider, self.runtime),
                registry,
                [LLMMessage("user", "An opportunity")],
                max_rounds=128,
            )
            self.assertEqual(text, "")
            self.assertEqual(self.store.run(context.execution_id)["calls"], 1)
            self.assertEqual(self.store.run(context.execution_id)["max_calls"], 128)
        self.assertEqual(self.runtime.activity_state()["status"], "resting")
        self.assertFalse(self.runtime.due())
        self.assertEqual(len(provider.prompts), 1)

    async def test_large_context_is_retained_and_only_generation_is_charged(self):
        provider = Provider()
        text = "Research material. " * 20000
        async with self.runtime.execution("autonomy") as context:
            await GovernedLLM(provider, self.runtime).chat([LLMMessage("user", text)])
            self.assertEqual(self.store.run(context.execution_id)["calls"], 1)
        self.assertEqual(provider.prompts[0][-1].content, text)
        self.assertEqual(len(provider.token_probes), 1)
        report = self.store.meta("model_context")
        self.assertEqual(report["mode"], "server_tokenizer")
        self.assertEqual(report["input_tokens_limit"], 131072 - 4096 - 1024)

    async def test_smaller_server_window_and_reply_override_govern_trimming(self):
        provider = Provider(window=16384)
        text = "detail " * 20000
        original = [LLMMessage("user", text)]
        async with self.runtime.execution("autonomy"):
            await GovernedLLM(provider, self.runtime).chat(original, max_tokens=8192)
        report = self.store.meta("model_context")
        self.assertEqual(report["model_window_tokens"], 16384)
        self.assertEqual(report["input_tokens_limit"], 16384 - 8192 - 1024)
        self.assertLessEqual(provider.token_probes[-1], report["input_tokens_limit"])
        self.assertEqual(original[0].content, text)
        self.assertGreater(len(provider.token_probes), 1)

    async def test_impossible_reply_reserve_never_dispatches_or_charges(self):
        provider = Provider(window=8192)
        async with self.runtime.execution("autonomy") as context:
            with self.assertRaises(ContextViewTooLarge):
                await GovernedLLM(provider, self.runtime).chat(
                    [LLMMessage("user", "Hello")], max_tokens=8192
                )
            self.assertEqual(self.store.run(context.execution_id)["calls"], 0)
        self.assertEqual(provider.prompts, [])

    async def test_missing_tokenizer_uses_conservative_fallback(self):
        provider = Provider(tokenizer=False)
        text = "detail " * 50000
        async with self.runtime.execution("autonomy"):
            await GovernedLLM(provider, self.runtime).chat([LLMMessage("user", text)])
        size = len(json.dumps([m.to_dict() for m in provider.prompts[0]]).encode())
        self.assertLess(size, 65536)
        self.assertEqual(self.store.meta("model_context")["mode"], "conservative_bytes")

    async def test_streaming_uses_same_measured_context(self):
        provider = Provider()
        text = "detail " * 20000
        result = [
            part
            async for part in GovernedLLM(provider, self.runtime).chat_stream(
                [LLMMessage("user", text)]
            )
        ]
        self.assertEqual(result, ["A response"])
        self.assertEqual(provider.prompts[0][-1].content, text)
        self.assertEqual(self.store.meta("model_context")["mode"], "server_tokenizer")

    async def test_user_preempts_a_long_activation(self):
        provider = Provider()
        entered = asyncio.Event()

        async def long_call(messages, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        provider.chat = long_call
        executions = []

        async def activate():
            async with self.runtime.execution("autonomy") as context:
                executions.append(context.execution_id)
                await GovernedLLM(provider, self.runtime).chat(
                    [LLMMessage("user", "An opportunity")]
                )

        task = asyncio.create_task(activate())
        await asyncio.wait_for(entered.wait(), 2)
        async with self.runtime.execution("user"):
            self.assertTrue(task.cancelled())
            self.assertEqual(self.store.run(executions[0])["status"], "interrupted")

    async def test_receipt_pages_can_exceed_the_old_thousand_character_limit(self):
        async with self.runtime.execution("autonomy") as context:
            receipt, _ = self.store.begin_receipt(
                context, "fixture", arguments={"source": "large text"}
            )
            self.store.finish_receipt(
                receipt["id"],
                "succeeded",
                {"success": True, "output": "detail " * 20000},
            )
            page = self.runtime.inspect_receipt(receipt["id"], limit=32768)
            self.assertEqual(len(page["text"]), 32768)
            self.assertEqual(page["next_offset"], 32768)

    async def test_large_tool_preview_retains_useful_text_and_bounds_control_bytes(
        self,
    ):
        for text in ["research " * 30000, "\x00\x01" * 100000]:
            raw = {
                "success": True,
                "output": text,
                "data": {"receipt_id": "receipt-source"},
            }
            preview = tool_result_json(raw, limit=32768)
            self.assertLessEqual(len(preview.encode()), 32768)
            self.assertEqual(json.loads(preview)["receipt_id"], "receipt-source")
            self.assertEqual(raw["output"], text)
        normal = json.loads(
            tool_result_json({"output": "research " * 30000}, limit=32768)
        )
        self.assertGreater(len(normal["output_excerpt"]), 10000)

    async def test_tokenizer_transport_preserves_tools_and_backs_off_unsupported_route(
        self,
    ):
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={"count": 72, "max_model_len": 131072})

        client = LLMClient(LLMConfig(base_url="http://model/v1"))
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        tools = [
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object"}},
            }
        ]
        try:
            result = await client.count_context_tokens(
                [LLMMessage("user", "Hello")], tools
            )
            self.assertEqual(result["count"], 72)
            self.assertEqual(str(requests[0].url), "http://model/tokenize")
            self.assertEqual(json.loads(requests[0].content)["tools"], tools)
        finally:
            await client._client.aclose()
        requests.clear()

        def missing(request):
            requests.append(request)
            return httpx.Response(404)

        client._client = httpx.AsyncClient(transport=httpx.MockTransport(missing))
        try:
            self.assertIsNone(
                await client.count_context_tokens([LLMMessage("user", "Hello")])
            )
            self.assertIsNone(
                await client.count_context_tokens([LLMMessage("user", "Hello")])
            )
            self.assertEqual(len(requests), 1)
        finally:
            await client._client.aclose()
