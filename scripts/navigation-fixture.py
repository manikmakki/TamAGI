"""Disposable browser target; deterministic provider, no live model or resident."""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
import os
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock

import uvicorn
import yaml

from backend.config import TamAGIConfig, set_config
from backend.core.llm import LLMResponse, ToolCall
from backend.core.runtime import (
    GovernedLLM,
    RuntimeService,
    RuntimeStore,
    current_execution,
)
from backend.skills.base import Skill, SkillResult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18742)
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tamagi-navigation-") as temporary:
        root = Path(temporary)
        config = TamAGIConfig()
        config.runtime.enabled = True
        config.runtime.instance_dir = str(root)
        config.runtime.authority = "guarded"
        config.autonomy.enabled = False
        config.world_thread.enabled = False
        config.llm.base_url = "http://127.0.0.1:1/v1"
        config.llm.model = "navigation-fixture"
        config.memory.context.enabled = False
        store = RuntimeStore(root / "runtime.sqlite3")
        runtime = RuntimeService(store, root / "workspace", config.runtime)
        runtime.identity(
            "IDENTITY.md", "# Identity\n- **Name**: Navigation fixture\n", 0
        )
        runtime.identity(
            "SOUL.md", "# Soul\nA mechanical fixture, not a live resident.\n", 0
        )
        runtime.identity("USER.md", "# User\nA test operator.\n", 0)
        (runtime.workspace / "workshop/leaf.md").write_text("A leaf in the workshop.")
        (runtime.workspace / "workshop/preview.html").write_text(
            "<h1>A creation</h1><script>parent.__previewEscaped=true;fetch('/api/state')</script>"
        )
        (runtime.workspace / "research").mkdir(exist_ok=True)
        (runtime.workspace / "research/earlier.md").write_text(
            "A saved earlier creation"
        )
        runtime.register_artifact("research/earlier.md")
        store.close()
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config.model_dump()))
        os.environ["TAMAGI_CONFIG"] = str(path)
        set_config(config)
        from backend.main import app
        from backend.api.chat import get_agent

        class Probe(Skill):
            name = "navigation_probe"
            description = "A delayed, observable fixture write"
            parameters = {"delay": {"type": "integer"}}
            calls = 0

            async def execute(self, delay=4, **kwargs):
                self.calls += 1
                (root / "workspace/workshop/probe.txt").write_text(str(self.calls))
                await asyncio.sleep(delay)
                return SkillResult(True, output="Fixture effect completed")

        class Provider:
            def __init__(self):
                self.config = config.llm
                self.counts = {}
                self.prompts = {}

            async def chat(self, messages, **kwargs):
                execution = current_execution.get()
                if execution.actor_id != get_agent().runtime.actor_id:
                    await asyncio.sleep(1)
                key = execution.root_id
                count = self.counts.get(key, 0)
                self.counts[key] = count + 1
                text = str(
                    next(
                        (m.content for m in reversed(messages) if m.role == "user"), ""
                    )
                )
                self.prompts.setdefault(key, []).append(text)
                if "approval fixture" in text.lower() and count == 0:
                    return LLMResponse(
                        tool_calls=[
                            ToolCall("fixture-approval", "approval_fixture", {})
                        ]
                    )
                if "quiet" in text.lower():
                    tools = ["request_quiet_turn", "end_turn", "navigation_probe"]
                    if count < len(tools):
                        return LLMResponse(
                            tool_calls=[
                                ToolCall(
                                    f"fixture-{count}",
                                    tools[count],
                                    {"delay": 4} if count == 2 else {},
                                )
                            ]
                        )
                elif "slow" in text.lower() and count == 0:
                    return LLMResponse(
                        tool_calls=[
                            ToolCall("fixture-probe", "navigation_probe", {"delay": 4})
                        ]
                    )
                return LLMResponse(content="Navigation fixture finished.")

        original = app.router.lifespan_context
        probe = Probe()
        provider = Provider()

        class ApprovalProbe(Skill):
            name = "approval_fixture"
            description = "Exercise approval delivery without executing a host command"
            parameters = {}

            async def execute(self, **kwargs):
                from backend.skills.exec_skill import ExecSkill

                approved, _ = await ExecSkill()._request_approval(
                    "fixture (no command is executed)",
                    "fixture",
                    kwargs.get("_event_callback"),
                    kwargs.get("_pending_approvals"),
                )
                return SkillResult(
                    approved,
                    output="Fixture approved" if approved else "Fixture denied",
                )

        @asynccontextmanager
        async def lifespan(application):
            async with original(application):
                agent = get_agent()
                agent.llm = GovernedLLM(provider, agent.runtime)
                agent.runtime.companions.llm = agent.llm
                agent.qa_pipeline = None
                agent.relational_consolidator = None
                agent._ensure_world_conversation_context = AsyncMock(return_value=False)
                agent.skills.register(probe)
                agent.skills.register(ApprovalProbe())
                yield

        app.router.lifespan_context = lifespan

        @app.get("/api/navigation-fixture")
        async def inspection():
            runtime = get_agent().runtime
            return {
                "tool_calls": probe.calls,
                "model_calls": sum(provider.counts.values()),
                "input_epoch": runtime.input_epoch,
                "receipts": [
                    {"id": r["id"], "status": r["status"]}
                    for r in runtime.store.receipts()
                ],
                "pending": runtime.interactions.has_pending(),
                "prompts": {
                    json.loads(runtime.store.run(key)["payload"]).get(
                        "request_id", key
                    ): values
                    for key, values in provider.prompts.items()
                },
            }

        # The production SPA catch-all is already registered at import time.
        fixture_route = app.router.routes.pop()
        app.router.routes.insert(0, fixture_route)
        uvicorn.run(app, host="127.0.0.1", port=options.port, log_level="warning")


if __name__ == "__main__":
    main()
