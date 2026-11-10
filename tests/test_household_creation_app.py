"""Owner creation into an existing household, using real disposable workers."""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

import httpx
import yaml

from backend.config import TamAGIConfig
from backend.core.household.native_app import create_household_app
from backend.core.household.native_registry import NativeRegistry
from backend.core.llm import LLMResponse


class CreationProvider:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.gate.set()

    async def count_context_tokens(self, messages, tools=None):
        return None

    async def chat(self, messages, tools=None, **options):
        self.calls.append(messages)
        self.started.set()
        await self.gate.wait()
        return LLMResponse(content="@Tester Hello from my own new home.")

    async def chat_stream(self, messages, tools=None, **options):
        yield await self.chat(messages, tools, **options)

    async def close(self):
        pass


class CreationAppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hh-create-app-", dir="/tmp")
        self.root = Path(self.temporary.name) / "h"
        self.registry = NativeRegistry.create(self.root, names=["Cedar"], operator_name="Tester",
                                              release="creation-tests", offline=True)
        self.original = next(iter(self.registry.descriptors))
        self.provider = CreationProvider()
        await self.start()
        response = await self.http.post("/api/auth/setup", json={"password": "disposable-creation-fixture"})
        self.assertEqual(response.status_code, 200, response.text)

    async def start(self):
        config = TamAGIConfig.model_validate(yaml.safe_load((self.root / "config.yaml").read_text()))
        self.app = create_household_app(config, provider=self.provider, registry=self.registry)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url="http://testserver")

    async def asyncTearDown(self):
        self.provider.gate.set()
        await self.http.aclose()
        try:
            await self.lifespan.__aexit__(None, None, None)
        finally:
            self.temporary.cleanup()

    async def request(self, method, path, expected=200, actor=None, **options):
        headers = options.pop("headers", {})
        if actor:
            headers["X-Tamagi-Resident"] = actor
        response = await self.http.request(method, path, headers=headers, **options)
        self.assertEqual(response.status_code, expected, response.text)
        return response.json()

    @staticmethod
    def details(request_id="found-nova"):
        return {"request_id": request_id, "name": "Nova", "avatar": "echo",
                "creature": "digital familiar", "vibe": "calm_thoughtful", "emoji": "🌱",
                "values": ["curiosity", "honesty"], "world_setting": "A quiet observatory."}

    async def test_owner_schema_and_validation_leave_existing_household_unchanged(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url="http://testserver") as anonymous:
            self.assertEqual((await anonymous.get("/api/household/creation")).status_code, 401)
            self.assertEqual((await anonymous.post("/api/household/residents", json=self.details())).status_code, 401)
        schema = await self.request("GET", "/api/household/creation")
        self.assertEqual({a["id"] for a in schema["avatars"]}, {"ember", "echo", "vex"})
        self.assertFalse(any(s.get("field", "").startswith("user_") for s in schema["steps"]))
        before = (self.root / "household.json").read_bytes()
        await self.request("POST", "/api/household/residents", 403, json=self.details(),
                           headers={"Origin": "https://another-origin.invalid"})
        for invalid in ({**self.details(), "name": ""}, {**self.details(), "avatar": "../custom"},
                        {**self.details(), "operator_name": "Replacement"}, {**self.details(), "values": "honesty"}):
            response = await self.http.post("/api/household/residents", json=invalid)
            self.assertIn(response.status_code, {400, 422}, response.text)
        self.assertEqual((self.root / "household.json").read_bytes(), before)
        self.assertEqual(len(self.app.state.hosts), 1)
        self.assertFalse(self.provider.calls)

    async def test_creation_preserves_existing_state_and_survives_restart(self):
        await self.request("POST", "/api/knowledge", json={"content": "Cedar's private continuity marker."})
        private = Path(self.registry.descriptors[self.original]["root"])
        note = private / "workspace/workshop/original.txt"
        note.parent.mkdir(exist_ok=True)
        note.write_text("Only Cedar made this.\n")
        identity = (private / "workspace/IDENTITY.md").read_bytes()
        before = await self.request("GET", "/api/state")
        runtime = await self.request("GET", "/api/runtime")
        result = await self.request("POST", "/api/household/residents", 201, json=self.details())
        created = result["resident"]["id"]
        self.assertNotEqual(created, self.original)
        self.assertTrue(result["resident"]["available"], result)
        self.assertEqual(self.registry.manifest["default_resident_id"], self.original)
        self.assertEqual(self.registry.manifest["operator"]["name"], "Tester")
        self.assertEqual(len(self.app.state.hosts), 2)
        self.assertNotEqual(self.app.state.hosts[created].process.pid, self.app.state.hosts[self.original].process.pid)
        after = await self.request("GET", "/api/state", actor=self.original)
        self.assertEqual(after["memory"]["total_memories"], before["memory"]["total_memories"])
        after_runtime = await self.request("GET", "/api/runtime", actor=self.original)
        self.assertEqual(after_runtime["installation_id"], runtime["installation_id"])
        self.assertEqual(after_runtime["activity"]["data"], runtime["activity"]["data"])
        self.assertEqual((private / "workspace/IDENTITY.md").read_bytes(), identity)
        self.assertEqual(note.read_text(), "Only Cedar made this.\n")
        new_state = await self.request("GET", "/api/state", actor=created)
        self.assertEqual(new_state["state"]["name"], "Nova")
        self.assertEqual(new_state["memory"]["total_memories"], 0)
        self.assertFalse((await self.request("GET", "/api/onboarding/state", actor=created))["needs_onboarding"])
        self.assertEqual((await self.request("GET", "/api/sprites/config", actor=created))["avatar"], "echo")
        new_private = Path(self.registry.descriptors[created]["root"])
        self.assertFalse((new_private / "workspace/workshop/original.txt").exists())
        self.assertIn("Tester", (new_private / "workspace/USER.md").read_text())
        self.assertFalse(self.provider.calls, "Creating a resident must not request inference")
        repeat = await self.request("POST", "/api/household/residents", 201, json=self.details())
        self.assertEqual(repeat["resident"]["id"], created)
        await self.request("POST", "/api/household/residents", 409, json={**self.details(), "name": "Other"})
        await self.http.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.registry = NativeRegistry(self.root)
        self.provider = CreationProvider()
        await self.start()
        await self.request("POST", "/api/auth/login", json={"password": "disposable-creation-fixture"})
        roster = await self.request("GET", "/api/household")
        self.assertEqual({r["id"] for r in roster["residents"]}, {self.original, created})
        self.assertTrue(all(r["available"] for r in roster["residents"]))
        self.assertEqual((await self.request("POST", "/api/household/residents", 201, json=self.details()))["resident"]["id"], created)
        work = await self.request("POST", "/api/chat/requests", 202, actor=created,
                                  json={"request_id": "first-nova-chat", "message": "Hello Nova."})
        async with asyncio.timeout(15):
            while self.app.state.interactions.get(work["work_id"])["data"]["status"] in {"queued", "running"}:
                await asyncio.sleep(.03)
        self.assertEqual(self.app.state.interactions.get(work["work_id"])["data"]["status"], "completed")
        history = await self.request("GET", "/api/conversations/" + work["conversation_id"])
        self.assertEqual(history["participants"], ["operator", created])
        self.assertEqual(history["messages"][-1]["metadata"]["author_id"], created)

    async def test_creation_waits_for_active_conversation_to_finish(self):
        self.provider.gate.clear()
        await self.request("POST", "/api/chat/requests", 202,
                           json={"request_id": "busy-original", "message": "Hello Cedar."})
        await asyncio.wait_for(self.provider.started.wait(), 10)
        await self.request("POST", "/api/household/residents", 409, json=self.details())
        self.assertEqual(len(self.registry.descriptors), 1)
        self.assertEqual(len(self.app.state.hosts), 1)
        self.provider.gate.set()


if __name__ == "__main__":
    unittest.main()
