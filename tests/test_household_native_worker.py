"""Native Agent/tool execution with immutable household input and private roots."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from backend.config import TamAGIConfig, get_config, set_config
from backend.core.agent import Message, TamAGIAgent
from backend.core.household.contracts import AccessDenied, ProtocolError, WorkerBinding
from backend.core.household.conversations import Conversations
from backend.core.household.model_transport import current_household_admission
from backend.core.household.native_registry import NativeRegistry
from backend.core.household.native_worker import NativeAuthentication, configure_app
from backend.core.household.native_world import NativeWorld
from backend.core.household.worker_bridge import NativeWorkerBridge, install_worker, validate_native_binding
from backend.core.identity import IdentityManager
from backend.core.llm import LLMResponse, ToolCall
from backend.core.personality import PersonalityEngine
from backend.core.runtime import GovernedLLM, RuntimeService, RuntimeStore, current_execution, set_runtime
from backend.core.runtime.interactions import AdmissionError, Interactions
from backend.core.runtime.memory import CanonicalMemory
from backend.skills.registry import SkillRegistry
from backend.skills.write_skill import WriteSkill
from backend.skills.read_world_graph_skill import ReadWorldGraphSkill
from backend.core.self_model.store import SelfModel
from backend.skills.runtime_skill import RuntimeSkill, param


class ControlledModel:
    """Exercise the real Agent/registry/receipt loop without a network provider."""

    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config
        self.starts, self.calls, self.bindings = [], [], {}
        self.responses = []
        self.contexts, self.active = {}, set()

    async def start_execution(self, context, max_calls, *, runtime=None):
        grant = copy.deepcopy(current_household_admission.get())
        self.starts.append((context.origin, grant))
        self.contexts[context.execution_id] = context
        self.active.add(context.execution_id)
        self.bindings[context.execution_id] = {"origin": grant["origin"] if grant else "autonomy",
            "exchange_id": grant["exchange_id"] if grant else context.root_id,
            "calls_remaining": 128, "segment": 0, "helper": bool(context.depth),
            "execution_id": context.execution_id, "actor_id": context.actor_id,
            "root_id": context.root_id, "first_response": False}

    async def end_execution(self, context, status):
        self.active.discard(context.execution_id)

    def lineage(self, context):
        return self.bindings[context.execution_id]

    async def resources(self, context):
        return self.lineage(context)

    def assert_active(self, context):
        self.lineage(context)
        if context.execution_id not in self.active:
            raise AccessDenied("This test execution is no longer active")

    def _active(self, resident, local_id):
        context = self.contexts[local_id]
        self.assert_active(context)
        if resident != self.runtime.actor_id:
            raise AccessDenied("Wrong test worker")
        return {}, {"actor_id": context.actor_id}, self.lineage(context)

    async def count_context_tokens(self, messages, tools=None):
        return None

    async def chat(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        context = current_execution.get()
        count = self.runtime.store.run(context.execution_id)["calls"] + 1
        self.runtime.project_model_charge({"attempt_id": f"controlled-{len(self.calls)}",
            "local_execution_id": context.execution_id, "local_calls": {context.execution_id: count},
            "bucket": "user", "charged_at": self.runtime.store.clock()})
        return self.responses.pop(0) if self.responses else LLMResponse(content="A source-aware answer.")


class NativeWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_config = get_config()
        self.temporary = tempfile.TemporaryDirectory(prefix="native-worker-test-", dir="/tmp")
        self.registry = NativeRegistry.create(Path(self.temporary.name) / "household",
            names=["First", "Second"], release="controlled-source", offline=True)
        self.descriptor, self.peer = list(self.registry.descriptors.values())
        self.actor = self.descriptor["resident_id"]
        self.binding = WorkerBinding.parse({key: self.descriptor[key] for key in WorkerBinding.__dataclass_fields__})
        self.root = Path(self.binding.root)
        self.config = TamAGIConfig()
        self.config.workspace.path = str(self.root / "workspace")
        self.config.history.persist_path = str(self.root / "data/history")
        self.config.history.context_compress_threshold = 0
        self.config.memory.context.enabled = False
        self.config.guardrails.allowed_write_paths = [str(self.root / "workspace")]
        self.config.guardrails.allowed_read_paths = [str(self.root / "workspace")]
        self.store = RuntimeStore(self.root / "runtime.sqlite3")
        self.runtime = RuntimeService(self.store, self.root / "workspace", self.config.runtime, self.root)
        set_config(self.config)
        set_runtime(self.runtime)
        self.model = ControlledModel(self.runtime, self.config.llm)
        self.world = NativeWorld(self.registry, self.model)
        self.grants, self.commits = {}, []
        self.conversations = Conversations(self.registry.store, self.registry.participants,
            validate_lineage=lambda supplied, author: {**supplied, "origin": "user"})
        self.bridge = NativeWorkerBridge(self.runtime, self.binding, "/unused.sock", "x" * 40,
            self.model, transport=httpx.MockTransport(self.coordinator_request))
        self.runtime.household = self.bridge
        self.skills = SkillRegistry()
        self.skills.register(WriteSkill())
        self.memory = CanonicalMemory(self.runtime)
        await self.memory.initialize()
        personality_file = patch.object(PersonalityEngine, "STATE_FILE", str(self.root / "data/personality.json"))
        personality_file.start()
        self.addCleanup(personality_file.stop)
        self.agent = TamAGIAgent(config=self.config, llm=GovernedLLM(self.model, self.runtime),
            memory=self.memory, personality=PersonalityEngine(name="First"), skills=self.skills,
            identity=IdentityManager(data_dir=str(self.root / "data"), workspace_dir=str(self.runtime.workspace)))
        self.agent.runtime = self.runtime
        self.bridge.agent = self.agent
        self.manager = Interactions(self.runtime, self.agent)
        self.runtime.interactions = self.manager

    async def asyncTearDown(self):
        await self.manager.close()
        await self.memory.close()
        self.bridge.close()
        set_runtime(None)
        set_config(self.previous_config)
        self.store.close()
        self.registry.close()
        self.temporary.cleanup()

    def coordinator_request(self, request):
        if request.url.path == "/_native/notebook" and json.loads(request.content)["action"] == "inspect":
            return httpx.Response(200, json=[])
        if request.url.path == "/_native/world":
            return httpx.Response(200, json=self.world.dispatch(self.actor, json.loads(request.content)))
        if request.url.path == "/_native/conversations/commit":
            body = json.loads(request.content)
            self.commits.append(body)
            grant = self.grants[body["grant_id"]]
            message = self.conversations.send(self.actor, grant["conversation_id"],
                body["local_message"]["content"], body["command_id"],
                lineage={"execution_id": "canonical-execution", "exchange_id": grant["exchange_id"], "root_id": grant["exchange_id"]})
            record = self.registry.store.get("message", message["id"])
            self.registry.store.put("message", message["id"], {**record["data"], "metadata": body["local_message"]["metadata"]})
            return httpx.Response(200, json={"message": {**message, "metadata": body["local_message"]["metadata"]}})
        return httpx.Response(404)

    def input(self, author=None, content="A question from another resident.", *, conversation=None, key="input"):
        if conversation is None:
            opened = self.conversations.create("operator", [self.actor, self.peer["resident_id"]], key + ":thread",
                group=True, opening="The shared discussion begins.")
            conversation = opened["conversation"]["id"]
        author = author or self.peer["resident_id"]
        message = self.conversations.send(author, conversation, content, key,
            lineage={"execution_id": "source-execution", "exchange_id": "shared-exchange", "root_id": "shared-exchange"})
        grant = {"id": key + ":grant", "resident_id": self.actor, "exchange_id": "shared-exchange",
                 "conversation_id": conversation, "origin": "user", "through": message["sequence"], "message": message}
        self.grants[grant["id"]] = grant
        snapshot = {"conversation": self.registry.store.get("conversation", conversation),
                    "messages": self.conversations.history(self.actor, conversation, through=message["sequence"]),
                    "watermark": message["sequence"]}
        return grant, snapshot

    async def run_input(self, grant, snapshot):
        work = await self.manager.submit_household({}, grant, snapshot, grant["id"])
        if self.manager.worker is None:
            self.manager.start()
        result = await asyncio.wait_for(self.manager.wait(work["work_id"]), 10)
        return work, result

    async def test_real_agent_executes_tool_and_commits_only_own_attributed_output(self):
        grant, snapshot = self.input()
        self.model.responses = [LLMResponse(tool_calls=[ToolCall("native-write", "write", {
            "path": "workshop/native-note.txt", "content": "An actual retained tool effect."})]),
            LLMResponse(content="I saved the note.")]
        epoch = self.runtime.input_epoch
        work, result = await self.run_input(grant, snapshot)
        self.assertEqual(result["turn_status"], "completed")
        self.assertEqual((self.runtime.workspace / "workshop/native-note.txt").read_text(), "An actual retained tool effect.")
        self.assertEqual(len(self.store.receipts()), 1)
        self.assertEqual(len(self.commits), 1)
        self.assertEqual(self.model.starts[0][0], "peer")
        self.assertEqual(self.runtime.input_epoch, epoch)
        prompt = "\n".join(str(message.content) for message in self.model.calls[0])
        self.assertIn("Attributed message from Second", prompt)
        self.assertIn(self.peer["resident_id"], prompt)
        self.assertFalse(self.store.list("conversation"))
        self.assertTrue(self.store.get("household_projection", grant["conversation_id"]))
        self.assertFalse(Path(self.config.history.persist_path, grant["conversation_id"] + ".json").exists())
        history = self.conversations.history("operator", grant["conversation_id"])
        self.assertEqual(history[-1]["author"], self.actor)
        self.assertEqual(history[-1]["content"], "I saved the note.")
        self.assertEqual(self.manager.get(work["work_id"])["data"]["grant_id"], grant["id"])

    async def test_only_actual_operator_input_advances_user_epoch(self):
        peer_grant, peer_snapshot = self.input(key="peer")
        await self.manager.submit_household({}, peer_grant, peer_snapshot, "peer")
        self.assertEqual(self.runtime.input_epoch, 0)
        human_grant, human_snapshot = self.input("operator", "Actual operator input.", key="human")
        await self.manager.submit_household({}, human_grant, human_snapshot, "human")
        self.assertEqual(self.runtime.input_epoch, 1)

    async def test_graph_read_refreshes_saved_shared_places_and_labels_partial_results(self):
        self.agent.self_model = SelfModel(self.root / "data/world-projection.json")
        # Simulate a different resident creating a place after this worker's
        # personal projection was loaded. Query must read canonical state.
        peer = self.peer["resident_id"]
        saved = self.world.world.create(peer, "peer-garden", peer + ":garden", "place", "The Garden", visibility="shared")
        read = ReadWorldGraphSkill(self.agent)
        result = await read.execute(node_type="location", query="The Garden")
        self.assertTrue(result.success)
        self.assertEqual(result.data["nodes"][0]["id"], "world:" + saved["id"])
        default = await read.execute(limit=2)
        self.assertLessEqual(default.data["count"], 2)
        self.assertGreater(default.data["total_count"], default.data["count"])
        self.assertIn("not an inventory", default.output)
        self.assertIn("Private to this resident", self.runtime.context())
        # Reopen canonical storage and a fresh personal projection, without
        # copying the old graph; this is actual read-back after connection close.
        self.registry.store.db.close()
        from backend.core.household.store import HouseholdStore
        self.registry.store = HouseholdStore(self.registry.root / "household.sqlite3")
        self.world.store = self.registry.store
        from backend.core.household.world import World
        self.world.world = World(self.registry.store)
        self.agent.self_model = SelfModel(self.root / "data/reopened-projection.json")
        reopened = await read.execute(node_id="world:" + saved["id"])
        self.assertTrue(reopened.success)
        self.assertEqual(reopened.data["node"]["name"], "The Garden")

    async def test_queued_future_input_cannot_replace_an_earlier_context_snapshot(self):
        first, first_snapshot = self.input(content="First peer input.", key="first")
        second, second_snapshot = self.input(content="FUTURE-QUEUED-INPUT", conversation=first["conversation_id"], key="second")
        accepted = await self.manager.submit_household({}, first, first_snapshot, "first")
        await self.manager.submit_household({}, second, second_snapshot, "second")
        self.assertNotIn(first["conversation_id"], self.agent.conversations)
        stored = self.store.get("interaction_input", accepted["work_id"])["data"]
        self.assertNotIn("FUTURE-QUEUED-INPUT", json.dumps(stored["household_snapshot"]))
        self.manager.start()
        await asyncio.wait_for(self.manager.wait(accepted["work_id"]), 10)
        self.assertTrue(self.model.calls)
        self.assertNotIn("FUTURE-QUEUED-INPUT", str([message.content for message in self.model.calls[0]]))

    async def test_replay_retains_grant_and_does_not_create_input_or_priority(self):
        grant, snapshot = self.input("operator", key="once")
        first = await self.manager.submit_household({}, grant, snapshot, "stable")
        again = await self.manager.submit_household({}, grant, snapshot, "stable")
        self.assertEqual(first["work_id"], again["work_id"])
        self.assertEqual(self.runtime.input_epoch, 1)
        changed = copy.deepcopy(grant)
        changed["id"] = "different-grant"
        with self.assertRaises(AdmissionError):
            await self.manager.submit_household({}, changed, snapshot, "stable")

    async def test_spoofed_author_future_snapshot_and_ordinary_chat_are_rejected(self):
        grant, snapshot = self.input()
        spoofed = copy.deepcopy(grant)
        spoofed["message"]["author"] = "operator"
        with self.assertRaises(ProtocolError):
            await self.manager.submit_household({}, spoofed, snapshot)
        future = copy.deepcopy(snapshot)
        future["messages"].append({"id": "future", "author": "operator", "sequence": grant["through"] + 1, "content": "secret future"})
        with self.assertRaises(ProtocolError):
            await self.manager.submit_household({}, grant, future)
        with self.assertRaises(AdmissionError):
            await self.manager.submit({"message": "Fake human input"})

    async def test_source_aware_memory_labels_peer_without_operator_impersonation(self):
        first, first_snapshot = self.input(content="I enjoy distant mountains.", key="memory-first")
        await self.run_input(first, first_snapshot)
        second, second_snapshot = self.input(content="I also enjoy the coast.", conversation=first["conversation_id"], key="memory-second")
        await self.run_input(second, second_snapshot)
        windows = [record for record in self.store.list("memory") if record["id"].startswith("conversation-window:")]
        self.assertTrue(windows)
        contents = json.dumps(windows)
        self.assertIn("Second: I enjoy distant mountains.", contents)
        self.assertNotIn("User: I enjoy distant mountains.", contents)

    async def test_generated_world_tool_mutates_canonical_topology_not_private_authority(self):
        self.skills.register(RuntimeSkill(self.runtime, "world_action", "Change the canonical world", {
            "action": param("string", required=True), "attributes": param("object")}, self.runtime.world_action))
        private_before = self.store.list("world_object")
        grant, snapshot = self.input(key="world-tool")
        self.model.responses = [LLMResponse(tool_calls=[ToolCall("native-world-create", "world_action", {
            "action": "create", "attributes": {"name": "A shared stone", "visibility": "shared"}})]),
            LLMResponse(content="I placed a stone in our shared world.")]
        _, result = await self.run_input(grant, snapshot)
        self.assertEqual(result["turn_status"], "completed")
        shared = [record for record in self.registry.store.list("world_object") if record["data"]["name"] == "A shared stone"]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["data"]["owner"], self.actor)
        self.assertEqual(shared[0]["data"]["location"], self.actor + ":home")
        self.assertEqual(self.store.list("world_object"), private_before)
        self.assertEqual(self.store.receipts()[0]["status"], "succeeded")

    async def test_world_receipt_replay_preserves_one_generated_identity_and_position(self):
        grant, _ = self.input(key="world-replay")
        token = current_household_admission.set(grant)
        try:
            async with self.runtime.execution("peer") as context:
                self.store.begin_receipt(context, "world_action", "world-receipt", {"action": "create"})
                first = self.runtime.world_action("create", attributes={"name": "One tree", "kind": "place"})
                again = self.runtime.world_action("create", attributes={"name": "One tree", "kind": "place"})
                self.assertEqual(first["id"], again["id"])
                self.store.finish_receipt(self.store.receipts()[0]["id"], "succeeded", {"success": True})
                moved = self.runtime.world_action("move", first["id"])
                self.assertEqual(moved["data"]["location"], first["id"])
                position = self.registry.store.get("world_position", self.actor)
                self.assertEqual(position["data"]["location"], first["id"])
        finally:
            current_household_admission.reset(token)
        self.assertEqual(len([record for record in self.registry.store.list("world_object") if record["data"]["name"] == "One tree"]), 1)
        NativeWorld(self.registry, self.model)
        self.assertEqual(len([record for record in self.registry.store.list("world_object") if record["data"]["name"] == "One tree"]), 1)

    async def test_world_visibility_and_revision_checks_are_canonical(self):
        grant, _ = self.input(key="world-access")
        token = current_household_admission.set(grant)
        try:
            async with self.runtime.execution("peer"):
                private = self.runtime.world_action("create", attributes={"name": "A private keepsake", "visibility": "private"})
                self.assertNotIn(private["id"], [record["id"] for record in self.world.world.list(self.peer["resident_id"])])
                with self.assertRaises(Exception):
                    self.world.world.get(self.peer["resident_id"], private["id"])
                updated = self.runtime.world_action("revise", private["id"], {"name": "A revised keepsake"}, private["revision"])
                self.assertEqual(updated["revision"], private["revision"] + 1)
                with self.assertRaises(Exception):
                    self.runtime.world_action("revise", private["id"], {"name": "Stale edit"}, private["revision"])
        finally:
            current_household_admission.reset(token)

    def test_binding_checks_identity_read_only_before_hydration(self):
        validate_native_binding(self.binding, Path(self.descriptor["config_path"]))
        from dataclasses import replace

        before = (self.root / "runtime.sqlite3").read_bytes()
        with self.assertRaises(Exception):
            validate_native_binding(replace(self.binding, resident_id="wrong"))
        self.assertEqual((self.root / "runtime.sqlite3").read_bytes(), before)

    def test_worker_refuses_browser_session_without_internal_bearer(self):
        app = FastAPI()

        @app.get("/private")
        async def private(request: Request):
            return {"authenticated": request.session.get("authenticated"), "grant": current_household_admission.get()}

        app.add_middleware(SessionMiddleware, secret_key="test-private-signing-key", session_cookie="tamagi_session")
        app.add_middleware(NativeAuthentication, token="z" * 40, session_secret="test-private-signing-key")
        with TestClient(app) as client:
            self.assertEqual(client.get("/private").status_code, 401)
            response = client.get("/private", headers={"X-Tamagi-Worker": "z" * 40, "X-Tamagi-Grant": "trusted-grant"})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["authenticated"])
            self.assertEqual(response.json()["grant"], {"id": "trusted-grant"})
            self.assertEqual(client.get("/private").status_code, 401)

    def test_internal_routes_precede_the_normal_spa_catchall(self):
        app = FastAPI()

        @app.get("/{path:path}")
        async def spa(path: str):
            return {"page": "ordinary TamAGI shell"}

        with patch.dict(os.environ, {"TAMAGI_NATIVE_TOKEN": "z" * 40}):
            configure_app(app)
        with patch("backend.api.chat.get_agent", return_value=self.agent), TestClient(app) as client:
            response = client.get("/_native/ready", headers={"X-Tamagi-Worker": "z" * 40})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["resident_id"], self.actor)
            ordinary = client.get("/map", headers={"X-Tamagi-Worker": "z" * 40})
            self.assertEqual(ordinary.json(), {"page": "ordinary TamAGI shell"})

    def test_installation_removes_control_credentials_from_tool_child_environment(self):
        import subprocess
        import sys

        credentials = {"TAMAGI_NATIVE_TOKEN": "private-worker-token", "TAMAGI_NATIVE_BINDING": "private-binding",
                       "TAMAGI_NATIVE_COORDINATOR_SOCKET": "/private/socket"}
        with patch.dict(os.environ, credentials):
            install_worker(self.runtime, self.agent, self.config)
            child = subprocess.run([sys.executable, "-c",
                "import os; assert not any(k in os.environ for k in ('TAMAGI_NATIVE_TOKEN','TAMAGI_NATIVE_BINDING','TAMAGI_NATIVE_COORDINATOR_SOCKET'))"],
                cwd=self.root, capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIs(self.runtime.household.agent, self.agent)
