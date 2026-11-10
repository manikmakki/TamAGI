"""Owner-created residents start fresh without rewriting an existing resident."""
from __future__ import annotations

import asyncio
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from backend.core.household.broker import InferenceBroker
from backend.core.household.contracts import Conflict, ProtocolError
from backend.core.household import native_creation as creation
from backend.core.household.native_registry import NativeRegistry
from backend.core.household.native_world import NativeWorld
from backend.core.runtime import RuntimeStore


def private_records(root):
    with closing(sqlite3.connect(f"file:{root / 'runtime.sqlite3'}?mode=ro", uri=True)) as database:
        return tuple(database.iterdump())


class HouseholdCreationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="household-create-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name) / "h"
        self.registry = NativeRegistry.create(self.home, names=["Ember"], release="creation-tests", offline=True)
        self.addCleanup(lambda: self.registry.close())
        self.ember = self.registry.manifest["default_resident_id"]
        self.ember_root = Path(self.registry.descriptors[self.ember]["root"])
        store = RuntimeStore(self.ember_root / "runtime.sqlite3")
        store.put("memory", "ember-private", self.ember, {"content": "Only Ember's private record"})
        store.close()
        self.registry.store.put("conversation", "existing-thread", {"participants": ["operator", self.ember]})
        self.registry.store.put("message", "existing-message", {"conversation_id": "existing-thread", "content": "Retained shared record"})
        self.registry.store.set_meta("native_world_schedule", {"next_at": 9999999999, "last_resident_id": self.ember})
        self.request = {"request_id": "create-first", "name": "Nova", "avatar": "echo", "world_setting": "A quiet observatory"}
        self.make_service()

    def make_service(self):
        self.model = SimpleNamespace(broker=InferenceBroker(self.registry.store))
        self.world = NativeWorld(self.registry, self.model)
        self.hosts = {self.ember: SimpleNamespace(process=SimpleNamespace(returncode=None))}
        self.maintenance_calls = []

        async def run(actor, action, work):
            self.maintenance_calls.append((actor, action))
            return work()

        self.recovery = SimpleNamespace(busy=False, closing=False, run=run)
        self.service = creation.NativeCreation(self.registry, self.recovery, self.model, self.hosts,
            self.world, self.home / "ipc/c.sock")

    def test_schema_excludes_operator_questions_and_has_no_effects(self):
        before = tuple(self.registry.store.db.iterdump())
        schema = self.service.schema()
        fields = [step["field"] for step in schema["steps"]]
        self.assertEqual(fields, ["name", "creature", "vibe", "emoji", "values", "world_setting"])
        self.assertEqual({avatar["id"] for avatar in schema["avatars"]}, {"vex", "echo", "ember"})
        self.assertEqual(schema["limit"], 16)
        self.assertEqual(schema["defaults"]["name"], "")
        schema["defaults"]["values"].clear()
        self.assertEqual(self.service.schema()["defaults"]["values"], ["curiosity", "honesty", "efficiency"])
        self.assertEqual(tuple(self.registry.store.db.iterdump()), before)

    def test_creation_preserves_existing_private_and_shared_state(self):
        private = private_records(self.ember_root)
        descriptor = deepcopy(self.registry.descriptors[self.ember])
        identity = (self.ember_root / "workspace/IDENTITY.md").read_bytes()
        messages = self.registry.store.list("message")
        clock = self.registry.store.meta("native_world_schedule")
        result = asyncio.run(self.service.create(self.request))
        actor = result["resident"]["id"]
        root = Path(self.registry.descriptors[actor]["root"])
        self.assertEqual(result["status"], "created")
        self.assertNotEqual(actor, self.ember)
        self.assertEqual(self.registry.manifest["default_resident_id"], self.ember)
        self.assertEqual(self.registry.descriptors[self.ember], descriptor)
        self.assertEqual(private_records(self.ember_root), private)
        self.assertEqual((self.ember_root / "workspace/IDENTITY.md").read_bytes(), identity)
        self.assertEqual(self.registry.store.list("message"), messages)
        self.assertEqual(self.registry.store.meta("native_world_schedule"), clock)
        config = yaml.safe_load((root / "config.yaml").read_text())
        self.assertEqual(config["runtime"]["authority"], "guarded")
        self.assertFalse(config["runtime"]["automatic_deployment"])
        self.assertFalse(config["runtime"]["index_projection"])
        self.assertFalse(config["autonomy"]["enabled"])
        self.assertFalse(config["world_thread"]["enabled"])
        self.assertEqual(config["mcp"]["servers"], [])
        self.assertEqual(config["llm"]["api_key"], "")
        self.assertEqual(config["auth"]["password_hash"], "")
        self.assertEqual(json.loads((root / "data/sprite_config.json").read_text()), {"avatar": "echo"})
        self.assertIn("Nova", (root / "workspace/IDENTITY.md").read_text())
        self.assertIn(self.registry.manifest["operator"]["name"], (root / "workspace/USER.md").read_text())
        store = RuntimeStore(root / "runtime.sqlite3")
        try:
            for kind in ("memory", "conversation", "companion_thread", "growth_recognition", "artifact"):
                self.assertEqual(store.list(kind), [], kind)
            self.assertEqual(store.meta("household_id"), self.registry.household_id)
            self.assertEqual(store.get("actor", actor)["data"]["name"], "Nova")
            self.assertEqual(store.get("personality", "primary")["data"]["name"], "Nova")
        finally:
            store.close()
        mapping = self.registry.store.get("world_import", actor + ":" + actor)["data"]["mapping"]
        home = self.registry.store.get("world_object", mapping["home"])["data"]
        self.assertEqual(home["visibility"], "private")
        self.assertEqual(home["attributes"]["description"], "A quiet observatory")
        self.assertFalse((self.home / creation.JOURNAL).exists())

    def test_repeated_request_has_one_identity_and_changed_input_conflicts(self):
        first = asyncio.run(self.service.create(self.request))
        second = asyncio.run(self.service.create(self.request))
        self.assertEqual(first["resident"]["id"], second["resident"]["id"])
        self.assertEqual(len(self.registry.descriptors), 2)
        self.assertEqual(len([event for event in self.registry.store.events() if event["kind"] == "resident_created"]), 1)
        with self.assertRaises(Conflict):
            asyncio.run(self.service.create({**self.request, "name": "Different"}))

    def test_validation_and_capacity_fail_before_maintenance(self):
        self.assertEqual(creation.validate_request({**self.request, "values": []})["values"], [])
        invalid = [[], {"request_id": "id", "name": ""}, {**self.request, "avatar": []},
            {**self.request, "avatar": "../custom"}, {**self.request, "values": ["unknown"]},
            {**self.request, "vibe": "custom"}, {**self.request, "name": "bad\nname"},
            {**self.request, "creature": "x" * 161}, {**self.request, "operator_name": "Other"}]
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(ProtocolError):
                asyncio.run(self.service.create(body))
        with self.assertRaises(Conflict):
            asyncio.run(self.service.create({**self.request, "name": "eMbEr"}))
        for index in range(15):
            self.registry.descriptors[f"synthetic-{index}"] = {"name": f"Synthetic {index}"}
        with self.assertRaises(Conflict):
            asyncio.run(self.service.create(self.request))
        self.assertEqual(self.maintenance_calls, [])

    def test_failure_before_journal_removes_only_unselected_fresh_root(self):
        manifest = (self.home / "household.json").read_bytes()
        roots = set((self.home / "residents").iterdir())
        with patch.object(creation, "sync_tree", side_effect=OSError("Disposable preparation failure")):
            with self.assertRaises(OSError):
                asyncio.run(self.service.create(self.request))
        self.assertEqual((self.home / "household.json").read_bytes(), manifest)
        self.assertEqual(set((self.home / "residents").iterdir()), roots)
        self.assertEqual(len(self.registry.store.list("resident")), 1)
        self.assertFalse((self.home / creation.JOURNAL).exists())

    def test_interrupted_database_manifest_selection_reconciles_on_startup(self):
        atomic = creation.atomic_json

        def fail_manifest(path, value):
            if Path(path) == self.home / "household.json":
                raise OSError("Disposable manifest failure after commit")
            return atomic(path, value)

        with patch.object(creation, "atomic_json", side_effect=fail_manifest):
            with self.assertRaises(OSError):
                asyncio.run(self.service.create(self.request))
        self.assertTrue((self.home / creation.JOURNAL).exists())
        self.assertEqual(len(self.registry.store.list("resident")), 2)
        self.assertEqual(len(self.registry.descriptors), 1)
        self.registry.close()
        self.registry = NativeRegistry(self.home)
        self.assertEqual(len(self.registry.descriptors), 2)
        self.assertFalse((self.home / creation.JOURNAL).exists())
        self.make_service()
        retried = asyncio.run(self.service.create(self.request))
        self.assertEqual(retried["resident"]["name"], "Nova")
        self.assertEqual(len(self.registry.descriptors), 2)
        self.assertEqual(self.registry.manifest["default_resident_id"], self.ember)

    def test_recovery_journal_blocks_creation_without_preparing_any_root(self):
        (self.home / "resident-recovery.json").write_text("{}")
        with self.assertRaises(Conflict):
            asyncio.run(self.service.create(self.request))
        self.assertEqual(len(list((self.home / "residents").iterdir())), 1)
        self.assertEqual(self.maintenance_calls, [])


class NativeCreationAppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from test_household_native_app import NativeAppTests
        self.resident_names = ["Ember", "Willow"]
        await NativeAppTests.asyncSetUp(self)

    async def asyncTearDown(self):
        from test_household_native_app import NativeAppTests
        await NativeAppTests.asyncTearDown(self)

    async def request(self, *args, **kwargs):
        from test_household_native_app import NativeAppTests
        return await NativeAppTests.request(self, *args, **kwargs)

    async def test_owner_creation_starts_fresh_worker_and_survives_coordinator_restart(self):
        old = deepcopy(self.registry.descriptors[self.first])
        result = await self.request("POST", "/api/household/residents", expected=201, json={
            "request_id": "new-resident", "name": "Nova", "avatar": "echo", "vibe": "calm_thoughtful"})
        actor = result["resident"]["id"]
        self.assertTrue(result["workers_available"])
        self.assertEqual(self.registry.descriptors[self.first], old)
        state = await self.request("GET", "/api/state", actor=actor)
        self.assertEqual(state["state"]["name"], "Nova")
        self.assertEqual(state["memory"]["total_memories"], 0)
        onboarding = await self.request("GET", "/api/onboarding/state", actor=actor)
        self.assertFalse(onboarding["needs_onboarding"])
        self.assertEqual(self.provider.calls, [])
        self.assertIn(actor, self.app.state.hosts)
        root = Path(self.registry.descriptors[actor]["root"])
        self.assertEqual(json.loads((root / "instance.json").read_text())["resident_id"], actor)
        repeated = await self.request("POST", "/api/household/residents", expected=201, json={
            "request_id": "new-resident", "name": "Nova", "avatar": "echo", "vibe": "calm_thoughtful"})
        self.assertEqual(repeated["resident"]["id"], actor)
        self.assertEqual(len(self.app.state.hosts), 3)
        await self.lifespan.__aexit__(None, None, None)
        self.registry = NativeRegistry(self.root)
        from backend.config import TamAGIConfig
        from backend.core.household.native_app import create_household_app
        from test_household_native_app import DeterministicProvider
        config = TamAGIConfig.model_validate(yaml.safe_load((self.root / "config.yaml").read_text()))
        self.provider = DeterministicProvider()
        self.app = create_household_app(config, registry=self.registry, provider=self.provider)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.assertEqual(self.app.state.hosts[actor].ready["resident_id"], actor)
        self.assertEqual(self.provider.calls, [])

    async def test_retry_resumes_same_created_resident_after_worker_start_failure(self):
        from backend.core.household.native_host import NativeHost
        original = NativeHost.start
        failed = False

        async def fail_new_once(host):
            nonlocal failed
            if host.actor_id not in {self.first, self.second} and not failed:
                failed = True
                raise RuntimeError("Disposable new worker failure")
            return await original(host)

        body = {"request_id": "retry-start", "name": "Nova", "avatar": "vex"}
        with patch.object(NativeHost, "start", fail_new_once):
            first = await self.request("POST", "/api/household/residents", expected=201, json=body)
            self.assertFalse(first["workers_available"])
            second = await self.request("POST", "/api/household/residents", expected=201, json=body)
        self.assertTrue(second["workers_available"])
        self.assertEqual(first["resident"]["id"], second["resident"]["id"])
        self.assertEqual(len(self.registry.descriptors), 3)
        self.assertEqual(self.provider.calls, [])

    async def test_creation_requires_owner_session_and_same_origin(self):
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url="http://testserver") as anonymous:
            response = await anonymous.post("/api/household/residents", json={"request_id": "forbidden", "name": "Nova"})
            self.assertEqual(response.status_code, 401)
        await self.request("POST", "/api/household/residents", expected=403,
            headers={"Origin": "https://another.example"}, json={"request_id": "forbidden", "name": "Nova"})
        self.assertEqual(len(self.registry.descriptors), 2)
        self.assertEqual(self.provider.calls, [])


if __name__ == "__main__":
    unittest.main()
