"""Resident checkpoints recover private continuity without rewinding peers."""
from __future__ import annotations

from contextlib import closing
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from backend.core.household.native_registry import NativeRegistry
from backend.core.household.resident_recovery import (
    JOURNAL, ResidentArchive, backup_resident, migrate_resident, restore_resident, stopped_household)
from backend.core.runtime.backup import ArchiveError, PortableBackup
from backend.core.runtime.store import RuntimeStore
from backend.core.memory import MemoryEntry, MemoryType

PASSWORD = "disposable resident recovery test password"


def rows(path, query):
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        return db.execute(query).fetchall()


RECOVERY_TERMINAL = {"reply_offer": "closed", "native_invitation_offer": "closed",
                     "conversation_invitation": "expired", "native_grant": "interrupted",
                     "broker_execution": "stopped"}


def seed_recovery_opportunities(store, actors, *, deferred=False):
    """Idle but unfinished canonical authority, with no provider operation."""
    from backend.core.household.broker import InferenceBroker
    broker = InferenceBroker(store, recover=False)
    records = {}
    for actor in actors:
        broker.register_resident(actor)
        exchange, execution = "restore-exchange-" + actor, "restore-execution-" + actor
        broker.open_exchange(exchange, "user")
        broker.open_execution(execution, exchange, actor)
        store.put("native_execution", "restore-binding-" + actor,
                  {"resident_id": actor, "broker_execution_id": execution})
        for kind in ("reply_offer", "native_invitation_offer"):
            store.put(kind, "restore-" + kind + "-" + actor,
                      {"recipient": actor, "status": "deferred" if deferred else "pending", "exchange_id": exchange})
        store.put("conversation_invitation", "restore-conversation_invitation-" + actor,
                  {"participant_id": actor, "status": "pending"})
        store.put("native_grant", "restore-native_grant-" + actor,
                  {"resident_id": actor, "status": "running"})
        records[actor] = {kind: store.get(kind, execution if kind == "broker_execution" else "restore-" + kind + "-" + actor)
                          for kind in RECOVERY_TERMINAL}
    return records


def check_recovery_opportunities(test, store, actor, peers):
    records = {}
    for kind, status in RECOVERY_TERMINAL.items():
        identity = "restore-execution-" + actor if kind == "broker_execution" else "restore-" + kind + "-" + actor
        record = store.get(kind, identity)
        test.assertEqual(record["data"]["status"], status, kind)
        records[kind] = record
    for expected in peers.values():
        for kind, record in expected.items():
            test.assertEqual(store.get(kind, record["id"]), record, "Peer opportunity changed: " + kind)
    return records


class ResidentRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hh-recover-", dir="/tmp")
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "h"
        registry = NativeRegistry.create(self.root, names=["Cedar", "Willow", "River"],
                                         release="recovery-test-release", offline=True)
        self.actors = list(registry.descriptors)
        self.actor, self.willow, self.river = self.actors
        self.roots = {a: Path(d["root"]) for a, d in registry.descriptors.items()}
        self.original_descriptor = dict(registry.descriptors[self.actor])
        for actor in self.actors:
            root = self.roots[actor]
            (root / "workspace/workshop").mkdir(exist_ok=True)
            (root / "workspace/workshop/identity.txt").write_text(actor)
            store = RuntimeStore(root / "runtime.sqlite3")
            entry = MemoryEntry("A memory belonging to " + actor, MemoryType.FACT, id="memory-" + actor)
            store.put("memory", entry.id, actor, entry.to_dict())
            store.close()
        private = self.roots[self.actor]
        (private / "data/history").mkdir(exist_ok=True)
        (private / "data/history/legacy.json").write_text('{"conversation":"retained legacy bytes"}')
        (private / "data/self_model.json").write_text(json.dumps({"version": 2, "nodes": [
            {"id": "private-garden", "node_type": "location", "name": "private garden", "description": "A remembered place"}],
            "edges": []}))
        (private / "workspace/skills/roller").mkdir(parents=True)
        (private / "workspace/skills/roller/main.py").write_text("print('known good dice roller')\n")
        (private / "data/secrets.enc").write_text("DEDICATED_CREDENTIAL_MUST_NOT_COPY")
        (private / "data/.session_secret").write_text("SESSION_MUST_NOT_COPY")
        registry.store.put("conversation", "group", {"participants": ["operator", *self.actors], "title": "Together"})
        registry.store.put("message", "message-1", {"conversation_id": "group", "author": self.willow,
                                                    "content": "A shared recollection", "sequence": 1})
        registry.store.put("conversation", "private-peer", {"participants": [self.willow, self.river]})
        registry.store.put("message", "private-peer-message", {"conversation_id": "private-peer", "content": "NOT_CEDARS_HISTORY"})
        registry.store.put("shared_artifact", "shared-dice", {"version": 1, "title": "Shared dice"})
        registry.store.set_meta("world_clock", {"last_resident_id": self.willow})
        registry.close()
        self.encrypted = self.directory / "cedar.tamagi.gpg"

    def tearDown(self):
        self.temporary.cleanup()

    def backup(self):
        with stopped_household(self.root) as registry:
            return backup_resident(registry, self.actor, self.encrypted, PASSWORD)

    def stage(self):
        archive = ResidentArchive(self.directory / "transfer")
        self.addCleanup(archive.cleanup)
        return archive, archive.stage(self.encrypted, PASSWORD)

    def restore(self, archive, preview):
        with stopped_household(self.root, damaged_resident=self.actor) as registry:
            return restore_resident(registry, archive, preview, self.actor,
                confirmed=True, source_stopped=True, authority="guarded")

    def test_regression_restore_retains_peer_changes_and_current_shared_state(self):
        self.backup()
        private = self.roots[self.actor]
        (private / "workspace/skills/roller/main.py").write_text("BROKEN_PRIVATE_REGRESSION")
        store = RuntimeStore(private / "runtime.sqlite3")
        store.put("memory", "memory-" + self.actor, self.actor,
                  MemoryEntry("regressed memory", MemoryType.FACT, id="memory-" + self.actor).to_dict())
        store.close()
        with stopped_household(self.root) as registry:
            registry.store.put("message", "message-2", {"conversation_id": "group", "author": self.river,
                                                        "content": "After the checkpoint", "sequence": 2})
            registry.store.put("shared_artifact", "shared-dice", {"version": 2, "title": "New shared dice"})
            registry.store.set_meta("world_clock", {"last_resident_id": self.river})
            shared = {k: registry.store.list(k) for k in ("message", "conversation", "shared_artifact")}
            clock = registry.store.meta("world_clock")
        peer_rows = {}
        for actor in (self.willow, self.river):
            store = RuntimeStore(self.roots[actor] / "runtime.sqlite3")
            store.put("memory", "post-checkpoint", actor,
                      MemoryEntry("Keep this new memory", MemoryType.FACT, id="post-checkpoint").to_dict())
            store.close()
            peer_rows[actor] = rows(self.roots[actor] / "runtime.sqlite3", "SELECT * FROM records ORDER BY kind,id")
        archive, preview = self.stage()
        result = self.restore(archive, preview)
        recovered = Path(result["root"])
        self.assertEqual((recovered / "workspace/skills/roller/main.py").read_text(), "print('known good dice roller')\n")
        self.assertEqual((private / "workspace/skills/roller/main.py").read_text(), "BROKEN_PRIVATE_REGRESSION")
        registry = NativeRegistry(self.root)
        try:
            self.assertEqual(list(registry.descriptors), self.actors)
            for kind, expected in shared.items():
                self.assertEqual(registry.store.list(kind), expected)
            self.assertEqual(registry.store.meta("world_clock"), clock)
            self.assertNotEqual(registry.descriptors[self.actor]["installation_id"], self.original_descriptor["installation_id"])
        finally:
            registry.close()
        for actor, expected in peer_rows.items():
            self.assertEqual(rows(self.roots[actor] / "runtime.sqlite3", "SELECT * FROM records ORDER BY kind,id"), expected)
        self.assertIn("A memory belonging to", rows(recovered / "runtime.sqlite3",
            "SELECT payload FROM records WHERE kind='memory'")[0][0])

    def test_complete_private_data_loss_can_be_replaced_without_original_root(self):
        self.backup()
        shutil.rmtree(self.roots[self.actor])
        archive, preview = self.stage()
        result = self.restore(archive, preview)
        recovered = Path(result["root"])
        self.assertEqual((recovered / "data/history/legacy.json").read_text(), '{"conversation":"retained legacy bytes"}')
        self.assertIn("private garden", (recovered / "data/self_model.json").read_text())
        registry = NativeRegistry(self.root)
        registry.close()

    def test_fresh_migration_uses_archive_only_and_does_not_invent_peers(self):
        self.backup()
        shutil.rmtree(self.root)
        archive, preview = self.stage()
        result = migrate_resident(archive, preview, self.directory / "new", release="recovery-test-release",
                                 confirmed=True, source_stopped=True, authority="guarded")
        registry = NativeRegistry(result["root"])
        try:
            self.assertEqual(list(registry.descriptors), [self.actor])
            self.assertEqual(registry.store.list("conversation"), [])
            restored = Path(registry.descriptors[self.actor]["root"])
            config = yaml.safe_load((restored / "config.yaml").read_text())
            self.assertEqual(config["workspace"]["path"], str(restored / "workspace"))
            self.assertEqual(config["history"]["persist_path"], str(restored / "data/history"))
            self.assertEqual(config["self_model"]["data_path"], str(restored / "data/self_model.json"))
            self.assertFalse(config["autonomy"]["enabled"])
            self.assertFalse(config["runtime"]["automatic_deployment"])
            self.assertEqual(config["auth"]["password_hash"], "")
            self.assertEqual(config["llm"]["api_key"], "")
            self.assertFalse((restored / "data/secrets.enc").exists())
            self.assertFalse((restored / "data/.session_secret").exists())
            context = json.loads(next((restored / "data/recovery-evidence").glob("context*.json")).read_text())
            self.assertEqual([r["id"] for r in context["messages"]], ["message-1"])
            self.assertNotIn("NOT_CEDARS_HISTORY", json.dumps(context))
            self.assertEqual(context["conversations"][0]["id"], "group")
            self.assertTrue(context["historical_only"])
        finally:
            registry.close()

    def test_queued_work_and_uncertain_effects_are_not_resumed(self):
        private = RuntimeStore(self.roots[self.actor] / "runtime.sqlite3")
        private.put("interaction", "old-work", self.actor, {"status": "queued"})
        private.put("evaluation_job", "old-job", self.actor, {"status": "queued"})
        private.put("household_outbox", "uncertain-command", self.actor, {"status": "pending", "payload": {"action": "publish"}})
        private.close()
        self.backup()
        archive, preview = self.stage()
        result = self.restore(archive, preview)
        private = RuntimeStore(Path(result["root"]) / "runtime.sqlite3")
        try:
            self.assertEqual(private.get("interaction", "old-work")["data"]["status"], "interrupted")
            self.assertEqual(private.get("evaluation_job", "old-job")["data"]["status"], "interrupted")
            self.assertEqual(private.get("household_outbox", "uncertain-command")["data"]["status"], "outcome_unknown")
        finally:
            private.close()

    def test_running_source_is_rejected_and_confirmation_identity_and_release_are_required(self):
        registry = NativeRegistry(self.root)
        try:
            with self.assertRaisesRegex(ArchiveError, "Stop this household"):
                with stopped_household(self.root):
                    self.fail("Running source acquired")
        finally:
            registry.close()
        for lock_path in (self.root / "ipc" / (self.actor + ".lock"), self.roots[self.actor] / "instance.lock"):
            with lock_path.open("a") as orphan:
                fcntl.flock(orphan, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ArchiveError, "worker is still running"):
                    with stopped_household(self.root):
                        self.fail("Orphan worker was ignored")
        self.backup()
        archive, preview = self.stage()
        with stopped_household(self.root) as registry:
            for changes in ({"confirmed": False}, {"source_stopped": False}, {"actor": self.willow},
                            {"preview": {**preview, "release": "incompatible"}}):
                arguments = dict(archive=archive, preview=preview, actor=self.actor,
                                 confirmed=True, source_stopped=True, authority="guarded")
                arguments.update(changes)
                with self.assertRaises(ValueError):
                    restore_resident(registry, **arguments)
            self.assertEqual(registry.descriptors[self.actor], self.original_descriptor)
        self.assertFalse((self.root / JOURNAL).exists())

    def test_wrong_password_and_corruption_do_not_change_source_or_leave_plaintext(self):
        self.backup()
        original = (self.root / "household.json").read_bytes()
        archive = ResidentArchive(self.directory / "transfer")
        with self.assertRaises(ArchiveError):
            archive.stage(self.encrypted, "wrong disposable password")
        damaged = self.directory / "damaged.gpg"
        damaged.write_bytes(self.encrypted.read_bytes()[:200])
        with self.assertRaises(ArchiveError):
            archive.stage(damaged, PASSWORD)
        self.assertEqual(archive.stages, {})
        self.assertEqual(list(archive.root.iterdir()), [])
        self.assertEqual((self.root / "household.json").read_bytes(), original)

    def test_failure_before_selection_keeps_old_root_and_interrupted_selection_reconciles(self):
        self.backup()
        archive, preview = self.stage()
        with patch("backend.core.household.resident_recovery.sync_tree", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.restore(archive, preview)
        self.assertEqual(json.loads((self.root / "household.json").read_text())["residents"][0], self.original_descriptor)
        self.assertFalse((self.root / JOURNAL).exists())
        self.assertEqual(sorted(p.name for p in (self.root / "residents").iterdir()), ["r0", "r1", "r2"])

        archive, preview = self.stage()
        from backend.core.household import resident_recovery as recovery
        original_write = recovery.atomic_json
        def fail_manifest(path, value):
            if Path(path).name == "household.json":
                raise OSError("interrupted after database commit")
            return original_write(path, value)
        with patch.object(recovery, "atomic_json", side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.restore(archive, preview)
        self.assertTrue((self.root / JOURNAL).exists())
        registry = NativeRegistry(self.root)
        try:
            self.assertNotEqual(registry.descriptors[self.actor]["root"], str(self.roots[self.actor]))
            self.assertEqual(registry.store.get("resident", self.actor)["data"], registry.descriptors[self.actor])
            self.assertEqual(len(registry.store.list("message")), 2)
        finally:
            registry.close()
        self.assertFalse((self.root / JOURNAL).exists())

    def test_interrupted_selection_durably_retires_only_target_opportunities(self):
        self.backup()
        archive, preview = self.stage()
        with stopped_household(self.root) as registry:
            seeded = seed_recovery_opportunities(registry.store, self.actors, deferred=True)
        peers = {actor: seeded[actor] for actor in (self.willow, self.river)}
        from backend.core.household import resident_recovery as recovery
        original_write = recovery.atomic_json
        def fail_manifest(path, value):
            if Path(path).name == "household.json":
                raise OSError("interrupted after database commit")
            return original_write(path, value)
        with patch.object(recovery, "atomic_json", side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.restore(archive, preview)
        self.assertTrue((self.root / JOURNAL).exists())
        # Registry startup replays the exact durable selection without needing
        # browser state or an available model server, and settlement is idempotent.
        registry = NativeRegistry(self.root)
        try:
            current = check_recovery_opportunities(self, registry.store, self.actor, peers)
            self.assertTrue(all(record["revision"] == 2 for record in current.values()))
            self.assertEqual(len([event for event in registry.store.events()
                                  if event["kind"] == "broker_execution_stopped"]), 1)
        finally:
            registry.close()
        self.assertFalse((self.root / JOURNAL).exists())

    def test_legacy_portable_archive_migrates_as_same_single_resident(self):
        root = self.roots[self.actor]
        store = RuntimeStore(root / "runtime.sqlite3")
        try:
            runtime = SimpleNamespace(store=store, actor_id=self.actor, workspace=root / "workspace",
                                      source_root=Path(__file__).resolve().parents[1])
            legacy = PortableBackup(runtime, self.directory / "legacy-export")
            exported = legacy.export(PASSWORD, yaml.safe_load((root / "config.yaml").read_text()))
        finally:
            store.close()
        archive = ResidentArchive(self.directory / "legacy-preview")
        self.addCleanup(archive.cleanup)
        preview = archive.stage(exported["path"], PASSWORD)
        self.assertEqual(preview["format"], PortableBackup.FORMAT)
        result = migrate_resident(archive, preview, self.directory / "legacy", release="legacy-compatible",
                                 confirmed=True, source_stopped=True, authority="guarded")
        registry = NativeRegistry(result["root"])
        self.assertEqual(list(registry.descriptors), [self.actor])
        registry.close()

    def test_later_backup_keeps_prior_household_context_after_migration(self):
        self.backup()
        archive, preview = self.stage()
        result = migrate_resident(archive, preview, self.directory / "first-migration", release="recovery-test-release",
                                 confirmed=True, source_stopped=True, authority="guarded")
        later = self.directory / "later.gpg"
        with stopped_household(result["root"]) as registry:
            backup_resident(registry, self.actor, later, PASSWORD)
        later_archive = ResidentArchive(self.directory / "later-preview")
        try:
            preview = later_archive.stage(later, PASSWORD)
            second = migrate_resident(later_archive, preview, self.directory / "second-migration", release="recovery-test-release",
                                     confirmed=True, source_stopped=True, authority="guarded")
            registry = NativeRegistry(second["root"])
            try:
                root = Path(registry.descriptors[self.actor]["root"])
                contexts = [json.loads(path.read_text()) for path in (root / "data/recovery-evidence").glob("context*.json")]
                self.assertEqual(len(contexts), 2)
                self.assertEqual(sorted(len(c["conversations"]) for c in contexts), [0, 1])
                self.assertEqual([m["id"] for c in contexts for m in c["messages"]], ["message-1"])
                self.assertEqual(list(registry.descriptors), [self.actor])
            finally:
                registry.close()
        finally:
            later_archive.cleanup()

    def test_all_three_residents_can_be_checkpointed_and_restored_independently(self):
        for actor in self.actors:
            encrypted = self.directory / (actor + ".gpg")
            with stopped_household(self.root) as registry:
                backup_resident(registry, actor, encrypted, PASSWORD)
            archive = ResidentArchive(self.directory / (actor + "-preview"))
            try:
                preview = archive.stage(encrypted, PASSWORD)
                with stopped_household(self.root, damaged_resident=actor) as registry:
                    result = restore_resident(registry, archive, preview, actor,
                        confirmed=True, source_stopped=True, authority="guarded")
                    self.assertEqual((Path(result["root"]) / "workspace/workshop/identity.txt").read_text(), actor)
                    self.assertEqual(len(registry.descriptors), 3)
            finally:
                archive.cleanup()
        registry = NativeRegistry(self.root)
        self.assertEqual(list(registry.descriptors), self.actors)
        registry.close()


class ResidentRecoveryWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_restored_workers_hydrate_distinct_residents_without_model_or_replay(self):
        import httpx
        from backend.config import TamAGIConfig
        from backend.core.household.native_app import create_household_app
        from test_household_native_app import DeterministicProvider

        fixture = ResidentRecoveryTests()
        fixture.setUp()
        try:
            fixture.backup()
            archive, preview = fixture.stage()
            try:
                result = fixture.restore(archive, preview)
            finally:
                archive.cleanup()
            registry = NativeRegistry(fixture.root)
            config = TamAGIConfig.model_validate(yaml.safe_load((fixture.root / "config.yaml").read_text()))
            provider = DeterministicProvider()
            app = create_household_app(config, registry=registry, provider=provider)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as client:
                    response = await client.post("/api/auth/setup", json={"password": "disposable-recovery-worker-password"})
                    self.assertEqual(response.status_code, 200, response.text)
                    for actor, name in zip(fixture.actors, ["Cedar", "Willow", "River"]):
                        response = await client.get("/api/state", headers={"X-Tamagi-Resident": actor})
                        self.assertEqual(response.status_code, 200,
                            response.text + app.state.hosts[actor].log_tail.decode(errors="replace")[-6000:])
                        self.assertEqual(response.json()["state"]["name"], name)
                        response = await client.get("/api/inventory/file?path=identity.txt&preview=true",
                                                    headers={"X-Tamagi-Resident": actor})
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(response.text, actor)
                    self.assertEqual(app.state.hosts[fixture.actor].descriptor["root"], result["root"])
                    self.assertEqual(provider.calls, [])
                    self.assertEqual(app.state.interactions.records(), [])
            # The migration has no access to the original household directories.
            archive, preview = fixture.stage()
            try:
                migrated = migrate_resident(archive, preview, fixture.directory / "migrated",
                    release="recovery-test-release", confirmed=True, source_stopped=True, authority="guarded")
            finally:
                archive.cleanup()
            shutil.rmtree(fixture.root)
            registry = NativeRegistry(migrated["root"])
            config = TamAGIConfig.model_validate(yaml.safe_load((Path(migrated["root"]) / "config.yaml").read_text()))
            config.household.provider_mode = "offline"
            provider = DeterministicProvider()
            app = create_household_app(config, registry=registry, provider=provider)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as client:
                    await client.post("/api/auth/setup", json={"password": "disposable-migrated-worker-password"})
                    response = await client.get("/api/state")
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["state"]["name"], "Cedar")
                    self.assertEqual(response.json()["memory"]["total_memories"], 1)
                    response = await client.get("/api/inventory/file?path=identity.txt&preview=true")
                    self.assertEqual(response.text, fixture.actor)
                    self.assertEqual(provider.calls, [])
                    self.assertEqual(list(app.state.hosts), [fixture.actor])
        finally:
            fixture.tearDown()
