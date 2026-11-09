"""Real-worker browser APIs for encrypted resident checkpoints and replacement."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.core.household import browser_recovery
from backend.core.household.resident_recovery import ResidentArchive
from backend.core.runtime.store import RuntimeStore
import test_household_native_app as fixture
from test_resident_recovery import check_recovery_opportunities, seed_recovery_opportunities

PASSWORD = "disposable browser backup password"


class BrowserRecoveryTests(unittest.IsolatedAsyncioTestCase):
    resident_names = ["Cedar", "Willow", "River"]
    asyncSetUp = fixture.NativeAppTests.asyncSetUp
    asyncTearDown = fixture.NativeAppTests.asyncTearDown
    request = fixture.NativeAppTests.request
    wait_until = fixture.NativeAppTests.wait_until
    completed = fixture.NativeAppTests.completed

    def pids(self):
        return [h.process.pid for h in self.app.state.hosts.values()]

    async def backup(self, actor=None):
        actor = actor or self.first
        result = await self.request("POST", "/api/backups", actor=actor, json={"password": PASSWORD})
        self.assertTrue(result["workers_available"])
        response = await self.http.get(f"/r/{actor}" + result["download"])
        self.assertEqual(response.status_code, 200, response.text[:100] if response.status_code != 200 else "")
        self.assertEqual(response.headers["cache-control"], "no-store")
        return result, response.content

    async def preview(self, data, actor=None, password=PASSWORD, expected=200):
        return await self.request("POST", "/api/restores/preview", actor=actor or self.first, expected=expected,
            files={"archive": ("resident.gpg", data, "application/octet-stream")}, data={"password": password})

    async def commit(self, identity, actor=None, expected=200, **body):
        return await self.request("POST", f"/api/restores/{identity}/commit", actor=actor or self.first, expected=expected,
            json={"confirmed": True, "source_stopped": True, "authority": "guarded", **body})

    async def test_browser_round_trip_restores_only_selected_private_state(self):
        root = Path(self.registry.descriptors[self.first]["root"])
        peer = Path(self.registry.descriptors[self.second]["root"])
        (root / "workspace/workshop/proof.txt").write_text("KNOWN_GOOD_PRIVATE_BYTES")
        before = self.pids()
        result, data = await self.backup()
        self.assertEqual(result["resident_id"], self.first)
        self.assertNotEqual(before, self.pids())
        preview_pids = self.pids()
        preview = await self.preview(data)
        another = await self.preview(data)
        self.assertEqual(preview["resident_id"], self.first)
        self.assertIn("Private changes", preview["continuity_gap"])
        self.assertEqual(preview_pids, self.pids(), "preview must not stop workers")
        (root / "workspace/workshop/proof.txt").write_text("DISPOSABLE_REGRESSION")
        (peer / "workspace/workshop/later.txt").write_text("KEEP_LATER_PEER_FILE")
        await self.request("POST", "/api/knowledge", actor=self.second, json={"content": "Keep this later peer memory"})
        self.app.state.model.broker.open_exchange("recovery-shared-exchange", "user")
        shared = self.app.state.interactions.conversations.create("operator", [self.first, self.second],
            "recovery-shared-thread", group=True, opening="Shared state created after the private checkpoint",
            lineage={"exchange_id": "recovery-shared-exchange", "origin": "user"})
        thread = shared["conversation"]["id"]
        messages = self.registry.store.list("message")
        clock = self.registry.store.meta("native_world_schedule")
        old_installation = self.registry.descriptors[self.first]["installation_id"]
        await self.commit(preview["id"], expected=400, confirmed=False)
        self.assertEqual(self.registry.descriptors[self.first]["installation_id"], old_installation)
        restored = await self.commit(preview["id"])
        self.assertEqual(restored["status"], "restored")
        self.assertTrue(restored["workers_available"])
        selected = Path(self.registry.descriptors[self.first]["root"])
        self.assertNotEqual(selected, root)
        self.assertEqual((selected / "workspace/workshop/proof.txt").read_text(), "KNOWN_GOOD_PRIVATE_BYTES")
        self.assertEqual((root / "workspace/workshop/proof.txt").read_text(), "DISPOSABLE_REGRESSION")
        self.assertEqual((peer / "workspace/workshop/later.txt").read_text(), "KEEP_LATER_PEER_FILE")
        self.assertEqual(self.registry.store.list("message"), messages)
        self.assertEqual(self.registry.store.meta("native_world_schedule"), clock)
        offers = self.registry.store.list("reply_offer")
        self.assertTrue(any(r["data"]["recipient"] == self.first and r["data"]["status"] == "closed" for r in offers))
        self.assertTrue(any(r["data"]["recipient"] == self.second and r["data"]["status"] == "pending" for r in offers))
        history = await self.request("GET", f"/api/conversations/{thread}")
        self.assertEqual(history["messages"][0]["content"], "Shared state created after the private checkpoint")
        state = await self.request("GET", "/api/state", actor=self.first)
        self.assertEqual(state["state"]["name"], "Cedar")
        peer_state = await self.request("GET", "/api/state", actor=self.second)
        self.assertEqual(peer_state["memory"]["total_memories"], 1)
        await self.commit(preview["id"], expected=404)
        await self.commit(another["id"], expected=409)
        self.assertEqual(self.provider.calls, [])

    async def test_busy_resident_is_not_interrupted_and_scope_confirmation_is_enforced(self):
        self.provider.gate.clear()
        work = await self.request("POST", "/api/chat/requests", actor=self.first, expected=202,
            json={"message": "An ongoing conversation"})
        await self.provider.started.wait()
        before = self.pids()
        await self.request("POST", "/api/backups", actor=self.second, expected=409, json={"password": PASSWORD})
        self.assertEqual(before, self.pids())
        self.assertFalse(self.app.state.interactions.maintenance)
        self.provider.gate.set()
        await self.completed(work["work_id"])
        private = RuntimeStore(Path(self.registry.descriptors[self.first]["root"]) / "runtime.sqlite3")
        private.put("evaluation_job", "queued-private-evaluation", self.first, {"status": "queued"})
        await self.request("POST", "/api/backups", actor=self.second, expected=409, json={"password": PASSWORD})
        self.assertEqual(before, self.pids(), "queued private work must not be discarded by a checkpoint")
        self.assertEqual(private.get("evaluation_job", "queued-private-evaluation")["data"]["status"], "queued")
        private.delete("evaluation_job", "queued-private-evaluation", self.first)
        private.close()
        result, data = await self.backup()
        await self.request("GET", result["download"], actor=self.second, expected=403)
        before = self.pids()
        await self.preview(data, actor=self.second, expected=400)
        await self.preview(data, password="wrong disposable password", expected=400)
        self.assertEqual(before, self.pids())
        preview = await self.preview(data)
        await self.commit(preview["id"], actor=self.second, expected=403)
        await self.commit(preview["id"], source_stopped=False, expected=400)
        await self.request("DELETE", f"/api/restores/{preview['id']}")
        await self.commit(preview["id"], expected=404)
        self.assertEqual(self.app.state.recovery.previews, {})

    async def test_disconnect_does_not_abandon_pause_and_mutations_are_blocked(self):
        entered, release = threading.Event(), threading.Event()
        export = browser_recovery.backup_resident
        def held(*args, **kwargs):
            entered.set()
            if not release.wait(15):
                raise RuntimeError("Test gate timed out")
            return export(*args, **kwargs)
        with patch.object(browser_recovery, "backup_resident", side_effect=held):
            request = asyncio.create_task(self.http.post("/api/backups", json={"password": PASSWORD}))
            try:
                await self.wait_until(entered.is_set)
                status = await self.request("GET", "/api/recovery/status")
                self.assertTrue(status["busy"])
                self.assertEqual(status["phase"], "backing_up")
                clock = self.registry.store.meta("native_world_schedule")
                tick = asyncio.create_task(self.app.state.world_scheduler.opportunity(manual=True))
                await asyncio.sleep(.05)
                self.assertFalse(tick.done(), "scheduler must share the maintenance boundary")
                tick.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await tick
                self.assertEqual(self.registry.store.meta("native_world_schedule"), clock)
                await self.request("POST", "/api/chat/requests", expected=503, json={"message": "Should not be admitted"})
                await self.request("POST", f"/r/{self.second}/api/knowledge", expected=503, json={"content": "Must not write"})
                await self.request("POST", "/api/backups", expected=409, json={"password": PASSWORD})
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request
            finally:
                release.set()
            await self.wait_until(lambda: not self.app.state.recovery.busy)
        for actor in self.registry.descriptors:
            await self.request("GET", "/api/state", actor=actor)
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.app.state.interactions.records(), [])
        self.assertEqual(len(self.app.state.recovery.downloads), 1, "disconnected exports must retain expiry bookkeeping")

    async def test_failed_restart_exposes_resume_control_and_retries(self):
        host = self.app.state.hosts[self.first]
        with patch.object(host, "start", side_effect=RuntimeError("Disposable startup failure")):
            result = await self.request("POST", "/api/backups", json={"password": PASSWORD})
        self.assertFalse(result["workers_available"])
        status = await self.request("GET", "/api/recovery/status")
        self.assertEqual(status["unavailable_residents"], [self.first])
        result = await self.request("POST", "/api/recovery/resume")
        self.assertTrue(result["workers_available"])
        await self.request("GET", "/api/state", actor=self.first)

    async def test_recovery_requires_auth_and_previews_expire_without_pausing(self):
        result, data = await self.backup()
        preview = await self.preview(data)
        before = self.pids()
        item = self.app.state.recovery.previews[preview["id"]]
        stage_root = item["root"]
        item["created"] -= 1801
        await self.request("GET", "/api/recovery/status")
        self.assertFalse(stage_root.exists())
        await self.commit(preview["id"], expected=404)
        self.assertEqual(before, self.pids())
        entered, release = threading.Event(), threading.Event()
        original = ResidentArchive.stage
        def held(archive, *args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Preview test gate timed out")
            return original(archive, *args, **kwargs)
        with patch.object(ResidentArchive, "stage", new=held):
            pending = asyncio.create_task(self.http.post("/api/restores/preview",
                files={"archive": ("resident.gpg", data, "application/octet-stream")}, data={"password": PASSWORD}))
            try:
                await self.wait_until(entered.is_set)
                pending.cancel()
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            await self.wait_until(lambda: not self.app.state.recovery.preview_tasks)
        self.assertEqual(list(self.app.state.recovery.root.glob("preview-*")), [])
        await self.request("POST", "/api/auth/logout")
        await self.request("GET", result["download"], expected=401)
        await self.request("POST", "/api/backups", expected=401, json={"password": PASSWORD})
        await self.preview(data, expected=401)

    async def test_incomplete_selection_stays_paused_until_resume_reconciles_it(self):
        from backend.core.household import resident_recovery
        from backend.core.household.resident_recovery import JOURNAL
        _, data = await self.backup()
        preview = await self.preview(data)
        original = resident_recovery.atomic_json
        old_root = self.registry.descriptors[self.first]["root"]
        seeded = seed_recovery_opportunities(self.registry.store, list(self.registry.descriptors))
        peers = {actor: records for actor, records in seeded.items() if actor != self.first}
        def failed_manifest(path, value):
            if Path(path).name == "household.json":
                raise OSError("Disposable disk failure")
            return original(path, value)
        with patch.object(resident_recovery, "atomic_json", side_effect=failed_manifest):
            await self.commit(preview["id"], expected=507)
        self.assertTrue((self.root / JOURNAL).exists())
        self.assertTrue(self.app.state.recovery.blocked)
        settled = check_recovery_opportunities(self, self.registry.store, self.first, peers)
        self.assertTrue(all(h.process.returncode is not None for h in self.app.state.hosts.values()))
        await self.request("GET", "/api/state", expected=503)
        await self.request("POST", "/api/backups", expected=409, json={"password": PASSWORD})
        result = await self.request("POST", "/api/recovery/resume")
        self.assertTrue(result["workers_available"])
        self.assertFalse(self.app.state.recovery.blocked)
        self.assertFalse((self.root / JOURNAL).exists())
        self.assertNotEqual(self.registry.descriptors[self.first]["root"], old_root)
        self.assertEqual(self.registry.store.get("resident", self.first)["data"], self.registry.descriptors[self.first])
        self.assertEqual(check_recovery_opportunities(self, self.registry.store, self.first, peers), settled,
                         "Retry must not repeat already-durable settlement")
        await self.request("GET", "/api/state")
        self.assertEqual(self.provider.calls, [])


class RecoveryStagingTests(unittest.IsolatedAsyncioTestCase):
    async def test_expired_previews_and_downloads_are_removed_without_http_requests(self):
        with tempfile.TemporaryDirectory(prefix="recovery-expiry-", dir="/tmp") as directory:
            root = Path(directory) / "home"
            root.mkdir()
            registry = SimpleNamespace(root=root, household_id="test-household")
            service = browser_recovery.BrowserRecovery(registry, None, None, {})
            preview = service.root / "preview-expired"
            preview.mkdir()
            (preview / "private.json").write_text("DISPOSABLE_PLAINTEXT")
            download = service.root / "expired.gpg"
            download.write_bytes(b"disposable encrypted-download fixture")
            service.previews["expired"] = {"archive": SimpleNamespace(cleanup=lambda: None), "root": preview, "created": 0}
            service.downloads["expired"] = {"path": download, "created": 0}
            service.EXPIRY_INTERVAL = .01
            service.start()
            try:
                async with asyncio.timeout(2):
                    while preview.exists() or download.exists():
                        await asyncio.sleep(.01)
                self.assertEqual(service.previews, {})
                self.assertEqual(service.downloads, {})
            finally:
                await service.close()

    async def test_owned_plaintext_previews_are_cleaned_on_next_start(self):
        with tempfile.TemporaryDirectory(prefix="recovery-staging-", dir="/tmp") as directory:
            root = Path(directory) / "home"
            root.mkdir()
            registry = SimpleNamespace(root=root, household_id="test-household")
            staging = root.parent / ".home-browser-recovery"
            staging.mkdir()
            owner = {"household_id": registry.household_id, "root": str(root)}
            (staging / "owner.json").write_text(json.dumps(owner))
            leftover = staging / "preview-interrupted"
            leftover.mkdir()
            (leftover / "private.json").write_text("DISPOSABLE_PLAINTEXT")
            service = browser_recovery.BrowserRecovery(registry, None, None, {})
            self.assertFalse(leftover.exists())
            self.assertEqual([p.name for p in staging.iterdir()], ["owner.json"])
            await service.close()
