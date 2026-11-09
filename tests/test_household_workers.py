"""Real subprocess fixture workers never reuse another resident's process globals."""

import asyncio
import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core.household.contracts import AccessDenied, Conflict, NotFound, ProtocolError, MAX_FRAME_BYTES
from backend.core.household.registry import SandboxRegistry, validate_binding
from backend.core.household.transport import ResidentProcess, decode_frame, encode_frame, worker_environment


class HouseholdWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.registry = SandboxRegistry.create("fixture-release", operator_name="Human")
        self.root = self.registry.root
        self.workers = []

    async def asyncTearDown(self):
        await asyncio.gather(*(worker.close() for worker in self.workers))
        self.registry.close()
        shutil.rmtree(self.root)

    async def worker(self, descriptor):
        worker = ResidentProcess(self.registry.binding(descriptor["resident_id"]))
        self.workers.append(worker)
        await worker.start()
        return worker

    async def test_three_processes_keep_distinct_identity_memory_and_cwd_across_restart(self):
        original_cwd = os.getcwd()
        descriptors = [self.registry.provision(f"Fixture {i}", seed=f"private-distinct-{i}") for i in range(3)]
        workers = [await self.worker(descriptor) for descriptor in descriptors]
        self.assertEqual(len({worker.process.pid for worker in workers}), 3)
        observations = await asyncio.gather(*(worker.request("inspect") for worker in workers))
        for i, state in enumerate(observations):
            self.assertEqual(state["resident_id"], descriptors[i]["resident_id"])
            self.assertEqual(state["cwd"], descriptors[i]["root"])
            self.assertEqual(state["identities"][0]["text"], f"private-distinct-{i}")
            self.assertEqual(state["memories"][0]["text"], f"private-distinct-{i}")
            self.assertFalse(state["generation_enabled"])
            encoded = json.dumps(state)
            for j in range(3):
                if j != i:
                    self.assertNotIn(f"private-distinct-{j}", encoded)
        self.assertEqual(os.getcwd(), original_cwd)
        await workers[0].close()
        restarted = await self.worker(descriptors[0])
        self.assertEqual((await restarted.request("read_identity"))["text"], "private-distinct-0")
        self.assertEqual(self.registry.resolve()["resident_id"], descriptors[0]["resident_id"])
        self.registry.resolve(descriptors[2]["resident_id"])
        self.assertEqual(self.registry.resolve()["resident_id"], descriptors[0]["resident_id"])
        with self.assertRaises(NotFound):
            self.registry.resolve("unknown-must-not-fall-back")

    async def test_conflicting_ownership_and_release_rejected_without_switching(self):
        descriptor = self.registry.provision("One")
        worker = await self.worker(descriptor)
        duplicate = ResidentProcess(self.registry.binding())
        self.workers.append(duplicate)
        with self.assertRaises(ProtocolError):
            await duplicate.start()
        self.assertEqual((await worker.request("inspect"))["resident_id"], descriptor["resident_id"])
        with self.assertRaises((BlockingIOError, Conflict)):
            SandboxRegistry(self.root, release="fixture-release")
        with self.assertRaises(Conflict):
            SandboxRegistry(self.root, release="other-release")

    async def test_wrong_or_missing_root_is_checked_before_database_creation(self):
        descriptor = self.registry.provision("One")
        binding = self.registry.binding()
        wrong = Path(descriptor["root"]).parent / "nonexistent"
        with self.assertRaises(AccessDenied):
            validate_binding(dataclasses.replace(binding, root=str(wrong)))
        self.assertFalse(wrong.exists())
        before = Path(binding.root, "runtime.sqlite3").read_bytes()
        with self.assertRaises(Conflict):
            validate_binding(dataclasses.replace(binding, resident_id="someone-else"))
        self.assertEqual(Path(binding.root, "runtime.sqlite3").read_bytes(), before)
        with self.assertRaises(Conflict):
            validate_binding(dataclasses.replace(binding, protocol=999))

    async def test_symlink_escape_and_unmarked_root_rejected(self):
        descriptor = self.registry.provision("One")
        binding = self.registry.binding()
        alias = self.root / "residents" / "alias"
        alias.symlink_to(descriptor["root"], target_is_directory=True)
        with self.assertRaises(AccessDenied):
            validate_binding(dataclasses.replace(binding, root=str(alias)))
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(AccessDenied):
                SandboxRegistry(raw, release="fixture-release")
        with self.assertRaises(AccessDenied):
            SandboxRegistry.create("bad", parent="/opt/TamAGI")
        with tempfile.TemporaryDirectory() as external:
            target = Path(external, "identity.txt")
            target.write_text("external state")
            identity = Path(binding.root, "workspace", "IDENTITY.md")
            identity.unlink()
            identity.symlink_to(target)
            with self.assertRaises(AccessDenied):
                validate_binding(binding)
            self.assertEqual(target.read_text(), "external state")

    async def test_malformed_auth_spoofed_identity_and_unknown_operations_fail_closed(self):
        self.registry.provision("One")
        worker = await self.worker(self.registry.resolve())
        original = worker.binding
        worker.process.stdin.write(encode_frame({"protocol": 1,
            "binding": dataclasses.replace(original, resident_id="spoofed").to_dict(),
            "token": worker.token, "request_id": "spoof", "operation": "inspect", "payload": {}}))
        await worker.process.stdin.drain()
        denial = decode_frame(await worker.process.stdout.readline())
        self.assertFalse(denial["ok"])
        self.assertIn("bound resident", denial["error"])
        original_token = worker.token
        worker.token = "bad"
        with self.assertRaisesRegex(ProtocolError, "credential"):
            await worker.request("inspect")
        worker.token = original_token
        with self.assertRaisesRegex(ProtocolError, "Unsupported"):
            await worker.request("read_file", {"path": "/opt/TamAGI/data"})
        self.assertEqual((await worker.request("inspect"))["resident_id"], original.resident_id)

    async def test_worker_death_is_visible_and_does_not_auto_replay(self):
        self.registry.provision("One")
        worker = await self.worker(self.registry.resolve())
        worker.process.kill()
        await worker.process.wait()
        with self.assertRaisesRegex(ProtocolError, "unavailable"):
            await worker.request("inspect", request_id="retained-id")

    async def test_cancelled_ipc_wait_poisons_channel_before_another_request(self):
        self.registry.provision("One")
        worker = await self.worker(self.registry.resolve())
        waiting = asyncio.Event()

        async def blocked(_):
            waiting.set()
            await asyncio.Event().wait()

        with patch.object(worker, "_receive", blocked):
            task = asyncio.create_task(worker.request("inspect"))
            await waiting.wait()
            queued = asyncio.create_task(worker.request("inspect"))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaisesRegex(ProtocolError, "unavailable"):
                await queued
        with self.assertRaisesRegex(ProtocolError, "unavailable"):
            await worker.request("inspect")

    async def test_malformed_response_poisons_channel(self):
        self.registry.provision("One")
        worker = await self.worker(self.registry.resolve())
        fake_reader = asyncio.StreamReader()
        fake_reader.feed_data(b"[broken]\n")
        with patch.object(worker.process, "stdout", fake_reader):
            with self.assertRaises(ProtocolError):
                await worker.request("inspect")
        self.assertTrue(worker.failed)
        with self.assertRaisesRegex(ProtocolError, "unavailable"):
            await worker.request("inspect")

    def test_frame_bounds_and_environment_deny_live_configuration(self):
        for raw in (b"[]\n", b"{}", b"{bad}\n", b'{"x":NaN}\n', b" " * MAX_FRAME_BYTES + b"\n"):
            with self.assertRaises(ProtocolError):
                decode_frame(raw)
        with self.assertRaises(ProtocolError):
            encode_frame({"text": "x" * MAX_FRAME_BYTES})
        self.assertEqual(decode_frame(encode_frame({"x": 1})), {"x": 1})
        with patch.dict(os.environ, {"TAMAGI_CONFIG": "/live/config", "TAMAGI_LLM_API_KEY": "secret",
                                     "TAMAGI_ES_URL": "http://live", "HTTPS_PROXY": "http://proxy"}):
            environment = worker_environment(self.root)
        self.assertNotIn("TAMAGI_LLM_API_KEY", environment)
        self.assertNotIn("TAMAGI_ES_URL", environment)
        self.assertNotIn("HTTPS_PROXY", environment)
        self.assertNotEqual(environment["TAMAGI_CONFIG"], "/live/config")
