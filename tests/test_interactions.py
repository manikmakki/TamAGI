"""Real execution receipts outlive observers, retries, and browser ownership."""

import asyncio
import base64
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from backend.config import RuntimeConfig
from backend.core.agent import Conversation, Message
from backend.core.runtime import RuntimeService, RuntimeStore, set_runtime
from backend.core.runtime.contracts import current_execution
from backend.core.runtime.interactions import AdmissionError, Interactions
from backend.skills.base import Skill, SkillResult
from backend.skills.registry import SkillRegistry


class WriteProbe(Skill):
    name = "probe"
    description = "A retained effect"
    parameters = {}

    def __init__(self, root):
        self.root = root
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def execute(self, **kwargs):
        self.calls += 1
        (self.root / "effect.txt").write_text(str(self.calls))
        self.started.set()
        await self.release.wait()
        return SkillResult(True, output="saved")


class Agent:
    def __init__(self, runtime):
        self.runtime = runtime
        self.conversations = {}
        self.personality = SimpleNamespace(state=SimpleNamespace(to_dict=lambda: {}))
        self._pending_approvals = {}
        self.registry = SkillRegistry()
        self.probe = WriteProbe(runtime.workspace)
        self.registry.register(self.probe)
        self.ended = []
        self.images = []

    def get_conversation(self, key):
        return self.conversations.get(key)

    def create_conversation(self, title):
        conv = Conversation(title=title)
        self.conversations[conv.id] = conv
        return conv

    def _save_conversation(self, conv):
        self.runtime.store.put(
            "conversation", conv.id, self.runtime.actor_id, conv.to_dict()
        )

    async def chat(
        self, *, conversation_id, event_callback, _admitted_message, **kwargs
    ):
        self.images.append(kwargs.get("image_data"))
        conv = self.get_conversation(conversation_id)
        if not any(m.id == _admitted_message["id"] for m in conv.messages):
            conv.messages.append(Message(**_admitted_message))
        self._save_conversation(conv)
        self.runtime.store.charge(current_execution.get(), None, 32)
        await event_callback({"type": "tool_start", "name": "probe"})
        result = await self.registry.execute("probe")
        await event_callback(
            {"type": "tool_result", "name": "probe", "output": result.output}
        )
        index = next(
            i for i, m in enumerate(conv.messages) if m.id == _admitted_message["id"]
        )
        conv.messages.insert(index + 1, Message(role="assistant", content="Done"))
        self._save_conversation(conv)
        return {
            "response": "Done",
            "conversation_id": conversation_id,
            "state": {},
            "skills_used": ["probe"],
            "memories_recalled": 0,
            "turn_status": "completed",
        }

    async def on_conversation_ended(self, key, *, schedule_resume):
        self.ended.append((key, schedule_resume))


class InteractionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.clock = [100000.0]
        self.store = RuntimeStore(
            self.root / "runtime.sqlite3", clock=lambda: self.clock[0]
        )
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.root
        )
        set_runtime(self.runtime)
        self.agent = Agent(self.runtime)
        self.manager = Interactions(self.runtime, self.agent)
        self.runtime.interactions = self.manager
        self.manager.start()

    async def asyncTearDown(self):
        await self.manager.close()
        set_runtime(None)
        self.store.close()
        self.temporary.cleanup()

    async def submit(self, key="one", **changes):
        return await self.manager.submit({"message": "Make a note", **changes}, key)

    async def test_accepted_input_precedes_execution_and_retries_do_not_interrupt(self):
        work = await self.submit()
        self.assertEqual(self.agent.probe.calls, 0)
        retained = self.manager.public(self.manager.get(work["work_id"]))
        self.assertEqual(retained["input_message"]["content"], "Make a note")
        self.assertEqual(
            self.store.get("conversation", work["conversation_id"])["data"]["messages"][
                0
            ]["id"],
            retained["input_message"]["id"],
        )
        epoch = self.runtime.input_epoch
        again = await self.submit()
        self.assertEqual(again["work_id"], work["work_id"])
        self.assertEqual(self.runtime.input_epoch, epoch)
        with self.assertRaises(AdmissionError) as error:
            await self.submit(message="Different content")
        self.assertEqual(error.exception.status, 409)
        self.agent.probe.release.set()
        await self.manager.wait(work["work_id"])
        self.assertEqual(self.agent.probe.calls, 1)
        self.assertEqual(len(self.store.receipts()), 1)

    async def test_observer_loss_and_slow_client_do_not_cancel_effects_or_result(self):
        work = await self.submit()
        observer = self.manager.subscribe()  # Deliberately never drained.
        waiter = asyncio.create_task(self.manager.wait(work["work_id"]))
        await self.agent.probe.started.wait()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        for _ in range(30):
            self.manager.emit(
                work["work_id"], {"type": "interim_text", "content": "working"}
            )
        self.assertEqual(observer.qsize(), 1)
        self.agent.probe.release.set()
        result = await self.manager.wait(work["work_id"])
        self.assertEqual(result["response"], "Done")
        self.assertEqual(self.store.receipts()[0]["status"], "succeeded")
        self.assertEqual(self.manager.get(work["work_id"])["data"]["result"], result)

    async def test_replay_is_attributed_and_expired_cursor_returns_snapshot(self):
        work = await self.submit()
        self.manager.EVENT_LIMIT = 3
        first = self.manager.snapshot(work["work_id"])
        for _ in range(6):
            self.manager.emit(
                work["work_id"], {"type": "interim_text", "content": "step"}
            )
        replay = self.manager.snapshot(work["work_id"], first["cursor"])
        self.assertTrue(replay["reset"])
        self.assertEqual(len(replay["events"]), 3)
        self.assertTrue(
            all(
                e["conversation_id"] == work["conversation_id"]
                for e in replay["events"]
            )
        )
        self.assertEqual(
            self.manager.snapshot(work["work_id"], replay["cursor"])["events"], []
        )
        ahead = self.manager.snapshot(work["work_id"], replay["cursor"] + 1000)
        self.assertTrue(ahead["reset"])
        self.assertEqual(len(ahead["events"]), 3)
        self.agent.probe.release.set()
        await self.manager.wait(work["work_id"])

    async def test_stop_retains_unknown_effect_and_next_request_still_runs(self):
        work = await self.submit()
        await self.agent.probe.started.wait()
        self.manager.stop(work["work_id"])
        result = await self.manager.wait(work["work_id"])
        self.assertEqual(result["turn_status"], "interrupted")
        self.assertEqual((self.runtime.workspace / "effect.txt").read_text(), "1")
        self.assertEqual(self.store.receipts()[0]["status"], "outcome_unknown")
        self.agent.probe.release.set()
        other = await self.submit("two")
        self.assertEqual(
            (await self.manager.wait(other["work_id"]))["turn_status"], "completed"
        )

    async def test_pending_queue_is_bounded_and_overload_has_no_admission(self):
        work = await self.submit()
        await self.agent.probe.started.wait()
        for index in range(self.manager.QUEUE_LIMIT):
            await self.submit(str(index))
        epoch = self.runtime.input_epoch
        with self.assertRaises(AdmissionError) as error:
            await self.submit("overflow")
        self.assertEqual(error.exception.status, 429)
        self.assertEqual(self.runtime.input_epoch, epoch)
        self.agent.probe.release.set()
        await self.manager.wait(work["work_id"])

    async def test_restart_recovers_input_and_unknown_outcome_without_replay(self):
        work = await self.submit()
        await self.agent.probe.started.wait()
        self.store.checkpoint(self.root / "checkpoint.sqlite3")
        restored_store = RuntimeStore(self.root / "checkpoint.sqlite3")
        try:
            restored = RuntimeService(
                restored_store,
                self.root / "restored-workspace",
                RuntimeConfig(),
                self.root,
            )
            manager = Interactions(restored, Agent(restored))
            state = manager.public(manager.get(work["work_id"]))
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["input_message"]["content"], "Make a note")
            self.assertEqual(restored_store.receipts()[0]["status"], "outcome_unknown")
            self.assertEqual(
                restored_store.db.execute("SELECT count(*) FROM charges").fetchone()[0],
                1,
            )
            self.assertFalse(manager.has_pending())
        finally:
            restored_store.close()

    async def test_presence_uses_completion_deadline_once_not_transport(self):
        work = await self.submit()
        await self.agent.probe.started.wait()
        self.clock[0] += 1800
        await self.manager.expire_visits()
        self.assertFalse(self.agent.ended)
        self.agent.probe.release.set()
        await self.manager.wait(work["work_id"])
        deadline = self.store.meta("decompression_until")
        self.assertEqual(deadline, self.clock[0] + 900)
        self.clock[0] = deadline
        await self.manager.expire_visits()
        await self.manager.expire_visits()
        self.assertEqual(self.agent.ended, [(work["conversation_id"], False)])
        self.assertEqual(self.store.meta("decompression_until"), deadline)
        self.assertTrue(self.runtime.due())

    async def test_encrypted_restore_retains_input_and_result_without_resubmission(
        self,
    ):
        from backend.core.runtime.backup import PortableBackup

        image = base64.b64encode(b"portable fixture image").decode()
        work = await self.submit(image_data=image, image_media_type="image/png")
        self.agent.probe.release.set()
        result = await self.manager.wait(work["work_id"])
        backup = PortableBackup(self.runtime)
        try:
            async with self.runtime.checkpointing():
                exported = backup.export("a long disposable backup password")
            preview = backup.stage(
                backup.root / (exported["id"] + ".tamagi.tar.gz.gpg"),
                "a long disposable backup password",
            )
            destination = self.root / "portable-restoration"
            backup.commit(preview["id"], destination, True, "guarded")
            restored_store = RuntimeStore(destination / "runtime.sqlite3")
            try:
                restored = RuntimeService(
                    restored_store,
                    destination / "workspace",
                    RuntimeConfig(),
                    self.root,
                )
                restored_manager = Interactions(restored, Agent(restored))
                state = restored_manager.public(restored_manager.get(work["work_id"]))
                self.assertEqual(state["result"], result)
                self.assertTrue(
                    state["input_message"]["content"].endswith("Make a note")
                )
                source = restored_store.get("interaction_input", work["work_id"])[
                    "data"
                ]
                self.assertEqual(
                    (
                        restored.workspace / source["payload"]["image_ref"]["path"]
                    ).read_bytes(),
                    b"portable fixture image",
                )
                self.assertFalse(restored_manager.has_pending())
                self.assertEqual(restored_store.resident_id, self.store.resident_id)
                self.assertNotEqual(
                    restored_store.meta("installation_id"),
                    self.store.meta("installation_id"),
                )
            finally:
                restored_store.close()
        finally:
            backup.cleanup()

    async def test_approval_is_request_scoped_and_expiry_does_not_reset(self):
        work = await self.submit()
        await self.agent.probe.started.wait()
        future = asyncio.get_running_loop().create_future()
        self.agent._pending_approvals["approval-one"] = future
        self.manager.emit(
            work["work_id"],
            {
                "type": "tool_approval_required",
                "approval_id": "approval-one",
                "timeout_seconds": 30,
                "command": "fixture",
            },
        )
        deadline = self.manager.snapshot(work["work_id"])["work"]["approval"][
            "expires_at"
        ]
        epoch = self.runtime.input_epoch
        with self.assertRaises(AdmissionError):
            self.manager.approve(work["work_id"], "another-approval", True)
        self.clock[0] = deadline
        with self.assertRaises(AdmissionError):
            self.manager.approve(work["work_id"], "approval-one", True)
        self.assertEqual(self.runtime.input_epoch, epoch)
        self.assertFalse(future.done())
        future.cancel()
        self.agent.probe.release.set()
        await self.manager.wait(work["work_id"])

    async def test_image_input_is_referenced_without_progress_revision_amplification(
        self,
    ):
        image = base64.b64encode(b"fixture image bytes" * 2048).decode()
        work = await self.submit(image_data=image, image_media_type="image/png")
        await self.agent.probe.started.wait()
        for _ in range(20):
            self.manager.emit(
                work["work_id"], {"type": "interim_text", "content": "progress"}
            )
        for revision in self.store.history("interaction", work["work_id"]):
            self.assertNotIn(image, str(revision["data"]))
            self.assertLess(len(str(revision["data"])), 5000)
        self.agent.probe.release.set()
        await self.manager.wait(work["work_id"])
        self.assertEqual(self.agent.images, [image])
