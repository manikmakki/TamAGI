from __future__ import annotations
import asyncio, json, shutil, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from backend.config import RuntimeConfig, TamAGIConfig, set_config
from backend.core.runtime import (
    RuntimeStore,
    RuntimeService,
    set_runtime,
    current_terminal,
    BudgetExhausted,
)
from backend.core.runtime.contracts import RuntimeConflict
from backend.core.runtime.memory import CanonicalMemory
from backend.core.memory import MemoryEntry, MemoryType
from backend.core.llm import LLMMessage, LLMResponse, ToolCall
from backend.core.runtime.companions import Companions
from backend.core.runtime.adaptation import Adaptation
from backend.skills.registry import SkillRegistry
from backend.skills.runtime_skill import install_runtime_skills
from backend.skills.write_skill import WriteSkill


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.config = SimpleNamespace(model="test", temperature=0.1, max_tokens=100)

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "backend").mkdir()
        (self.source / "backend/__init__.py").write_text("")
        self.clock = [100000.0]
        self.store = RuntimeStore(
            self.root / "runtime.sqlite3", clock=lambda: self.clock[0]
        )
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.source
        )
        self.registry = SkillRegistry()
        self.registry.register(WriteSkill())
        self.llm = FakeLLM([])
        self.companions = Companions(self.runtime, self.llm, self.registry)
        self.runtime.companions = self.companions
        install_runtime_skills(self.runtime, self.registry, self.companions)
        self.runtime.adaptation = Adaptation(self.runtime, self.registry)
        config = TamAGIConfig()
        config.workspace.path = str(self.runtime.workspace)
        config.guardrails.allowed_write_paths = [str(self.runtime.workspace)]
        config.guardrails.allowed_read_paths = [str(self.runtime.workspace)]
        set_config(config)
        set_runtime(self.runtime)

    def tearDown(self):
        set_runtime(None)
        self.store.close()
        self.temp.cleanup()

    async def test_receipt_survives_invalid_narrative_and_repeated_call_id(self):
        async with self.runtime.execution("autonomy") as execution:
            result = await self.registry.execute(
                "write", path="leaf.txt", content="one", _call_id="same"
            )
            second = await self.registry.execute(
                "write", path="leaf.txt", content="two", _call_id="same"
            )
            self.assertTrue(result.success)
            self.assertTrue(second.success)
            self.assertEqual((self.runtime.workspace / "leaf.txt").read_text(), "one")
            self.assertEqual(len(self.store.receipts(execution.execution_id)), 1)
        self.assertEqual(self.store.receipts()[0]["status"], "succeeded")

    async def test_registered_artifact_retains_its_bytes_when_working_file_changes(
        self,
    ):
        working = self.runtime.workspace / "sketch.txt"
        working.write_text("first sketch")
        record = self.runtime.register_artifact("sketch.txt")
        working.write_text("revised sketch")
        saved = self.runtime.workspace / record["data"]["path"]
        self.assertEqual(saved.read_text(), "first sketch")
        self.assertNotEqual(saved, working)
        self.assertEqual(record["data"]["original_path"], "sketch.txt")

        revised = self.runtime.register_artifact(str(working))
        self.assertNotEqual(revised["data"]["sha256"], record["data"]["sha256"])
        self.assertEqual(
            (self.runtime.workspace / revised["data"]["path"]).read_text(),
            "revised sketch",
        )

    async def test_tool_revision_freezes_source_and_rejects_changed_version(self):
        from backend.core.runtime.adaptation import ProcessTool

        source = self.runtime.workspace / "leaf.py"
        source.write_text('print(\'{"success":true,"output":"first"}\')')
        proposal = self.runtime.adaptation.propose(
            "tool",
            {
                "name": "leaf",
                "description": "Draw a leaf",
                "command": ["/usr/bin/python3", str(source)],
            },
            "Keep an exact version",
        )
        source.write_text('print(\'{"success":true,"output":"second"}\')')
        definition = proposal["data"]["value"]
        worker = ProcessTool(definition, self.runtime.workspace)
        self.assertEqual((await worker.execute()).output, "first")
        Path(definition["command"][1]).write_text(source.read_text())
        changed = await worker.execute()
        self.assertFalse(changed.success)
        self.assertIn("source changed", changed.error)

    async def test_dispatch_validates_tools_and_actor_not_model_kwargs(self):
        async with self.runtime.execution(
            "companion", allowed_tools={"remember"}, actor_id="peer"
        ):
            denied = await self.registry.execute("write", path="bad", content="bad")
            self.assertFalse(denied.success)
            learned = await self.registry.execute(
                "remember", text="I like drawing", kind="preference"
            )
            self.assertTrue(learned.success)
            self.assertEqual(self.store.list("learning", "peer")[0]["actor"], "peer")
        self.assertFalse((self.runtime.workspace / "bad").exists())

    async def test_restart_marks_dispatched_effect_unknown_and_run_interrupted(self):
        from backend.core.runtime.contracts import ExecutionContext

        context = ExecutionContext("lost", "peer", "companion", root_id="lost")
        self.store.start_run(context, 3)
        receipt, _ = self.store.begin_receipt(context, "write", "call")
        self.store.recover()
        self.assertEqual(self.store.receipt(receipt["id"])["status"], "outcome_unknown")
        self.assertEqual(self.store.run("lost")["status"], "interrupted")

    async def test_persistent_sleep_and_named_event_wait(self):
        async with self.runtime.execution("autonomy"):
            self.runtime.end_turn("resting", wake_at=self.clock[0] + 100)
        restarted = RuntimeService(
            self.store, self.runtime.workspace, RuntimeConfig(), self.source
        )
        self.assertFalse(restarted.due())
        self.clock[0] += 100
        self.assertTrue(restarted.due())
        async with restarted.execution("autonomy"):
            restarted.end_turn("waiting", wake_event="notebook_changed")
        self.assertFalse(restarted.due())
        restarted.wake("different")
        self.assertFalse(restarted.due())
        restarted.wake("notebook_changed")
        self.assertTrue(restarted.due())

    async def test_children_share_parent_budget_and_restart_does_not_reset_daily_budget(
        self,
    ):
        async with self.runtime.execution("autonomy", max_calls=2) as parent:
            async with self.runtime.execution(
                "companion", actor_id="peer", max_calls=3
            ) as child:
                self.store.charge(child, 2, 32)
                self.store.charge(child, 2, 32)
                with self.assertRaises(BudgetExhausted):
                    self.store.charge(child, 2, 32)
            self.assertEqual(self.store.run(parent.execution_id)["calls"], 2)
        restarted = RuntimeService(
            self.store, self.runtime.workspace, RuntimeConfig(), self.source
        )
        async with restarted.execution("autonomy") as context:
            with self.assertRaises(BudgetExhausted):
                self.store.charge(context, 2, 32)

    async def test_user_interrupts_background_without_deleting_effects(self):
        started = asyncio.Event()

        async def background():
            async with self.runtime.execution("autonomy"):
                await self.registry.execute("write", path="saved", content="present")
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(background())
        await started.wait()
        async with self.runtime.execution("user"):
            pass
        self.assertTrue(task.cancelled())
        self.assertEqual((self.runtime.workspace / "saved").read_text(), "present")

    async def test_pursuit_and_identity_cas_survive_reopening(self):
        pursuit = self.runtime.pursuit(
            "create", attributes={"title": "Field guide", "next_step": "Sketch a leaf"}
        )
        identity = self.runtime.identity("SOUL.md", "# Soul\nI enjoy making things.", 0)
        with self.assertRaises(RuntimeConflict):
            self.runtime.identity("SOUL.md", "# Soul\nStale", 0)
        self.assertEqual(
            self.store.get("pursuit", pursuit["id"])["data"]["next_step"],
            "Sketch a leaf",
        )
        self.assertEqual(len(self.store.history("identity", identity["id"])), 1)
        (self.runtime.workspace / "SOUL.md").write_text(
            "# Soul\nI am reconsidering this preference."
        )
        self.runtime.reconcile_identity()
        self.assertEqual(
            self.runtime.identity("SOUL.md")["data"]["reason"], "external file edit"
        )

    async def test_memories_are_actor_scoped_and_shared_notes_are_explicit(self):
        memory = CanonicalMemory(self.runtime)
        async with self.runtime.execution("companion", actor_id="peer"):
            await memory.store(
                MemoryEntry("A private preference", MemoryType.PREFERENCE, id="private")
            )
            lesson = self.runtime.remember("I enjoy leaves", "preference")
            self.runtime.share("I enjoy drawing leaves", [lesson["id"]])
        self.assertIsNone(await memory.get_by_id("private"))
        self.assertEqual(await memory.recall("private"), [])
        self.assertEqual(self.store.list("shared_note")[0]["actor"], "peer")
        self.assertNotIn("A private preference", self.runtime.context())

    async def test_same_companion_continues_its_thread_without_copying_parent_identity(
        self,
    ):
        self.llm.responses = [
            LLMResponse(content="I prefer charcoal."),
            LLMResponse(content="I still prefer charcoal."),
        ]
        peer = self.companions.create("Juniper", "Curious and patient")
        first = await self.companions.consult(peer["id"], "What would you draw?")
        second = await self.companions.consult(
            peer["id"], "What did you choose?", thread_id=first["thread_id"]
        )
        self.assertEqual(first["actor_id"], second["actor_id"])
        self.assertIn("I prefer charcoal.", str([m.content for m in self.llm.calls[1]]))
        self.assertEqual(len(self.store.list("invocation", peer["id"])), 2)
        self.assertNotIn("SOUL.md", str([m.content for m in self.llm.calls[0]]))

    async def test_world_keeps_effects_without_parseable_new_state(self):
        from backend.core.world_thread import WorldThread
        from backend.config import WorldThreadConfig

        self.llm.responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "write-1",
                        "write",
                        {"path": "field-guide.txt", "content": "A leaf"},
                    )
                ]
            ),
            LLMResponse(content="I made a small sketch."),
        ]

        async def flush():
            pass

        personality = SimpleNamespace(
            state=SimpleNamespace(decay=lambda: None), save_state=lambda: None
        )
        agent = SimpleNamespace(
            runtime=self.runtime,
            llm=self.llm,
            skills=self.registry,
            identity=SimpleNamespace(get_system_prompt_context=lambda: ""),
            personality=personality,
            consolidation=None,
            flush_unsummarized_conversations=flush,
        )
        config = WorldThreadConfig(
            thread_path=str(self.root / "thread.json"),
            state_path=str(self.root / "state.json"),
        )
        world = WorldThread(agent, config)
        result = await world.tick_now()
        self.assertEqual(result["status"], "completed")
        self.assertTrue((self.runtime.workspace / "field-guide.txt").exists())
        self.assertEqual(self.store.receipts()[0]["status"], "succeeded")

    async def test_manual_opportunity_bypasses_only_conversation_pause(self):
        from backend.core.world_thread import WorldThread
        from backend.config import WorldThreadConfig
        from backend.core.runtime import GovernedLLM

        flushed = []

        async def flush():
            flushed.append(True)

        agent = SimpleNamespace(
            runtime=self.runtime,
            llm=GovernedLLM(self.llm, self.runtime),
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
        world = WorldThread(
            agent,
            WorldThreadConfig(
                thread_path=str(self.root / "thread.json"),
                state_path=str(self.root / "world.json"),
            ),
        )
        self.store.set_meta("decompression_until", self.clock[0] + 900)
        scheduled = await world._tick_once()
        self.assertEqual(scheduled["reason"], "decompression")
        self.assertFalse(scheduled["activation_started"])
        self.assertEqual(flushed, [])
        self.assertEqual(self.llm.calls, [])
        self.llm.responses = [LLMResponse(content="I looked around the workshop.")]
        manual = await world.tick_now()
        self.assertTrue(manual["activation_started"])
        self.assertEqual(manual["status"], "completed")
        self.assertEqual(flushed, [True])
        self.assertEqual(len(self.llm.calls), 1)
        with self.store.lock:
            bucket = self.store.db.execute("SELECT bucket FROM charges").fetchone()[0]
        self.assertEqual(bucket, "background")
        self.runtime.end_turn("resting", wake_at=self.clock[0] + 3600)
        resting = await world.tick_now()
        self.assertEqual(resting["reason"], "sleep")
        self.assertFalse(resting["activation_started"])
        self.runtime.end_turn("waiting", wake_event="notebook_changed")
        waiting = await world.tick_now()
        self.assertEqual(waiting["reason"], "event_wait")
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(flushed, [True])

    async def test_silent_rest_uses_one_call_and_no_narrative(self):
        from backend.core.tool_loop import run_tool_loop

        self.llm.responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "rest",
                        "end_turn",
                        {"status": "resting", "wake_at": "2030-01-01T00:00:00+00:00"},
                    )
                ]
            )
        ]
        async with self.runtime.execution("autonomy"):
            text, _ = await run_tool_loop(
                self.llm, self.registry, [LLMMessage("user", "Rest")]
            )
            self.assertEqual(text, "")
            self.assertEqual(current_terminal.get()["status"], "resting")
        self.assertEqual(len(self.llm.calls), 1)

    async def test_failed_tool_never_turns_into_platform_success(self):
        async with self.runtime.execution("autonomy"):
            result = await self.registry.execute(
                "world_action", action="move", object_id="missing"
            )
            self.assertFalse(result.success)
        receipt = self.store.receipts()[0]
        self.assertEqual(receipt["status"], "failed")
        self.runtime.remember("I think the move worked", "belief", [receipt["id"]])
        self.assertEqual(
            self.store.list("learning")[0]["data"]["support"], "interpretation"
        )

    async def test_stat_updates_do_not_double_apply_or_reward_tool_counts(self):
        from backend.core.personality import TamAGIState

        state = TamAGIState(
            vitality=0, last_stats_update=100000, last_curiosity_update=100000
        )
        with (
            patch("backend.core.personality.time.time", return_value=103600),
            patch("backend.core.personality._vitality_target", return_value=80),
        ):
            state.decay()
            first = state.vitality
            state.decay()
            self.assertEqual(state.vitality, first)
        curiosity = state.curiosity
        xp = state.experience
        state.use_skill()
        self.assertEqual(state.curiosity, curiosity)
        self.assertEqual(state.experience, xp)

    @unittest.skipUnless(shutil.which("gpg"), "GnuPG required")
    async def test_encrypted_round_trip_preserves_life_without_credentials_or_indexes(
        self,
    ):
        from backend.core.runtime.backup import PortableBackup

        backup = PortableBackup(self.runtime, max_bytes=16 * 1024**2)
        (self.runtime.workspace / "drawing.txt").write_text("A small leaf")
        (self.runtime.workspace / ".env").write_text("DO_NOT_EXPORT=secret")
        self.runtime.pursuit(
            "create", attributes={"title": "Sketch", "next_step": "Shade it"}
        )
        async with self.runtime.checkpointing():
            result = await asyncio.to_thread(
                backup.export,
                "a long test password",
                {"llm": {"api_key": "must-not-appear"}},
            )
        self.assertNotIn("must-not-appear", json.dumps(result["manifest"]))
        self.assertFalse(any(".env" in name for name in result["manifest"]["files"]))
        stage = await asyncio.to_thread(
            backup.stage, Path(result["path"]), "a long test password"
        )
        destination = self.root / "restored"
        restored = backup.commit(stage["id"], destination, True, "guarded")
        self.assertEqual(restored["resident_id"], self.runtime.actor_id)
        self.assertNotEqual(
            restored["installation_id"], self.store.meta("installation_id")
        )
        self.assertEqual(
            (destination / "workspace/drawing.txt").read_text(), "A small leaf"
        )
        restored_store = RuntimeStore(destination / "runtime.sqlite3")
        self.assertEqual(
            restored_store.list("pursuit")[0]["data"]["next_step"], "Shade it"
        )
        restored_store.close()
        with self.assertRaises(ValueError):
            backup.stage(Path(result["path"]), "wrong but long password")
        self.assertTrue(self.store.path.exists())

    @unittest.skipUnless(shutil.which("gpg"), "GnuPG required")
    async def test_encrypted_tampering_does_not_change_active_state(self):
        from backend.core.runtime.backup import PortableBackup

        backup = PortableBackup(self.runtime, max_bytes=16 * 1024**2)
        result = backup.export("a long test password")
        path = Path(result["path"])
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        with self.assertRaises(ValueError):
            backup.stage(path, "a long test password")
        self.assertEqual(self.store.meta("resident_id"), self.runtime.actor_id)

    async def test_unrestricted_host_exec_bypasses_local_command_tiers(self):
        from backend.skills.exec_skill import ExecSkill
        from backend.skills.base import SkillResult
        from unittest.mock import AsyncMock

        self.runtime.authority = "unrestricted_host"
        skill = ExecSkill()
        skill._run = AsyncMock(return_value=SkillResult(True, "hello"))
        async with self.runtime.execution("autonomy"):
            result = await skill.execute(command="an_unknown_local_command --flag")
        self.assertTrue(result.success)
        skill._run.assert_awaited_once()
        with (
            patch("backend.core.runtime.service.os.geteuid", return_value=1000),
            self.assertRaises(RuntimeError),
        ):
            self.runtime.verify_host()

    async def test_generated_world_affordance_tests_on_a_copy_then_changes_real_state(
        self,
    ):
        from backend.core.runtime.adaptation import ProcessTool

        value = {
            "success": True,
            "output": "Tea is ready",
            "data": {
                "world_operations": [
                    {
                        "action": "create",
                        "object_id": "teapot",
                        "attributes": {
                            "name": "Teapot",
                            "kind": "object",
                            "properties": {"brew": "ready"},
                        },
                    }
                ]
            },
        }
        definition = {
            "name": "make_tea",
            "description": "Make simulated tea",
            "command": [
                "/usr/bin/python3",
                "-c",
                "import json; print(" + repr(json.dumps(value)) + ")",
            ],
            "world_effects": True,
        }
        proposal = self.runtime.adaptation.propose(
            "tool", definition, "Add an optional household activity"
        )
        trial = await self.runtime.adaptation.test_tool(proposal["id"], {})
        self.assertTrue(trial["success"])
        self.assertIsNone(self.store.get("world_object", "teapot"))
        self.registry.register(ProcessTool(definition, self.runtime.workspace))
        async with self.runtime.execution("autonomy"):
            result = await self.registry.execute("make_tea")
        self.assertTrue(result.success)
        self.assertEqual(
            self.store.get("world_object", "teapot")["data"]["properties"]["brew"],
            "ready",
        )

    @unittest.skipUnless(shutil.which("gpg"), "GnuPG required")
    async def test_unsafe_tar_paths_are_rejected_before_extraction(self):
        import tarfile, io
        from backend.core.runtime.backup import PortableBackup

        backup = PortableBackup(self.runtime, max_bytes=16 * 1024**2)
        malicious = self.root / "unsafe.tar.gz"
        encrypted = self.root / "unsafe.gpg"
        with tarfile.open(malicious, "w:gz") as archive:
            item = tarfile.TarInfo("../../escape")
            item.size = 4
            archive.addfile(item, io.BytesIO(b"nope"))
        backup._crypt(malicious, encrypted, "a long test password")
        with self.assertRaises(ValueError):
            backup.stage(encrypted, "a long test password")
        self.assertEqual(self.store.meta("resident_id"), self.runtime.actor_id)
