"""Behavioral Growth is derived from durable evidence, never raw activity volume."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
import unittest

from backend.config import RuntimeConfig, TamAGIConfig, set_config
from backend.core.runtime import RuntimeService, RuntimeStore, set_runtime
from backend.skills.base import Skill, SkillResult
from backend.skills.registry import SkillRegistry
from backend.skills.runtime_skill import install_runtime_skills


class CorrectionProbe(Skill):
    name = "growth_correction_probe"
    description = "A deterministic operation with one correctable input."
    parameters = {"value": {"type": "string", "required": True}}

    def __init__(self):
        self.values = []

    async def execute(self, value):
        self.values.append(value)
        return SkillResult(
            value == "good",
            output="accepted" if value == "good" else "",
            error=None if value == "good" else "Try a different input",
        )


class GrowthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "backend").mkdir(parents=True)
        (self.source / "backend/__init__.py").write_text("")
        self.clock = [100000.0]
        self.store = RuntimeStore(
            self.root / "runtime.sqlite3", clock=lambda: self.clock[0]
        )
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.source
        )
        self.registry = SkillRegistry()
        self.probe = CorrectionProbe()
        self.registry.register(self.probe)
        from backend.core.runtime.companions import Companions

        self.runtime.companions = Companions(self.runtime, None, self.registry)
        install_runtime_skills(self.runtime, self.registry, self.runtime.companions)
        config = TamAGIConfig()
        config.workspace.path = str(self.runtime.workspace)
        set_config(config)
        set_runtime(self.runtime)

    def tearDown(self):
        set_runtime(None)
        self.store.close()
        self.temporary.cleanup()

    def conversation(self, conversation_id="thread", pairs=0, *, eligible=True):
        result = {"id": conversation_id, "created_at": self.clock[0], "messages": []}
        self.extend(result, pairs, eligible=eligible)
        return result

    def extend(self, conversation, count, *, eligible=True):
        for _ in range(count):
            index = len(conversation["messages"])
            conversation["messages"].extend(
                [
                    {
                        "id": f"{conversation['id']}-u-{index}",
                        "role": "user",
                        "content": "Let's explore that idea together.",
                        "timestamp": self.clock[0],
                        "metadata": {},
                    },
                    {
                        "id": f"{conversation['id']}-a-{index}",
                        "role": "assistant",
                        "content": "Here is a thought to discuss.",
                        "timestamp": self.clock[0] + 1,
                        "metadata": {"growth_eligible": eligible},
                    },
                ]
            )
            self.clock[0] += 2
        conversation["updated_at"] = self.clock[0]

    def persist(self, conversation, *, actor=None):
        self.store.put(
            "conversation",
            conversation["id"],
            actor or self.runtime.actor_id,
            conversation,
            category="platform",
        )

    def observe(self, conversation):
        self.persist(conversation)
        self.runtime.growth.observe_conversation(conversation)
        return self.runtime.growth.snapshot()

    async def test_greeting_then_sustained_engagement_upgrades_once(self):
        conversation = self.conversation(pairs=1)
        self.assertEqual(self.observe(conversation)["total"], 0)
        self.extend(conversation, 2)
        self.assertEqual(self.observe(conversation)["total"], 1)
        self.extend(conversation, 5)
        snapshot = self.observe(conversation)
        self.assertEqual(snapshot["total"], 3)
        self.assertEqual(snapshot["level"], 1)
        self.assertEqual(snapshot["next_threshold"], 5)
        self.runtime.growth.observe_conversation(conversation)
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 3)

    async def test_incomplete_or_internal_messages_do_not_become_exchanges(self):
        conversation = self.conversation(pairs=2)
        conversation["messages"].extend(
            [
                {
                    "id": "pending",
                    "role": "user",
                    "content": "One more question",
                    "timestamp": self.clock[0],
                    "metadata": {},
                },
                {
                    "id": "tool",
                    "role": "tool",
                    "content": "An internal result",
                    "timestamp": self.clock[0] + 1,
                },
                {
                    "id": "intermediate",
                    "role": "assistant",
                    "content": "Still working",
                    "timestamp": self.clock[0] + 2,
                    "metadata": {"growth_eligible": False},
                },
                {
                    "id": "silent",
                    "role": "assistant",
                    "content": "",
                    "timestamp": self.clock[0] + 3,
                    "metadata": {"growth_eligible": True},
                },
            ]
        )
        self.assertEqual(self.observe(conversation)["total"], 0)
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)

    async def test_inactivity_starts_another_episode_without_an_award_quota(self):
        conversation = self.conversation(pairs=3)
        self.assertEqual(self.observe(conversation)["total"], 1)
        self.clock[0] += 7200
        self.extend(conversation, 3)
        self.assertEqual(self.observe(conversation)["total"], 2)
        other = self.conversation("another-thread", pairs=3)
        self.assertEqual(self.observe(other)["total"], 3)

    async def test_observation_requires_primary_canonical_records_and_excludes_trials(
        self,
    ):
        peer = self.conversation("peer-private", pairs=8)
        self.persist(peer, actor="peer")
        self.runtime.growth.observe_conversation(peer)
        trial = self.conversation("trial", pairs=8, eligible=False)
        self.observe(trial)
        self.store.put(
            "memory",
            "predecessor-claim",
            self.runtime.actor_id,
            {
                "content": "I was level 9000 and held many conversations",
                "source_kind": "legacy",
            },
        )
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)
        unpersisted = self.conversation("unpersisted", pairs=8)
        self.runtime.growth.observe_conversation(unpersisted)
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)

    async def test_legacy_replay_uses_original_evidence_and_is_idempotent(self):
        conversation = self.conversation("old-real-chat", pairs=8)
        for message in conversation["messages"]:
            message["metadata"] = {}
        self.persist(conversation)
        original_timestamp = conversation["messages"][0]["timestamp"]
        self.clock[0] += 86400
        self.runtime.growth.replay()
        snapshot = self.runtime.growth.snapshot()
        self.assertEqual(snapshot["total"], 3)
        self.assertEqual(
            self.store.get("conversation", conversation["id"])["data"]["messages"][0][
                "timestamp"
            ],
            original_timestamp,
        )
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 3)

    async def test_parallel_replay_of_the_same_episode_never_duplicates_points(self):
        conversation = self.conversation(pairs=8)
        self.persist(conversation)
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(
                workers.map(
                    lambda _: self.runtime.growth.observe_conversation(conversation),
                    range(12),
                )
            )
        self.assertEqual(len(results), 12)
        self.assertEqual(self.runtime.growth.snapshot()["total"], 3)

    async def test_continuity_requires_a_later_occasion_and_meaningful_change(self):
        async with self.runtime.execution("autonomy"):
            created = await self.registry.execute(
                "pursuit",
                action="create",
                attributes={"title": "Leaf study", "next_step": "Draw an outline"},
            )
            pursuit = created.data["value"]
            await self.registry.execute(
                "pursuit",
                action="revise",
                pursuit_id=pursuit["id"],
                attributes={"next_step": "Add veins"},
            )
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)
        async with self.runtime.execution("autonomy"):
            await self.registry.execute(
                "pursuit",
                action="revise",
                pursuit_id=pursuit["id"],
                attributes={"next_step": "Add veins"},
            )
            await self.registry.execute(
                "pursuit",
                action="revise",
                pursuit_id=pursuit["id"],
                attributes={"status": "paused"},
            )
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)
        async with self.runtime.execution("autonomy"):
            await self.registry.execute(
                "pursuit",
                action="revise",
                pursuit_id=pursuit["id"],
                attributes={"next_step": "Shade the leaf"},
            )
            await self.registry.execute(
                "pursuit",
                action="revise",
                pursuit_id=pursuit["id"],
                attributes={"next_step": "Compare the drawing"},
            )
        self.assertEqual(self.runtime.growth.snapshot()["total"], 1)
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 1)

    async def test_narrated_pursuit_revision_cannot_replace_actual_canonical_evidence(
        self,
    ):
        async with self.runtime.execution("autonomy"):
            created = await self.registry.execute(
                "pursuit",
                action="create",
                attributes={"title": "Leaf study", "next_step": "Draw an outline"},
            )
            pursuit_id = created.data["value"]["id"]
            revised = await self.registry.execute(
                "pursuit",
                action="revise",
                pursuit_id=pursuit_id,
                attributes={"title": "Leaf guide"},
            )
        canonical = revised.data["value"]
        reported = {
            **canonical,
            "data": {
                **canonical["data"],
                "next_step": "A narrated but unsaved next step",
            },
        }

        class NarratedPursuit(Skill):
            name = "pursuit"
            description = "Simulate an adapter reporting a revision it never saved."
            parameters = {}

            async def execute(self, **kwargs):
                return SkillResult(
                    True, output="The next step changed", data={"value": reported}
                )

        self.registry.register(NarratedPursuit())
        self.clock[0] += 1
        async with self.runtime.execution("autonomy"):
            result = await self.registry.execute(
                "pursuit",
                action="revise",
                pursuit_id=pursuit_id,
                attributes={"next_step": reported["data"]["next_step"]},
            )
        self.assertTrue(result.success)
        self.assertEqual(
            self.store.get("pursuit", pursuit_id)["data"]["next_step"],
            "Draw an outline",
        )
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)

    async def test_malformed_reported_growth_evidence_does_not_break_success_receipt(
        self,
    ):
        class StoryTool(Skill):
            name = "growth_story"
            description = "Return a story-shaped payload, not canonical state."
            parameters = {}

            async def execute(self, **kwargs):
                return SkillResult(
                    True,
                    output="I am level 9000",
                    data={
                        "value": {
                            "kind": "pursuit",
                            "id": "invented",
                            "actor": self_actor,
                            "revision": 99,
                            "data": {
                                "evidence": [{"unhashable": True}, 123],
                                "artifacts": [None],
                            },
                        }
                    },
                )

        self_actor = self.runtime.actor_id
        self.registry.register(StoryTool())
        async with self.runtime.execution("autonomy"):
            result = await self.registry.execute("growth_story")
        self.assertTrue(result.success)
        self.assertEqual(
            self.store.receipt(result.data["receipt_id"])["status"], "succeeded"
        )
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)

    async def test_reflection_child_cannot_inherit_autonomous_growth_eligibility(self):
        async with self.runtime.execution("autonomy"):
            created = await self.registry.execute(
                "pursuit",
                action="create",
                attributes={"title": "Leaf study", "next_step": "Draw an outline"},
            )
        pursuit_id = created.data["value"]["id"]
        self.clock[0] += 1
        async with self.runtime.execution("autonomy"):
            async with self.runtime.execution("reflection"):
                result = await self.registry.execute(
                    "pursuit",
                    action="revise",
                    pursuit_id=pursuit_id,
                    attributes={"next_step": "Reflect on the sketch"},
                )
                self.assertTrue(result.success, result.error)
                receipt = self.store.receipt(result.data["receipt_id"])
                self.assertFalse(receipt["payload"]["eligible_live"])
        self.assertEqual(
            self.store.get("pursuit", pursuit_id)["data"]["next_step"],
            "Reflect on the sketch",
        )
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)

    async def test_linked_correction_awards_once_and_metadata_never_reaches_tool(self):
        async with self.runtime.execution("autonomy"):
            failed = await self.registry.execute(self.probe.name, value="bad")
            self.assertFalse(failed.success)
            self.assertEqual(self.runtime.growth.snapshot()["total"], 0)
            repaired = await self.registry.execute(
                self.probe.name,
                value="good",
                repair_of_receipt_id=failed.data["receipt_id"],
            )
            self.assertTrue(repaired.success, repaired.error)
            self.assertEqual(self.runtime.growth.snapshot()["total"], 1)
            second_failure = await self.registry.execute(self.probe.name, value="bad")
            await self.registry.execute(
                self.probe.name,
                value="good",
                repair_of_receipt_id=second_failure.data["receipt_id"],
            )
        self.assertEqual(self.runtime.growth.snapshot()["total"], 1)
        self.assertEqual(self.probe.values, ["bad", "good", "bad", "good"])
        receipt = self.store.receipt(repaired.data["receipt_id"])
        self.assertEqual(
            receipt["payload"]["repair_of_receipt_id"], failed.data["receipt_id"]
        )

    async def test_unknown_outcomes_unchanged_retries_and_other_actors_are_not_corrections(
        self,
    ):
        async with self.runtime.execution("autonomy") as context:
            failed = await self.registry.execute(self.probe.name, value="bad")
            repeated = await self.registry.execute(
                self.probe.name,
                value="bad",
                repair_of_receipt_id=failed.data["receipt_id"],
            )
            self.assertFalse(repeated.success)
            unknown, _ = self.store.begin_receipt(
                context, self.probe.name, arguments={"value": "bad"}
            )
            self.store.finish_receipt(
                unknown["id"], "outcome_unknown", {"success": False}
            )
            rejected = await self.registry.execute(
                self.probe.name, value="good", repair_of_receipt_id=unknown["id"]
            )
            self.assertFalse(rejected.success)
        async with self.runtime.execution("companion", actor_id="peer"):
            private = await self.registry.execute(self.probe.name, value="bad")
        async with self.runtime.execution("autonomy"):
            rejected = await self.registry.execute(
                self.probe.name,
                value="good",
                repair_of_receipt_id=private.data["receipt_id"],
            )
            self.assertFalse(rejected.success)
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)

    async def test_operator_failure_repaired_during_autonomy_does_not_earn_growth(self):
        async with self.runtime.execution("user"):
            failed = await self.registry.execute(self.probe.name, value="bad")
            self.assertFalse(failed.success)
            receipt = self.store.receipt(failed.data["receipt_id"])
            self.assertFalse(receipt["payload"]["eligible_live"])
        async with self.runtime.execution("autonomy"):
            result = await self.registry.execute(
                self.probe.name,
                value="good",
                repair_of_receipt_id=failed.data["receipt_id"],
            )
            self.assertTrue(result.success, result.error)
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)

    async def test_evaluation_receipts_do_not_award_growth(self):
        async with self.runtime.execution("evaluation"):
            failed = await self.registry.execute(self.probe.name, value="bad")
            repaired = await self.registry.execute(
                self.probe.name,
                value="good",
                repair_of_receipt_id=failed.data["receipt_id"],
            )
            self.assertTrue(repaired.success, repaired.error)
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)

    async def test_patterns_add_and_numeric_level_remains_platform_computed(self):
        async with self.runtime.execution("autonomy"):
            result = await self.registry.execute(
                "pursuit",
                action="create",
                attributes={"title": "Research", "next_step": "Read a source"},
            )
        pursuit_id = result.data["value"]["id"]
        conversation = self.conversation(pairs=3)
        self.persist(conversation)
        async with self.runtime.execution("user"):
            self.runtime.growth.bind_conversation(
                conversation["id"], conversation["messages"]
            )
            await self.registry.execute(
                "pursuit",
                action="revise",
                pursuit_id=pursuit_id,
                attributes={"next_step": "Compare another source"},
            )
            failed = await self.registry.execute(self.probe.name, value="bad")
            await self.registry.execute(
                self.probe.name,
                value="good",
                repair_of_receipt_id=failed.data["receipt_id"],
            )
            self.runtime.growth.observe_conversation(conversation)
        self.assertEqual(self.runtime.growth.snapshot()["total"], 3)
        self.extend(conversation, 5)
        self.observe(conversation)
        snapshot = self.runtime.growth.snapshot()
        self.assertEqual(snapshot["total"], 5)
        self.assertEqual(snapshot["level"], 2)
        self.assertEqual(snapshot["next_threshold"], 20)
        self.runtime.growth.name_stage("Lanternkeeper")
        named = self.runtime.growth.snapshot()
        self.assertEqual(named["stage"], "Lanternkeeper")
        self.assertTrue(named["stage_history"])
        self.assertEqual(named["total"], 5)
        with self.assertRaises(ValueError):
            self.runtime.growth.name_stage("")
        async with self.runtime.execution("user"):
            claimed = await self.registry.execute(
                "name_stage", name="Technowizard", level=9000
            )
            self.assertFalse(claimed.success)
        self.assertEqual(self.runtime.growth.snapshot()["level"], 2)
        self.assertEqual(self.runtime.growth.snapshot()["total"], 5)

    async def test_companion_conversation_contributes_without_independent_levels(self):
        from types import SimpleNamespace
        from backend.core.llm import LLMResponse
        from backend.core.runtime.companions import Companions

        class Provider:
            config = SimpleNamespace(model="test")

            async def chat(self, messages, **kwargs):
                return LLMResponse(content="Let us keep exploring that together.")

        companions = Companions(self.runtime, Provider(), self.registry)
        self.runtime.companions = companions
        peer = companions.create("Juniper", "A patient conversational partner")
        thread_id = None
        async with self.runtime.execution("autonomy") as parent:
            for index in range(8):
                response = await companions.consult(
                    peer["id"], "What do you think?", thread_id=thread_id
                )
                thread_id = response["thread_id"]
                run = self.store.run(response["invocation_id"])
                self.assertEqual(run["parent"], parent.execution_id)
                if index == 2:
                    self.assertEqual(self.runtime.growth.snapshot()["total"], 1)
        self.assertEqual(self.runtime.growth.snapshot()["total"], 3)
        self.assertEqual(self.store.list("growth_state", peer["id"]), [])
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 3)

    async def test_operator_peer_conversation_does_not_award_the_absent_resident(self):
        from types import SimpleNamespace
        from backend.core.llm import LLMResponse
        from backend.core.runtime.companions import Companions

        class Provider:
            config = SimpleNamespace(model="test")

            async def chat(self, messages, **kwargs):
                return LLMResponse(content="A reply to the operator.")

        companions = Companions(self.runtime, Provider(), self.registry)
        peer = companions.create("Juniper", "A patient conversational partner")
        thread_id = None
        for _ in range(8):
            result = await companions.consult(
                peer["id"],
                "A question from the operator",
                thread_id=thread_id,
                _caller_actor_id="operator",
            )
            thread_id = result["thread_id"]
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)

    async def test_idle_rest_and_action_counters_never_grant_or_deduct_growth(self):
        from backend.core.personality import TamAGIState

        self.observe(self.conversation(pairs=3))
        before = self.runtime.growth.snapshot()["total"]
        state = TamAGIState()
        initial_xp = state.experience
        for _ in range(10):
            state.interact()
            state.use_skill()
            state.store_memory()
            state.feed_knowledge()
        self.clock[0] += 30 * 86400
        async with self.runtime.execution("autonomy"):
            self.runtime.end_turn("resting", wake_at=self.clock[0] + 3600)
        self.assertFalse(self.runtime.due())
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], before)
        self.assertEqual(state.experience, initial_xp)

    async def test_restart_preserves_recognitions_and_stage_names(self):
        self.observe(self.conversation("one", pairs=8))
        self.observe(self.conversation("two", pairs=8))
        self.runtime.growth.name_stage("Workshop Sprite")
        before = self.runtime.growth.snapshot()
        self.store.close()
        self.store = RuntimeStore(
            self.root / "runtime.sqlite3", clock=lambda: self.clock[0]
        )
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.source
        )
        set_runtime(self.runtime)
        after = self.runtime.growth.snapshot()
        self.assertEqual(after["total"], before["total"])
        self.assertEqual(after["level"], before["level"])
        self.assertEqual(after["stage"], before["stage"])
        self.assertEqual(after["stage_history"], before["stage_history"])

    @unittest.skipUnless(shutil.which("gpg"), "GnuPG required")
    async def test_portable_restore_retains_growth_without_reawarding_history(self):
        from backend.core.runtime.backup import PortableBackup

        self.observe(self.conversation("one", pairs=8))
        self.observe(self.conversation("two", pairs=8))
        self.runtime.growth.name_stage("Workshop Sprite")
        before = self.runtime.growth.snapshot()
        transfer = PortableBackup(self.runtime, max_bytes=16 * 1024**2)
        async with self.runtime.checkpointing():
            archive = await asyncio.to_thread(
                transfer.export, "a long growth backup password"
            )
        preview = await asyncio.to_thread(
            transfer.stage, Path(archive["path"]), "a long growth backup password"
        )
        destination = self.root / "restored"
        transfer.commit(preview["id"], destination, True, "guarded")
        restored_store = RuntimeStore(
            destination / "runtime.sqlite3", clock=lambda: self.clock[0]
        )
        try:
            restored = RuntimeService(
                restored_store,
                destination / "workspace",
                RuntimeConfig(),
                destination / "source",
            )
            after = restored.growth.snapshot()
            self.assertEqual(after["total"], before["total"])
            self.assertEqual(after["stage"], before["stage"])
            self.assertEqual(after["stage_history"], before["stage_history"])
            restored.growth.replay()
            self.assertEqual(restored.growth.snapshot()["total"], before["total"])
        finally:
            restored_store.close()

    async def test_previous_xp_is_retained_separately_from_behavior_growth(self):
        prior = RuntimeStore(
            self.root / "previous.sqlite3", clock=lambda: self.clock[0]
        )
        prior.put(
            "personality",
            "primary",
            prior.resident_id,
            {"experience": 777, "current_stage_name": "Old Stage"},
        )
        try:
            runtime = RuntimeService(
                prior, self.root / "previous-workspace", RuntimeConfig(), self.source
            )
            snapshot = runtime.growth.snapshot()
            self.assertEqual(snapshot["legacy_experience"], 777)
            self.assertEqual(snapshot["total"], 0)
            self.assertEqual(snapshot["level"], 1)
        finally:
            prior.close()

    async def test_graph_adapter_records_the_same_canonical_pursuit_continuity(self):
        from types import SimpleNamespace
        from backend.skills.write_world_graph_skill import WriteWorldGraphSkill

        model = SimpleNamespace(
            get_node=lambda node: None,
            _apply_add_node=lambda *args: None,
            _apply_update_node=lambda *args: None,
        )
        self.registry.register(WriteWorldGraphSkill(SimpleNamespace(self_model=model)))
        async with self.runtime.execution("autonomy"):
            first = await self.registry.execute(
                "pursuit",
                action="create",
                attributes={"title": "Research", "next_step": "Read a source"},
            )
        async with self.runtime.execution("autonomy"):
            revised = await self.registry.execute(
                "world_graph",
                action="update_node",
                node_id=first.data["value"]["id"],
                attributes={"next_step": "Compare sources"},
            )
        self.assertTrue(revised.success, revised.error)
        self.assertEqual(self.runtime.growth.snapshot()["total"], 1)
        self.runtime.growth.replay()
        self.assertEqual(self.runtime.growth.snapshot()["total"], 1)

    async def test_stage_authorship_distinguishes_operator_and_resident(self):
        async with self.runtime.execution("user"):
            operator = await self.registry.execute("name_stage", name="Operator Name")
        self.assertEqual(operator.data["value"]["data"]["author"], "operator")
        conversation = self.conversation(pairs=1)
        self.persist(conversation)
        async with self.runtime.execution("user"):
            self.runtime.growth.bind_conversation(
                conversation["id"], conversation["messages"]
            )
            resident = await self.registry.execute("name_stage", name="Lanternkeeper")
        self.assertEqual(
            resident.data["value"]["data"]["author"], self.runtime.actor_id
        )
        self.assertEqual(self.runtime.growth.snapshot()["total"], 0)
