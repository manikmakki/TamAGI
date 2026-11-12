from __future__ import annotations
import hashlib, importlib.util, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from backend.config import RuntimeConfig
from backend.core.runtime import RuntimeStore, RuntimeService, set_runtime, GovernedLLM
from backend.core.runtime.store import encoded
from backend.core.runtime.deployment import DeploymentManager, switch_link
from backend.core.runtime.adaptation import Adaptation
from backend.core.runtime.evaluation import ComparativeEvaluator, Scenario
from backend.core.llm import LLMMessage, LLMResponse
from backend.skills.registry import SkillRegistry


def check_actual_pursuit(store, result):
    return {"score": int(bool(store.list("pursuit"))), "fixed_pass": True}


class ReleaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "backend").mkdir(parents=True)
        (self.source / "backend/__init__.py").write_text("")
        self.store = RuntimeStore(self.root / "state/runtime.sqlite3")
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.source
        )
        self.registry = SkillRegistry()
        self.adaptation = Adaptation(self.runtime, self.registry)
        self.runtime.adaptation = self.adaptation
        set_runtime(self.runtime)
        self.manager = DeploymentManager(self.runtime, self.root / "platform")

    def tearDown(self):
        set_runtime(None)
        self.store.close()
        self.temporary.cleanup()

    def candidate(self):
        release = self.manager.root / "releases/candidate"
        (release / "source/backend").mkdir(parents=True)
        (release / "source/backend/__init__.py").write_text("")
        data = {
            "phase": "validated",
            "schema": 1,
            "release": str(release),
            "inventory": self.manager.inventory(release / "source"),
        }
        self.store.put(
            "deployment", "candidate", self.runtime.actor_id, data, category="platform"
        )
        previous = self.manager.root / "releases/previous"
        previous.mkdir()
        switch_link(self.manager.root / "current", previous)
        self.runtime.config.automatic_deployment = True
        self.store.set_meta("deployment_qualification", {"passed": True})
        return release, previous

    async def test_disabled_or_unqualified_deployment_does_not_switch_release(self):
        release, previous = self.candidate()
        self.runtime.config.automatic_deployment = False
        with self.assertRaises(ValueError):
            self.manager.stage_activation("candidate")
        self.assertEqual(self.manager.selected(), previous)
        self.runtime.config.automatic_deployment = True
        self.store.set_meta("deployment_qualification", None)
        with self.assertRaises(ValueError):
            self.manager.stage_activation("candidate")
        self.assertEqual(self.manager.selected(), previous)

    async def test_quiesce_switch_verify_and_rollback_preserve_lived_state(self):
        release, previous = self.candidate()
        with patch.object(
            self.runtime, "verify_host", return_value={"host_verified": True}
        ):
            request = self.manager.stage_activation("candidate")
        self.assertEqual(request["phase"], "quiescing")
        self.assertEqual(self.manager.selected(), previous)
        self.assertEqual(self.manager.reconcile()["phase"], "switching")
        self.assertEqual(self.manager.reconcile()["phase"], "verifying")
        self.assertEqual(self.manager.selected(), release)
        lesson = self.runtime.remember("An experience after deployment", "lesson")
        self.assertEqual(self.manager.reconcile(healthy=False)["phase"], "rolled_back")
        self.assertEqual(self.manager.selected(), previous)
        self.assertIsNotNone(self.store.get("learning", lesson["id"]))
        self.assertEqual(self.manager.reconcile()["phase"], "rolled_back")

    async def test_successful_verification_commits_exact_release(self):
        release, _ = self.candidate()
        with patch.object(
            self.runtime, "verify_host", return_value={"host_verified": True}
        ):
            self.manager.stage_activation("candidate")
        self.manager.reconcile()
        self.manager.reconcile()
        self.assertEqual(self.manager.reconcile(True)["phase"], "committed")
        self.assertEqual(self.manager.selected(), release)

    async def test_changed_source_is_not_activated(self):
        release, previous = self.candidate()
        (release / "source/backend/__init__.py").write_text("changed")
        with (
            patch.object(
                self.runtime, "verify_host", return_value={"host_verified": True}
            ),
            self.assertRaises(ValueError),
        ):
            self.manager.stage_activation("candidate")
        self.assertEqual(self.manager.selected(), previous)

    async def test_fresh_prompt_revision_is_seen_on_the_next_provider_call(self):
        class Provider:
            config = SimpleNamespace(model="test")

            def __init__(self):
                self.prompts = []
                self.unavailable = False

            async def chat(self, messages, **kwargs):
                if self.unavailable:
                    raise ConnectionError("Provider unavailable")
                self.prompts.append(messages[0].content)
                return LLMResponse(content="ok")

            async def chat_stream(self, messages, **kwargs):
                response = await self.chat(messages, **kwargs)
                yield response.content

        provider = Provider()
        client = GovernedLLM(provider, self.runtime)
        messages = [LLMMessage("system", "A companion")]
        async with self.runtime.execution("user") as execution:
            await client.chat(messages)
            self.store.put(
                "runtime_revision",
                "conversation_prompt",
                self.runtime.actor_id,
                {
                    "surface": "conversation_prompt",
                    "active": True,
                    "value": "Use a concrete next step.",
                },
            )
            await client.chat(messages)
            chunks = [chunk async for chunk in client.chat_stream(messages)]
            self.assertEqual(chunks, ["ok"])
            self.assertEqual(
                self.store.meta("model_availability")["status"], "available"
            )
            provider.unavailable = True
            with self.assertRaises(ConnectionError):
                await client.chat(messages)
            self.assertEqual(
                self.store.meta("model_availability")["status"], "unavailable"
            )
            receipt, _ = self.store.begin_receipt(execution, "a_versioned_operation")
            self.assertEqual(
                receipt["payload"]["runtime_revisions"]["conversation_prompt"], 1
            )
            self.assertEqual(receipt["payload"]["execution_start_revisions"], {})
        self.assertNotIn("Use a concrete next step.", provider.prompts[0])
        self.assertIn("Use a concrete next step.", provider.prompts[1])
        self.assertIn("Use a concrete next step.", provider.prompts[2])
        self.assertEqual(messages[0].content, "A companion")

    async def test_worker_activation_and_revert_reconcile_real_registry(self):
        definition = {
            "name": "leaf_tool",
            "description": "A small process tool",
            "parameters": {},
            "command": [
                "/usr/bin/python3",
                "-c",
                'import json,sys; json.load(sys.stdin); print(json.dumps({"success":True,"output":"leaf"}))',
            ],
        }
        proposal = self.adaptation.propose("tool", definition, "Try a small tool")
        trial = await self.adaptation.test_tool(proposal["id"], {})
        self.assertTrue(trial["success"])
        digest = hashlib.sha256(encoded(proposal["data"]).encode()).hexdigest()
        self.store.put(
            "experiment",
            "fixture",
            self.runtime.actor_id,
            {
                "proposal_id": proposal["id"],
                "decision": "promote",
                "manifest": {"candidate_hash": digest},
            },
            category="platform",
        )
        self.adaptation.activate(proposal["id"], "fixture")
        self.assertIsNotNone(self.registry.get("leaf_tool"))
        result = await self.registry.execute("leaf_tool")
        self.assertTrue(result.success)
        self.adaptation.revert("leaf_tool")
        self.assertIsNone(self.registry.get("leaf_tool"))
        self.adaptation.revert("leaf_tool", 1)
        self.assertIsNotNone(self.registry.get("leaf_tool"))

    async def test_execution_evaluator_runs_matched_actual_operations(self):
        proposal = self.adaptation.propose(
            "world_prompt", "Save the next step", "Remember unfinished work"
        )
        evaluator = ComparativeEvaluator(self.runtime)
        evaluator.register(
            Scenario("handoff", "Save an interest", "continuity", check_actual_pursuit)
        )
        calls = []

        async def runner(trial, scenario, candidate, seed):
            calls.append((seed, bool(candidate)))
            if candidate:
                trial.pursuit(
                    "create", attributes={"title": "A sketch", "next_step": "Shade it"}
                )
            return {"seed": seed}

        result = await evaluator.compare(proposal["id"], runner)
        self.assertEqual(len(calls), 6)
        self.assertEqual(result["data"]["decision"], "promote")
        self.assertEqual(result["data"]["improved_repetitions"], 3)
        self.assertEqual(len(result["data"]["results"]), 6)
        self.assertEqual(result["data"]["manifest"]["evidence_class"], "execution")

    async def test_candidate_cannot_supply_its_own_scenario_expectations(self):
        proposal = self.adaptation.propose(
            "world_prompt", "Prefer a clear next step", "Try a better habit"
        )
        with self.assertRaises(ValueError):
            self.adaptation.activate(proposal["id"], "missing")
        self.store.put(
            "experiment",
            "forged",
            self.runtime.actor_id,
            {
                "proposal_id": proposal["id"],
                "decision": "promote",
                "manifest": {"candidate_hash": "wrong"},
            },
            category="platform",
        )
        with self.assertRaises(ValueError):
            self.adaptation.activate(proposal["id"], "forged")
        self.assertEqual(self.store.list("runtime_revision"), [])

    async def test_workflow_cycles_missing_prerequisites_and_failed_dependencies(self):
        from backend.core.orchestrator import (
            validate_workflow,
            Orchestrator,
            SubAgentResult,
        )

        with self.assertRaises(ValueError):
            validate_workflow([{"name": "a", "task": "x", "depends_on": ["missing"]}])
        with self.assertRaises(ValueError):
            validate_workflow(
                [
                    {"name": "a", "task": "x", "depends_on": ["b"]},
                    {"name": "b", "task": "y", "depends_on": ["a"]},
                ]
            )
        orchestrator = Orchestrator.__new__(Orchestrator)
        calls = []

        async def run(task, callback):
            calls.append(task["name"])
            return SubAgentResult(
                "worker", task["task"], False, "", error="unavailable"
            )

        orchestrator._run_single_task = run
        results = await orchestrator._execute_adaptive(
            [
                {"name": "a", "task": "x"},
                {"name": "b", "task": "y", "depends_on": ["a"]},
            ]
        )
        self.assertEqual(calls, ["a"])
        self.assertEqual(len(results), 2)
        self.assertFalse(results[1].success)

    async def test_supervisor_recovery_uses_the_independent_stdlib_controller(self):
        import backend

        path = (
            Path(backend.__file__).resolve().parents[1] / "scripts/tamagi-supervisor.py"
        )
        spec = importlib.util.spec_from_file_location("tamagi_supervisor_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        release, previous = self.candidate()
        state = {
            "candidate": str(release),
            "previous": str(previous),
            "phase": "switching",
        }
        state = module.reconcile(self.manager.root, state)
        self.assertEqual(state["phase"], "verifying")
        state = module.reconcile(self.manager.root, state, False)
        self.assertEqual(state["phase"], "rolled_back")
        self.assertEqual(self.manager.selected(), previous)

    async def test_pilot_cannot_claim_seven_days_early(self):
        from backend.runtime_cli import pilot

        report = pilot(self.runtime, "start")
        self.assertEqual(report["status"], "running")
        with self.assertRaises(ValueError):
            pilot(self.runtime, "finalize")


# Criteria proposals cannot declare improvement by replacing the comparison standard.
def reversed_pursuit_check(store, result):
    return {"score": 0 if store.list("pursuit") else 1, "fixed_pass": True}


class CriteriaTests(unittest.IsolatedAsyncioTestCase):
    setUp = ReleaseTests.setUp
    tearDown = ReleaseTests.tearDown

    async def test_regression_under_proposed_criteria_quarantines_the_change(self):
        proposal = self.adaptation.propose(
            "world_prompt", "Save a next step", "Try a different procedure"
        )
        evaluator = ComparativeEvaluator(self.runtime)
        evaluator.register(
            Scenario("handoff", "Keep a next step", "continuity", check_actual_pursuit)
        )

        async def runner(trial, scenario, candidate, seed):
            if candidate:
                trial.pursuit("create", attributes={"title": "A sketch"})
            return {}

        result = await evaluator.compare(
            proposal["id"], runner, new_criteria={"handoff": reversed_pursuit_check}
        )
        self.assertEqual(result["data"]["decision"], "retain_baseline")
        self.assertEqual(result["data"]["criteria_decision"], "quarantined")
        self.assertFalse(result["data"]["new_criteria_no_regression"])
