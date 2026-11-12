"""Resident guidance follows real proposal, trial, and activation prerequisites."""

import asyncio
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from backend.config import RuntimeConfig
from backend.core.runtime import RuntimeService, RuntimeStore
from backend.core.runtime.adaptation import Adaptation
from backend.core.runtime.context_view import tool_result_json
from backend.core.runtime.live_evaluation import install_evaluator
from backend.core.runtime.store import encoded
from backend.skills.registry import SkillRegistry
from backend.skills.runtime_skill import install_development_skills


class RuntimeChangeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = RuntimeStore(self.root / "runtime.sqlite3")
        self.addCleanup(self.store.close)
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.root / "source"
        )
        self.registry = SkillRegistry()
        install_development_skills(self.runtime, self.registry)
        self.runtime.adaptation = Adaptation(self.runtime, self.registry)
        self.skill = self.registry.get("runtime_change")
        self.runtime.submit_evaluation = Mock(
            return_value={"job_id": "job-test", "status": "queued"}
        )

    async def call(self, **arguments):
        return (await self.skill.execute(**arguments)).data["value"]

    async def propose(self, name="health"):
        return await self.call(
            action="propose",
            surface="tool",
            intention="Read system health",
            value={
                "name": name,
                "description": "Report health",
                "command": [
                    sys.executable,
                    "-c",
                    "import json; print(json.dumps({'success':True,'data':{'ok':True}}))",
                ],
            },
        )

    def experiment(
        self,
        proposal_id,
        experiment_id="experiment-approved",
        decision="promote",
        **extra,
    ):
        proposal = self.store.get("proposal", proposal_id)
        return self.store.put(
            "experiment",
            experiment_id,
            self.runtime.actor_id,
            {
                "proposal_id": proposal_id,
                "decision": decision,
                "manifest": {
                    "candidate_hash": hashlib.sha256(
                        encoded(proposal["data"]).encode()
                    ).hexdigest()
                },
                **extra,
            },
        )

    async def test_lifecycle_returns_executable_next_calls_without_automatic_activation(
        self,
    ):
        proposed = await self.propose()
        proposal_id = proposed["proposal_id"]
        self.assertEqual(proposed["status"], "needs_worker_test")
        tested = await self.call(**proposed["next_call"]["arguments"])
        self.assertTrue(tested["success"])
        self.assertEqual(tested["status"], "needs_evaluation")
        queued = await self.call(**tested["next_call"]["arguments"])
        self.runtime.submit_evaluation.assert_called_once_with(proposal_id)
        self.assertEqual(queued["job_id"], "job-test")
        self.assertEqual(
            queued["wait_call"]["arguments"]["wake_event"], "evaluation_completed"
        )
        self.experiment(proposal_id)
        inspected = await self.call(**queued["next_call"]["arguments"])
        ready = inspected["changes"][0]
        self.assertEqual(ready["status"], "ready_to_activate")
        self.assertEqual(
            ready["next_call"]["arguments"]["experiment_id"], "experiment-approved"
        )
        self.assertEqual(self.store.list("runtime_revision"), [])
        activated = await self.call(**ready["next_call"]["arguments"])
        self.assertEqual(activated["status"], "active")
        self.assertIsNotNone(self.registry.get("health"))

    async def test_embers_missing_id_error_names_existing_approval_in_context_preview(
        self,
    ):
        proposed = await self.propose()
        proposal_id = proposed["proposal_id"]
        await self.call(**proposed["next_call"]["arguments"])
        self.experiment(proposal_id)
        with self.assertRaises(ValueError) as caught:
            await self.call(action="activate", proposal_id=proposal_id)
        error = str(caught.exception)
        self.assertIn("requires experiment_id", error)
        self.assertIn("experiment-approved", error)
        preview = tool_result_json(
            {"success": False, "error": error, "output": "x" * 10000}
        )
        self.assertIn("experiment-approved", preview)
        self.assertEqual(self.store.list("runtime_revision"), [])

    async def test_evaluate_with_experiment_id_and_missing_proposal_are_actionable(
        self,
    ):
        proposed = await self.propose()
        with self.assertRaisesRegex(ValueError, "does not accept experiment_id"):
            await self.call(
                action="evaluate",
                proposal_id=proposed["proposal_id"],
                experiment_id="experiment-approved",
            )
        for action in ("test_tool", "evaluate", "activate"):
            with (
                self.subTest(action=action),
                self.assertRaisesRegex(ValueError, "requires proposal_id"),
            ):
                await self.call(action=action, experiment_id=proposed["proposal_id"])
        with self.assertRaisesRegex(ValueError, "Unknown proposal_id"):
            await self.call(action="evaluate", proposal_id="missing")
        self.runtime.submit_evaluation.assert_not_called()

    async def test_inspect_filters_and_summarizes_large_evidence_without_losing_it(
        self,
    ):
        proposed = await self.propose()
        proposal_id = proposed["proposal_id"]
        await self.call(**proposed["next_call"]["arguments"])
        other = await self.propose("other")
        self.experiment(other["proposal_id"], "experiment-other")
        results = [{"score": 1, "events": ["x" * 10000]}] * 24
        self.experiment(proposal_id, results=results)
        compact = await self.call(action="inspect", proposal_id=proposal_id)
        self.assertEqual(len(compact["changes"]), 1)
        self.assertEqual(
            [r["id"] for r in compact["experiments"]], ["experiment-approved"]
        )
        self.assertEqual(compact["experiments"][0]["trial_count"], 24)
        self.assertNotIn("results", compact["experiments"][0])
        rendered = tool_result_json(
            (
                await self.skill.execute(action="inspect", proposal_id=proposal_id)
            ).to_dict()
        )
        self.assertIn("experiment-approved", rendered)
        self.assertIn("ready_to_activate", rendered)
        detail = await self.call(
            action="inspect", proposal_id=proposal_id, include_evidence=True
        )
        self.assertEqual(detail["experiments"][0]["data"]["results"], results)
        self.assertEqual(
            self.store.get("experiment", "experiment-approved")["data"]["results"],
            results,
        )

    async def test_pending_interrupted_failed_and_retained_evaluations_have_distinct_guidance(
        self,
    ):
        proposed = await self.propose()
        proposal_id = proposed["proposal_id"]
        await self.call(**proposed["next_call"]["arguments"])
        self.experiment(proposal_id, decision="running")
        for state, expected in [
            ("queued", "evaluation_pending"),
            ("interrupted", "evaluation_interrupted"),
            ("failed", "evaluation_failed"),
        ]:
            with self.subTest(state=state):
                self.store.put(
                    "evaluation_job",
                    "job",
                    self.runtime.actor_id,
                    {
                        "proposal_id": proposal_id,
                        "status": state,
                        "error": "RuntimeError",
                    },
                )
                view = await self.call(action="inspect", proposal_id=proposal_id)
                self.assertEqual(view["changes"][0]["status"], expected)
                self.assertEqual(view["jobs"][0]["status"], state)
        self.store.put(
            "evaluation_job",
            "job",
            self.runtime.actor_id,
            {"proposal_id": proposal_id, "status": "completed"},
        )
        self.experiment(proposal_id, decision="retain_baseline")
        view = await self.call(action="inspect", proposal_id=proposal_id)
        self.assertEqual(view["changes"][0]["status"], "no_qualifying_evaluation")
        self.assertIn("overall decision=promote", view["changes"][0]["message"])
        self.experiment(proposal_id)
        view = await self.call(action="inspect", proposal_id=proposal_id)
        self.assertEqual(view["changes"][0]["status"], "ready_to_activate")

    async def test_activation_errors_keep_existing_guards_and_explain_each_failure(
        self,
    ):
        proposed = await self.propose()
        proposal_id = proposed["proposal_id"]
        other = await self.propose("other")
        self.experiment(other["proposal_id"], "wrong-proposal")
        with self.assertRaisesRegex(ValueError, "different proposal"):
            await self.call(
                action="activate",
                proposal_id=proposal_id,
                experiment_id="wrong-proposal",
            )
        with self.assertRaisesRegex(ValueError, "Unknown experiment_id"):
            await self.call(
                action="activate", proposal_id=proposal_id, experiment_id="job-missing"
            )
        self.experiment(proposal_id, decision="retain_baseline")
        with self.assertRaisesRegex(ValueError, "overall decision='retain_baseline'"):
            await self.call(
                action="activate",
                proposal_id=proposal_id,
                experiment_id="experiment-approved",
            )
        self.experiment(proposal_id)
        with self.assertRaisesRegex(ValueError, "test_tool"):
            await self.call(
                action="activate",
                proposal_id=proposal_id,
                experiment_id="experiment-approved",
            )
        await self.call(**proposed["next_call"]["arguments"])
        record = self.store.get("proposal", proposal_id)
        self.store.put(
            "proposal",
            proposal_id,
            record["actor"],
            {**record["data"], "intention": "Changed version"},
        )
        view = await self.call(action="inspect", proposal_id=proposal_id)
        self.assertIsNone(view["changes"][0]["qualifying_experiment_id"])
        with self.assertRaisesRegex(ValueError, "different version"):
            await self.call(
                action="activate",
                proposal_id=proposal_id,
                experiment_id="experiment-approved",
            )
        self.assertEqual(self.store.list("runtime_revision"), [])

    async def test_non_tool_changes_skip_worker_test_and_worker_failure_explains_revision(
        self,
    ):
        proposed = await self.call(
            action="propose",
            surface="world_prompt",
            value="A small change",
            intention="Try a prompt",
        )
        self.assertEqual(proposed["status"], "needs_evaluation")
        with self.assertRaisesRegex(ValueError, "surface='tool'"):
            await self.call(action="test_tool", proposal_id=proposed["proposal_id"])
        broken = await self.call(
            action="propose",
            surface="tool",
            intention="Observe a failed worker",
            value={
                "name": "broken",
                "description": "Fails",
                "command": [sys.executable, "-c", "raise SystemExit(1)"],
            },
        )
        result = await self.call(**broken["next_call"]["arguments"])
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "worker_test_failed")
        self.assertIn("propose a new version", result["message"])

    async def test_background_comparison_exposes_actual_experiment_id_for_activation(
        self,
    ):
        proposed = await self.propose()
        proposal_id = proposed["proposal_id"]
        await self.call(**proposed["next_call"]["arguments"])
        llm = SimpleNamespace(
            config=SimpleNamespace(model="test", temperature=0, max_tokens=100)
        )
        install_evaluator(self.runtime, llm)
        submit = self.runtime.submit_evaluation
        self.runtime.submit_evaluation = lambda proposal_id: submit(
            proposal_id, ["worker_protocol"]
        )
        queued = await self.call(action="evaluate", proposal_id=proposal_id)
        await asyncio.gather(*self.runtime.evaluation_tasks)
        inspected = await self.call(**queued["next_call"]["arguments"])
        self.assertEqual(inspected["jobs"][0]["status"], "completed")
        ready = inspected["changes"][0]
        self.assertEqual(ready["status"], "ready_to_activate")
        self.assertEqual(
            ready["qualifying_experiment_id"], inspected["jobs"][0]["experiment_id"]
        )
        self.assertEqual(inspected["experiments"][0]["trial_count"], 6)
        self.assertEqual(
            (await self.call(**ready["next_call"]["arguments"]))["status"], "active"
        )
