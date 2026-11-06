"""Settled or interrupted evaluations must not strand the resident's event wait."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock

from backend.config import RuntimeConfig
from backend.core.llm import LLMResponse
from backend.core.runtime import RuntimeService, RuntimeStore
from backend.core.runtime.live_evaluation import install_evaluator


class EvaluationLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = RuntimeStore(self.root / "runtime.sqlite3")
        self.addCleanup(self.store.close)
        self.config = RuntimeConfig()
        self.runtime = RuntimeService(self.store, self.root / "workspace", self.config)
        self.llm = SimpleNamespace(
            config=SimpleNamespace(model="test", temperature=0, max_tokens=100),
            chat=AsyncMock(return_value=LLMResponse(content="Done")),
        )
        install_evaluator(self.runtime, self.llm)
        self.store.put(
            "proposal", "proposal", self.runtime.actor_id,
            {"surface": "world_prompt", "value": "Try a different prompt"},
        )

    def activity(self):
        return self.runtime.activity_state()

    async def wait(self, event="evaluation_completed"):
        async with self.runtime.execution("autonomy"):
            self.runtime.end_turn("waiting", wake_event=event)

    def job(self, status, job_id="job", actor=None):
        return self.store.put(
            "evaluation_job", job_id, actor or self.runtime.actor_id,
            {"proposal_id": "proposal", "status": status},
        )

    async def test_rolling_budget_failure_wakes_without_claiming_success(self):
        self.config.daily_evaluation_calls = 1
        async with self.runtime.execution("evaluation") as context:
            self.store.charge(context, evaluation_limit=1)
        await self.wait()
        queued = self.runtime.submit_evaluation("proposal", ["chosen_rest"])
        await asyncio.gather(*self.runtime.evaluation_tasks)
        job = self.store.get("evaluation_job", queued["job_id"])["data"]
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"], "BudgetExhausted")
        self.llm.chat.assert_not_awaited()
        self.assertTrue(self.runtime.wake_signal.is_set())
        self.assertTrue(self.runtime.due())
        self.assertIsNone(self.activity()["wake_event"])
        self.assertIsNone(self.activity()["wait_reason"])
        self.assertFalse(any(
            record["data"].get("decision") == "promote"
            for record in self.store.list("experiment")
        ))

    async def test_success_and_unexpected_failure_both_deliver_terminal_wake(self):
        for error in (None, ValueError("fixture failure")):
            with self.subTest(error=error):
                self.runtime.evaluator.compare = AsyncMock(
                    return_value={"id": "experiment"}, side_effect=error
                )
                await self.wait()
                self.runtime.wake_signal.clear()
                queued = self.runtime.submit_evaluation("proposal")
                await asyncio.gather(*self.runtime.evaluation_tasks)
                job = self.store.get("evaluation_job", queued["job_id"])["data"]
                self.assertEqual(job["status"], "failed" if error else "completed")
                self.assertTrue(self.runtime.wake_signal.is_set())
                self.assertTrue(self.runtime.due())

    async def test_cancellation_records_interruption_and_wakes(self):
        entered = asyncio.Event()

        async def compare(*args):
            entered.set()
            await asyncio.Event().wait()

        self.runtime.evaluator.compare = compare
        await self.wait()
        queued = self.runtime.submit_evaluation("proposal")
        task = next(iter(self.runtime.evaluation_tasks))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.gather(task)
        self.assertEqual(
            self.store.get("evaluation_job", queued["job_id"])["data"]["status"],
            "interrupted",
        )
        self.assertTrue(self.runtime.wake_signal.is_set())
        self.assertTrue(self.runtime.due())

    async def test_restart_interrupts_orphan_jobs_and_repairs_wait_once(self):
        self.job("queued", "queued")
        self.job("running", "running")
        await self.wait()
        restarted = RuntimeService(self.store, self.root / "workspace", self.config)
        self.assertTrue(restarted.due())
        self.assertTrue(restarted.wake_signal.is_set())
        for job_id in ("queued", "running"):
            self.assertEqual(
                self.store.get("evaluation_job", job_id)["data"]["status"],
                "interrupted",
            )
        revision = self.store.get("activity", restarted.actor_id)["revision"]
        restarted.reconcile_evaluation_wait()
        self.assertEqual(self.store.get("activity", restarted.actor_id)["revision"], revision)

    async def test_wait_after_job_settles_recovers_a_missed_wake(self):
        for status in ("completed", "failed", "interrupted"):
            with self.subTest(status=status):
                self.job(status)
                self.runtime.wake("evaluation_completed")
                await self.wait()
                self.assertTrue(self.runtime.due())
                self.assertIsNone(self.activity()["wake_event"])

    async def test_pending_jobs_other_actors_and_other_waits_are_preserved(self):
        self.job("failed", actor="another-resident")
        await self.wait()
        self.assertFalse(self.runtime.due())
        self.job("failed")
        self.job("queued", "pending")
        self.assertFalse(self.runtime.due())
        self.job("completed", "pending")
        await self.wait("world_changed")
        self.assertFalse(self.runtime.due())
        async with self.runtime.execution("autonomy"):
            self.runtime.end_turn("resting")
        self.assertFalse(self.runtime.due())


if __name__ == "__main__":
    unittest.main()
