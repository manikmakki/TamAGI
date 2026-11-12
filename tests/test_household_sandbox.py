import tempfile
import unittest

from backend.core.household.sandbox import run_scenario


class HouseholdSandboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_resident_script_runs_without_real_models_or_notifications(self):
        with tempfile.TemporaryDirectory(prefix="tamagi-scenario-test-", dir="/tmp") as parent:
            report = await run_scenario("fixture-release", parent=parent)
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(len(report["residents"]), 3)
            self.assertEqual(len(report["provider_timeline"]), 6)
            self.assertEqual(report["messages"], 6)
            self.assertEqual(report["push_dispatches"], 0)
            self.assertEqual(report["network_model_calls"], 0)
