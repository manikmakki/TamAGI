import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core.household.broker import InferenceBroker
from backend.core.household.contracts import ReconciliationRequired
from backend.core.household.driver import BrokerDriver, ProviderFinishedError, ProviderResult
from backend.core.household.store import HouseholdStore


class Provider:
    def __init__(self, clock):
        self.clock = clock
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.maximum = 0
        self.error = None
        self.usage = {"completion_tokens": 3}

    async def generate(self, attempt):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.started.set()
        try:
            await self.release.wait()
            self.clock[0] += 7
            if self.error:
                raise self.error
            return ProviderResult("retained fixture reply", self.usage)
        finally:
            self.active -= 1


class HouseholdDriverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = [1000.0]
        self.store = HouseholdStore(Path(self.temp.name, "household.sqlite3"), clock=lambda: self.clock[0])
        self.broker = InferenceBroker(self.store)
        for actor in ("a", "b"):
            self.broker.register_resident(actor)
            self.broker.open_exchange(actor, origin="autonomy")
            self.broker.open_execution(actor, actor, actor)
            self.broker.request("attempt-" + actor, actor)
        self.provider = Provider(self.clock)
        self.driver = BrokerDriver(self.broker, self.provider)

    async def asyncTearDown(self):
        self.provider.release.set()
        await self.driver.close()
        self.store.close()
        self.temp.cleanup()

    async def test_cancelled_observer_and_stopped_execution_do_not_cancel_provider_or_overlap(self):
        first = self.driver.dispatch()
        await self.provider.started.wait()
        waiter = asyncio.create_task(self.driver.wait(first))
        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.broker.stop_execution(self.broker.attempt(first)["execution_id"])
        self.assertIsNone(self.driver.dispatch())
        self.assertEqual(self.provider.active, 1)
        self.provider.release.set()
        result = await self.driver.wait(first)
        self.assertFalse(result["delivery_permitted"])
        self.assertIsNone(result["text"])
        self.assertEqual(self.broker.attempt(first)["service_seconds"], 7)
        second = self.driver.dispatch()
        self.assertIsNotNone(second)
        await self.driver.wait(second)
        self.assertEqual(self.provider.maximum, 1)

    async def test_result_is_retained_independently_of_an_observer(self):
        identity = self.driver.dispatch()
        self.provider.release.set()
        result = await self.driver.wait(identity)
        another_observer = BrokerDriver(self.broker, self.provider)
        self.assertEqual(await another_observer.wait(identity), result)
        self.assertEqual(result["text"], "retained fixture reply")
        self.assertEqual(self.broker.attempt(identity)["usage"]["completion_tokens"], 3)

    async def test_ambiguous_disconnect_holds_provider_admission(self):
        self.provider.error = ConnectionError("transport lost")
        identity = self.driver.dispatch()
        self.provider.release.set()
        result = await self.driver.wait(identity)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(await BrokerDriver(self.broker, self.provider).wait(identity), result)
        with self.assertRaises(ReconciliationRequired):
            self.driver.dispatch()
        self.assertEqual(len(self.store.list("broker_charge")), 1)

    async def test_proven_finished_failure_keeps_charge_and_shared_backoff(self):
        self.provider.error = ProviderFinishedError("fixture operation ended with a known failure")
        identity = self.driver.dispatch()
        self.provider.release.set()
        result = await self.driver.wait(identity)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(await BrokerDriver(self.broker, self.provider).wait(identity), result)
        self.assertIsNone(self.driver.dispatch())
        self.clock[0] += 1
        self.assertIsNotNone(self.driver.dispatch())

    async def test_nonfinite_provider_usage_enters_visible_reconciliation(self):
        self.provider.usage = {"completion_tokens": float("nan")}
        identity = self.driver.dispatch()
        self.provider.release.set()
        result = await self.driver.wait(identity)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(self.broker.attempt(identity)["status"], "unknown")
        with self.assertRaises(ReconciliationRequired):
            self.driver.dispatch()
        self.assertEqual(await BrokerDriver(self.broker, self.provider).wait(identity), result)

    async def test_crash_between_completion_and_result_rolls_back_into_recorded_hold(self):
        original = self.store.put
        failed = False

        def fail_once(kind, *args, **kwargs):
            nonlocal failed
            if kind == "broker_result" and not failed:
                failed = True
                raise RuntimeError("Injected result persistence failure")
            return original(kind, *args, **kwargs)

        identity = self.driver.dispatch()
        self.provider.release.set()
        with patch.object(self.store, "put", fail_once):
            result = await self.driver.wait(identity)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("Settlement failed", result["error"])
        self.assertEqual(self.broker.attempt(identity)["status"], "unknown")
        with self.assertRaises(ReconciliationRequired):
            self.driver.dispatch()
        self.assertEqual(len(self.store.list("broker_charge")), 1)

    async def test_failed_outcome_projection_cannot_silently_release_admission(self):
        original = self.store.put

        def fail_result(kind, *args, **kwargs):
            if kind == "broker_result":
                raise RuntimeError("Persistent result projection failure")
            return original(kind, *args, **kwargs)

        self.provider.error = ProviderFinishedError("Known provider failure")
        identity = self.driver.dispatch()
        self.provider.release.set()
        with patch.object(self.store, "put", fail_result):
            with self.assertRaises(ReconciliationRequired):
                await self.driver.wait(identity)
        self.assertEqual(self.broker.attempt(identity)["status"], "unknown")
        with self.assertRaises(ReconciliationRequired):
            self.driver.dispatch()
