import asyncio
import shutil
import unittest

from backend.core.household.coordinator import SandboxCoordinator
from backend.core.household.contracts import AccessDenied, Conflict
from backend.core.household.driver import ProviderResult
from backend.core.household.registry import SandboxRegistry


class GatedProvider:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, attempt):
        self.started.set()
        await self.release.wait()
        return ProviderResult("A late fixture response")


class HouseholdCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.registry = SandboxRegistry.create("coordinator-fixture")
        self.descriptor = self.registry.provision("First fixture")
        self.actor = self.descriptor["resident_id"]
        self.coordinator = SandboxCoordinator(self.registry)
        await self.coordinator.start_worker(self.actor)
        self.provider = GatedProvider()
        self.driver = self.coordinator.driver(self.provider)

    async def asyncTearDown(self):
        self.provider.release.set()
        await self.driver.close()
        await self.coordinator.close()
        self.registry.close()
        shutil.rmtree(self.registry.root)

    async def test_only_one_activation_and_resident_origin_cannot_grant_user_priority(self):
        lineage = self.coordinator.begin_autonomy(self.actor)
        with self.assertRaises(Conflict):
            self.coordinator.begin_autonomy(self.actor)
        opening = self.coordinator.initiate(self.actor, ["operator"], "A chosen topic", "chosen")
        self.assertEqual(opening["message"]["lineage"]["exchange_id"], lineage["exchange_id"])
        attempt = self.coordinator.queue_generation(self.actor, "autonomy-call")
        self.assertFalse(attempt["priority"])
        self.assertEqual(len(self.registry.store.list("broker_exchange")), 1)
        self.coordinator.stop(self.actor)
        self.assertEqual(self.coordinator.shared_outcome(self.actor, "chosen")["result"], opening)
        with self.assertRaises(AccessDenied):
            self.coordinator.initiate(self.actor, ["operator"], "A new topic", "after-stop")
        self.assertEqual(len(self.registry.store.list("message")), 1)

    async def test_new_fixture_after_construction_registers_account_before_worker_start(self):
        descriptor = self.registry.provision("Added later")
        ready = await self.coordinator.start_worker(descriptor["resident_id"])
        self.assertEqual(ready["resident_id"], descriptor["resident_id"])
        self.assertTrue(self.coordinator.broker.state()["accounts"][descriptor["resident_id"]]["available"])
        with self.assertRaises(Conflict):
            SandboxCoordinator(self.registry)

    async def test_queued_generation_does_not_dispatch_for_a_dead_worker(self):
        self.coordinator.begin_autonomy(self.actor)
        self.coordinator.queue_generation(self.actor, "queued-before-exit")
        worker = self.coordinator.workers[self.actor]
        worker.process.kill()
        await worker.process.wait()
        self.assertIsNone(self.driver.dispatch())
        attempt = self.coordinator.broker.attempt("queued-before-exit")
        self.assertEqual(attempt["status"], "cancelled")
        self.assertFalse(attempt["charged"])
        self.assertFalse(self.provider.started.is_set())

    async def test_lost_worker_during_generation_drains_and_suppresses_response(self):
        self.coordinator.begin_autonomy(self.actor)
        self.coordinator.queue_generation(self.actor, "running-before-exit")
        identity = self.driver.dispatch()
        await self.provider.started.wait()
        worker = self.coordinator.workers[self.actor]
        worker.process.kill()
        await worker.process.wait()
        self.coordinator.refresh_workers()
        self.assertEqual(self.coordinator.broker.state()["active_attempt"], identity)
        self.assertIsNone(self.driver.dispatch())
        self.provider.release.set()
        result = await self.driver.wait(identity)
        self.assertFalse(result["delivery_permitted"])
        self.assertIsNone(result["text"])
        self.assertEqual(len(self.registry.store.list("broker_charge")), 1)

    async def test_poisoned_worker_channel_revokes_admission(self):
        self.coordinator.begin_autonomy(self.actor)
        self.coordinator.queue_generation(self.actor, "queued-before-poison")
        self.coordinator.workers[self.actor].failed = True
        self.assertIsNone(self.driver.dispatch())
        self.assertEqual(self.coordinator.broker.attempt("queued-before-poison")["status"], "cancelled")
