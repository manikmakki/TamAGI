"""Reproducible three-process scripted scenario, with a synthetic provider clock."""

from __future__ import annotations

import asyncio

from .coordinator import SandboxCoordinator
from .driver import ProviderResult
from .registry import SandboxRegistry


class DeterministicProvider:
    def __init__(self, clock, durations):
        self.clock = clock
        self.durations = durations
        self.timeline = []
        self.active = 0
        self.maximum_active = 0

    async def generate(self, attempt):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            started = self.clock[0]
            await asyncio.sleep(0)
            self.clock[0] += self.durations[attempt["attempt_id"]]
            self.timeline.append({"attempt_id": attempt["attempt_id"], "actor_id": attempt["actor_id"],
                "sponsor_id": attempt["sponsor_id"], "exchange_id": attempt["exchange_id"],
                "start": started, "end": self.clock[0]})
            return ProviderResult("A scripted fixture response.", {"completion_tokens": 5})
        finally:
            self.active -= 1


async def run_scenario(release: str, *, parent="/tmp") -> dict:
    registry = SandboxRegistry.create(release, parent=parent, operator_name="FixtureHuman")
    coordinator = None
    driver = None
    clock = [1000.0]
    registry.store.clock = lambda: clock[0]
    try:
        descriptors = [registry.provision(name, seed=f"Distinct private seed for {name}.")
                       for name in ("Juniper fixture", "Lark fixture", "Moss fixture")]
        ids = [descriptor["resident_id"] for descriptor in descriptors]
        coordinator = SandboxCoordinator(registry)
        ready = [await coordinator.start_worker(resident_id) for resident_id in ids]
        assert len({state["cwd"] for state in ready}) == 3
        for index, state in enumerate(ready):
            assert state["identities"][0]["text"] == f"Distinct private seed for {descriptors[index]['name']}."
            assert state["resident_id"] == ids[index]

        group = coordinator.operator_thread(ids, "A scripted shared question.", "operator-opening", group=True)
        thread_id = group["conversation"]["id"]
        assert coordinator.operator_thread(ids, "A scripted shared question.", "operator-opening", group=True) == group
        for resident_id in ids:
            offer = coordinator.conversations.offers(resident_id)[0]
            coordinator.begin_reply(resident_id, offer["id"])
        durations = {f"fixture-{round_number}-{index}": duration
                     for round_number, values in enumerate(((24.0, 3.0, 9.0), (2.0, 2.0, 2.0)))
                     for index, duration in enumerate(values)}
        provider = DeterministicProvider(clock, durations)
        driver = coordinator.driver(provider)
        for round_number in range(2):
            for index, resident_id in enumerate(ids):
                coordinator.queue_generation(resident_id, f"fixture-{round_number}-{index}")
            while identity := driver.dispatch():
                await driver.wait(identity)
        assert provider.maximum_active == 1
        assert len(registry.store.list("broker_charge")) == 6
        for index, resident_id in enumerate(ids):
            coordinator.send(resident_id, thread_id, "A scripted attributed reply.", f"group-reply-{index}", mentions=["operator"])
            coordinator.stop(resident_id)

        # Private continuity stays separate while only selected topology is shared.
        coordinator.world.import_records(ids[0], "import-juniper", "juniper", [
            {"id": "home", "kind": "place", "name": "Home", "visibility": "shared"}])
        coordinator.world.import_records(ids[1], "import-lark", "lark", [
            {"id": "home", "kind": "place", "name": "Home", "visibility": "shared"}])
        assert coordinator.world.get(ids[2], "juniper:home")["id"] != coordinator.world.get(ids[2], "lark:home")["id"]
        world = coordinator.world.create(ids[0], "shared-stone", "shared-stone", "object", "A stone",
                                          visibility="shared", location="juniper:home")
        moved = coordinator.world.update(ids[0], "move-stone", "shared-stone", {"location": "lark:home"}, world["revision"])
        assert coordinator.world.update(ids[0], "move-stone", "shared-stone", {"location": "lark:home"}, world["revision"]) == moved

        # A resident chooses an opening in its existing autonomous allowance.
        lineage = coordinator.begin_autonomy(ids[0])
        before_exchanges = len(registry.store.list("broker_exchange"))
        chosen = coordinator.initiate(ids[0], ["operator", ids[1]], "Hello @FixtureHuman, a scripted observation.",
                                      "chosen-opening", title="A chosen topic", group=True)
        chosen_thread = chosen["conversation"]["id"]
        assert len(registry.store.list("broker_exchange")) == before_exchanges
        assert chosen["message"]["lineage"]["origin"] == "autonomy"
        assert chosen["message"]["lineage"]["exchange_id"] == lineage["exchange_id"]
        # An unaddressed joined peer may read without being offered cognition.
        assert coordinator.conversations.history(ids[1], chosen_thread)
        assert not coordinator.conversations.offers(ids[1])
        snapshot = coordinator.conversations.snapshot("operator", chosen_thread)
        second = coordinator.send(ids[0], chosen_thread, "A second scripted observation.", "chosen-followup", mentions=["operator"])
        coordinator.conversations.mark_read("operator", chosen_thread, snapshot["watermark"], "foreground-read",
                                             snapshot_id=snapshot["snapshot_id"])
        unread = coordinator.conversations.unread("operator")
        chosen_unread = next(thread for thread in unread["threads"] if thread["conversation_id"] == chosen_thread)
        assert chosen_unread["count"] == 1 and chosen_unread["latest"]["id"] == second["id"]
        coordinator.stop(ids[0])
        assert coordinator.shared_outcome(ids[0], "chosen-opening")["result"] == chosen
        pending = registry.store.list("notification_intent")
        report = {"scenario_version": 1, "release": release, "fixture_root": str(registry.root),
            "household_id": registry.household_id, "residents": descriptors,
            "checks": {"distinct_private_worker_processes": True, "one_provider_operation": provider.maximum_active == 1,
                "measured_time_and_attribution": True, "group_and_resident_contact": True,
                "shared_command_replay": True, "namespaced_homes": True,
                "snapshot_unread_race": True, "no_fresh_contact_allowance": True},
            "provider_timeline": provider.timeline, "broker_state": coordinator.broker.state(),
            "messages": len(registry.store.list("message")), "notification_intents": len(pending),
            "push_dispatches": 0, "network_model_calls": 0,
            "scope": "Scripted disposable foundation scenario; production runtime/UI/migration/push and full ADR qualification remain pending."}
        return report
    finally:
        if driver:
            await driver.close()
        if coordinator:
            await coordinator.close()
        registry.close()
