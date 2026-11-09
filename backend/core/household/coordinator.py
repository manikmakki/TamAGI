"""Small offline composition root for the ADR-0019 foundations.

This harness exercises real private runtime processes and canonical shared commands.
It does not expose production APIs or connect existing Agent/model/tool execution.
"""

from __future__ import annotations

import asyncio

from .broker import InferenceBroker
from .contracts import AccessDenied, Conflict, NotFound
from .conversations import Conversations
from .store import uid
from .transport import ResidentProcess
from .world import World


class SandboxCoordinator:
    def __init__(self, registry):
        if getattr(registry, "_coordinator", None) is not None:
            raise Conflict("This registry already has a coordinator")
        self.registry = registry
        self.store = registry.store
        self.broker = InferenceBroker(self.store)
        self.conversations = Conversations(self.store, registry.participants, self.broker.validate_message_lineage)
        self.world = World(self.store)
        self.workers = {}
        self._monitors = {}
        # Reconnecting a coordinator does not resurrect old executions or effects.
        for lease in self.store.list("resident_lease"):
            if lease["data"]["status"] == "running":
                self.broker.stop_execution(lease["data"]["execution_id"])
                self.store.put("resident_lease", lease["id"], {**lease["data"], "status": "interrupted"})
        for resident in self.store.list("resident"):
            self.broker.register_resident(resident["id"], available=False)
            self.broker.set_available(resident["id"], False)
        registry._coordinator = self

    async def start_worker(self, resident_id):
        if resident_id in self.workers:
            raise Conflict("This resident already has a worker")
        binding = self.registry.binding(resident_id)
        self.broker.register_resident(resident_id, available=False)
        worker = ResidentProcess(binding)
        try:
            ready = await worker.start()
            self.broker.set_available(resident_id, True)
            self.workers[resident_id] = worker
            worker.on_unavailable = lambda: self._resident_lost(resident_id)
            self._monitors[resident_id] = asyncio.create_task(self._watch_exit(resident_id, worker))
            return ready
        except BaseException:
            await worker.close()
            self.broker.set_available(resident_id, False)
            raise

    async def _watch_exit(self, resident_id, worker):
        await worker.process.wait()
        self._resident_lost(resident_id)

    def _resident_lost(self, resident_id):
        with self.store.transaction():
            lease = self.store.get("resident_lease", resident_id)
            if lease and lease["data"]["status"] == "running":
                self.broker.stop_execution(lease["data"]["execution_id"])
                self.store.put("resident_lease", resident_id, {**lease["data"], "status": "interrupted"})
            self.broker.set_available(resident_id, False)

    def refresh_workers(self):
        for resident_id, worker in self.workers.items():
            if worker.failed or worker.process.returncode is not None:
                self._resident_lost(resident_id)

    def driver(self, provider):
        from .driver import BrokerDriver
        return BrokerDriver(self.broker, provider, before_dispatch=self.refresh_workers)

    def _worker(self, resident_id):
        self.registry.resolve(resident_id)
        worker = self.workers.get(resident_id)
        if worker is None or worker.failed or worker.process.returncode is not None:
            self._resident_lost(resident_id)
            raise AccessDenied("The explicitly selected resident worker is unavailable")
        return worker

    async def inspect(self, resident_id):
        return await self._worker(resident_id).request("inspect")

    def _lease(self, resident_id):
        self._worker(resident_id)
        record = self.store.get("resident_lease", resident_id)
        if record is None or record["data"]["status"] != "running":
            raise AccessDenied("No admitted execution for this resident")
        return record["data"]

    def _admit(self, resident_id, exchange_id, *, offer_id=None, operator_offer=False):
        self._worker(resident_id)
        previous = self.store.get("resident_lease", resident_id)
        if previous and previous["data"]["status"] == "running":
            raise Conflict("Resident already has an active execution lease")
        execution_id = uid("execution")
        self.broker.open_execution(execution_id, exchange_id, resident_id)
        lease = {"execution_id": execution_id, "exchange_id": exchange_id,
                 "status": "running", "offer_id": offer_id, "operator_offer": operator_offer}
        self.store.put("resident_lease", resident_id, lease)
        return self.broker.validate_lineage(execution_id, resident_id)

    def begin_autonomy(self, resident_id):
        """Manual fixture activation; does not schedule or wake the live application."""
        with self.store.transaction():
            self._worker(resident_id)
            exchange_id = uid("exchange")
            self.broker.open_exchange(exchange_id, origin="autonomy")
            return self._admit(resident_id, exchange_id)

    def operator_thread(self, participants, content, command_id, *, title="", group=False):
        """Trusted operator admission is separate from every resident tool path."""
        operator = self.registry.marker["operator_id"]

        def apply():
            exchange_id = uid("exchange")
            self.broker.open_exchange(exchange_id, origin="user")
            return self.conversations.create(operator, participants, command_id + ":message", title=title,
                group=group, opening=content, lineage={"exchange_id": exchange_id, "root_id": exchange_id,
                                                     "origin": "user"})

        return self.store.command(operator, command_id, "sandbox.operator_thread",
                                  {"participants": participants, "content": content, "title": title, "group": group}, apply)

    def begin_reply(self, resident_id, offer_id):
        with self.store.transaction():
            offer = self.store.get("reply_offer", offer_id)
            if offer is None:
                raise NotFound("Unknown reply offer")
            data = offer["data"]
            if data["recipient"] != resident_id or data["status"] != "pending":
                raise AccessDenied("This is not a pending offer for the selected resident")
            self.conversations.history(resident_id, data["conversation_id"])
            human = self.registry.marker["operator_id"]
            addressed = [self.store.get("message", identity)["data"] for identity in data["message_ids"]]
            operator_offer = any(message["author"] == human for message in addressed)
            return self._admit(resident_id, data["exchange_id"], offer_id=offer_id, operator_offer=operator_offer)

    def queue_generation(self, resident_id, attempt_id):
        lease = self._lease(resident_id)
        return self.broker.request(attempt_id, lease["execution_id"], first_response=lease["operator_offer"])

    def initiate(self, resident_id, participants, content, command_id, *, title="", group=False, mentions=None):
        lease = self._lease(resident_id)
        lineage = {"execution_id": lease["execution_id"], "exchange_id": lease["exchange_id"], "root_id": lease["exchange_id"]}
        return self.conversations.create(resident_id, participants, command_id, title=title, group=group,
                                         opening=content, mentions=mentions, lineage=lineage)

    def send(self, resident_id, conversation_id, content, command_id, *, mentions=None):
        lease = self._lease(resident_id)
        lineage = {"execution_id": lease["execution_id"], "exchange_id": lease["exchange_id"], "root_id": lease["exchange_id"]}
        return self.conversations.send(resident_id, conversation_id, content, command_id,
                                        mentions=mentions, lineage=lineage, reply_offer_id=lease["offer_id"])

    def shared_outcome(self, resident_id, command_id):
        # Transport/receipt recovery is observation, even after the execution ended.
        self.registry.resolve(resident_id)
        return self.store.command_result(resident_id, command_id)

    def stop(self, resident_id):
        lease = self._lease(resident_id)
        with self.store.transaction():
            self.broker.stop_execution(lease["execution_id"])
            self.store.put("resident_lease", resident_id, {**lease, "status": "stopped"})

    async def close(self):
        for resident_id in self.workers:
            record = self.store.get("resident_lease", resident_id)
            if record and record["data"]["status"] == "running":
                self.broker.stop_execution(record["data"]["execution_id"])
                self.store.put("resident_lease", resident_id, {**record["data"], "status": "interrupted"})
            self.broker.set_available(resident_id, False)
        for worker in self.workers.values():
            await worker.close()
        if self._monitors:
            await asyncio.gather(*self._monitors.values())
        self._monitors.clear()
        self.workers.clear()
        self.registry._coordinator = None
