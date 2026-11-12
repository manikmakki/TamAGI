"""Application-owned provider-driver seam, exercised only with deterministic fixtures.

No endpoint or LLM client is constructed here. The production streaming/GovernedLLM
adapter must qualify its transport's failure evidence before using this interface.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .contracts import Conflict, ReconciliationRequired
from .store import encoded


@dataclass(frozen=True)
class ProviderResult:
    text: str
    usage: dict = field(default_factory=dict)


class ProviderFinishedError(RuntimeError):
    """A driver has affirmative evidence that this failed operation has ended."""


class BrokerDriver:
    def __init__(self, broker, provider, *, before_dispatch=None):
        self.broker = broker
        self.provider = provider
        self.before_dispatch = before_dispatch
        self._tasks: dict[str, asyncio.Task] = {}
        self._closed = False

    def dispatch(self):
        if self._closed:
            raise Conflict("Provider driver is closed")
        if self.before_dispatch:
            self.before_dispatch()
        attempt = self.broker.dispatch()
        if attempt is None:
            return None
        identity = attempt["attempt_id"]
        self._tasks[identity] = asyncio.create_task(self._perform(identity, attempt))
        return identity

    async def _perform(self, identity, attempt):
        try:
            result = await self.provider.generate(attempt)
            if not isinstance(result, ProviderResult) or not isinstance(result.text, str) or not isinstance(result.usage, dict):
                raise ValueError("Provider result violates the driver contract")
            encoded(result.usage)
        except asyncio.CancelledError:
            self._failure(identity, "unknown", "Driver cancelled; upstream state requires reconciliation")
            raise
        except ProviderFinishedError as error:
            return self._failure(identity, "failed", str(error)[:1000], provider_error=True)
        except Exception as error:
            # A timeout/disconnect is not proof that upstream stopped. Keep the slot
            # held until the operator records evidence or acknowledges uncertainty.
            return self._failure(identity, "unknown", type(error).__name__, provider_error=True)
        try:
            if self.before_dispatch:
                self.before_dispatch()
            with self.broker.store.transaction():
                settled = self.broker.complete(identity, usage=result.usage)
                delivery = bool(settled["result_delivery_permitted"])
                data = {"attempt_id": identity, "actor_id": attempt["actor_id"], "status": settled["status"],
                        "delivery_permitted": delivery, "text": result.text if delivery else None}
                self.broker.store.put("broker_result", identity, data, expected=0)
            return data
        except Exception as error:
            try:
                return self._failure(identity, "unknown", "Settlement failed: " + type(error).__name__)
            except Exception as persistence_error:
                # If even the failure record cannot be written, occupancy remains
                # held in the store. Report that state explicitly to inspection.
                raise ReconciliationRequired("Provider settlement failed; occupancy remains held and must be inspected") from persistence_error

    def _failure(self, identity, outcome, error, *, provider_error=False):
        try:
            if self.before_dispatch:
                self.before_dispatch()
            with self.broker.store.transaction():
                settled = self.broker.complete(identity, outcome=outcome, provider_error=provider_error)
                data = {"attempt_id": identity, "status": settled["status"], "error": error,
                        "delivery_permitted": False, "text": None}
                self.broker.store.put("broker_result", identity, data, expected=0)
                return data
        except Exception as persistence_error:
            try:
                # A broken result projection need not prevent a canonical hold.
                self.broker.complete(identity, outcome="unknown")
            except Exception:
                pass  # Total store failure leaves the original occupancy retained.
            raise ReconciliationRequired("Provider outcome could not be retained; admission is held for inspection") from persistence_error

    async def wait(self, identity):
        task = self._tasks.get(identity)
        if task is None:
            record = self.broker.store.get("broker_result", identity)
            if record:
                return record["data"]
            raise ReconciliationRequired("No retained driver result; never replay a dispatched attempt")
        # The driver owns generation. Cancelling a waiter cannot release occupancy.
        return await asyncio.shield(task)

    async def close(self):
        self._closed = True
        # Healthy operations drain without a newly imposed generation time limit.
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
