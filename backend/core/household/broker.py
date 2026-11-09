"""Experimental, provider-independent household admission and accounting.

Only the trusted coordinator calls these methods. Execution/actor/sponsorship
arguments are admitted server state, never model tool arguments or client claims.
``dispatch`` commits a unique charge before the provider driver contacts upstream;
``complete`` settles that driver-owned operation. The driver must continue draining
an operation after worker Stop/disconnection. This module neither contacts a model
nor wires the production GovernedLLM, streaming transport, workers, or tool leases.
Those integrations and process-level qualification remain separate release gates.
"""

from __future__ import annotations

import math

from .contracts import (
    AccessDenied, BudgetExhausted, Conflict, NotFound, ReconciliationRequired,
)


class InferenceBroker:
    def __init__(
        self, store, *, user_priority_seconds=30.0, evaluation_limit=32,
        daily_background_limit=None, backoff_initial=1.0, backoff_max=300.0,
        recover=True,
    ):
        self.store = store
        self.priority_lead = self._nonnegative(user_priority_seconds)
        self.evaluation_limit = self._limit(evaluation_limit)
        self.background_limit = (
            None if daily_background_limit is None else self._limit(daily_background_limit)
        )
        self.backoff_initial = self._nonnegative(backoff_initial)
        self.backoff_max = self._nonnegative(backoff_max)
        if self.backoff_max < self.backoff_initial:
            raise ValueError("Maximum backoff is less than initial backoff")
        with self.store.transaction():
            if self.store.meta("broker_control") is None:
                self._save_control({
                    "active_attempt": None, "hold": [], "floor": 0.0,
                    "last_account": None, "next_sequence": 0,
                    "provider_failures": 0, "backoff_until": 0.0,
                })
        if recover:
            self.recover()

    @staticmethod
    def _id(value):
        if not isinstance(value, str) or not value:
            raise ValueError("A nonempty stable identity is required")
        return value

    @staticmethod
    def _nonnegative(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("A finite nonnegative number is required")
        if not math.isfinite(value) or value < 0:
            raise ValueError("A finite nonnegative number is required")
        return float(value)

    @staticmethod
    def _limit(value):
        if type(value) is not int or value < 0:
            raise ValueError("A nonnegative integer limit is required")
        return value

    def _control(self):
        return self.store.meta("broker_control")

    def _save_control(self, data):
        self.store.set_meta("broker_control", data)

    def _get(self, kind, identity):
        record = self.store.get("broker_" + kind, identity)
        if record is None:
            raise NotFound(identity)
        return record["data"]

    def _put(self, kind, identity, data):
        return self.store.put("broker_" + kind, identity, data)["data"]

    def _list(self, kind):
        return [(r["id"], r["data"]) for r in self.store.list("broker_" + kind)]

    def _now(self):
        return self._nonnegative(self.store.clock())

    def register_resident(self, resident_id, available=True):
        self._id(resident_id)
        if type(available) is not bool:
            raise ValueError("Availability must be a boolean")
        with self.store.transaction():
            old = self.store.get("broker_actor", resident_id)
            definition = {"role": "resident", "owner": resident_id}
            if old:
                if old["data"] != definition:
                    raise Conflict("Actor identity already has another role")
                return self._get("account", resident_id)
            self._put("actor", resident_id, definition)
            return self._put("account", resident_id, {
                "service": self._control()["floor"], "available": available,
                "runnable": False, "measured_service_seconds": 0.0,
            })

    def register_companion(self, actor_id, owner_resident_id):
        self._id(actor_id)
        with self.store.transaction():
            self._get("account", owner_resident_id)
            definition = {"role": "companion", "owner": owner_resident_id}
            old = self.store.get("broker_actor", actor_id)
            if old and old["data"] != definition:
                raise Conflict("Actor identity already has another role or owner")
            return old["data"] if old else self._put("actor", actor_id, definition)

    def set_available(self, resident_id, available):
        if type(available) is not bool:
            raise ValueError("Availability must be a boolean")
        with self.store.transaction():
            self._boundary()
            account = self._get("account", resident_id)
            account["available"] = available
            self._put("account", resident_id, account)
            self._sync_runnable()
            return self._get("account", resident_id)

    def open_exchange(self, exchange_id, origin="user"):
        self._id(exchange_id)
        if origin not in {"user", "autonomy", "evaluation"}:
            raise ValueError("Unsupported exchange origin")
        with self.store.transaction():
            old = self.store.get("broker_exchange", exchange_id)
            if old:
                if old["data"]["origin"] != origin:
                    raise Conflict("Exchange identity already has another origin")
                return old["data"]
            return self._put("exchange", exchange_id, {
                "origin": origin, "status": "running", "calls": 0,
                "max_calls": {"user": 120, "autonomy": 128,
                              "evaluation": self.evaluation_limit}[origin],
                "segment": 0, "segment_calls": 0,
                "continuations_used": 0, "continuation_pending": False,
                "continuation_requested_by": None, "continuation_requesters": [],
                "priority_claims": {},
            })

    def open_execution(
        self, execution_id, exchange_id, actor_id, parent_id=None, commissioned=False, world_sponsor=None,
    ):
        self._id(execution_id)
        if type(commissioned) is not bool:
            raise ValueError("Commissioned must be a boolean")
        with self.store.transaction():
            self._boundary()
            actor = self._get("actor", actor_id)
            exchange = self._get("exchange", exchange_id)
            if exchange["status"] != "running":
                raise BudgetExhausted("Exchange is closed")
            parent = self._get("execution", parent_id) if parent_id else None
            if world_sponsor is not None:
                if parent or actor["role"] != "resident" or exchange["origin"] != "autonomy":
                    raise AccessDenied("World sponsorship requires a full resident autonomous root")
                self._get("account", world_sponsor)
            if commissioned and parent is None:
                raise ValueError("Commissioned work requires a sponsoring parent")
            if parent:
                self._validate_lineage(parent_id, parent["actor_id"])
                if parent["exchange_id"] != exchange_id:
                    raise AccessDenied("Parent belongs to another exchange")
                if parent["helper"]:
                    raise AccessDenied("A bounded helper cannot create descendant work")
            definition = {
                "exchange_id": exchange_id, "actor_id": actor_id,
                "parent_id": parent_id, "root_id": exchange_id,
                "commissioned": commissioned,
                "sponsor_id": parent["sponsor_id"] if commissioned else world_sponsor or actor["owner"],
                "helper": bool(parent) or actor["role"] == "companion",
            }
            if world_sponsor or parent and parent.get("world_sponsored"):
                definition["world_sponsored"] = True
            old = self.store.get("broker_execution", execution_id)
            if old:
                if any(old["data"].get(k) != v for k, v in definition.items()):
                    raise Conflict("Execution identity already has different lineage")
                return old["data"]
            result = self._put("execution", execution_id, {
                **definition, "status": "running", "calls": 0,
                "max_calls": 16 if definition["helper"] else exchange["max_calls"],
            })
            self._sync_runnable()
            return result

    def _ancestors(self, execution_id):
        result, visited = [], set()
        while execution_id:
            if execution_id in visited:
                raise Conflict("Execution lineage contains a cycle")
            visited.add(execution_id)
            execution = self._get("execution", execution_id)
            result.append((execution_id, execution))
            execution_id = execution["parent_id"]
        return result

    def _remaining(self, execution_id):
        ancestors = self._ancestors(execution_id)
        execution = ancestors[0][1]
        exchange = self._get("exchange", execution["exchange_id"])
        if exchange["status"] != "running" or any(e["status"] != "running" for _, e in ancestors):
            raise BudgetExhausted("Execution or exchange is no longer running")
        if any(e["exchange_id"] != execution["exchange_id"] for _, e in ancestors):
            raise AccessDenied("Execution lineage crosses an exchange")
        amounts = [exchange["max_calls"] - exchange["calls"]]
        amounts.extend(e["max_calls"] - e["calls"] for _, e in ancestors)
        if exchange["origin"] == "user":
            amounts.append(20 - exchange["segment_calls"])
        bucket = "evaluation" if exchange["origin"] == "evaluation" else (
            "user" if exchange["origin"] == "user" else "background"
        )
        rolling_limit = self.evaluation_limit if bucket == "evaluation" else self.background_limit
        if bucket != "user" and rolling_limit is not None:
            used = sum(
                c["bucket"] == bucket and c["charged_at"] > self._now() - 86400
                for _, c in self._list("charge")
            )
            amounts.append(rolling_limit - used)
        if min(amounts) <= 0:
            raise BudgetExhausted("Shared execution allowance is exhausted")
        return min(amounts), ancestors, exchange, bucket

    def _validate_lineage(self, execution_id, actor_id):
        execution = self._get("execution", execution_id)
        if execution["actor_id"] != actor_id:
            raise AccessDenied("Execution does not belong to this actor")
        remaining, _, exchange, _ = self._remaining(execution_id)
        return {
            "execution_id": execution_id, "actor_id": actor_id,
            "exchange_id": execution["exchange_id"], "root_id": execution["root_id"],
            "sponsor_id": execution["sponsor_id"], "origin": exchange["origin"],
            "segment": exchange["segment"], "calls_remaining": remaining,
        }

    def validate_lineage(self, execution_id, actor_id):
        """Validate a trusted current binding without consuming a generation."""
        with self.store.transaction():
            return self._validate_lineage(execution_id, actor_id)

    def validate_message_lineage(self, lineage, author):
        if not isinstance(lineage, dict):
            raise AccessDenied("A trusted execution lineage is required")
        with self.store.transaction():
            valid = self._validate_lineage(lineage.get("execution_id"), author)
            if any(lineage.get(k) != valid[k] for k in ("exchange_id", "root_id")):
                raise AccessDenied("Message lineage does not match the execution")
            return valid

    def request(self, attempt_id, execution_id, *, first_response=False, retry_of=None):
        self._id(attempt_id)
        if type(first_response) is not bool:
            raise ValueError("First response must be a boolean")
        definition = {"execution_id": execution_id, "first_response": first_response,
                      "retry_of": retry_of}
        with self.store.transaction():
            old = self.store.get("broker_attempt", attempt_id)
            if old:
                if old["data"]["request"] != definition:
                    raise Conflict("Attempt identity already has different input")
                return old["data"]
            self._boundary()
            execution = self._get("execution", execution_id)
            self._validate_lineage(execution_id, execution["actor_id"])
            exchange = self._get("exchange", execution["exchange_id"])
            if retry_of:
                earlier = self._get("attempt", retry_of)
                if earlier["execution_id"] != execution_id or earlier["status"] != "failed":
                    raise Conflict("Retry must reference this execution's known failed attempt")
            priority = bool(
                first_response and exchange["origin"] == "user"
                and exchange["segment"] == 0 and not execution["parent_id"]
                and execution["actor_id"] not in exchange["priority_claims"]
            )
            if priority:
                exchange["priority_claims"][execution["actor_id"]] = attempt_id
                self._put("exchange", execution["exchange_id"], exchange)
            control = self._control()
            sequence = control["next_sequence"]
            control["next_sequence"] += 1
            self._save_control(control)
            result = self._put("attempt", attempt_id, {
                "request": definition, "execution_id": execution_id,
                "exchange_id": execution["exchange_id"], "root_id": execution["root_id"],
                "actor_id": execution["actor_id"], "sponsor_id": execution["sponsor_id"],
                "segment": exchange["segment"], "priority": priority,
                "status": "queued", "charged": False, "created_at": self._now(),
                "sequence": sequence, "service_seconds": 0.0,
                "cancel_requested": False, "result_delivery_permitted": False,
            })
            self._sync_runnable()
            return result

    def _settle(self):
        """Settle only known, live-driver occupancy; never invent downtime usage."""
        identity = self._control()["active_attempt"]
        if not identity:
            return
        attempt = self._get("attempt", identity)
        if attempt["status"] != "running":
            return
        now = max(self._now(), attempt["settled_at"])
        elapsed = now - attempt["settled_at"]
        if elapsed:
            account = self._get("account", attempt["sponsor_id"])
            account["service"] += elapsed
            account["measured_service_seconds"] += elapsed
            attempt["service_seconds"] += elapsed
            attempt["settled_at"] = now
            self._put("account", attempt["sponsor_id"], account)
            self._put("attempt", identity, attempt)

    def _update_floor(self):
        services = [a["service"] for _, a in self._list("account") if a["runnable"]]
        if services:
            control = self._control()
            control["floor"] = max(control["floor"], min(services))
            self._save_control(control)

    def _boundary(self):
        self._settle()
        self._update_floor()

    def _eligible_execution(self, execution):
        # Sponsorship pays for cognition; it does not override the cognizing
        # resident's availability or a companion owner's wake eligibility.
        actor = self._get("actor", execution["actor_id"])
        return ((execution.get("world_sponsored", False) or self._get("account", execution["sponsor_id"])["available"])
                and self._get("account", actor["owner"])["available"])

    def _sync_runnable(self):
        sponsors = set()
        # A trusted live activation remains eligible between its generations.
        # Completing a call must not discard its measured-time position merely
        # because its next call has not yet been enqueued. The coordinator sets
        # availability false for rest/unavailability and closes ended activations.
        # This eligibility never reserves the provider slot while tools run.
        for identity, execution in self._list("execution"):
            if execution["status"] == "running" and self._eligible_execution(execution):
                try:
                    self._remaining(identity)
                except (BudgetExhausted, NotFound):
                    continue
                sponsors.add(execution["sponsor_id"])
        for _, attempt in self._list("attempt"):
            if attempt["status"] in {"running", "unknown"}:
                sponsors.add(attempt["sponsor_id"])
        floor = self._control()["floor"]
        held = bool(self._control()["hold"])
        for identity, account in self._list("account"):
            runnable = identity in sponsors
            # Unknown in-flight occupancy must be settled/reconciled before a
            # new account can join its floor. Queuing during the hold is safe,
            # but must not bank service that preceded that account's admission.
            if held and not account["runnable"]:
                runnable = False
            if runnable and not account["runnable"]:
                account["service"] = max(account["service"], floor)
            account["runnable"] = runnable
            self._put("account", identity, account)
        self._update_floor()

    def dispatch(self):
        """Reserve one provider attempt and charge atomically; None means wait."""
        with self.store.transaction():
            self._boundary()
            control = self._control()
            if control["hold"]:
                raise ReconciliationRequired("Provider occupancy requires recorded reconciliation")
            if control["active_attempt"] or self._now() < control["backoff_until"]:
                return None
            candidates = []
            for identity, attempt in self._list("attempt"):
                if attempt["status"] != "queued":
                    continue
                try:
                    _, _, exchange, _ = self._remaining(attempt["execution_id"])
                    if attempt["segment"] != exchange["segment"]:
                        raise BudgetExhausted("Queued attempt belongs to an earlier segment")
                except BudgetExhausted:
                    attempt["status"] = "budget_exhausted"
                    self._put("attempt", identity, attempt)
                    continue
                if self._eligible_execution(self._get("execution", attempt["execution_id"])):
                    candidates.append((identity, attempt))
            self._sync_runnable()
            if not candidates:
                return None
            minimum = min(self._get("account", a["sponsor_id"])["service"] for _, a in candidates)
            priority = [(i, a) for i, a in candidates if a["priority"] and
                        self._get("account", a["sponsor_id"])["service"] <= minimum + self.priority_lead]
            accounts = [i for i, _ in self._list("account")]
            last = self._control()["last_account"]
            offset = (accounts.index(last) + 1) % len(accounts) if last in accounts else 0
            rotation = accounts[offset:] + accounts[:offset]
            def order(item):
                identity, attempt = item
                return (self._get("account", attempt["sponsor_id"])["service"],
                        attempt["created_at"], rotation.index(attempt["sponsor_id"]),
                        attempt["sequence"], identity)
            identity, attempt = min(priority or candidates, key=order)
            _, ancestors, exchange, bucket = self._remaining(attempt["execution_id"])
            for execution_id, execution in ancestors:
                execution["calls"] += 1
                self._put("execution", execution_id, execution)
            exchange["calls"] += 1
            exchange["segment_calls"] += 1
            self._put("exchange", attempt["exchange_id"], exchange)
            self._put("charge", identity, {
                "execution_id": attempt["execution_id"], "exchange_id": attempt["exchange_id"],
                "actor_id": attempt["actor_id"], "sponsor_id": attempt["sponsor_id"],
                "bucket": bucket, "charged_at": self._now(),
            })
            attempt.update(status="running", charged=True, started_at=self._now(), settled_at=self._now())
            self._put("attempt", identity, attempt)
            control = self._control()
            control.update(active_attempt=identity, last_account=attempt["sponsor_id"])
            self._save_control(control)
            self.store.event("broker_dispatched", attempt["actor_id"], {"attempt_id": identity, **attempt})
            return {"attempt_id": identity, **attempt}

    def complete(self, attempt_id, *, outcome="completed", usage=None, provider_error=False):
        if outcome not in {"completed", "failed", "unknown"}:
            raise ValueError("Unsupported provider outcome")
        if type(provider_error) is not bool:
            raise ValueError("Provider error must be a boolean")
        with self.store.transaction():
            attempt = self._get("attempt", attempt_id)
            completion = {"outcome": outcome, "usage": usage or {}, "provider_error": provider_error}
            if attempt["status"] in {"completed", "failed"}:
                if attempt.get("completion") != completion:
                    raise Conflict("Attempt already has another completion")
                return attempt
            if attempt["status"] == "unknown":
                if attempt.get("completion") == completion:
                    return attempt
                raise ReconciliationRequired("Unknown occupancy requires reconciliation")
            if attempt["status"] != "running" or self._control()["active_attempt"] != attempt_id:
                raise Conflict("Attempt is not the active provider operation")
            self._boundary()
            attempt = self._get("attempt", attempt_id)
            attempt.update(status=outcome, completion=completion, finished_at=self._now(), usage=usage or {})
            try:
                self._remaining(attempt["execution_id"])
                live = True
            except BudgetExhausted:
                # Exhausting this very call still permits its final response.
                exchange = self._get("exchange", attempt["exchange_id"])
                live = exchange["status"] == "running" and all(
                    e["status"] == "running" for _, e in self._ancestors(attempt["execution_id"])
                )
            attempt["result_delivery_permitted"] = bool(live and outcome == "completed" and not attempt["cancel_requested"])
            attempt["orphaned"] = not live or attempt["cancel_requested"]
            self._put("attempt", attempt_id, attempt)
            control = self._control()
            if outcome == "unknown":
                control["hold"] = sorted(set(control["hold"]) | {attempt_id})
            else:
                control["active_attempt"] = None
            if provider_error:
                control["provider_failures"] += 1
                exponent = min(control["provider_failures"] - 1, 30)
                delay = min(self.backoff_max, self.backoff_initial * 2 ** exponent)
                control["backoff_until"] = max(control["backoff_until"], self._now() + delay)
            elif outcome == "completed":
                control.update(provider_failures=0, backoff_until=0.0)
            self._save_control(control)
            self._sync_runnable()
            self.store.event("broker_completed", attempt["actor_id"], {"attempt_id": attempt_id, **attempt})
            return attempt

    def stop_execution(self, execution_id):
        with self.store.transaction():
            self._boundary()
            self._get("execution", execution_id)
            stopped = {execution_id}
            changed = True
            while changed:
                changed = False
                for identity, execution in self._list("execution"):
                    if execution["parent_id"] in stopped and identity not in stopped:
                        stopped.add(identity)
                        changed = True
            for identity in stopped:
                execution = self._get("execution", identity)
                execution["status"] = "stopped"
                self._put("execution", identity, execution)
            for identity, attempt in self._list("attempt"):
                if attempt["execution_id"] not in stopped:
                    continue
                if attempt["status"] == "queued":
                    attempt["status"] = "cancelled"
                elif attempt["status"] in {"running", "unknown"}:
                    attempt["cancel_requested"] = True
                self._put("attempt", identity, attempt)
            for identity, exchange in self._list("exchange"):
                if set(exchange["continuation_requesters"]) & stopped:
                    remaining = [i for i in exchange["continuation_requesters"] if i not in stopped]
                    exchange.update(continuation_pending=bool(remaining),
                                    continuation_requesters=remaining,
                                    continuation_requested_by=remaining[0] if remaining else None)
                    self._put("exchange", identity, exchange)
            self._sync_runnable()
            self.store.event("broker_execution_stopped", "coordinator", {"execution_ids": sorted(stopped)})

    def finish_execution(self, execution_id):
        """Complete one participant without ending the shared exchange.

        The trusted controller must first finish/cancel this participant's
        queued work, drain running operations, and close any bounded children.
        Completion releases eligibility, never resets the retained allowance.
        """
        with self.store.transaction():
            self._boundary()
            execution = self._get("execution", execution_id)
            if execution["status"] == "completed":
                return execution
            if execution["status"] != "running":
                raise Conflict("Only a running participant can complete")
            descendants = {execution_id}
            changed = True
            while changed:
                changed = False
                for identity, child in self._list("execution"):
                    if child["parent_id"] in descendants and identity not in descendants:
                        descendants.add(identity)
                        changed = True
            if any(a["execution_id"] in descendants and a["status"] in {"queued", "running", "unknown"}
                   for _, a in self._list("attempt")):
                raise Conflict("Participant still has outstanding provider work")
            if any(identity != execution_id and child["status"] == "running"
                   for identity, child in self._list("execution") if identity in descendants):
                raise Conflict("Participant still has an active bounded child")
            exchange = self._get("exchange", execution["exchange_id"])
            if execution_id in exchange["continuation_requesters"]:
                raise Conflict("Participant still has a pending quiet continuation")
            execution["status"] = "completed"
            self._put("execution", execution_id, execution)
            self._sync_runnable()
            self.store.event("broker_execution_completed", execution["actor_id"], {"execution_id": execution_id})
            return execution

    def close_exchange(self, exchange_id, status="stopped"):
        if status not in {"completed", "stopped", "failed", "budget_exhausted"}:
            raise ValueError("Unsupported exchange terminal status")
        with self.store.transaction():
            for identity, execution in self._list("execution"):
                if execution["exchange_id"] == exchange_id:
                    self.stop_execution(identity)
            exchange = self._get("exchange", exchange_id)
            exchange.update(status=status, continuation_pending=False, continuation_requested_by=None,
                            continuation_requesters=[])
            return self._put("exchange", exchange_id, exchange)

    def request_continuation(self, exchange_id, execution_id):
        with self.store.transaction():
            execution = self._get("execution", execution_id)
            self._validate_lineage(execution_id, execution["actor_id"])
            exchange = self._get("exchange", exchange_id)
            if execution["exchange_id"] != exchange_id or exchange["origin"] != "user":
                raise AccessDenied("Continuation requires this user-origin exchange")
            actor = self._get("actor", execution["actor_id"])
            if actor["role"] != "resident" or execution["helper"]:
                raise AccessDenied("Only a participating full resident may request continuation")
            if exchange["continuations_used"] >= 5:
                raise BudgetExhausted("No quiet continuations remain")
            requesters = sorted(set(exchange["continuation_requesters"]) | {execution_id})
            exchange.update(continuation_pending=True, continuation_requested_by=requesters[0],
                            continuation_requesters=requesters)
            return self._put("exchange", exchange_id, exchange)

    def advance_segment(self, exchange_id):
        """Trusted controller calls only after a successfully completed segment."""
        with self.store.transaction():
            exchange = self._get("exchange", exchange_id)
            if exchange["status"] != "running" or exchange["origin"] != "user" or not exchange["continuation_pending"]:
                raise Conflict("No successful quiet continuation is pending")
            if exchange["continuations_used"] >= 5 or exchange["calls"] >= exchange["max_calls"]:
                raise BudgetExhausted("Exchange allowance is exhausted")
            if any(a["exchange_id"] == exchange_id and a["status"] in {"queued", "running", "unknown"}
                   for _, a in self._list("attempt")):
                raise Conflict("A segment still has outstanding provider work")
            exchange.update(segment=exchange["segment"] + 1, segment_calls=0,
                            continuations_used=exchange["continuations_used"] + 1,
                            continuation_pending=False, continuation_requested_by=None,
                            continuation_requesters=[])
            return self._put("exchange", exchange_id, exchange)

    def recover(self):
        """Broker startup: never infer remote idleness or replay dispatched work."""
        with self.store.transaction():
            control = self._control()
            held = set(control["hold"])
            for identity, attempt in self._list("attempt"):
                if attempt["status"] in {"running", "unknown"}:
                    attempt.update(status="unknown", result_delivery_permitted=False,
                                   recovery_reason="Broker restart; remote generation may remain active")
                    self._put("attempt", identity, attempt)
                    held.add(identity)
            control["hold"] = sorted(held)
            self._save_control(control)
            if held:
                self.store.event("broker_reconciliation_required", "coordinator", {"attempt_ids": sorted(held)})
            return self.state()

    def reconcile(
        self, attempt_id, *, operator_id, evidence, confirmed_stopped=False,
        acknowledge_uncertainty=False, measured_service_seconds=None,
    ):
        self._id(operator_id)
        if not evidence or not isinstance(evidence, (str, dict)):
            raise ValueError("Recorded reconciliation evidence is required")
        if type(confirmed_stopped) is not bool or type(acknowledge_uncertainty) is not bool:
            raise ValueError("Reconciliation choices must be booleans")
        if not confirmed_stopped and not acknowledge_uncertainty:
            raise ReconciliationRequired("Confirm remote stop or explicitly acknowledge unresolved uncertainty")
        if measured_service_seconds is not None:
            measured_service_seconds = self._nonnegative(measured_service_seconds)
        definition = {"operator_id": operator_id, "evidence": evidence,
                      "confirmed_stopped": confirmed_stopped,
                      "acknowledge_uncertainty": acknowledge_uncertainty,
                      "measured_service_seconds": measured_service_seconds}
        with self.store.transaction():
            attempt = self._get("attempt", attempt_id)
            if attempt["status"] == "reconciled":
                if attempt["reconciliation_request"] != definition:
                    raise Conflict("Attempt already has different reconciliation evidence")
                return attempt
            if attempt["status"] != "unknown":
                raise Conflict("Attempt does not require reconciliation")
            if measured_service_seconds is not None:
                measured = self._nonnegative(measured_service_seconds)
                if measured < attempt["service_seconds"]:
                    raise Conflict("Measured service cannot erase already settled occupancy")
                account = self._get("account", attempt["sponsor_id"])
                account["service"] += measured - attempt["service_seconds"]
                account["measured_service_seconds"] += measured - attempt["service_seconds"]
                self._put("account", attempt["sponsor_id"], account)
                attempt["service_seconds"] = measured
            attempt.update(status="reconciled", reconciled_at=self._now(),
                           reconciliation_request=definition,
                           reconciliation={"operator_id": operator_id, "evidence": evidence,
                                           "confirmed_stopped": confirmed_stopped,
                                           "acknowledged_uncertainty": acknowledge_uncertainty,
                                           "unmeasured_interval": measured_service_seconds is None},
                           result_delivery_permitted=False)
            self._put("attempt", attempt_id, attempt)
            control = self._control()
            control["hold"] = [i for i in control["hold"] if i != attempt_id]
            if control["active_attempt"] == attempt_id:
                control["active_attempt"] = None
            self._save_control(control)
            self._update_floor()
            self._sync_runnable()
            self.store.event("broker_reconciled", operator_id, {"attempt_id": attempt_id, **attempt["reconciliation"]})
            return attempt

    def state(self):
        with self.store.transaction():
            return {**self._control(), "accounts": dict(self._list("account"))}

    def attempt(self, attempt_id):
        with self.store.transaction():
            return self._get("attempt", attempt_id)

    def execution(self, execution_id):
        with self.store.transaction():
            return self._get("execution", execution_id)

    def exchange(self, exchange_id):
        with self.store.transaction():
            return self._get("exchange", exchange_id)
