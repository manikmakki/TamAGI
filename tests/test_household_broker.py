"""Deterministic broker mechanics; no resident state, runtime, or provider calls."""

import tempfile
import unittest
from pathlib import Path

from backend.core.household.broker import InferenceBroker
from backend.core.household.contracts import (
    AccessDenied, BudgetExhausted, Conflict, ReconciliationRequired,
)
from backend.core.household.store import HouseholdStore


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "household.sqlite3"
        self.now = 100000.0
        self.store = HouseholdStore(self.path, clock=lambda: self.now)
        self.broker = InferenceBroker(self.store)
        for actor in ("a", "b", "c"):
            self.broker.register_resident(actor)
        self.broker.register_companion("a-companion", "a")
        self.serial = 0

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def execution(self, actor="a", *, exchange=None, origin="autonomy", identity=None, **kwargs):
        self.serial += 1
        exchange = exchange or f"exchange-{self.serial}"
        identity = identity or f"execution-{self.serial}"
        self.broker.open_exchange(exchange, origin)
        self.broker.open_execution(identity, exchange, actor, **kwargs)
        return identity

    def request(self, execution, **kwargs):
        self.serial += 1
        identity = f"attempt-{self.serial}"
        self.broker.request(identity, execution, **kwargs)
        return identity

    def finish(self, identity, duration=1, **kwargs):
        self.now += duration
        return self.broker.complete(identity, **kwargs)

    def call(self, execution, duration=1):
        identity = self.request(execution)
        self.assertEqual(self.broker.dispatch()["attempt_id"], identity)
        return identity, self.finish(identity, duration)

    def test_queued_attempt_is_uncharged_and_dispatch_is_atomic_idempotent(self):
        execution = self.execution()
        attempt = self.request(execution)
        self.assertFalse(self.broker.attempt(attempt)["charged"])
        self.assertEqual(self.broker.exchange(self.broker.execution(execution)["exchange_id"])["calls"], 0)
        first = self.broker.dispatch()
        self.assertEqual(first["attempt_id"], attempt)
        self.assertIsNone(self.broker.dispatch())
        self.assertEqual(len(self.store.list("broker_charge")), 1)
        self.assertEqual(self.broker.request(attempt, execution)["status"], "running")
        with self.assertRaises(Conflict):
            self.broker.request(attempt, execution, first_response=True)
        self.finish(attempt, 7, usage={"total_tokens": 9})
        repeated = self.broker.complete(attempt, usage={"total_tokens": 9})
        self.assertEqual(repeated["service_seconds"], 7)
        self.assertEqual(self.broker.state()["accounts"]["a"]["service"], 7)
        self.assertEqual(len(self.store.list("broker_charge")), 1)

    def test_equal_measured_time_does_not_mean_equal_call_counts(self):
        a, b = self.execution("a"), self.execution("b")
        first = self.request(a)
        self.now += 0.01
        self.request(b)
        self.assertEqual(self.broker.dispatch()["attempt_id"], first)
        self.finish(first, 10)
        waiting_a = self.request(a)
        for n in range(10):
            turn = self.broker.dispatch()
            self.assertEqual(turn["sponsor_id"], "b")
            self.finish(turn["attempt_id"], 1)
            if n < 9:
                self.request(b)
        self.assertEqual(self.broker.execution(a)["calls"], 1)
        self.assertEqual(self.broker.execution(b)["calls"], 10)
        self.assertEqual(self.broker.dispatch()["attempt_id"], waiting_a)

    def test_all_three_accounts_receive_turns_and_rotation_survives_restart(self):
        executions = [self.execution(actor) for actor in ("a", "b", "c")]
        for execution in executions:
            self.request(execution)
        order = []
        for _ in range(3):
            attempt = self.broker.dispatch()
            order.append(attempt["sponsor_id"])
            self.finish(attempt["attempt_id"], 0)
        self.assertEqual(order, ["a", "b", "c"])
        self.broker = InferenceBroker(self.store)
        self.assertEqual(self.broker.state()["last_account"], "c")
        for execution in executions:
            self.request(execution)
        self.assertEqual(self.broker.dispatch()["sponsor_id"], "a")

    def test_newly_runnable_account_is_clamped_after_midflight_settlement(self):
        a = self.execution("a")
        first = self.request(a)
        self.broker.dispatch()
        self.now += 40
        b = self.execution("b")
        self.request(b)
        accounts = self.broker.state()["accounts"]
        self.assertEqual(accounts["a"]["service"], 40)
        self.assertEqual(accounts["b"]["service"], 40)
        self.assertEqual(self.broker.attempt(first)["service_seconds"], 40)
        self.assertIsNone(self.broker.dispatch())
        self.finish(first, 10)
        self.assertEqual(self.broker.state()["accounts"]["a"]["service"], 50)
        self.assertEqual(self.broker.dispatch()["sponsor_id"], "b")

    def test_sleep_earns_no_credit_and_existing_lead_is_preserved(self):
        a, b = self.execution("a"), self.execution("b")
        self.call(a, 60)
        self.broker.set_available("b", False)
        pending = self.request(b)
        self.call(a, 40)
        self.broker.set_available("b", True)
        self.assertEqual(self.broker.state()["accounts"]["b"]["service"], 100)
        self.assertEqual(self.broker.dispatch()["attempt_id"], pending)
        self.finish(pending, 30)
        self.broker.set_available("b", False)
        self.broker.set_available("b", True)
        self.request(b)
        self.assertEqual(self.broker.state()["accounts"]["b"]["service"], 130)

    def test_priority_is_bounded_by_thirty_service_seconds(self):
        a = self.execution("a")
        self.call(a, 31)
        # B was not runnable while A worked, so it joins at the current floor.
        b = self.execution("b")
        self.request(b)
        # Create a genuine ahead-of-peer account while B is now waiting.
        user_a = self.execution("a", origin="user")
        candidate = self.request(user_a, first_response=True)
        selected = self.broker.dispatch()
        self.assertEqual(selected["attempt_id"], candidate)
        self.finish(candidate, 31)
        user_again = self.execution("a", origin="user")
        later = self.request(user_again, first_response=True)
        selected = self.broker.dispatch()
        self.assertEqual(selected["sponsor_id"], "b")
        self.assertNotEqual(selected["attempt_id"], later)

    def test_first_response_claim_cannot_be_minted_by_retry_or_another_execution(self):
        execution = self.execution("a", origin="user", exchange="user")
        first = self.request(execution, first_response=True)
        self.assertTrue(self.broker.attempt(first)["priority"])
        second = self.request(execution, first_response=True)
        another = self.execution("a", origin="user", exchange="user")
        third = self.request(another, first_response=True)
        self.assertFalse(self.broker.attempt(second)["priority"])
        self.assertFalse(self.broker.attempt(third)["priority"])
        background = self.execution("b", origin="autonomy")
        invalid = self.request(background, first_response=True)
        self.assertFalse(self.broker.attempt(invalid)["priority"])

    def test_each_operator_broadcast_resident_can_receive_one_initial_priority(self):
        for actor in ("a", "b", "c"):
            execution = self.execution(actor, exchange="broadcast", origin="user")
            attempt = self.request(execution, first_response=True)
            self.assertTrue(self.broker.attempt(attempt)["priority"])
        self.assertEqual(len(self.broker.exchange("broadcast")["priority_claims"]), 3)

    def test_twenty_calls_are_shared_across_speakers_and_queued_fanout(self):
        a = self.execution("a", exchange="shared", origin="user")
        b = self.execution("b", exchange="shared", origin="user")
        queued = [self.request(a if n % 2 else b) for n in range(24)]
        for _ in range(20):
            work = self.broker.dispatch()
            self.finish(work["attempt_id"], 0)
        self.assertIsNone(self.broker.dispatch())
        self.assertEqual(self.broker.exchange("shared")["calls"], 20)
        self.assertEqual(len(self.store.list("broker_charge")), 20)
        self.assertEqual(sum(self.broker.attempt(i)["status"] == "budget_exhausted" for i in queued), 4)
        with self.assertRaises(BudgetExhausted):
            self.request(a)

    def test_five_coalesced_continuations_share_at_most_120_calls(self):
        execution = self.execution("a", origin="user", exchange="long-user")
        for segment in range(6):
            if segment < 5:
                self.broker.request_continuation("long-user", execution)
                self.broker.request_continuation("long-user", execution)
            for _ in range(20):
                self.call(execution, 0)
            if segment < 5:
                self.broker.advance_segment("long-user")
        exchange = self.broker.exchange("long-user")
        self.assertEqual(exchange["calls"], 120)
        self.assertEqual(exchange["continuations_used"], 5)
        with self.assertRaises(BudgetExhausted):
            self.request(execution)
        another = self.execution("b", exchange="long-user", origin="user")
        with self.assertRaises(BudgetExhausted):
            self.request(another)

    def test_autonomy_stops_at_128_and_helper_cannot_open_continuation(self):
        execution = self.execution("a")
        helper = self.execution("a-companion", exchange=self.broker.execution(execution)["exchange_id"],
                                parent_id=execution, commissioned=True)
        with self.assertRaises(AccessDenied):
            self.broker.request_continuation(self.broker.execution(helper)["exchange_id"], helper)
        for _ in range(128):
            self.call(execution, 0)
        with self.assertRaises(BudgetExhausted):
            self.request(execution)

    def test_sponsorship_is_distinct_from_cognizing_actor(self):
        parent = self.execution("a", origin="user", exchange="help")
        commissioned = self.execution("b", origin="user", exchange="help", parent_id=parent, commissioned=True)
        helper = self.request(commissioned)
        work = self.broker.dispatch()
        self.assertEqual(work["attempt_id"], helper)
        self.assertEqual(work["actor_id"], "b")
        self.assertEqual(work["sponsor_id"], "a")
        self.finish(helper, 4)
        ordinary = self.execution("b", origin="user", exchange="help")
        self.assertEqual(self.broker.execution(ordinary)["sponsor_id"], "b")
        self.assertEqual(self.broker.execution(parent)["calls"], 1)
        self.assertEqual(self.broker.execution(commissioned)["calls"], 1)
        direct = self.execution("a-companion", origin="user")
        self.assertEqual(self.broker.execution(direct)["sponsor_id"], "a")
        self.assertNotIn("a-companion", self.broker.state()["accounts"])

    def test_helper_16_is_also_debited_from_initiating_exchange(self):
        parent = self.execution("a", origin="user", exchange="child-budget")
        child = self.execution("a-companion", origin="user", exchange="child-budget",
                               parent_id=parent, commissioned=True)
        for _ in range(16):
            self.call(child, 0)
        with self.assertRaises(BudgetExhausted):
            self.request(child)
        self.assertEqual(self.broker.validate_lineage(parent, "a")["calls_remaining"], 4)
        with self.assertRaises(BudgetExhausted):
            self.broker.validate_lineage(child, "a-companion")

    def test_companion_cannot_mint_independent_time_or_quiet_grants(self):
        execution = self.execution("a-companion", origin="user", exchange="companion")
        with self.assertRaises(AccessDenied):
            self.broker.request_continuation("companion", execution)
        with self.assertRaises(AccessDenied):
            self.execution("b", origin="user", exchange="companion", parent_id=execution)
        with self.assertRaises(Conflict):
            self.broker.register_companion("a-companion", "b")

    def test_stop_queued_attempt_and_descendants_costs_no_calls(self):
        parent = self.execution("a")
        child = self.execution("b", exchange=self.broker.execution(parent)["exchange_id"], parent_id=parent,
                               commissioned=True)
        queued = self.request(child)
        self.broker.stop_execution(parent)
        self.assertEqual(self.broker.attempt(queued)["status"], "cancelled")
        self.assertFalse(self.broker.attempt(queued)["charged"])
        self.assertEqual(self.broker.execution(child)["status"], "stopped")
        self.assertIsNone(self.broker.dispatch())
        self.assertEqual(len(self.store.list("broker_charge")), 0)

    def test_running_stop_retains_slot_until_drained_and_suppresses_tools(self):
        a, b = self.execution("a"), self.execution("b")
        first = self.request(a)
        self.broker.dispatch()
        self.now += 3
        self.broker.stop_execution(a)
        second = self.request(b)
        self.assertIsNone(self.broker.dispatch())
        result = self.finish(first, 8, usage={"total_tokens": 22})
        self.assertEqual(result["service_seconds"], 11)
        self.assertTrue(result["orphaned"])
        self.assertFalse(result["result_delivery_permitted"])
        self.assertEqual(self.broker.dispatch()["attempt_id"], second)

    def test_restart_holds_uncertain_generation_without_replay_or_health_shortcut(self):
        execution = self.execution()
        attempt = self.request(execution)
        self.broker.dispatch()
        self.now += 20
        self.store.close()
        self.store = HouseholdStore(self.path, clock=lambda: self.now)
        self.broker = InferenceBroker(self.store)
        self.assertEqual(self.broker.attempt(attempt)["status"], "unknown")
        self.assertEqual(self.broker.attempt(attempt)["service_seconds"], 0)
        self.assertEqual(len(self.store.list("broker_charge")), 1)
        self.store.set_meta("model_health", {"status": "available"})
        with self.assertRaises(ReconciliationRequired):
            self.broker.dispatch()
        with self.assertRaises(ReconciliationRequired):
            self.broker.reconcile(attempt, operator_id="operator", evidence="models endpoint answered")
        with self.assertRaises(ReconciliationRequired):
            self.broker.complete(attempt)
        self.broker.reconcile(attempt, operator_id="operator", evidence="Provider was stopped and inspected",
                              confirmed_stopped=True, measured_service_seconds=12)
        self.assertEqual(self.broker.state()["hold"], [])
        self.assertEqual(self.broker.state()["accounts"]["a"]["service"], 12)
        self.assertEqual(self.broker.execution(execution)["calls"], 1)

    def test_operator_can_release_hold_with_recorded_uncertainty_without_fabricated_usage(self):
        execution = self.execution()
        attempt = self.request(execution)
        self.broker.dispatch()
        self.finish(attempt, 3, outcome="unknown", provider_error=True)
        self.now += 300
        result = self.broker.reconcile(attempt, operator_id="operator", evidence={"reason": "acknowledged uncertainty"},
                                       acknowledge_uncertainty=True)
        self.assertTrue(result["reconciliation"]["unmeasured_interval"])
        self.assertEqual(result["service_seconds"], 3)
        self.assertFalse(result["result_delivery_permitted"])
        events = self.store.events()
        self.assertEqual(events[-1]["kind"], "broker_reconciled")
        self.assertEqual(events[-1]["actor"], "operator")

    def test_shared_backoff_and_unique_retry_charges_survive_restart(self):
        execution = self.execution()
        failed = self.request(execution)
        self.broker.dispatch()
        self.finish(failed, 2, outcome="failed", provider_error=True)
        retry = self.request(execution, retry_of=failed)
        self.now += 0.01
        b = self.execution("b")
        self.request(b)
        self.assertIsNone(self.broker.dispatch())
        self.broker = InferenceBroker(self.store)
        self.assertIsNone(self.broker.dispatch())
        self.now += 1
        selected = self.broker.dispatch()
        self.assertEqual(selected["attempt_id"], retry)
        self.finish(retry, 1)
        self.assertEqual(len(self.store.list("broker_charge")), 2)
        self.assertEqual(self.broker.state()["provider_failures"], 0)

    def test_evaluation_rolling_limit_survives_new_exchange_and_restart(self):
        self.broker = InferenceBroker(self.store, evaluation_limit=2)
        execution = self.execution(origin="evaluation")
        self.call(execution, 0)
        self.call(execution, 0)
        self.broker = InferenceBroker(self.store, evaluation_limit=2)
        other = self.execution("b", origin="evaluation")
        with self.assertRaises(BudgetExhausted):
            self.request(other)
        self.now += 86401
        self.call(other, 0)

    def test_trusted_lineage_validation_does_not_charge_and_rejects_wrong_actor_or_root(self):
        execution = self.execution()
        lineage = self.broker.validate_lineage(execution, "a")
        self.assertEqual(lineage["calls_remaining"], 128)
        self.assertEqual(self.broker.validate_message_lineage(lineage, "a"), lineage)
        self.assertEqual(len(self.store.list("broker_charge")), 0)
        with self.assertRaises(AccessDenied):
            self.broker.validate_lineage(execution, "b")
        with self.assertRaises(AccessDenied):
            self.broker.validate_message_lineage({**lineage, "root_id": "forged"}, "a")
        self.broker.stop_execution(execution)
        with self.assertRaises(BudgetExhausted):
            self.broker.validate_lineage(execution, "a")

    def test_exchange_and_execution_identity_reuse_cannot_reset_budgets(self):
        execution = self.execution(origin="user", exchange="stable", identity="stable-execution")
        self.call(execution, 0)
        self.assertEqual(self.broker.open_exchange("stable", "user")["calls"], 1)
        self.assertEqual(self.broker.open_execution(execution, "stable", "a")["calls"], 1)
        with self.assertRaises(Conflict):
            self.broker.open_exchange("stable", "autonomy")
        with self.assertRaises(Conflict):
            self.broker.open_execution(execution, "stable", "b")

    def test_token_counting_and_tool_wait_time_are_not_provider_charges(self):
        execution = self.execution()
        self.now += 500
        self.broker.validate_lineage(execution, "a")
        self.assertEqual(len(self.store.list("broker_charge")), 0)
        identity, result = self.call(execution, 4)
        self.now += 500
        self.broker.validate_lineage(execution, "a")
        self.assertEqual(self.broker.attempt(identity)["service_seconds"], 4)
        self.assertEqual(self.broker.state()["accounts"]["a"]["service"], 4)

    def test_commissioned_sponsorship_does_not_wake_unavailable_cognizing_resident(self):
        parent = self.execution("a", exchange="commission")
        child = self.execution("b", exchange="commission", parent_id=parent, commissioned=True)
        pending = self.request(child)
        self.broker.set_available("b", False)
        self.assertIsNone(self.broker.dispatch())
        self.assertEqual(len(self.store.list("broker_charge")), 0)
        self.broker.set_available("b", True)
        self.assertEqual(self.broker.dispatch()["attempt_id"], pending)
        self.assertEqual(self.finish(pending, 4)["sponsor_id"], "a")

    def test_account_joining_unknown_hold_is_clamped_after_reconciliation_settles_service(self):
        a = self.execution("a")
        first = self.request(a)
        self.broker.dispatch()
        self.now += 40
        self.broker = InferenceBroker(self.store)
        b = self.execution("b")
        self.request(b)
        self.assertFalse(self.broker.state()["accounts"]["b"]["runnable"])
        self.broker.reconcile(first, operator_id="operator", evidence="confirmed provider completion",
                              confirmed_stopped=True, measured_service_seconds=40)
        accounts = self.broker.state()["accounts"]
        self.assertEqual(accounts["a"]["service"], 40)
        self.assertEqual(accounts["b"]["service"], 40)
        self.assertEqual(accounts["a"]["measured_service_seconds"], 40)
        self.assertEqual(accounts["b"]["measured_service_seconds"], 0)

    def test_lost_reconciliation_acknowledgment_does_not_duplicate_service_or_events(self):
        execution = self.execution()
        attempt = self.request(execution)
        self.broker.dispatch()
        self.finish(attempt, 2, outcome="unknown")
        self.assertEqual(self.broker.complete(attempt, outcome="unknown")["status"], "unknown")
        arguments = dict(operator_id="operator", evidence="provider completion receipt",
                         confirmed_stopped=True, measured_service_seconds=8)
        first = self.broker.reconcile(attempt, **arguments)
        count = len(self.store.events())
        self.assertEqual(self.broker.reconcile(attempt, **arguments), first)
        self.assertEqual(len(self.store.events()), count)
        self.assertEqual(self.broker.state()["accounts"]["a"]["service"], 8)
        with self.assertRaises(Conflict):
            self.broker.reconcile(attempt, **{**arguments, "evidence": "different evidence"})

    def test_shared_continuation_is_retained_when_one_of_two_requesters_stops(self):
        a = self.execution("a", exchange="shared-quiet", origin="user")
        b = self.execution("b", exchange="shared-quiet", origin="user")
        self.broker.request_continuation("shared-quiet", a)
        self.broker.request_continuation("shared-quiet", b)
        self.broker.stop_execution(b)
        exchange = self.broker.exchange("shared-quiet")
        self.assertTrue(exchange["continuation_pending"])
        self.assertEqual(exchange["continuation_requesters"], [a])
        self.broker.advance_segment("shared-quiet")
        self.assertEqual(self.broker.exchange("shared-quiet")["continuations_used"], 1)

    def test_successful_participant_completion_preserves_exchange_and_other_speakers(self):
        a = self.execution("a", exchange="discussion", origin="user")
        b = self.execution("b", exchange="discussion", origin="user")
        self.call(a, 4)
        self.broker.finish_execution(a)
        self.assertFalse(self.broker.state()["accounts"]["a"]["runnable"])
        self.assertEqual(self.broker.execution(a)["status"], "completed")
        self.assertEqual(self.broker.execution(b)["status"], "running")
        self.assertEqual(self.broker.exchange("discussion")["calls"], 1)
        self.assertEqual(self.broker.exchange("discussion")["status"], "running")
        self.call(b, 1)
        self.assertEqual(self.broker.exchange("discussion")["calls"], 2)
        self.assertEqual(self.broker.finish_execution(a)["calls"], 1)
        with self.assertRaises(BudgetExhausted):
            self.request(a)

    def test_world_sponsored_full_resident_does_not_consume_recipient_service_or_helper_limit(self):
        execution = self.execution('b', exchange='world-discussion', world_sponsor='a')
        self.broker.set_available('a', False)
        self.call(execution, 7)
        accounts = self.broker.state()['accounts']
        self.assertEqual(accounts['a']['measured_service_seconds'], 7)
        self.assertEqual(accounts['b']['measured_service_seconds'], 0)
        self.assertFalse(self.broker.execution(execution)['helper'])
        self.assertEqual(self.broker.execution(execution)['max_calls'], 128)
        self.broker.set_available('b', False)
        self.request(execution)
        self.assertIsNone(self.broker.dispatch(), 'Sponsorship must not override the recipient\'s own rest')
        with self.assertRaises(AccessDenied):
            self.execution('b', exchange='operator-discussion', origin='user', world_sponsor='a')

    def test_participant_cannot_complete_before_queued_or_running_generation_drains(self):
        execution = self.execution()
        pending = self.request(execution)
        with self.assertRaises(Conflict):
            self.broker.finish_execution(execution)
        self.broker.dispatch()
        with self.assertRaises(Conflict):
            self.broker.finish_execution(execution)
        self.finish(pending, 3)
        self.broker.finish_execution(execution)
        self.assertEqual(self.broker.execution(execution)["status"], "completed")


if __name__ == "__main__":
    unittest.main()
