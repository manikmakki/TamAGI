"""Disposable canonical conversation tests; no resident, provider, or push traffic."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core.household.broker import InferenceBroker
from backend.core.household.conversations import Conversations
from backend.core.household.contracts import AccessDenied, BudgetExhausted, Conflict, ProtocolError
from backend.core.household.store import HouseholdStore


class HouseholdConversationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tamagi-conversations-")
        self.path = Path(self.temporary.name) / "household.sqlite3"
        self.now = 1000.0
        self.store = HouseholdStore(self.path, clock=lambda: self.now)
        self.roster = [
            {"id": "a", "name": "Ember fixture", "kind": "resident"},
            {"id": "b", "name": "Peer", "kind": "resident"},
            {"id": "c", "name": "Third", "kind": "resident"},
            {"id": "helper", "name": "Helper", "kind": "companion"},
            {"id": "operator", "name": "Alex", "kind": "human"},
            {"id": "other-human", "name": "Jordan", "kind": "human"},
        ]
        self.allowed = True
        self.validations = []
        self.service = Conversations(self.store, lambda: self.roster, self.validate)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def validate(self, lineage, author):
        self.validations.append((lineage, author))
        if not self.allowed:
            raise BudgetExhausted("Fixture execution is exhausted or stopped")
        execution_id = lineage.get("execution_id")
        if not execution_id or not execution_id.startswith(author + "-"):
            raise AccessDenied("Execution is not bound to author")
        exchange = "exchange-2" if execution_id.endswith("2") else "exchange-1"
        return {"execution_id": execution_id, "actor_id": author,
                "exchange_id": exchange, "root_id": exchange, "origin": "autonomy",
                "calls_remaining": 12}

    @staticmethod
    def lineage(actor="a", number=1):
        return {"execution_id": f"{actor}-execution-{number}"}

    def create(self, participants=None, **kwargs):
        return self.service.create("a", participants or ["operator"], kwargs.pop("command_id", "create"),
                                   opening=kwargs.pop("opening", "A chosen opening"),
                                   lineage=kwargs.pop("lineage", self.lineage()), **kwargs)

    def test_resident_initiation_replay_survives_execution_end(self):
        result = self.create(opening="Hello @Alex", title="An observation",
                             context_refs=[{"kind": "artifact", "id": "leaf-snapshot", "summary": "A selected leaf"}])
        message = result["message"]
        self.assertEqual(message["author"], "a")
        self.assertEqual(message["addressees"], ["operator"])
        self.assertEqual(message["lineage"]["exchange_id"], "exchange-1")
        self.assertEqual(message["context_refs"][0]["id"], "leaf-snapshot")
        self.assertEqual(self.store.list("interaction"), [])
        self.assertEqual(len(self.store.list("notification_intent")), 1)
        self.assertEqual(self.service.unread("operator")["count"], 1)
        self.allowed = False
        replay = self.create(opening="Hello @Alex", title="An observation",
                             context_refs=[{"kind": "artifact", "id": "leaf-snapshot", "summary": "A selected leaf"}])
        self.assertEqual(replay, result)
        self.assertEqual(len(self.validations), 1)
        self.assertEqual(len(self.store.list("message")), 1)
        self.assertEqual(len(self.store.list("notification_intent")), 1)
        with self.assertRaises(Conflict):
            self.create(opening="Changed input")

    def test_message_and_notification_intent_rollback_together(self):
        original_put = self.store.put

        def fail_notification(kind, *args, **kwargs):
            if kind == "notification_intent":
                raise RuntimeError("Injected crash before notification intent commit")
            return original_put(kind, *args, **kwargs)

        with patch.object(self.store, "put", side_effect=fail_notification):
            with self.assertRaises(RuntimeError):
                self.create()
        for kind in ("conversation", "message", "notification_intent", "thread_read"):
            self.assertEqual(self.store.list(kind), [])
        self.assertEqual(self.store.events(), [])
        result = self.create()
        self.assertEqual(len(self.store.list("conversation")), 1)
        self.assertEqual(self.store.list("notification_intent")[0]["data"]["message_id"], result["message"]["id"])

    def test_stable_mentions_do_not_change_visibility_or_follow_renames(self):
        result = self.create(["operator", "b", "c"], group=True, mentions=["b"])
        conversation_id = result["conversation"]["id"]
        self.assertEqual(self.store.list("notification_intent"), [])
        self.assertEqual(self.service.unread("operator")["count"], 1)
        self.assertEqual(self.service.history("operator", conversation_id)[0]["addressees"], ["b"])
        self.roster[1]["name"] = "Renamed peer"
        self.assertEqual(self.service.history("operator", conversation_id)[0]["addressees"], ["b"])
        self.roster[1]["name"] = self.roster[2]["name"] = "Same"
        with self.assertRaises(ProtocolError):
            self.service.send("a", conversation_id, "Hi @Same", "ambiguous", lineage=self.lineage())
        self.assertEqual(len(self.service.history("a", conversation_id)), 1)
        self.service.send("a", conversation_id, "Hi", "resolved", mentions=["c"], lineage=self.lineage())
        self.assertEqual(self.service.history("operator", conversation_id)[-1]["addressees"], ["c"])

    def test_reply_offers_coalesce_and_closed_offer_never_reopens(self):
        result = self.create(["b", "c"], group=True)
        conversation_id = result["conversation"]["id"]
        self.service.send("a", conversation_id, "A follow-up", "follow-up", lineage=self.lineage())
        self.assertEqual(len(self.service.offers("b")), 1)
        offer = self.service.offers("b")[0]
        self.assertEqual(len(offer["data"]["message_ids"]), 2)
        self.service.decide_offer("b", offer["id"], "silent", "silent")
        self.service.send("a", conversation_id, "Later in the same exchange", "late", lineage=self.lineage())
        self.assertEqual(self.service.offers("b"), [])
        self.assertEqual(len(self.service.offers("c")), 1)
        self.allowed = False
        with self.assertRaises(BudgetExhausted):
            self.service.send("a", conversation_id, "No allowance", "exhausted", lineage=self.lineage())
        self.assertEqual(len(self.service.history("a", conversation_id)), 3)

    def test_companion_membership_is_not_an_invitation_for_later_exchanges(self):
        result = self.create(["helper"])
        conversation_id = result["conversation"]["id"]
        first = self.service.offers("helper")[0]
        self.service.decide_offer("helper", first["id"], "answered", "answered")
        self.service.send("a", conversation_id, "A different exchange", "exchange-2", lineage=self.lineage(number=2))
        self.assertEqual(self.service.offers("helper"), [])
        self.service.invite_companion("a", conversation_id, "helper", "invite-2", self.lineage(number=2))
        self.service.send("a", conversation_id, "Explicitly invited help", "after-invite", lineage=self.lineage(number=2))
        self.assertEqual(len(self.service.offers("helper")), 1)
        self.assertEqual(self.service.offers("helper")[0]["data"]["exchange_id"], "exchange-2")

    def test_reciprocal_replies_retain_answer_links_with_one_exchange_allowance(self):
        result = self.create(["b"])
        conversation_id = result["conversation"]["id"]
        first_offer = self.service.offers("b")[0]
        reply = self.service.send("b", conversation_id, "A peer response", "peer-response",
                                  lineage=self.lineage("b"), reply_offer_id=first_offer["id"])
        self.assertEqual(reply["reply_to"], [result["message"]["id"]])
        author_offer = self.service.offers("a")[0]
        self.service.send("a", conversation_id, "Continuing reciprocally", "author-response",
                          lineage=self.lineage(), reply_offer_id=author_offer["id"])
        repeated_offer = self.service.offers("b")[0]
        self.assertEqual(repeated_offer["id"], first_offer["id"])
        self.assertEqual(repeated_offer["data"]["exchange_id"], "exchange-1")
        self.assertEqual(repeated_offer["data"]["generation"], 2)
        self.assertEqual(repeated_offer["data"]["history"][0]["response_message_id"], reply["id"])
        with self.assertRaises(AccessDenied):
            self.service.send("a", conversation_id, "Cannot answer another participant's offer", "forged-response",
                              lineage=self.lineage(), reply_offer_id=repeated_offer["id"])

    def test_join_requires_choice_grants_history_and_trigger_baseline(self):
        result = self.create(["b"], group=True, opening="Old history")
        conversation_id = result["conversation"]["id"]
        trigger = self.service.send("a", conversation_id, "Let's invite Alex", "trigger", lineage=self.lineage())
        invitation = self.service.invite("a", conversation_id, "operator", "invite", trigger["id"])
        self.assertIn("full prior thread", invitation["data"]["history_notice"])
        with self.assertRaises(AccessDenied):
            self.service.history("operator", conversation_id)
        with self.assertRaises(AccessDenied):
            self.service.join("a", conversation_id, "operator", "forced-join")
        joined = self.service.join("operator", conversation_id, "operator", "join")
        self.assertEqual(joined["data"]["membership_version"], 2)
        self.assertEqual(len(self.service.history("operator", conversation_id)), 2)
        unread = self.service.unread("operator")
        self.assertEqual(unread["threads"][0]["count"], 1)
        self.assertEqual(unread["threads"][0]["latest"]["id"], trigger["id"])
        self.assertEqual(self.store.list("notification_intent"), [])  # no historical alerts on joining
        self.service.leave("operator", conversation_id, "leave")
        self.assertEqual(self.service.unread("operator")["count"], 0)
        with self.assertRaises(AccessDenied):
            self.service.history("operator", conversation_id)

    def test_join_without_trigger_baselines_at_current_end(self):
        result = self.create(["b"], group=True)
        conversation_id = result["conversation"]["id"]
        self.service.invite("a", conversation_id, "operator", "invite")
        self.service.join("operator", conversation_id, "operator", "join")
        self.assertEqual(len(self.service.history("operator", conversation_id)), 1)
        self.assertEqual(self.service.unread("operator")["count"], 0)

    def test_read_snapshot_does_not_clear_unread_or_include_later_messages(self):
        result = self.create()
        conversation_id = result["conversation"]["id"]
        snapshot = self.service.snapshot("operator", conversation_id)
        self.assertEqual(self.service.unread("operator")["count"], 1)
        second = self.service.send("a", conversation_id, "Arrived after rendering", "second", lineage=self.lineage())
        with self.assertRaises(Conflict):
            self.service.mark_read("operator", conversation_id, second["sequence"], "too-new",
                                   snapshot_id=snapshot["snapshot_id"])
        self.service.mark_read("operator", conversation_id, snapshot["watermark"], "read-first",
                               snapshot_id=snapshot["snapshot_id"])
        unread = self.service.unread("operator")
        self.assertEqual(unread["threads"][0]["count"], 1)
        self.assertEqual(unread["threads"][0]["latest"]["id"], second["id"])
        self.assertEqual(len(self.service.history("operator", conversation_id, through=snapshot["watermark"])), 1)

    def test_read_state_survives_restart_reconciles_devices_and_excludes_own_messages(self):
        first = self.create()
        conversation_id = first["conversation"]["id"]
        old_snapshot = self.service.snapshot("operator", conversation_id)
        self.service.send("a", conversation_id, "New message", "new", lineage=self.lineage())
        self.service.mark_read("operator", conversation_id, 2, "explicit-read", explicit=True)
        self.service.mark_read("operator", conversation_id, 1, "older-device", snapshot_id=old_snapshot["snapshot_id"])
        self.assertEqual(self.service.unread("operator")["count"], 0)
        self.service.send("operator", conversation_id, "My own reply", "own")
        self.assertEqual(self.service.unread("operator")["count"], 0)
        second = self.create(command_id="another-thread")
        self.service.send("a", second["conversation"]["id"], "Same unread thread", "another-message", lineage=self.lineage())
        self.assertEqual(self.service.unread("operator")["count"], 1)
        self.store.close()
        self.store = HouseholdStore(self.path, clock=lambda: self.now)
        self.service = Conversations(self.store, lambda: self.roster, self.validate)
        self.assertEqual(self.service.unread("operator")["count"], 1)
        self.assertEqual(self.service.unread("operator")["threads"][0]["count"], 2)

    def test_private_access_forged_lineage_and_snapshot_ownership_fail_closed(self):
        result = self.create(["operator", "other-human"], group=True)
        conversation_id = result["conversation"]["id"]
        with self.assertRaises(AccessDenied):
            self.service.history("b", conversation_id)
        with self.assertRaises(AccessDenied):
            self.service.send("b", conversation_id, "Not a member", "outsider", lineage=self.lineage("b"))
        with self.assertRaises(AccessDenied):
            self.service.send("a", conversation_id, "Forged exchange", "forged",
                              lineage={**self.lineage(), "exchange_id": "different"})
        snapshot = self.service.snapshot("other-human", conversation_id)
        with self.assertRaises(AccessDenied):
            self.service.mark_read("operator", conversation_id, 1, "stolen-snapshot", snapshot_id=snapshot["snapshot_id"])
        with self.assertRaises(AccessDenied):
            self.service.mark_read("a", conversation_id, 1, "resident-read", explicit=True)
        unvalidated = Conversations(self.store, lambda: self.roster)
        with self.assertRaises(AccessDenied):
            unvalidated.send("a", conversation_id, "No trusted callback", "unchecked", lineage=self.lineage())

    def test_invalid_opening_and_context_leave_no_partial_thread(self):
        with self.assertRaises(ProtocolError):
            self.create(opening=" ")
        with self.assertRaises(ProtocolError):
            self.create(context_refs=[{"kind": "private_filesystem", "id": "/private/path"}])
        with self.assertRaises(ProtocolError):
            self.create(participants=[{"id": "operator"}])
        with self.assertRaises(ProtocolError):
            self.create(group="false")
        self.assertEqual(self.store.list("conversation"), [])
        self.assertEqual(self.store.list("message"), [])

    def test_actual_broker_quiet_lineage_preserves_budget_and_committed_replay(self):
        broker = InferenceBroker(self.store)
        broker.register_resident("a")
        broker.open_exchange("user-exchange", "user")
        broker.open_execution("quiet-execution", "user-exchange", "a")
        broker.request_continuation("user-exchange", "quiet-execution")
        broker.advance_segment("user-exchange")
        trusted = broker.validate_lineage("quiet-execution", "a")
        service = Conversations(self.store, lambda: self.roster, broker.validate_message_lineage)
        result = service.create("a", ["operator"], "quiet-opening", opening="An explicit quiet-work message",
                                lineage=trusted)
        self.assertEqual(result["message"]["lineage"]["origin"], "user")
        self.assertEqual(result["message"]["lineage"]["segment"], 1)
        self.assertEqual(broker.exchange("user-exchange")["calls"], 0)
        self.assertEqual(broker.exchange("user-exchange")["continuations_used"], 1)
        self.assertEqual(len(self.store.list("broker_exchange")), 1)
        broker.stop_execution("quiet-execution")
        self.assertEqual(service.create("a", ["operator"], "quiet-opening",
                                        opening="An explicit quiet-work message", lineage=trusted), result)
        with self.assertRaises(BudgetExhausted):
            service.send("a", result["conversation"]["id"], "No new work after Stop", "after-stop", lineage=trusted)
        self.assertEqual(len(self.store.list("message")), 1)
        self.assertEqual(self.store.list("broker_charge"), [])


if __name__ == "__main__":
    unittest.main()
