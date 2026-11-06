"""Canonical commands commit once, independently of any resident installation."""

import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.core.household.contracts import Conflict
from backend.core.household.store import HouseholdStore


class HouseholdStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tamagi-store-test-")
        self.path = Path(self.temp.name) / "household.sqlite3"
        self.store = HouseholdStore(self.path, clock=lambda: 100.0)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_replay_retains_one_effect_event_and_original_result_after_restart(self):
        def apply():
            self.store.put("object", "stone", {"position": 1}, expected=0)
            return {"object_id": "stone", "event": self.store.event("moved", "a", {})["id"]}
        original = self.store.command("a", "cmd", "move", {"position": 1}, apply)
        self.store.close()
        self.store = HouseholdStore(self.path)
        replay = self.store.command("a", "cmd", "move", {"position": 1}, lambda: self.fail("replayed effect"))
        self.assertEqual(original, replay)
        self.assertEqual(len(self.store.list("object")), 1)
        self.assertEqual([row["kind"] for row in self.store.events()], ["moved", "command_committed"])
        with self.assertRaises(Conflict):
            self.store.command("a", "cmd", "move", {"position": 2}, lambda: {})

    def test_failed_command_and_caught_nested_transaction_roll_back(self):
        def fail():
            self.store.put("object", "bad", {})
            self.store.event("bad", "a", {})
            raise RuntimeError("crash before commit")
        with self.assertRaises(RuntimeError):
            self.store.command("a", "failed", "move", {}, fail)
        self.assertEqual(self.store.list("object"), [])
        self.assertEqual(self.store.events(), [])
        with self.store.transaction():
            self.store.put("object", "outer", {})
            try:
                with self.store.transaction():
                    self.store.put("object", "inner", {})
                    raise ValueError("caught inner failure")
            except ValueError:
                pass
        self.assertIsNotNone(self.store.get("object", "outer"))
        self.assertIsNone(self.store.get("object", "inner"))

    def test_concurrent_connections_admit_identical_command_once(self):
        other = HouseholdStore(self.path)
        def commit(store):
            return store.command("a", "same", "add", {}, lambda: store.put("object", "one", {"saved": True}, expected=0))
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(commit, (self.store, other)))
            self.assertEqual(results[0], results[1])
            self.assertEqual(self.store.get("object", "one")["revision"], 1)
        finally:
            other.close()

    def test_foreign_and_future_database_rejected_before_mutation(self):
        path = self.path.parent / "resident.sqlite3"
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE private(value)")
        db.commit()
        db.close()
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "not a household"):
            HouseholdStore(path)
        self.assertEqual(path.read_bytes(), before)
        self.store.set_meta("schema", 999)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            HouseholdStore(self.path)

    def test_optimistic_revisions_and_finite_json(self):
        one = self.store.put("object", "one", {"v": 1}, expected=0)
        with self.assertRaises(Conflict):
            self.store.put("object", "one", {"v": 2}, expected=0)
        self.store.put("object", "one", {"v": 2}, expected=one["revision"])
        with self.assertRaises(ValueError):
            self.store.put("object", "nonfinite", {"v": float("nan")})
