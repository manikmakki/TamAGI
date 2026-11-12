"""Disposable shared-world command, visibility, import, and artifact contracts."""

import tempfile
import unittest
from pathlib import Path

from backend.core.household.contracts import AccessDenied, Conflict, NotFound, ProtocolError
from backend.core.household.store import HouseholdStore
from backend.core.household.world import World


class HouseholdWorldTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tamagi-world-")
        self.store = HouseholdStore(Path(self.temporary.name) / "household.sqlite3", clock=lambda: 1000)
        self.world = World(self.store)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def create(self, object_id="home", **kwargs):
        return self.world.create("a", kwargs.pop("command_id", "create-" + object_id), object_id,
                                 kwargs.pop("kind", "place"), kwargs.pop("name", "My home"), **kwargs)

    def test_visibility_ownership_revision_and_command_replay(self):
        private = self.create()
        self.assertEqual(self.world.list("b"), [])
        with self.assertRaises(AccessDenied):
            self.world.get("b", "home")
        shared = self.world.update("a", "share", "home", {"visibility": "shared"}, private["revision"])
        self.assertEqual(self.world.get("b", "home"), shared)
        with self.assertRaises(AccessDenied):
            self.world.update("b", "edit-owner", "home", {"name": "Taken"}, shared["revision"])
        with self.assertRaises(Conflict):
            self.world.update("a", "stale", "home", {"name": "Stale"}, private["revision"])
        renamed = self.world.update("a", "rename", "home", {"name": "Renamed"}, shared["revision"])
        replay = self.world.update("a", "rename", "home", {"name": "Renamed"}, shared["revision"])
        self.assertEqual(replay, renamed)
        self.assertEqual(self.world.get("a", "home")["revision"], 3)
        effects = [event for event in self.store.events() if event["kind"] == "world_changed"]
        self.assertEqual([event["data"]["revision"] for event in effects], [1, 2, 3])
        with self.assertRaises(Conflict):
            self.world.update("a", "rename", "home", {"name": "Different"}, shared["revision"])
        with self.assertRaises(ProtocolError):
            self.world.update("a", "change-owner", "home", {"owner": "b"}, renamed["revision"])

    def test_explicit_visibility_grants_do_not_publish_entire_home(self):
        self.create(visible_to=["b"])
        self.create("private-object", kind="object", name="Private notebook", location="home")
        self.assertEqual([record["id"] for record in self.world.list("b")], ["home"])
        with self.assertRaises(AccessDenied):
            self.world.get("b", "private-object")
        self.assertEqual(self.world.list("c"), [])

    def test_location_types_and_cycles_fail_without_mutation(self):
        self.create()
        self.create("room", location="home")
        self.create("leaf", kind="object", location="room")
        with self.assertRaises(Conflict):
            self.world.update("a", "cycle", "home", {"location": "room"}, 1)
        with self.assertRaises(Conflict):
            self.create("inside-object", location="leaf")
        self.assertIsNone(self.world.get("a", "home")["data"]["location"])
        self.assertIsNone(self.store.get("world_object", "inside-object"))

    def test_artifact_snapshot_is_immutable_and_explicitly_shared(self):
        self.create(visibility="shared")
        published = self.world.publish_artifact("a", "publish", "home", "leaf-v1", "a" * 64, 1,
                                                filename="leaf.html", size=123)
        reference_id = published["artifact"]["id"]
        self.assertEqual(self.world.artifact("b", "home", reference_id)["data"]["sha256"], "a" * 64)
        replay = self.world.publish_artifact("a", "publish", "home", "leaf-v1", "a" * 64, 1,
                                             filename="leaf.html", size=123)
        self.assertEqual(replay, published)
        with self.assertRaises(Conflict):
            self.world.publish_artifact("a", "rewrite", "home", "leaf-v1", "b" * 64, 2,
                                        filename="leaf.html", size=123)
        with self.assertRaises(ProtocolError):
            self.world.publish_artifact("a", "private-path", "home", "leaf-v2", "b" * 64, 2,
                                        filename="/private/workshop/leaf.html")
        with self.assertRaises(AccessDenied):
            self.world.publish_artifact("b", "foreign-publish", "home", "leaf-v2", "b" * 64, 2)
        self.create("unpublished", visibility="shared")
        with self.assertRaises(NotFound):
            self.world.artifact("b", "unpublished", reference_id)
        self.assertEqual(len(self.store.list("world_artifact")), 1)
        self.assertEqual(self.world.get("a", "home")["revision"], 2)

    def test_failed_publish_rolls_back_new_artifact_reference(self):
        self.create()
        self.world.update("a", "advance", "home", {"name": "Changed"}, 1)
        with self.assertRaises(Conflict):
            self.world.publish_artifact("a", "stale-publish", "home", "leaf-v1", "a" * 64, 1)
        self.assertEqual(self.store.list("world_artifact"), [])
        self.assertEqual(self.world.get("a", "home")["data"]["artifact_refs"], [])

    def test_namespaced_import_retains_mapping_without_merging_names(self):
        records = [{"id": "leaf", "kind": "object", "name": "Leaf", "location": "home"},
                   {"id": "home", "kind": "place", "name": "Home"}]
        imported = self.world.import_records("a", "import-a", "a-domain", records)
        other = self.world.import_records("b", "import-b", "b-domain", records)
        self.assertEqual(imported["data"]["mapping"], {"home": "a-domain:home", "leaf": "a-domain:leaf"})
        self.assertEqual(self.world.get("a", "a-domain:leaf")["data"]["location"], "a-domain:home")
        self.assertEqual(other["data"]["mapping"]["home"], "b-domain:home")
        self.assertEqual(len(self.store.list("world_object")), 4)
        self.assertEqual(self.world.import_records("a", "reimport", "a-domain", records), imported)
        with self.assertRaises(AccessDenied):
            self.world.get("b", "a-domain:home")
        changed = [{"id": "home", "kind": "place", "name": "Changed"}]
        with self.assertRaises(Conflict):
            self.world.import_records("a", "different-manifest", "a-domain", changed)

    def test_import_failure_and_collisions_leave_no_partial_state(self):
        records = [{"id": "home", "kind": "place", "name": "Home"},
                   {"id": "leaf", "kind": "object", "name": "Leaf", "location": "absent"}]
        with self.assertRaises(NotFound):
            self.world.import_records("a", "missing-location", "domain", records)
        self.assertEqual(self.store.list("world_object"), [])
        self.assertEqual(self.store.list("world_import"), [])
        self.assertEqual(self.store.events(), [])
        cyclic = [{"id": "one", "kind": "place", "name": "One", "location": "two"},
                  {"id": "two", "kind": "place", "name": "Two", "location": "one"}]
        with self.assertRaises(Conflict):
            self.world.import_records("a", "cycle", "domain", cyclic)
        self.assertEqual(self.store.list("world_object"), [])
        self.create("domain:home")
        with self.assertRaises(Conflict):
            self.world.import_records("a", "collision", "domain", [{"id": "home", "kind": "place", "name": "Home"}])
        self.assertEqual(len(self.store.list("world_object")), 1)

    def test_malformed_types_cannot_silently_change_visibility(self):
        with self.assertRaises(ProtocolError):
            self.create(visible_to="everyone")
        with self.assertRaises(ProtocolError):
            self.create(attributes=[1, 2])
        with self.assertRaises(ProtocolError):
            self.create(location=["private-path"])
        self.assertEqual(self.store.list("world_object"), [])


if __name__ == "__main__":
    unittest.main()
