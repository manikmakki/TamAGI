"""Graph writes must report persistence failures and retain the last good file."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from backend.config import RuntimeConfig
from backend.core.runtime import RuntimeService, RuntimeStore
from backend.core.self_model.store import SelfModel
from backend.skills.registry import SkillRegistry
from backend.skills.write_world_graph_skill import WriteWorldGraphSkill


class WorldGraphPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tamagi-graph-save-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime_patch = patch("backend.core.runtime.get_runtime", return_value=None)
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    def model(self, name="graph"):
        model = SelfModel(self.root / name / "self_model.json")
        model._apply_add_node("lore", {"id": "source", "description": "A lasting memory"})
        model._apply_add_node("lore", {"id": "target", "description": "Another memory"})
        model.save()
        return model

    def operation(self, name):
        return {
            "new": {"action": "add_node", "node_type": "lore", "attributes": {
                "id": "new", "description": "A different discovery"}},
            "merge": {"action": "add_node", "node_type": "lore", "attributes": {
                "description": "A lasting memory", "context": "world_genre"}},
            "update": {"action": "update_node", "node_id": "source", "attributes": {
                "description": "A revised memory"}},
            "edge": {"action": "add_edge", "source_id": "source", "target_id": "target",
                     "edge_type": "relates_to"},
            "delete": {"action": "delete_node", "node_id": "source"},
        }[name]

    def assert_saved_graph(self, model):
        restored = SelfModel(model._data_path)
        restored.load()
        self.assertEqual(restored.get_all_nodes(), model.get_all_nodes())
        self.assertEqual(restored.get_edges(), model.get_edges())

    async def test_all_write_paths_report_save_failure_instead_of_success(self):
        for name in ("new", "merge", "update", "edge", "delete"):
            with self.subTest(operation=name):
                model = self.model(name)
                before = model._data_path.read_bytes()
                skill = WriteWorldGraphSkill(SimpleNamespace(self_model=model))
                with patch.object(model, "save", side_effect=OSError("disk full")):
                    with self.assertLogs("tamagi", level="ERROR") as logged:
                        result = await skill.execute(**self.operation(name))
                self.assertFalse(result.success)
                self.assertIn("disk full", result.error)
                self.assertIn("saving", result.output.lower())
                self.assertNotIn("sm_mutation", result.data)
                self.assertIn("disk full", "\n".join(logged.output))
                self.assertTrue(any(record.exc_info for record in logged.records))
                self.assertEqual(model._data_path.read_bytes(), before)
                restored = SelfModel(model._data_path)
                restored.load()
                self.assertEqual(len(restored.get_all_nodes()), 2)

    async def test_successful_writes_survive_reloading(self):
        for name in ("new", "merge", "update", "edge", "delete"):
            with self.subTest(operation=name):
                model = self.model(name)
                before = model._data_path.read_bytes()
                result = await WriteWorldGraphSkill(SimpleNamespace(self_model=model)).execute(
                    **self.operation(name))
                self.assertTrue(result.success, result.error)
                self.assertNotEqual(model._data_path.read_bytes(), before)
                self.assert_saved_graph(model)

    async def test_optional_edge_failures_are_logged_without_claiming_added_edges(self):
        for name in ("new", "merge", "update", "context"):
            with self.subTest(operation=name):
                model = self.model(name)
                agent = SimpleNamespace(self_model=model)
                if name == "context":
                    model._apply_add_node("location", {"id": "harbor", "name": "Harbor"})
                    agent._world_thread = SimpleNamespace(get_current_location=lambda: "Harbor")
                    operation = {"action": "add_node", "node_type": "event",
                                 "attributes": {"id": "visit", "description": "Arrived today"}}
                else:
                    operation = self.operation(name)
                    operation["relationships"] = [{"target_id": "target", "edge_type": "relates_to"}]
                with patch.object(model, "_apply_add_edge", side_effect=ValueError("edge rejected")):
                    with self.assertLogs("tamagi", level="ERROR") as logged:
                        result = await WriteWorldGraphSkill(agent).execute(**operation)
                self.assertTrue(result.success, result.error)
                self.assertEqual(model.get_edges(), [])
                self.assertNotIn("(+1", result.output)
                self.assertNotIn("1 explicit", result.output)
                self.assertIn("edge rejected", "\n".join(logged.output))
                self.assertTrue(any(record.exc_info for record in logged.records))
                self.assert_saved_graph(model)

    def test_auto_wire_failure_is_logged(self):
        model = self.model()
        model._apply_add_node("location", {"id": "harbor", "name": "Harbor"})
        model._apply_add_node("quest", {"id": "quest", "title": "Explore Harbor"})
        with patch.object(model, "_apply_add_edge", side_effect=ValueError("auto edge rejected")):
            with self.assertLogs("tamagi.self_model", level="ERROR") as logged:
                self.assertEqual(model.auto_wire_node("quest"), 0)
        self.assertIn("auto edge rejected", "\n".join(logged.output))

    def test_mid_write_failure_preserves_previous_file_and_cleans_temporary_file(self):
        model = self.model()
        before = model._data_path.read_bytes()
        model._apply_update_node("source", {"description": "Unsaved"})

        def partial_write(data, stream, **kwargs):
            stream.write('{"nodes": [')
            stream.flush()
            raise OSError("disk full during write")

        with patch("backend.core.self_model.store.json.dump", side_effect=partial_write):
            with self.assertRaisesRegex(OSError, "disk full during write"):
                model.save()
        self.assertEqual(model._data_path.read_bytes(), before)
        self.assertEqual(list(model._data_path.parent.iterdir()), [model._data_path])
        self.assertEqual(json.loads(before)["node_count"], 2)

    def test_serialization_failure_preserves_previous_file(self):
        model = self.model()
        before = model._data_path.read_bytes()
        circular = {}
        circular["self"] = circular
        model._apply_update_node("source", {"invalid": circular})
        with self.assertRaisesRegex(ValueError, "Circular reference"):
            model.save()
        self.assertEqual(model._data_path.read_bytes(), before)
        self.assertEqual(list(model._data_path.parent.iterdir()), [model._data_path])

    def test_sync_and_replace_failures_preserve_previous_file(self):
        for failure in ("os.fsync", "os.replace"):
            with self.subTest(failure=failure):
                model = self.model(failure.replace(".", "-"))
                before = model._data_path.read_bytes()
                model._apply_update_node("source", {"description": "Unsaved"})
                with patch("backend.core.self_model.store." + failure, side_effect=OSError("storage unavailable")):
                    with self.assertRaisesRegex(OSError, "storage unavailable"):
                        model.save()
                self.assertEqual(model._data_path.read_bytes(), before)
                self.assertEqual(list(model._data_path.parent.iterdir()), [model._data_path])

    def test_failed_first_save_leaves_no_partial_destination(self):
        model = SelfModel(self.root / "new" / "self_model.json")
        model._apply_add_node("lore", {"id": "source", "description": "A memory"})
        with patch("backend.core.self_model.store.json.dump", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                model.save()
        self.assertFalse(model._data_path.exists())
        self.assertEqual(list(model._data_path.parent.iterdir()), [])

    def test_successful_replacement_preserves_permissions_and_supports_export_path(self):
        model = self.model()
        model._data_path.chmod(0o640)
        model._apply_update_node("source", {"description": "Saved revision"})
        self.assertEqual(model.save(), model._data_path)
        self.assertEqual(model._data_path.stat().st_mode & 0o777, 0o640)
        self.assert_saved_graph(model)
        exported = self.root / "exports" / "world.json"
        self.assertEqual(model.save(exported), exported)
        self.assertEqual(json.loads(exported.read_text())["nodes"], model.get_all_nodes())
        self.assertEqual(list(exported.parent.iterdir()), [exported])

    async def test_runtime_receipt_reports_failed_save_even_if_canonical_record_committed(self):
        model = self.model()
        before = model._data_path.read_bytes()
        store = RuntimeStore(self.root / "runtime.sqlite3")
        self.addCleanup(store.close)
        runtime = RuntimeService(store, self.root / "workspace", RuntimeConfig(), self.root)
        registry = SkillRegistry()
        registry.register(WriteWorldGraphSkill(SimpleNamespace(self_model=model)))
        with patch("backend.core.runtime.get_runtime", return_value=runtime):
            with patch("backend.core.self_model.store.os.replace", side_effect=OSError("replacement denied")):
                with self.assertLogs("tamagi", level="ERROR"):
                    result = await registry.execute("world_graph", **self.operation("update"))
            self.assertFalse(result.success)
            receipt = store.receipt(result.data["receipt_id"])
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse(receipt["payload"]["result"]["success"])
            self.assertIn("replacement denied", receipt["payload"]["result"]["error"])
            self.assertEqual(model._data_path.read_bytes(), before)
            restored = SelfModel(model._data_path)
            restored.load()
            self.assertEqual(restored.get_node("source")["description"], "A revised memory")
