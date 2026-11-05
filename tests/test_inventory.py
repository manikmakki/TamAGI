"""Workshop discovery and downloads cannot traverse into other host files."""

from pathlib import Path
import tempfile
import unittest

from backend.core.runtime import RuntimeService, RuntimeStore
from backend.core.runtime.inventory import Inventory


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = RuntimeStore(self.root / "runtime.sqlite3")
        self.addCleanup(self.store.close)
        self.runtime = RuntimeService(self.store, self.root / "workspace")
        self.inventory = Inventory(self.runtime)

    def test_unregistered_files_and_separate_saved_versions_are_discovered(self):
        creation = self.runtime.workspace / "workshop/leaf.md"
        creation.write_text("first")
        saved = self.runtime.register_artifact("workshop/leaf.md")
        creation.write_text("current")
        (self.runtime.workspace / "workshop/new.txt").write_text("not registered")
        (self.runtime.workspace / "research/earlier.txt").write_text("earlier")
        earlier = self.runtime.register_artifact("research/earlier.txt")
        data = self.inventory.list()
        entries = {r["path"]: r for r in data["items"]}
        self.assertEqual(set(entries), {"leaf.md", "new.txt"})
        self.assertEqual(entries["leaf.md"]["versions"][0]["id"], saved["id"])
        self.assertEqual(data["earlier_saved_artifacts"][0]["id"], earlier["id"])
        with self.inventory.open("leaf.md") as file:
            self.assertEqual(file.read(), b"current")
        self.assertEqual(
            (self.runtime.workspace / saved["data"]["path"]).read_text(), "first"
        )

    def test_parent_paths_absolute_paths_and_symlinks_are_not_available(self):
        secret = self.root / "outside.txt"
        secret.write_text("outside")
        (self.inventory.root / "outside-link").symlink_to(secret)
        (self.inventory.root / "directory-link").symlink_to(
            self.root, target_is_directory=True
        )
        for path in [
            "../outside.txt",
            str(secret),
            "outside-link",
            "directory-link/outside.txt",
        ]:
            with self.subTest(path=path), self.assertRaises((OSError, ValueError)):
                self.inventory.open(path)
        self.assertEqual(self.inventory.list()["items"], [])

    def test_open_descriptor_keeps_selected_file_when_path_is_replaced(self):
        path = self.inventory.root / "selected.txt"
        path.write_text("selected")
        with self.inventory.open("selected.txt") as file:
            path.unlink()
            path.symlink_to(self.root / "private.txt")
            (self.root / "private.txt").write_text("unrelated")
            self.assertEqual(file.read(), b"selected")
