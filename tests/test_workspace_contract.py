"""Directory guidance and generated workers agree on the active instance."""

from pathlib import Path
import sys
import tempfile
import unittest

from backend.config import RuntimeConfig
from backend.core.runtime import RuntimeService, RuntimeStore
from backend.core.runtime.adaptation import Adaptation, ProcessTool
from backend.core.runtime.workspace import MARKER, paths, prepare
from backend.skills.registry import SkillRegistry


class WorkspaceContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = RuntimeStore(self.root / "home/runtime.sqlite3")
        self.addCleanup(self.store.close)
        self.runtime = RuntimeService(
            self.store,
            self.root / "home/workspace",
            RuntimeConfig(),
            self.root / "canonical-source",
        )
        self.adaptation = Adaptation(self.runtime, SkillRegistry())

    def test_hydration_reports_and_prepares_real_paths_without_moving_work(self):
        workspace = self.runtime.workspace
        note = workspace / "research/earlier.md"
        note.write_text("An existing working note")
        self.runtime.release_source_root = self.root / "selected-release/source"
        RuntimeService(self.store, workspace, RuntimeConfig(), self.runtime.source_root)
        context = self.runtime.context()
        for directory in paths(workspace).values():
            self.assertTrue(Path(directory).is_dir())
            self.assertIn(directory, context)
        self.assertIn(str(self.runtime.release_source_root), context)
        self.assertIn(str(self.runtime.source_root), context)
        self.assertEqual(note.read_text(), "An existing working note")
        self.assertIn("not automatically a callable", context)
        self.assertIn("JSON object", (workspace / "skills/README.md").read_text())

    def test_owned_guides_refresh_but_authored_guides_are_preserved(self):
        workspace = self.runtime.workspace
        (workspace / "WORKSPACE.md").write_text(MARKER + "old platform guide")
        (workspace / "skills/README.md").write_text("My own development notes")
        prepare(workspace)
        self.assertIn("# Your workspace", (workspace / "WORKSPACE.md").read_text())
        self.assertEqual(
            (workspace / "skills/README.md").read_text(), "My own development notes"
        )

    async def test_relative_worker_is_frozen_and_receives_workspace_in_both_modes(self):
        workspace = self.runtime.workspace
        source = workspace / "skills/probe/main.py"
        source.parent.mkdir()
        source.write_text(
            "import json,os,sys\n"
            "args=json.load(sys.stdin)\n"
            "print(json.dumps({'success':True,'output':args['label'],'data':"
            "{'cwd':os.getcwd(),'workspace':os.environ['TAMAGI_WORKSPACE'],"
            "'workshop':os.environ['TAMAGI_WORKSHOP'],"
            "'skills':os.environ['TAMAGI_SKILLS'],"
            "'scratch':os.environ['TAMAGI_SCRATCH']}}))\n"
        )
        proposal = self.adaptation.propose(
            "tool",
            {
                "name": "path_probe",
                "description": "Observe the actual process paths",
                "command": [sys.executable, "skills/probe/main.py"],
                "parameters": {"label": {"type": "string", "required": True}},
            },
            "Verify the tool contract independently of backend cwd",
        )
        source.write_text("raise RuntimeError('editable source has changed')")
        definition = proposal["data"]["value"]
        frozen = Path(definition["command"][1])
        self.assertTrue(frozen.is_relative_to(workspace / ".tools"))
        result = await ProcessTool(definition, workspace).execute(label="live")
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output, "live")
        self.assertEqual(result.data["cwd"], str(workspace))
        self.assertEqual(result.data["workspace"], str(workspace))
        for directory in ("workshop", "skills", "scratch"):
            self.assertEqual(result.data[directory], str(workspace / directory))

        trial = await self.adaptation.test_tool(proposal["id"], {"label": "trial"})
        self.assertTrue(trial["success"], trial.get("error"))
        self.assertEqual(trial["output"], "trial")
        trial_workspace = Path(trial["data"]["workspace"])
        self.assertNotEqual(trial_workspace, workspace)
        self.assertEqual(trial["data"]["cwd"], str(trial_workspace))
        for directory in ("workshop", "skills", "scratch"):
            self.assertEqual(trial["data"][directory], str(trial_workspace / directory))
        self.assertEqual(self.store.list("tool_trial")[0]["data"]["success"], True)

    def test_full_json_schema_is_rejected_with_actionable_guidance(self):
        with self.assertRaisesRegex(ValueError, "skills/README.md"):
            self.adaptation.propose(
                "tool",
                {
                    "name": "incorrect_schema",
                    "description": "A tool with the wrong parameter shape",
                    "command": [sys.executable, "-c", "pass"],
                    "parameters": {"type": "object", "properties": {}},
                },
                "Show the supported definition",
            )
