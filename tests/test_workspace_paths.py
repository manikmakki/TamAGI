"""Run commands against the active workspace rather than the backend's cwd."""

import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.config import TamAGIConfig
from backend.skills.exec_skill import ExecSkill


class WorkspaceExecTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "instance" / "workspace"
        self.workspace.mkdir(parents=True)
        self.config = TamAGIConfig()
        self.config.workspace.path = str(self.workspace)
        self.config.guardrails.exec_trust.safe.append(Path(sys.executable).name)
        self.runtime = SimpleNamespace(authority="guarded")
        self.config_patch = patch(
            "backend.skills.exec_skill.get_config", return_value=self.config
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.runtime_patch = patch(
            "backend.core.runtime.get_runtime", return_value=self.runtime
        )
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.skill = ExecSkill()

    @staticmethod
    def command(program):
        return f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"

    async def test_relative_directory_and_environment_use_active_workspace(self):
        workshop = self.workspace / "workshop"
        workshop.mkdir()
        unrelated = self.root / "instance" / "workshop"
        unrelated.mkdir()
        program = (
            "import json,os;from pathlib import Path;"
            "Path('probe.txt').write_text('created here');"
            "print(json.dumps({'cwd':os.getcwd(),'paths':"
            "{k:os.environ[k] for k in "
            "('TAMAGI_WORKSPACE','TAMAGI_WORKSHOP','TAMAGI_SKILLS','TAMAGI_SCRATCH')},"
            "'inherited':os.environ['TAMAGI_PATH_TEST']}))"
        )
        for authority in ("guarded", "unrestricted_host"):
            with self.subTest(authority=authority):
                self.runtime.authority = authority
                with patch.dict(
                    os.environ,
                    {
                        "TAMAGI_WORKSPACE": "/stale/installation/workspace",
                        "TAMAGI_WORKSHOP": "/stale/installation/workshop",
                        "TAMAGI_SKILLS": "/stale/installation/skills",
                        "TAMAGI_SCRATCH": "/stale/installation/scratch",
                        "TAMAGI_PATH_TEST": "preserved",
                    },
                ):
                    result = await self.skill.execute(
                        command=self.command(program), working_dir="workshop"
                    )
                self.assertTrue(result.success, result.error)
                output = json.loads(result.output)
                self.assertEqual(output["cwd"], str(workshop))
                self.assertEqual(result.data["working_dir"], str(workshop))
                self.assertEqual(output["inherited"], "preserved")
                self.assertEqual(
                    output["paths"],
                    {
                        "TAMAGI_WORKSPACE": str(self.workspace),
                        "TAMAGI_WORKSHOP": str(workshop),
                        "TAMAGI_SKILLS": str(self.workspace / "skills"),
                        "TAMAGI_SCRATCH": str(self.workspace / "scratch"),
                    },
                )
                self.assertEqual((workshop / "probe.txt").read_text(), "created here")
                self.assertFalse((unrelated / "probe.txt").exists())

    async def test_default_and_absolute_directories_preserve_host_authority(self):
        absolute = self.root / "host-directory"
        absolute.mkdir()
        for authority in ("guarded", "unrestricted_host"):
            self.runtime.authority = authority
            for requested, expected in (("", self.workspace), (str(absolute), absolute)):
                with self.subTest(authority=authority, requested=requested):
                    result = await self.skill.execute(
                        command=self.command("import os;print(os.getcwd())"),
                        working_dir=requested,
                    )
                    self.assertTrue(result.success, result.error)
                    self.assertEqual(result.output, str(expected))

    async def test_tilde_in_directory_argument_is_literal_workspace_path(self):
        literal = self.workspace / "~" / "notes"
        literal.mkdir(parents=True)
        for authority in ("guarded", "unrestricted_host"):
            with self.subTest(authority=authority):
                self.runtime.authority = authority
                result = await self.skill.execute(
                    command=self.command("import os;print(os.getcwd())"),
                    working_dir="~/notes",
                )
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.output, str(literal))
