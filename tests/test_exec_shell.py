"""Exercise the real subprocess contract, without network or model calls."""

import asyncio
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

from backend.config import RuntimeConfig, TamAGIConfig, get_config, set_config
from backend.core.runtime import RuntimeService, RuntimeStore, set_runtime
from backend.skills.exec_skill import ExecSkill
from backend.skills.registry import SkillRegistry


class ExecShellTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.previous_config = get_config()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = RuntimeStore(self.root / "runtime.sqlite3")
        self.runtime = RuntimeService(
            self.store, self.root / "workspace", RuntimeConfig(), self.root
        )
        self.runtime.authority = "unrestricted_host"
        self.config = TamAGIConfig()
        self.config.workspace.path = str(self.runtime.workspace)
        set_config(self.config)
        set_runtime(self.runtime)
        self.registry = SkillRegistry()
        self.registry.register(ExecSkill())

    def tearDown(self):
        set_runtime(None)
        set_config(self.previous_config)
        self.store.close()
        self.temporary.cleanup()

    async def test_shell_pipeline_redirect_and_followup_command(self):
        result = await self.registry.execute(
            "exec",
            command="printf '%s\\n' alpha beta | head -n 1 > note.txt && cat note.txt",
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output, "alpha")
        self.assertEqual((self.runtime.workspace / "note.txt").read_text(), "alpha\n")
        receipt = self.store.receipts()[0]
        self.assertEqual(receipt["status"], "succeeded")
        self.assertEqual(receipt["payload"]["result"]["data"]["execution_mode"], "bash")

    async def test_cd_variables_and_quoted_arguments(self):
        result = await self.registry.execute(
            "exec",
            command="mkdir 'research notes' && cd 'research notes' && value='a b' && printf '%s' \"$value\" > note.txt && pwd",
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output, str(self.runtime.workspace / "research notes"))
        self.assertEqual(
            (self.runtime.workspace / "research notes/note.txt").read_text(), "a b"
        )

    async def test_pipeline_failure_is_not_verified_success(self):
        result = await self.registry.execute("exec", command="exit 7 | cat")
        self.assertFalse(result.success)
        self.assertEqual(result.data["return_code"], 7)
        self.assertIn("status 7", result.error)
        self.assertEqual(self.store.receipts()[0]["status"], "failed")

    async def test_shell_syntax_error_is_a_failed_receipt(self):
        result = await self.registry.execute("exec", command="printf 'unclosed")
        self.assertFalse(result.success)
        self.assertEqual(result.data["return_code"], 2)
        self.assertEqual(self.store.receipts()[0]["status"], "failed")

    async def test_guarded_arguments_do_not_gain_shell_authority(self):
        self.runtime.authority = "guarded"
        self.config.guardrails.exec_trust.safe.append("printf")
        result = await self.registry.execute(
            "exec", command="printf '%s' '$(touch injected)'"
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output, "$(touch injected)")
        self.assertEqual(result.data["execution_mode"], "argv")
        self.assertFalse((self.runtime.workspace / "injected").exists())
        self.config.guardrails.exec_trust.block.append("bash")
        denied = await self.registry.execute("exec", command="bash -c 'touch injected'")
        self.assertFalse(denied.success)
        self.assertIn("blocked", denied.error)
        self.assertFalse((self.runtime.workspace / "injected").exists())

    def child_command(self):
        program = "import os,time;from pathlib import Path;Path('child.pid').write_text(str(os.getpid()));time.sleep(30);Path('late-effect').touch()"
        return f"{shlex.quote(sys.executable)} -c {shlex.quote(program)} & wait"

    async def child_pid(self):
        path = self.runtime.workspace / "child.pid"
        for _ in range(200):
            if path.exists() and path.read_text():
                return int(path.read_text())
            await asyncio.sleep(0.01)
        self.fail("Command did not start its child")

    async def assert_child_stopped(self, pid):
        for _ in range(100):
            try:
                state = (
                    Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0]
                )
            except FileNotFoundError:
                return
            if state == "Z":
                return
            await asyncio.sleep(0.01)
        os.kill(pid, 9)
        self.fail("Command child survived interruption")

    async def test_timeout_stops_descendants_and_records_partial_effect_warning(self):
        self.config.guardrails.exec_timeout = 1
        task = asyncio.create_task(
            self.registry.execute("exec", command=self.child_command())
        )
        pid = await self.child_pid()
        result = await asyncio.wait_for(task, 4)
        self.assertFalse(result.success)
        self.assertIn("partial effects", result.error)
        await self.assert_child_stopped(pid)
        self.assertFalse((self.runtime.workspace / "late-effect").exists())

    async def test_interruption_stops_descendants_and_keeps_unknown_outcome(self):
        async def activate():
            async with self.runtime.execution("autonomy"):
                return await self.registry.execute("exec", command=self.child_command())

        task = asyncio.create_task(activate())
        pid = await self.child_pid()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 4)
        await self.assert_child_stopped(pid)
        self.assertEqual(self.store.receipts()[0]["status"], "outcome_unknown")
        self.assertFalse((self.runtime.workspace / "late-effect").exists())
