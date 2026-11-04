"""Full application smoke in a disposable process and instance."""

import os
import subprocess
import sys
import unittest
from pathlib import Path
import backend


class ApplicationTests(unittest.TestCase):
    def test_operator_session_ws_origin_and_portable_backup_flow(self):
        root = Path(backend.__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(root)}
        env.pop("TAMAGI_CONFIG", None)
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(os.environ.get("TAMAGI_GATE_SCRIPTS", root / "scripts"))
                    / "platform-smoke.py"
                ),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-5000:])
        self.assertIn('"encrypted_download_upload_preview": "passed"', result.stdout)
        self.assertIn('"growth_conversation_and_inspection": "passed"', result.stdout)

    def test_real_supervisor_recovers_and_commits_without_a_model(self):
        root = Path(backend.__file__).resolve().parents[1]
        environment = {**os.environ, "PYTHONPATH": str(root)}
        environment.pop("TAMAGI_CONFIG", None)
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(os.environ.get("TAMAGI_GATE_SCRIPTS", root / "scripts"))
                    / "platform-supervisor-smoke.py"
                ),
            ],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-6000:])
        self.assertIn('"healthy_release": "committed"', result.stdout)
        self.assertIn('"lived_state": "preserved"', result.stdout)
