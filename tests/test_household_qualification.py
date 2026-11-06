"""Offline qualification must fail closed and propagate to smoke-test children."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import backend


SOURCE = Path(backend.__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("household_qualification", SOURCE / "scripts/qualify_household.py")
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tamagi-guard-test-")
        self.root = Path(self.temporary.name)
        self.guard = self.root / "guard"
        qualification.prepare_guard(self.guard)
        self.cwd = self.root / "cwd"
        self.cwd.mkdir()
        self.environment = qualification._guard_environment(self.guard, SOURCE, self.cwd)

    def tearDown(self):
        self.temporary.cleanup()

    def python(self, program):
        return subprocess.run(
            [sys.executable, "-c", program], cwd=self.cwd, env=self.environment,
            capture_output=True, text=True, timeout=15,
        )

    def test_provider_dns_and_udp_denials_are_retained_even_if_caught(self):
        result = self.python("""
import socket
for operation in (
    lambda: socket.socket().connect(('127.0.0.1', 11434)),
    lambda: socket.getaddrinfo('outside.example', 443),
    lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'x', ('127.0.0.1', 53)),
):
    try: operation()
    except PermissionError: pass
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        denied = [json.loads(line) for line in (self.guard / "denied.jsonl").read_text().splitlines()]
        self.assertEqual([item["operation"] for item in denied], ["connect", "getaddrinfo", "sendto"])

    def test_loopback_connection_requires_a_port_bound_by_this_run(self):
        result = self.python("""
import socket
import threading
server = socket.socket()
server.bind(('127.0.0.1', 0))
server.listen()
def serve():
    connection, _ = server.accept()
    with connection: connection.sendall(b'local')
thread = threading.Thread(target=serve)
thread.start()
with socket.create_connection(server.getsockname()) as client:
    assert client.recv(5) == b'local'
thread.join()
server.close()
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.guard / "denied.jsonl").read_text(), "")

    def test_child_replacing_pythonpath_keeps_guard(self):
        result = self.python("""
import os, subprocess, sys
environment = {'PATH': os.environ.get('PATH', ''), 'PYTHONPATH': '/nonexistent'}
program = "import socket; socket.socket().connect(('127.0.0.1', 11434))"
child = subprocess.run([sys.executable, '-c', program], env=environment, capture_output=True, text=True)
assert child.returncode != 0
assert 'Offline qualification blocked network access' in child.stderr, child.stderr
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len((self.guard / "activations.jsonl").read_text().splitlines()), 2)
        self.assertEqual(len((self.guard / "denied.jsonl").read_text().splitlines()), 1)

    def test_child_bound_port_supports_disposable_supervisor_health_checks(self):
        result = self.python('''
import socket, subprocess, sys
program = """import socket
server = socket.socket()
server.bind(('127.0.0.1', 0))
server.listen()
print(server.getsockname()[1], flush=True)
connection, _ = server.accept()
with connection: connection.sendall(b'healthy')
server.close()
"""
child = subprocess.Popen([sys.executable, '-c', program], stdout=subprocess.PIPE, text=True)
try:
    port = int(child.stdout.readline())
    with socket.create_connection(('127.0.0.1', port)) as client:
        assert client.recv(7) == b'healthy'
    assert child.wait(timeout=5) == 0
finally:
    if child.poll() is None:
        child.kill()
        child.wait()
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.guard / "denied.jsonl").read_text(), "")

    def test_live_paths_rejected_and_inherited_configuration_scrubbed(self):
        with self.assertRaises(ValueError):
            qualification._outside_live_tree(Path("/opt/TamAGI/data"))
        with patch.dict(os.environ, {"TAMAGI_LLM_BASE_URL": "http://secret-provider", "TAMAGI_CONFIG": "/live/config.yaml", "HTTP_PROXY": "http://secret-proxy", "PYTHONHOME": "/other/python", "TMPDIR": "/live/temp"}):
            environment = qualification._guard_environment(self.guard, SOURCE, self.cwd)
        self.assertNotIn("TAMAGI_LLM_BASE_URL", environment)
        self.assertNotIn("HTTP_PROXY", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("TMPDIR", environment)
        self.assertEqual(environment["TAMAGI_CONFIG"], str(self.cwd / "no-live-config.yaml"))
        self.assertEqual(environment["TAMAGI_AUTONOMY_ENABLED"], "false")

    def test_successful_test_process_with_network_denial_fails_qualification(self):
        source = self.root / "source"
        tests = source / "tests"
        tests.mkdir(parents=True)
        (tests / "test_fixture.py").write_text('''import socket, unittest
class Fixture(unittest.TestCase):
    def test_caught_network_attempt(self):
        try: socket.socket().connect(('127.0.0.1', 11434))
        except PermissionError: pass
''')
        with patch.object(qualification, "_manifest", return_value={"revision": "fixture", "diff_sha256": "fixture", "dirty": False}):
            report = qualification.run_qualification(source, self.root / "evidence")
        self.assertEqual(report["returncode"], 0)
        self.assertFalse(report["passed"])
        self.assertEqual(report["denied_network_operations"], 1)
        self.assertTrue((self.root / "evidence/report.json").is_file())
