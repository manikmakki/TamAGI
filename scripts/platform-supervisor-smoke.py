"""Exercise the real supervisor and application against disposable lived state."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import yaml
import backend

from backend.config import TamAGIConfig
from backend.core.runtime import RuntimeService, RuntimeStore
from backend.core.runtime.deployment import DeploymentManager, atomic_json, switch_link

SOURCE = Path(backend.__file__).resolve().parents[1]


def eventually(predicate, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise AssertionError("Supervisor did not converge before the deadline")


with tempfile.TemporaryDirectory(prefix="tamagi-supervisor-check-") as temporary:
    root = Path(temporary)
    platform = root / "platform"
    platform.mkdir()
    instance = root / "instance"
    store = RuntimeStore(instance / "runtime.sqlite3")
    config = TamAGIConfig()
    config.runtime.enabled = True
    config.runtime.instance_dir = str(instance)
    config.runtime.canonical_repo = str(root)
    config.runtime.authority = "guarded"
    config.autonomy.enabled = False
    config.llm.base_url = "http://127.0.0.1:1/v1"
    config.mcp.servers = []
    runtime = RuntimeService(store, instance / "workspace", config.runtime, root)
    lesson = runtime.remember("Preserve this experience across code recovery", "lesson")
    store.close()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    config.server.host = "127.0.0.1"
    config.server.port = port
    config_path = instance / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump()))

    def release(name):
        target = platform / "releases" / name
        for filename in DeploymentManager.inventory(SOURCE):
            destination = target / "source" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / filename, destination)
        (target / "venv").symlink_to(Path(sys.prefix), target_is_directory=True)
        return target

    previous = release("known-good")
    candidate = release("candidate-good")
    broken = platform / "releases" / "candidate-broken"
    (broken / "venv/bin").mkdir(parents=True)
    program = broken / "venv/bin/python"
    program.write_text("#!/bin/sh\nexit 72\n")
    program.chmod(0o700)
    switch_link(platform / "current", previous)
    atomic_json(
        platform / "active.json",
        {"instance": str(instance), "config": str(config_path)},
    )
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("TAMAGI_CONFIG", None)

    def healthy(expected):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/health/ready", timeout=1
        ) as response:
            value = json.load(response)
        return value.get("data_ready") and value.get("release") == expected

    def phase(expected):
        return (
            json.loads((platform / "deployment.json").read_text())["phase"] == expected
        )

    with (root / "supervisor.log").open("w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(SOURCE / "scripts/tamagi-supervisor.py"),
                "--root",
                str(platform),
                "--port",
                str(port),
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            eventually(lambda: healthy(previous.name))
            for target, expected in [(broken, "rolled_back"), (candidate, "committed")]:
                atomic_json(
                    platform / "deployment.json",
                    {
                        "phase": "quiescing",
                        "candidate_id": target.name,
                        "candidate": str(target),
                        "previous": str(previous),
                        "instance": str(instance),
                        "created_at": time.time(),
                    },
                )
                eventually(lambda: phase(expected))
                selected = previous if expected == "rolled_back" else candidate
                eventually(lambda: healthy(selected.name))
                check = RuntimeStore(instance / "runtime.sqlite3")
                try:
                    assert check.get("learning", lesson["id"]), (
                        "Code recovery erased lived state"
                    )
                finally:
                    check.close()
            # No provider is reachable; local data readiness still commits the release.
            assert process.poll() is None
            print(
                json.dumps(
                    {
                        "broken_release": "rolled_back",
                        "healthy_release": "committed",
                        "lived_state": "preserved",
                        "model_outage": "readiness_independent",
                    }
                )
            )
        except BaseException:
            log.flush()
            log.seek(0)
            print(log.read()[-5000:], file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
