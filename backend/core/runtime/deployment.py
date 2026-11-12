"""Cooperative autonomous releases. Root can change this machinery too."""

from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from .store import uid, encoded


class DeploymentManager:
    def __init__(self, runtime, root=None):
        self.runtime = runtime
        self.root = Path(root or runtime.source_root / ".platform").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "releases").mkdir(exist_ok=True)
        (self.root / "candidates").mkdir(exist_ok=True)

    def prepare(self, intention):
        self.synchronize()
        if not intention.strip():
            raise ValueError("A core change needs an intention")
        candidate_id = uid("candidate")
        directory = self.root / "candidates" / candidate_id
        command = [
            "git",
            "-C",
            str(self.runtime.source_root),
            "worktree",
            "add",
            "--detach",
            str(directory),
            "HEAD",
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        baseline = getattr(
            self.runtime, "release_source_root", self.runtime.source_root
        )
        baseline_inventory = self.inventory(baseline)
        for name in self.inventory(directory):
            if name not in baseline_inventory:
                (directory / name).unlink()
        for name in baseline_inventory:
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline / name, target)
        record = self.runtime.store.put(
            "deployment",
            candidate_id,
            self.runtime.scope(),
            {
                "phase": "prepared",
                "intention": intention,
                "worktree": str(directory),
                "baseline": str(self.selected() or ""),
                "schema": self.runtime.store.SCHEMA,
                "authority": self.runtime.authority,
            },
            category="platform",
        )
        return record

    @staticmethod
    def inventory(source):
        from .backup import PortableBackup

        files = {}
        for top in (
            "backend",
            "frontend",
            "scripts",
            "tests",
            "docs",
            "README.md",
            "Dockerfile",
            "config.example.yaml",
            "pyproject.toml",
            "requirements.txt",
            "requirements.lock",
            "LICENSE.md",
            "tamagi-platform.service",
        ):
            base = source / top
            paths = (
                [base] if base.is_file() else (base.rglob("*") if base.is_dir() else [])
            )
            for path in paths:
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and PortableBackup._eligible(path, source)
                ):
                    files[str(path.relative_to(source))] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
        return files

    def synchronize(self):
        path = self.root / "deployment.json"
        if not path.exists():
            return
        state = json.loads(path.read_text())
        record = self.runtime.store.get("deployment", state.get("candidate_id", ""))
        if (
            record
            and state.get("phase") in {"committed", "rolled_back", "recovery_required"}
            and record["data"].get("phase") != state["phase"]
        ):
            self.runtime.store.put(
                "deployment",
                record["id"],
                record["actor"],
                {**record["data"], "phase": state["phase"]},
                category="platform",
            )
            self.runtime.wake("deployment_completed")

    def selected(self):
        current = self.root / "current"
        return current.resolve() if current.is_symlink() else None

    def build(self, candidate_id):
        record = self.runtime.store.get("deployment", candidate_id)
        if not record or record["data"]["phase"] not in {
            "prepared",
            "validation_failed",
        }:
            raise ValueError("Choose a prepared candidate")
        source = Path(record["data"]["worktree"]).resolve()
        if not source.is_relative_to(self.root / "candidates") or not source.is_dir():
            raise ValueError("Candidate source is outside the release workspace")
        inventory = self.inventory(source)
        source_hash = hashlib.sha256(encoded(inventory).encode()).hexdigest()
        release = self.root / "releases" / source_hash
        if release.exists():
            raise ValueError(
                "This exact release already exists; inspect its retained validation"
            )
        release.mkdir()
        data = {
            **record["data"],
            "phase": "building",
            "release": str(release),
            "source_hash": source_hash,
            "inventory": inventory,
        }
        self.runtime.store.put(
            "deployment", candidate_id, self.runtime.actor_id, data, category="platform"
        )
        try:
            target = release / "source"
            target.mkdir()
            for name in inventory:
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, destination)
            if self.inventory(target) != inventory:
                raise ValueError("Source changed while the release was being captured")
            lock = target / "requirements.lock"
            if not lock.is_file():
                raise ValueError("A reproducible release requires requirements.lock")
            subprocess.run(
                [sys.executable, "-m", "venv", str(release / "venv")],
                check=True,
                capture_output=True,
                timeout=120,
            )
            python = release / "venv/bin/python"
            subprocess.run(
                [str(python), "-m", "pip", "install", "-r", str(lock)],
                check=True,
                capture_output=True,
                timeout=600,
            )
            # Freeze baseline tests separately; candidate test edits cannot remove the existing gates.
            gates = release / "baseline-tests"
            shutil.copytree(
                getattr(self.runtime, "release_source_root", self.runtime.source_root)
                / "tests",
                gates,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            probes = release / "baseline-probes"
            probes.mkdir()
            baseline_source = getattr(
                self.runtime, "release_source_root", self.runtime.source_root
            )
            for probe in ("platform-smoke.py", "platform-supervisor-smoke.py"):
                shutil.copy2(baseline_source / "scripts" / probe, probes / probe)
            env = {
                **os.environ,
                "TAMAGI_GATE_SCRIPTS": str(probes),
                "PYTHONPATH": str(target),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TAMAGI_AUTONOMY_ENABLED": "false",
            }
            with tempfile.TemporaryDirectory(prefix="release-check-") as cwd:
                env.pop("TAMAGI_CONFIG", None)
                completed = subprocess.run(
                    [str(python), "-m", "unittest", "discover", "-s", str(gates), "-q"],
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    timeout=600,
                )
            if completed.returncode:
                raise ValueError(
                    "Frozen baseline checks failed; inspect the candidate in its worktree"
                )
            self.boot_check(release)
            if self.inventory(target) != inventory:
                raise ValueError("Validated source changed during checks")
            manifest = {
                "candidate_id": candidate_id,
                "source_hash": source_hash,
                "schema": self.runtime.store.SCHEMA,
                "python": sys.version,
                "inventory": inventory,
                "tests": "frozen baseline suite passed",
                "evidence_class": "scripted",
            }
            (release / "manifest.json").write_text(encoded(manifest))
            data = {**data, "phase": "validated", "manifest": manifest}
            self.runtime.store.put(
                "deployment",
                candidate_id,
                self.runtime.actor_id,
                data,
                category="platform",
            )
            return data
        except BaseException as exc:
            self.runtime.store.put(
                "deployment",
                candidate_id,
                self.runtime.actor_id,
                {**data, "phase": "validation_failed", "error": type(exc).__name__},
                category="platform",
            )
            raise

    def boot_check(self, release):
        import yaml
        from backend.config import get_config
        from .backup import sanitized

        with tempfile.TemporaryDirectory(prefix="candidate-home-") as directory:
            home = Path(directory)
            self.runtime.store.checkpoint(home / "runtime.sqlite3")
            shutil.copytree(
                self.runtime.workspace,
                home / "workspace",
                ignore=lambda directory, names: [
                    name
                    for name in names
                    if (Path(directory) / name).is_symlink()
                    or name
                    in {
                        ".venv",
                        "node_modules",
                        "__pycache__",
                        ".env",
                        ".secrets.key",
                        "secrets.enc",
                    }
                ],
                symlinks=False,
            )
            config = sanitized(get_config().model_dump())
            config["runtime"] = {
                **config["runtime"],
                "enabled": True,
                "instance_dir": str(home),
                "canonical_repo": str(release / "source"),
                "authority": "guarded",
                "automatic_deployment": False,
                "index_projection": False,
            }
            config["autonomy"]["enabled"] = False
            config["mcp"]["servers"] = []
            config_path = home / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            env = {
                **os.environ,
                "PYTHONPATH": str(release / "source"),
                "TAMAGI_CONFIG": str(config_path),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            code = "from fastapi.testclient import TestClient; from backend.main import app; from backend.core.runtime import get_runtime; "
            code += "exec(\"with TestClient(app) as client:\\n response=client.get('/api/health/ready')\\n assert response.status_code==200 and response.json()['data_ready']\")"
            result = subprocess.run(
                [str(release / "venv/bin/python"), "-c", code],
                cwd=home,
                env=env,
                capture_output=True,
                timeout=90,
            )
            if result.returncode:
                raise ValueError(
                    "Candidate could not boot against a copy of current lived state"
                )
        return {"data_ready": True, "model_required": False, "live_integrations": False}

    def stage_activation(self, candidate_id):
        if not getattr(self.runtime.config, "automatic_deployment", False):
            raise ValueError(
                "Automatic deployment remains disabled until recovery qualification"
            )
        if not self.runtime.verify_host().get("host_verified"):
            raise ValueError(
                "Automatic core deployment requires the dedicated host-root installation"
            )
        qualification = self.runtime.store.meta("deployment_qualification")
        if not qualification or qualification.get("passed") is not True:
            raise ValueError("Deployment recovery qualification has not passed")
        record = self.runtime.store.get("deployment", candidate_id)
        if not record or record["data"]["phase"] != "validated":
            raise ValueError("Candidate is not validated")
        data = record["data"]
        release = Path(data["release"])
        if self.inventory(release / "source") != data["inventory"]:
            raise ValueError("Validated release no longer matches its manifest")
        # All ordinary releases must remain compatible with the current schema.
        if data["schema"] != self.runtime.store.SCHEMA:
            raise ValueError(
                "Schema-changing deployment needs a separately qualified restore path"
            )
        snapshot = self.root / f"{candidate_id}.sqlite3"
        self.runtime.store.checkpoint(snapshot)
        handoff = {
            "candidate_id": candidate_id,
            "previous": str(self.selected() or ""),
            "candidate": str(release),
            "phase": "quiescing",
            "created_at": self.runtime.store.clock(),
            "checkpoint": str(snapshot),
            "instance": str(self.runtime.store.path.parent),
            "unfinished": [
                r["id"]
                for r in self.runtime.store.list("pursuit")
                if r["data"]["status"] in {"active", "blocked", "paused"}
            ],
        }
        atomic_json(self.root / "deployment.json", handoff)
        self.runtime.store.put(
            "deployment",
            candidate_id,
            self.runtime.actor_id,
            {**data, "phase": "quiescing", "handoff": handoff},
            category="platform",
        )
        self.runtime.maintenance = True
        return handoff

    def reconcile(self, healthy=False):
        path = self.root / "deployment.json"
        if not path.exists():
            return {"phase": "idle"}
        state = json.loads(path.read_text())
        if state["phase"] in {"committed", "rolled_back", "recovery_required"}:
            return state
        if state["phase"] == "quiescing":
            state["phase"] = "switching"
        elif state["phase"] == "switching":
            switch_link(self.root / "current", Path(state["candidate"]))
            state["phase"] = "verifying"
        elif state["phase"] == "verifying":
            if healthy:
                state["phase"] = "committed"
            elif state.get("previous") and Path(state["previous"]).is_dir():
                switch_link(self.root / "current", Path(state["previous"]))
                state["phase"] = "rolled_back"
            else:
                state["phase"] = "recovery_required"
        atomic_json(path, state)
        record = self.runtime.store.get("deployment", state["candidate_id"])
        if record:
            self.runtime.store.put(
                "deployment",
                record["id"],
                record["actor"],
                {**record["data"], "phase": state["phase"]},
                category="platform",
            )
        return state


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as output:
        output.write(encoded(value))
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def switch_link(path, target):
    temporary = path.with_name(path.name + ".next")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target, target_is_directory=True)
    temporary.replace(path)
