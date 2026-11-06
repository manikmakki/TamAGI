"""Explicit registry for new disposable homes; live enrollment is deliberately absent."""

from __future__ import annotations

import fcntl
import json
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import quote

from .contracts import AccessDenied, Conflict, NotFound, PROTOCOL_VERSION, WorkerBinding
from .store import HouseholdStore, uid

MARKER = "household-sandbox.json"


def reject_symlinks(root: Path):
    # Fixture roots are small. Check all supported projections before bootstrap
    # writes any of them, including identity files and SQLite sidecars.
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise AccessDenied("Disposable resident state cannot contain symlinks")


def canonical_sandbox(root: str | Path) -> tuple[Path, dict]:
    supplied = Path(root).absolute()
    resolved = supplied.resolve()
    if supplied != resolved or not resolved.is_relative_to(Path("/tmp").resolve()):
        raise AccessDenied("Experimental household roots must be canonical disposable paths under /tmp")
    marker_path = resolved / MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        raise AccessDenied("Missing disposable household marker")
    marker = json.loads(marker_path.read_text())
    if marker.get("format") != 1 or marker.get("disposable") is not True or not marker.get("household_id"):
        raise AccessDenied("Invalid disposable household marker")
    if (resolved / "residents").is_symlink():
        raise AccessDenied("Resident root directory cannot be a symlink")
    return resolved, marker


def validate_binding(binding: WorkerBinding) -> dict:
    """Inspect identity read-only before locks, hydration, or recover() can mutate it."""
    if binding.protocol != PROTOCOL_VERSION:
        raise Conflict("Unsupported worker protocol")
    root = Path(binding.root)
    if not root.is_absolute() or root != root.resolve():
        raise AccessDenied("Resident root must be canonical")
    household_root, marker = canonical_sandbox(root.parent.parent)
    if root.parent != household_root / "residents" or marker["household_id"] != binding.household_id:
        raise AccessDenied("Resident does not belong to this disposable household")
    if marker["release"] != binding.release:
        raise Conflict("Worker and household release differ")
    for path in (root, root / "runtime.sqlite3", root / "instance.json", root / "workspace"):
        if not path.exists() or path.is_symlink():
            raise AccessDenied("Resident fixture is missing or uses a symlink")
    reject_symlinks(root)
    manifest = json.loads((root / "instance.json").read_text())
    expected = {**binding.to_dict(), "disposable": True}
    if manifest != expected:
        raise Conflict("Resident manifest does not match the registered worker binding")
    database = sqlite3.connect(f"file:{quote(str(root / 'runtime.sqlite3'), safe='/')}?mode=ro", uri=True)
    try:
        database.execute("PRAGMA query_only=ON")
        metadata = {key: json.loads(value) for key, value in database.execute(
            "SELECT key,value FROM meta WHERE key IN ('resident_id','installation_id','household_sandbox_id')"
        )}
    finally:
        database.close()
    if metadata != {"resident_id": binding.resident_id, "installation_id": binding.installation_id,
                    "household_sandbox_id": binding.household_id}:
        raise Conflict("Resident database identity does not match its descriptor")
    return manifest


class SandboxRegistry:
    """Own one disposable coordinator, with an immutable selected application release."""

    @classmethod
    def create(cls, release: str, *, parent: str | Path = "/tmp", operator_name="Operator"):
        if not isinstance(release, str) or not release:
            raise ValueError("An explicit source release is required")
        supplied = Path(parent).absolute()
        if supplied != supplied.resolve() or not supplied.is_relative_to(Path("/tmp").resolve()):
            raise AccessDenied("Disposable households may only be created under /tmp")
        root = Path(tempfile.mkdtemp(prefix="tamagi-household-fixture-", dir=supplied))
        (root / "residents").mkdir()
        (root / MARKER).write_text(json.dumps({"format": 1, "disposable": True,
            "household_id": uid("household"), "release": release, "protocol": PROTOCOL_VERSION,
            "operator_id": "operator", "operator_name": operator_name}, indent=2) + "\n")
        return cls(root, release=release)

    def __init__(self, root: str | Path, *, release: str):
        self.root, self.marker = canonical_sandbox(root)
        if release != self.marker["release"] or self.marker["protocol"] != PROTOCOL_VERSION:
            raise Conflict("Household release or protocol mismatch")
        for path in self.root.glob("household.sqlite3*"):
            if path.is_symlink():
                raise AccessDenied("Household database and sidecars cannot be symlinks")
        self.household_id = self.marker["household_id"]
        self.release = release
        lock_path = self.root / "coordinator.lock"
        if lock_path.is_symlink():
            raise AccessDenied("Coordinator lock cannot be a symlink")
        self._lock = lock_path.open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.store = HouseholdStore(self.root / "household.sqlite3")
            with self.store.transaction():
                old = self.store.meta("household_id")
                if old is not None and old != self.household_id:
                    raise Conflict("Household identity mismatch")
                self.store.set_meta("household_id", self.household_id)
                self.store.set_meta("release", release)
        except BaseException:
            if hasattr(self, "store"):
                self.store.close()
            self._lock.close()
            raise

    def provision(self, name: str, *, seed: str = "A disposable test resident.") -> dict:
        """Seed an independent resident, never copy or infer an existing installation."""
        if not isinstance(name, str) or not name.strip() or not isinstance(seed, str):
            raise ValueError("A fixture needs a name and text seed")
        from backend.core.runtime.store import RuntimeStore

        root = self.root / "residents" / uid("fixture")
        root.mkdir()
        (root / "workspace").mkdir()
        (root / "data").mkdir()
        private = RuntimeStore(root / "runtime.sqlite3")
        try:
            resident_id = private.resident_id
            installation_id = private.meta("installation_id")
            private.set_meta("household_sandbox_id", self.household_id)
            private.put("actor", resident_id, resident_id, {
                "name": name.strip(), "seed": seed, "primary": True, "status": "active"
            })
            private.put("identity", "fixture-identity", resident_id, {"name": "IDENTITY.md", "text": seed})
            private.put("learning", "fixture-memory", resident_id, {"text": seed, "kind": "interpretation"})
            (root / "workspace" / "IDENTITY.md").write_text(seed)
            (root / "workspace" / "private-seed.txt").write_text(seed)
            binding = WorkerBinding(self.household_id, resident_id, installation_id, str(root), self.release)
            (root / "instance.json").write_text(json.dumps({**binding.to_dict(), "disposable": True}, indent=2) + "\n")
        finally:
            private.close()
        descriptor = {**binding.to_dict(), "name": name.strip(), "role": "resident", "availability": "offline"}
        with self.store.transaction():
            self.store.put("resident", resident_id, descriptor, expected=0)
            if self.store.meta("default_resident_id") is None:
                self.store.set_meta("default_resident_id", resident_id)
            self.store.event("resident_fixture_provisioned", resident_id, {"resident_id": resident_id})
        return descriptor

    def resolve(self, resident_id: str | None = None) -> dict:
        selected = resident_id if resident_id is not None else self.store.meta("default_resident_id")
        record = self.store.get("resident", selected) if selected else None
        if record is None:
            raise NotFound("Unknown resident; selection cannot fall back to another home")
        return record["data"]

    def binding(self, resident_id: str | None = None) -> WorkerBinding:
        descriptor = self.resolve(resident_id)
        binding = WorkerBinding.parse({key: descriptor[key] for key in WorkerBinding.__dataclass_fields__})
        validate_binding(binding)
        return binding

    def participants(self) -> list[dict]:
        return [{"id": self.marker["operator_id"], "name": self.marker["operator_name"], "kind": "human"}] + [
            {"id": record["id"], "name": record["data"]["name"], "kind": "resident"}
            for record in self.store.list("resident")
        ]

    def close(self):
        self.store.close()
        self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
