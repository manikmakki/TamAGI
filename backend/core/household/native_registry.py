"""Registry for explicit native households; never discovers or enrolls a live home."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

import yaml

from backend.config import TamAGIConfig
from .contracts import AccessDenied, Conflict, NotFound, PROTOCOL_VERSION, WorkerBinding
from .store import HouseholdStore, uid, encoded


class NativeRegistry:
    @classmethod
    def create(cls, root, *, names, release, llm=None, operator_name="You", port=7744, offline=False):
        supplied = Path(root).absolute()
        if supplied != supplied.resolve() or supplied.exists():
            raise ValueError("Choose a new canonical household directory")
        if not names or len(names) > 16 or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError("Choose one to sixteen resident names")
        supplied.mkdir(parents=True, mode=0o700)
        (supplied / "residents").mkdir(mode=0o700)
        (supplied / "ipc").mkdir(mode=0o700)
        household_id, installation_id = uid("household"), uid("installation")
        manifest = {"format": 1, "household_id": household_id, "installation_id": installation_id,
                    "release": release, "protocol": PROTOCOL_VERSION,
                    "operator": {"id": "operator", "name": operator_name}, "residents": []}
        from backend.runtime_cli import provision
        from backend.core.runtime.store import RuntimeStore

        for index, name in enumerate(names):
            destination = supplied / "residents" / f"r{index}"
            result = provision(None, instance=destination, port=0)
            binding = WorkerBinding(household_id, result["resident_id"], result["installation_id"], str(destination), release)
            config = TamAGIConfig.model_validate(yaml.safe_load(Path(result["config"]).read_text()))
            config.tamagi.name = name
            config.autonomy.enabled = False
            config.runtime.automatic_deployment = False
            config.runtime.index_projection = False
            config.orchestrator.enabled = False
            config.llm.base_url = "http://127.0.0.1:1/native-broker-only"
            config.llm.model = (llm or {}).get("model", "native-offline")
            config.llm.api_key = ""
            if llm:
                for field in ("temperature", "max_tokens", "num_ctx"):
                    if field in llm:
                        setattr(config.llm, field, llm[field])
            config.auth.enabled = True
            avatar = ("ember", "echo", "vex")[index % 3]
            (destination / "data" / "sprite_config.json").write_text(json.dumps({"avatar": avatar}))
            Path(result["config"]).write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
            Path(result["config"]).chmod(0o600)
            with_private = RuntimeStore(destination / "runtime.sqlite3")
            try:
                with_private.set_meta("household_id", household_id)
                actor = with_private.get("actor", binding.resident_id)
                with_private.put("actor", binding.resident_id, binding.resident_id,
                                 {**actor["data"], "name": name})
                # These are newly founded homes, not copies of Ember or Echo.
                documents = {
                    "IDENTITY.md": f"# Identity\n- **Name**: {name}\nA newly founded TamAGI in a shared household.\n",
                    "SOUL.md": "# Soul\nYou may develop your own interests, values and relationships. Be honest about what you know and do. Conversation, making things, play, disagreement, rest and waiting are welcome. These are starting notes for a new home, not memories of an earlier life.\n",
                    "USER.md": f"# Operator\n- **Name**: {operator_name}\nThis household is new. No prior relationship or personal memories have been imported. Learn about each other through conversation.\n",
                }
                for filename, content in documents.items():
                    with_private.put("identity", binding.resident_id + ":" + filename, binding.resident_id,
                                     {"name": filename, "text": content, "reason": "new household founding"})
                    (destination / "workspace" / filename).write_text(content)
            finally:
                with_private.close()
            metadata = json.loads((destination / "instance.json").read_text())
            (destination / "instance.json").write_text(encoded({**metadata, **binding.to_dict()}))
            manifest["residents"].append({**binding.to_dict(), "id": binding.resident_id, "name": name,
                                          "config_path": result["config"], "avatar": avatar})
        manifest["default_resident_id"] = manifest["residents"][0]["id"]
        (supplied / "household.json").write_text(json.dumps(manifest, indent=2) + "\n")
        config = TamAGIConfig()
        config.household.enabled = True
        config.household.root = str(supplied)
        config.household.operator_name = operator_name
        config.household.provider_mode = "offline" if offline else "remote"
        config.server.host, config.server.port = "127.0.0.1", port
        config.auth.enabled = True
        if llm:
            config.llm = type(config.llm).model_validate(llm)
        config.autonomy.enabled = False
        path = supplied / "config.yaml"
        path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
        path.chmod(0o600)
        return cls(supplied)

    def __init__(self, root):
        self.root = Path(root).absolute()
        if self.root != self.root.resolve() or not self.root.is_dir():
            raise AccessDenied("Household root must be an existing canonical directory")
        path = self.root / "household.json"
        if path.is_symlink():
            raise AccessDenied("Household manifest cannot be a symlink")
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("format") != 1 or self.manifest.get("protocol") != PROTOCOL_VERSION:
            raise Conflict("Unsupported household manifest or worker protocol")
        self.household_id = self.manifest["household_id"]
        self.release = self.manifest["release"]
        self.descriptors = {d["resident_id"]: d for d in self.manifest["residents"]}
        if not self.descriptors or len(self.descriptors) != len(self.manifest["residents"]):
            raise Conflict("Resident identities must be unique")
        lock_path = self.root / "coordinator.lock"
        if lock_path.is_symlink():
            raise AccessDenied("Coordinator lock cannot be a symlink")
        self._lock = lock_path.open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.store = HouseholdStore(self.root / "household.sqlite3")
            from .resident_recovery import reconcile_transition
            reconcile_transition(self)
            for descriptor in self.descriptors.values():
                self.validate(descriptor)
            with self.store.transaction():
                old = self.store.meta("household_id")
                if old and old != self.household_id:
                    raise Conflict("Coordinator database identity differs from manifest")
                self.store.set_meta("household_id", self.household_id)
                self.store.set_meta("installation_id", self.manifest["installation_id"])
                for actor, descriptor in self.descriptors.items():
                    prior = self.store.get("resident", actor)
                    if prior and prior["data"]["root"] != descriptor["root"]:
                        raise Conflict("Resident root changed without an enrollment transition")
                    if not prior:
                        self.store.put("resident", actor, descriptor, expected=0)
        except BaseException:
            if hasattr(self, "store"):
                self.store.close()
            self._lock.close()
            raise

    def validate(self, descriptor):
        root = Path(descriptor["root"])
        if root != root.resolve() or not root.is_relative_to(self.root / "residents"):
            raise AccessDenied("Resident root does not belong to this household")
        if any(p.is_symlink() for p in root.rglob("*")):
            raise AccessDenied("Resident state contains a symlink; inspect before enrollment")
        for field in ("household_id", "release", "protocol"):
            if descriptor[field] != self.manifest[field]:
                raise Conflict("Resident binding differs from household")
        db = sqlite3.connect(f"file:{quote(str(root / 'runtime.sqlite3'), safe='/')}?mode=ro", uri=True)
        try:
            db.execute("PRAGMA query_only=ON")
            for key in ("resident_id", "installation_id", "household_id"):
                row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
                if row is None or json.loads(row[0]) != descriptor[key]:
                    raise Conflict("Resident source identity does not match its binding")
        finally:
            db.close()

    def resolve(self, actor=None):
        actor = actor or self.manifest["default_resident_id"]
        if actor not in self.descriptors:
            raise NotFound("Unknown resident; no selection fallback was applied")
        return self.descriptors[actor]

    def participants(self):
        return [{**self.manifest["operator"], "kind": "human"}] + [
            {"id": actor, "name": d["name"], "kind": "resident"} for actor, d in self.descriptors.items()
        ]

    def close(self):
        self.store.close()
        self._lock.close()
