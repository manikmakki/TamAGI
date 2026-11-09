"""Cold, resident-first checkpoints. Shared state is evidence, never rollback input.

The coordinator lock is mandatory for every source/destination household. A
replacement is prepared in a new root and selected by a durable, replayable
registry transition; the displaced root is retained for operator inspection.
No worker, uploaded source, model, or resident tool is executed here.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace

import yaml

from backend.config import TamAGIConfig
from backend.core.runtime.backup import ArchiveError, PortableBackup
from backend.core.runtime.store import RuntimeStore
from .contracts import Conflict, PROTOCOL_VERSION
from .store import HouseholdStore, encoded, uid

SOURCE = Path(__file__).resolve().parents[3]
JOURNAL = "resident-recovery.json"


def atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".recovery-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(encoded(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_tree(path):
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise ArchiveError("Recovered roots cannot contain symbolic links")
        if entry.is_file():
            with entry.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in sorted([path, *(p for p in path.rglob("*") if p.is_dir())],
                            key=lambda p: len(p.parts), reverse=True):
        sync_directory(directory)
    sync_directory(path.parent)


def settle_restored_actor(store, actor):
    """Retire this resident's old authority as part of its durable root selection."""
    from .broker import InferenceBroker
    from .native_service import TERMINAL

    with store.transaction():
        for kind in ("reply_offer", "native_invitation_offer"):
            for record in store.list(kind):
                data = record["data"]
                if data.get("recipient") == actor and data.get("status") in {"pending", "deferred", "running"}:
                    store.put(kind, record["id"], {**data, "status": "closed",
                        "recovery_note": "Resident checkpoint restored; invite or address them again explicitly"})
        for record in store.list("conversation_invitation"):
            data = record["data"]
            if data.get("participant_id") == actor and data.get("status") == "pending":
                store.put("conversation_invitation", record["id"], {**data, "status": "expired",
                    "recovery_note": "Resident checkpoint restored; invite again explicitly"})
        for record in store.list("native_grant"):
            if record["data"].get("resident_id") == actor and record["data"].get("status") not in TERMINAL:
                store.put("native_grant", record["id"], {**record["data"], "status": "interrupted"})
        broker = None
        for record in store.list("native_execution"):
            if record["data"].get("resident_id") == actor:
                identity = record["data"]["broker_execution_id"]
                execution = store.get("broker_execution", identity)
                if execution is None:
                    raise Conflict("Recovery execution has no canonical broker binding")
                if execution["data"]["status"] == "running":
                    # Reuse canonical stop/accounting behavior without performing
                    # a broker restart or changing other residents' opportunities.
                    broker = broker or InferenceBroker(store, recover=False)
                    broker.stop_execution(identity)


def reconcile_transition(registry):
    """Finish only an explicit, durable operator-selected replacement under lock."""
    from .native_registry import NativeRegistry

    path = registry.root / JOURNAL
    if not path.exists():
        return
    if path.is_symlink():
        raise Conflict("Recovery transition cannot be a symbolic link")
    if registry.store.meta("household_id") != registry.household_id:
        raise Conflict("Recovery database belongs to another household")
    transition = json.loads(path.read_text())
    before, after = transition["before"], transition["after"]
    actor = before["resident_id"]
    current = next((r for r in registry.manifest["residents"] if r["resident_id"] == actor), None)
    if (transition.get("format") != 1 or after["resident_id"] != actor
            or current not in (before, after)):
        raise Conflict("Recovery transition does not match the selected resident")
    NativeRegistry.validate(registry, after)
    saved = registry.store.get("resident", actor)
    if not saved or saved["data"] not in (before, after):
        raise Conflict("Recovery transition conflicts with the coordinator registry")
    # Select the binding and retire its old opportunities together. No shared
    # conversation, read marker, publication, world object or clock is imported.
    with registry.store.transaction():
        if saved["data"] != after:
            registry.store.put("resident", actor, after)
            registry.store.event("resident_restored", actor, {
                "source_installation_id": before["installation_id"],
                "installation_id": after["installation_id"],
                "retained_root": before["root"],
            })
        # This shares the binding transaction and runs on every journal replay.
        # No worker can resume between selecting private time and retiring its
        # newer opportunities, even after a failed manifest write or a crash.
        settle_restored_actor(registry.store, actor)
    registry.manifest = {**registry.manifest, "residents": [
        after if r["resident_id"] == actor else r for r in registry.manifest["residents"]]}
    atomic_json(registry.root / "household.json", registry.manifest)
    registry.descriptors = {r["resident_id"]: r for r in registry.manifest["residents"]}
    path.unlink()
    sync_directory(registry.root)


@contextmanager
def stopped_household(root, *, damaged_resident=None):
    """Open no workers. Permit only the named repair target to be damaged."""
    from .native_registry import NativeRegistry

    root = Path(root).absolute()
    if root != root.resolve() or not root.is_dir():
        raise ArchiveError("Choose an existing canonical household directory")
    for name in ("coordinator.lock", "household.json", "household.sqlite3"):
        if (root / name).is_symlink():
            raise ArchiveError("Household recovery paths cannot be symbolic links")
    lock = (root / "coordinator.lock").open("a")
    registry = None
    worker_locks = []
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ArchiveError("Stop this household before backup or recovery") from error
        manifest = json.loads((root / "household.json").read_text())
        if manifest.get("format") != 1 or manifest.get("protocol") != PROTOCOL_VERSION:
            raise ArchiveError("Unsupported household manifest")
        for descriptor in manifest["residents"]:
            actor = descriptor["resident_id"]
            if not actor.startswith("resident-") or any(c not in "0123456789abcdef" for c in actor[9:]) or len(actor) != 41:
                raise ArchiveError("Invalid resident identity")
            resident_root = Path(descriptor["root"])
            if resident_root != resident_root.resolve() or not resident_root.is_relative_to(root / "residents"):
                raise ArchiveError("Resident root is outside this household")
            paths = [root / "ipc" / (actor + ".lock")]
            # Also detect workers from releases predating the IPC lifetime lock.
            if resident_root.is_dir():
                paths.append(resident_root / "instance.lock")
            for worker_path in paths:
                if worker_path != worker_path.resolve() or worker_path.is_symlink():
                    raise ArchiveError("Worker lock path cannot redirect outside this household")
                worker_lock = worker_path.open("a")
                worker_locks.append(worker_lock)
                try:
                    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ArchiveError("A resident worker is still running; finish shutdown before recovery") from error
        if not (root / "household.sqlite3").is_file():
            raise ArchiveError("Shared state is missing; resident repair cannot rebuild it")
        registry = SimpleNamespace(root=root, manifest=manifest, household_id=manifest["household_id"],
            release=manifest["release"], descriptors={r["resident_id"]: r for r in manifest["residents"]},
            store=HouseholdStore(root / "household.sqlite3"))
        if len(registry.descriptors) != len(manifest["residents"]):
            raise ArchiveError("Duplicate resident identity")
        if registry.store.meta("household_id") != registry.household_id:
            raise ArchiveError("Shared state belongs to another household")
        reconcile_transition(registry)
        for actor, descriptor in registry.descriptors.items():
            path = Path(descriptor["root"])
            if path != path.resolve() or not path.is_relative_to(root / "residents"):
                raise ArchiveError("Resident root is outside this household")
            if actor != damaged_resident:
                NativeRegistry.validate(registry, descriptor)
        yield registry
    finally:
        if registry:
            registry.store.close()
        for worker_lock in worker_locks:
            worker_lock.close()
        lock.close()


class ResidentArchive(PortableBackup):
    FORMAT = "tamagi.resident.v1"
    FORMATS = {FORMAT, PortableBackup.FORMAT}

    def __init__(self, job_root, *, runtime=None, metadata=None, context=None, max_bytes=10 * 1024**3):
        runtime = runtime or SimpleNamespace(store=SimpleNamespace(clock=time.time, SCHEMA=RuntimeStore.SCHEMA), source_root=SOURCE)
        self.metadata, self.context = metadata, context
        super().__init__(runtime, job_root, max_bytes)

    def export_sources(self):
        sources = super().export_sources()
        # The private native data directory includes legacy history, the private
        # world graph and uploaded artwork, not only sprite configuration.
        sources[2] = ("data", self.runtime.store.path.parent / "data", None)
        return [*sources, ("data/recovery-evidence", self.evidence, None)]

    def describe_manifest(self, manifest):
        return {**manifest, "resident": self.metadata,
                "recovery_context_path": "data/recovery-evidence/" + self.context_filename,
                "shared_state": "Historical evidence only; never applied over current household state"}

    def export(self, password, config=None):
        # Overlay context in staging rather than writing anything into the home.
        with tempfile.TemporaryDirectory(prefix="resident-evidence-", dir=self.root) as directory:
            self.evidence = Path(directory)
            self.context_filename = uid("context") + ".json"
            (self.evidence / self.context_filename).write_text(encoded(self.context))
            try:
                return super().export(password, config)
            finally:
                self.evidence = None

    def stage(self, encrypted, password):
        preview = super().stage(encrypted, password)
        stage = self.stages[preview["id"]]
        manifest = stage["manifest"]
        try:
            if manifest["format"] == self.FORMAT:
                descriptor = manifest.get("resident")
                if (not isinstance(descriptor, dict) or descriptor.get("resident_id") != preview["resident_id"]
                        or not isinstance(descriptor.get("name"), str) or not descriptor["name"].strip()
                        or descriptor.get("protocol") != PROTOCOL_VERSION):
                    raise ArchiveError("Resident archive metadata is inconsistent")
                context_path = manifest.get("recovery_context_path", "data/recovery-evidence/context.json")
                if (not isinstance(context_path, str) or context_path not in manifest["files"]
                        or not context_path.startswith("data/recovery-evidence/")):
                    raise ArchiveError("Recovery context is not a verified archive member")
                context = json.loads((Path(stage["directory"]) / context_path).read_text())
                if context.get("resident_id") != preview["resident_id"]:
                    raise ArchiveError("Recovery context belongs to another resident")
                preview.update(name=descriptor["name"], release=descriptor["release"],
                               conversation_count=len(context["conversations"]))
            else:
                preview.update(name=manifest.get("configuration", {}).get("tamagi", {}).get("name", "TamAGI"),
                               release=None, conversation_count=None)
            preview.update(format=manifest["format"], mode="resident checkpoint replacement",
                shared_state="Current household state is retained; archive context remains historical evidence",
                continuity_gap="Private changes after the checkpoint will not be present in the restored root")
            return preview
        except BaseException:
            self.stages.pop(preview["id"], None)
            shutil.rmtree(Path(stage["directory"]).parent, ignore_errors=True)
            raise


def backup_resident(registry, actor, output, password):
    """Caller owns stopped_household; only this resident's data is exported."""
    from .native_registry import NativeRegistry
    descriptor = registry.descriptors.get(actor)
    if descriptor is None:
        raise ArchiveError("Unknown resident")
    NativeRegistry.validate(registry, descriptor)
    root = Path(descriptor["root"])
    config = yaml.safe_load((root / "config.yaml").read_text())
    if Path(config["runtime"]["instance_dir"]) != root or Path(config["workspace"]["path"]) != root / "workspace":
        raise ArchiveError("Resident configuration selects a different home")
    output = Path(output).absolute()
    if output != output.resolve() or output.exists() or output.is_relative_to(registry.root):
        raise ArchiveError("Choose a new backup file outside the household")
    conversations = [r for r in registry.store.list("conversation") if actor in r["data"]["participants"]]
    identities = {r["id"] for r in conversations}
    context = {"format": 1, "resident_id": actor, "household_id": registry.household_id,
        "historical_only": True, "participants": [registry.manifest["operator"], *[
            {"id": r["resident_id"], "name": r["name"]} for r in registry.manifest["residents"]]],
        "conversations": conversations,
        "messages": [r for r in registry.store.list("message") if r["data"]["conversation_id"] in identities]}
    store = RuntimeStore(root / "runtime.sqlite3")
    try:
        runtime = SimpleNamespace(store=store, actor_id=actor, workspace=root / "workspace", source_root=SOURCE)
        with tempfile.TemporaryDirectory(prefix="tamagi-resident-export-", dir=output.parent) as directory:
            archive = ResidentArchive(directory, runtime=runtime, metadata=descriptor, context=context)
            # Add immutable transcript evidence after the normal private data
            # inputs, giving the reserved recovery-evidence name one authority.
            result = archive.export(password, config)
            # Refuse a path created concurrently instead of replacing it.
            os.link(result["path"], output)
            output.chmod(0o600)
            sync_directory(output.parent)
            return {"path": str(output), "resident_id": actor, "name": descriptor["name"],
                    "exported_at": result["manifest"]["exported_at"],
                    "conversation_count": len(conversations), "omissions": result["manifest"]["omissions"]}
    finally:
        store.close()


def prepare_resident(archive, preview, destination, household_id, release, *, authority, source_stopped):
    """Materialize unselected private state, rebind paths and interrupt work."""
    if preview.get("release") and preview["release"] != release:
        raise ArchiveError("Application release differs; qualify compatibility before restoring this checkpoint")
    result = archive.commit(preview["id"], destination, source_stopped, authority)
    root = Path(result["instance_dir"])
    actor = result["resident_id"]
    try:
        store = RuntimeStore(root / "runtime.sqlite3")
        try:
            store.set_meta("household_id", household_id)
            # Pending private execution is evidence, not a restoration instruction.
            for record in store.list("interaction", actor, limit=100000):
                if record["data"].get("status") in {"queued", "running"}:
                    store.put("interaction", record["id"], actor, {**record["data"], "status": "interrupted",
                        "detail": "Resident restored; previous work was not replayed"}, category="platform")
            for record in store.list("household_outbox", actor, limit=100000):
                if record["data"].get("status") == "pending":
                    store.put("household_outbox", record["id"], actor, {**record["data"], "status": "outcome_unknown",
                        "recovery_note": "Inspect the canonical outcome before issuing another command"}, category="platform")
        finally:
            store.close()
        descriptor = {"id": actor, "resident_id": actor, "installation_id": result["installation_id"],
            "household_id": household_id, "root": str(root), "release": release, "protocol": PROTOCOL_VERSION,
            "name": preview["name"], "config_path": str(root / "config.yaml"), "avatar": "vex"}
        atomic_json(root / "instance.json", descriptor)
        config = TamAGIConfig.model_validate(yaml.safe_load((root / "config.yaml").read_text()))
        config.tamagi.name = preview["name"]
        config.autonomy.enabled = False
        config.runtime.automatic_deployment = False
        config.runtime.index_projection = False
        config.orchestrator.enabled = False
        config.llm.base_url, config.llm.api_key = "http://127.0.0.1:1/native-broker-only", ""
        config.self_model.data_path = str(root / "data/self_model.json")
        config.world_thread.consolidation.state_path = str(root / "data/consolidation_state.json")
        # Preserve integration descriptions in the encrypted manifest, but do
        # not reconnect services or launch archived stdio commands on hydration.
        config.mcp.servers = []
        (root / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
        (root / "config.yaml").chmod(0o600)
        sync_tree(root)
        return descriptor
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def restore_resident(registry, archive, preview, actor, *, confirmed, source_stopped, authority):
    if not confirmed:
        raise ArchiveError("Confirm the resident checkpoint replacement after reviewing the preview")
    if preview["resident_id"] != actor or actor not in registry.descriptors:
        raise ArchiveError("Checkpoint identity does not match the selected resident")
    destination = registry.root / "residents" / uid("recovered")
    descriptor = prepare_resident(archive, preview, destination, registry.household_id, registry.release,
                                  authority=authority, source_stopped=source_stopped)
    before = registry.descriptors[actor]
    # Once the intent is durable, startup can finish this exact selection even
    # if the process exits between database and manifest updates.
    atomic_json(registry.root / JOURNAL, {"format": 1, "before": before, "after": descriptor})
    reconcile_transition(registry)
    return {"resident_id": actor, "root": str(destination), "retained_root": before["root"],
            "installation_id": descriptor["installation_id"], "shared_state": "retained", "restart_required": True}


def migrate_resident(archive, preview, destination, *, release, confirmed, source_stopped, authority, port=7744):
    """Restore into an empty one-resident household; no peers are fabricated."""
    if not confirmed or not source_stopped:
        raise ArchiveError("Review the preview and confirm the source resident has stopped")
    destination = Path(destination).absolute()
    if destination != destination.resolve() or destination.exists():
        raise ArchiveError("Migration requires a new canonical household directory")
    destination.mkdir(mode=0o700)
    try:
        (destination / "residents").mkdir(mode=0o700)
        (destination / "ipc").mkdir(mode=0o700)
        household_id, installation_id = uid("household"), uid("installation")
        descriptor = prepare_resident(archive, preview, destination / "residents/r0", household_id, release,
                                      authority=authority, source_stopped=True)
        manifest = {"format": 1, "household_id": household_id, "installation_id": installation_id,
            "release": release, "protocol": PROTOCOL_VERSION, "operator": {"id": "operator", "name": "You"},
            "default_resident_id": descriptor["resident_id"], "residents": [descriptor]}
        store = HouseholdStore(destination / "household.sqlite3")
        try:
            store.set_meta("household_id", household_id)
            store.set_meta("installation_id", installation_id)
            store.put("resident", descriptor["resident_id"], descriptor)
        finally:
            store.close()
        config = TamAGIConfig()
        config.household.enabled, config.household.root = True, str(destination)
        config.server.port = port
        config.auth.enabled = True
        config.autonomy.enabled = False
        config.llm.base_url, config.llm.api_key = "http://127.0.0.1:1/reconnect-provider", ""
        (destination / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
        (destination / "config.yaml").chmod(0o600)
        atomic_json(destination / "household.json", manifest)
        sync_tree(destination)
        return {"root": str(destination), "resident_id": descriptor["resident_id"],
                "installation_id": installation_id, "credentials_required": True, "world_thread_enabled": False,
                "shared_state": "Historical context retained privately; shared-state recovery is separate"}
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
