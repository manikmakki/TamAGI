"""Explicit household-public, immutable file versions and deliberate private imports."""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from backend.core.runtime import current_execution
from backend.skills.runtime_skill import RuntimeSkill, param
from .contracts import AccessDenied, Conflict, NotFound, ProtocolError
from .store import uid

MAX_BYTES = 4 * 1024 * 1024
MAX_FILES = 64


def relative(value):
    if (not isinstance(value, str) or not value or len(value) > 512 or "\\" in value
            or "\x00" in value or PurePosixPath(value).is_absolute()
            or any(p in {"", ".", ".."} for p in value.split("/"))):
        raise ProtocolError("Choose a relative file or directory path without dot segments")
    return value


def local_path(workspace, value):
    path = workspace / relative(value)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(workspace.resolve()):
        raise AccessDenied("Shared files must stay within your workspace without symlinks")
    return path


def verified_files(files):
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ProtocolError("Publish 1–64 regular files, at most 4 MiB total")
    result, total = {}, 0
    for name, descriptor in files.items():
        relative(name)
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("data"), str) or len(descriptor["data"]) > 6 * 1024 * 1024:
            raise ProtocolError("Invalid shared file bytes")
        try:
            content = base64.b64decode(descriptor["data"], validate=True)
        except (ValueError, TypeError):
            raise ProtocolError("Invalid shared file encoding") from None
        total += len(content)
        if total > MAX_BYTES or descriptor.get("size") != len(content) or descriptor.get("sha256") != hashlib.sha256(content).hexdigest():
            raise ProtocolError("Shared file size or digest does not match its bytes")
        result[name] = content
    if any(parent.as_posix() in result for name in result for parent in PurePosixPath(name).parents):
        raise ProtocolError("A published path cannot be both a file and a directory")
    return result


class NativeArtifacts:
    def __init__(self, registry, model):
        self.registry, self.model, self.store = registry, model, registry.store

    @staticmethod
    def summary(record):
        return {"id": record["id"], **{k: v for k, v in record["data"].items() if k != "files"},
                "files": {name: {k: v for k, v in file.items() if k != "data"}
                          for name, file in record["data"]["files"].items()}}

    def listed(self):
        return [self.summary(r) for r in self.store.list("shared_artifact")][-100:]

    def version(self, identity, version=None):
        artifact = self.store.get("shared_artifact", identity)
        if not artifact:
            raise NotFound("No such household-public artifact")
        version = artifact["data"]["version"] if version is None else version
        if type(version) is not int or version < 1:
            raise ProtocolError("Choose a positive published version number")
        record = self.store.get("shared_artifact_version", f"{identity}:v{version}")
        if not record:
            raise NotFound("No such published version")
        return record

    def dispatch(self, actor, body):
        self.registry.resolve(actor)
        action = body.get("action")
        if action == "list":
            return self.listed()
        if action in {"inspect", "fetch"}:
            record = self.version(body.get("artifact_id"), body.get("version"))
            if action == "fetch":
                verified_files(record["data"]["files"])
                return {"id": record["id"], **record["data"]}
            return self.summary(record)
        if action != "publish":
            raise ProtocolError("Choose publish, list or inspect")
        def publish():
            _, execution, _ = self.model._active(actor, body.get("local_execution_id"))
            if execution["helper"] or execution["actor_id"] != actor:
                raise AccessDenied("Only a full resident may publish household artifacts")
            files = body.get("files")
            verified_files(files)
            title, description = body.get("title"), body.get("description", "")
            if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120 or not isinstance(description, str) or len(description) > 2000:
                raise ProtocolError("Provide a title up to 120 characters and description up to 2000")
            identity = body.get("artifact_id") or uid("artifact")
            prior = self.store.get("shared_artifact", identity)
            if prior:
                if type(body.get("expected")) is not int or body["expected"] != prior["data"]["version"]:
                    raise Conflict("The artifact changed; inspect its current version before publishing")
            elif body.get("artifact_id"):
                raise NotFound("No such household-public artifact; omit artifact_id for a first publication")
            version = prior["data"]["version"] + 1 if prior else 1
            data = {"artifact_id": identity, "version": version, "title": title,
                    "description": description, "author_id": actor, "published_at": self.store.clock(),
                    "visibility": "household_public", "files": files}
            self.store.put("shared_artifact_version", f"{identity}:v{version}", data, expected=0)
            record = self.store.put("shared_artifact", identity, data,
                                    expected=prior["revision"] if prior else 0)
            self.store.event("artifact_published", actor, {"artifact_id": identity, "version": version})
            return self.summary(record)
        return self.store.command(actor, body["command_id"], "artifact.publish", body, publish)


def install_artifacts(runtime, skills):
    def artifact(action, path=None, title="", description="", artifact_id=None, version=None,
                 expected=None, destination=None):
        context = current_execution.get()
        if not context or context.actor_id != runtime.actor_id or context.depth:
            raise AccessDenied("Only a full resident may use household artifacts")
        runtime.household.assert_active(context)
        if action in {"list", "inspect"}:
            return runtime.household._request("POST", "/_native/artifacts", json={
                "action": action, "artifact_id": artifact_id, "version": version})
        if action == "publish":
            source = local_path(runtime.workspace, path)
            if source.is_dir():
                entries = list(source.rglob("*"))
                if any(p.is_symlink() for p in entries):
                    raise AccessDenied("Published directories cannot contain symlinks")
                paths = [p for p in entries if p.is_file()]
                base = source
            else:
                paths, base = [source], source.parent
            if not 1 <= len(paths) <= MAX_FILES:
                raise ProtocolError("Publish 1–64 regular files, at most 4 MiB total")
            files, total = {}, 0
            for entry in paths:
                with entry.open("rb") as stream:
                    content = stream.read(MAX_BYTES + 1)
                total += len(content)
                if total > MAX_BYTES:
                    raise ProtocolError("A publication is limited to 4 MiB total")
                files[entry.relative_to(base).as_posix()] = {"data": base64.b64encode(content).decode(),
                    "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            receipt = next((r for r in runtime.store.receipts(context.execution_id, limit=100)
                            if r["tool"] == "household_artifact" and r["status"] == "started"), None)
            command = "artifact:" + (receipt["id"] if receipt else uid("operation"))
            body = {"action": "publish", "local_execution_id": context.execution_id, "command_id": command,
                    "title": title, "description": description, "artifact_id": artifact_id, "expected": expected, "files": files}
            runtime.store.put("household_outbox", command, runtime.actor_id, {"status": "pending", "payload": body}, category="platform")
            result = runtime.household._request("POST", "/_native/artifacts", json=body)
            runtime.store.put("household_outbox", command, runtime.actor_id, {"status": "committed", "payload": body}, category="platform")
            return result
        if action != "import":
            raise ProtocolError("Choose publish, list, inspect or import")
        record = runtime.household._request("POST", "/_native/artifacts", json={"action": "fetch", "artifact_id": artifact_id, "version": version})
        files = verified_files(record["files"])
        target = local_path(runtime.workspace, destination or f"shared/{record['artifact_id']}/v{record['version']}")
        if target.exists():
            actual = [p for p in target.rglob("*") if p.is_file()] if target.is_dir() else []
            if (target.is_dir() and not any(p.is_symlink() for p in target.rglob("*"))
                    and {p.relative_to(target).as_posix(): p.read_bytes() for p in actual} == files):
                return {"artifact_id": record["artifact_id"], "version": record["version"], "destination": str(target.relative_to(runtime.workspace)), "already_imported": True}
            raise Conflict("Import destination already exists with different bytes; choose a new directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".artifact-import-", dir=target.parent))
        try:
            for name, content in files.items():
                file = temporary / name
                file.parent.mkdir(parents=True, exist_ok=True)
                with file.open("wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            runtime.household.assert_active(context)
            for directory in sorted([temporary, *(p for p in temporary.rglob("*") if p.is_dir())], key=lambda p: len(p.parts), reverse=True):
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            os.rename(temporary, target)
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        result = {"artifact_id": record["artifact_id"], "version": record["version"],
                  "author_id": record["author_id"], "destination": str(target.relative_to(runtime.workspace)),
                  "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
        runtime.store.put("household_artifact_import", str(target.relative_to(runtime.workspace)), runtime.actor_id, result, category="platform")
        return result

    skills.register(RuntimeSkill(runtime, "household_artifact",
        "Publish a file/directory as a versioned artifact PUBLIC TO ALL HOUSEHOLD RESIDENTS, list/inspect publications, or import exact published bytes into your private workspace. Publish requires path and title; updates require artifact_id and expected current version. Import requires artifact_id, optional version and a new destination directory. Imports never overwrite files or activate executable skills; review and use runtime_change for activation. Other residents may publish attributed new versions, including nonparticipants in the original collaboration.",
        {"action": param("string", required=True, enum=["publish", "list", "inspect", "import"]),
         "path": param("string", "Own workspace file/directory to explicitly publish"),
         "title": param("string"), "description": param("string"), "artifact_id": param("string"),
         "version": param("integer", "Published version to inspect/import; defaults to latest"),
         "expected": param("integer", "Required current version for publishing an update"),
         "destination": param("string", "New private import directory; defaults to shared/<artifact>/v<version>")}, artifact))
