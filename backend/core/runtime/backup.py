"""Portable encrypted checkpoints. Import is staged; no uploaded code is executed."""

from __future__ import annotations
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from contextlib import closing
from urllib.parse import urlsplit, urlunsplit
from .store import uid, encoded

EXCLUDED_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".aws",
    ".ssh",
    ".gnupg",
    ".env",
    "secrets.enc",
    ".secrets.key",
    ".session_secret",
    "operator.key",
    "config.yaml",
    "config.platform.yaml",
    "chromadb",
    "models",
    ".cache",
    "backups",
    "archives",
    ".platform",
    ".worktrees",
    ".releases",
}
EXCLUDED_SUFFIXES = {".gguf", ".safetensors", ".onnx", ".pyc"}
SECRET_FIELDS = {
    "api_key",
    "password",
    "password_hash",
    "secret_key",
    "session_secret",
    "token",
    "access_token",
    "private_key",
    "client_secret",
}


class ArchiveError(ValueError):
    pass


def sanitized(value):
    if isinstance(value, dict):
        return {
            k: ("" if k.lower() in SECRET_FIELDS else sanitized(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitized(v) for v in value]
    if isinstance(value, str) and "://" in value:
        try:
            url = urlsplit(value)
            if url.username or url.password:
                return urlunsplit(
                    (url.scheme, url.hostname or "", url.path, url.query, url.fragment)
                )
        except ValueError:
            pass
    return value


class PortableBackup:
    FORMAT = "tamagi.portable.v1"

    def __init__(self, runtime, job_root=None, max_bytes=10 * 1024**3):
        self.runtime = runtime
        self.root = Path(job_root or runtime.store.path.parent / "transfer").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.stages = {}
        # One owning instance process; abandon decrypted previews from a previous process.
        for leftover in self.root.iterdir():
            if leftover.name.startswith(
                ("restore-", "checkpoint-", "encrypted-upload-")
            ):
                if leftover.is_dir():
                    shutil.rmtree(leftover)
                elif leftover.is_file():
                    leftover.unlink()

    @staticmethod
    def _password(password):
        if (
            not isinstance(password, str)
            or len(password) < 12
            or len(password) > 1024
            or "\n" in password
            or "\r" in password
        ):
            raise ArchiveError(
                "Use a password of 12–1024 characters without line breaks"
            )

    def _crypt(self, source, target, password, decrypt=False):
        self._password(password)
        if not shutil.which("gpg"):
            raise ArchiveError("GnuPG is required for encrypted portable archives")
        with tempfile.TemporaryDirectory(prefix="tamagi-gpg-") as directory:
            os.chmod(directory, 0o700)
            read_fd, write_fd = os.pipe()
            os.write(write_fd, password.encode() + b"\n")
            os.close(write_fd)
            command = [
                "gpg",
                "--no-options",
                "--homedir",
                directory,
                "--batch",
                "--yes",
                "--no-tty",
                "--pinentry-mode",
                "loopback",
                "--no-symkey-cache",
                "--passphrase-fd",
                str(read_fd),
                "--output",
                str(target),
            ]
            command += (
                ["--decrypt"]
                if decrypt
                else [
                    "--cipher-algo",
                    "AES256",
                    "--compress-algo",
                    "none",
                    "--symmetric",
                ]
            )
            try:
                process = (
                    subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE
                        if callable(source)
                        else subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        pass_fds=(read_fd,),
                    )
                    if callable(source)
                    else subprocess.Popen(
                        command + [str(source)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        pass_fds=(read_fd,),
                    )
                )
                os.close(read_fd)
                read_fd = -1
                try:
                    if callable(source):
                        source(process.stdin)
                        process.stdin.close()
                        process.stdin = None
                    _, stderr = process.communicate(timeout=300)
                except BaseException:
                    process.kill()
                    process.communicate()
                    raise
                if process.returncode:
                    target.unlink(missing_ok=True)
                    raise ArchiveError(
                        "Archive decryption/integrity check failed"
                        if decrypt
                        else "Archive encryption failed"
                    )
            finally:
                if read_fd >= 0:
                    os.close(read_fd)
                if shutil.which("gpgconf"):
                    subprocess.run(
                        ["gpgconf", "--homedir", directory, "--kill", "gpg-agent"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                    )
        os.chmod(target, 0o600)

    @staticmethod
    def _eligible(path, base):
        rel = path.relative_to(base)
        return (
            not any(p in EXCLUDED_NAMES for p in rel.parts)
            and path.suffix.lower() not in EXCLUDED_SUFFIXES
        )

    def export(self, password, config=None):
        """Caller holds RuntimeService.checkpointing; only encrypted output survives."""
        export_id = uid("backup")
        destination = self.root / (export_id + ".tamagi.tar.gz.gpg")
        with tempfile.TemporaryDirectory(
            prefix="checkpoint-", dir=self.root
        ) as directory:
            stage = Path(directory)
            os.chmod(stage, 0o700)
            self.runtime.store.checkpoint(stage / "runtime.sqlite3")
            files = {"runtime.sqlite3": stage / "runtime.sqlite3"}
            omissions = []
            sources = [
                ("workspace", self.runtime.workspace, None),
                (
                    "source",
                    getattr(
                        self.runtime, "release_source_root", self.runtime.source_root
                    ),
                    {
                        "backend",
                        "frontend",
                        "scripts",
                        "tests",
                        "docs",
                        "pyproject.toml",
                        "requirements.txt",
                        "requirements.lock",
                        "Dockerfile",
                        "docker-compose.yml",
                        "tamagi.service",
                        "tamagi-platform.service",
                        "LICENSE.md",
                    },
                ),
                (
                    "data",
                    self.runtime.store.path.parent / "data",
                    {"sprites", "sprite_config.json"},
                ),
            ]
            for prefix, base, allowed_roots in sources:
                if not base.exists():
                    continue
                for directory, subdirs, names in os.walk(base, followlinks=False):
                    directory = Path(directory)
                    subdirs[:] = sorted(
                        d
                        for d in subdirs
                        if d not in EXCLUDED_NAMES
                        and not (directory / d).is_symlink()
                        and (
                            directory != base
                            or allowed_roots is None
                            or d in allowed_roots
                        )
                    )
                    for name in sorted(names):
                        path = directory / name
                        if not self._eligible(path, base):
                            continue
                        rel = path.relative_to(base)
                        if (
                            allowed_roots is not None
                            and rel.parts[0] not in allowed_roots
                        ):
                            continue
                        if path.is_symlink():
                            omissions.append(
                                f"{prefix}/{rel}: symbolic link not copied"
                            )
                        elif path.is_file():
                            files[f"{prefix}/{rel.as_posix()}"] = path
            manifest = {
                "format": self.FORMAT,
                "schema": self.runtime.store.SCHEMA,
                "resident_id": self.runtime.actor_id,
                "installation_id": self.runtime.store.meta("installation_id"),
                "exported_at": self.runtime.store.clock(),
                "configuration": sanitized(config or {}),
                "paths": {
                    "workspace": str(self.runtime.workspace),
                    "canonical_repo": str(self.runtime.source_root),
                    "source": str(
                        getattr(
                            self.runtime,
                            "release_source_root",
                            self.runtime.source_root,
                        )
                    ),
                },
                "files": {},
                "omissions": omissions,
                "excludes": [
                    "credential stores",
                    "model weights",
                    "derived indexes",
                    "predecessor archive",
                    "unregistered host changes",
                ],
                "note": "Artifacts and conversation text may contain embedded secrets; this is not a complete machine image.",
            }
            # MCP environments are integration configuration; carry secret names, not raw environment values.
            for server in manifest["configuration"].get("mcp", {}).get("servers", []):
                server["env"] = {
                    k: (v if isinstance(v, dict) and set(v) == {"secret"} else "")
                    for k, v in server.get("env", {}).items()
                }
            for name, path in files.items():
                size = path.stat().st_size
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                manifest["files"][name] = {"size": size, "sha256": digest.hexdigest()}
            total = sum(f["size"] for f in manifest["files"].values())
            if (
                total > self.max_bytes
                or shutil.disk_usage(self.root).free < total + 1024**2
            ):
                raise ArchiveError(
                    "Checkpoint exceeds transfer limit or available space"
                )

            def pack(output):
                with tarfile.open(fileobj=output, mode="w|gz") as archive:
                    payload = encoded(manifest).encode()
                    member = tarfile.TarInfo("manifest.json")
                    member.size = len(payload)
                    member.mode = 0o600
                    archive.addfile(member, io.BytesIO(payload))
                    for name, path in files.items():
                        archive.add(path, arcname=name, recursive=False)

            try:
                self._crypt(pack, destination, password)
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
        self.runtime.store.set_meta("last_export_at", manifest["exported_at"])
        self.runtime.store.event(
            self.runtime.actor_id,
            "backup_created",
            {"id": export_id, "exported_at": manifest["exported_at"]},
        )
        return {
            "id": export_id,
            "path": str(destination),
            "filename": destination.name,
            "manifest": manifest,
        }

    def stage(self, encrypted, password):
        stage_id = uid("restore")
        directory = self.root / stage_id
        directory.mkdir(mode=0o700)
        decrypted = directory / "checkpoint.tar.gz"
        try:
            if Path(encrypted).stat().st_size > self.max_bytes:
                raise ArchiveError("Upload exceeds transfer limit")
            self._crypt(Path(encrypted), decrypted, password, decrypt=True)
            extracted = directory / "contents"
            extracted.mkdir(mode=0o700)
            with tarfile.open(decrypted, "r:gz") as archive:
                members = []
                seen = set()
                total = 0

                for member in archive:
                    members.append(member)
                    if len(members) > 100000:
                        raise ArchiveError("Archive has too many members")
                    path = PurePosixPath(member.name)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or "\\" in member.name
                        or member.name in seen
                        or not (member.isfile() or member.isdir())
                    ):
                        raise ArchiveError("Unsafe or duplicate archive member")
                    if not path.parts or path.parts[0] not in {
                        "manifest.json",
                        "runtime.sqlite3",
                        "workspace",
                        "source",
                        "data",
                    }:
                        raise ArchiveError("Unrecognized archive root")
                    seen.add(member.name)
                    total += member.size
                    if total > self.max_bytes:
                        raise ArchiveError("Expanded archive exceeds transfer limit")
                if shutil.disk_usage(directory).free < total + 1024**2:
                    raise ArchiveError("Insufficient space to stage restoration")
                # Never use extractall on untrusted input. No links or device members are accepted.
                for member in members:
                    target = extracted / member.name
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with (
                            archive.extractfile(member) as source,
                            target.open("xb") as output,
                        ):
                            shutil.copyfileobj(source, output)
                        target.chmod(0o600)
            manifest = json.loads((extracted / "manifest.json").read_text())
            if (
                manifest.get("format") != self.FORMAT
                or manifest.get("schema") != self.runtime.store.SCHEMA
            ):
                raise ArchiveError("Unsupported archive format or schema")
            listed = set(manifest["files"]) | {"manifest.json"}
            actual = {
                str(p.relative_to(extracted).as_posix())
                for p in extracted.rglob("*")
                if p.is_file()
            }
            if actual != listed:
                raise ArchiveError("Archive contents do not match its manifest")
            for name, expected in manifest["files"].items():
                path = extracted / name
                if path.stat().st_size != expected["size"]:
                    raise ArchiveError("Archive file size mismatch")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected["sha256"]:
                    raise ArchiveError("Archive content hash mismatch")
            with closing(
                sqlite3.connect(
                    f"file:{extracted / 'runtime.sqlite3'}?mode=ro", uri=True
                )
            ) as db:
                if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ArchiveError("Checkpoint database is damaged")
                resident = json.loads(
                    db.execute(
                        "SELECT value FROM meta WHERE key='resident_id'"
                    ).fetchone()[0]
                )
                if resident != manifest["resident_id"]:
                    raise ArchiveError("Resident identity mismatch")
            decrypted.unlink()
            self.stages[stage_id] = {
                "directory": str(extracted),
                "manifest": manifest,
                "created": self.runtime.store.clock(),
            }
            return {
                "id": stage_id,
                "resident_id": resident,
                "exported_at": manifest["exported_at"],
                "source_installation_id": manifest["installation_id"],
                "omissions": manifest["omissions"],
                "requires": [
                    "source installation stopped",
                    "installation authority choice",
                    "credential reconnection",
                ],
                "file_count": len(manifest["files"]),
            }
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def commit(self, stage_id, destination, source_stopped, authority):
        if not source_stopped:
            raise ArchiveError(
                "Confirm the source installation is stopped before continuing its identity"
            )
        if authority not in {"guarded", "unrestricted_host"}:
            raise ArchiveError("Choose the authority for this installation")
        stage = self.stages.get(stage_id)
        if not stage or self.runtime.store.clock() - stage["created"] > 1800:
            raise ArchiveError("Restore preview expired; upload the archive again")
        destination = Path(destination).resolve()
        if destination.exists():
            raise ArchiveError(
                "Restore destination must be new; active state is never overwritten"
            )
        source = Path(stage["directory"])
        try:
            shutil.copytree(source, destination)
            from .store import RuntimeStore

            store = RuntimeStore(destination / "runtime.sqlite3")
            installation_id = uid("installation")
            store.set_meta("installation_id", installation_id)
            store.set_meta("authority", authority)
            store.set_meta("deployment_qualification", None)
            store.set_meta("offhost_copy_acknowledged", False)
            old_paths = stage["manifest"].get("paths", {})
            for revision in store.list("runtime_revision"):
                data = revision["data"]
                if data.get("surface") == "tool" and data.get("active"):
                    definition = dict(data["value"])
                    command = []
                    for argument in definition["command"]:
                        for kind, new_base in [
                            ("workspace", destination / "workspace"),
                            ("source", destination / "source"),
                            ("canonical_repo", self.runtime.source_root),
                        ]:
                            old_base = old_paths.get(kind)
                            if old_base and (
                                argument == old_base
                                or argument.startswith(old_base + "/")
                            ):
                                argument = str(new_base) + argument[len(old_base) :]
                        command.append(argument)
                    definition["command"] = command
                    store.put(
                        "runtime_revision",
                        revision["id"],
                        revision["actor"],
                        {**data, "value": definition, "installation_rebound": True},
                        category="platform",
                    )
            store.set_meta(
                "restore",
                {
                    "exported_at": stage["manifest"]["exported_at"],
                    "restored_at": store.clock(),
                    "source_installation_id": stage["manifest"]["installation_id"],
                    "gap": "Post-checkpoint history is unavailable unless separately recovered.",
                },
            )
            store.recover()
            store.event(store.resident_id, "timeline_restored", store.meta("restore"))
            store.close()
            # Generate location-independent configuration. Credentials and host authority never carry implicitly.
            config = stage["manifest"].get("configuration") or {}
            config["runtime"] = {
                **config.get("runtime", {}),
                "enabled": True,
                "instance_dir": str(destination),
                "authority": authority,
                "canonical_repo": str(self.runtime.source_root),
                "automatic_deployment": False,
            }
            config["workspace"] = {"path": str(destination / "workspace")}
            config["history"] = {
                **config.get("history", {}),
                "persist_path": str(destination / "data/history"),
            }
            config["auth"] = {"enabled": True, "password_hash": ""}
            config["world_thread"] = {
                **config.get("world_thread", {}),
                "state_path": str(destination / "data/world_state.json"),
                "thread_path": str(destination / "data/world_thread.json"),
            }
            config["self_model"] = {
                **config.get("self_model", {}),
                "data_path": str(destination / "data/self_model.json"),
            }
            import yaml

            (destination / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False)
            )
            return {
                "resident_id": stage["manifest"]["resident_id"],
                "installation_id": installation_id,
                "instance_dir": str(destination),
                "config_path": str(destination / "config.yaml"),
                "activation_required": True,
                "credentials_required": True,
            }
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(source.parent, ignore_errors=True)
            self.stages.pop(stage_id, None)

    def expire(self):
        now = self.runtime.store.clock()
        for stage_id, stage in list(self.stages.items()):
            if now - stage["created"] > 1800:
                shutil.rmtree(Path(stage["directory"]).parent, ignore_errors=True)
                self.stages.pop(stage_id, None)
        for path in self.root.glob("backup-*.gpg"):
            if now - path.stat().st_mtime > 86400:
                path.unlink(missing_ok=True)

    def cleanup(self):
        for stage_id, stage in list(self.stages.items()):
            shutil.rmtree(Path(stage["directory"]).parent, ignore_errors=True)
            self.stages.pop(stage_id, None)
