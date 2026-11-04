"""Provision, preserve, verify, and run TamAGI without mutating a predecessor."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .core.runtime.store import RuntimeStore, uid, encoded
from .core.runtime.service import RuntimeService
from .core.runtime.backup import sanitized
from .config import TamAGIConfig, RuntimeConfig

SOURCE = Path(__file__).resolve().parents[1]


def logical_legacy_export(raw, destination):
    memory = raw.get("memory", {})
    backend = memory.get("backend", "chromadb")
    if backend == "elasticsearch":
        import httpx

        cfg = memory.get("elasticsearch", {})
        url = cfg.get("url", "http://localhost:9200").rstrip("/")
        headers = {}
        auth = None
        if cfg.get("api_key"):
            headers["Authorization"] = "ApiKey " + cfg["api_key"]
        elif cfg.get("username"):
            auth = (cfg["username"], cfg.get("password", ""))
        count = 0
        scroll = None
        with (
            httpx.Client(timeout=15, headers=headers, auth=auth) as client,
            destination.open("w") as output,
        ):
            response = client.post(
                f"{url}/{cfg.get('index', 'tamagi_memory')}/_search",
                params={"scroll": "1m"},
                json={"size": 500, "sort": ["_doc"], "query": {"match_all": {}}},
            )
            response.raise_for_status()
            while True:
                result = response.json()
                scroll = result.get("_scroll_id")
                hits = result.get("hits", {}).get("hits", [])
                if not hits:
                    break
                for hit in hits:
                    source = hit.get("_source", {})
                    if isinstance(source.get("data"), dict):
                        source["data"].pop("content_vector", None)
                    output.write(
                        encoded(
                            {
                                "id": hit["_id"],
                                "source": source,
                                "source_kind": "legacy",
                            }
                        )
                        + "\n"
                    )
                    count += 1
                if not scroll:
                    break
                response = client.post(
                    f"{url}/_search/scroll", json={"scroll": "1m", "scroll_id": scroll}
                )
                response.raise_for_status()
            if scroll:
                client.request(
                    "DELETE", f"{url}/_search/scroll", json={"scroll_id": [scroll]}
                )
        return {"status": "complete", "backend": backend, "records": count}
    fallback = SOURCE / "data/memory_fallback.json"
    if fallback.exists():
        shutil.copy2(fallback, destination)
        return {"status": "complete", "backend": "fallback", "format": "original JSON"}
    directory = Path(
        memory.get("chromadb", {}).get("persist_directory", "data/chromadb")
    )
    if not directory.is_absolute():
        directory = SOURCE / directory
    if not directory.exists():
        destination.write_text("")
        return {"status": "complete", "backend": backend, "records": 0}
    try:
        import chromadb
    except ImportError:
        return {
            "status": "needs_chroma_export",
            "backend": backend,
            "physical_source": str(directory),
        }
    client = chromadb.PersistentClient(path=str(directory))
    collection = client.get_collection(
        memory.get("chromadb", {}).get("collection_name", "tamagi_memory")
    )
    result = collection.get(include=["documents", "metadatas"])
    with destination.open("w") as output:
        for index, key in enumerate(result["ids"]):
            output.write(
                encoded(
                    {
                        "id": key,
                        "content": result["documents"][index],
                        "metadata": result["metadatas"][index],
                        "source_kind": "legacy",
                    }
                )
                + "\n"
            )
    return {"status": "complete", "backend": backend, "records": len(result["ids"])}


def archive_legacy(config_path, destination=None):
    import yaml

    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    destination = Path(
        destination or SOURCE / ".platform/operator-archives" / uid("predecessor")
    ).resolve()
    if destination.exists():
        raise ValueError("Archive destination must not exist")
    destination.mkdir(parents=True, mode=0o700)
    manifest = {
        "kind": "operator_predecessor_archive",
        "created_at": time.time(),
        "source_revision": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "operator_only": True,
        "memory": {"status": "pending"},
        "files": {},
        "credential_stores_excluded": True,
    }
    try:
        for name, source in [
            ("data", SOURCE / "data"),
            (
                "workspace",
                Path(raw.get("workspace", {}).get("path", SOURCE / "workspace")),
            ),
        ]:
            if not source.is_absolute():
                source = SOURCE / source
            if source.exists():
                shutil.copytree(
                    source,
                    destination / name,
                    ignore=shutil.ignore_patterns(
                        "instances",
                        "operator-archives",
                        "transfer",
                        "__pycache__",
                        "secrets.enc",
                        ".secrets.key",
                        ".session_secret",
                    ),
                    symlinks=True,
                )
        (destination / "configuration.json").write_text(encoded(sanitized(raw)))
        archive = destination / "source.tar"
        with archive.open("wb") as output:
            subprocess.run(
                ["git", "-C", str(SOURCE), "archive", "HEAD"], stdout=output, check=True
            )
        try:
            manifest["memory"] = logical_legacy_export(
                raw, destination / "memory.jsonl"
            )
        except Exception as exc:
            manifest["memory"] = {
                "status": "unavailable",
                "error_class": type(exc).__name__,
            }
        for path in destination.rglob("*"):
            if path.is_file() and not path.is_symlink():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                manifest["files"][str(path.relative_to(destination))] = (
                    digest.hexdigest()
                )
        manifest["complete"] = manifest["memory"]["status"] == "complete"
        (destination / "manifest.json").write_text(encoded(manifest))
        return {
            "archive": str(destination),
            "complete": manifest["complete"],
            "memory": manifest["memory"],
        }
    except BaseException:
        # Preserve incomplete evidence; do not delete the predecessor or pretend the snapshot succeeded.
        (destination / "manifest.json").write_text(encoded(manifest))
        raise


def provision(
    archive, authority="guarded", instance=None, legacy_config=None, port=7742
):
    import yaml

    if archive:
        archive = Path(archive).resolve()
        manifest = json.loads((archive / "manifest.json").read_text())
        if (
            not manifest.get("complete")
            or manifest.get("kind") != "operator_predecessor_archive"
        ):
            raise ValueError(
                "Complete the predecessor archive and logical memory export before fresh cutover"
            )
    elif (SOURCE / "workspace/IDENTITY.md").exists() or (
        SOURCE / "data/tamagi_state.json"
    ).exists():
        raise ValueError("An existing resident must be archived before a fresh start")
    if authority not in {"guarded", "unrestricted_host"}:
        raise ValueError("Choose an installation authority")
    instance = Path(instance or SOURCE / "data/instances" / uid("home")).resolve()
    if instance.exists():
        raise ValueError("Fresh instance directory must not exist")
    instance.mkdir(parents=True, mode=0o700)
    (instance / "data").mkdir()
    config = TamAGIConfig()
    if legacy_config:
        # Transport configuration is reusable; persona, user memory, and integrations are not inherited.
        raw = yaml.safe_load(Path(legacy_config).read_text()) or {}
        if raw.get("llm"):
            config.llm = type(config.llm).model_validate(raw["llm"])
    config.runtime = RuntimeConfig(
        enabled=True,
        instance_dir=str(instance),
        authority=authority,
        canonical_repo=str(SOURCE),
    )
    config.workspace.path = str(instance / "workspace")
    config.history.persist_path = str(instance / "data/history")
    config.history.context_compress_threshold = 0
    config.history.max_messages_per_conversation = 1000
    config.server.host = "127.0.0.1"
    config.server.port = port
    config.auth.enabled = True
    runtime = RuntimeService(
        RuntimeStore(instance / "runtime.sqlite3"),
        instance / "workspace",
        config.runtime,
        SOURCE,
    )
    if archive:
        runtime.store.set_meta("predecessor_archive", str(archive))
    metadata = {
        "resident_id": runtime.actor_id,
        "installation_id": runtime.store.meta("installation_id"),
        "created_at": time.time(),
        "fresh": True,
    }
    (instance / "instance.json").write_text(encoded(metadata))
    (instance / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(), sort_keys=False)
    )
    (instance / "config.yaml").chmod(0o600)
    runtime.store.close()
    return {
        **metadata,
        "config": str(instance / "config.yaml"),
        "instance": str(instance),
        "operator_setup_required": True,
    }


def open_runtime(config_path):
    import yaml

    config = TamAGIConfig.model_validate(yaml.safe_load(Path(config_path).read_text()))
    root = Path(config.runtime.instance_dir)
    from .config import set_config

    set_config(config)
    return RuntimeService(
        RuntimeStore(root / "runtime.sqlite3"),
        root / "workspace",
        config.runtime,
        SOURCE,
        recover=False,
    )


def qualify(runtime):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(SOURCE / "tests"),
            "-q",
        ],
        cwd=SOURCE,
        env=env,
        capture_output=True,
        timeout=600,
    )
    report = {
        "passed": result.returncode == 0,
        "evidence_class": "scripted",
        "completed_at": time.time(),
        "command": "python -m unittest discover -s tests -q",
        "output": result.stderr.decode(errors="replace")[-4000:]
        if isinstance(result.stderr, bytes)
        else result.stderr[-4000:],
        "source_revision": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    runtime.store.set_meta("deployment_qualification", report)
    return report


def install_supervisor(runtime, config_path):
    from .core.runtime.deployment import DeploymentManager, atomic_json, switch_link

    manager = DeploymentManager(runtime)
    # Capture the implemented working source, including authored files not yet committed.
    inventory = manager.inventory(SOURCE)
    source_hash = hashlib.sha256(encoded(inventory).encode()).hexdigest()
    release = manager.root / "releases" / source_hash
    if release.exists():
        raise ValueError("Initial release already exists")
    (release / "source").mkdir(parents=True)
    for name in inventory:
        target = release / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    subprocess.run(
        [sys.executable, "-m", "venv", str(release / "venv")],
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run(
        [
            str(release / "venv/bin/python"),
            "-m",
            "pip",
            "install",
            "-r",
            str(release / "source/requirements.lock"),
        ],
        check=True,
        capture_output=True,
        timeout=600,
    )
    (release / "manifest.json").write_text(
        encoded(
            {
                "source_hash": source_hash,
                "inventory": inventory,
                "schema": runtime.store.SCHEMA,
                "initial_release": True,
            }
        )
    )
    manager.boot_check(release)
    shutil.copy2(SOURCE / "scripts/tamagi-supervisor.py", manager.root / "bootstrap.py")
    switch_link(manager.root / "current", release)
    atomic_json(
        manager.root / "active.json",
        {
            "instance": str(runtime.store.path.parent),
            "config": str(Path(config_path).resolve()),
        },
    )
    return {
        "release": str(release),
        "supervisor": str(manager.root / "bootstrap.py"),
        "root": str(manager.root),
    }


def pilot(runtime, operation):
    from .core.runtime.deployment import DeploymentManager

    DeploymentManager(runtime).synchronize()
    now = time.time()
    report = runtime.store.meta("pilot")
    if operation == "start":
        if report:
            raise ValueError("A retained pilot already exists; do not overwrite it")
        report = {
            "started_at": now,
            "ends_at": now + 7 * 86400,
            "evidence_class": "sustained_operation",
            "status": "running",
        }
        runtime.store.set_meta("pilot", report)
    if not report:
        raise ValueError("No pilot has started")
    turns = [
        r["data"]
        for r in runtime.store.list("world_turn", limit=100000)
        if r["created"] >= report["started_at"]
    ]
    deployments = [
        r["data"]
        for r in runtime.store.list("deployment")
        if r["updated"] >= report["started_at"]
    ]
    observed = {
        "independent_choice": bool(turns),
        "chosen_rest": any(t["status"] == "resting" for t in turns),
        "companion_conversation": any(
            r["created"] >= report["started_at"]
            for r in runtime.store.list("invocation", limit=100000)
        ),
        "tool_failure": any(
            r["status"] == "failed" and r["started"] >= report["started_at"]
            for r in runtime.store.receipts(limit=10000)
        ),
        "deployment": any(d["phase"] == "committed" for d in deployments),
        "rollback": any(d["phase"] == "rolled_back" for d in deployments),
    }
    with runtime.store.lock:
        resumed = runtime.store.db.execute(
            "SELECT count(*) FROM events WHERE kind='pursuit_continued_after_restart' AND created>=?",
            (report["started_at"],),
        ).fetchone()[0]
    observed["accepted_work_across_restart"] = resumed > 0
    report = {
        **report,
        "observed": observed,
        "remaining_seconds": max(0, report["ends_at"] - now),
    }
    if operation == "finalize":
        if now < report["ends_at"]:
            raise ValueError(
                "Seven actual days have not elapsed; this is not sustained qualification"
            )
        report["status"] = (
            "observed" if all(observed.values()) else "incomplete_observations"
        )
        runtime.store.set_meta("pilot", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    arch = commands.add_parser("archive")
    arch.add_argument("--config", default=str(SOURCE / "config.yaml"))
    arch.add_argument("--destination")
    fresh = commands.add_parser("fresh")
    fresh.add_argument("--archive")
    fresh.add_argument(
        "--authority", choices=["guarded", "unrestricted_host"], required=True
    )
    fresh.add_argument("--instance")
    fresh.add_argument("--legacy-config")
    fresh.add_argument("--port", type=int, default=7742)
    restore = commands.add_parser("restore")
    restore.add_argument("--archive", required=True)
    restore.add_argument("--destination", required=True)
    restore.add_argument(
        "--authority", choices=["guarded", "unrestricted_host"], required=True
    )
    restore.add_argument("--source-stopped", action="store_true", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--config", required=True)
    for name in ("qualify", "install-supervisor", "pilot"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "pilot":
            p.add_argument("operation", choices=["start", "status", "finalize"])
    args = parser.parse_args()
    if args.command == "archive":
        result = archive_legacy(args.config, args.destination)
    elif args.command == "fresh":
        result = provision(
            args.archive, args.authority, args.instance, args.legacy_config, args.port
        )
    elif args.command == "restore":
        import getpass
        import tempfile
        from .core.runtime.backup import PortableBackup

        with tempfile.TemporaryDirectory(prefix="tamagi-recovery-") as temporary:
            root = Path(temporary)
            recovery = RuntimeService(
                RuntimeStore(root / "runtime.sqlite3"),
                root / "workspace",
                RuntimeConfig(),
                SOURCE,
            )
            try:
                transfer = PortableBackup(recovery)
                password = getpass.getpass("Archive password: ")
                preview = transfer.stage(Path(args.archive), password)
                password = ""
                result = transfer.commit(
                    preview["id"],
                    Path(args.destination),
                    args.source_stopped,
                    args.authority,
                )
            finally:
                recovery.store.close()
    elif args.command == "serve":
        os.environ["TAMAGI_CONFIG"] = str(Path(args.config).resolve())
        from .main import main as run

        run()
        return
    else:
        runtime = open_runtime(args.config)
        try:
            if args.command == "qualify":
                result = qualify(runtime)
            elif args.command == "install-supervisor":
                result = install_supervisor(runtime, args.config)
            else:
                result = pilot(runtime, args.operation)
        finally:
            runtime.store.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
