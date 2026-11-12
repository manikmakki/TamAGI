"""Instance configuration and deterministic hydration; never imports a predecessor."""

from pathlib import Path
import os
from . import RuntimeStore, RuntimeService, set_runtime


def configure_instance(config):
    if not config.runtime.enabled:
        return config
    root = Path(config.runtime.instance_dir).resolve()
    if (
        not (root / "instance.json").is_file()
        and not (root / "runtime.sqlite3").is_file()
    ):
        raise ValueError(
            "Provision a fresh instance or stage a restore before enabling the platform"
        )
    config.runtime.instance_dir = str(root)
    config.workspace.path = str(root / "workspace")
    if config.runtime.authority == "guarded":
        config.guardrails.allowed_read_paths = [str(root / "workspace")]
        config.guardrails.allowed_write_paths = [str(root / "workspace")]
    config.history.persist_path = str(root / "data/history")
    config.self_model.data_path = str(root / "data/self_model.json")
    config.world_thread.thread_path = str(root / "data/world_thread.json")
    config.world_thread.state_path = str(root / "data/world_state.json")
    config.world_thread.consolidation.state_path = str(
        root / "data/consolidation_state.json"
    )
    config.memory.chromadb.persist_directory = str(root / "data/chromadb")
    # A new identity must never query the predecessor's embedding namespace.
    suffix = root.name.replace("-", "_")
    config.memory.chromadb.collection_name = "tamagi_" + suffix
    config.memory.elasticsearch.index = "tamagi_" + suffix
    config.auth.enabled = True
    return config


def initialize_runtime(config, source_root):
    root = Path(config.runtime.instance_dir)
    (root / "data").mkdir(exist_ok=True)
    import fcntl

    lock = (root / "instance.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("This installation already has an active process")
    try:
        runtime = RuntimeService(
            RuntimeStore(root / "runtime.sqlite3"),
            root / "workspace",
            config.runtime,
            config.runtime.canonical_repo or source_root,
        )
        runtime.verify_host()
    except BaseException:
        lock.close()
        raise
    runtime.instance_lock = lock
    runtime.release_source_root = Path(source_root).resolve()
    set_runtime(runtime)
    for record in runtime.store.list("identity", runtime.actor_id):
        if record["data"].get("name") in {"IDENTITY.md", "SOUL.md", "USER.md"}:
            (runtime.workspace / record["data"]["name"]).write_text(
                record["data"]["text"]
            )
    graph = runtime.store.get("graph", "world")
    if graph:
        import json

        Path(config.self_model.data_path).write_text(json.dumps(graph["data"]))
    projection = runtime.store.get("world_projection", "current")
    if projection:
        import json

        Path(config.world_thread.state_path).write_text(json.dumps(projection["data"]))
    runtime.store.set_meta("last_hydrated_at", runtime.store.clock())
    runtime.store.event(
        runtime.actor_id,
        "hydration_completed",
        {
            "method": "canonical source recovery",
            "unfinished_pursuits": [
                r["id"]
                for r in runtime.store.list("pursuit")
                if r["data"]["status"] in {"active", "paused", "blocked"}
            ],
            "unknown_receipts": [
                r["id"]
                for r in runtime.store.receipts()
                if r["status"] == "outcome_unknown"
            ],
        },
    )
    # Legacy relative file adapters now project into this instance only.
    os.environ["TAMAGI_CONFIG"] = str(
        Path(os.environ.get("TAMAGI_CONFIG", root / "config.yaml")).resolve()
    )
    os.chdir(root)
    return runtime
