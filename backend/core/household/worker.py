"""Offline resident fixture worker. Never starts main, a model client, or schedulers."""

from __future__ import annotations

import fcntl
import hmac
import os
import sys
from pathlib import Path

from .contracts import MAX_FRAME_BYTES, PROTOCOL_VERSION, ProtocolError, WorkerBinding
from .registry import validate_binding
from .transport import SOURCE_ROOT, decode_frame, encode_frame


def hydrate(binding):
    validate_binding(binding)
    root = Path(binding.root)
    lock_path = root / "instance.lock"
    if lock_path.is_symlink():
        raise ProtocolError("Instance lock cannot be a symlink")
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Imports/construction happen only after read-only identity and lock checks.
        from backend.config import TamAGIConfig, set_config
        from backend.core.runtime import RuntimeService, RuntimeStore, set_runtime

        config = TamAGIConfig()
        config.runtime.enabled = True
        config.runtime.instance_dir = str(root)
        config.runtime.canonical_repo = str(SOURCE_ROOT)
        config.runtime.index_projection = False
        config.runtime.automatic_deployment = False
        config.autonomy.enabled = False
        config.workspace.path = str(root / "workspace")
        config.history.persist_path = str(root / "data/history")
        config.guardrails.allowed_read_paths = [str(root / "workspace")]
        config.guardrails.allowed_write_paths = [str(root / "workspace")]
        # This worker has no completion transport; accidental clients fail locally.
        config.llm.base_url = "http://127.0.0.1:1/offline-household-fixture"
        set_config(config)
        store = RuntimeStore(root / "runtime.sqlite3")
        try:
            runtime = RuntimeService(store, root / "workspace", config.runtime, SOURCE_ROOT)
            set_runtime(runtime)
            for record in store.list("identity", runtime.actor_id):
                name = record["data"].get("name")
                if name in {"IDENTITY.md", "SOUL.md", "USER.md"}:
                    (runtime.workspace / name).write_text(record["data"]["text"])
            os.chdir(root)
            store.event(runtime.actor_id, "household_fixture_hydrated", {"protocol": PROTOCOL_VERSION,
                "household_id": binding.household_id, "release": binding.release})
            return runtime, lock
        except BaseException:
            store.close()
            raise
    except BaseException:
        lock.close()
        raise


def inspect(runtime):
    return {"resident_id": runtime.actor_id, "cwd": os.getcwd(), "workspace": str(runtime.workspace),
        "actor": runtime.store.get("actor", runtime.actor_id)["data"],
        "identities": [record["data"] for record in runtime.store.list("identity", runtime.actor_id)],
        "memories": [record["data"] for record in runtime.store.list("learning", runtime.actor_id)],
        "activity": runtime.store.get("activity", runtime.actor_id)["data"],
        "fixture_only": True, "generation_enabled": False}


def dispatch(runtime, operation, payload):
    if operation == "inspect" and payload == {}:
        return inspect(runtime)
    if operation == "read_identity" and payload == {}:
        return {"text": (runtime.workspace / "IDENTITY.md").read_text(), "resident_id": runtime.actor_id}
    # There is deliberately no arbitrary path, shell, model endpoint, or peer-root API.
    raise ProtocolError("Unsupported fixture worker operation or payload")


def main():
    binding = None
    runtime = lock = None
    request_id = "startup"

    def reply(ok, **body):
        sys.stdout.buffer.write(encode_frame({"protocol": PROTOCOL_VERSION, "request_id": request_id,
            "binding": binding.to_dict() if binding else None, "ok": ok, **body}))
        sys.stdout.buffer.flush()

    try:
        hello = decode_frame(sys.stdin.buffer.readline(MAX_FRAME_BYTES + 1))
        if set(hello) != {"protocol", "binding", "token"} or type(hello["protocol"]) is not int or hello["protocol"] != PROTOCOL_VERSION:
            raise ProtocolError("Unsupported worker handshake")
        binding = WorkerBinding.parse(hello["binding"])
        token = hello["token"]
        if not isinstance(token, str) or not 32 <= len(token) <= 256:
            raise ProtocolError("Invalid worker session credential")
        runtime, lock = hydrate(binding)
        reply(True, data=inspect(runtime))
        while raw := sys.stdin.buffer.readline(MAX_FRAME_BYTES + 1):
            request_id = "invalid"
            try:
                request = decode_frame(raw)
                expected = {"protocol", "binding", "token", "request_id", "operation", "payload"}
                if set(request) != expected:
                    raise ProtocolError("Invalid worker request fields")
                request_id = request["request_id"]
                if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
                    request_id = "invalid"
                    raise ProtocolError("Invalid request identity")
                if type(request["protocol"]) is not int or request["protocol"] != PROTOCOL_VERSION or request["binding"] != binding.to_dict():
                    raise ProtocolError("Request does not match the bound resident or protocol")
                if not isinstance(request["token"], str) or not hmac.compare_digest(request["token"], token):
                    raise ProtocolError("Invalid worker session credential")
                if not isinstance(request["operation"], str) or not isinstance(request["payload"], dict):
                    raise ProtocolError("Invalid operation or payload")
                reply(True, data=dispatch(runtime, request["operation"], request["payload"]))
            except (ValueError, LookupError, PermissionError, RuntimeError) as error:
                reply(False, error=str(error))
    except Exception as error:
        reply(False, error=f"Worker unavailable: {type(error).__name__}: {error}")
        return 1
    finally:
        if runtime:
            runtime.store.close()
        if lock:
            lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
