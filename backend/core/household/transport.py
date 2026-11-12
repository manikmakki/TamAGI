"""Bounded, versioned parent/worker IPC over private inherited stdio pipes."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
from pathlib import Path

from .contracts import MAX_FRAME_BYTES, PROTOCOL_VERSION, ProtocolError, WorkerBinding
from .registry import validate_binding
from .store import encoded, uid

SOURCE_ROOT = Path(__file__).resolve().parents[3]


def decode_frame(raw: bytes) -> dict:
    if len(raw) > MAX_FRAME_BYTES or not raw.endswith(b"\n"):
        raise ProtocolError("IPC frame is oversized or incomplete")
    try:
        value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ProtocolError("Malformed IPC JSON") from error
    if not isinstance(value, dict):
        raise ProtocolError("IPC frame must be an object")
    return value


def encode_frame(value: dict) -> bytes:
    frame = (encoded(value) + "\n").encode()
    if len(frame) > MAX_FRAME_BYTES:
        raise ProtocolError("IPC frame is oversized")
    return frame


def worker_environment(root: Path) -> dict[str, str]:
    """Positive allowlist: no live config, provider secrets, proxy, or index overrides."""
    result = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT") if key in os.environ}
    result.update({"PYTHONPATH": str(SOURCE_ROOT), "PYTHONDONTWRITEBYTECODE": "1",
        "TAMAGI_CONFIG": str(root / "offline-worker-config.yaml"), "TAMAGI_AUTONOMY_ENABLED": "false"})
    # Preserve an offline qualification guard only through its explicit run manifest.
    guard = os.environ.get("TAMAGI_QUALIFICATION_GUARD")
    if guard:
        result["TAMAGI_QUALIFICATION_GUARD"] = guard
        result["PYTHONPATH"] = str(Path(guard).parent) + os.pathsep + str(SOURCE_ROOT)
    return result


class ResidentProcess:
    """One fixture runtime per process; disconnect never replays a mutation."""

    def __init__(self, binding: WorkerBinding, *, timeout: float = 10):
        self.binding = binding
        self.timeout = timeout
        self.token = secrets.token_urlsafe(32)
        self.process: asyncio.subprocess.Process | None = None
        self._gate = asyncio.Lock()
        self._stderr_task = None
        self.stderr = b""
        self._failed = False
        self.on_unavailable = None

    @property
    def failed(self):
        return self._failed

    @failed.setter
    def failed(self, value):
        changed = bool(value) and not self._failed
        self._failed = bool(value)
        if changed and self.on_unavailable:
            self.on_unavailable()

    async def start(self):
        if self.process is not None:
            raise RuntimeError("Worker already started")
        validate_binding(self.binding)
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "backend.core.household.worker",
            cwd=SOURCE_ROOT, env=worker_environment(Path(self.binding.root)),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=MAX_FRAME_BYTES + 1,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            self.process.stdin.write(encode_frame({"protocol": PROTOCOL_VERSION,
                "binding": self.binding.to_dict(), "token": self.token}))
            await self.process.stdin.drain()
            response = await self._receive("startup")
            if response["data"]["resident_id"] != self.binding.resident_id:
                raise ProtocolError("Hydrated resident mismatch")
            return response["data"]
        except BaseException:
            await self.close()
            raise

    async def _drain_stderr(self):
        while chunk := await self.process.stderr.read(4096):
            self.stderr = (self.stderr + chunk)[-32768:]

    async def _receive(self, request_id):
        try:
            raw = await asyncio.wait_for(self.process.stdout.readline(), self.timeout)
        except (ValueError, asyncio.TimeoutError) as error:
            self.failed = True
            raise ProtocolError("Worker response timed out or exceeded the frame limit; outcome is not replayed") from error
        if not raw:
            self.failed = True
            raise ProtocolError("Worker exited before acknowledging; outcome may be unknown")
        try:
            response = decode_frame(raw)
        except ProtocolError:
            self.failed = True
            raise
        if type(response.get("protocol")) is not int or response.get("protocol") != PROTOCOL_VERSION or response.get("request_id") != request_id:
            self.failed = True
            raise ProtocolError("Worker response protocol/request mismatch")
        if response.get("binding") != self.binding.to_dict():
            self.failed = True
            raise ProtocolError("Worker response identity mismatch")
        if type(response.get("ok")) is not bool:
            self.failed = True
            raise ProtocolError("Worker response lacks an outcome")
        if set(response) != {"protocol", "request_id", "binding", "ok", "data" if response["ok"] else "error"}:
            self.failed = True
            raise ProtocolError("Worker response has invalid fields")
        if (response["ok"] and not isinstance(response["data"], dict)) or (not response["ok"] and not isinstance(response["error"], str)):
            self.failed = True
            raise ProtocolError("Worker response has an invalid payload")
        if not response["ok"]:
            raise ProtocolError(response.get("error", "Worker rejected request"))
        return response

    async def request(self, operation: str, payload: dict | None = None, *, request_id: str | None = None):
        if self.process is None or self.process.returncode is not None or self.failed:
            raise ProtocolError("Worker is unavailable; reconcile before resubmitting effects")
        request_id = request_id or uid("ipc")
        async with self._gate:
            if self.process.returncode is not None or self.failed:
                raise ProtocolError("Worker is unavailable; reconcile before resubmitting effects")
            self.process.stdin.write(encode_frame({"protocol": PROTOCOL_VERSION,
                "binding": self.binding.to_dict(), "token": self.token, "request_id": request_id,
                "operation": operation, "payload": {} if payload is None else payload}))
            try:
                await self.process.stdin.drain()
                return (await self._receive(request_id))["data"]
            except asyncio.CancelledError:
                self.failed = True
                raise
            except (BrokenPipeError, ConnectionResetError) as error:
                self.failed = True
                raise ProtocolError("Worker connection lost; outcome is not automatically replayed") from error

    async def close(self):
        if self.process is None:
            return
        if self.process.stdin:
            self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), 2)
        except asyncio.TimeoutError:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 2)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self._stderr_task:
            await self._stderr_task

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.close()
