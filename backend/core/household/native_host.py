"""Local native resident hosting; one unchanged application context per process."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import secrets
import sys

import httpx

from .contracts import WorkerBinding

SOURCE = Path(__file__).resolve().parents[3]


class NativeHost:
    def __init__(self, registry, descriptor, coordinator_socket, *, lost=None):
        self.registry, self.descriptor = registry, descriptor
        self.actor_id = descriptor["resident_id"]
        index = list(registry.descriptors).index(self.actor_id)
        self.socket = registry.root / "ipc" / f"r{index}.sock"
        self.coordinator_socket = coordinator_socket
        self.token = secrets.token_urlsafe(32)
        self.process = None
        self.client = None
        self.log_tail = b""
        self._drainer = None
        self._watcher = None
        self.lost = lost
        self.closing = False
        self.ready = None

    async def start(self):
        self.closing = False
        self.registry.validate(self.descriptor)
        if len(os.fsencode(self.socket)) > 100:
            raise ValueError("Household IPC path is too long; choose a shorter data root")
        if self.socket.exists():
            if self.socket.is_symlink() or not self.socket.is_socket():
                raise ValueError("Worker socket path contains unexpected state")
            self.socket.unlink()
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TZ") if key in os.environ}
        binding = WorkerBinding.parse({key: self.descriptor[key] for key in WorkerBinding.__dataclass_fields__})
        env.update({"PYTHONPATH": str(SOURCE), "PYTHONDONTWRITEBYTECODE": "1",
            "TAMAGI_CONFIG": self.descriptor["config_path"], "TAMAGI_NATIVE_BINDING": json.dumps(binding.to_dict()),
            "TAMAGI_NATIVE_TOKEN": self.token, "TAMAGI_NATIVE_COORDINATOR_SOCKET": str(self.coordinator_socket)})
        guard = os.environ.get("TAMAGI_QUALIFICATION_GUARD")
        if guard:
            env["TAMAGI_QUALIFICATION_GUARD"] = guard
            env["PYTHONPATH"] = str(Path(guard).parent) + os.pathsep + str(SOURCE)
        self.process = await asyncio.create_subprocess_exec(sys.executable, "-m", "backend.core.household.native_worker",
            "--config", self.descriptor["config_path"], "--socket", str(self.socket), env=env, cwd=SOURCE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        self._drainer = asyncio.create_task(self._drain())
        self.client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(self.socket)),
                                       base_url="http://native-worker", headers={"X-Tamagi-Worker": self.token}, timeout=None)
        try:
            for _ in range(300):
                if self.process.returncode is not None:
                    raise RuntimeError("Resident worker exited during startup: " + self.log_tail.decode(errors="replace")[-1600:])
                try:
                    response = await asyncio.wait_for(self.client.get("/_native/ready"), 1)
                    if response.status_code == 200:
                        self.ready = response.json()
                        if self.ready.get("resident_id") != self.actor_id:
                            raise RuntimeError("Worker hydrated a different resident")
                        self._watcher = asyncio.create_task(self._watch())
                        return self.ready
                except (httpx.TransportError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(0.05)
            raise RuntimeError("Resident worker did not become ready")
        except BaseException:
            await self.close()
            raise

    async def _drain(self):
        directory = self.registry.root / "logs"
        directory.mkdir(exist_ok=True, mode=0o700)
        path = directory / (self.actor_id + ".log")
        with path.open("ab") as stream:
            path.chmod(0o600)
            while data := await self.process.stdout.read(8192):
                self.log_tail = (self.log_tail + data)[-65536:]
                stream.write(data)
                stream.flush()

    async def _watch(self):
        await self.process.wait()
        if not self.closing and self.lost:
            await self.lost(self.actor_id)

    async def request(self, method, path, **kwargs):
        if self.process is None or self.process.returncode is not None:
            raise RuntimeError("Resident worker is unavailable")
        return await self.client.request(method, path, **kwargs)

    async def json(self, method, path, body=None):
        response = await self.request(method, path, json=body, timeout=10) if body is not None else await self.request(method, path, timeout=10)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text[:500])
            except ValueError:
                detail = response.text[:500]
            raise RuntimeError(f"Worker rejected request ({response.status_code}): {detail}")
        return response.json()

    async def close(self):
        self.closing = True
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 15)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.client:
            await self.client.aclose()
        if self._drainer:
            await self._drainer
        if self._watcher and self._watcher is not asyncio.current_task():
            await self._watcher
