"""Operator browser recovery with an idle household maintenance boundary."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
import fcntl
import json
from pathlib import Path
import re
import secrets
import shutil
import tempfile
import time

import anyio
from fastapi import HTTPException
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

from backend.core.runtime.backup import ArchiveError
from .resident_recovery import JOURNAL, ResidentArchive, atomic_json, backup_resident, reconcile_transition, restore_resident
from .native_service import TERMINAL
from .store import uid


class BrowserRecovery:
    MAX_UPLOAD = 1024**3
    EXPIRY_INTERVAL = 60

    def __init__(self, registry, manager, model, hosts, *, scheduler=None):
        self.registry, self.manager, self.model, self.hosts = registry, manager, model, hosts
        # A known, owned directory lets the next coordinator remove plaintext
        # previews after a crash. The registry already holds the household lock.
        self.root = registry.root.parent / ("." + registry.root.name + "-browser-recovery")
        owner = {"household_id": registry.household_id, "root": str(registry.root)}
        if self.root != self.root.resolve() or self.root.is_symlink():
            raise ArchiveError("Browser recovery directory must be canonical")
        if self.root.exists():
            marker = self.root / "owner.json"
            if marker.is_symlink() or not marker.is_file() or json.loads(marker.read_text()) != owner:
                raise ArchiveError("Browser recovery directory belongs to another installation")
            shutil.rmtree(self.root)
        self.root.mkdir(mode=0o700)
        atomic_json(self.root / "owner.json", owner)
        self.previews, self.downloads = {}, {}
        self.preview_tasks = set()
        self.task = None
        self.inflight = 0
        self.phase, self.actor, self.resume_errors = "idle", None, []
        self.closing = False
        self.blocked = False
        self.last_operation = None
        self.schedule_lock = scheduler.lock if scheduler is not None else asyncio.Lock()
        self.expiry_task = None

    def start(self):
        self.expiry_task = asyncio.create_task(self._expire_loop())

    async def _expire_loop(self):
        while True:
            await asyncio.sleep(self.EXPIRY_INTERVAL)
            self.expire()

    @staticmethod
    def handles(path):
        return path == "api/recovery/status" or path == "api/recovery/resume" or path.startswith(("api/backups", "api/restores"))

    @property
    def busy(self):
        return self.task is not None and not self.task.done()

    @property
    def paused(self):
        return self.busy or self.blocked

    def status(self):
        return {"busy": self.busy, "phase": self.phase, "resident_id": self.actor,
                "unavailable_residents": self.resume_errors, "last_operation": self.last_operation}

    def expire(self):
        if self.busy:
            return
        now = time.time()
        for identity, item in list(self.previews.items()):
            if now - item["created"] > 1800:
                item["archive"].cleanup()
                shutil.rmtree(item["root"], ignore_errors=True)
                del self.previews[identity]
        for identity, item in list(self.downloads.items()):
            if now - item["created"] > 86400:
                item["path"].unlink(missing_ok=True)
                del self.downloads[identity]

    async def run(self, actor, action, work):
        if self.busy or self.closing:
            raise HTTPException(409, "Another recovery operation is already in progress")
        if self.blocked and action != "resuming":
            raise HTTPException(409, "Finish the pending recovery with Retry resuming residents first")
        self.actor = actor
        self.task = asyncio.create_task(self._maintain(work, action))
        # A disconnect must not abandon a stopped worker or half-selected root.
        return await asyncio.shield(self.task)

    async def _maintain(self, work, action):
        self.phase = "checking_idle"
        try:
            await asyncio.wait_for(self.schedule_lock.acquire(), 10)
        except TimeoutError as error:
            self.phase = "resume_required" if self.resume_errors else "idle"
            raise HTTPException(409, "The World Thread scheduler is still finishing. Try recovery again shortly") from error
        try:
            return await self._checkpoint(work, action)
        finally:
            self.schedule_lock.release()

    async def _checkpoint(self, work, action):
        self.manager.maintenance = True
        stopped = False
        installation = self.registry.descriptors[self.actor]["installation_id"]
        self.phase = "checking_idle"
        try:
            self.check_idle_work()
            async with asyncio.timeout(10):
                while self.inflight:
                    await asyncio.sleep(.02)
            self.check_idle_work()
            for host in self.hosts.values():
                if host.process and host.process.returncode is None:
                    ready = await host.json("GET", "/_native/ready")
                    if ready.get("recovery_pending", ready.get("pending")):
                        raise HTTPException(409, "A resident has active private work. Try again after it finishes")
            self.model.maintenance = True
            self.phase = "pausing"
            stopped = True
            for host in self.hosts.values():
                await host.close()
            self.phase = action
            with self._worker_locks():
                result = await asyncio.to_thread(work)
        except TimeoutError as error:
            raise HTTPException(409, "Other requests are still finishing. Try recovery again shortly") from error
        finally:
            if stopped:
                self.phase = "resuming"
                self.resume_errors = []
                self.blocked = False
                if (self.registry.root / JOURNAL).exists():
                    try:
                        await asyncio.to_thread(reconcile_transition, self.registry)
                    except Exception:
                        self.blocked = True
                        self.resume_errors = list(self.hosts)
                if not self.blocked and getattr(self, "creation", None) is not None:
                    try:
                        await asyncio.to_thread(self.creation.reconcile)
                    except Exception:
                        self.blocked = True
                        self.resume_errors = list(self.hosts)
                if not self.blocked:
                    for actor, host in self.hosts.items():
                        host.descriptor = self.registry.descriptors[actor]
                        host.token = secrets.token_urlsafe(32)
                        try:
                            ready = await host.start()
                            host.ready = ready
                        except Exception:
                            self.resume_errors.append(actor)
            self.manager.maintenance = self.model.maintenance = self.blocked
            if stopped:
                for actor, host in self.hosts.items():
                    if host.process and host.process.returncode is None and host.ready:
                        self.manager.refresh_availability(actor, host.ready)
                    else:
                        self.model.broker.set_available(actor, False)
            self.phase = "resume_required" if self.resume_errors else "idle"
            self.last_operation = {"action": action, "resident_id": self.actor,
                "resident_replaced": installation != self.registry.descriptors[self.actor]["installation_id"],
                "workers_available": not self.resume_errors}
        return {**result, "workers_available": not self.resume_errors,
                "unavailable_residents": list(self.resume_errors)}

    def check_idle_work(self):
        if any(r["data"]["status"] not in TERMINAL for r in self.manager.records()):
            raise HTTPException(409, "A resident is busy. Wait for conversations and World Thread work to finish, then try again")
        if any(r["data"]["status"] not in {"completed", "failed", "cancelled", "stopped"}
               for r in self.registry.store.list("broker_attempt")):
            raise HTTPException(409, "Model work is pending or its outcome is unknown. Resolve it before recovery")

    @contextmanager
    def _worker_locks(self):
        with ExitStack() as stack:
            for actor, descriptor in self.registry.descriptors.items():
                root = Path(descriptor["root"])
                paths = [self.registry.root / "ipc" / (actor + ".lock")]
                if root.is_dir():
                    paths.append(root / "instance.lock")
                for path in paths:
                    if path != path.resolve() or path.is_symlink():
                        raise ArchiveError("Worker lock cannot be a symbolic link")
                    stream = stack.enter_context(path.open("a"))
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise ArchiveError("A resident worker did not stop; recovery was not applied") from error
            yield

    async def route(self, actor, path, request):
        self.expire()
        try:
            if path == "api/recovery/status" and request.method == "GET":
                return self.status()
            if path == "api/recovery/resume" and request.method == "POST":
                return await self.run(actor, "resuming", lambda: {"status": "resumed"})
            if path == "api/backups" and request.method == "POST":
                body = await request.json()
                if not isinstance(body, dict):
                    raise HTTPException(400, "Choose a backup password")
                password = body.get("password")
                ResidentArchive._password(password)
                identity = uid("backup")
                target = self.root / (identity + ".tamagi.tar.gz.gpg")
                def export():
                    result = backup_resident(self.registry, actor, target, password)
                    # Retain expiry bookkeeping even if the browser loses the
                    # acknowledgment while the independent operation finishes.
                    self.downloads[identity] = {"path": target, "actor": actor, "created": time.time()}
                    return result
                result = await self.run(actor, "backing_up", export)
                return {"id": identity, "resident_id": actor, "name": result["name"],
                    "filename": target.name, "download": "/api/backups/" + identity,
                    "exported_at": result["exported_at"], "omissions": result["omissions"],
                    "workers_available": result["workers_available"]}
            match = re.fullmatch(r"api/backups/(backup-[a-f0-9]{32})", path)
            if match and request.method == "GET":
                item = self.downloads.get(match[1])
                if not item or not item["path"].is_file():
                    raise HTTPException(404, "Download expired; create another checkpoint")
                if item["actor"] != actor:
                    raise HTTPException(403, "This backup belongs to a different resident")
                return FileResponse(item["path"], filename=item["path"].name, media_type="application/octet-stream",
                                    headers={"Cache-Control": "no-store"})
            if path == "api/restores/preview" and request.method == "POST":
                return await self.preview(actor, request)
            match = re.fullmatch(r"api/restores/(restore-[a-f0-9]{32})(/commit)?", path)
            if match and request.method in {"POST", "DELETE"}:
                item = self.previews.get(match[1])
                if not item:
                    raise HTTPException(404, "Restore preview expired; upload the archive again")
                if item["actor"] != actor:
                    raise HTTPException(403, "Preview belongs to a different resident")
                if request.method == "DELETE" and not match[2]:
                    if self.busy:
                        raise HTTPException(409, "A confirmed operation is in progress; it cannot be cancelled by closing its preview")
                    self.discard(match[1])
                    return {"status": "discarded"}
                if request.method == "POST" and match[2]:
                    body = await request.json()
                    if not isinstance(body, dict):
                        raise HTTPException(400, "Review and confirm the preview")
                    if body.get("confirmed") is not True or body.get("source_stopped") is not True:
                        raise HTTPException(400, "Confirm replacement and that no other copy of this resident is active")
                    if item["installation_id"] != self.registry.descriptors[actor]["installation_id"]:
                        raise HTTPException(409, "The resident changed since preview; upload and review the archive again")
                    def commit():
                        return restore_resident(self.registry, item["archive"], item["preview"], actor,
                            confirmed=True, source_stopped=True, authority=body.get("authority"))
                    result = await self.run(actor, "restoring", commit)
                    self.discard(match[1])
                    return {"status": "restored", "resident_id": actor, "shared_state": "retained",
                            "workers_available": result["workers_available"], "credentials_required": True}
            raise HTTPException(404, "Unknown recovery operation")
        except ArchiveError as error:
            raise HTTPException(400, str(error)) from error
        except OSError as error:
            if self.last_operation:
                self.last_operation["error"] = True
            raise HTTPException(507, "Recovery could not finish writing files. Check recovery status and available disk space before retrying") from error

    async def preview(self, actor, request):
        if self.busy or self.closing:
            raise HTTPException(409, "Recovery is in progress; wait before uploading another archive")
        installation_id = self.registry.descriptors[actor]["installation_id"]
        if int(request.headers.get("content-length", 0)) > self.MAX_UPLOAD + 16384:
            raise HTTPException(413, "Archive exceeds the browser upload limit")
        directory = Path(tempfile.mkdtemp(prefix="preview-", dir=self.root))
        archive = ResidentArchive(directory)
        uploaded = directory / "upload.gpg"
        try:
            form = await request.form(max_files=1, max_fields=2, max_part_size=2048)
            # Close the multipart spool before decrypting, including when the
            # request's AnyIO cancellation scope has already been cancelled.
            with anyio.CancelScope(shield=True):
                try:
                    file, password = form.get("archive"), form.get("password")
                    if not isinstance(file, UploadFile):
                        raise HTTPException(400, "Choose an encrypted resident archive")
                    size = 0
                    with uploaded.open("xb") as stream:
                        while data := await file.read(1024 * 1024):
                            size += len(data)
                            if size > self.MAX_UPLOAD:
                                raise HTTPException(413, "Archive exceeds the browser upload limit")
                            stream.write(data)
                finally:
                    await form.close()
            task = asyncio.create_task(asyncio.to_thread(archive.stage, uploaded, password))
            self.preview_tasks.add(task)
            try:
                try:
                    preview = await asyncio.shield(task)
                except asyncio.CancelledError:
                    with anyio.CancelScope(shield=True):
                        await asyncio.gather(task, return_exceptions=True)
                    raise
            finally:
                self.preview_tasks.discard(task)
            if self.closing or installation_id != self.registry.descriptors[actor]["installation_id"]:
                raise HTTPException(409, "Resident changed during preview; upload and review the archive again")
            if preview["resident_id"] != actor:
                raise ArchiveError("This checkpoint belongs to another resident. Select that resident before uploading")
            if preview.get("release") and preview["release"] != self.registry.release:
                raise ArchiveError("This checkpoint needs a different application release; use qualified offline recovery")
            self.previews[preview["id"]] = {"actor": actor, "preview": preview, "archive": archive,
                "root": directory, "created": time.time(),
                "installation_id": installation_id}
            return {**preview, "requires": ["checkpoint replacement confirmation", "other copies of this resident stopped"],
                    "household_pause": "All workers pause briefly once idle; other residents' state and shared records are retained"}
        except BaseException:
            archive.cleanup()
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            uploaded.unlink(missing_ok=True)

    def discard(self, identity):
        item = self.previews.pop(identity, None)
        if item:
            item["archive"].cleanup()
            shutil.rmtree(item["root"], ignore_errors=True)

    async def close(self):
        self.closing = True
        if self.expiry_task:
            self.expiry_task.cancel()
            await asyncio.gather(self.expiry_task, return_exceptions=True)
        if self.task:
            await asyncio.gather(self.task, return_exceptions=True)
        await asyncio.gather(*self.preview_tasks, return_exceptions=True)
        for identity in list(self.previews):
            self.discard(identity)
        shutil.rmtree(self.root, ignore_errors=True)
