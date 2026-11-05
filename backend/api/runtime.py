"""Operator inspection, companion surfaces, and staged portability controls."""

from __future__ import annotations
import asyncio
import os
import re
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, SecretStr
from backend.core.runtime import get_runtime
from backend.core.runtime.contracts import RuntimeConflict
from backend.core.runtime.store import uid


async def checkpoint_work(function, *args):
    # A disconnected browser must not release the mutation barrier while a checkpoint thread is still copying.
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def operator(request: Request):
    if not request.session.get("authenticated"):
        raise HTTPException(401, "Operator login required")
    runtime = get_runtime()
    if not runtime:
        raise HTTPException(409, "Provision a platform instance to use this feature")
    return runtime


router = APIRouter(
    prefix="/api", tags=["home platform"], dependencies=[Depends(operator)]
)


@router.get("/inventory")
async def inventory(runtime=Depends(operator)):
    from backend.core.runtime.inventory import Inventory
    try:
        return Inventory(runtime).list()
    except (OSError, ValueError):
        raise HTTPException(409, "Workshop directory unavailable") from None


@router.get("/inventory/file")
async def inventory_file(path: str, preview: bool = False, runtime=Depends(operator)):
    import mimetypes
    from urllib.parse import quote
    from backend.core.runtime.inventory import Inventory

    try:
        file = Inventory(runtime).open(path)
    except (OSError, ValueError):
        raise HTTPException(404, "Workshop file unavailable") from None
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}
    if preview:
        try:
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            safe_images = {"image/png", "image/jpeg", "image/gif", "image/webp"}
            limit = 16 * 1024 * 1024 if mime in safe_images else 256 * 1024
            content = file.read(limit + 1)
            if len(content) > limit:
                raise HTTPException(413, "Use download for this larger file")
            headers["Content-Security-Policy"] = "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'; frame-ancestors 'self'"
            if mime not in safe_images | {"text/html", "image/svg+xml"}:
                try:
                    content.decode("utf-8")
                except UnicodeError:
                    raise HTTPException(415, "Preview unavailable; use download") from None
                mime = "text/plain"
            return Response(content, media_type=mime, headers=headers)
        finally:
            file.close()

    def chunks():
        try:
            while chunk := file.read(65536):
                yield chunk
        finally:
            file.close()
    headers["Content-Disposition"] = "attachment; filename*=UTF-8''" + quote(Path(path).name, safe="")
    return StreamingResponse(chunks(), media_type="application/octet-stream", headers=headers)


class Operation(BaseModel):
    action: str
    id: str = ""
    attributes: dict = Field(default_factory=dict)
    expected: int | None = None


class CompanionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    seed: str = Field(min_length=1, max_length=2000)


class Consultation(BaseModel):
    content: str = Field(min_length=1, max_length=16000)
    mode: str = "conversation"
    thread_id: str | None = None
    allowed_tools: list[str] | None = None
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    background: bool = False


class BackupRequest(BaseModel):
    password: SecretStr


class RestoreCommit(BaseModel):
    source_stopped: bool
    authority: str


class ChangeRequest(BaseModel):
    surface: str
    value: dict | str
    intention: str


class Activation(BaseModel):
    experiment_id: str


class EvaluationRequest(BaseModel):
    proposal_id: str
    scenarios: list[str] | None = None


def bad_request(call):
    try:
        return call()
    except RuntimeConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/runtime")
async def state(runtime=Depends(operator)):
    runtime.deployments.synchronize()
    return {
        "enabled": True,
        "resident_id": runtime.actor_id,
        "installation_id": runtime.store.meta("installation_id"),
        "authority": runtime.authority,
        "activity": {**runtime.store.get("activity", runtime.actor_id), "data": runtime.activity_state()},
        "home": runtime.store.list("world_object"),
        "pursuits": runtime.store.list("pursuit", runtime.actor_id),
        "companions": runtime.companions.list(),
        "notebook": runtime.store.list("shared_note", limit=100),
        "artifacts": runtime.store.list("artifact"),
        "growth": runtime.growth.snapshot(),
        "automatic_deployment": runtime.config.automatic_deployment,
        "resources": {
            "background_calls_per_activation": runtime.config.background_calls_per_activation,
            "daily_background_calls": runtime.config.daily_background_calls,
            "companion_calls": runtime.config.companion_calls,
            "user_calls": runtime.config.user_calls,
            "user_silent_turns": runtime.config.user_silent_turns,
            "daily_evaluation_calls": runtime.config.daily_evaluation_calls,
            "model_context": runtime.store.meta("model_context"),
        },
        "backup_note": "A download kept on another device can restore this life. It is not a machine image.",
    }


@router.get("/runtime/growth")
async def growth(runtime=Depends(operator)):
    """Explain the primary resident's deterministic growth and retained evidence."""
    return runtime.growth.snapshot()


@router.get("/runtime/events")
async def events(after: int = 0, runtime=Depends(operator)):
    return {
        "events": runtime.store.events(after=after, limit=100),
        "receipts": runtime.store.receipts(limit=30),
    }


@router.post("/runtime/world")
async def world(body: Operation, runtime=Depends(operator)):
    return bad_request(
        lambda: runtime.world_action(
            body.action, body.id, body.attributes, body.expected
        )
    )


@router.get("/pursuits")
async def pursuits(runtime=Depends(operator)):
    return {"pursuits": runtime.pursuit("list")}


@router.post("/pursuits")
async def change_pursuit(body: Operation, runtime=Depends(operator)):
    return bad_request(
        lambda: runtime.pursuit(body.action, body.id, body.attributes, body.expected)
    )


@router.get("/companions")
async def companions(runtime=Depends(operator)):
    return {"companions": runtime.companions.list()}


@router.post("/companions")
async def create_companion(body: CompanionCreate, runtime=Depends(operator)):
    return bad_request(lambda: runtime.companions.create(body.name, body.seed))


@router.get("/companions/{actor_id}")
async def inspect_companion(actor_id: str, runtime=Depends(operator)):
    actor = runtime.store.get("actor", actor_id)
    if not actor:
        raise HTTPException(404, "Unknown companion")
    return {
        "actor": actor,
        "personal_memory": runtime.store.list("memory", actor_id),
        "learning": runtime.store.list("learning", actor_id),
        "invocations": runtime.store.list("invocation", actor_id),
    }


@router.post("/companions/{actor_id}/consult")
async def consult(actor_id: str, body: Consultation, runtime=Depends(operator)):
    if getattr(runtime, "interactions", None):
        from backend.core.runtime.interactions import AdmissionError
        try:
            work = await runtime.interactions.submit({"actor_id": actor_id, **body.model_dump(exclude={"request_id", "background"})}, body.request_id, kind="companion")
            return work if body.background else await runtime.interactions.wait(work["work_id"])
        except AdmissionError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
    try:
        return await runtime.companions.consult(
            actor_id, body.content, body.mode, body.thread_id, body.allowed_tools,
            _caller_actor_id="operator",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/runtime/notebook")
async def share(body: dict, runtime=Depends(operator)):
    text = body.get("text")
    if not isinstance(text, str) or not 1 <= len(text) <= 16000:
        raise HTTPException(400, "Write a note of 1–16000 characters")
    record = runtime.store.put(
        "shared_note", uid("note"), "operator", {"text": text, "sources": []}
    )
    runtime.wake("notebook_changed")
    return record


@router.get("/runtime/revisions/{kind}/{record_id:path}")
async def history(kind: str, record_id: str, runtime=Depends(operator)):
    return {"revisions": runtime.store.history(kind, record_id)}


@router.get("/runtime/changes")
async def changes(runtime=Depends(operator)):
    return {
        "proposals": runtime.store.list("proposal"),
        "active": runtime.store.list("runtime_revision"),
        "experiments": runtime.store.list("experiment"),
        "jobs": runtime.store.list("evaluation_job"),
    }


@router.post("/runtime/changes")
async def propose(body: ChangeRequest, runtime=Depends(operator)):
    return bad_request(
        lambda: runtime.adaptation.propose(body.surface, body.value, body.intention)
    )


@router.post("/runtime/changes/{proposal_id}/test-tool")
async def test_tool(proposal_id: str, body: dict, runtime=Depends(operator)):
    try:
        return await runtime.adaptation.test_tool(proposal_id, body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/runtime/changes/{proposal_id}/activate")
async def activate(proposal_id: str, body: Activation, runtime=Depends(operator)):
    return bad_request(
        lambda: runtime.adaptation.activate(proposal_id, body.experiment_id)
    )


@router.post("/runtime/changes/{target}/revert")
async def revert(target: str, body: dict, runtime=Depends(operator)):
    return bad_request(
        lambda: runtime.adaptation.revert(target, int(body.get("revision", 0)))
    )


@router.post("/runtime/evaluate")
async def evaluate(body: EvaluationRequest, runtime=Depends(operator)):
    return bad_request(
        lambda: runtime.submit_evaluation(body.proposal_id, body.scenarios)
    )


@router.get("/deployments")
async def deployments(runtime=Depends(operator)):
    runtime.deployments.synchronize()
    return {
        "deployments": runtime.store.list("deployment"),
        "automatic": runtime.config.automatic_deployment,
        "qualification": runtime.store.meta("deployment_qualification"),
    }


@router.post("/deployments")
async def prepare_deployment(body: dict, runtime=Depends(operator)):
    return bad_request(
        lambda: runtime.deployments.prepare(str(body.get("intention", "")))
    )


@router.post("/deployments/{candidate_id}/build")
async def build(candidate_id: str, runtime=Depends(operator)):
    try:
        return await asyncio.to_thread(runtime.deployments.build, candidate_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/deployments/{candidate_id}/activate")
async def deploy(candidate_id: str, runtime=Depends(operator)):
    # Middleware already holds the activation lock, so no supported background mutation overlaps this checkpoint.
    return bad_request(lambda: runtime.deployments.stage_activation(candidate_id))


@router.post("/backups")
async def backup(body: BackupRequest, runtime=Depends(operator)):
    from backend.config import get_config

    async with runtime.checkpointing():
        try:
            result = await checkpoint_work(
                runtime.backups.export,
                body.password.get_secret_value(),
                get_config().model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    return {
        "id": result["id"],
        "filename": result["filename"],
        "download": f"/api/backups/{result['id']}",
        "excludes": result["manifest"]["excludes"],
        "omissions": result["manifest"]["omissions"],
    }


@router.get("/backups/{backup_id}")
async def download_backup(backup_id: str, runtime=Depends(operator)):
    if not re.fullmatch(r"backup-[a-f0-9]{32}", backup_id):
        raise HTTPException(404, "Unknown backup")
    path = runtime.backups.root / (backup_id + ".tamagi.tar.gz.gpg")
    if not path.is_file():
        raise HTTPException(404, "Backup not found")
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/restores/preview")
async def restore_preview(
    password: str = Form(...),
    archive: UploadFile = File(...),
    runtime=Depends(operator),
):
    fd, name = tempfile.mkstemp(prefix="encrypted-upload-", dir=runtime.backups.root)
    os.chmod(name, 0o600)
    size = 0
    try:
        with os.fdopen(fd, "wb") as output:
            while chunk := await archive.read(1024 * 1024):
                size += len(chunk)
                if size > runtime.backups.max_bytes:
                    raise HTTPException(413, "Archive is too large")
                output.write(chunk)
        try:
            return await asyncio.to_thread(runtime.backups.stage, Path(name), password)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    finally:
        Path(name).unlink(missing_ok=True)
        await archive.close()


@router.post("/restores/{stage_id}/commit")
async def restore_commit(stage_id: str, body: RestoreCommit, runtime=Depends(operator)):
    from backend.core.runtime.deployment import atomic_json

    destination = runtime.source_root / "data/instances" / uid("installation")
    async with runtime.checkpointing():
        try:
            result = await checkpoint_work(
                runtime.backups.commit,
                stage_id,
                destination,
                body.source_stopped,
                body.authority,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        # A configured supervisor notices this pointer and restarts into the restored identity.
        active = runtime.deployments.root / "active.json"
        if active.exists() and runtime.deployments.selected():
            atomic_json(
                active,
                {"instance": result["instance_dir"], "config": result["config_path"]},
            )
            result["activation_required"] = False
            result["status"] = "supervisor_switch_requested"
        else:
            result["status"] = "restored_restart_required"
    return result


@router.get("/runtime/artifacts/{artifact_id}")
async def artifact(artifact_id: str, runtime=Depends(operator)):
    record = runtime.store.get("artifact", artifact_id)
    if not record:
        raise HTTPException(404, "Unknown artifact")
    path = (runtime.workspace / record["data"]["path"]).resolve()
    if not path.is_relative_to(runtime.workspace) or not path.is_file():
        raise HTTPException(404, "Artifact unavailable")
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record["data"]["sha256"]:
        raise HTTPException(
            409, "Saved artifact content no longer matches its registered version"
        )
    return FileResponse(
        path,
        filename=Path(record["data"].get("original_path", path.name)).name,
        media_type="application/octet-stream",
    )


@router.get("/runtime/grants")
async def grants(runtime=Depends(operator)):
    tools = [
        {"name": skill.name, "scope": skill.external_scope}
        for skill in runtime.adaptation.registry._skills.values()
        if getattr(skill, "external_effect", False)
    ]
    return {"tools": tools, "grants": runtime.store.list("external_grant")}


@router.post("/runtime/grants")
async def grant(body: dict, runtime=Depends(operator)):
    scope = body.get("scope")
    known = {
        skill.external_scope
        for skill in runtime.adaptation.registry._skills.values()
        if getattr(skill, "external_effect", False)
    }
    if scope not in known:
        raise HTTPException(400, "Choose a configured external service tool")
    seconds = body.get("seconds", 3600)
    if not isinstance(seconds, int) or not 0 <= seconds <= 30 * 86400:
        raise HTTPException(400, "Choose an allowance up to 30 days")
    record = runtime.store.put(
        "external_grant",
        scope,
        "operator",
        {"enabled": seconds > 0, "expires_at": runtime.store.clock() + seconds},
        category="platform",
    )
    runtime.wake("external_authorization_changed")
    return record


@router.post("/deployments/settings")
async def deployment_settings(body: dict, runtime=Depends(operator)):
    from backend.config import get_config
    import yaml

    qualification = runtime.store.meta("deployment_qualification")
    if body.get("enabled"):
        if not qualification or not qualification.get("passed"):
            raise HTTPException(409, "Run the recovery qualification first")
        if not runtime.verify_host().get("host_verified"):
            raise HTTPException(
                409, "Automatic deployment requires the dedicated host-root profile"
            )
    config = get_config()
    config.runtime.automatic_deployment = bool(body.get("enabled"))
    path = Path(os.environ["TAMAGI_CONFIG"])
    path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
    path.chmod(0o600)
    if config.runtime.automatic_deployment:
        runtime.wake("automatic_deployment_enabled")
    return {"automatic_deployment": config.runtime.automatic_deployment}


@router.get("/operator-archives")
async def predecessor_archives(runtime=Depends(operator)):
    import json

    root = runtime.source_root / ".platform/operator-archives"
    archives = []
    if root.exists():
        for path in root.glob("predecessor-*/manifest.json"):
            manifest = json.loads(path.read_text())
            archives.append(
                {
                    "id": path.parent.name,
                    "created_at": manifest.get("created_at"),
                    "complete": manifest.get("complete", False),
                    "memory": manifest.get("memory", {}),
                }
            )
    return {
        "archives": archives,
        "access": "operator only; excluded from resident hydration and retrieval",
    }


@router.post("/operator-archives/{archive_id}/export")
async def predecessor_export(
    archive_id: str, body: BackupRequest, runtime=Depends(operator)
):
    import tarfile

    if not re.fullmatch(r"predecessor-[a-f0-9]{32}", archive_id):
        raise HTTPException(404, "Unknown archive")
    source = runtime.source_root / ".platform/operator-archives" / archive_id
    if not (source / "manifest.json").is_file():
        raise HTTPException(404, "Unknown archive")
    backup_id = uid("backup")
    destination = runtime.backups.root / (backup_id + ".tamagi.tar.gz.gpg")

    def pack(output):
        with tarfile.open(fileobj=output, mode="w|gz") as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.add(
                        path, arcname=str(path.relative_to(source)), recursive=False
                    )

    try:
        await asyncio.to_thread(
            runtime.backups._crypt, pack, destination, body.password.get_secret_value()
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "id": backup_id,
        "download": f"/api/backups/{backup_id}",
        "filename": archive_id + ".tar.gz.gpg",
        "kind": "operator_predecessor_archive",
        "restore": "Use the archived source revision and legacy files; do not import this as the new resident.",
    }


@router.post("/runtime/memory/reindex")
async def reindex(runtime=Depends(operator)):
    from backend.api.chat import get_agent

    memory = get_agent().memory
    if not hasattr(memory, "schedule_rebuild"):
        raise HTTPException(409, "Canonical memory is not active")
    memory.schedule_rebuild()
    return {
        "status": "queued" if memory.projection else "not_configured",
        "source_records_available": True,
    }
