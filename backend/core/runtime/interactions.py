"""Application-owned interactions. Connections observe work; they never own it."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import json
import os
import tempfile

from .contracts import current_terminal
from .store import encoded, uid

TERMINAL = {
    "completed",
    "resting",
    "waiting",
    "blocked",
    "declined",
    "interrupted",
    "failed",
    "budget_exhausted",
}


class AdmissionError(ValueError):
    def __init__(self, detail, status=400):
        super().__init__(detail)
        self.status = status


class Interactions:
    QUEUE_LIMIT = 8
    EVENT_LIMIT = 256

    def __init__(self, runtime, agent, resume_minutes=15):
        self.runtime = runtime
        self.store = runtime.store
        self.agent = agent
        self.resume_seconds = resume_minutes * 60
        self.admission = asyncio.Lock()
        self.queue = asyncio.Queue(maxsize=self.QUEUE_LIMIT)
        self.tasks = {}
        self.observers = set()
        self.worker = None
        self.presence = None
        self.closing = False
        for record in self.store.list("interaction", runtime.actor_id, limit=100000):
            if record["data"]["status"] in {"queued", "running"}:
                self._update(
                    record["id"],
                    status="interrupted",
                    detail="Application restarted; inspect retained effects before continuing.",
                )
                self.emit(record["id"], {"type": "work_ended", "status": "interrupted"})

    def start(self):
        self.worker = asyncio.create_task(self._worker(), context=contextvars.Context())
        self.presence = asyncio.create_task(
            self._presence(), context=contextvars.Context()
        )

    async def close(self):
        self.closing = True
        tasks = [t for t in [self.worker, self.presence, *self.tasks.values()] if t]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for record in self.records():
            if record["data"]["status"] in {"queued", "running"}:
                self._update(
                    record["id"],
                    status="interrupted",
                    detail="Application stopped; work was not replayed.",
                )

    def records(self, limit=100000):
        with self.store.lock:
            rows = self.store.db.execute(
                "SELECT * FROM records WHERE kind='interaction' AND actor=? ORDER BY created DESC LIMIT ?",
                (self.runtime.actor_id, limit),
            ).fetchall()
            return [self.store._record(r) for r in rows]

    def has_pending(self, exclude_work_id=None):
        with self.store.lock:
            return bool(
                self.store.db.execute(
                    "SELECT 1 FROM records WHERE kind='interaction' AND actor=? AND id!=? AND json_extract(payload,'$.status') IN ('queued','running') LIMIT 1",
                    (self.runtime.actor_id, exclude_work_id or ""),
                ).fetchone()
            )

    def get(self, work_id):
        record = self.store.get("interaction", work_id)
        if not record or record["actor"] != self.runtime.actor_id:
            raise AdmissionError("Unknown request", 404)
        return record

    def public(self, record):
        data = record["data"]
        source = self.store.get("interaction_input", record["id"])
        return {
            "work_id": record["id"],
            "created_at": record["created"],
            **{
                k: v
                for k, v in data.items()
                if k not in {"payload", "fingerprint", "input_epoch"}
            },
            "input_message": source["data"]["message"]
            if source
            else data.get("input_message"),
        }

    def _image_reference(self, content):
        digest = hashlib.sha256(content).hexdigest()
        directory = self.runtime.workspace / ".attachments"
        directory.mkdir(exist_ok=True)
        destination = directory / (digest + ".image")
        if not destination.exists():
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                name = temporary.name
            os.replace(name, destination)
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return {
            "path": str(destination.relative_to(self.runtime.workspace)),
            "sha256": digest,
            "size": len(content),
        }

    def _update(self, work_id, **changes):
        record = self.get(work_id)
        return self.store.put(
            "interaction",
            work_id,
            record["actor"],
            {**record["data"], **changes},
            category="platform",
        )

    async def submit(self, payload, client_id=None, *, kind="chat"):
        if getattr(self.runtime, "household", None) is not None and kind == "chat":
            raise AdmissionError("Shared chat requires coordinator admission", 403)
        client_id = client_id or uid("client")
        if not isinstance(client_id, str) or not 1 <= len(client_id) <= 128:
            raise AdmissionError("A submission identifier must have 1–128 characters")
        payload = dict(payload)
        fingerprint = hashlib.sha256(
            encoded({"kind": kind, "payload": payload}).encode()
        ).hexdigest()
        identity = [
            self.store.meta("installation_id"),
            self.runtime.actor_id,
            client_id,
        ]
        work_id = "request-" + hashlib.sha256(encoded(identity).encode()).hexdigest()
        async with self.admission:
            prior = self.store.get("interaction", work_id)
            if prior:
                if prior["data"]["fingerprint"] != fingerprint:
                    raise AdmissionError(
                        "Submission identifier already belongs to different input", 409
                    )
                return self.public(prior)
            if self.closing or self.runtime.maintenance:
                raise AdmissionError(
                    "Application is preparing a checkpoint or release", 503
                )
            if self.queue.full():
                raise AdmissionError(
                    "The pending work queue is full; this input was not accepted", 429
                )
            content = payload.get("message" if kind == "chat" else "content", "")
            if (
                not isinstance(content, str)
                or not content.strip()
                or len(content) > 32000
            ):
                raise AdmissionError("Enter a message of 1–32000 characters")
            image = payload.get("image_data")
            image_bytes = None
            if image:
                try:
                    image_bytes = (
                        base64.b64decode(image, validate=True)
                        if isinstance(image, str) and len(image) <= 12 * 1024 * 1024
                        else None
                    )
                    if image_bytes is None or len(image_bytes) > 8 * 1024 * 1024:
                        raise ValueError()
                except (ValueError, TypeError):
                    raise AdmissionError(
                        "Image data must be valid base64 and at most 8 MiB"
                    ) from None
            if kind == "chat":
                conversation_id = payload.get("conversation_id")
                conv = (
                    self.agent.get_conversation(conversation_id)
                    if conversation_id
                    else None
                )
                if conversation_id and not conv:
                    raise AdmissionError("Unknown conversation", 404)
                if not conv:
                    conv = self.agent.create_conversation(content[:60])
                conversation_id = conv.id
                payload["conversation_id"] = conversation_id
            elif kind == "companion":
                actor = self.store.get("actor", payload.get("actor_id", ""))
                if (
                    not actor
                    or actor["data"].get("primary")
                    or actor["data"].get("status") != "active"
                ):
                    raise AdmissionError("Companion is unavailable", 404)
                if payload.get("mode", "conversation") not in {"conversation", "task"}:
                    raise AdmissionError("Choose conversation or task")
                conversation_id = payload.get("thread_id") or uid("thread")
                thread = self.store.get("companion_thread", conversation_id)
                if thread and thread["data"]["participants"] not in (
                    ["operator", actor["id"]],
                    [self.runtime.actor_id, actor["id"]],
                ):
                    raise AdmissionError(
                        "Conversation is not visible to these participants", 403
                    )
                if payload.get("thread_id") and not thread:
                    raise AdmissionError("Unknown companion conversation", 404)
                payload["thread_id"] = conversation_id
            else:
                raise AdmissionError("Unknown interaction type")
            now = self.store.clock()
            # Input and attachments are canonical before acknowledgment, even before execution starts.
            message = {
                "id": uid("message"),
                "role": "user",
                "content": (
                    f"[image: {payload.get('image_media_type', 'image/jpeg')}] "
                    if image
                    else ""
                )
                + content,
                "timestamp": now,
                "metadata": {"request_id": client_id},
            }
            with self.store.transaction():
                stored_payload = dict(payload)
                if image_bytes is not None:
                    stored_payload.pop("image_data", None)
                    stored_payload["image_ref"] = self._image_reference(image_bytes)
                self.store.put(
                    "interaction_input",
                    work_id,
                    self.runtime.actor_id,
                    {"payload": stored_payload, "message": message},
                    category="platform",
                )
                if kind == "chat":
                    from backend.core.agent import Message

                    conv.messages.append(Message(**message))
                    conv.updated_at = now
                    conv.world_summarized = False
                    self.agent._save_conversation(conv)
                elif not thread:
                    self.store.put(
                        "companion_thread",
                        conversation_id,
                        "operator",
                        {"participants": ["operator", actor["id"]], "messages": []},
                        category="platform",
                    )
                self.store.put(
                    "interaction",
                    work_id,
                    self.runtime.actor_id,
                    {
                        "request_id": client_id,
                        "kind": kind,
                        "fingerprint": fingerprint,
                        "conversation_id": conversation_id,
                        "actor_id": payload.get("actor_id", self.runtime.actor_id),
                        "status": "queued",
                        "execution_id": None,
                        "result": None,
                        "detail": None,
                        "cursor": 0,
                        "approval": None,
                    },
                    category="platform",
                )
            epoch = self.runtime.notice_user_input()
            self._update(work_id, input_epoch=epoch)
            if kind == "chat":
                self._touch_visit(conversation_id)
            self.queue.put_nowait(work_id)
            self.emit(work_id, {"type": "typing", "status": True})
            return self.public(self.get(work_id))

    async def submit_household(self, payload, grant, snapshot, client_id=None):
        """Admit immutable canonical input from the authenticated coordinator only."""
        bridge = getattr(self.runtime, "household", None)
        if bridge is None:
            raise AdmissionError("This is not a household worker", 403)
        grant, snapshot = bridge.validate_admission(grant, snapshot)
        invitation = grant.get("kind") == "invitation"
        world = grant.get("kind") == "world"
        client_id = client_id or grant["id"]
        if not isinstance(client_id, str) or not 1 <= len(client_id) <= 128:
            raise AdmissionError("A submission identifier must have 1–128 characters")
        payload = dict(payload)
        if set(payload) - {"image_data", "image_media_type"}:
            raise AdmissionError("Household input content is selected by its admission")
        payload["message"] = bridge.input_record(grant)["content"] if invitation or world else grant["message"]["content"]
        payload["conversation_id"] = grant["conversation_id"]
        if not payload["message"].strip() or len(payload["message"]) > 32000:
            raise AdmissionError("Enter a message of 1–32000 characters")
        fingerprint = hashlib.sha256(encoded({"kind": "household", "payload": payload,
                                             "grant": grant, "snapshot": snapshot}).encode()).hexdigest()
        work_id = "request-" + hashlib.sha256(encoded([
            self.store.meta("installation_id"), self.runtime.actor_id, client_id,
        ]).encode()).hexdigest()
        async with self.admission:
            prior = self.store.get("interaction", work_id)
            if prior:
                if prior["data"]["fingerprint"] != fingerprint:
                    raise AdmissionError("Submission identifier already belongs to different input", 409)
                return self.public(prior)
            if self.closing or self.runtime.maintenance:
                raise AdmissionError("Application is preparing a checkpoint or release", 503)
            if self.queue.full():
                raise AdmissionError("The pending work queue is full; this input was not accepted", 429)
            stored_payload = dict(payload)
            if image := stored_payload.pop("image_data", None):
                try:
                    if not isinstance(image, str) or len(image) > 12 * 1024 * 1024:
                        raise ValueError()
                    content = base64.b64decode(image, validate=True)
                    if len(content) > 8 * 1024 * 1024:
                        raise ValueError()
                except (ValueError, TypeError):
                    raise AdmissionError("Image data must be valid base64 and at most 8 MiB") from None
                stored_payload["image_ref"] = self._image_reference(content)
            incoming = bridge.input_record(grant)
            human = not invitation and not world and not grant.get("invitation_reply") and grant["origin"] == "user" and grant["input_author"] == "operator"
            epoch = self.runtime.notice_user_input() if human else self.runtime.input_epoch
            with self.store.transaction():
                self.store.put("interaction_input", work_id, self.runtime.actor_id,
                    {"payload": stored_payload, "message": incoming, "household_grant": grant,
                     "household_snapshot": snapshot}, category="platform")
                self.store.put("interaction", work_id, self.runtime.actor_id,
                    {"request_id": client_id, "kind": "world" if world else "invitation" if invitation else "chat", "household": True,
                     "grant_id": grant["id"], "fingerprint": fingerprint,
                     "conversation_id": grant["conversation_id"], "actor_id": self.runtime.actor_id,
                     "input_author": grant["input_author"], "origin": grant["origin"],
                     "input_epoch": epoch, "status": "queued", "execution_id": None,
                     "result": None, "detail": None, "cursor": 0, "approval": None}, category="platform")
            if human:
                self._touch_visit(grant["conversation_id"])
                for identity, task in list(self.tasks.items()):
                    if self.get(identity)["data"]["kind"] == "world":
                        task.cancel()
            self.queue.put_nowait(work_id)
            self.emit(work_id, {"type": "typing", "status": True})
            return self.public(self.get(work_id))

    def emit(self, work_id, event):
        event = dict(event)
        for field in ("content", "output"):
            if isinstance(event.get(field), str):
                event[field] = event[field][:16000]
        with self.store.transaction():
            record = self.get(work_id)
            data = record["data"]
            event.update(
                work_id=work_id,
                request_id=data["request_id"],
                conversation_id=data["conversation_id"],
            )
            if data.get("household"):
                event["resident_id"] = self.runtime.actor_id
                event["grant_id"] = data["grant_id"]
            event_id = self.store.event(
                self.runtime.actor_id, "interaction_progress", event
            )
            seq = self.store.get_event(event_id)["seq"]
            changes = {"cursor": seq}
            if event.get("type") == "quiet_turn":
                changes["quiet_turns"] = event["ordinal"]
            if event.get("type") == "tool_approval_required":
                changes["approval"] = {
                    **event,
                    "expires_at": self.store.clock() + event.get("timeout_seconds", 30),
                }
            elif event.get("type") in {"tool_result", "message", "work_ended"}:
                changes["approval"] = None
            self._update(work_id, **changes)
            self.store.db.execute(
                "DELETE FROM events WHERE seq IN (SELECT seq FROM events WHERE kind='interaction_progress' AND actor=? AND json_extract(payload,'$.work_id')=? ORDER BY seq DESC LIMIT -1 OFFSET ?)",
                (self.runtime.actor_id, work_id, self.EVENT_LIMIT),
            )
        for queue in tuple(self.observers):
            if queue.empty():
                queue.put_nowait(True)

    def snapshot(self, work_id, after=0):
        if not isinstance(after, int) or after < 0:
            raise AdmissionError("Invalid event cursor")
        with self.store.transaction():
            record = self.get(work_id)
            rows = self.store.db.execute(
                "SELECT seq,payload FROM events WHERE kind='interaction_progress' AND actor=? AND json_extract(payload,'$.work_id')=? ORDER BY seq",
                (self.runtime.actor_id, work_id),
            ).fetchall()
            reset = bool(
                after > record["data"]["cursor"]
                or (after and rows and after < rows[0]["seq"])
            )
            return {
                "work": self.public(record),
                "events": [
                    {**json.loads(r["payload"]), "event_id": r["seq"]}
                    for r in rows
                    if reset or r["seq"] > after
                ],
                "cursor": record["data"]["cursor"],
                "reset": reset,
            }

    def subscribe(self):
        queue = asyncio.Queue(maxsize=1)
        self.observers.add(queue)
        queue.put_nowait(True)
        return queue

    async def wait(self, work_id):
        queue = self.subscribe()
        try:
            while True:
                data = self.get(work_id)["data"]
                if data["status"] not in {"queued", "running"}:
                    if data["result"] is not None:
                        return data["result"]
                    return self._interrupted_result(data)
                await queue.get()
        finally:
            self.observers.discard(queue)

    def _interrupted_result(self, data):
        if data["kind"] == "companion":
            actor = self.store.get("actor", data["actor_id"])
            return {
                "actor_id": data["actor_id"],
                "thread_id": data["conversation_id"],
                "speaker": actor["data"]["name"] if actor else "Companion",
                "text": "",
                "status": data["status"],
                "detail": data.get("detail"),
                "skills_used": [],
                "receipts": [],
            }
        return {
            "response": "",
            "conversation_id": data["conversation_id"],
            "state": self.agent.personality.state.to_dict(),
            "skills_used": [],
            "memories_recalled": 0,
            "turn_status": data["status"],
            "status_detail": data.get("detail"),
            "quiet_turns": 0,
        }

    def stop(self, work_id):
        record = self.get(work_id)
        if record["data"]["status"] in {"queued", "running"}:
            task = self.tasks.get(work_id)
            if task:
                task.cancel()
            else:
                self._update(
                    work_id, status="interrupted", detail="Stopped by the operator"
                )
                self.emit(work_id, {"type": "work_ended", "status": "interrupted"})
        return self.public(self.get(work_id))

    def approve(self, work_id, approval_id, approved=False, allow_always=False):
        approval = self.get(work_id)["data"].get("approval")
        if (
            not approval
            or approval["approval_id"] != approval_id
            or approval["expires_at"] <= self.store.clock()
            or approval_id not in self.agent._pending_approvals
        ):
            raise AdmissionError("Approval is unavailable or expired", 409)
        self.agent.resolve_approval(approval_id, approved, allow_always)
        self._update(work_id, approval=None)
        return {"status": "received"}

    async def _worker(self):
        while not self.closing:
            work_id = await self.queue.get()
            try:
                if self.get(work_id)["data"]["status"] != "queued":
                    continue
                task = asyncio.create_task(
                    self._run(work_id), context=contextvars.Context()
                )
                self.tasks[work_id] = task
                await task
            except asyncio.CancelledError:
                if self.closing:
                    raise
                self._update(
                    work_id,
                    status="interrupted",
                    detail="Stopped before execution started",
                )
                self.emit(work_id, {"type": "work_ended", "status": "interrupted"})
            finally:
                self.tasks.pop(work_id, None)
                self.queue.task_done()

    async def _run(self, work_id):
        data = self.get(work_id)["data"]
        household_variable = household_token = None
        try:
            source = self.store.get("interaction_input", work_id)
            payload = dict(source["data"]["payload"] if source else data["payload"])
            message = source["data"]["message"] if source else data["input_message"]
            if reference := payload.pop("image_ref", None):
                path = (self.runtime.workspace / reference["path"]).resolve()
                if (
                    not path.is_relative_to(self.runtime.workspace)
                    or not path.is_file()
                ):
                    raise ValueError("Attached image is unavailable")
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != reference["sha256"]:
                    raise ValueError("Attached image content changed")
                payload["image_data"] = base64.b64encode(content).decode()
            grant = source["data"].get("household_grant") if source else None
            origin = "user"
            if grant:
                from backend.core.household.worker_bridge import admission_context

                household_variable = admission_context()
                household_token = household_variable.set(grant)
                origin = "user" if grant["origin"] == "user" and grant["input_author"] == "operator" and grant.get("kind") != "invitation" and not grant.get("invitation_reply") else "peer"
            invitation = bool(grant and grant.get("kind") == "invitation")
            world = bool(grant and grant.get("kind") == "world")
            if world:
                origin = "autonomy"
            async with self.runtime.execution(origin, **({"allowed_tools": ["contact", "household_contacts", "end_turn"], "max_calls": 4} if invitation else {})) as execution:
                if grant and not invitation and not world:
                    self.runtime.household.project(self.agent, source["data"]["household_snapshot"], grant)
                self._update(
                    work_id, status="running", execution_id=execution.execution_id
                )
                self.store.set_run_provenance(
                    execution.execution_id, request_id=data["request_id"],
                    **({"household_grant_id": grant["id"], "conversation_id": grant["conversation_id"],
                        "input_author": grant["input_author"], "eligible_live": True} if grant else {}),
                )

                async def progress(event):
                    self.emit(work_id, event)

                if world:
                    from .world import tick
                    result = await tick(self.agent._world_thread, execution=execution, work_id=work_id, event_callback=progress)
                    result = {**result, "response": "", "conversation_id": None, "state": {}}
                elif invitation:
                    result = await self.runtime.household.decide_invitation(self.agent, grant, progress)
                elif data["kind"] == "chat":
                    result = await self.agent.chat(
                        user_message=payload["message"],
                        conversation_id=data["conversation_id"],
                        image_data=payload.get("image_data"),
                        image_media_type=payload.get("image_media_type", "image/jpeg"),
                        event_callback=progress,
                        _input_epoch=data["input_epoch"],
                        _admitted_message=message,
                    )
                else:
                    result = await self.runtime.companions.consult(
                        payload["actor_id"],
                        payload["content"],
                        payload.get("mode", "conversation"),
                        data["conversation_id"],
                        payload.get("allowed_tools"),
                        _caller_actor_id="operator",
                        _event_callback=progress,
                        _pending_approvals=self.agent._pending_approvals,
                    )
                current_terminal.set(
                    {
                        "status": result.get(
                            "turn_status", result.get("status", "completed")
                        )
                    }
                )
        except asyncio.CancelledError:
            self._update(
                work_id,
                status="interrupted",
                detail="Work interrupted; completed effects are saved",
            )
            result = self._interrupted_result(self.get(work_id)["data"])
        except Exception as exc:
            self._update(
                work_id,
                status="failed",
                detail=f"Work stopped after {type(exc).__name__}; inspect retained evidence",
            )
            result = self._interrupted_result(self.get(work_id)["data"])
        finally:
            if household_token is not None:
                household_variable.reset(household_token)
        status = result.get("turn_status", result.get("status", "completed"))
        self._update(
            work_id, status=status, result=result, finished_at=self.store.clock()
        )
        self.emit(work_id, {"type": "message", **result})
        if data["kind"] == "chat" and (not data.get("household") or (data.get("input_author") == "operator" and data.get("origin") == "user")):
            self._touch_visit(data["conversation_id"])

    def _touch_visit(self, conversation_id):
        now = self.store.clock()
        self.store.put(
            "conversation_visit",
            conversation_id,
            self.runtime.actor_id,
            {"deadline": now + self.resume_seconds, "status": "open"},
            category="platform",
        )
        self.store.set_meta("decompression_until", now + self.resume_seconds)

    async def expire_visits(self):
        if self.has_pending() or self.runtime.maintenance:
            return
        for visit in self.store.list("conversation_visit", self.runtime.actor_id):
            if (
                visit["data"]["status"] != "open"
                or visit["data"]["deadline"] > self.store.clock()
            ):
                continue
            async with self.runtime.execution("reflection"):
                current = self.store.get("conversation_visit", visit["id"])
                if self.has_pending() or current["revision"] != visit["revision"]:
                    return
                conv = self.agent.get_conversation(visit["id"])
                watermark = conv.messages[-1].id if conv and conv.messages else None
                previous = self.store.get("visit_watermark", visit["id"])
                if watermark and (
                    not previous or previous["data"].get("message_id") != watermark
                ):
                    await self.agent.on_conversation_ended(
                        visit["id"], schedule_resume=False
                    )
                    self.store.put(
                        "visit_watermark",
                        visit["id"],
                        self.runtime.actor_id,
                        {"message_id": watermark},
                        category="platform",
                    )
                self.store.put(
                    "conversation_visit",
                    visit["id"],
                    self.runtime.actor_id,
                    {**current["data"], "status": "closed"},
                    category="platform",
                )

    async def _presence(self):
        while not self.closing:
            try:
                await asyncio.sleep(5)
                await self.expire_visits()
            except asyncio.CancelledError:
                if self.closing:
                    raise
                asyncio.current_task().uncancel()
