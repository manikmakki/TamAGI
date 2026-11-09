"""Canonical native chat admission and resident work supervision.

Browser connections observe work; they never own its lifetime. Every resident
activation receives a fixed transcript watermark and one coordinator grant.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import httpx

from backend.core.runtime.interactions import AdmissionError
from .conversations import Conversations
from .contracts import AccessDenied, Conflict, NotFound
from .store import uid


TERMINAL = {"completed", "failed", "stopped", "interrupted", "resting", "waiting", "blocked", "declined", "budget_exhausted"}


class NativeInteractions:
    def __init__(self, registry, model, hosts):
        self.registry, self.store, self.model, self.hosts = registry, registry.store, model, hosts
        def validate(lineage, author):
            if lineage.get("completion_attempt_id"):
                return model.validate_completion_lineage(lineage, author)
            return model.broker.validate_message_lineage(lineage, author)
        self.conversations = Conversations(self.store, registry.participants, validate)
        self.tasks, self.observers = {}, set()
        self.maintenance = False
        self._commands = set()
        self._pulse_task = None
        # An interrupted coordinator never silently repeats uncertain effects.
        for record in self.records():
            if record["data"]["status"] not in TERMINAL:
                self._update(record["id"], status="interrupted", status_detail="Coordinator restarted; inspect retained work before continuing.")
        for record in self.store.list("native_grant"):
            if record["data"]["status"] not in TERMINAL:
                self.store.put("native_grant", record["id"], {**record["data"], "status": "interrupted"})
        for record in self.store.list("reply_offer"):
            if record["data"]["status"] in {"pending", "deferred"}:
                self.store.put("reply_offer", record["id"], {**record["data"], "status": "closed",
                    "recovery_note": "Coordinator restarted; previous opportunities were not automatically replayed."})
        for record in self.store.list("native_invitation_offer"):
            if record["data"]["status"] in {"pending", "deferred", "running"}:
                self.store.put("native_invitation_offer", record["id"], {**record["data"], "status": "closed",
                    "recovery_note": "Coordinator restarted; decision was not automatically replayed."})

    def start(self):
        self._pulse_task = asyncio.create_task(self._pulse())

    def refresh_availability(self, actor, ready):
        if self.maintenance:
            return
        # An independent evaluation can acquire the worker lease as a world/chat
        # activation ends. Its live execution outranks an earlier ready snapshot.
        executing = any(r['data']['actor_id'] == actor and r['data']['status'] == 'running'
                        and self.model.broker.exchange(r['data']['exchange_id'])['status'] == 'running'
                        for r in self.store.list('broker_execution'))
        if executing:
            self.model.broker.set_available(actor, True)
        elif not ready.get('pending'):
            self.model.broker.set_available(actor, ready.get('due', True))

    async def _pulse(self):
        while True:
            await asyncio.sleep(2)
            if self.maintenance:
                continue
            await self._deliver_shared_events()
            for invitation in self.store.list("native_invitation_offer"):
                data = invitation["data"]
                if data["status"] != "deferred":
                    continue
                try:
                    if self.model.broker.exchange(data["exchange_id"])["status"] != "running":
                        self.store.put("native_invitation_offer", invitation["id"], {**data, "status": "closed"})
                        continue
                    ready = await self.hosts[data["recipient"]].json("GET", "/_native/ready")
                    if ready.get("due"):
                        self.store.put("native_invitation_offer", invitation["id"], {**data, "status": "pending"})
                        message = self.store.get("message", data["trigger_message_id"])
                        self.enqueue_contact(data["recipient"], data["conversation_id"], {"id": message["id"], **message["data"]}, data["lineage"])
                except (RuntimeError, NotFound, httpx.TransportError):
                    continue
            for offer in self.store.list("reply_offer"):
                data = offer["data"]
                if data["status"] != "deferred":
                    continue
                actor = data["recipient"]
                try:
                    ready = await self.hosts[actor].json("GET", "/_native/ready")
                    self.refresh_availability(actor, ready)
                    if not ready.get("due"):
                        continue
                    if self.model.broker.exchange(data["exchange_id"])["status"] != "running":
                        self.store.put("reply_offer", offer["id"], {**data, "status": "closed"})
                        continue
                    self.store.put("reply_offer", offer["id"], {**data, "status": "pending"})
                    message = self.store.get("message", data["message_ids"][-1])
                    self.enqueue_contact(actor, data["conversation_id"], {"id": message["id"], **message["data"]}, data["lineage"])
                except (RuntimeError, NotFound, httpx.TransportError):
                    continue

    async def _deliver_shared_events(self):
        # Persist an independent delivery cursor per resident. These wake
        # observations never create an execution or inference allowance.
        from .world import World
        world = World(self.store)
        for actor, host in self.hosts.items():
            if self.maintenance:
                return
            key = "native_wake_cursor:" + actor
            for event in self.store.events(self.store.meta(key, 0))[:128]:
                eligible = event["kind"] in {"notebook_changed", "world_changed"}
                if eligible and event["kind"] == "world_changed":
                    try:
                        world.get(actor, event["data"]["object_id"])
                    except (AccessDenied, NotFound):
                        eligible = False
                if eligible:
                    try:
                        await host.json("POST", "/_native/wake", {k: event[k] for k in ("id", "kind", "created")})
                    except (RuntimeError, httpx.TransportError):
                        break
                self.store.set_meta(key, event["seq"])

    def get(self, identity):
        record = self.store.get("native_work", identity)
        if not record:
            raise AdmissionError("Unknown request", 404)
        return record

    def records(self, limit=100000):
        return list(reversed(self.store.list("native_work")))[0:limit]

    def public(self, record):
        return {"work_id": record["id"], "created_at": record["created"],
                **{k: v for k, v in record["data"].items() if k not in {"events", "payload", "children"}}}

    def _update(self, identity, **changes):
        record = self.get(identity)
        return self.store.put("native_work", identity, {**record["data"], **changes}, expected=record["revision"])

    def emit(self, identity, event):
        record = self.get(identity)
        data = record["data"]
        cursor = data.get("cursor", 0) + 1
        event = {**event, "work_id": identity, "request_id": data["request_id"],
                 "conversation_id": data["conversation_id"], "event_id": cursor}
        changes = {"cursor": cursor, "events": [*data.get("events", []), event][-512:]}
        if event.get("type") == "tool_approval_required":
            changes["approval"] = event
        elif event.get("type") in {"tool_result", "message", "work_ended"}:
            changes["approval"] = None
        self._update(identity, **changes)
        for queue in tuple(self.observers):
            if queue.empty():
                queue.put_nowait(True)

    def snapshot(self, identity, after=0):
        if type(after) is not int or after < 0:
            raise AdmissionError("Invalid event cursor")
        record = self.get(identity)
        events, cursor = record["data"].get("events", []), record["data"].get("cursor", 0)
        reset = after > cursor or bool(after and events and after < events[0]["event_id"] - 1)
        return {"work": self.public(record), "cursor": cursor, "reset": reset,
                "events": [e for e in events if reset or e["event_id"] > after]}

    def subscribe(self):
        queue = asyncio.Queue(maxsize=1)
        self.observers.add(queue)
        queue.put_nowait(True)
        return queue

    async def wait(self, identity):
        queue = self.subscribe()
        try:
            while True:
                data = self.get(identity)["data"]
                if data["status"] in TERMINAL:
                    return data.get("result") or {"response": "", "conversation_id": data["conversation_id"],
                        "state": {}, "skills_used": [], "memories_recalled": 0, "turn_status": data["status"]}
                await queue.get()
        finally:
            self.observers.discard(queue)

    async def submit(self, payload, client_id=None, *, kind="chat"):
        if self.maintenance:
            raise AdmissionError("Household paused briefly for resident recovery", 503)
        payload = copy.deepcopy(payload)
        resident = self.registry.resolve(payload.get("resident_id"))["resident_id"]
        request_id = client_id or uid("request")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise AdmissionError("Invalid request identity")
        text = payload.get("message", "")
        if not isinstance(text, str) or not text.strip() or len(text) > 32000:
            raise AdmissionError("Enter a message of at most 32000 characters")
        payload["resident_id"] = resident
        image = payload.get("image_data")
        image_bytes = None
        if image is not None:
            try:
                if not isinstance(image, str) or not image or len(image) > 12 * 1024 * 1024:
                    raise ValueError()
                image_bytes = base64.b64decode(image, validate=True)
                if not image_bytes or len(image_bytes) > 8 * 1024 * 1024:
                    raise ValueError()
            except (ValueError, TypeError):
                raise AdmissionError("Image data must be valid base64 and at most 8 MiB") from None
            if payload.get("image_media_type", "image/jpeg") not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
                raise AdmissionError("Choose a JPEG, PNG, GIF or WebP image")

        def admit():
            exchange = uid("exchange")
            self.model.broker.open_exchange(exchange, "user")
            lineage = {"exchange_id": exchange, "origin": "user"}
            conversation_id = payload.get("conversation_id")
            if conversation_id:
                conversation = self.conversations._conversation(conversation_id, "operator")
                if resident not in conversation["data"]["participants"]:
                    raise AccessDenied("Selected resident is not a participant in this thread")
                message = self.conversations.send("operator", conversation_id, text,
                    "input:" + request_id, mentions=payload.get("mentions"), lineage=lineage)
            else:
                members = payload.get("participants") or [resident]
                if not isinstance(members, list) or resident not in members or any(m not in self.hosts and m != "operator" for m in members):
                    raise AccessDenied("Choose available full residents for this thread")
                members = list(dict.fromkeys(["operator", *members]))
                result = self.conversations.create("operator", members, "input:" + request_id,
                    title=text[:80], group=len(members) > 2, opening=text,
                    mentions=payload.get("mentions"), lineage=lineage)
                conversation_id, message = result["conversation"]["id"], result["message"]
            stored_payload = dict(payload)
            if image_bytes is not None:
                descriptor = {"media_type": payload.get("image_media_type", "image/jpeg"),
                              "sha256": hashlib.sha256(image_bytes).hexdigest(), "size": len(image_bytes)}
                self.store.put("conversation_attachment", message["id"], {
                    **descriptor, "conversation_id": conversation_id, "image_data": image})
                record = self.store.get("message", message["id"])
                message = {**record["data"], "id": record["id"],
                           "metadata": {**record["data"].get("metadata", {}), "image": descriptor}}
                self.store.put("message", message["id"], {k: v for k, v in message.items() if k != "id"})
                stored_payload.pop("image_data", None)
            identity = uid("work")
            self.store.put("native_work", identity, {"request_id": request_id, "conversation_id": conversation_id,
                "resident_id": resident, "exchange_id": exchange, "status": "queued", "kind": kind,
                "cursor": 0, "events": [], "children": [], "responders": [], "input_message": self.ui_message(message),
                "payload": stored_payload, "approval": None, "result": None})
            return identity

        identity = self.store.command("operator", request_id, "native.chat", payload, admit)
        if self.get(identity)["data"]["status"] == "queued" and identity not in self.tasks:
            self.tasks[identity] = asyncio.create_task(self._run(identity))
        return self.public(self.get(identity))

    def attachment(self, viewer, conversation_id, message_id, through=None):
        self.conversations._conversation(conversation_id, viewer)
        message = self.store.get("message", message_id)
        source = self.store.get("conversation_attachment", message_id)
        if (not message or message["data"]["conversation_id"] != conversation_id
                or through is not None and message["data"]["sequence"] > through
                or not source or source["data"]["conversation_id"] != conversation_id):
            raise NotFound("Attached image is unavailable in this transcript")
        return source["data"]

    def ui_message(self, message):
        names = {p["id"]: p["name"] for p in self.registry.participants()}
        author = message["author"]
        return {"id": message["id"], "role": "user" if author == "operator" else "assistant",
                "content": message["content"], "timestamp": message["timestamp"], "sequence": message["sequence"],
                "metadata": {**message.get("metadata", {}), "author_id": author, "author_name": names.get(author, author),
                    "mentions": message.get("addressees", []) if message.get("explicit_mentions") else [],
                    "sequence": message["sequence"], "message_id": message["id"]}}

    def grant(self, identity, resident):
        record = self.store.get("native_grant", identity)
        if not record or record["data"]["resident_id"] != resident or record["data"]["status"] in TERMINAL:
            raise AccessDenied("Resident grant is unavailable")
        return record["data"]

    def enqueue_contact(self, actor, conversation_id, message, lineage):
        # One dispatcher owns this exchange/thread, including messages posted by
        # tools while its current activation is still in progress.
        for record in self.records():
            data = record["data"]
            if data["exchange_id"] == lineage["exchange_id"] and data["conversation_id"] == conversation_id and data["status"] not in TERMINAL:
                return record["id"]
        identity = uid("work")
        members = self.conversations._conversation(conversation_id)["data"]["participants"]
        owner = actor if actor in self.hosts and actor in members else next(member for member in members if member in self.hosts)
        self.store.put("native_work", identity, {"request_id": identity, "conversation_id": conversation_id,
            "resident_id": owner, "exchange_id": lineage["exchange_id"], "status": "queued", "kind": "chat",
            "cursor": 0, "events": [], "children": [], "responders": [], "input_message": self.ui_message(message),
            "payload": {}, "approval": None, "result": None, "resident_initiated": True})
        # Dispatch after the transaction commits; a lost browser is irrelevant.
        def start():
            self.tasks[identity] = asyncio.create_task(self._run(identity))
        asyncio.get_running_loop().call_soon(start)
        return identity

    def offer_invitation_reply(self, actor, invitation, lineage):
        """Acceptance offers only the invitation trigger, never historical backlog."""
        from .conversations import _stable
        data = invitation["data"]
        self.conversations._conversation(data["conversation_id"], actor)
        trigger = self.store.get("message", data.get("trigger_message_id"))
        if not trigger:
            return
        identity = _stable("offer", data["conversation_id"], lineage["exchange_id"], actor)
        previous = self.store.get("reply_offer", identity)
        if previous and previous["data"]["status"] in {"silent", "declined", "exhausted", "closed"}:
            return
        self.store.put("reply_offer", identity, {"conversation_id": data["conversation_id"],
            "exchange_id": lineage["exchange_id"], "recipient": actor, "status": "pending",
            "message_ids": list(dict.fromkeys([*(previous["data"]["message_ids"] if previous else []), trigger["id"]])),
            "lineage": lineage, "created_at": self.store.clock(), "generation": 1,
            "history": [], "invitation_id": invitation["id"]})
        self.enqueue_contact(actor, data["conversation_id"], {"id": trigger["id"], **trigger["data"]}, lineage)

    def submit_world(self, actor, scheduled_at):
        if self.maintenance:
            raise AdmissionError("Household paused briefly for resident recovery", 503)
        self.registry.resolve(actor)
        identity, exchange, grant_id = uid("world-work"), uid("world-exchange"), uid("grant")
        with self.store.transaction():
            self.model.broker.open_exchange(exchange, "autonomy")
            self.store.put("native_world_exchange", exchange, {"sponsor_id": actor, "work_id": identity, "scheduled_at": scheduled_at})
            grant = {"id": grant_id, "kind": "world", "resident_id": actor, "exchange_id": exchange,
                     "conversation_id": None, "origin": "autonomy", "first_response": False,
                     "status": "running", "work_id": identity}
            self.store.put("native_grant", grant_id, grant)
            self.store.put("native_work", identity, {"request_id": identity, "conversation_id": None,
                "resident_id": actor, "exchange_id": exchange, "status": "queued", "kind": "world",
                "cursor": 0, "events": [], "children": [], "responders": [], "payload": {},
                "approval": None, "result": None, "grant_id": grant_id, "scheduled_at": scheduled_at})
            self.store.event("world_tick_admitted", actor, {"work_id": identity, "exchange_id": exchange, "scheduled_at": scheduled_at})
        self.tasks[identity] = asyncio.create_task(self._run(identity))
        return self.public(self.get(identity))

    async def _run_world(self, identity):
        while self.maintenance:
            await asyncio.sleep(.05)
        data = self.get(identity)["data"]
        if data["status"] != "queued":
            return
        actor, grant_id = data["resident_id"], data["grant_id"]
        self._update(identity, status="running")
        try:
            child = await self.hosts[actor].json("POST", "/_native/respond", {
                "grant": {"id": grant_id, **self.grant(grant_id, actor)}, "snapshot": {}, "request_id": grant_id, "payload": {}})
            self._update(identity, children=[{"resident_id": actor, "work_id": child["work_id"], "grant_id": grant_id}])
            cursor = 0
            while self.get(identity)["data"]["status"] not in TERMINAL:
                result = await self.hosts[actor].json("GET", f"/_native/work/{child['work_id']}?after={cursor}")
                for event in result["events"]:
                    if event.get("type") not in {"accepted", "message", "work_ended"}:
                        self.emit(identity, {**event, "resident_id": actor, "grant_id": grant_id})
                cursor = result["cursor"]
                current = result["work"]
                if current["status"] in TERMINAL:
                    self.store.put("native_grant", grant_id, {**self.grant(grant_id, actor), "status": current["status"]})
                    self._update(identity, status=current["status"], result=current.get("result"))
                    self.emit(identity, {"type": "work_ended", "status": current["status"]})
                    ready = await self.hosts[actor].json("GET", "/_native/ready")
                    self.refresh_availability(actor, ready)
                    return
                await asyncio.sleep(.08)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.get(identity)["data"]["status"] not in TERMINAL:
                self.stop(identity)
                self._update(identity, status="failed", status_detail=str(exc)[:500])

    async def _invitation_decision(self, identity, offer):
        data = offer["data"]
        actor = data["recipient"]
        ready = await self.hosts[actor].json("GET", "/_native/ready")
        if not ready.get("due", True):
            self.store.put("native_invitation_offer", offer["id"], {**data, "status": "deferred"})
            return
        exchange = self.model.broker.exchange(data["exchange_id"])
        if exchange["status"] != "running" or exchange["calls"] >= exchange["max_calls"] or (exchange["origin"] == "user" and exchange["segment_calls"] >= 20):
            self.store.put("native_invitation_offer", offer["id"], {**data, "status": "exhausted"})
            return
        invitation = self.store.get("conversation_invitation", data["invitation_id"])
        if invitation["data"]["status"] != "pending" or invitation["revision"] != data["version"]:
            self.store.put("native_invitation_offer", offer["id"], {**data, "status": "closed"})
            return
        from .native_messaging import NativeMessaging
        card = NativeMessaging(self.registry, self.model, self).card(invitation)
        grant_id = uid("grant")
        grant = {"id": grant_id, "kind": "invitation", "resident_id": actor, "exchange_id": data["exchange_id"],
            "conversation_id": data["conversation_id"], "origin": exchange["origin"], "first_response": False,
            "invitation_id": invitation["id"], "invitation_version": invitation["revision"], "invitation": card,
            "status": "running", "work_id": identity}
        self.store.put("native_grant", grant_id, grant)
        self.store.put("native_invitation_offer", offer["id"], {**data, "status": "running"})
        child = await self.hosts[actor].json("POST", "/_native/respond", {
            "grant": grant, "snapshot": {}, "request_id": grant_id, "payload": {}})
        children = [*self.get(identity)["data"]["children"], {"resident_id": actor, "work_id": child["work_id"], "grant_id": grant_id}]
        self._update(identity, children=children)
        # Pre-membership decision reasoning stays private to the invitee's store.
        while True:
            result = await self.hosts[actor].json("GET", f"/_native/work/{child['work_id']}")
            current = result["work"]
            if current["status"] in TERMINAL:
                break
            if self.get(identity)["data"]["status"] in TERMINAL:
                return
            await asyncio.sleep(.08)
        self.store.put("native_grant", grant_id, {**self.grant(grant_id, actor), "status": current["status"]})
        latest = self.store.get("native_invitation_offer", offer["id"])
        if latest["data"]["status"] == "running":
            status = ("deferred" if current["status"] in {"resting", "waiting"} else
                      "exhausted" if current["status"] == "budget_exhausted" else "silent")
            self.store.put("native_invitation_offer", offer["id"], {**latest["data"], "status": status})
        ready = await self.hosts[actor].json("GET", "/_native/ready")
        self.refresh_availability(actor, ready)
        self.emit(identity, {"type": "membership_changed", "resident_id": actor})
        if current["status"] == "failed":
            raise RuntimeError("Invitation decision execution failed")

    async def _run(self, identity):
        while self.maintenance:
            await asyncio.sleep(.05)
        if self.get(identity)["data"]["kind"] == "world":
            return await self._run_world(identity)
        if self.get(identity)["data"]["status"] != "queued":
            return
        self._update(identity, status="running")
        counts = {}
        try:
            while self.get(identity)["data"]["status"] not in TERMINAL:
                while self.maintenance:
                    await asyncio.sleep(.05)
                data = self.get(identity)["data"]
                invitations = [r for r in self.store.list("native_invitation_offer")
                    if r["data"]["status"] == "pending" and r["data"]["exchange_id"] == data["exchange_id"]
                    and r["data"]["conversation_id"] == data["conversation_id"]]
                if invitations:
                    await self._invitation_decision(identity, invitations[0])
                    continue
                pending = [offer for actor in self.hosts for offer in self.conversations.offers(actor)
                           if offer["data"]["exchange_id"] == data["exchange_id"]
                           and offer["data"]["conversation_id"] == data["conversation_id"]]
                if not pending:
                    break
                # Everyone gets an initial opportunity before another participant
                # takes a second turn; the shared broker is the resource bound.
                pending.sort(key=lambda o: (counts.get(o["data"]["recipient"], 0), o["created"], o["id"]))
                offer = pending[0]
                actor = offer["data"]["recipient"]
                trigger = self.store.get("message", offer["data"]["message_ids"][-1])["data"]
                ready = await self.hosts[actor].json("GET", "/_native/ready")
                actual_human = trigger["author"] == "operator" and not offer["data"].get("invitation_id")
                if not actual_human and not ready.get("due", True):
                    self.conversations.decide_offer(actor, offer["id"], "deferred", uid("defer"))
                    self.refresh_availability(actor, ready)
                    continue
                self.model.broker.set_available(actor, True)
                exchange = self.model.broker.exchange(data["exchange_id"])
                if exchange["status"] != "running" or exchange["calls"] >= exchange["max_calls"] or (exchange["origin"] == "user" and exchange["segment_calls"] >= 20):
                    for remaining in pending:
                        self.conversations.decide_offer(remaining["data"]["recipient"], remaining["id"], "exhausted", uid("decision"))
                    break
                snapshot = self.conversations.snapshot(actor, data["conversation_id"])
                messages = snapshot["messages"]
                addressed = [m for m in messages if m["id"] in offer["data"]["message_ids"]]
                if not addressed:
                    self.conversations.decide_offer(actor, offer["id"], "closed", uid("decision"))
                    continue
                grant_id = uid("grant")
                grant = {"id": grant_id, "resident_id": actor, "exchange_id": data["exchange_id"],
                    "conversation_id": data["conversation_id"], "origin": "user" if addressed[-1]["author"] == "operator" else "autonomy",
                    "first_response": actual_human and not counts.get(actor),
                    "through": snapshot["watermark"], "message": addressed[-1], "message_ids": [m["id"] for m in addressed],
                    "offer_id": offer["id"], "status": "running", "work_id": identity}
                if offer["data"].get("invitation_id"):
                    grant["invitation_reply"] = True
                # Peer activations share the actual initiating exchange; their
                # transport origin does not mint a new allowance or user signal.
                grant["origin"] = exchange["origin"]
                self.store.put("native_grant", grant_id, grant)
                child = await self.hosts[actor].json("POST", "/_native/respond", {
                    "grant": grant, "snapshot": snapshot, "request_id": grant_id,
                    "payload": {}})
                children = [*self.get(identity)["data"]["children"], {"resident_id": actor, "work_id": child["work_id"], "grant_id": grant_id}]
                self._update(identity, children=children, responders=list(dict.fromkeys([*data["responders"], actor])))
                cursor = 0
                while True:
                    result = await self.hosts[actor].json("GET", f"/_native/work/{child['work_id']}?after={cursor}")
                    for event in result["events"]:
                        if event.get("type") not in {"accepted", "work_ended", "message"}:
                            self.emit(identity, {**event, "resident_id": actor, "grant_id": grant_id,
                                                 "resident_work_id": child["work_id"]})
                        elif event.get("type") == "message":
                            self.emit(identity, {"type": "thread_changed", "resident_id": actor,
                                                 "grant_id": grant_id, "resident_work_id": child["work_id"]})
                    cursor = result["cursor"]
                    current = result["work"]
                    if current["status"] in TERMINAL:
                        if actor == data["resident_id"] and current.get("result"):
                            self._update(identity, result={**current["result"], "resident_id": actor})
                        break
                    if self.get(identity)["data"]["status"] in TERMINAL:
                        return
                    await asyncio.sleep(.08)
                self.store.put("native_grant", grant_id, {**self.grant(grant_id, actor), "status": current["status"]})
                counts[actor] = counts.get(actor, 0) + 1
                ready = await self.hosts[actor].json("GET", "/_native/ready")
                self.refresh_availability(actor, ready)
                current_offer = self.store.get("reply_offer", offer["id"])
                if current_offer["data"]["status"] == "pending" and not self.store.get("native_grant_commit", grant_id):
                    self.conversations.decide_offer(actor, offer["id"], "silent", uid("decision"))
                if current["status"] == "failed":
                    raise RuntimeError(current.get("status_detail") or "Resident execution failed")
            if self.get(identity)["data"]["status"] not in TERMINAL:
                self._update(identity, status="completed")
                self.emit(identity, {"type": "work_ended", "status": "completed"})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.get(identity)["data"]["status"] not in TERMINAL:
                # A lost acknowledgment does not prove the resident stopped.
                # Revoke derived execution authority before reporting failure.
                self.stop(identity)
                self._update(identity, status="failed", status_detail=str(exc)[:500])
                self.emit(identity, {"type": "work_ended", "status": "failed", "message": str(exc)[:500]})

    def commit(self, actor, body):
        grant_id, local = body["grant_id"], body["local_message"]
        def apply():
            grant = self.grant(grant_id, actor)
            if grant.get("kind") in {"invitation", "world"}:
                raise AccessDenied("Use contact to publish an explicit message from this activation")
            execution = self.store.get("native_grant_execution", grant_id)
            if not execution:
                raise AccessDenied("No model execution was admitted for this response")
            lineage = self.model.completion_lineage_for_grant(grant_id, actor)
            if local.get("role") != "assistant":
                raise AccessDenied("A resident may author only its own response")
            offer_id = None if self.store.get("native_grant_commit", grant_id) else grant.get("offer_id")
            late = []
            if offer_id:
                offer = self.store.get("reply_offer", offer_id)
                late = [m for m in offer["data"]["message_ids"] if m not in grant["message_ids"]]
                self.store.put("reply_offer", offer_id, {**offer["data"], "message_ids": grant["message_ids"]})
            message = self.conversations.send(actor, grant["conversation_id"], local["content"],
                "canonical:" + body["command_id"], lineage=lineage, reply_offer_id=offer_id,
                admitted_id=local["id"], admitted_timestamp=local.get("timestamp"),
                metadata={**(local.get("metadata") or {}), "household_grant_id": grant_id})
            self.store.put("native_grant_commit", grant_id, {"message_id": message["id"]})
            if late:
                answered = self.store.get("reply_offer", offer_id)["data"]
                self.store.put("reply_offer", offer_id, {**answered, "status": "pending", "message_ids": late,
                    "generation": answered.get("generation", 1) + 1, "history": [*answered.get("history", []), answered]})
            return message
        return self.store.command(actor, body["command_id"], "native.commit", body, apply)

    def stop(self, identity):
        record = self.get(identity)
        if record["data"]["status"] not in TERMINAL:
            exchange_id = record["data"]["exchange_id"]
            affected = [r for r in self.records() if r["data"]["exchange_id"] == exchange_id and r["data"]["status"] not in TERMINAL]
            for work in affected:
                self._update(work["id"], status="stopped")
            for grant in self.store.list("native_grant"):
                if grant["data"].get("exchange_id") == exchange_id and grant["data"]["status"] not in TERMINAL:
                    self.store.put("native_grant", grant["id"], {**grant["data"], "status": "stopped"})
                    execution = self.store.get("native_grant_execution", grant["id"])
                    if execution:
                        binding = self.model._binding(grant["data"]["resident_id"], execution["data"]["local_execution_id"])
                        self.model.broker.stop_execution(binding["broker_execution_id"])
            self.model.broker.close_exchange(exchange_id, "stopped")
            for offer in self.store.list("reply_offer"):
                if offer["data"]["exchange_id"] == exchange_id and offer["data"]["status"] in {"pending", "deferred"}:
                    self.store.put("reply_offer", offer["id"], {**offer["data"], "status": "closed", "decided_at": self.store.clock()})
            for offer in self.store.list("native_invitation_offer"):
                if offer["data"]["exchange_id"] == exchange_id and offer["data"]["status"] in {"pending", "deferred", "running"}:
                    self.store.put("native_invitation_offer", offer["id"], {**offer["data"], "status": "closed"})
            for work in affected:
                for child in work["data"]["children"]:
                    self._schedule_command(child["resident_id"], f"/_native/work/{child['work_id']}/stop", {})
                self.emit(work["id"], {"type": "work_ended", "status": "stopped"})
        return self.public(self.get(identity))

    def _schedule_command(self, actor, path, body):
        async def perform():
            try:
                await self.hosts[actor].json("POST", path, body)
            except Exception as exc:
                self.store.event("native_control_unconfirmed", actor,
                    {"operation": path.rsplit("/", 1)[-1], "error_class": type(exc).__name__})
        task = asyncio.create_task(perform())
        self._commands.add(task)
        task.add_done_callback(self._commands.discard)

    def approve(self, identity, approval_id, approved=False, allow_always=False):
        data = self.get(identity)["data"]
        approval = data.get("approval") or {}
        if approval.get("approval_id") != approval_id:
            raise AdmissionError("Approval is no longer pending", 409)
        child = next(c for c in reversed(data["children"]) if c["resident_id"] == approval["resident_id"])
        self._schedule_command(child["resident_id"], f"/_native/work/{child['work_id']}/approval",
            {"approval_id": approval_id, "approved": approved, "allow_always": allow_always})
        return {"accepted": True}

    async def close(self):
        if self._pulse_task:
            self._pulse_task.cancel()
            await asyncio.gather(self._pulse_task, return_exceptions=True)
        for record in self.records():
            if record["data"]["status"] not in TERMINAL:
                self.stop(record["id"])
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        await asyncio.gather(*self._commands, return_exceptions=True)
