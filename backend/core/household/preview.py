"""Local development UI for household fixtures, with no production/model startup."""

from __future__ import annotations

import asyncio
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.auth import hash_password, verify_password
from .contracts import AccessDenied, BudgetExhausted, Conflict, NotFound, ProtocolError, ReconciliationRequired
from .coordinator import SandboxCoordinator
from .driver import ProviderResult
from .store import uid

FRONTEND = Path(__file__).resolve().parents[3] / "frontend" / "household-preview.html"
SCOPE = "Disposable household preview. Replies and generation durations are scripted; no real model or push delivery."


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginInput(Input):
    password: str = Field(min_length=1, max_length=256)


class MessageInput(Input):
    content: str = Field(min_length=1, max_length=32000)
    mentions: list[str] | None = Field(default=None, max_length=8)


class ThreadInput(MessageInput):
    participants: list[str] = Field(min_length=1, max_length=4)
    title: str = Field(default="", max_length=200)
    group: bool = False


class ReadInput(Input):
    snapshot_id: str = Field(min_length=1, max_length=256)
    watermark: int = Field(ge=0, strict=True)


class ReplyInput(Input):
    offer_id: str = Field(min_length=1, max_length=256)


class AvailabilityInput(Input):
    available: bool = Field(strict=True)


class WorldInput(Input):
    owner: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=200)
    visibility: str = "shared"


class PreviewProvider:
    def __init__(self, clock, names):
        self.clock, self.names = clock, names
        self.timeline = []
        self.durations = {actor: seconds for actor, seconds in zip(names, (12.0, 3.0, 6.0))}

    async def generate(self, attempt):
        started = self.clock[0]
        await asyncio.sleep(0)
        self.clock[0] += self.durations[attempt["actor_id"]]
        self.timeline.append({"attempt_id": attempt["attempt_id"], "actor_id": attempt["actor_id"],
            "sponsor_id": attempt["sponsor_id"], "start": started, "end": self.clock[0]})
        return ProviderResult(f"Scripted fixture reply from {self.names[attempt['actor_id']]}. "
                              "This is a test response; no language model was called.", {"completion_tokens": 5})


class Preview:
    def __init__(self, registry):
        self.registry, self.store = registry, registry.store
        self.clock = [time.time()]
        self.store.clock = lambda: self.clock[0]
        self.coordinator = SandboxCoordinator(registry)
        self.names = {r["id"]: r["data"]["name"] for r in self.store.list("resident")}
        self.provider = PreviewProvider(self.clock, self.names)
        self.driver = self.coordinator.driver(self.provider)
        self.mutations = asyncio.Lock()

    @staticmethod
    def command_id(request):
        key = request.headers.get("Idempotency-Key") or uid("preview-command")
        if not 1 <= len(key) <= 128 or not key.isascii() or any(ord(c) < 33 for c in key):
            raise ProtocolError("Use an ASCII Idempotency-Key of at most 128 characters")
        return key

    def finish(self, actor, execution_id):
        self.coordinator.broker.finish_execution(execution_id)
        lease = self.store.get("resident_lease", actor)
        if lease and lease["data"]["execution_id"] == execution_id:
            self.store.put("resident_lease", actor, {**lease["data"], "status": "completed"})

    def state(self):
        self.coordinator.refresh_workers()
        broker = self.coordinator.broker.state()
        unread = self.coordinator.conversations.unread("operator")
        counts = {r["conversation_id"]: r["count"] for r in unread["threads"]}
        threads = [{"id": record["id"], **record["data"], "unread": counts.get(record["id"], 0)}
                   for record in self.store.list("conversation") if "operator" in record["data"]["participants"]]
        visible = {thread["id"] for thread in threads}
        return {"household_id": self.registry.household_id, "scope": SCOPE,
            "residents": [{"id": actor, "name": name, **broker["accounts"][actor],
                           "activity": "available" if broker["accounts"][actor]["available"] else "unavailable"}
                          for actor, name in self.names.items()],
            "conversations": threads, "unread_count": unread["count"],
            "world": self.coordinator.world.list("operator"),
            "attempts": self.store.list("broker_attempt"),
            "offers": [r for r in self.store.list("reply_offer") if r["data"]["status"] == "pending"
                       and r["data"]["conversation_id"] in visible],
            "notifications": [r for r in self.store.list("notification_intent") if r["data"]["recipient"] == "operator"],
            "provider_timeline": list(self.provider.timeline), "broker": broker}

    def operator_send(self, thread, body, key):
        def apply():
            self.coordinator.conversations.history("operator", thread)
            exchange = uid("exchange")
            self.coordinator.broker.open_exchange(exchange, origin="user")
            return self.coordinator.conversations.send("operator", thread, body.content, key + ":message",
                mentions=body.mentions, lineage={"exchange_id": exchange, "root_id": exchange, "origin": "user"})
        return self.store.command("operator", key, "preview.operator_message", {"thread": thread, **body.model_dump()}, apply)

    def initiate(self, actor, body, key):
        def apply():
            if not self.coordinator.broker.state()["accounts"][actor]["available"]:
                raise Conflict("This fixture is unavailable; make it available before initiating")
            lineage = self.coordinator.begin_autonomy(actor)
            result = self.coordinator.initiate(actor, body.participants, body.content, key + ":message",
                title=body.title, group=body.group, mentions=body.mentions)
            self.finish(actor, lineage["execution_id"])
            return result
        self.registry.resolve(actor)
        return self.store.command(actor, key, "preview.resident_initiate", body.model_dump(), apply)

    async def reply(self, actor, body, key):
        self.registry.resolve(actor)
        offer = self.store.get("reply_offer", body.offer_id)
        if offer is None:
            raise NotFound("Unknown reply offer")
        self.coordinator.conversations.history("operator", offer["data"]["conversation_id"])

        def admit():
            if not self.coordinator.broker.state()["accounts"][actor]["available"]:
                raise Conflict("This fixture is unavailable; its reply offer can wait")
            lineage = self.coordinator.begin_reply(actor, body.offer_id)
            attempt = uid("preview-attempt")
            self.coordinator.queue_generation(actor, attempt)
            offer = self.store.get("reply_offer", body.offer_id)["data"]
            return {"attempt_id": attempt, "lineage": lineage, "conversation_id": offer["conversation_id"]}

        admission = self.store.command(actor, key, "preview.reply", body.model_dump(), admit)
        attempt_id = admission["attempt_id"]
        status = self.coordinator.broker.attempt(attempt_id)["status"]
        if status == "queued":
            selected = self.driver.dispatch()
            if selected != attempt_id:
                raise ReconciliationRequired("The admitted fixture is not dispatchable; inspect broker state")
        result = await self.driver.wait(attempt_id)
        if not result.get("delivery_permitted"):
            raise ReconciliationRequired("Fixture response cannot be delivered; inspect the retained outcome")

        def commit():
            message = self.coordinator.conversations.send(actor, admission["conversation_id"], result["text"],
                key + ":message", lineage=admission["lineage"], reply_offer_id=body.offer_id)
            self.finish(actor, admission["lineage"]["execution_id"])
            return {"message": message, "attempt_id": attempt_id}

        return self.store.command(actor, key + ":result", "preview.reply_result", {"attempt_id": attempt_id}, commit)

    async def scheduling_demo(self, key):
        def admit():
            admissions = []
            for actor, account in self.coordinator.broker.state()["accounts"].items():
                if account["available"]:
                    lineage = self.coordinator.begin_autonomy(actor)
                    attempt = uid("preview-attempt")
                    self.coordinator.queue_generation(actor, attempt)
                    admissions.append({"actor": actor, "attempt": attempt, "execution": lineage["execution_id"]})
            return admissions

        admissions = self.store.command("operator", key, "preview.scheduling_demo", {}, admit)
        wanted = {a["attempt"] for a in admissions}
        while any(self.coordinator.broker.attempt(identity)["status"] == "queued" for identity in wanted):
            identity = self.driver.dispatch()
            if identity is None:
                raise Conflict("Scheduling demonstration is waiting for eligibility or recovery")
            await self.driver.wait(identity)
        for admission in admissions:
            if self.coordinator.broker.execution(admission["execution"])["status"] == "running":
                self.finish(admission["actor"], admission["execution"])
        return {"attempt_ids": list(wanted), "scope": "Synthetic provider-time demonstration; no messages or real model calls."}


LOGIN_HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Household preview login</title>
<style>body{background:#181c23;color:#f2eee6;font:17px system-ui;margin:12vh auto;padding:24px;max-width:460px}input,button{box-sizing:border-box;width:100%;padding:13px;margin:10px 0;font:inherit;border-radius:8px}button{background:#f1bd83;color:#201b16;border:0;cursor:pointer}p{line-height:1.6;color:#c5c8d0}#error{color:#ffb7a5}</style>
<h1>Household preview</h1><p>A separate development household with scripted residents. No real model calls, Ember data, or push notifications.</p>
<form id="login"><label for="password">Preview password</label><input id="password" type="password" autocomplete="current-password" required><button>Open test household</button></form><p id="error" role="alert"></p>
<script>document.getElementById('login').onsubmit=async event=>{event.preventDefault();try{const r=await fetch('/api/preview/login',{method:'POST',headers:{'Content-Type':'application/json','X-Household-Preview':'1'},body:JSON.stringify({password:document.getElementById('password').value})});if(!r.ok)throw new Error('Incorrect preview password');location.replace('/')}catch(e){document.getElementById('error').textContent=e.message}};</script></html>"""


def create_preview_app(registry, password: str) -> FastAPI:
    """Own the supplied disposable registry for one interactive process lifespan."""
    password_hash = hash_password(password)

    @asynccontextmanager
    async def lifespan(app):
        preview = None
        try:
            if not registry.store.list("resident"):
                for name in ("Juniper", "Lark", "Moss"):
                    registry.provision(name, seed=f"{name}: a disposable scripted preview fixture. No lived predecessor material.")
            preview = Preview(registry)
            for actor in preview.names:
                await preview.coordinator.start_worker(actor)
            app.state.preview = preview
            yield
        finally:
            if preview:
                await preview.driver.close()
                await preview.coordinator.close()
            registry.close()

    app = FastAPI(title="TamAGI household development preview", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def access(request, call_next):
        def headers(response):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "same-origin"
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            return response

        public = request.url.path in {"/login", "/api/preview/login", "/api/preview/health"}
        if not public and request.session.get("household") != registry.household_id:
            if request.url.path.startswith("/api/"):
                return headers(JSONResponse({"detail": "Preview login required"}, status_code=401))
            return headers(RedirectResponse("/login", status_code=303))
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if request.headers.get("X-Household-Preview") != "1" or request.headers.get("content-type", "").split(";")[0] != "application/json":
                return headers(JSONResponse({"detail": "Preview JSON request header required"}, status_code=403))
            origin = request.headers.get("origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                except ValueError:
                    return headers(JSONResponse({"detail": "Malformed preview Origin"}, status_code=403))
                if parsed.scheme != request.url.scheme or parsed.netloc != request.url.netloc:
                    return headers(JSONResponse({"detail": "Cross-origin preview mutation rejected"}, status_code=403))
        response = await call_next(request)
        return headers(response)

    app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32),
        session_cookie="tamagi_household_preview_" + registry.household_id[-12:], same_site="strict", max_age=43200)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    async def error_response(request, error):
        code = 403 if isinstance(error, AccessDenied) else 404 if isinstance(error, NotFound) else 409 if isinstance(error, (Conflict, BudgetExhausted, ReconciliationRequired)) else 400
        return JSONResponse({"detail": str(error)}, status_code=code)

    for exception in (AccessDenied, NotFound, Conflict, BudgetExhausted, ReconciliationRequired, ProtocolError):
        app.add_exception_handler(exception, error_response)

    @app.get("/api/preview/health")
    async def health():
        return {"ready": hasattr(app.state, "preview"), "fixture_only": True, "real_model": False}

    @app.get("/login")
    async def login_page():
        return HTMLResponse(LOGIN_HTML)

    @app.post("/api/preview/login")
    async def login(body: LoginInput, request: Request):
        if not verify_password(body.password, password_hash):
            return JSONResponse({"detail": "Incorrect preview password"}, status_code=401)
        request.session.clear()
        request.session["household"] = registry.household_id
        return {"ok": True}

    @app.post("/api/preview/logout")
    async def logout(request: Request):
        request.session.clear()
        return {"ok": True}

    @app.get("/")
    async def page():
        return FileResponse(FRONTEND, media_type="text/html")

    @app.get("/api/preview/state")
    async def state():
        return app.state.preview.state()

    @app.get("/api/preview/residents/{actor}")
    async def resident(actor: str):
        p = app.state.preview
        result = await p.coordinator.inspect(actor)
        return {**result, "world": p.coordinator.world.list(actor)}

    @app.get("/api/preview/threads/{thread}")
    async def thread(thread: str):
        return app.state.preview.coordinator.conversations.snapshot("operator", thread)

    @app.post("/api/preview/threads/{thread}/read")
    async def read(thread: str, body: ReadInput, request: Request):
        p = app.state.preview
        async with p.mutations:
            return p.coordinator.conversations.mark_read("operator", thread, body.watermark, p.command_id(request), snapshot_id=body.snapshot_id)

    @app.post("/api/preview/threads")
    async def create_thread(body: ThreadInput, request: Request):
        p = app.state.preview
        async with p.mutations:
            key = p.command_id(request)
            if body.mentions is not None:
                def apply():
                    exchange = uid("exchange")
                    p.coordinator.broker.open_exchange(exchange, "user")
                    return p.coordinator.conversations.create("operator", body.participants, key + ":message",
                        title=body.title, group=body.group, opening=body.content, mentions=body.mentions,
                        lineage={"exchange_id": exchange, "root_id": exchange, "origin": "user"})
                return p.store.command("operator", key, "preview.operator_thread", body.model_dump(), apply)
            return p.coordinator.operator_thread(body.participants, body.content, key, title=body.title, group=body.group)

    @app.post("/api/preview/threads/{thread}/messages")
    async def send(thread: str, body: MessageInput, request: Request):
        p = app.state.preview
        async with p.mutations:
            return p.operator_send(thread, body, p.command_id(request))

    @app.post("/api/preview/residents/{actor}/initiate")
    async def initiate(actor: str, body: ThreadInput, request: Request):
        p = app.state.preview
        async with p.mutations:
            return p.initiate(actor, body, p.command_id(request))

    @app.post("/api/preview/residents/{actor}/reply")
    async def reply(actor: str, body: ReplyInput, request: Request):
        p = app.state.preview
        async with p.mutations:
            return await p.reply(actor, body, p.command_id(request))

    @app.post("/api/preview/residents/{actor}/availability")
    async def availability(actor: str, body: AvailabilityInput):
        p = app.state.preview
        async with p.mutations:
            p.registry.resolve(actor)
            return p.coordinator.broker.set_available(actor, body.available)

    @app.post("/api/preview/world")
    async def world(body: WorldInput, request: Request):
        p = app.state.preview
        async with p.mutations:
            p.registry.resolve(body.owner)
            key = p.command_id(request)
            def apply():
                return p.coordinator.world.create(body.owner, key + ":object", uid("preview-object"), "object", body.name, visibility=body.visibility)
            return p.store.command(body.owner, key, "preview.world_create", body.model_dump(), apply)

    @app.post("/api/preview/scheduling-demo")
    async def schedule(body: Input, request: Request):
        p = app.state.preview
        async with p.mutations:
            return await p.scheduling_demo(p.command_id(request))

    return app
