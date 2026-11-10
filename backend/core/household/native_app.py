"""Native household gateway serving the ordinary TamAGI interface."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, nullcontext
import hmac
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
import yaml

from backend.auth import hash_password, load_session_secret, verify_password
from backend.core.runtime.interactions import AdmissionError
from .contracts import AccessDenied, Conflict, NotFound, ProtocolError
from .native_host import NativeHost
from .native_registry import NativeRegistry
from .native_service import NativeInteractions
from .model_server import NativeModelServer
from .store import uid


SOURCE = Path(__file__).resolve().parents[3]
FRONTEND = SOURCE / "frontend"


def create_household_app(config, *, provider=None, registry=None):
    if config.household.provider_mode == "offline" and provider is None:
        raise ValueError("Offline households require an explicitly injected test provider")
    registry = registry or NativeRegistry(config.household.root)
    store, hosts = registry.store, {}

    def worker_auth(request):
        token = request.headers.get("x-tamagi-worker", "")
        for actor, host in hosts.items():
            if hmac.compare_digest(token, host.token):
                return actor
        raise AccessDenied("A current internal resident credential is required")

    model = NativeModelServer(store, provider or config.llm, worker_auth)
    coordinator_socket = registry.root / "ipc" / "c.sock"
    for actor, descriptor in registry.descriptors.items():
        hosts[actor] = NativeHost(registry, descriptor, coordinator_socket)
        model.broker.register_resident(actor)
    manager = NativeInteractions(registry, model, hosts)
    from .native_scheduler import NativeWorldScheduler
    scheduler = NativeWorldScheduler(manager, config)
    model.grant_lookup = manager.grant
    from .native_world import NativeWorld
    world = NativeWorld(registry, model)
    from .native_messaging import NativeMessaging
    messaging = NativeMessaging(registry, model, manager)
    from .native_artifacts import NativeArtifacts
    artifacts = NativeArtifacts(registry, model)
    from .browser_recovery import BrowserRecovery
    recovery = BrowserRecovery(registry, manager, model, hosts, scheduler=scheduler)
    async def lost(actor):
        model.broker.set_available(actor, False)
        for record in manager.records():
            if record["data"]["status"] in {"queued", "running"} and any(c["resident_id"] == actor for c in record["data"].get("children", [])):
                manager.stop(record["id"])
        for record in store.list("native_execution"):
            if record["data"]["resident_id"] == actor:
                execution = model.broker.execution(record["data"]["broker_execution_id"])
                if execution["status"] == "running":
                    model.broker.stop_execution(record["data"]["broker_execution_id"])
    for host in hosts.values():
        host.lost = lost
    from .native_creation import NativeCreation
    creation = NativeCreation(registry, recovery, model, hosts, world, coordinator_socket, lost=lost)

    @asynccontextmanager
    async def lifespan(app):
        if coordinator_socket.exists():
            if not coordinator_socket.is_socket():
                raise RuntimeError("Coordinator socket path is occupied by a non-socket")
            coordinator_socket.unlink()
        server = uvicorn.Server(uvicorn.Config(app, uds=str(coordinator_socket), lifespan="off",
                                               log_level="warning", access_log=False))
        # The foreground public server owns process signals. The private
        # listener must not replace its Ctrl+C/shutdown handler.
        server.capture_signals = nullcontext
        serving = asyncio.create_task(server.serve())
        try:
            while not server.started:
                if serving.done():
                    await serving
                    raise RuntimeError("Private coordinator server did not start")
                await asyncio.sleep(.01)
            for host in hosts.values():
                await host.start()
            manager.start()
            scheduler.start()
            recovery.start()
            yield
        finally:
            await scheduler.close()
            await recovery.close()
            await manager.close()
            for host in hosts.values():
                await host.close()
            await model.close()
            server.should_exit = True
            await serving
            registry.close()

    app = FastAPI(title="TamAGI household", lifespan=lifespan)
    app.state.household = registry
    app.state.interactions, app.state.model, app.state.hosts = manager, model, hosts
    app.state.world_scheduler = scheduler
    app.state.recovery = recovery
    app.state.creation = creation
    session_root = registry.root / "session"
    session_root.mkdir(mode=0o700, exist_ok=True)

    @app.middleware("http")
    async def authentication(request, call_next):
        path = request.url.path
        if path.startswith("/_native/"):
            # Even on the public listener a browser session cannot authorize IPC.
            try:
                worker_auth(request)
            except AccessDenied as exc:
                return JSONResponse({"detail": str(exc)}, status_code=401)
        elif not (path in {"/login", "/login.html", "/favicon.ico"} or path.startswith("/api/auth/") or path.startswith("/assets/")):
            if not request.session.get("authenticated"):
                return JSONResponse({"detail": "Authentication required"}, status_code=401) if "/api/" in path else RedirectResponse("/login", status_code=303)
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not path.startswith("/_native/"):
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Same-origin request required"}, status_code=403)
        recovery_path = path.removeprefix("/")
        if recovery_path.startswith("r/"):
            recovery_path = recovery_path.split("/", 2)[-1]
        tracked = not path.startswith("/_native/") and ("/api/" in path or path.startswith(("/r/", "/panels/"))) and not recovery.handles(recovery_path) and not path.startswith("/api/auth/") and path != "/api/household/residents"
        if tracked and recovery.paused:
            return JSONResponse({"detail": "Household paused briefly for resident recovery; please try again shortly"}, status_code=503,
                                headers={"Retry-After": "2"})
        if tracked:
            recovery.inflight += 1
        try:
            return await call_next(request)
        except (AccessDenied, Conflict, NotFound, ProtocolError, AdmissionError) as exc:
            status = exc.status if isinstance(exc, AdmissionError) else 403 if isinstance(exc, AccessDenied) else 404 if isinstance(exc, NotFound) else 409 if isinstance(exc, Conflict) else 400
            return JSONResponse({"detail": str(exc)}, status_code=status)
        finally:
            if tracked:
                recovery.inflight -= 1

    # SessionMiddleware must be outermost so authentication can inspect session.
    app.add_middleware(SessionMiddleware, secret_key=load_session_secret(str(session_root)),
                       session_cookie="tamagi_household_" + registry.household_id[-12:], same_site="strict")

    @app.exception_handler(AccessDenied)
    @app.exception_handler(Conflict)
    @app.exception_handler(NotFound)
    @app.exception_handler(ProtocolError)
    @app.exception_handler(AdmissionError)
    async def errors(request, exc):
        status = exc.status if isinstance(exc, AdmissionError) else 403 if isinstance(exc, AccessDenied) else 404 if isinstance(exc, NotFound) else 409 if isinstance(exc, Conflict) else 400
        return JSONResponse({"detail": str(exc)}, status_code=status)

    @app.get("/api/auth/status")
    async def auth_status(request: Request):
        return {"enabled": True, "has_password": bool(config.auth.password_hash),
                "authenticated": bool(request.session.get("authenticated"))}

    @app.post("/api/auth/{action}")
    async def auth_action(action: str, request: Request):
        if action == "logout":
            request.session.clear()
            return {"status": "ok"}
        body = await request.json()
        password = body.get("password")
        if not isinstance(password, str) or len(password) > 1024:
            raise HTTPException(400, "Enter a valid password")
        if action == "setup":
            if config.auth.password_hash:
                raise HTTPException(403, "Password already set")
            if len(password) < 8:
                raise HTTPException(400, "Password must be at least 8 characters")
            config.auth.password_hash = hash_password(password)
            path = registry.root / "config.yaml"
            saved = yaml.safe_load(path.read_text())
            saved["auth"]["password_hash"] = config.auth.password_hash
            temporary = path.with_suffix(".yaml.new")
            temporary.write_text(yaml.safe_dump(saved, sort_keys=False))
            temporary.chmod(0o600)
            temporary.replace(path)
        elif action == "login":
            if not config.auth.password_hash or not verify_password(password, config.auth.password_hash):
                raise HTTPException(401, "Invalid password")
        else:
            raise HTTPException(404, "Unknown authentication operation")
        request.session["authenticated"] = True
        return {"status": "ok"}

    def bootstrap():
        preferences = store.meta("preferences", {"colors": {}})
        def avatar(descriptor):
            path = Path(descriptor["root"]) / "data" / "sprite_config.json"
            try:
                if path.is_symlink():
                    return descriptor.get("avatar", "vex")
                value = json.loads(path.read_text()).get("avatar")
                return value if value in {"ember", "echo", "vex", "custom"} else descriptor.get("avatar", "vex")
            except (OSError, ValueError, AttributeError):
                return descriptor.get("avatar", "vex")
        return {"household_id": registry.household_id, "operator": registry.manifest["operator"],
                "resident_recovery_available": True,
                "residents": [{"id": actor, "name": d["name"], "avatar": avatar(d),
                               "available": host.process is not None and host.process.returncode is None}
                              for actor, d in registry.descriptors.items() for host in [hosts[actor]]],
                "default_resident_id": registry.manifest["default_resident_id"], "preferences": preferences,
                "unread_count": manager.conversations.unread("operator")["count"],
                "addressed_thread_count": sum(bool(t["addressed_count"]) for t in manager.conversations.unread("operator")["threads"])}

    def operator_work(record):
        conversation = store.get("conversation", record["data"].get("conversation_id"))
        return bool(conversation and "operator" in conversation["data"]["participants"])

    @app.get("/api/household/creation")
    async def creation_options():
        return creation.schema()

    @app.post("/api/household/residents", status_code=201)
    async def create_resident(request: Request):
        raw = await request.body()
        if len(raw) > 32 * 1024:
            raise ProtocolError("Choose shorter starting notes for your new TamAGI")
        try:
            body = json.loads(raw)
        except ValueError:
            raise ProtocolError("Choose valid creation details") from None
        return await creation.create(body)

    def resident(request, bound=None):
        return registry.resolve(bound or request.headers.get("x-tamagi-resident") or request.query_params.get("resident_id"))["resident_id"]

    def history(identity):
        snap = manager.conversations.snapshot("operator", identity)
        return {"id": identity, **snap["conversation"]["data"], "snapshot_id": snap["snapshot_id"],
                "watermark": snap["watermark"], "messages": [manager.ui_message(m) for m in snap["messages"]],
                "invitations": [messaging.card(r) for r in store.list("conversation_invitation")
                                if r["data"]["conversation_id"] == identity]}

    def threads(actor=None, unread=False):
        unseen = {r["conversation_id"]: r for r in manager.conversations.unread("operator")["threads"]}
        output = []
        for record in reversed(store.list("conversation")):
            d = record["data"]
            if "operator" not in d["participants"] or (actor and actor not in d["participants"]) or (unread and record["id"] not in unseen):
                continue
            output.append({"id": record["id"], **d, "created_at": record["created"], "updated_at": record["updated"],
                           "message_count": d["last_sequence"], "unread_count": unseen.get(record["id"], {}).get("count", 0),
                           "addressed_count": unseen.get(record["id"], {}).get("addressed_count", 0),
                           "mention_count": unseen.get(record["id"], {}).get("mention_count", 0)})
        return output

    app.include_router(model.router)

    @app.post("/_native/artifacts")
    async def private_artifacts(request: Request):
        actor = worker_auth(request)
        raw = await request.body()
        if len(raw) > 6 * 1024 * 1024:
            raise ProtocolError("Artifact request is too large")
        try:
            body = json.loads(raw)
        except ValueError:
            raise ProtocolError("Choose a valid artifact request") from None
        if not isinstance(body, dict):
            raise ProtocolError("Choose an artifact operation")
        return artifacts.dispatch(actor, body)

    @app.get("/api/household/artifacts")
    async def public_artifacts():
        return {"artifacts": artifacts.listed()}

    @app.get("/api/household/artifacts/{identity}/versions/{version}/files/{path:path}")
    async def artifact_file(identity: str, version: int, path: str):
        from .native_artifacts import verified_files
        files = verified_files(artifacts.version(identity, version)["data"]["files"])
        if path not in files:
            raise NotFound("No such published file")
        return Response(files[path], media_type="application/octet-stream", headers={
            "Content-Disposition": "attachment", "X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"})

    @app.post("/_native/world")
    async def shared_world(request: Request):
        return world.dispatch(worker_auth(request), await request.json())

    @app.get("/_native/contacts")
    async def contacts(request: Request):
        return messaging.contacts(worker_auth(request))

    @app.post("/_native/contact")
    async def contact(request: Request):
        return messaging.contact(worker_auth(request), await request.json())

    def publish_note(actor, text, sources, command):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 16000:
            raise ProtocolError("Write a note of 1–16000 characters")
        if not isinstance(sources, list) or len(sources) > 32 or any(not isinstance(s, str) or not 1 <= len(s) <= 256 for s in sources):
            raise ProtocolError("Choose bounded explicit source IDs")
        def apply():
            record = store.put("shared_note", uid("note"), {"text": text, "sources": sources, "author": actor})
            store.event("notebook_changed", actor, {"note_id": record["id"]})
            return {**record, "actor": actor}
        return store.command(actor, command, "native.notebook.publish", {"text": text, "sources": sources}, apply)

    @app.post("/_native/notebook")
    async def notebook(request: Request):
        actor = worker_auth(request)
        body = await request.json()
        if body.get("action") == "inspect":
            return [{**r, "actor": r["data"]["author"]} for r in store.list("shared_note")]
        if body.get("action") != "publish":
            raise ProtocolError("Unknown notebook operation")
        _, execution, _ = model._active(actor, body["local_execution_id"])
        return publish_note(execution["actor_id"], body["text"], body.get("source_ids") or [], body["command_id"])

    @app.post("/api/runtime/notebook")
    async def operator_note(request: Request):
        body = await request.json()
        return publish_note("operator", body.get("text"), [], body.get("request_id") or uid("operator-note"))

    @app.post("/_native/conversations/commit")
    async def commit(request: Request):
        return manager.commit(worker_auth(request), await request.json())

    @app.get("/_native/conversations/{identity}")
    async def private_history(identity: str, request: Request, grant_id: str = "", through: int | None = None):
        actor = worker_auth(request)
        conv = manager.conversations._conversation(identity, actor)
        watermark = conv["data"]["last_sequence"]
        if grant_id:
            grant = manager.grant(grant_id, actor)
            if grant.get("kind") == "invitation":
                raise AccessDenied("An invitation card does not authorize transcript access")
            if grant.get("conversation_id") == identity:
                watermark = min(watermark, grant["through"])
        if through is not None:
            watermark = min(watermark, through)
        return {"conversation": conv, "messages": manager.conversations.history(actor, identity, through=watermark), "watermark": watermark}

    @app.get("/_native/conversations/{identity}/attachments/{message_id}")
    async def private_attachment(identity: str, message_id: str, request: Request, grant_id: str):
        actor = worker_auth(request)
        grant = manager.grant(grant_id, actor)
        if grant.get("kind") == "invitation":
            raise AccessDenied("An invitation card does not authorize attachment access")
        through = grant["through"] if grant["conversation_id"] == identity else None
        return manager.attachment(actor, identity, message_id, through)

    @app.get("/api/household")
    async def household():
        return bootstrap()

    @app.get("/api/household/broker")
    async def broker_state():
        return model.broker.state()

    @app.get("/api/household/world")
    async def world_schedule():
        return scheduler.status()

    @app.post("/api/world/tick", status_code=202)
    async def world_tick():
        return await scheduler.opportunity(manual=True)

    @app.post("/api/household/world/{identity}/stop")
    async def stop_world(identity: str):
        if manager.get(identity)["data"]["kind"] != "world":
            raise HTTPException(400, "Choose World Thread work")
        return manager.stop(identity)

    @app.put("/api/household/preferences")
    async def preferences(request: Request):
        import re
        colors = (await request.json()).get("colors")
        participants = {p["id"] for p in registry.participants()}
        if not isinstance(colors, dict) or any(k not in participants or not isinstance(v, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", v) for k, v in colors.items()):
            raise HTTPException(400, "Choose valid participant colors")
        result = {"colors": colors}
        store.set_meta("preferences", result)
        return {"preferences": result, **result}

    @app.get("/api/household/threads")
    async def household_threads(resident_id: str | None = None, unread: int = 0):
        if resident_id:
            registry.resolve(resident_id)
        return {"threads": threads(resident_id, bool(unread)), "unread_count": manager.conversations.unread("operator")["count"]}

    @app.post("/api/household/threads/{identity}/read")
    async def read(identity: str, request: Request):
        body = await request.json()
        manager.conversations.mark_read("operator", identity, body["watermark"], uid("read"), snapshot_id=body["snapshot_id"])
        return {"unread_count": manager.conversations.unread("operator")["count"]}

    @app.post("/api/household/threads/{identity}/participants")
    async def participants(identity: str, request: Request):
        body = await request.json()
        conv = manager.conversations._conversation(identity, "operator")
        desired = body.get("participants")
        if not isinstance(desired, list) or any(p not in hosts for p in desired):
            raise HTTPException(400, "Choose household residents")
        if any(p != "operator" and p not in desired for p in conv["data"]["participants"]):
            raise Conflict("Participants choose whether to leave their shared thread")
        def apply():
            if all(actor in conv["data"]["participants"] for actor in desired):
                return {"conversation_id": identity, "invitations": [], "notice": "Those residents have already joined."}
            active = next((r for r in manager.records() if r["data"]["conversation_id"] == identity
                           and r["data"]["status"] in {"queued", "running"}), None)
            exchange = active["data"]["exchange_id"] if active else uid("invitation-exchange")
            if not active:
                model.broker.open_exchange(exchange, "autonomy")
            source = manager.conversations.history("operator", identity)[-1]
            return messaging.invite_members("operator", identity, desired,
                "The operator invites you to join this existing conversation.", command,
                {"exchange_id": exchange, "origin": model.broker.exchange(exchange)["origin"]}, trigger=source)
        command = body.get("request_id") or uid("operator-invite")
        result = store.command("operator", command, "native.operator.invite", {"conversation_id": identity, "participants": desired}, apply)
        return {"conversation": history(identity), **result}

    @app.get("/api/conversations")
    async def conversations(request: Request):
        return {"conversations": threads(resident(request))}

    @app.get("/api/conversations/{identity}")
    async def conversation(identity: str):
        return history(identity)

    @app.get("/api/conversations/{identity}/attachments/{message_id}")
    async def attachment(identity: str, message_id: str):
        import base64
        source = manager.attachment("operator", identity, message_id)
        return Response(base64.b64decode(source["image_data"], validate=True), media_type=source["media_type"],
                        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"})

    @app.post("/api/chat/requests", status_code=202)
    async def submit(request: Request):
        body = await request.json()
        body["resident_id"] = resident(request)
        return await manager.submit(body, body.pop("request_id", None))

    @app.get("/api/chat/requests")
    async def requests(request: Request):
        actor = resident(request)
        return {"requests": [manager.public(r) for r in manager.records(50)
                              if actor in [r["data"]["resident_id"], *r["data"]["responders"]]
                              and operator_work(r)]}

    @app.get("/api/chat/requests/{identity}")
    async def work(identity: str, after: int = 0):
        manager.conversations._conversation(manager.get(identity)["data"]["conversation_id"], "operator")
        return manager.snapshot(identity, after)

    @app.post("/api/chat/requests/{identity}/stop")
    async def stop(identity: str):
        manager.conversations._conversation(manager.get(identity)["data"]["conversation_id"], "operator")
        return manager.stop(identity)

    @app.post("/api/chat/requests/{identity}/approval")
    async def approve(identity: str, request: Request):
        manager.conversations._conversation(manager.get(identity)["data"]["conversation_id"], "operator")
        body = await request.json()
        return manager.approve(identity, body["approval_id"], body.get("approved") is True, body.get("allow_always") is True)

    @app.websocket("/api/ws")
    async def websocket(socket: WebSocket):
        origin = socket.headers.get("origin")
        expected = ("https" if socket.url.scheme == "wss" else "http") + "://" + socket.headers.get("host", "")
        if not socket.session.get("authenticated") or (origin and origin != expected):
            await socket.close(code=1008)
            return
        await socket.accept()
        from backend.api.chat import observe_websocket
        class OperatorObserver:
            observers = manager.observers
            def subscribe(self):
                return manager.subscribe()
            def get(self, identity):
                record = manager.get(identity)
                manager.conversations._conversation(record["data"]["conversation_id"], "operator")
                return record
            def records(self, limit=50):
                return [r for r in manager.records(limit) if operator_work(r)]
            def public(self, record):
                return manager.public(record)
            def snapshot(self, identity, after=0):
                self.get(identity)
                return manager.snapshot(identity, after)
            def stop(self, identity):
                self.get(identity)
                return manager.stop(identity)
            def approve(self, identity, *args):
                self.get(identity)
                return manager.approve(identity, *args)
            async def submit(self, *args):
                raise AdmissionError("Use the resident-scoped HTTP chat admission endpoint")
        await observe_websocket(socket, OperatorObserver())

    async def proxy(request, path, actor):
        if path.startswith(("/", "\\")) or "\\" in path or any(part in {".", ".."} for part in path.split("/")):
            raise HTTPException(404, "Invalid resident route")
        if path.startswith(("_native/", "api/chat", "api/conversations")):
            raise HTTPException(404, "This route belongs to the household coordinator")
        if recovery.handles(path):
            return await recovery.route(actor, path, request)
        host = hosts[actor]
        if path == "api/world/tick" and request.method == "POST":
            return JSONResponse(await scheduler.opportunity(manual=True), status_code=202)
        if path.startswith(("api/backup", "api/restore", "api/deploy", "api/transfer")):
            raise HTTPException(409, "Household deployment and shared-state recovery remain unavailable")
        if path == "api/config" and request.method != "GET":
            raise HTTPException(409, "Household provider configuration belongs to the coordinator")
        headers = {key: value for key, value in request.headers.items()
                   if key.lower() not in {"host", "cookie", "origin", "content-length", "authorization"}
                   and not key.lower().startswith("x-tamagi-")}
        if request.method not in {"GET", "HEAD", "OPTIONS"} and path != "api/world/tick":
            grant_id, exchange = uid("grant"), uid("exchange")
            model.broker.open_exchange(exchange, "user")
            store.put("native_grant", grant_id, {"id": grant_id, "resident_id": actor,
                "exchange_id": exchange, "origin": "user", "first_response": False, "status": "running", "purpose": path})
            headers["X-Tamagi-Grant"] = grant_id
        result = await host.request(request.method, "/" + path + ("?" + request.url.query if request.url.query else ""),
                                    content=await request.body(), headers=headers)
        content = result.content
        content_type = result.headers.get("content-type", "")
        if "text/html" in content_type and not path.startswith("api/"):
            return HTMLResponse(inject(content.decode()), status_code=result.status_code)
        return Response(content, status_code=result.status_code,
                        headers={k: v for k, v in result.headers.items() if k.lower() in {"content-type", "content-disposition", "cache-control", "content-security-policy", "x-content-type-options"}})

    def inject(content):
        encoded = json.dumps(bootstrap(), ensure_ascii=True).replace("<", "\\u003c")
        return content.replace("<head>", "<head><script>window.TAMAGI_HOUSEHOLD_BOOTSTRAP=" + encoded + ";</script><script src='/assets/household-ui.js?v=6'></script>", 1)

    @app.api_route("/r/{actor}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])
    async def scoped(actor: str, path: str, request: Request):
        actor = registry.resolve(actor)["resident_id"]
        # Scoped canonical endpoints must never reach a worker's private cache.
        if path == "api/conversations" and request.method == "GET":
            return {"conversations": threads(actor)}
        if path.startswith("api/conversations/") and request.method == "GET":
            return history(path.split("/")[-1])
        if path == "api/runtime/notebook" and request.method == "POST":
            return await operator_note(request)
        if path == "api/chat/requests":
            if request.method == "GET":
                return {"requests": [manager.public(r) for r in manager.records(50)
                    if actor in [r["data"]["resident_id"], *r["data"]["responders"]]
                    and operator_work(r)]}
            if request.method == "POST":
                body = await request.json()
                body["resident_id"] = actor
                return JSONResponse(await manager.submit(body, body.pop("request_id", None)), status_code=202)
        if path.startswith("api/chat/requests/"):
            parts = path.split("/")
            if len(parts) == 4 and request.method == "GET":
                return await work(parts[3], int(request.query_params.get("after", 0)))
            if len(parts) == 5 and request.method == "POST":
                if parts[4] == "stop":
                    return await stop(parts[3])
                if parts[4] == "approval":
                    return await approve(parts[3], request)
        return await proxy(request, path, actor)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])
    async def fallback(path: str, request: Request):
        if path in {"login", "login.html"}:
            return FileResponse(FRONTEND / "login.html")
        if path in {"", "index.html"}:
            return HTMLResponse(inject((FRONTEND / "index.html").read_text()))
        if path.startswith("assets/") or path in {"manifest.json", "sw.js", "favicon.ico"}:
            file = (FRONTEND / path).resolve()
            if not file.is_relative_to(FRONTEND) or not file.is_file():
                raise HTTPException(404)
            return FileResponse(file)
        return await proxy(request, path, resident(request))

    return app
