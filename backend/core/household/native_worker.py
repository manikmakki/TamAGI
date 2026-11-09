"""Private Unix-socket worker running the ordinary TamAGI application and tools."""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
from pathlib import Path

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from .contracts import WorkerBinding
from .worker_bridge import admission_context, validate_native_binding


class RespondInput(BaseModel):
    grant: dict
    snapshot: dict
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    payload: dict = Field(default_factory=dict)


class NativeAuthentication:
    """Only an internal bearer can reach any ordinary or internal worker endpoint."""

    def __init__(self, app, token, session_secret):
        from itsdangerous import TimestampSigner

        self.app, self.token = app, token
        self.signer = TimestampSigner(session_secret)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        provided = headers.get(b"x-tamagi-worker", b"").decode("latin1")
        authorization = headers.get(b"authorization", b"").decode("latin1")
        if not provided and authorization.startswith("Bearer "):
            provided = authorization[7:]
        if not hmac.compare_digest(provided.encode("latin1"), self.token.encode("utf-8")):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"detail":"Internal resident authentication required"}'})
            return
        # The ordinary application still enforces its mutation/session rules.
        # Browser cookies are never accepted as internal worker credentials.
        session = base64.b64encode(json.dumps({"authenticated": True}).encode())
        cookie = b"tamagi_session=" + self.signer.sign(session)
        scope = dict(scope)
        scope["headers"] = [(key, value) for key, value in scope.get("headers", []) if key.lower() not in {b"cookie", b"origin"}]
        scope["headers"].append((b"cookie", cookie))
        grant = headers.get(b"x-tamagi-grant", b"").decode("latin1")
        variable = admission_context()
        marker = variable.set({"id": grant} if grant else None)
        try:
            await self.app(scope, receive, send)
        finally:
            variable.reset(marker)


def configure_app(app):
    """Install authenticated local transport before the normal lifespan starts."""
    from backend.auth import load_session_secret
    from backend.config import get_config

    original_route_count = len(app.router.routes)

    @app.post("/_native/respond", status_code=202)
    async def respond(body: RespondInput):
        from backend.api.chat import get_agent
        from backend.core.runtime.interactions import AdmissionError

        agent = get_agent()
        try:
            return await agent.runtime.interactions.submit_household(body.payload, body.grant, body.snapshot, body.request_id)
        except AdmissionError as error:
            raise HTTPException(error.status, str(error)) from error
        except (ValueError, PermissionError, LookupError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/_native/ready")
    async def ready():
        from backend.api.chat import get_agent

        agent = get_agent()
        return {"resident_id": agent.runtime.actor_id, "ready": True,
                "pending": agent.runtime.interactions.has_pending() or agent.runtime.root_lock.locked(),
                "activity": agent.runtime.activity_state(),
                "due": agent.runtime.due(ignore_decompression=True),
                "world_due": agent.runtime.due() and not getattr(agent.identity, "needs_onboarding", False)}

    @app.post("/_native/wake")
    async def shared_event(body: dict):
        from backend.api.chat import get_agent
        import math

        runtime = get_agent().runtime
        identity, name, created = body.get("id"), body.get("kind"), body.get("created")
        if (not isinstance(identity, str) or not 1 <= len(identity) <= 256
                or name not in {"world_changed", "notebook_changed"}
                or type(created) not in {int, float} or not math.isfinite(created)):
            raise HTTPException(400, "Invalid canonical wake event")
        with runtime.store.transaction():
            if not runtime.store.get("household_event", identity):
                activity = runtime.store.get("activity", runtime.actor_id)
                # A late delivery must not satisfy a wait chosen after the event.
                if activity["updated"] <= created:
                    runtime.wake(name)
                runtime.store.put("household_event", identity, runtime.actor_id, body, category="platform")
        return {"received": identity}

    @app.get("/_native/work/{work_id}")
    async def work(work_id: str, after: int = 0):
        from backend.api.chat import interactions

        return interactions().snapshot(work_id, after)

    @app.post("/_native/work/{work_id}/stop")
    async def stop(work_id: str):
        from backend.api.chat import interactions

        return interactions().stop(work_id)

    @app.post("/_native/work/{work_id}/approval")
    async def approval(work_id: str, body: dict):
        from backend.api.chat import interactions

        return interactions().approve(work_id, body.get("approval_id"), body.get("approved") is True, body.get("allow_always") is True)

    secret = load_session_secret(str(Path(get_config().history.persist_path).parent))
    # The ordinary SPA's catch-all is already registered at import time.
    # Internal routes must precede it rather than return the application HTML.
    internal_routes = app.router.routes[original_route_count:]
    del app.router.routes[original_route_count:]
    app.router.routes[0:0] = internal_routes
    app.add_middleware(NativeAuthentication, token=os.environ["TAMAGI_NATIVE_TOKEN"], session_secret=secret)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    arguments = parser.parse_args()
    binding = WorkerBinding.parse(json.loads(os.environ["TAMAGI_NATIVE_BINDING"]))
    validate_native_binding(binding, arguments.config)
    socket = arguments.socket.absolute()
    household = Path(binding.root).parent.parent
    if socket != socket.resolve() or not socket.is_relative_to(household / "ipc") or socket.exists():
        parser.error("Worker socket must be a new canonical path in its household ipc directory")
    token = os.environ.get("TAMAGI_NATIVE_TOKEN", "")
    if len(token) < 32:
        parser.error("Worker authentication token is missing")
    os.environ["TAMAGI_CONFIG"] = str(arguments.config)
    from backend.config import load_config, set_config

    config = load_config(arguments.config)
    if Path(config.runtime.instance_dir).resolve() != Path(binding.root):
        parser.error("Worker configuration selects another resident root")
    config.runtime.automatic_deployment = False
    set_config(config)
    from backend.main import app
    import uvicorn

    configure_app(app)
    uvicorn.run(app, uds=str(socket), log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
