"""
Auth API — login, logout, status, and first-time password setup.

All endpoints are public (exempt from the auth middleware) so the login
page itself can reach them without a session.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.auth import hash_password, verify_password
from backend.config import get_config

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Helpers ──────────────────────────────────────────────────────────────────


def _persist_auth_config(password_hash: str) -> None:
    """
    Write the new password_hash into config.yaml using a regex line-replacement
    so that all comments, ordering, and formatting are preserved exactly.

    Finds the first `password_hash:` line (anywhere in the file) and replaces
    only its value. If no such line exists, appends an `auth:` section.
    """
    config_path = Path(os.environ.get("TAMAGI_CONFIG", "config.yaml"))
    content = config_path.read_text() if config_path.exists() else ""

    # Replace just the value on the password_hash line, leaving everything else
    # (including inline comments and surrounding lines) completely untouched.
    new_line = f"  password_hash: {password_hash}"
    updated, n = re.subn(
        r"^[ \t]*password_hash:.*$",
        new_line,
        content,
        count=1,
        flags=re.MULTILINE,
    )

    if n == 0:
        # No existing password_hash line — append a minimal auth section
        updated = content.rstrip("\n") + f"\nauth:\n  enabled: true\n{new_line}\n"

    config_path.write_text(updated)

    # Keep the in-memory singleton in sync — no restart required
    get_config().auth.password_hash = password_hash


# ── Endpoints ─────────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    password: str


@router.get("/status")
async def auth_status(request: Request):
    """
    Return auth state: whether auth is enabled, whether a password has been
    set, and whether the current session is authenticated.

    The login page polls this to decide whether to show the login form or
    the first-time setup form.
    """
    config = get_config()
    return {
        "enabled": config.auth.enabled,
        "authenticated": bool(request.session.get("authenticated")),
        "has_password": bool(config.auth.password_hash),
    }


@router.post("/login")
async def login(request: Request, body: LoginRequest):
    """Verify the password and create an authenticated session."""
    config = get_config()

    # If auth is turned off, accept any login attempt transparently
    if not config.auth.enabled:
        request.session["authenticated"] = True
        return {"status": "ok"}

    if not config.auth.password_hash:
        raise HTTPException(status_code=400, detail="No password set — use /api/auth/setup first")

    if not verify_password(body.password, config.auth.password_hash):
        raise HTTPException(status_code=401, detail="Invalid password")

    request.session["authenticated"] = True
    return {"status": "ok"}


@router.post("/logout")
async def logout(request: Request):
    """Clear the session cookie."""
    request.session.clear()
    return {"status": "ok"}


@router.post("/setup")
async def setup(request: Request, body: LoginRequest):
    """
    First-time password setup.

    This endpoint is only open when no password has been set yet (password_hash
    is empty). Once a password exists it returns 403 — use a config file edit
    or a container restart with a cleared hash to reset.
    """
    config = get_config()

    if config.auth.password_hash:
        raise HTTPException(
            status_code=403,
            detail="Password already set. Clear auth.password_hash in config.yaml to reset.",
        )

    if not body.password or len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    new_hash = hash_password(body.password)
    _persist_auth_config(new_hash)

    # Log the user in immediately after setup
    request.session["authenticated"] = True
    return {"status": "ok"}
