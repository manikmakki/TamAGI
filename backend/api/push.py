"""
Push Notification API — VAPID key distribution and subscription management.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("tamagi.api.push")

router = APIRouter(prefix="/api/push", tags=["push"])

# Set by main.py lifespan (same pattern as set_agent / set_dream_engine)
_push_service = None


def set_push_service(service) -> None:
    global _push_service
    _push_service = service


def get_push_service():
    if _push_service is None:
        raise HTTPException(status_code=503, detail="Push service not initialized")
    return _push_service


# ── Pydantic Models ────────────────────────────────────────────────────────────

class PushSubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class PushSubscriptionPayload(BaseModel):
    endpoint: str
    keys: PushSubscriptionKeys
    expirationTime: Optional[float] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/vapid-public-key")
async def get_vapid_public_key():
    """
    Return the VAPID public key in URL-safe base64 (no padding).
    The frontend passes this to pushManager.subscribe() as applicationServerKey.
    """
    svc = get_push_service()
    if not svc.enabled:
        raise HTTPException(status_code=503, detail="Push notifications not available")
    return {"publicKey": svc.public_key}


@router.post("/subscribe")
async def subscribe(payload: PushSubscriptionPayload):
    """
    Register a push subscription from the browser.
    Overwrites any previously stored subscription (single-user model).
    """
    svc = get_push_service()
    svc.save_subscription(payload.model_dump())
    return {"status": "subscribed"}


@router.delete("/subscribe")
async def unsubscribe():
    """Remove the stored push subscription."""
    svc = get_push_service()
    svc.delete_subscription()
    return {"status": "unsubscribed"}


@router.get("/status")
async def push_status():
    """
    Return current push notification state.
    Used by the frontend bell icon to show current subscription status on page load.
    """
    svc = get_push_service()
    return {
        "enabled": svc.enabled,
        "subscribed": svc.has_subscription,
    }


@router.post("/test")
async def test_push():
    """
    Send a test push notification immediately. Returns detailed error on failure.
    Useful for verifying the backend-to-browser push path works end-to-end.
    """
    svc = get_push_service()
    if not svc.enabled:
        raise HTTPException(status_code=503, detail="Push notifications not available (pywebpush not installed)")
    if not svc.has_subscription:
        raise HTTPException(status_code=400, detail="No push subscription registered — enable notifications in the browser first")
    try:
        # Call synchronously here so we can capture and return any error
        svc._send_sync(
            title="TamAGI Test",
            body="Push notifications are working!",
            url="/",
            tag="tamagi-test",
        )
        return {"status": "sent"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
