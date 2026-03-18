"""
Push Notification Service — VAPID key management, subscription storage,
and Web Push dispatch via pywebpush.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tamagi.push")


class PushNotificationService:
    """
    Single-user Web Push notification service.

    Responsibilities:
    - Generate and persist VAPID key pair on first run (data/vapid_keys.json)
    - Load/save a single push subscription (data/push_subscription.json)
    - Send push notifications using pywebpush (fire-and-forget friendly)
    """

    def __init__(self, data_dir: str = "data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._vapid_path = self._data_dir / "vapid_keys.json"
        self._sub_path   = self._data_dir / "push_subscription.json"

        self._private_key: str = ""
        self._public_key: str  = ""   # URL-safe base64, no padding
        self._subscription: Optional[dict] = None
        self._enabled: bool = True    # set False if pywebpush is missing

        self._load_or_generate_vapid()
        self._load_subscription()

    # ── VAPID ──────────────────────────────────────────────────────────────────

    def _load_or_generate_vapid(self) -> None:
        if self._vapid_path.exists():
            try:
                data = json.loads(self._vapid_path.read_text())
                self._private_key = data["private_key"]   # PEM string
                self._public_key  = data["public_key"]    # URL-safe base64
                logger.info("VAPID keys loaded from %s", self._vapid_path)
                return
            except (KeyError, json.JSONDecodeError) as exc:
                logger.warning("VAPID keys file corrupt, regenerating: %s", exc)

        self._generate_and_save_vapid()

    def _generate_and_save_vapid(self) -> None:
        try:
            import base64
            from py_vapid import Vapid
            from cryptography.hazmat.primitives.serialization import (
                Encoding, PrivateFormat, PublicFormat, NoEncryption,
            )
        except ImportError:
            logger.error(
                "pywebpush / py_vapid not installed — push notifications disabled. "
                "Add 'pywebpush>=2.0.0' to requirements.txt and reinstall."
            )
            self._enabled = False
            return

        v = Vapid()
        v.generate_keys()

        private_pem = v.private_key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        ).decode()

        # Serialize as uncompressed EC point (04 || x || y, 65 bytes) then
        # URL-safe base64 without padding — this is what browsers expect for
        # pushManager.subscribe({ applicationServerKey: ... })
        public_bytes = v.public_key.public_bytes(
            Encoding.X962, PublicFormat.UncompressedPoint
        )
        public_b64 = base64.urlsafe_b64encode(public_bytes).rstrip(b"=").decode()

        self._private_key = private_pem
        self._public_key  = public_b64

        self._vapid_path.write_text(json.dumps({
            "private_key": private_pem,
            "public_key":  public_b64,
        }, indent=2))
        logger.info("VAPID key pair generated and saved to %s", self._vapid_path)

    @property
    def public_key(self) -> str:
        return self._public_key

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._public_key)

    # ── Subscription ───────────────────────────────────────────────────────────

    def _load_subscription(self) -> None:
        if self._sub_path.exists():
            try:
                self._subscription = json.loads(self._sub_path.read_text())
                logger.info("Push subscription loaded from disk")
            except json.JSONDecodeError:
                logger.warning("Push subscription file corrupt, ignoring")
                self._subscription = None

    def save_subscription(self, subscription: dict) -> None:
        """Persist a PushSubscription object sent from the browser."""
        self._subscription = subscription
        self._sub_path.write_text(json.dumps(subscription, indent=2))
        logger.info(
            "Push subscription saved (endpoint: %s...)",
            subscription.get("endpoint", "")[:60],
        )

    def delete_subscription(self) -> None:
        self._subscription = None
        if self._sub_path.exists():
            self._sub_path.unlink()
        logger.info("Push subscription removed")

    @property
    def has_subscription(self) -> bool:
        return self._subscription is not None

    # ── Send ───────────────────────────────────────────────────────────────────

    async def send_notification(
        self,
        title: str,
        body: str,
        url: str = "/",
        tag: str = "tamagi-response",
    ) -> bool:
        """
        Fire a Web Push notification. Returns True on success.

        The payload is JSON; the service worker push handler unpacks it
        via event.data.json().

        Error handling:
        - HTTP 410/404 from the push endpoint means the subscription is gone —
          delete it automatically (RFC 8030 requirement).
        - All other errors are logged but never propagated (best-effort delivery).
        - webpush() is blocking, so it runs in the thread pool to avoid stalling
          the event loop during the WebSocket response cycle.
        """
        if not self.enabled:
            logger.debug("Push notifications disabled — skipping")
            return False
        if not self._subscription:
            logger.debug("No push subscription registered — skipping")
            return False

        try:
            import asyncio
            from functools import partial
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(self._send_sync, title=title, body=body, url=url, tag=tag),
            )
            logger.info("Push notification sent: %s", title)
            return True

        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (404, 410):
                logger.info(
                    "Push subscription expired (HTTP %s) — removing", status
                )
                self.delete_subscription()
            else:
                logger.warning("Push notification failed: %s", exc)
            return False

    def _send_sync(self, title: str, body: str, url: str, tag: str) -> None:
        """Synchronous pywebpush call — executed inside run_in_executor."""
        import json as _json
        from pywebpush import webpush

        payload = _json.dumps({
            "title": title,
            "body":  body,
            "url":   url,
            "tag":   tag,
        })

        webpush(
            subscription_info=self._subscription,
            data=payload,
            vapid_private_key=self._private_key,
            vapid_claims={
                # sub must be a mailto: or https: URI per the VAPID spec
                "sub": "mailto:tamagi@localhost",
            },
            content_encoding="aes128gcm",   # RFC 8291 modern encryption
        )
