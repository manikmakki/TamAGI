"""
Authentication utilities for TamAGI.

Provides password hashing (PBKDF2-HMAC-SHA256, stdlib only — no extra deps)
and a session-secret loader that persists the signing key across restarts.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path


def hash_password(password: str) -> str:
    """
    Hash a plaintext password and return a 'salt:hash' string.

    Uses PBKDF2-HMAC-SHA256 with a random 256-bit salt and 260 000 iterations
    (OWASP 2023 recommendation for PBKDF2-SHA256).
    """
    salt = secrets.token_hex(32)  # 256-bit random salt, hex-encoded
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        260_000,
    )
    return f"{salt}:{digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """
    Verify a plaintext password against a stored 'salt:hash' string.

    Uses hmac.compare_digest for constant-time comparison to prevent
    timing-based side-channel attacks.
    """
    try:
        salt, stored_hash = stored.split(":", 1)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            260_000,
        )
        return hmac.compare_digest(digest.hex(), stored_hash)
    except Exception:
        return False


def load_session_secret(data_dir: str = "./data") -> str:
    """
    Return the session-cookie signing secret, creating it on first boot.

    Stored in data/.session_secret rather than config.yaml so it is not
    accidentally committed to version control. The file is created with
    restricted permissions (0o600) on POSIX systems.
    """
    path = Path(data_dir) / ".session_secret"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        secret = path.read_text().strip()
        if secret:
            return secret

    # Generate a fresh 256-bit secret on first boot
    secret = secrets.token_hex(32)
    path.write_text(secret)
    try:
        path.chmod(0o600)  # owner read/write only (POSIX)
    except OSError:
        pass  # Windows — ignore

    return secret
