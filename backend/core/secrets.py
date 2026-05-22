"""
Local-first secret store.

Priority order:
  1. OS keyring — GNOME Keyring / KWallet on Linux, macOS Keychain, Windows
     Credential Vault.  Encrypted by the OS and tied to the user session.
  2. Fernet-encrypted JSON file — used when no keyring daemon is available
     (e.g. headless servers).  The encryption key is generated once with
     os.urandom and stored at data/.secrets.key (chmod 0600).

Config.yaml only stores *names*, never values:

    mcp:
      servers:
        - name: brave-search
          env:
            BRAVE_API_KEY:
              secret: BRAVE_API_KEY   # <-- name reference, not the value

Values are injected as subprocess env vars at MCP server spawn time.
They are never written into any LLM message.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("tamagi.secrets")

_KEYRING_SERVICE = "tamagi"


def _keyring_get(name: str) -> str | None:
    try:
        import keyring
        return keyring.get_password(_KEYRING_SERVICE, name)
    except Exception:
        return None


def _keyring_set(name: str, value: str) -> bool:
    try:
        import keyring
        keyring.set_password(_KEYRING_SERVICE, name, value)
        return True
    except Exception:
        return False


def _keyring_delete(name: str) -> bool:
    try:
        import keyring
        keyring.delete_password(_KEYRING_SERVICE, name)
        return True
    except Exception:
        return False


class SecretStore:
    """
    Local-first secret store.

    Uses OS keyring when available; falls back to a Fernet-encrypted file.
    Both storage paths keep secrets off disk in plaintext.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._data_dir = Path(data_dir)
        self._enc_path = self._data_dir / "secrets.enc"
        self._key_path = self._data_dir / ".secrets.key"
        self._fernet = None  # lazy-init on first file-store access

    # ── Fernet helpers ─────────────────────────────────────────────────────

    def _get_fernet(self):
        if self._fernet is not None:
            return self._fernet
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise RuntimeError(
                "cryptography package is required for the secret-store fallback. "
                "Install it with: pip install cryptography"
            )
        self._data_dir.mkdir(parents=True, exist_ok=True)
        if self._key_path.exists():
            key = self._key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            self._key_path.write_bytes(key)
            os.chmod(self._key_path, 0o600)
            logger.info("Secret store: generated new encryption key at %s", self._key_path)
        self._fernet = Fernet(key)
        return self._fernet

    def _file_load(self) -> dict[str, str]:
        if not self._enc_path.exists():
            return {}
        try:
            raw = self._get_fernet().decrypt(self._enc_path.read_bytes())
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Secret store: could not read secrets file: %s", exc)
            return {}

    def _file_save(self, data: dict[str, str]) -> None:
        encrypted = self._get_fernet().encrypt(json.dumps(data).encode())
        self._enc_path.write_bytes(encrypted)
        os.chmod(self._enc_path, 0o600)

    # ── Public API ─────────────────────────────────────────────────────────

    def get(self, name: str) -> str | None:
        """Return the secret value, or None if not found."""
        value = _keyring_get(name)
        if value is not None:
            return value
        return self._file_load().get(name)

    def set(self, name: str, value: str) -> None:
        """Store a secret.  Tries OS keyring first, falls back to encrypted file."""
        if _keyring_set(name, value):
            logger.info("Secret '%s' stored in OS keyring", name)
            return
        data = self._file_load()
        data[name] = value
        self._file_save(data)
        logger.info("Secret '%s' stored in encrypted file (no keyring backend)", name)

    def delete(self, name: str) -> bool:
        """Delete a secret.  Returns True if it existed in either store."""
        deleted_keyring = _keyring_delete(name)
        data = self._file_load()
        deleted_file = name in data
        if deleted_file:
            del data[name]
            self._file_save(data)
        return deleted_keyring or deleted_file

    def list_names(self) -> list[str]:
        """
        List secret names known to the encrypted file.
        Keyring contents are not enumerable — only file-backed names are returned.
        """
        return list(self._file_load().keys())

    def resolve_env(self, env_spec: dict[str, Any]) -> dict[str, str]:
        """
        Resolve an env-var spec dict from config into plain string values.

        Two forms are supported:

          PLAIN_VAR: "value"             # passed through as-is
          SECRET_VAR:                    # looked up in the secret store
            secret: SECRET_NAME
        """
        resolved: dict[str, str] = {}
        for key, spec in env_spec.items():
            if isinstance(spec, str):
                resolved[key] = spec
            elif isinstance(spec, dict) and "secret" in spec:
                secret_name = spec["secret"]
                value = self.get(secret_name)
                if value is None:
                    logger.warning(
                        "Secret '%s' (env var '%s') not found in store — "
                        "use POST /api/secrets/%s to set it",
                        secret_name, key, secret_name,
                    )
                else:
                    resolved[key] = value
            else:
                logger.warning("Unknown env spec for '%s': %r — skipping", key, spec)
        return resolved


# ── Global singleton ───────────────────────────────────────────────────────

_store: SecretStore | None = None


def get_secret_store(data_dir: str | Path = "data") -> SecretStore:
    global _store
    if _store is None:
        _store = SecretStore(data_dir)
    return _store
