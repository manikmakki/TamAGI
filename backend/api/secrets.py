"""
Secrets management API.

Provides endpoints to store and delete secrets without ever returning
their values.  Secret *values* are write-only from the API's perspective —
only names are exposed in list/get responses.

Endpoints
---------
GET    /api/secrets          — list known secret names (file-backed only;
                               keyring-backed names are not enumerable)
POST   /api/secrets/{name}   — create or update a secret
DELETE /api/secrets/{name}   — delete a secret
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.secrets import get_secret_store

logger = logging.getLogger("tamagi.api.secrets")
router = APIRouter(prefix="/api/secrets", tags=["secrets"])


class SecretPayload(BaseModel):
    value: str


class SecretListResponse(BaseModel):
    names: list[str]


@router.get("", response_model=SecretListResponse)
async def list_secrets():
    """Return the names of all file-backed secrets (values are never exposed)."""
    store = get_secret_store()
    return SecretListResponse(names=store.list_names())


@router.post("/{name}", status_code=204)
async def set_secret(name: str, payload: SecretPayload):
    """
    Store or update a secret value.
    The value is accepted exactly once and is never returned by the API.
    """
    if not name.strip():
        raise HTTPException(status_code=422, detail="Secret name must not be empty")
    store = get_secret_store()
    store.set(name, payload.value)
    # Deliberately no body — the value must not appear in any response.


@router.delete("/{name}", status_code=204)
async def delete_secret(name: str):
    """Delete a secret by name."""
    store = get_secret_store()
    existed = store.delete(name)
    if not existed:
        raise HTTPException(status_code=404, detail=f"Secret '{name}' not found")
