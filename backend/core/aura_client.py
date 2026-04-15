"""
AURA client for TamAGI — Phase 3

Wraps AURA's HTTP API so TamAGI can forward chat traffic to AURA's reasoning
engine and surface deliberation/approval state in the TamAGI UI.

Used only when config.brain.mode == "aura".
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.config import BrainConfig

logger = logging.getLogger("tamagi.core.aura_client")


class AuraClient:
    def __init__(self, config: BrainConfig) -> None:
        self.base_url = config.aura_base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.aura_timeout, connect=10.0),
        )

    # ── Core chat path ────────────────────────────────────────

    async def input(
        self,
        user_message: str,
        conversation_id: str | None = None,
        persona_context: str | None = None,
    ) -> dict[str, Any]:
        """Send a user message to AURA and return its full response dict.

        AURA POST /api/input body:
            { "input": "...", "conversation_id": "...", "persona_context": "..." }

        AURA response fields used by TamAGI:
            response          — text to show the user
            conversation_id   — ID to thread future messages
            llm_error         — set when AURA's LLM call failed
            task              — TaskFrame dict if a task was created
            deliberations     — list of pending Tier-2 items
            pending_proposals — list of proposal IDs awaiting approval
        """
        payload: dict[str, Any] = {"input": user_message}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if persona_context:
            payload["persona_context"] = persona_context

        try:
            resp = await self._client.post(
                f"{self.base_url}/api/input",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "AURA /api/input HTTP %s: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise
        except httpx.RequestError as exc:
            logger.error("AURA /api/input connection error: %s", exc)
            raise

    # ── Status & observability ────────────────────────────────

    async def get_status(self) -> dict[str, Any]:
        """GET /api/status — AURA system snapshot."""
        resp = await self._client.get(f"{self.base_url}/api/status")
        resp.raise_for_status()
        return resp.json()

    # ── Deliberation / approval ───────────────────────────────

    async def get_pending_approvals(self) -> dict[str, Any]:
        """GET /api/deliberation/pending — Tier-2 proposals awaiting human sign-off."""
        resp = await self._client.get(f"{self.base_url}/api/deliberation/pending")
        resp.raise_for_status()
        return resp.json()

    async def respond_to_deliberation(
        self,
        proposal_id: str,
        choice: str,
        justification: str = "",
    ) -> dict[str, Any]:
        """POST /api/deliberation/respond — submit human response to a pending proposal.

        choice values (from AURA's HumanDeliberationChoice enum):
            accept_system | accept_original | modify | force_override | reject
        """
        resp = await self._client.post(
            f"{self.base_url}/api/deliberation/respond",
            json={
                "proposal_id": proposal_id,
                "choice": choice,
                "justification": justification,
            },
        )
        resp.raise_for_status()
        return resp.json()

    # ── Lifecycle ─────────────────────────────────────────────

    async def aclose(self) -> None:
        await self._client.aclose()
