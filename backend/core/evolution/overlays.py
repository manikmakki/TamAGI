"""Promoted profile registry and prompt overlay rendering."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.core.evolution.audit import EvolutionAuditLog


class PromotedProfileRegistry:
    """Store approved soft overlays without editing production prompts or code."""

    def __init__(
        self,
        path: str | Path = "data/evolution/promoted_profiles.json",
        audit_root: str | Path = "data/evolution",
        monologue_log=None,
    ) -> None:
        self.path = Path(path)
        self.audit = EvolutionAuditLog(audit_root)
        self.monologue_log = monologue_log

    def list_profiles(self) -> list[dict[str, Any]]:
        return self._load().get("profiles", [])

    def promote_from_run(self, run: dict[str, Any]) -> dict[str, Any]:
        candidate = run.get("candidate_profile") or {}
        proposal = candidate.get("proposal") or {}
        tier = run.get("tier_classification") or {}
        summary = run.get("summary") or {}

        if not candidate:
            raise ValueError("Run does not contain a candidate profile")
        if tier.get("decision") != "allowed" or tier.get("tier") != "tier_1":
            raise ValueError("Only allowed Tier 1 candidates can be promoted as prompt overlays")
        if summary.get("recommendation") != "promote_candidate":
            raise ValueError("Run does not recommend candidate promotion")

        profile = {
            "candidate_id": candidate.get("candidate_id"),
            "proposal_id": proposal.get("proposal_id"),
            "run_id": run.get("run_id"),
            "lane": proposal.get("lane"),
            "target": proposal.get("target"),
            "change": proposal.get("change"),
            "intention": proposal.get("intention"),
            "risk": proposal.get("risk"),
            "rollback_plan": proposal.get("rollback_plan"),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        data = self._load()
        profiles = [
            p for p in data.get("profiles", [])
            if p.get("candidate_id") != profile["candidate_id"]
        ]
        profiles.append(profile)
        self._save({"profiles": profiles})
        self.audit.append_event(
            event_type="evolution_profile_promoted",
            summary=f"Promoted Evolution profile overlay: {profile['candidate_id']}",
            detail={"profile": profile},
        )
        self._record_monologue(
            event_type="evolution_promoted",
            title=f"Kept a change to {profile.get('target', 'my behavior')}",
            content=(
                f"[Intention]\n{profile.get('intention', '')}\n\n"
                f"[Change]\n{profile.get('change', '')}\n\n"
                f"[Safety Net]\n{profile.get('rollback_plan', '')}"
            ),
            profile=profile,
        )
        return profile

    def rollback(self, candidate_id: str) -> dict[str, Any] | None:
        data = self._load()
        profiles = data.get("profiles", [])
        kept = [p for p in profiles if p.get("candidate_id") != candidate_id]
        removed = next((p for p in profiles if p.get("candidate_id") == candidate_id), None)
        if removed is None:
            return None
        removed = dict(removed)
        removed["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
        removed["status"] = "rolled_back"
        self._save({"profiles": kept})
        self.audit.append_event(
            event_type="evolution_profile_rolled_back",
            summary=f"Rolled back Evolution profile overlay: {candidate_id}",
            detail={"profile": removed},
        )
        self._record_monologue(
            event_type="evolution_rolled_back",
            title=f"Released a change to {removed.get('target', 'my behavior')}",
            content=(
                f"[Change Released]\n{removed.get('change', '')}\n\n"
                "[Result]\nThe prior behavior is active again. The experiment remains in the audit trail."
            ),
            profile=removed,
        )
        return removed

    def render_prompt_block(self, lane: str = "conversation") -> str:
        profiles = [
            p for p in self.list_profiles()
            if p.get("status", "active") == "active" and p.get("lane") == lane
        ]
        if not profiles:
            return ""

        lines = [
            "[Evolution Profile Overlays]",
            "These are approved, reversible practice overlays. Follow them when relevant, "
            "but do not let them override safety, user intent, identity, tool rules, or explicit instructions.",
        ]
        for profile in profiles:
            candidate_id = profile.get("candidate_id", "unknown")
            target = profile.get("target", "behavior")
            change = profile.get("change", "")
            lines.append(f"- {candidate_id} ({target}): {change}")
        return "\n".join(lines)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"profiles": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"profiles": []}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def _record_monologue(
        self,
        event_type: str,
        title: str,
        content: str,
        profile: dict[str, Any],
    ) -> None:
        if self.monologue_log is None:
            return
        try:
            self.monologue_log.append(
                type=event_type,
                source="self",
                title=title,
                content=content,
                metadata={
                    "run_id": profile.get("run_id"),
                    "candidate_id": profile.get("candidate_id"),
                    "target": profile.get("target"),
                },
            )
        except Exception:
            # Audit and rollback behavior remain authoritative if display logging fails.
            return
