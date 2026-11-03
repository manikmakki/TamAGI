"""Audit persistence for Evolution Lab Phase 0 runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class EvolutionAuditLog:
    """Persist run records and append hash-chained audit entries."""

    def __init__(self, root: str | Path = "data/evolution") -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.audit_path = self.root / "audit.jsonl"
        self.review_queue_path = self.root / "review_queue.json"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def persist_run(self, run: dict[str, Any]) -> Path:
        run_path = self.runs_dir / f"{run['run_id']}.json"
        run_path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")
        candidate = run.get("candidate_profile") or {}
        candidate_id = candidate.get("candidate_id", "static-fixtures")
        if candidate:
            self._append_review_entry(run, run_path)
        self.append_event(
            event_type="evolution_run_completed",
            summary=(
                f"Evolution Phase 0 run completed for {candidate_id}: "
                f"{run['summary'].get('recommendation', 'unknown')}."
            ),
            detail={"run_id": run["run_id"], "run_path": str(run_path), "summary": run["summary"]},
        )
        return run_path

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent run summaries, newest first."""
        runs = []
        for path in sorted(self.runs_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            runs.append({
                "run_id": data.get("run_id"),
                "created_at": data.get("created_at"),
                "candidate_id": (data.get("candidate_profile") or {}).get("candidate_id"),
                "summary": data.get("summary", {}),
                "path": str(path),
            })
            if len(runs) >= limit:
                break
        return runs

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def review_queue(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.review_queue_path.exists():
            return []
        try:
            data = json.loads(self.review_queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        entries = data.get("entries", [])
        return list(reversed(entries[-limit:]))

    def append_event(self, event_type: str, summary: str, detail: dict[str, Any]) -> None:
        prev_hash = self._last_hash()
        payload = {
            "event_type": event_type,
            "summary": summary,
            "detail": detail,
            "prev_hash": prev_hash,
        }
        payload["entry_hash"] = self._hash_payload(payload)
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def _last_hash(self) -> str:
        if not self.audit_path.exists():
            return "0" * 64
        last = ""
        with self.audit_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        if not last:
            return "0" * 64
        try:
            return str(json.loads(last).get("entry_hash", "0" * 64))
        except json.JSONDecodeError:
            return "0" * 64

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> str:
        body = json.dumps({k: v for k, v in payload.items() if k != "entry_hash"}, sort_keys=True)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def _append_review_entry(self, run: dict[str, Any], run_path: Path) -> None:
        candidate = run.get("candidate_profile") or {}
        tier = run.get("tier_classification") or {}
        summary = run.get("summary") or {}
        entry = {
            "run_id": run.get("run_id"),
            "created_at": run.get("created_at"),
            "candidate_id": candidate.get("candidate_id"),
            "proposal_id": ((candidate.get("proposal") or {}).get("proposal_id")),
            "target": ((candidate.get("proposal") or {}).get("target")),
            "lane": ((candidate.get("proposal") or {}).get("lane")),
            "tier": tier.get("tier"),
            "tier_decision": tier.get("decision"),
            "recommendation": summary.get("recommendation"),
            "recommendation_reason": summary.get("recommendation_reason"),
            "run_path": str(run_path),
        }
        existing = {"entries": []}
        if self.review_queue_path.exists():
            try:
                existing = json.loads(self.review_queue_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {"entries": []}
        entries = existing.get("entries", [])
        entries.append(entry)
        self.review_queue_path.write_text(
            json.dumps({"entries": entries[-200:]}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
