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
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def persist_run(self, run: dict[str, Any]) -> Path:
        run_path = self.runs_dir / f"{run['run_id']}.json"
        run_path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")
        self.append_event(
            event_type="evolution_run_completed",
            summary=(
                f"Evolution Phase 0 run completed: {run['summary'].get('candidate_wins', 0)} "
                f"candidate wins, {run['summary'].get('baseline_wins', 0)} baseline wins."
            ),
            detail={"run_id": run["run_id"], "run_path": str(run_path), "summary": run["summary"]},
        )
        return run_path

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
