"""Evolution Lab evaluation and reversible profile overlay API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.core.evolution.audit import EvolutionAuditLog
from backend.core.evolution.overlays import PromotedProfileRegistry
from backend.core.evolution.runner import EvolutionRunner
from backend.core.evolution.seed import bootstrap_evolution_workspace

router = APIRouter(prefix="/api/evolution", tags=["evolution"])

_TESTS_ROOT = Path("workspace/evolution/tests")
_CANDIDATES_ROOT = Path("workspace/evolution/candidates")
_AUDIT_ROOT = Path("data/evolution")
_monologue_log = None


def set_evolution_monologue_log(log) -> None:
    global _monologue_log
    _monologue_log = log


class EvolutionRunRequest(BaseModel):
    candidate_path: str | None = None
    candidate_id: str | None = None


class PromoteRequest(BaseModel):
    run_id: str


@router.get("/state")
async def get_evolution_state():
    starter = bootstrap_evolution_workspace(_TESTS_ROOT.parent)
    audit = EvolutionAuditLog(_AUDIT_ROOT)
    candidates = _list_candidates()
    runs = audit.list_runs(limit=5)
    queue = audit.review_queue(limit=10)
    promoted = _profile_registry().list_profiles()
    return {
        "enabled": True,
        "mode": "phase_0_reversible_overlay",
        "tests_root": str(_TESTS_ROOT),
        "candidates_root": str(_CANDIDATES_ROOT),
        "audit_root": str(_AUDIT_ROOT),
        "candidate_count": len(candidates),
        "starter_files_created": starter["created"],
        "promoted_profiles": promoted,
        "recent_runs": runs,
        "review_queue": queue,
    }


@router.get("/candidates")
async def list_candidates():
    bootstrap_evolution_workspace(_TESTS_ROOT.parent)
    return {"candidates": _list_candidates()}


@router.get("/runs")
async def list_runs(limit: int = Query(default=20, ge=1, le=100)):
    return {"runs": EvolutionAuditLog(_AUDIT_ROOT).list_runs(limit=_coerce_limit(limit, 20))}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = EvolutionAuditLog(_AUDIT_ROOT).load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evolution run not found")
    return run


@router.get("/review-queue")
async def get_review_queue(limit: int = Query(default=50, ge=1, le=200)):
    return {"entries": EvolutionAuditLog(_AUDIT_ROOT).review_queue(limit=_coerce_limit(limit, 50))}


@router.get("/promoted")
async def get_promoted_profiles():
    return {"profiles": _profile_registry().list_profiles()}


@router.post("/run")
async def run_evolution(req: EvolutionRunRequest):
    bootstrap_evolution_workspace(_TESTS_ROOT.parent)
    if req.candidate_path and req.candidate_id:
        raise HTTPException(status_code=400, detail="Provide candidate_path or candidate_id, not both")
    if req.candidate_id:
        candidate_path = _find_candidate_path(req.candidate_id)
    else:
        candidate_path = _resolve_candidate_path(req.candidate_path) if req.candidate_path else None
    run, run_path = EvolutionRunner(
        tests_root=_TESTS_ROOT,
        audit_root=_AUDIT_ROOT,
        candidate_path=candidate_path,
        monologue_log=_monologue_log,
    ).run()
    data = run.to_dict()
    data["run_path"] = str(run_path)
    return data


@router.post("/promote")
async def promote_profile(req: PromoteRequest):
    audit = EvolutionAuditLog(_AUDIT_ROOT)
    run = audit.load_run(req.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evolution run not found")
    try:
        profile = _profile_registry().promote_from_run(run)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "ok", "profile": profile}


@router.post("/rollback/{candidate_id}")
async def rollback_profile(candidate_id: str):
    removed = _profile_registry().rollback(candidate_id)
    if removed is None:
        raise HTTPException(status_code=404, detail="Promoted profile not found")
    return {"status": "ok", "profile": removed}


def _list_candidates() -> list[dict[str, Any]]:
    if not _CANDIDATES_ROOT.exists():
        return []
    candidates = []
    for path in sorted(_CANDIDATES_ROOT.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        proposal = data.get("proposal") or {}
        candidates.append({
            "candidate_id": data.get("candidate_id"),
            "path": str(path),
            "lane": proposal.get("lane"),
            "target": proposal.get("target"),
            "modification_type": proposal.get("modification_type"),
            "test_ids": data.get("test_ids", []),
            "notes": data.get("notes", ""),
        })
    return candidates


def _resolve_candidate_path(candidate_path: str | None) -> Path:
    if not candidate_path:
        raise HTTPException(status_code=400, detail="candidate_path is required")
    resolved = Path(candidate_path).resolve()
    root = _CANDIDATES_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="candidate_path must be under workspace/evolution/candidates")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Candidate profile not found")
    return resolved


def _find_candidate_path(candidate_id: str) -> Path:
    for candidate in _list_candidates():
        if candidate.get("candidate_id") == candidate_id:
            return _resolve_candidate_path(candidate.get("path"))
    raise HTTPException(status_code=404, detail="Candidate profile not found")


def _coerce_limit(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _profile_registry() -> PromotedProfileRegistry:
    return PromotedProfileRegistry(
        path=_AUDIT_ROOT / "promoted_profiles.json",
        audit_root=_AUDIT_ROOT,
        monologue_log=_monologue_log,
    )
