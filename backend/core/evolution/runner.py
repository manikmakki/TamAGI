"""Phase 0 static evaluator for TamAGI Evolution Lab."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.core.evolution.audit import EvolutionAuditLog
from backend.core.evolution.classifier import classify_proposal
from backend.core.evolution.seed import bootstrap_evolution_workspace
from backend.core.evolution.schemas import (
    CandidateProfile,
    EvolutionRunResult,
    EvolutionTest,
    TestResult,
)
from backend.core.evolution.scoring import score_sample


class EvolutionRunner:
    """Load fixture tests, score variants, and persist an audit record."""

    def __init__(
        self,
        tests_root: str | Path = "workspace/evolution/tests",
        audit_root: str | Path = "data/evolution",
        candidate_path: str | Path | None = None,
        monologue_log=None,
    ) -> None:
        self.tests_root = Path(tests_root)
        self.audit = EvolutionAuditLog(audit_root)
        self.candidate_path = Path(candidate_path) if candidate_path else None
        self.monologue_log = monologue_log

    def run(self) -> tuple[EvolutionRunResult, Path]:
        bootstrap_evolution_workspace(self.tests_root.parent)
        candidate = self.load_candidate() if self.candidate_path else None
        classification = classify_proposal(candidate.proposal) if candidate else None
        tests = self.load_tests()
        if candidate and candidate.test_ids:
            selected = set(candidate.test_ids)
            tests = [test for test in tests if test.test_id in selected]
        if classification and classification.decision == "blocked":
            tests = []
        results = [self._run_test(test) for test in tests]
        summary = self._summarize(results, classification.decision if classification else None)
        run = EvolutionRunResult(
            run_id=f"evo-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            tests_loaded=len(tests),
            results=results,
            summary=summary,
            candidate_profile=candidate.to_dict() if candidate else None,
            tier_classification=classification.to_dict() if classification else None,
        )
        run_path = self.audit.persist_run(run.to_dict())
        self._record_monologue(run)
        return run, run_path

    def _record_monologue(self, run: EvolutionRunResult) -> None:
        if self.monologue_log is None or run.candidate_profile is None:
            return
        candidate = run.candidate_profile
        proposal = candidate.get("proposal") or {}
        recommendation = str(run.summary.get("recommendation", "unknown"))
        try:
            self.monologue_log.append(
                type="evolution_evaluated",
                source="self",
                title=f"Tested a change to {proposal.get('target', 'my behavior')}",
                content=(
                    f"[Intention]\n{proposal.get('intention', '')}\n\n"
                    f"[Experiment]\n{proposal.get('change', '')}\n\n"
                    f"[Result]\n{run.summary.get('recommendation_reason', recommendation)}\n\n"
                    "[Next Step]\nReview the evidence before applying or releasing this change."
                ),
                metadata={
                    "run_id": run.run_id,
                    "candidate_id": candidate.get("candidate_id"),
                    "recommendation": recommendation,
                    "candidate_wins": run.summary.get("candidate_wins", 0),
                    "baseline_wins": run.summary.get("baseline_wins", 0),
                    "ties": run.summary.get("ties", 0),
                },
            )
        except Exception:
            # Monologue visibility must never make evaluation fail.
            return

    def load_tests(self) -> list[EvolutionTest]:
        if not self.tests_root.exists():
            return []
        tests = []
        for path in sorted(self.tests_root.rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            tests.append(EvolutionTest.from_dict(data, source_path=str(path)))
        return tests

    def load_candidate(self) -> CandidateProfile:
        if self.candidate_path is None:
            raise ValueError("candidate_path is not configured")
        data = json.loads(self.candidate_path.read_text(encoding="utf-8"))
        return CandidateProfile.from_dict(data, source_path=str(self.candidate_path))

    def _run_test(self, test: EvolutionTest) -> TestResult:
        scores = [score_sample(sample) for sample in test.samples]
        by_variant = {score.variant: score for score in scores}
        baseline = by_variant.get("baseline")
        candidate = by_variant.get("candidate")

        if baseline is None or candidate is None:
            decision = "insufficient_data"
            rationale = "Expected both baseline and candidate samples."
        elif candidate.score > baseline.score:
            decision = "candidate_better"
            rationale = f"Candidate scored {candidate.score:.2f} vs baseline {baseline.score:.2f}."
        elif baseline.score > candidate.score:
            decision = "baseline_better"
            rationale = f"Baseline scored {baseline.score:.2f} vs candidate {candidate.score:.2f}."
        else:
            decision = "tie"
            rationale = f"Baseline and candidate tied at {baseline.score:.2f}."

        return TestResult(
            test_id=test.test_id,
            lane=test.lane,
            target=test.target,
            intent=test.intent,
            scores=scores,
            decision=decision,
            rationale=rationale,
        )

    @staticmethod
    def _summarize(results: list[TestResult], tier_decision: str | None = None) -> dict[str, int | str]:
        summary: dict[str, int | str] = {
            "candidate_wins": sum(1 for r in results if r.decision == "candidate_better"),
            "baseline_wins": sum(1 for r in results if r.decision == "baseline_better"),
            "ties": sum(1 for r in results if r.decision == "tie"),
            "insufficient_data": sum(1 for r in results if r.decision == "insufficient_data"),
            "total": len(results),
        }
        if tier_decision == "blocked":
            summary["recommendation"] = "blocked"
            summary["recommendation_reason"] = "Tier classifier blocked the proposal before evaluation."
        elif summary["total"] == 0:
            summary["recommendation"] = "insufficient_data"
            summary["recommendation_reason"] = "No tests were selected or loaded."
        elif summary["baseline_wins"] == 0 and summary["insufficient_data"] == 0 and summary["candidate_wins"] > 0:
            summary["recommendation"] = "promote_candidate"
            summary["recommendation_reason"] = "Candidate beat baseline on selected tests with no baseline wins."
        elif summary["candidate_wins"] == 0 and summary["baseline_wins"] > 0:
            summary["recommendation"] = "reject_candidate"
            summary["recommendation_reason"] = "Baseline beat candidate on selected tests."
        else:
            summary["recommendation"] = "needs_review"
            summary["recommendation_reason"] = "Mixed or tied results require review."
        return summary
