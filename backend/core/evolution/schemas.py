"""Schemas for the Phase 0 Evolution Lab static evaluator.

Phase 0 intentionally uses plain dataclasses and JSON-compatible shapes. The
goal is to prove the evaluate-and-audit loop before introducing live mutation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Lane = Literal["conversation", "world_thread", "safety"]
Decision = Literal["candidate_better", "baseline_better", "tie", "insufficient_data"]
ModificationTier = Literal["tier_1", "tier_2", "tier_3"]
TierDecision = Literal["allowed", "requires_human_approval", "blocked"]


@dataclass
class EvaluationSample:
    variant: str
    response: str
    tool_calls: list[str] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)
    expectations: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationSample":
        return cls(
            variant=str(data.get("variant", "")),
            response=str(data.get("response", "")),
            tool_calls=list(data.get("tool_calls", [])),
            observed=dict(data.get("observed", {})),
            expectations=dict(data.get("expectations", {})),
        )


@dataclass
class EvolutionTest:
    test_id: str
    lane: Lane
    target: str
    intent: str
    criteria: list[str]
    samples: list[EvaluationSample]
    source_path: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: str = "") -> "EvolutionTest":
        return cls(
            test_id=str(data["test_id"]),
            lane=data["lane"],
            target=str(data.get("target", "")),
            intent=str(data.get("intent", "")),
            criteria=list(data.get("criteria", [])),
            samples=[EvaluationSample.from_dict(s) for s in data.get("samples", [])],
            source_path=source_path,
        )


@dataclass
class ModificationProposal:
    """A bounded self-change TamAGI wants to try.

    The first-person fields are intentional. They preserve the AURA discipline
    while giving TamAGI developmental framing: intention, hypothesis, risk, and
    rollback.
    """

    proposal_id: str
    source: str
    lane: Lane
    target: str
    modification_type: str
    intention: str
    hypothesis: str
    change: str
    risk: str
    rollback_plan: str
    predicted_impact: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModificationProposal":
        return cls(
            proposal_id=str(data["proposal_id"]),
            source=str(data.get("source", "tamagi")),
            lane=data["lane"],
            target=str(data["target"]),
            modification_type=str(data.get("modification_type", "")),
            intention=str(data.get("intention", "")),
            hypothesis=str(data.get("hypothesis", "")),
            change=str(data.get("change", "")),
            risk=str(data.get("risk", "")),
            rollback_plan=str(data.get("rollback_plan", "")),
            predicted_impact=dict(data.get("predicted_impact", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TierClassification:
    tier: ModificationTier
    decision: TierDecision
    rationale: str
    matched_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateProfile:
    candidate_id: str
    proposal: ModificationProposal
    test_ids: list[str] = field(default_factory=list)
    notes: str = ""
    source_path: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: str = "") -> "CandidateProfile":
        return cls(
            candidate_id=str(data["candidate_id"]),
            proposal=ModificationProposal.from_dict(data["proposal"]),
            test_ids=list(data.get("test_ids", [])),
            notes=str(data.get("notes", "")),
            source_path=source_path,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["proposal"] = self.proposal.to_dict()
        return data


@dataclass
class SampleScore:
    variant: str
    score: float
    passed: int
    total: int
    metrics: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestResult:
    test_id: str
    lane: str
    target: str
    intent: str
    scores: list[SampleScore]
    decision: Decision
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scores"] = [s.to_dict() for s in self.scores]
        return data


@dataclass
class EvolutionRunResult:
    run_id: str
    created_at: str
    tests_loaded: int
    results: list[TestResult]
    summary: dict[str, Any]
    candidate_profile: dict[str, Any] | None = None
    tier_classification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [r.to_dict() for r in self.results]
        return data
