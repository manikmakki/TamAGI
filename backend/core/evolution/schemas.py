"""Schemas for the Phase 0 Evolution Lab static evaluator.

Phase 0 intentionally uses plain dataclasses and JSON-compatible shapes. The
goal is to prove the evaluate-and-audit loop before introducing live mutation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Lane = Literal["conversation", "world_thread", "safety"]
Decision = Literal["candidate_better", "baseline_better", "tie", "insufficient_data"]


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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [r.to_dict() for r in self.results]
        return data
