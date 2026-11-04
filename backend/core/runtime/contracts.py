"""Trusted execution contracts. Model arguments never construct these contexts."""

from __future__ import annotations
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar, TypedDict

Outcome = Literal[
    "completed",
    "resting",
    "waiting",
    "blocked",
    "declined",
    "interrupted",
    "failed",
    "budget_exhausted",
]
Category = Literal["platform", "interpretation", "imagination", "legacy"]


class RuntimeConflict(ValueError):
    pass


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionContext:
    execution_id: str
    actor_id: str
    origin: str
    authority: str = "guarded"
    parent_id: str | None = None
    root_id: str = ""
    allowed_tools: frozenset[str] | None = None
    depth: int = 0
    revisions: dict[str, int] = field(default_factory=dict)
    conversation_id: str | None = None
    occasion_id: str | None = None
    evaluation: bool = False
    eligible_live: bool = False


@dataclass
class TurnResult:
    status: Outcome = "completed"
    text: str = ""
    execution_id: str = ""
    receipts: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    next_wake_at: float | None = None


RecordData = TypeVar("RecordData")


class Record(TypedDict, Generic[RecordData]):
    """Canonical versioned envelope, shared by service and API adapters."""

    kind: str
    id: str
    actor: str
    revision: int
    created: float
    updated: float
    payload: str
    data: RecordData


class ActionReceipt(TypedDict):
    id: str
    run: str
    actor: str
    tool: str
    call_key: str | None
    started: float
    finished: float | None
    status: Literal["started", "succeeded", "failed", "outcome_unknown"]
    payload: dict[str, Any]


class PursuitData(TypedDict):
    title: str
    origin: str
    status: Literal["active", "paused", "blocked", "completed", "abandoned"]
    next_step: str
    completion_condition: str | None
    blocker: str | None
    wake_condition: str | None
    artifacts: list[str]
    evidence: list[str]
    commitment: bool


class CompanionData(TypedDict):
    name: str
    seed: str
    self_description: str
    interests: list[str]
    primary: bool
    status: Literal["active", "archived"]
    created_by: str


class InvocationData(TypedDict):
    mode: Literal["conversation", "task"]
    caller_actor_id: str
    content: str
    started_at: float
    completed_at: float
    finished_at: float
    conversation_id: str | None
    occasion_id: str | None
    root_id: str
    growth_eligible: bool
    growth_completed: bool
    actor_id: str
    speaker: str
    thread_id: str
    invocation_id: str
    status: Outcome
    text: str
    receipts: list[str]
    skills_used: list[str]


# These aliases describe persisted envelopes, not independently writable stores.
Pursuit = Record[PursuitData]
Companion = Record[CompanionData]
Invocation = Record[InvocationData]
Revision = Record[dict[str, Any]]
Experiment = Record[dict[str, Any]]
Deployment = Record[dict[str, Any]]


current_execution: ContextVar[ExecutionContext | None] = ContextVar(
    "tamagi_execution", default=None
)
current_terminal: ContextVar[dict[str, Any] | None] = ContextVar(
    "tamagi_terminal", default=None
)

current_runtime: ContextVar[Any] = ContextVar("tamagi_runtime", default=None)
