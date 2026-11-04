"""TamAGI's durable, local-first platform."""

from .contracts import (
    ExecutionContext,
    TurnResult,
    ActionReceipt,
    Pursuit,
    Companion,
    Invocation,
    Revision,
    Experiment,
    Deployment,
    BudgetExhausted,
    RuntimeConflict,
    current_execution,
    current_terminal,
    current_runtime,
)
from .store import RuntimeStore
from .service import RuntimeService, GovernedLLM

_active = None


def set_runtime(runtime):
    global _active
    _active = runtime


def get_runtime():
    return current_runtime.get() or _active


__all__ = [
    "ExecutionContext",
    "TurnResult",
    "ActionReceipt",
    "Pursuit",
    "Companion",
    "Invocation",
    "Revision",
    "Experiment",
    "Deployment",
    "BudgetExhausted",
    "RuntimeConflict",
    "current_execution",
    "current_terminal",
    "current_runtime",
    "RuntimeStore",
    "RuntimeService",
    "GovernedLLM",
    "get_runtime",
    "set_runtime",
]
