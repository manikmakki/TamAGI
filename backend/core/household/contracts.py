"""Versioned coordinator contracts; authority comes from admitted worker bindings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1_048_576


class Conflict(ValueError):
    pass


class NotFound(LookupError):
    pass


class AccessDenied(PermissionError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class ReconciliationRequired(RuntimeError):
    pass


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class WorkerBinding:
    household_id: str
    resident_id: str
    installation_id: str
    root: str
    release: str
    protocol: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def parse(cls, value: dict[str, Any]) -> WorkerBinding:
        if not isinstance(value, dict) or set(value) != {
            "household_id", "resident_id", "installation_id", "root", "release", "protocol"
        }:
            raise ProtocolError("Invalid worker binding")
        if type(value["protocol"]) is not int or value["protocol"] != PROTOCOL_VERSION:
            raise ProtocolError("Unsupported household worker protocol")
        if any(not isinstance(value[key], str) or not value[key] for key in value if key != "protocol"):
            raise ProtocolError("Worker binding fields must be nonempty strings")
        return cls(**value)
