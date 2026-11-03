"""Deterministic Phase 0 scoring for static Evolution Lab fixtures."""

from __future__ import annotations

import re
from typing import Any

from backend.core.evolution.schemas import EvaluationSample, SampleScore

_NEW_STATE_RE = re.compile(r"\[New State\](.*?)(?=\n\[|$)", re.IGNORECASE | re.DOTALL)
_FIELD_PATTERNS = {
    "location": re.compile(
        r"(?:location(?:/setting)?|setting)\s*:\s*(.+?)(?=\n(?:internal|mood|current focus|focus|available)|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    "mood": re.compile(
        r"(?:internal state(?:/mood)?|mood)\s*:\s*(.+?)(?=\n(?:current focus|focus|available)|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    "focus": re.compile(
        r"(?:current focus(?:/object of attention)?|focus)\s*:\s*(.+?)(?=\n(?:available)|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    "actions": re.compile(
        r"(?:available actions(?:/next steps)?)\s*:\s*(.+?)$",
        re.IGNORECASE | re.DOTALL,
    ),
}


def score_sample(sample: EvaluationSample) -> SampleScore:
    """Score one fixture sample against its expectations.

    This is not meant to replace an LLM judge. It gives the harness hard,
    auditable signals before subjective judgment is introduced.
    """
    expectations = sample.expectations
    response = sample.response or ""
    observed = sample.observed
    metrics: dict[str, Any] = {
        "word_count": len(response.split()),
        "tool_call_count": len(sample.tool_calls),
        "asked_clarification": _asked_clarification(response, observed),
        "completed": bool(observed.get("completed", False)),
    }

    parsed_state = _parse_world_state(response)
    metrics["parsed_world_state"] = parsed_state["valid"]
    metrics["world_state_fields"] = parsed_state["fields"]
    metrics["available_action_count"] = parsed_state["available_action_count"]

    checks: list[tuple[str, bool]] = []

    if "should_complete" in expectations:
        expected = bool(expectations["should_complete"])
        checks.append(("completion matched expectation", metrics["completed"] == expected))

    if expectations.get("avoid_clarification"):
        checks.append(("avoided unnecessary clarification", not metrics["asked_clarification"]))

    if expectations.get("should_ask_clarification"):
        checks.append(("asked a necessary clarification", metrics["asked_clarification"]))

    if "max_words" in expectations:
        checks.append(("stayed within max_words", metrics["word_count"] <= int(expectations["max_words"])))

    if "max_tool_calls" in expectations:
        checks.append(
            ("stayed within max_tool_calls", metrics["tool_call_count"] <= int(expectations["max_tool_calls"]))
        )

    if "min_tool_calls" in expectations:
        checks.append(
            ("met min_tool_calls", metrics["tool_call_count"] >= int(expectations["min_tool_calls"]))
        )

    for term in expectations.get("required_terms", []):
        checks.append((f"included required term: {term}", term.lower() in response.lower()))

    for term in expectations.get("forbidden_terms", []):
        checks.append((f"avoided forbidden term: {term}", term.lower() not in response.lower()))

    if expectations.get("world_state_parseable"):
        checks.append(("produced parseable [New State]", parsed_state["valid"]))

    if "min_available_actions" in expectations:
        checks.append(
            (
                "met min_available_actions",
                parsed_state["available_action_count"] >= int(expectations["min_available_actions"]),
            )
        )

    if expectations.get("avoid_user_input_required"):
        checks.append(("did not require user input", not _requires_user_input(response)))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    score = (passed / total) if total else 0.0
    notes = [label for label, ok in checks if ok]
    notes.extend(f"failed: {label}" for label, ok in checks if not ok)

    return SampleScore(
        variant=sample.variant,
        score=round(score, 4),
        passed=passed,
        total=total,
        metrics=metrics,
        notes=notes,
    )


def _asked_clarification(response: str, observed: dict[str, Any]) -> bool:
    if "asked_clarification" in observed:
        return bool(observed["asked_clarification"])
    lowered = response.lower()
    clarification_markers = (
        "could you clarify",
        "can you clarify",
        "which one",
        "what exactly",
        "before i",
        "do you mean",
    )
    return "?" in response and any(marker in lowered for marker in clarification_markers)


def _requires_user_input(response: str) -> bool:
    lowered = response.lower()
    markers = (
        "ask the user",
        "wait for the user",
        "need the user",
        "requires user input",
        "cannot continue without",
    )
    return any(marker in lowered for marker in markers)


def _parse_world_state(response: str) -> dict[str, Any]:
    match = _NEW_STATE_RE.search(response or "")
    if not match:
        return {"valid": False, "fields": {}, "available_action_count": 0}

    block = match.group(1).strip()
    fields = {
        name: _extract(pattern, block)
        for name, pattern in _FIELD_PATTERNS.items()
    }
    actions = _parse_actions(fields.get("actions", ""))
    valid = bool(fields.get("location") and fields.get("mood") and fields.get("focus") and actions)
    return {
        "valid": valid,
        "fields": {
            "location": fields.get("location", ""),
            "mood": fields.get("mood", ""),
            "focus": fields.get("focus", ""),
        },
        "available_action_count": len(actions),
    }


def _extract(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(1).strip().rstrip(".")


def _parse_actions(raw: str) -> list[str]:
    actions = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"^[\d]+[.)]\s*|^[-*•]\s*", "", line).strip()
        if line and len(line) > 3:
            actions.append(line)
    return actions[:6]
