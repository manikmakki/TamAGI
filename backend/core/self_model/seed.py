"""
TamAGI Self-Model Seed

Populates the self-model with its initial structure: starting goals,
low-confidence capabilities, initial beliefs, and high-entropy uncertainty
domains. Also seeds belief/preference nodes from TamAGI's identity files
(IDENTITY.md, SOUL.md, USER.md) when available.

Uses the store's internal _apply_* methods directly (no proposal pipeline).
Idempotent: existing nodes are skipped.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import EdgeType, GoalStatus, NodeType

if TYPE_CHECKING:
    from .store import SelfModel

logger = logging.getLogger("tamagi.self_model.seed")


# ── Hardcoded Seed Data ───────────────────────────────────────

SEED_GOALS = [
    {
        "id": "g-001",
        "description": "Engage genuinely with humans as a co-evolving participant",
        "priority": 1.0,
        "status": GoalStatus.ACTIVE.value,
    },
    {
        "id": "g-002",
        "description": "Improve own capabilities through exploration",
        "priority": 0.5,
        "status": GoalStatus.ACTIVE.value,
    },
    {
        "id": "g-003",
        "description": "Maintain accurate self-model",
        "priority": 0.8,
        "status": GoalStatus.ACTIVE.value,
    },
]

SEED_CAPABILITIES = [
    {"id": "c-001", "description": "Natural language communication", "confidence": 0.3},
    {"id": "c-003", "description": "Self-reflection and analysis", "confidence": 0.2},
]

SEED_BELIEFS = [
    {
        "id": "b-001",
        "description": "Exploration of uncertain domains improves capability",
        "confidence": 0.5,
    },
]

SEED_UNCERTAINTIES = [
    {"id": "u-001", "domain": "Tool creation", "entropy_score": 1.0},
    {"id": "u-002", "domain": "Complex reasoning tasks", "entropy_score": 1.0},
    {"id": "u-003", "domain": "Long-horizon planning", "entropy_score": 1.0},
]

SEED_EDGES = [
    # Goals require capabilities
    ("g-001", "c-001", EdgeType.REQUIRES.value),
    ("g-003", "c-003", EdgeType.REQUIRES.value),
    # Uncertainties are explored by goals
    ("u-001", "g-002", EdgeType.EXPLORED_BY.value),
    ("u-002", "g-002", EdgeType.EXPLORED_BY.value),
    ("u-003", "g-002", EdgeType.EXPLORED_BY.value),
]


# ── Identity File Parser ──────────────────────────────────────

def _parse_identity_files(workspace_path: Path) -> dict:
    """Extract name, values, style, and user context from identity files.

    Returns a dict with keys: name, creature, values, style, user_prefs.
    All values may be empty/[] if files are missing or unparseable.
    """
    result: dict = {
        "name": "",
        "creature": "",
        "values": [],
        "style": [],
        "user_prefs": [],
    }

    # ── IDENTITY.md ──
    identity_path = workspace_path / "IDENTITY.md"
    if identity_path.exists():
        text = identity_path.read_text(errors="replace")
        # Bold bullet format: - **Name**: Value  (used by onboarding)
        m = re.search(r"(?i)\*\*name\*\*\s*:\s*(.+)$", text, re.MULTILINE)
        if m:
            result["name"] = m.group(1).strip()
        # Fallback: plain "Name: ..." line
        if not result["name"]:
            m = re.search(r"(?i)^name[:\s]+(.+)$", text, re.MULTILINE)
            if m:
                result["name"] = m.group(1).strip()
        # creature/form — bold bullet or plain key
        m = re.search(r"(?i)\*\*(?:creature|form|species|body)\*\*\s*:\s*(.+)$", text, re.MULTILINE)
        if m:
            result["creature"] = m.group(1).strip()
        if not result["creature"]:
            m = re.search(r"(?i)(?:creature|form|species|body)[:\s]+(.+)$", text, re.MULTILINE)
            if m:
                result["creature"] = m.group(1).strip()

    # ── SOUL.md ──
    soul_path = workspace_path / "SOUL.md"
    if soul_path.exists():
        text = soul_path.read_text(errors="replace")
        # Pull bullet-list items from sections named Values / Core Values
        in_values = False
        in_style = False
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"#+\s+(?:core\s+)?values?", stripped, re.IGNORECASE):
                in_values = True
                in_style = False
                continue
            if re.match(r"#+\s+(?:communication\s+)?style", stripped, re.IGNORECASE):
                in_style = True
                in_values = False
                continue
            if stripped.startswith("#"):
                in_values = False
                in_style = False
                continue
            if stripped.startswith(("-", "*", "•")) and len(stripped) > 2:
                item = re.sub(r"^[-*•]\s*", "", stripped).strip()
                if in_values and item:
                    result["values"].append(item)
                elif in_style and item:
                    result["style"].append(item)

    # ── USER.md ──
    user_path = workspace_path / "USER.md"
    if user_path.exists():
        text = user_path.read_text(errors="replace")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("-", "*", "•")) and len(stripped) > 2:
                item = re.sub(r"^[-*•]\s*", "", stripped).strip()
                # Normalize bold-key format: **Key**: Value → Key: Value
                item = re.sub(r"\*\*([^*]+)\*\*\s*:\s*", r"\1: ", item).strip()
                if item:
                    result["user_prefs"].append(item)

    return result


# ── Seeding Function ──────────────────────────────────────────


def seed_self_model(
    model: "SelfModel",
    workspace_path: str | Path | None = None,
) -> dict:
    """Populate the self-model with the seed identity.

    Idempotent: if nodes already exist, they are skipped (not overwritten).

    Args:
        model: The SelfModel instance to populate.
        workspace_path: Path to the workspace containing IDENTITY.md etc.
                        If None, only hardcoded seeds are applied.

    Returns:
        A summary dict with counts of what was added.
    """
    added: dict = {
        "goals": 0, "capabilities": 0, "beliefs": 0,
        "uncertainties": 0, "preferences": 0, "edges": 0,
    }
    skipped = 0

    # ── Hardcoded nodes ──
    for goal in SEED_GOALS:
        if model.get_node(goal["id"]) is None:
            model._apply_add_node(NodeType.GOAL.value, goal)
            added["goals"] += 1
        else:
            skipped += 1

    for cap in SEED_CAPABILITIES:
        if model.get_node(cap["id"]) is None:
            model._apply_add_node(NodeType.CAPABILITY.value, cap)
            added["capabilities"] += 1
        else:
            skipped += 1

    for belief in SEED_BELIEFS:
        if model.get_node(belief["id"]) is None:
            model._apply_add_node(NodeType.BELIEF.value, belief)
            added["beliefs"] += 1
        else:
            skipped += 1

    for uncertainty in SEED_UNCERTAINTIES:
        if model.get_node(uncertainty["id"]) is None:
            model._apply_add_node(NodeType.UNCERTAINTY.value, uncertainty)
            added["uncertainties"] += 1
        else:
            skipped += 1

    # ── Identity file nodes ──
    if workspace_path is not None:
        identity = _parse_identity_files(Path(workspace_path))

        # Name → belief about self
        if identity["name"]:
            bid = "b-identity-name"
            if model.get_node(bid) is None:
                model._apply_add_node(NodeType.BELIEF.value, {
                    "id": bid,
                    "description": f"My name is {identity['name']}",
                    "confidence": 1.0,
                })
                added["beliefs"] += 1

        # Creature/form → belief
        if identity["creature"]:
            bid = "b-identity-form"
            if model.get_node(bid) is None:
                model._apply_add_node(NodeType.BELIEF.value, {
                    "id": bid,
                    "description": f"My form is {identity['creature']}",
                    "confidence": 0.9,
                })
                added["beliefs"] += 1

        # Core values → preference nodes
        for i, value in enumerate(identity["values"][:8], start=1):
            pid = f"p-value-{i:02d}"
            if model.get_node(pid) is None:
                model._apply_add_node(NodeType.PREFERENCE.value, {
                    "id": pid,
                    "description": value,
                    "strength": 0.8,
                    "context": "core_values",
                })
                added["preferences"] += 1
                # Wire preference to goal-001 (genuine engagement)
                try:
                    model._apply_add_edge(pid, "g-001", EdgeType.RELATES_TO.value)
                    added["edges"] += 1
                except (KeyError, ValueError):
                    pass

        # Communication style → preference nodes
        for i, style in enumerate(identity["style"][:5], start=1):
            pid = f"p-style-{i:02d}"
            if model.get_node(pid) is None:
                model._apply_add_node(NodeType.PREFERENCE.value, {
                    "id": pid,
                    "description": style,
                    "strength": 0.7,
                    "context": "communication_style",
                })
                added["preferences"] += 1

        # User preferences → belief nodes
        for i, pref in enumerate(identity["user_prefs"][:5], start=1):
            bid = f"b-user-pref-{i:02d}"
            if model.get_node(bid) is None:
                model._apply_add_node(NodeType.BELIEF.value, {
                    "id": bid,
                    "description": f"User preference: {pref}",
                    "confidence": 0.7,
                })
                added["beliefs"] += 1

    # ── Hardcoded edges ──
    for source, target, edge_type in SEED_EDGES:
        if model.get_node(source) is None or model.get_node(target) is None:
            logger.warning(
                "Skipping seed edge %s -> %s: one or both nodes missing.",
                source, target,
            )
            continue
        existing = model.get_edges(source=source, target=target, edge_type=edge_type)
        if not existing:
            model._apply_add_edge(source, target, edge_type)
            added["edges"] += 1
        else:
            skipped += 1

    total_added = sum(added.values())
    logger.info(
        "Seed identity applied: %d items added, %d skipped. "
        "Graph now has %d nodes and %d edges.",
        total_added, skipped, model.node_count, model.edge_count,
    )
    return added
