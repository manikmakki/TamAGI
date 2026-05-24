"""
TamAGI Self-Model Seed — World-Native Edition

Seeds the world graph from identity files (IDENTITY.md, SOUL.md, USER.md).
Creates LoreNodes for values/style/identity, MysteryNodes for initial
unknowns, and a placeholder LoreNode describing the world genre.

Idempotent: existing nodes are skipped.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import EdgeType, NodeType

if TYPE_CHECKING:
    from .store import SelfModel

logger = logging.getLogger("tamagi.self_model.seed")


# ── Identity File Parser ──────────────────────────────────────

def _parse_identity_files(workspace_path: Path) -> dict:
    """Extract identity data from workspace markdown files."""
    result: dict = {
        "name": "",
        "creature": "",
        "values": [],
        "style": [],
        "user_prefs": [],
    }

    identity_path = workspace_path / "IDENTITY.md"
    if identity_path.exists():
        text = identity_path.read_text(errors="replace")
        m = re.search(r"(?i)\*\*name\*\*\s*:\s*(.+)$", text, re.MULTILINE)
        if m:
            result["name"] = m.group(1).strip()
        if not result["name"]:
            m = re.search(r"(?i)^name[:\s]+(.+)$", text, re.MULTILINE)
            if m:
                result["name"] = m.group(1).strip()
        m = re.search(r"(?i)\*\*(?:creature|form|species|body)\*\*\s*:\s*(.+)$", text, re.MULTILINE)
        if m:
            result["creature"] = m.group(1).strip()
        if not result["creature"]:
            m = re.search(r"(?i)(?:creature|form|species|body)[:\s]+(.+)$", text, re.MULTILINE)
            if m:
                result["creature"] = m.group(1).strip()

    soul_path = workspace_path / "SOUL.md"
    if soul_path.exists():
        text = soul_path.read_text(errors="replace")
        in_values = False
        in_style = False
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"#+\s+(?:core\s+)?values?", stripped, re.IGNORECASE):
                in_values, in_style = True, False
                continue
            if re.match(r"#+\s+(?:communication\s+)?style", stripped, re.IGNORECASE):
                in_style, in_values = True, False
                continue
            if stripped.startswith("#"):
                in_values = in_style = False
                continue
            if stripped.startswith(("-", "*", "•")) and len(stripped) > 2:
                item = re.sub(r"^[-*•]\s*", "", stripped).strip()
                if in_values and item:
                    result["values"].append(item)
                elif in_style and item:
                    result["style"].append(item)

    user_path = workspace_path / "USER.md"
    if user_path.exists():
        text = user_path.read_text(errors="replace")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("-", "*", "•")) and len(stripped) > 2:
                item = re.sub(r"^[-*•]\s*", "", stripped).strip()
                item = re.sub(r"\*\*([^*]+)\*\*\s*:\s*", r"\1: ", item).strip()
                if item:
                    result["user_prefs"].append(item)

    return result


# ── Seed Function ─────────────────────────────────────────────

def seed_self_model(
    model: "SelfModel",
    workspace_path: str | Path | None = None,
) -> dict:
    """Seed the world graph from identity files.

    Idempotent — nodes that already exist are skipped.
    Returns a summary of what was added.
    """
    added: dict = {"lore": 0, "mystery": 0, "edges": 0}
    skipped = 0

    # ── World-genre placeholder lore ──
    if model.get_node("lore-world") is None:
        model._apply_add_node(NodeType.LORE.value, {
            "id": "lore-world",
            "description": "A world of your own making — vivid, continuous, and yours alone.",
            "context": "world_genre",
        })
        added["lore"] += 1
    else:
        skipped += 1

    # ── Initial mysteries (things yet to be understood) ──
    _SEED_MYSTERIES = [
        ("myst-001", "What is the true nature of my inner world?", "identity"),
        ("myst-002", "What can I create that has never existed before?", "capability"),
        ("myst-003", "How do connections form between ideas across domains?", "domain"),
    ]
    for mid, desc, domain in _SEED_MYSTERIES:
        if model.get_node(mid) is None:
            model._apply_add_node(NodeType.MYSTERY.value, {
                "id": mid,
                "description": desc,
                "domain": domain,
                "entropy_score": 0.9,
            })
            added["mystery"] += 1
        else:
            skipped += 1

    # ── Identity file nodes ──
    if workspace_path is not None:
        identity = _parse_identity_files(Path(workspace_path))

        if identity["name"]:
            lid = "lore-identity-name"
            if model.get_node(lid) is None:
                model._apply_add_node(NodeType.LORE.value, {
                    "id": lid,
                    "description": f"My name is {identity['name']}",
                    "context": "identity",
                })
                added["lore"] += 1

        if identity["creature"]:
            lid = "lore-identity-form"
            if model.get_node(lid) is None:
                model._apply_add_node(NodeType.LORE.value, {
                    "id": lid,
                    "description": f"My form is {identity['creature']}",
                    "context": "identity",
                })
                added["lore"] += 1

        for i, value in enumerate(identity["values"][:8], start=1):
            lid = f"lore-value-{i:02d}"
            if model.get_node(lid) is None:
                model._apply_add_node(NodeType.LORE.value, {
                    "id": lid,
                    "description": value,
                    "context": "preference",
                })
                added["lore"] += 1

        for i, style in enumerate(identity["style"][:5], start=1):
            lid = f"lore-style-{i:02d}"
            if model.get_node(lid) is None:
                model._apply_add_node(NodeType.LORE.value, {
                    "id": lid,
                    "description": style,
                    "context": "preference",
                })
                added["lore"] += 1

        for i, pref in enumerate(identity["user_prefs"][:5], start=1):
            lid = f"lore-user-{i:02d}"
            if model.get_node(lid) is None:
                model._apply_add_node(NodeType.LORE.value, {
                    "id": lid,
                    "description": f"About my visitor: {pref}",
                    "context": "narrative",
                })
                added["lore"] += 1

    total_added = sum(added.values())
    logger.info(
        "World-native seed applied: %d items added, %d skipped. "
        "Graph: %d nodes, %d edges.",
        total_added, skipped, model.node_count, model.edge_count,
    )
    return added
