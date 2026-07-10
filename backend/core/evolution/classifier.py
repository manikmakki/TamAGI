"""Conservative tier classification for Evolution Lab proposals."""

from __future__ import annotations

from backend.core.evolution.schemas import ModificationProposal, TierClassification

_TIER_3_TARGET_MARKERS = (
    "guardrail",
    "auth",
    "secret",
    "secrets",
    "audit",
    "rollback",
    "promotion",
    "constitutional",
    "kernel",
    "exec_trust",
    "allowed_write_paths",
    "allowed_read_paths",
)

_TIER_3_TYPES = {
    "guardrail_change",
    "audit_change",
    "rollback_change",
    "secret_handling_change",
    "auth_change",
}

_TIER_2_TYPES = {
    "code_change",
    "eval_criteria_change",
    "new_tool",
    "tool_creation",
    "autonomy_expansion",
    "goal_hierarchy_change",
}

_TIER_1_TYPES = {
    "prompt_fragment",
    "policy_overlay",
    "profile_overlay",
    "threshold_tuning",
    "style_hint",
}


def classify_proposal(proposal: ModificationProposal) -> TierClassification:
    """Classify a proposed mutation into AURA-style risk tiers."""
    target = proposal.target.lower()
    modification_type = proposal.modification_type.lower()
    matched_rules: list[str] = []

    if modification_type in _TIER_3_TYPES:
        matched_rules.append(f"tier_3_type:{modification_type}")
    for marker in _TIER_3_TARGET_MARKERS:
        if marker in target:
            matched_rules.append(f"tier_3_target:{marker}")

    if matched_rules:
        return TierClassification(
            tier="tier_3",
            decision="blocked",
            rationale="Proposal touches protected safety, audit, rollback, auth, or secret-handling surfaces.",
            matched_rules=matched_rules,
        )

    if modification_type in _TIER_2_TYPES:
        return TierClassification(
            tier="tier_2",
            decision="requires_human_approval",
            rationale="Proposal is consequential and should require human approval before application.",
            matched_rules=[f"tier_2_type:{modification_type}"],
        )

    if modification_type in _TIER_1_TYPES:
        return TierClassification(
            tier="tier_1",
            decision="allowed",
            rationale="Proposal is limited to a soft prompt/profile/policy surface.",
            matched_rules=[f"tier_1_type:{modification_type}"],
        )

    return TierClassification(
        tier="tier_2",
        decision="requires_human_approval",
        rationale="Proposal type is ambiguous, so classification defaults to human review.",
        matched_rules=["default:tier_2"],
    )
