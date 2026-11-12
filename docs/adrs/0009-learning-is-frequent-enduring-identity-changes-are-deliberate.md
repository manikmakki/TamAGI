# ADR-0009: Learning is frequent; enduring identity changes are deliberate

Status: Accepted design; implementation evidence tracked in the roadmap.

The progression portion is superseded by [ADR-0015: Growth through sustained behavior](0015-growth-through-sustained-behavior.md). Identity, reflection, learning, and resource-separation decisions remain in force.

## Context

Consolidation rewrites identity from narrative, while tool counts mechanically deplete curiosity and award XP.

## Decision and alternatives

**Decision:** Keep recognizable IDENTITY, SOUL, and USER documents, managed through a versioned identity service. Preserve authorship, rationale, previous revisions, and relevant sources.

Reflection may produce a practical lesson, belief correction, preference observation, unresolved question, proposed identity revision, or no change. A frustrating episode need not become a trait.

Founding identity remains historical; current self-description can evolve. Normal bounded revisions do not require approval for every change.

**Reject:** Obligatory whole-file identity rewrites and consolidation based predominantly on location/mood prose.

**Consequence:** All supported identity-writing paths use revision checks. Direct host-root edits are detected and recorded as external revisions rather than silently accepted as service-authored changes.

Keep playful stats and avatar expression, but separate them from actual resources. Remove per-tool curiosity depletion and XP farming. Make elapsed-time updates incremental and idempotent; low scores must not compel existential or recovery narratives.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
