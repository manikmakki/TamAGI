# ADR-0004: Evidence, interpretation, and imagination remain distinct

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

Narrative, subjective belief, simulated events, and host actions currently lose their evidence distinctions.

## Decision and alternatives

**Decision:** Persist three distinguishable categories:

- Platform events and tool receipts: what an operation attempted or reported.
- Agent-authored memories, beliefs, and preferences: interpretations with attribution and uncertainty.
- Imaginative expression: stories, atmosphere, and fictional experiences.

A successful tool receipt does not automatically establish the agent’s broader interpretation. A simulated event can be authoritative within the simulation without implying a physical-world event.

**Reject:** Promoting fluent narration, repeated claims, or the presence of a citation into verified history.

**Consequence:** Memory, graph updates, summaries, and consolidation retain source references and claim categories. Fiction remains welcome; technical labels stay in inspection views rather than interrupting ordinary conversation.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
