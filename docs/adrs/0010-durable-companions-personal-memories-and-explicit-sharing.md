# ADR-0010: Durable companions, personal memories, and explicit sharing

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

The orchestrator reconstructs stateless specialist prompts and loses individual continuity.

## Decision and alternatives

**Decision:** Add durable companion actors with names, founding seeds, evolving self-descriptions, interests, personal memories, and attributed conversation histories.

Provide separate conversation and task invocation modes. Conversation requires no deliverable. Task responses can accept, complete, decline, renegotiate, report blockage, or request help.

Companions remain inactive between invocations. They receive no independent population scheduler. Shared messages, task results, artifacts, and selected notes enter a household notebook explicitly.

**Reject:** A shared autobiography, automatic copying of TamAGI’s identity, compulsory permanent professions, and an always-running resident community.

**Consequence:** Operator inspection covers all actors. Memory separation controls normal context and retrieval, not confidentiality against host root.

Serialize simultaneous activations of the same companion. Retain four workers per workflow, enforce depth at dispatch, and prevent companion calls from creating unbounded descendant activations. Preserve attribution during synthesis.

Repair the existing positional dispatch defect, schema-only permission filtering, round-exhaustion reporting, and blind workflow retries before extending orchestration.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
