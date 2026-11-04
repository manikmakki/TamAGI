# ADR-0005: One execution lifecycle and durable action receipts

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

Tool effects precede narrative parsing, and structured results are discarded from persisted world ticks.

## Decision and alternatives

**Decision:** Route chat, autonomy, companions, planning, and tools through one execution service. Supply trusted actor, authority, budget, and provenance through an `ExecutionContext`, separate from model-generated arguments.

Record a tool attempt before dispatch and its result afterward. Preserve completed effects even if later narration fails. On interruption or restart, an unfinished dispatched operation becomes `outcome_unknown` until reconciled.

**Reject:** Treating a parseable `[New State]` as proof of successful action, blindly retrying effectful tasks, or promising exactly-once execution for arbitrary shell and MCP operations.

**Consequence:** Narrative rendering becomes optional after execution. Built-in operations use idempotent identifiers where supported; unknown external effects require inspection before retry. Dispatch validates tool eligibility rather than merely hiding schemas.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
