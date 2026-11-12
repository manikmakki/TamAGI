# ADR-0007: Pursuits and commitments persist without turning every interest into a task

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

Graph quests and TASKS.md lack one durable pursuit lifecycle and execution references.

## Decision and alternatives

**Decision:** Introduce pursuits with stable IDs, origin, status, next step, optional completion condition, blocker/wake condition, and artifact/evidence references.

Support active, paused, blocked, completed, and abandoned states. Distinguish voluntary interests from accepted commitments. Creative play need not acquire a completion test.

Make graph quests and `TASKS.md` views of these records rather than separate task systems. Preserve the current task tool as a compatibility adapter.

**Reject:** String-only task lists, automatic obligation creation from every conversation, and inferring completion from narrative.

**Consequence:** Workflow dependencies must be validated before execution. Missing or failed prerequisites block dependent work; cycles are rejected. Relevant unfinished work is recovered during hydration and after interruption.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
