# ADR-0008: Scheduling offers opportunities; rest actually stops cognition

Status: Accepted design; implementation evidence tracked in the roadmap.

The initial call-count defaults below are superseded by [ADR-0016](0016-room-for-self-directed-work.md). Scheduling, rest, interruption, and durable accounting remain in force.

## Context

Cron forces repeated narrative opportunities; sleep, waiting, and allowances do not survive restart.

## Decision and alternatives

**Decision:** Retain the fifteen-minute schedule as an opportunity to consider autonomous activity. Add durable sleep, event-specific waiting, silent completion, and bounded continuation.

Keep periodic opportunities for independent choice even when no external event arrives. Explicit sleep or waiting suppresses unnecessary calls until its wake condition.

User input takes priority over background inference. Preserve completed effects and record uncertain outcomes when interruption cannot establish what happened.

**Reject:** Mandatory monologues every tick, permanent inactivity under event-only scheduling, and restart-reset resource allowances.

**Consequence:** Persist rolling budgets and wake state. Use UTC timestamps and one configured timezone for local schedules. Provider outages produce a visible waiting state with bounded backoff.

Initial configurable defaults:

- One concurrent provider call.
- Eight model calls per background activation.
- Three calls per companion activation, drawn from the parent allowance.
- Ninety-six ordinary background calls and thirty-two evaluation calls per rolling 24 hours.
- Twenty calls per user turn; user turns do not replenish background allowances.
- Existing fifteen-minute post-conversation decompression.

Planner, reviewer, synthesis, companion, and retry calls all count. Limits are resource policy, never rewards for agreement or productivity.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
