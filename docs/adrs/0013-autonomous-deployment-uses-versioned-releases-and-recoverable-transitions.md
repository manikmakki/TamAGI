# ADR-0013: Autonomous deployment uses versioned releases and recoverable transitions

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

The service executes a mutable checkout; Python edits do not establish a tested and recoverable release.

## Decision and alternatives

**Decision:** Keep `/opt/TamAGI` canonical. Prepare changes in isolated Git worktrees and build exact releases with pinned dependencies and recorded source revisions.

Separate source, selected release, previous known-good release, and lived data. Use a supervisor outside the ordinary agent process to reconcile deployment state and restart the selected release.

The normal autonomous lifecycle is:

> Prepare → validate → build → save handoff → quiesce → activate → verify → commit or recover.

**Reject:** Editing the running checkout and assuming Python code changes immediately take effect, or reverting code while leaving incompatible data.

**Consequence:** Require rollback-compatible migrations during ordinary deployment. An incompatible migration needs a tested snapshot-restoration path. Code rollback preserves subsequent lived experience wherever schema compatibility permits.

Separate liveness, data readiness, model availability, and hydration status. Model outages alone must not trigger deployment rollback loops.

The completed platform permits automatic core deployment without per-change approval. Supervisor and deployment-policy changes receive their own recovery tests; their root-mutability remains explicit.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
