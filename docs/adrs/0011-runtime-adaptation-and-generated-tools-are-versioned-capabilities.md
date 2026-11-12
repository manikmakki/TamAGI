# ADR-0011: Runtime adaptation and generated tools are versioned capabilities

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

Workspace Python skills currently import directly into the backend process.

## Decision and alternatives

**Decision:** Maintain a typed registry of mutable prompt fragments, retrieval settings, working procedures, and tool versions. Record inspection, proposed change, validation, activation, and revert.

Prompt/config changes become visible at the next model-call boundary. Each dispatched operation retains the versions it started with.

Generated executable tools run through a subprocess/MCP-style contract rather than being imported into FastAPI. In the host-root profile, those processes may have host-root authority; process separation provides fault isolation, not confinement.

**Reject:** Untracked global prompt replacement, generated Python imported directly into the backend, and “rollback” that leaves obsolete tools callable.

**Consequence:** Tools have explicit inputs, outputs, version, dependencies, health checks, and retirement behavior. Activation reconciles the actual registry. Reverting runtime configuration does not erase unrelated memories or undo external effects.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
