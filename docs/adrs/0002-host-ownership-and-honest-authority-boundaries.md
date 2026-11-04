# ADR-0002: Host ownership and honest authority boundaries

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

Existing command tiers and container credentials do not describe the requested dedicated host-root deployment.

## Decision and alternatives

**Decision:** Implement an explicit `unrestricted_host` deployment profile that runs with host UID 0, host filesystem access, and ordinary process/service authority. It must not silently substitute container-root access or retain the existing command approval tiers for local host operations.

Publishing, messaging, purchases, and externally consequential service actions continue to require operator authorization unless previously delegated.

Under this profile, budgets, audit records, evaluation gates, and deployment procedures are cooperative operating rules. Root can alter them; they are not tamper-proof boundaries.

**Reject:** Describing same-host controls as immutable, or permanently confining TamAGI while claiming it owns the machine.

**Consequence:** Keep authority behind a common execution interface so a future restricted personal-computer profile can enforce different permissions. Portable restoration must require an installation-specific authority choice rather than silently inheriting root.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
