# ADR-0003: Fresh identity, separate lived state, and canonical storage

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

History, identity, graph, pending events, and world state currently have independent file authorities.

## Decision and alternatives

**Decision:** Use standard-library SQLite with WAL as the canonical store for actors, conversations, memories, events, executions, pursuits, world objects, identity revisions, experiments, and deployments. Store large artifacts as referenced files with content hashes.

Separate application releases from lived state. Give each resident a stable `resident_id` and each installation a separate `installation_id`.

Archive the existing instance before cutover. Create new identity, world, memory, and retrieval namespaces without importing its autobiography.

**Reject:** Maintaining several independently writable JSON/Markdown authorities or treating a vector database as the only copy of experience.

**Consequence:** ChromaDB and Elasticsearch become rebuildable retrieval projections. Existing graph, history, identity Markdown, and task views become service-managed projections or exports. Predecessor preservation includes its original files and logical memory export; no retrospective evidence is invented.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
