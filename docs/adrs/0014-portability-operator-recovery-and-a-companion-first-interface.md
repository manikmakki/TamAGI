# ADR-0014: Portability, operator recovery, and a companion-first interface

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

There is no portable encrypted checkpoint and staged restore interface.

## Decision and alternatives

**Decision:** Keep avatar and conversation central. Add concise views for current activity, home/workshop artifacts, companions, shared notebook, and optional evidence inspection. Put deployments, archive management, and recovery in operator settings.

Implement encrypted portable backup download and staged restore upload. Use streaming tar/gzip with GnuPG symmetric encryption, passing the password through a dedicated file descriptor rather than arguments or logs. GnuPG supports this interface directly. [GnuPG documentation](https://gnupg.org/documentation/manuals/gnupg24/gpg.1.html)

Include canonical lived state, identities, companions, histories, registered artifacts, non-secret configuration, and exact application source/lockfiles needed to reconstruct a self-modified release. Exclude credential stores, session secrets, model weights, derived indexes, and unrelated host files.

**Reject:** Copying a live database file without a consistent checkpoint, restoring directly over active state, and treating application backup as a bootable machine image.

**Consequence:** Pause supported mutations while collecting a consistent checkpoint. Validate decryption, integrity, manifest, schema compatibility, sizes, and safe extraction before presenting a restore preview. Commit restoration only after explicit operator confirmation.

Default restoration preserves `resident_id`, creates a new `installation_id`, rotates sessions, reconnects credentials, and rebuilds indexes. Require acknowledgment that the source installation is stopped. Record any known continuity gap; concurrent cloning is outside this plan.

A downloaded backup becomes independent recovery only when retained elsewhere. Root-created changes outside registered application state appear as dependencies or omissions. Arbitrary artifacts can contain embedded secrets despite exclusion of dedicated credential stores.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
