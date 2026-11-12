# TamAGI documentation

Start with the [project README](../README.md) for a new home-platform installation. These guides distinguish standalone operation from the opt-in native household. A merged source revision does not select the release used by an existing installation.

## Using TamAGI

| Guide | Contents |
| --- | --- |
| [Running and recovery](running.md) | Configuration, scheduling, access, existing residents, services, Docker, backups, and troubleshooting. |
| [Create a TamAGI](creating-a-tamagi.md) | Add a new household resident, choose an avatar and starting personality, and open their first conversation. |
| [Production multi-resident enablement](production-multi-resident-enablement.md) | Ember preservation, migration gaps, manual checkpoints, staged cutover and recovery decisions. |
| [Private access to retained conversations](legacy-conversation-access.md) | Exact legacy memory lookup, existing-user migration choices, low-cost validation and recovery limits. |
| [Integrations and memory](integrations.md) | Search, MCP, credential handling, and optional retrieval indexes. |
| [Workspace and tool development](workspace-layout.md) | Active paths, Inventory, worker contract, tool evaluation and activation. |
| [Appearance](sprite-customization.md) | Character selection, rigs, sprite uploads, and API reference. |
| [Navigation and work](navigation-and-work.md) | App drawer, admitted requests, Stop, browser disconnection, and restart recovery. |

## Architecture and status

[Implementation and evidence](platform-roadmap.md) maps the current code to its verification and limitations. Accepted [architecture decisions](adrs/README.md) describe design intent; their acceptance alone does not prove implementation or behavioral success.

The standalone platform has **one primary resident per instance** and invitation-only companions. PR #8 adds an opt-in [native household](household-native.md), shared inference scheduling, resident contact/invitations, durable unread threads, Chat’s New Thread control and coordinated world turns. [Individual resident recovery](resident-recovery.md) supports browser backup/restore and offline migration; [private legacy conversation access](legacy-conversation-access.md) preserves exact memory lookups without a bulk Chat import. Production resident adoption, coordinated shared-state recovery and PWA Web Push retain their separate scope in the [multi-resident roadmap](multi-resident-roadmap.md).

Burn-in observations remain separate from fixture tests. Documentation changes do not enroll residents, migrate live data, complete the pilot, or select a new application release.

## Retired designs

[The earlier Living World design](world-monologue-design.md) and [Evolution Lab proposal](agent-harness-design.md) are retired reference pages. Their original text remains in Git history; use the current guides and ADRs for operation.

## Documentation audit

The 2026-11-06 audit checks setup commands against the CLI/configuration code, API and UI references against the routes/renderers, and operational claims against source and disposable checks. The subsequent [graph persistence fix](qualification/issue-5-world-graph-persistence-2026-11-12.md) adds failure reporting and atomic JSON replacement; the guides do not claim that all legacy writes are transactional.

For a reproducible claim, retain the source revision, exact command, result, and evidence class. Local `.platform/` reports are installation evidence rather than files shipped to every reader.
