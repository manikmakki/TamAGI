# TamAGI documentation

Start with the [project README](../README.md) for a new home-platform installation. These guides describe the `codex/dependable-home` development branch. The public default `main` branch still uses the older runtime as of the 2026-11-06 audit.

## Using TamAGI

| Guide | Contents |
| --- | --- |
| [Running and recovery](running.md) | Configuration, scheduling, access, existing residents, services, Docker, backups, and troubleshooting. |
| [Integrations and memory](integrations.md) | Search, MCP, credential handling, and optional retrieval indexes. |
| [Workspace and tool development](workspace-layout.md) | Active paths, Inventory, worker contract, tool evaluation and activation. |
| [Appearance](sprite-customization.md) | Character selection, rigs, sprite uploads, and API reference. |
| [Navigation and work](navigation-and-work.md) | App drawer, admitted requests, Stop, browser disconnection, and restart recovery. |

## Architecture and status

[Implementation and evidence](platform-roadmap.md) maps the current code to its verification and limitations. Accepted [architecture decisions](adrs/README.md) describe design intent; their acceptance alone does not prove implementation or behavioral success.

The running platform has **one primary resident per instance** and invitation-only companions. ADR-0019’s [multi-resident roadmap](multi-resident-roadmap.md) is pending implementation. Shared resident households, fair cross-process inference scheduling, resident-initiated messaging, durable unread-thread counts, Chat’s proposed New Thread control, and PWA Web Push are future work.

Burn-in observations remain separate from fixture tests. Documentation changes do not enroll residents, migrate live data, complete the pilot, or select a new application release.

## Retired designs

[The earlier Living World design](world-monologue-design.md) and [Evolution Lab proposal](agent-harness-design.md) are retired reference pages. Their original text remains in Git history; use the current guides and ADRs for operation.

## Documentation audit

The 2026-11-06 audit checks setup commands against the CLI/configuration code, API and UI references against the routes/renderers, and operational claims against source and disposable checks. Known graph-persistence failures remain a separate bug fix; the guides do not claim that all legacy writes are transactional.

For a reproducible claim, retain the source revision, exact command, result, and evidence class. Local `.platform/` reports are installation evidence rather than files shipped to every reader.
