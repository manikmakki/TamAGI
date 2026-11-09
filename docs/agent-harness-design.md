# Evolution Lab — retired design

This was an early proposal for a bounded self-improvement harness. It is retained as a reference destination; the original proposal is available in this file’s Git history. It is not an installation guide or a description of the current evaluator.

The home platform implements versioned runtime proposals, subprocess tools, retained trials, comparative evaluation, activation/revert, and separately validated application releases. Use:

- [Workspace and tool development](workspace-layout.md) for the supported tool lifecycle.
- [Implementation and evidence](platform-roadmap.md) for current mechanics and their limits.
- [ADR-0011](adrs/0011-runtime-adaptation-and-generated-tools-are-versioned-capabilities.md) for runtime adaptation.
- [ADR-0012](adrs/0012-improvement-claims-require-retained-comparative-evidence.md) for comparative evidence.
- [ADR-0013](adrs/0013-autonomous-deployment-uses-versioned-releases-and-recoverable-transitions.md) for releases and recovery.

The earlier proposal’s immutable same-host safety boundary, mandatory narrative grading, static sample-response scores as proof of improvement, and proposed Evolution Lab interfaces are superseded. Under host-root authority, application controls are cooperative and mutable. The implemented default evaluator checks a limited set of behaviors; it cannot establish arbitrary tool usefulness or sustained autonomous improvement. Those claims require task-specific and live evidence.
