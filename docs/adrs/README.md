# Architecture decisions

The parent-child framing describes the builder’s responsibility: honest conditions, meaningful choices, recoverable mistakes, and growing independence. It does not prescribe the resident’s persona.

| ADR | Decision | Status |
|---|---|---|
| 0001 | [Developmental commitments and meaningful choice](0001-developmental-commitments-and-meaningful-choice.md) | Accepted design |
| 0002 | [Host ownership and honest authority boundaries](0002-host-ownership-and-honest-authority-boundaries.md) | Accepted design |
| 0003 | [Fresh identity, separate lived state, and canonical storage](0003-fresh-identity-separate-lived-state-and-canonical-storage.md) | Accepted design |
| 0004 | [Evidence, interpretation, and imagination remain distinct](0004-evidence-interpretation-and-imagination-remain-distinct.md) | Accepted design |
| 0005 | [One execution lifecycle and durable action receipts](0005-one-execution-lifecycle-and-durable-action-receipts.md) | Accepted design |
| 0006 | [A small persistent home connected to a real workshop](0006-a-small-persistent-home-connected-to-a-real-workshop.md) | Accepted design |
| 0007 | [Pursuits and commitments persist without turning every interest into a task](0007-pursuits-and-commitments-persist-without-turning-every-interest-into-a-task.md) | Accepted design |
| 0008 | [Scheduling offers opportunities; rest actually stops cognition](0008-scheduling-offers-opportunities-rest-actually-stops-cognition.md) | Accepted design |
| 0009 | [Learning is frequent; enduring identity changes are deliberate](0009-learning-is-frequent-enduring-identity-changes-are-deliberate.md) | Accepted design |
| 0010 | [Durable companions, personal memories, and explicit sharing](0010-durable-companions-personal-memories-and-explicit-sharing.md) | Accepted design |
| 0011 | [Runtime adaptation and generated tools are versioned capabilities](0011-runtime-adaptation-and-generated-tools-are-versioned-capabilities.md) | Accepted design |
| 0012 | [Improvement claims require retained comparative evidence](0012-improvement-claims-require-retained-comparative-evidence.md) | Accepted design |
| 0013 | [Autonomous deployment uses versioned releases and recoverable transitions](0013-autonomous-deployment-uses-versioned-releases-and-recoverable-transitions.md) | Accepted design |
| 0014 | [Portability, operator recovery, and a companion-first interface](0014-portability-operator-recovery-and-a-companion-first-interface.md) | Accepted design |
| 0015 | [Growth through sustained behavior](0015-growth-through-sustained-behavior.md) | Accepted design |
| 0016 | [Room for self-directed work](0016-room-for-self-directed-work.md) | Accepted design |
| 0017 | [Bounded user turns and quiet continuations](0017-bounded-user-turns-and-quiet-continuations.md) | Accepted design |
| 0018 | [An app drawer, chat-first navigation, and independent work lifetimes](0018-app-drawer-and-independent-work-lifecycle.md) | Accepted design; implemented; mechanical evidence in roadmap |
| 0019 | [Distinct residents, a shared world, and fair access to cognition](0019-distinct-residents-and-a-shared-world.md) | Accepted design; documentation only; household implementation pending |

[Implementation and evidence](../platform-roadmap.md).

[Multi-resident implementation sequence and qualification gates](../multi-resident-roadmap.md). ADR-0019 does not enable household mode or change the running resident during burn-in.

ADR-0019's 2026-11-06 extension covers [resident-initiated contact and PWA notifications](0019-distinct-residents-and-a-shared-world.md#resident-initiated-contact-and-operator-notifications), plus durable unread-thread counts and New Thread access directly in Chat. Implementation remains pending.
