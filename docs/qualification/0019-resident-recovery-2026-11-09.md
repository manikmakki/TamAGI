# ADR-0019 individual resident recovery — 2026-11-09

The operator prioritized individual recovery after regression, private data loss,
and migration to fresh hardware. Coordinated shared-state/artifact recovery is a
second layer. This development candidate implements an offline resident command;
browser recovery and deployment remain disabled.

## Retained evidence

- [Full offline qualification](0019-resident-recovery-2026-11-09/suite.json):
  **319 tests passed in 231.995 seconds**, 171 guarded Python processes, zero
  denied network operations, no real model. Source before/after hashes match.
  [Test output](0019-resident-recovery-2026-11-09/suite.log).
- [Operator CLI round trips](0019-resident-recovery-2026-11-09/cli.json): three
  independent backups/previews, one replacement and one migration in fresh
  disposable homes; nine guarded Python processes, zero denials. Passwords used
  inherited file descriptors, not command arguments or environment variables.
- [Actual resident checkpoint checks](0019-resident-recovery-2026-11-09/retained-data.json):
  Cedar, Willow and River from the retained test household each passed all three
  scenarios in inactive disposable migrations. Preserved canonical record hashes
  and file hashes are compared by the retained
  [check script](0019-resident-recovery-2026-11-09/retained-data-check.py).
- [Repeated backup](0019-resident-recovery-2026-11-09/repeated-backup.json): a new
  backup of restored Cedar retained both the earlier household transcript evidence
  and the new household context. The initial native archive layout remains readable;
  new exports use unique evidence filenames so subsequent backups do not erase it.
- [Actual-data guard summary](0019-resident-recovery-2026-11-09/retained-data-guard.json):
  three guarded Python processes, zero denied operations, no model calls or active
  restored workers. This ledger excludes the original household's normal resumed
  World Thread activity.

| Actual resident | Private memories preserved | File hashes verified per scenario | Scenarios |
|---|---:|---:|---|
| Cedar | 38 | 336 | Migration, private regression, complete private-root loss |
| Willow | 36 | 337 | Migration, private regression, complete private-root loss |
| River | 17 | 334 | Migration, private regression, complete private-root loss |

The file counts include archived source and resident data/workspace files; the
SQLite file itself is compared through selected canonical lived-record hashes
rather than its physical bytes, since restore deliberately changes installation
metadata and records recovery. Growth recognition counts in these actual homes
were zero; nonzero Growth preservation/replay is covered separately by the full
suite's existing portable recovery tests.

## Boundaries exercised

The new tests restore one of three residents while retaining later peer memories,
shared messages, publications and the World Thread clock. They replace a missing
private root, rebind paths, accept the existing portable v1 archive for one-resident
migration, reject incompatible native release bindings, require matching identity
and explicit confirmation/stopped-source acknowledgment, reject a held coordinator
or resident lock, and exercise corruption/wrong passwords without retained plaintext.

Failure injection before selection leaves the old root selected. Interruption
between coordinator database commit and manifest selection is completed through
the fsynced explicit transition journal on startup. Private interactions/evaluations
are interrupted and pending outbox effects become outcome-unknown. Existing safe
extraction/resource-limit tests remain in the full suite. Real disposable worker
processes hydrate a replacement alongside its unchanged peers, then hydrate a
fresh migration after the original directories have been removed, without model
calls or replayed work.

## Retained household operation

Preflight found no active resident runs, queued/running private interactions or
evaluations, active canonical chat work, or unsettled broker attempts. Only the
test coordinator PID 359591 received SIGINT. It and its three workers exited
cleanly; no force termination was needed. All three resident archives were taken
under the stopped-household/instance locks. Original resident roots and the
household manifest were not replaced.

The retained home `/tmp/tamagi-native-vk2tks0j/home` resumed on `0.0.0.0:7744` as
PID 367928, interactive session 49338, with its existing password, all three workers,
and the shared `*/15 * * * *` World Thread clock enabled. Normal restart recovery
still applies to prior offers/work. Production Ember PID 152015 and its
`127.0.0.1:7742` listener were unchanged.

Encrypted archives and disposable recovered roots remain outside the repository
in `/tmp/tamagi-current-resident-recovery-20261109`; their password is private there
and is not part of published evidence. These local rehearsal copies are not an
off-host disaster recovery guarantee.

## Remaining qualification

This evidence does not qualify hot resident maintenance, browser restore,
selective memory merging, arbitrary cross-release recovery, migration into an
established household, historical transcript viewing in the new household's UI,
canonical shared-state/public-library recovery, or actual Echo rehydration.
The original shared transcripts travel as historical evidence, not active migrated
threads. Restoring a private checkpoint does not itself roll back shared code.

NATIVE-010's earlier shutdown hang with stranded queued evaluations remains open:
this clean idle shutdown did not reproduce that condition. Household deployment,
production Ember enrollment/history migration and sustained real-model burn-in
retain their separate acceptance gates. The offline guard covers Python sockets
and DNS, not OS confinement of arbitrary native programs.
