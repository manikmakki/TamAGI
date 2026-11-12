# Private access to retained conversations

An existing resident can keep useful continuity without importing every old
thread into household Chat. Identity and relationship files, saved memories and
their summaries remain the everyday context. Retained transcripts supply exact
surrounding evidence when an existing conversation-memory pointer needs it; they
do not need to be replayed or included in every request.

This guide covers the private legacy conversation lookup used by
`expand_conversation_memory`. It does not enroll a resident, select a release or
restore canonical household conversations. See [individual resident recovery](resident-recovery.md)
for the existing backup, preview and migration commands, and the
[production enablement plan](production-multi-resident-enablement.md) for a full
cutover.

## What the resident can retrieve

For a native resident, expansion first asks the coordinator for the conversation
referenced by the surfaced memory. If that conversation does not exist there,
the dedicated memory lookup may read a private SQL conversation with that exact
stable ID and owned by the primary resident. The ordinary conversation lookup,
Chat, Archive and admitted conversation snapshots continue to use canonical
household history.

| Result | Behavior |
| --- | --- |
| Accessible canonical conversation | Use the coordinator's conversation; it takes precedence over a private copy with the same ID. |
| Canonical conversation is missing; exact primary-owned private SQL conversation exists | Use its saved messages to expand the memory's surrounding window privately. |
| Coordinator denies access or transport fails | Preserve the error. These outcomes do not authorize a private fallback. |
| Neither supported source has that conversation | Return only the memory's stored window/snippet when available. Do not invent missing dialogue. |

Other residents' records and companion-owned conversations are outside this
fallback. It does not search history JSON files, recovery-context exports,
unrelated databases or predecessor archives. It does not publish old messages,
join anyone to a thread, create notifications or reply offers, or replay Growth
recognition. New household conversations are already canonical and need no
legacy import.

## Before migrating an existing resident

1. Identify the primary resident by stable ID, not a display name. A companion
   can have the same name. Record the selected application release, source paths,
   chosen rest/wait, and a few known conversation-memory IDs and their exact
   conversation IDs. Keep this inventory private.
2. Stop the source installation through its service/supervisor and confirm that
   all resident workers are stopped before taking a cold checkpoint. Retain the
   complete instance, coherent SQLite database and any WAL, private workspace and
   data, application source/environment, and service/release-selection metadata.
   Record external dependencies and credential stores separately.
3. Keep a verified, untouched checkpoint and work on a disposable copy. Older
   standalone portable exporters can include only `data/sprites` and
   `data/sprite_config.json` from private data. The runtime SQL database travels,
   but history JSON, artwork or other private files can still be missing. Inspect
   the archive manifest and omissions; do not infer full coverage from successful
   encryption or startup.
4. Use the selected release's existing preview/migration procedure to create a
   new one-resident household with the original resident ID. Where the legacy
   exporter omits eligible private files, qualify an expanded export or a
   hash-checked overlay from the cold checkpoint into the unstarted destination.
   Keep excluded credentials separate. Do not edit an encrypted archive without
   regenerating and validating its manifest.
5. Rehearse with guarded authority, an injected provider, no production
   credentials or active MCP services, and disabled scheduling/deployment. Verify
   memory IDs, exact owned conversation rows and message bytes before hydration,
   then validate the access behavior below. Retain the old root and release.

Existing files and summaries can be the first recovery layer if an operator
accepts that scope. Keep the full checkpoint so omitted transcripts and files
can be assessed later. A successful summary-only recovery is not evidence that
exact historical lookup works.

## If the resident has already migrated

1. Check which application release actually runs the worker. Updating a checkout
   alone does not activate this fix in a pinned service. Use a compatible,
   qualified release and retain the previous release/checkpoint.
2. Take a current resident checkpoint and, for a household change, a coherent
   whole-household checkpoint. Stop the affected installation before any offline
   inspection or replacement. Work against a disposable restored copy first.
3. Check whether the current private SQL database still contains the exact
   conversation and memory IDs. If it does, this access fix needs no transcript
   import or copy of the original database. Rehearse the lookup on that copy.
4. If the conversation is absent, inspect the original checkpoint offline. The
   fallback cannot read missing rows from history files or another installation.
   Choose between keeping the available files/summaries, explicit checkpoint
   replacement with its reported continuity gap, or separately qualifying a
   selective recovery procedure. Selective merging is not provided here.

Do not copy the old database over a newer resident home. That can erase later
memories, private work and installation metadata, and it bypasses the restore
transition. Do not point a new worker at an original live database or run two
active continuations of the same resident.

## Validation and cost

The [legacy conversation regressions](../tests/test_legacy_conversation_access.py)
exercise exact lookup and ownership boundaries without using a live resident.

Use offline fixtures or a guarded disposable restore to check:

- A known memory expands to the expected saved message window by exact ID.
- A canonical thread with the same ID wins over private legacy content.
- A missing canonical thread permits only the primary resident's exact SQL record;
  peer/companion ownership, a missing record, access denial and transport failure
  retain their intended boundaries.
- No legacy thread appears in Chat/Archive merely because it was expanded. No
  reply, invitation, notification, inference grant or Growth replay is created.
- A clean restart and native resident archive round trip retain the IDs and
  message bytes needed by the lookup.

This exact lookup adds no model call, embedding/index rebuild or bulk transcript
conversion. A surrounding conversation or interactive check may itself use the
configured model; keep mechanical checks offline. Passive retrieval settings
still determine which memory pointers are surfaced. This change does not turn
off existing retrieval or require adding an index.

Generated-tool interpreter and absolute-path portability are separate follow-up
checks. This lookup does not execute those tools. Record any known mismatch and
validate an affected tool before enabling or invoking it; that issue does not
prevent qualifying the private conversation lookup itself.

## Backup and restore limits

Native resident archives include the private SQL database and eligible private
files. They can also retain accessible household transcripts in
`data/recovery-evidence/` as historical evidence. Those context files are not
restored as active canonical threads and are not searched by this fallback.
Canonical shared messages, membership/read state, world state and public-library
bytes require coordinated household recovery.

Preserve the whole checkpoint even when only files and summaries are used day to
day. Credentials, session authority, machine services, model weights, symlinks
and other portable exclusions need their own inventory and deliberate recovery.
Verify file hashes and logical records on a restored copy, record every omitted
dependency and continuity gap, and keep an independent backup before removing
the original. Source restoration, private transcript access and full household
recovery remain separate claims.
