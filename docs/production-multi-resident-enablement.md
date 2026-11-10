# Production multi-resident enablement: preserve Ember first

Planning baseline: 2026-11-09, America/New_York, after PR #8 merged as
`d01b06c`; inspected checkout `8945161`. This is an execution plan, not evidence
that production qualification has happened. Initial planning involved read-only
production inspection; subsequent authorized checkpoint/rehearsal results need
their own evidence and do not by themselves select a production release.

## Activation progress

On November 10 the operator [activated the rest-policy release](qualification/0019-rest-chat-eligibility-2026-11-10.md)
for the existing Ember and Echo household. Ordinary chosen rest now pauses World
Thread work while direct chat, joined-peer replies and invitation decisions remain
eligible. A complete stopped checkpoint, exact release transition, two-worker
offline preservation and authenticated live health checks passed. Normal scheduled
World Thread activity resumed; the upgrade created no resident and requested no
operator test chat or manual tick. Automatic household deployment remains off.

On November 10 the operator [activated the creation release](qualification/0019-resident-creation-activation-2026-11-10.md),
making the [creation workflow](creating-a-tamagi.md) available in Ember's existing
household. Its stopped-source checkpoint, explicit pinned release transition,
offline startup and authenticated live schema/asset checks passed. The upgrade
retained 187 current memories, 28 Growth recognitions, legacy private dialogue,
the current canonical conversation, operator configuration/session and Ember's
chosen rest until 1:00 PM EST. It added zero model generation attempts and
created no resident. New-peer creation remains an operator action; recovering a
historical resident or promoting a companion is separate work.

On November 9 the operator selected manual one-resident adoption. Ember's original
ID has been migrated into the durable `/opt/TamAGI/data/households/ember` home,
using pinned source and environment, explicit original authority/login/provider
mapping, and the retained chosen rest. Offline native startup and a same-path,
same-host whole-household cold restore passed with zero model calls. Initial
`tamagi-household.service` is serving on loopback port 7742, with authenticated
identity, memory, Growth, private-history and rest checks passed before and after
a controlled service restart. Zero generation attempts were recorded; the
original rest and quarter-hour New York clock are retained. Details and remaining
scope are in the
[activation record](qualification/0019-ember-household-activation-2026-11-09.md).
That adoption enrolled no second resident and kept automatic household deployment
off. The original adoption record and baseline below remain historical
observations; the latest November 10 record identifies the current two-resident
release and wake state. The sequence remains useful for subsequent rehearsals
and recovery.

## Recommended outcome

Continue **the existing Ember**, initially as the only resident in a native
household. Preserve the original resident ID and lived records. Add a second
resident only after Ember's continuity, restart and recovery checks pass.
Creating a fresh resident named Ember is not migration. Recovering the historical
Echo is a separate assessment; a fresh resident with that name would be a new
identity.

Use a planned maintenance window and manual backup/restore if necessary.
Automated deployment and browser household recovery are not prerequisites for
this limited rollout if a pinned manual service and complete cold restore have
been rehearsed. Successful startup alone is insufficient. The agreed history
scope preserves everyday continuity through existing files, memories and
summaries, with exact private transcript access through existing memory pointers.
Importing all old threads into Chat/Archive is follow-up work, not a cutover
requirement. See the [existing-user action plan](legacy-conversation-access.md).

## Observed starting point

Recheck these observations immediately before any maintenance; they are not
permanent service identifiers.

| Item | Observed baseline |
| --- | --- |
| Production | Standalone home platform under `tamagi-platform.service`; supervisor PID 724, Ember PID 152015; `127.0.0.1:7742`. |
| Selected application | `.platform/current` selects release content hash `fde27560bde70e63201a6c6928021fc2449ac98a3cf3c84084fbb94812961abb`. Merging source did not activate it. |
| Live instance | `/opt/TamAGI/data/instances/home-a01785dc2a784b0da6dc21fb27a7998d`; `.platform/active.json` identifies its configuration. |
| Primary identity | `resident-e3395ef821ce4bdbb5bc58bfe1fe27b5`; personality name Ember, legacy actor/config name Tama. A separate companion is also named Ember: preserve both identities and map by ID. |
| Storage | Runtime schema 1; SQLite, workspace and private `data/` files. An active SQLite WAL exists: copying the database alone while running is insufficient. |
| Lived-state sample | 182 memories, 196 learning records, 12 conversations/390 stored messages, 4 companion actors/5 threads, 28 Growth recognitions, 12 pursuits, 14 artifacts, 30 world objects and 100 world turns. These counts are a planning baseline, not final checkpoint totals. |
| Model and schedule | `qwen36-heretic`, reply allowance 8192, timeout 180 seconds; World Thread every 15 minutes, all hours, timezone `America/New_York`. |
| Chosen rest | Resting until November 10, 2026, 6:00 AM EST; no wake event. Migration must not treat maintenance or a new household as permission to wake early. |
| Pilot | Stored status is `running`; the seven-day time boundary is November 10, 2026, 1:08 PM EST. Review observations after that boundary; elapsed time does not establish completion. |
| Authority and effects | `unrestricted_host`; one existing outcome-unknown receipt. Preserve that evidence; do not retry the effect merely because the resident moved. |
| Deployment | Standalone automatic deployment is currently enabled. Freeze its selection during the eventual checkpoint/cutover procedure. |
| Development household | Separate retained Cedar/Willow/River household on port 7744 under `/tmp`; it is neither Ember's destination nor a durable production backup. |
| Existing evidence | [Final branch review](qualification/0019-final-branch-review-2026-11-09.md): 331 offline tests and injected-provider Chromium checks. These are not an Ember migration rehearsal or sustained production qualification. |

## Preservation contract

Keep the full original snapshot even where some fields deliberately change in
the destination. Compare logical records and file hashes; restore records,
installation IDs, path bindings and normal hydration bookkeeping mean the new
SQLite file will not be byte-identical.

| State | Required result |
| --- | --- |
| Personal identity | Same `resident_id`, actor, identity revisions and resident-authored identity files; no fresh onboarding or predecessor identity imported. |
| Memory and development | Same memory IDs/content/provenance, pursuits, commitments, Growth recognitions, stage history and evidence links. No duplicated recognition or reset of rolling allowances. |
| Conversations | Retain original threads, messages, authors, timestamps, order, relationships and available attachments. Verify exact private legacy SQL transcript access through surfaced conversation-memory pointers. Canonical history takes precedence; denied access never permits a fallback. Old threads need not appear in native Chat/Archive. Preserve companion-private threads under their original privacy boundary. |
| Workspace and creations | Preserve eligible file bytes, generated skills/tools and their versions, artifacts, saved versions and artwork. Inventory and tool paths resolve to the new private root. Record any excluded file or unsupported external dependency. |
| World and Journal | Preserve graph/world records, private history, World Thread entries/state and consolidation state. Verify references and ownership when adapting private world objects into household places. |
| Companions | Retain their identities, private memories and conversation records; do not promote them into independently scheduled residents. Qualify existing companion invocation behavior separately from deferred household group activation. |
| Rest and waits | Preserve chosen rest/wait reason, deadline or event condition and post-conversation pause. No missed-tick catch-up. Completion waits for interrupted work must reflect the interruption rather than wait forever. |
| Execution evidence | Preserve receipts and historical results. Unfinished work is interrupted; uncertain effects remain outcome-unknown. No automatic replay of tools, invitations, messages or external actions. |
| New installation | New household and private installation authority, rebound paths, fresh operator session/password and explicit provider/integration reconnection. Preserve Ember's name and appearance in both worker configuration and household metadata. |

Keep predecessor/operator archives separately recoverable and outside hydration
and retrieval. Dedicated secrets, machine configuration and external services
need a separate recovery inventory; the portable archive intentionally excludes
several of them.

## Known gaps to close before cutover

1. **Full legacy data export.** The running standalone
   [portable exporter](../backend/core/runtime/backup.py) includes only
   `data/sprites` and `data/sprite_config.json`, alongside the runtime database,
   workspace and source. A stock archive can omit history and other private data.
   Take a complete stopped-instance snapshot. For migration, either qualify an
   expanded legacy exporter or overlay the missing eligible private files from
   that snapshot into the unstarted migrated root. Produce a manifest of the
   additional files and verify hashes before first hydration. Do not modify an
   encrypted archive without regenerating and validating its manifest. Preserve
   excluded files separately and reconnect credentials deliberately.
2. **Private legacy memory access.** Existing `migrate` keeps the private database
   but starts an empty canonical household conversation store. Qualify the
   dedicated memory-expansion lookup on Ember's retained records: canonical
   history first, then an exact primary-owned SQL conversation only on canonical
   not-found. Access denial and transport failure must remain errors; a missing
   transcript leaves only the stored memory window/snippet. Preserve stable IDs
   and verify surrounding message bytes for known pointers. This path must not
   create canonical threads, reply offers, notifications, inference grants or
   Growth replay. Private companion transcripts remain excluded. General
   conversation lookup and Chat/Archive continue to use canonical history; bulk
   legacy UI import is outside this rollout's agreed scope. Existing files and
   summaries provide ordinary continuity while retained transcripts remain
   available for exact private lookup. The [action plan](legacy-conversation-access.md)
   covers missing-source choices and older export gaps.
3. **Metadata and visibility mapping.** Current migration defaults the household
   operator to `You` and the resident descriptor avatar to `vex`; the original
   configuration must be mapped explicitly. Use the primary personality's Ember
   name while retaining the original IDs and legacy naming provenance; do not
   select the same-named companion as the migrating primary. On initial native
   startup, private `world_object` places can become namespaced household places with shared
   visibility, and canonical artifact-reference lists start empty. Inspect that
   mapping and reconnect supported artifact references before adding peers;
   preserve private interpretations as private. Map timezone, active hours, schedule, model
   parameters and authority deliberately rather than inheriting fresh defaults.
4. **Pinned household service.** The development launcher executes its checkout
   while retaining older enrollment release labels. Package an exact source tree,
   Python environment and dependency lock; bind coordinator, workers and frontend
   to those same bytes and record their digest. Do not relabel an old root to
   bypass compatibility checks. The existing supervisor probes standalone
   readiness and its boot check imports the standalone app; the household requires
   a qualified service entry point and aggregate readiness check. A manually
   managed, pinned service is acceptable; household automatic deployment stays
   disabled. Readiness must distinguish liveness, store/identity/hydration readiness
   and model availability, without restarting a healthy app for a model outage.
5. **Complete household recovery.** Individual archives retain historical shared
   context, not a restorable canonical household. Rehearse manual cold recovery of
   the entire household, shared database, private roots, published artifact bytes,
   configuration and exact application together. Protect installation secrets
   separately. Demonstrate restoration with the original paths unavailable and
   no simultaneous active source. Cross-release restoration remains unqualified
   until tested; native archives currently compare recorded release labels.
6. **Add a new resident to the retained household.** The
   [qualified creation workflow](creating-a-tamagi.md) now provisions a separate
   identity through an idle maintenance boundary and durable registry/manifest
   transition. Its production release was activated on November 10; no peer was
   created by that deployment. Reopening the development launcher still does not
   enroll anyone, and archive migration still targets an empty household. Use the
   creation interface rather than hand-editing a live household into existence.

## Execution sequence and exit checks

### 1. Prepare the release and evidence before maintenance

- Review Ember's existing pilot observations and stored status without using a
  CLI that instantiates the live runtime merely to inspect it. Seven elapsed days
  alone do not complete the behavioral pilot; missing observations remain explicit
  limitations. Do not manufacture activity to fill them.
- Select the exact candidate revision, apply the adoption fixes above, and build
  its pinned environment. Run the offline suite and browser checks for that final
  revision; retain command, source digest, result and limitations.
  Enrollment work may be prepared in parallel, but its activation follows Ember's
  one-resident verification.
- Inventory the complete instance, external path dependencies, credentials,
  provider transport, service unit and `.platform` release-selection/recovery
  metadata. Read secret values only through the eventual private reconnection
  procedure. Review generated-tool absolute paths, excluded files and symlinks.
- Preserve the external predecessor archive named in instance metadata separately.
  Credential files also exist under `/opt/TamAGI/data/`, outside the instance;
  neither a private-root copy nor a portable archive alone covers them. No MCP
  servers are currently configured. The initial instance inventory was about
  94 MB, excluding application environments, predecessor archive and host services;
  measure the complete recovery set before allocating staging space.
- Choose a new durable household root with a short Unix-socket path and enough
  space for staging, the old instance, snapshots and restore copies. Use a free
  loopback validation port distinct from 7742 and the retained test port 7744.
- Identify a durable backup location, a verified off-host destination, password
  custody, the maintenance window and who performs cutover/recovery. Keep private
  evidence and archives out of Git. These locations and timing remain to be set.

Exit: selected release, preservation inventory, private legacy lookup checks, service design
and executable rehearsal procedure are reviewable. Before the final adoption
decision, review the pilot observations after November 10, 2026, 1:08 PM EST;
changing the accepted post-burn-in gate is an explicit operator decision. No
production activation yet.

### 2. Capture a coherent Ember checkpoint

- In an agreed checkpoint window, block new supported work, freeze automatic
  release changes, and drain active chat, evaluations and model requests. Record
  wake state, pending work and any uncertain provider occupancy. A healthy model
  request should drain; health alone does not prove an unknown attempt is idle.
- Use the running release's supported checkpoint boundary for an encrypted
  portable archive. Do not start a second runtime against the live instance to
  export it. A qualified offline exporter can instead operate on the cold copy.
- Stop the supervisor/service, not only its child, so it cannot respawn Ember.
  Verify the child has exited, the listener is closed, instance locks are free and
  external work is settled. A forced stop requires reconciliation before restart.
- Capture the complete stopped instance, exact selected release/environment,
  configuration, service and deployment-selection metadata. Preserve the SQLite
  database and any WAL as a coherent set; validate integrity on a disposable copy.
  Retain encrypted copies locally and off host, checking their hashes and opening
  a restored copy. Keep credentials/password custody separate from portable state.
- Record stable IDs, per-kind counts and logical hashes, file hashes, wake state,
  source release and timestamp. The cold copy is authoritative for any difference
  from an earlier portable export. The original production root is retained.

Perform source-stopped migration and worker rehearsals during the checkpoint
window. Production may resume on its unchanged old release after all rehearsal
workers are stopped and destinations quarantined. Outside that window, continue
static archive/mapping comparisons without activating copied residents or making
false source-stopped acknowledgments. A real-model rehearsal also requires the
production source stopped.

### 3. Rehearse the complete migration on disposable state

- Validate the archive, materialize the new one-resident household, restore missing
  eligible private data, apply the reviewed history/metadata mapping, then run
  integrity and provenance checks before any worker starts. Leave the original
  and the backup untouched.
- Use temporary guarded rehearsal authority, no production credentials, inactive
  MCP, disabled autonomy/orchestrator/index projection and an injected provider.
  Enforce filesystem/network isolation or use inert effect adapters; the Python
  qualification network guard alone does not confine native commands. Test actual
  generated commands only inside the confined disposable environment. Record
  rehearsal authority separately from the deliberate final authority mapping.
- Hydrate with those controls in place. Confirm only
  Ember is enrolled; canonical memories and exact primary-owned old transcript
  windows are retrievable through the dedicated memory lookup; identity files,
  world, companions, Growth, tools and
  artifacts agree with the checkpoint. Verify attachment bytes and old source
  references as well as counts.
- Compare identity documents and graph/world files with their canonical database
  records before hydration overwrites projections. Keep unregistered differences
  as recoverable evidence and reconcile them explicitly. Verify old memory-to-
  transcript expansion; retained memory text alone does not prove that access.
  Index projection is already disabled in production; preserve that retrieval
  behavior rather than adding an index as an incidental migration change.
- Exercise login, new canonical-thread continuation, private legacy memory lookup, Inventory
  read-back, Journal/Map, companion behavior, chosen rest/wake, browser-independent
  accepted work, clean stop/start and interrupted-work handling. Check for no
  duplicate historical messages, effects, Growth or onboarding.
- Disconnect the source paths and repeat hydration/recovery checks. Rehearse
  interruption during migration and service startup; partial state must remain
  unselected or recover to a clearly identified selection.
- Take a new native resident archive and a complete cold household snapshot.
  Restore both into separate disposable destinations. Verify the resident-only
  and whole-household scopes separately, including shared attachment/publication
  bytes, identities, privacy and no historical replay.
- Prove the pre-cutover return path to the exact old standalone release and state.
  Retain a comparison report with every intentional difference and unresolved
  loss. Delete or quarantine disposable copies after the rehearsal.

Exit: a verified migration report and two working recovery paths, not merely a
backup file. No unexplained difference in the preservation contract.

### 4. Perform the final cutover

- Present the exact release, affected paths, enrolled ID, history/world mapping,
  wake state, credential/authority changes, checkpoint and tested rollback steps
  for the operator's execution decision. This planning request is not that live
  cutover decision.
- Repeat the drain/stop/checkpoint procedure against the latest Ember state.
  Reusing a rehearsal checkpoint would discard intervening history. Rerun the
  private lookup and preservation comparison against the final checkpoint.
- Start the pinned household with only Ember, on the validation listener, with
  household scheduling, integrations and automatic deployment disabled. Complete
  fresh operator login and offline continuity checks. Retain the original source
  stopped while this destination is active.
- Reconnect the authorized model centrally, then check an operator-led exchange
  and available tool behavior. Reconnect needed integrations individually; archived
  MCP descriptions do not authorize launching their commands. Keep an explicit
  path/credential/authority difference log.
- Once readiness and continuity pass, move the existing browser access to the
  household service on the intended production address/port. Keep exposure at
  loopback unless a separate access change is chosen. Check login, cached assets,
  resident selection and new canonical conversation routing from the actual
  browser. Validate retained legacy memory access separately; old transcripts
  are not expected in the Chat/Archive thread list.
- Resume the single household clock using Ember's previous schedule/timezone and
  retained rest/wait eligibility, without catch-up. Worker timers remain disabled.
  Record the selected release, roots, IDs, final checkpoint and observations.

### 5. Observe Ember, then introduce another resident

Run Ember alone through conversation, a scheduled opportunity or deliberate rest,
and a clean restart. Observe for at least one normal day before considering a
second resident; this is an operational checkpoint, not a completed behavioral
pilot. Stop progression on identity/history/byte mismatches, replayed actions,
privacy leakage or unexplained wake changes. A model outage needs diagnosis,
not automatic release rollback.

Before adding a second resident, take a new complete snapshot and use the
[creation workflow](creating-a-tamagi.md) in the qualified production release.
The launcher does not enroll into a retained household, and archive migration
only targets a new empty household. Confirm separate
private stores, workspaces and tools; test addressed replies, optional invitations,
rest, fair model admission and recovery. Add one resident at a time. Companion
group activation, device push, autonomous deployment and automatic shared-state
recovery remain separate follow-up work.

## Recovery decisions

| Situation | Response and continuity consequence |
| --- | --- |
| Before destination accepts any new lived work | Stop destination and ensure no orphan worker/provider action remains; restore the original service/selection/configuration and resume the retained old root. Verify original ID, data and wake state. No post-cutover history exists to lose. |
| After destination has new conversations or creations | Stop/drain it and checkpoint **all** destination state first. Prefer repair on the native release or a separately qualified compatible native release. There is no qualified native-to-standalone reverse migration; switching back alone would omit the new interval. Retain and report that interval and get an explicit recovery decision before accepting the gap. |
| One native private root regresses or is lost | Use the rehearsed resident restore with a compatible recorded release. Other residents and current shared state remain current; private changes after the checkpoint are absent. Retain the displaced root and settle old execution authority. |
| Shared store, publications or complete host are lost | Restore the coherent whole-household snapshot and compatible application, with source stopped and paths/credentials deliberately reconciled. Per-resident archives alone cannot restore canonical shared history. Report the household-wide checkpoint gap. |

Retain the old instance and release until Ember's native continuity and complete
recovery are accepted. Keep recurring encrypted resident backups **and** coherent
manual whole-household snapshots while automated household recovery is pending.
Define retention and verify off-host restore before removing any original copy.

Related operating guides: [resident recovery](resident-recovery.md),
[native household](household-native.md), [running and recovery](running.md),
[World Thread](household-world-thread.md), and
[adoption requirements](multi-resident-roadmap.md#phase-f-adoption-is-separate-from-building-the-framework).
