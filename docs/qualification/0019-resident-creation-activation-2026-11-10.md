# Resident creation release activation — 2026-11-10

The operator authorized activation of the qualified creation workflow and a
production service restart. The existing Ember household now serves the new
release on `127.0.0.1:7742`. This activation created no resident and requested no
model generation. The operator can use **New TamAGI** to create a new identity
when ready; that action is separate from deploying its interface.

The [source qualification](0019-resident-creation-2026-11-10.md) retains the
358-test offline pass and 40 Chromium checks against disposable residents. This
record covers the subsequent manual production transition and live verification.

## Selected release and continuity

| Item | Result |
| --- | --- |
| Household | Existing `/opt/TamAGI/data/households/ember`; one enrolled resident. |
| Resident | Ember, `resident-e3395ef821ce4bdbb5bc58bfe1fe27b5`; resident and installation IDs retained. |
| Previous release | `sha256:758fc06923a751e755382cf3f93840c2d5fc6ec58595408bebb45d0f88a38d58`. |
| Selected release | `sha256:3adf86aa2bb240f3a14ba0ce41c1e0a39a9018a6b43a73389388566742615b2b`; pinned source/environment under the matching directory in `/opt/TamAGI/.platform/household/releases/`. |
| Service | `tamagi-household.service`, enabled and active/running after startup at 8:54 AM EST; coordinator PID 442287, `NRestarts=0`, `ExecMainStatus=0`. |
| Previous services | `tamagi.service` and `tamagi-platform.service` remain disabled/inactive. |
| Lived state at upgrade | 187 memories, 28 Growth recognitions, 12 legacy private conversations/390 messages, and one canonical shared conversation/11 messages. These include activity since the earlier November 9 checkpoint. |
| Rest | Original `chosen_rest` retained until November 10, 2026, 1:00 PM EST (`1794333600`); no wake event. |
| Clock | Enabled, `*/15 * * * *`, `America/New_York`; the upgrade does not advance the chosen wake time. |
| Operator access | Existing operator configuration and session bytes retained; no password reset. |

The [release report](0019-resident-creation-activation-2026-11-10/release.json)
records the pin and qualification provenance. No clean committed source revision
is claimed; the selected bytes contain the previously qualified working-tree
implementation.

## Checkpoint and stopped-household transition

The source household was stopped and its old worker's exit verified before the
binding change. The local cold checkpoint is:

```text
/opt/tamagi-creation-checkpoint-20261110T134600Z/household-before-creation-release.tar.gz
```

It is 149,301,468 bytes; SHA-256 is
`db2e259a24c809d155793ba327ce41eba9d7c70b8820c7b10fc2441048941a76`.
The gzip stream validated. All 12,608 compared files matched stopped-source
content, modes and ownership. Two ephemeral SQLite SHM modification-time
differences were recorded; their file bytes still matched. See the
[checkpoint report](0019-resident-creation-activation-2026-11-10/checkpoint.json).

An explicit transition, `3c2118e3e16d4fceb189997fa9188a42`, selected the qualified
release in the household and resident bindings. At that boundary the private
database and 286 private files retained their exact bytes, with resident and
installation identities unchanged. The transition started no worker or model
request. See the [transition report](0019-resident-creation-activation-2026-11-10/release-transition.json).
This was a recorded application release switch, not relabeling a resident archive
to bypass its compatibility check.

The operational transition helper was separately checked on disposable metadata
and SQLite state, including interruption at four points and idempotent recovery.
The [transition qualification](0019-resident-creation-activation-2026-11-10/transition-qualification.json)
records those cases. They are distinct from the live upgrade; no production
failure was injected.

## Offline startup and service helpers

The final [offline boot](0019-resident-creation-activation-2026-11-10/offline-boot.json)
passed with an injected provider, no public listener, zero provider calls and
clean worker exit. Its [guard report](0019-resident-creation-activation-2026-11-10/offline-guard.json)
records two guarded Python processes and zero denied network operations.

Resident-authored files and lived record payloads remained intact, including
activity, installation metadata, private dialogue and exact shared content. The
check covered 105 private data/workspace files while allowing documented startup
refresh of `graph.timestamp`, `personality.last_stats_update` and serialization
of three derived JSON files. Those projection/bookkeeping changes are not new
memories or a reset of lived state.

Startup verification now accepts one through sixteen enrolled residents. A
separate [disposable helper smoke check](0019-resident-creation-activation-2026-11-10/startup-helper-smoke.json)
verified two residents, three after creation, and three after an application
restart, with zero model calls or network denials and all workers exited. This
supports restarting an expanded household; it did not create a production peer.

## Authenticated live checks

At 8:55:11 AM EST, [authenticated verification](0019-resident-creation-activation-2026-11-10/live.json)
confirmed the selected pin, the unchanged state/counts above, one resident, no
active work and the retained historical outcome-unknown receipt. Generation
attempts were 23 both before and after activation: the activation added zero.
The existing operator configuration and session matched their cold checkpoint.

`GET /api/household/creation` returned the three built-in avatar choices and the
creation-only schema without operator questions. Served JavaScript (`v6`) and
stylesheet (`v3`) matched the new pinned source. No production creation POST,
conversation input, manual tick or generated-tool execution was used to verify
deployment.

Refresh the existing application address and open the character chooser's
**New TamAGI** action, or **Create a new TamAGI** in Settings. Follow the
[creation guide](../creating-a-tamagi.md). Existing login access is retained.

## Recovery and remaining scope

Sanitized aggregate reports linked here exclude credentials and resident
transcripts. The protected local checkpoint's `operations/` directory retains
the unit, startup helpers, transition helper, verification scripts, manifest and
index. Keep that operational material with the cold archive.

This is a qualified manual transition between the two recorded releases on the
existing host. It does not establish arbitrary cross-release archive recovery,
off-host restoration, an OS image or automatic household deployment. The earlier
[Ember adoption record](0019-ember-household-activation-2026-11-09.md) separately
records its exercised same-path cold restoration.

Keep the pre-upgrade checkpoint and previous release. If the household acquires
new lived work, checkpoint that complete interval before selecting a recovery
path; restoring the earlier snapshot alone would omit it. New residents still
start with guarded authority and no inherited private integrations. Existing
tool-interpreter repair, the original behavioral pilot and sustained operation
with multiple full residents retain their separate scope.
