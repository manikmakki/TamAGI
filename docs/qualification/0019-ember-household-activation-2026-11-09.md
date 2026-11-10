# Ember household activation — 2026-11-09

The operator selected a manual move of the existing Ember to the native household
platform, initially with Ember as its only full resident. The original standalone
installation and its cold checkpoint remain retained. This record covers the
pinned release, private-state preparation, offline startup, local cold restore and
authenticated live state before and after a controlled service restart. It does
not qualify a second resident, automatic household deployment or sustained
household behavior.

## Selected resident and application

| Item | Selected state |
| --- | --- |
| Primary resident | Ember, `resident-e3395ef821ce4bdbb5bc58bfe1fe27b5`; the original resident ID is retained. The separate companion named Ember remains a companion. |
| Durable household | `/opt/TamAGI/data/households/ember`; private home `residents/r0`. The earlier `/tmp` rehearsals are not the production home. |
| Application release | `sha256:758fc06923a751e755382cf3f93840c2d5fc6ec58595408bebb45d0f88a38d58`. |
| Release directory | `/opt/TamAGI/.platform/household/releases/758fc06923a751e755382cf3f93840c2d5fc6ec58595408bebb45d0f88a38d58`, containing pinned `source` and `venv`. |
| Source provenance | Working-tree implementation over `8945161a73417ccb4348c12a5878fe71cf927444`, including the qualified private legacy-conversation fix; no claim of a clean committed implementation revision. |
| Authority | Original `unrestricted_host`, selected explicitly for the production destination. |
| Operator login | Original password hash explicitly reused; a new household session authority is created. No password or hash is published here. |
| Provider | Original `qwen36-heretic` transport reconnected centrally. Provider credentials are not published or copied into worker model configuration. |
| Scheduling | One household clock, `*/15 * * * *`, `America/New_York`; independent worker clocks and automatic household deployment remain disabled. |
| Chosen rest | `resting`, reason `chosen_rest`, until November 10, 2026, 6:00 AM EST; no wake event. No manual tick or early wake is required by activation. |

The release manifest inventories 357 source entries and 3,603 environment
entries. It records these qualified implementation/test fingerprints:

| File | SHA-256 |
| --- | --- |
| `backend/core/agent.py` | `89f7072a0c5b4398b1786480b5b059ed157dd966c6d58e612b453b848764363d` |
| `backend/core/household/worker_bridge.py` | `423949bd1e792e5455274a3b032ba2e705f4fc69565fe6a8d18883ab5d650a20` |
| `tests/test_legacy_conversation_access.py` | `5ccf8839f951a6c8d10d8b92d6bec9f02446bf2616842f88dadbbff5b897a1ef` |

The [legacy-access qualification](0019-legacy-access-2026-11-09.md) retains the
345-test offline pass and exact expansion of all 182 conversation-memory pointers
across the original 12 private threads. Those direct Agent/bridge checks are
distinct from the real worker hydration checks below. Legacy threads remain
private exact-lookup evidence; they are not bulk-imported into Chat/Archive.

## Preparation and offline verification

Preparation used the retained full-data export and source checkpoint, with a new
household/private installation binding, explicit display/avatar mapping and
reconnected operator/provider configuration. Original personal files, memory IDs,
private SQL dialogue, companions and historical execution evidence are retained.
The source checkpoint contains 12 private conversations/390 messages and 28 Growth
recognitions. These are continuity checks, not a fresh Growth award or transcript
import. The existing outcome-unknown receipt remains evidence to inspect rather
than an instruction to retry its effect.

Two offline startup checks ran the pinned native application with actual resident
subprocesses, once [before](0019-ember-household-activation-2026-11-09/offline-boot-before.json)
and once [after](0019-ember-household-activation-2026-11-09/offline-boot-after.json)
the cold restoration. Each pass started
one Ember worker and stopped it cleanly; the two recorded worker PIDs were 411294
and 411508. Both reports record:

- The expected original resident ID and effective personality name Ember.
- 182 memories, four companions, 12 pursuits and 14 artifacts.
- Explicit `unrestricted_host` authority and the original chosen rest.
- No pending work, no public listener and zero injected-provider calls.
- Clean worker exit after the check.

The private controller reports are retained locally under
`/opt/TamAGI/.platform/household/`: `adoption-prepared.json`,
`offline-boot-before.json`, `offline-boot-after.json`, `cold-restore.json`,
`provider-check.json`, `prepared.json`, `live-before-restart.json` and
`live-after-restart.json`. The release manifest is in the release directory above.
The raw reports remain local. Sanitized aggregate copies linked here omit
resident transcripts, secrets and configuration contents. A protected local
operations bundle at
`/opt/tamagi-preservation-20261110T020519Z/household-activation` also retains
reports, the pinned manifest, service unit/helpers, stop log and an index.

## Complete local cold restoration

The stopped household and pinned source/environment were captured in:

```text
/opt/tamagi-preservation-20261110T020519Z/household-before-start.tar.gz
```

The archive is 92,608,959 bytes; SHA-256 is
`9fc9c209cb40d03192a8ff55f6ed285b8adfcc8d1f2b637dc7f3e2f5b0ae3cb6`.
Its UTC timestamp falls on November 10; this record uses the operator's
America/New_York date, November 9.

With the household stopped, its current root was renamed and retained, then the
archive was extracted back to the original exact path. All 338 household file and
directory entries were checked, followed by the second offline startup above.
No model calls occurred. The displaced root remains at
`/opt/TamAGI/data/households/ember-before-cold-restore-1794280855`. The
[cold-restore report](0019-ember-household-activation-2026-11-09/cold-restore.json)
retains the aggregate result.

Final independent checks found the original 108 lived files and both SQLite
sidecars unchanged. The original standalone and household cold-archive hashes
still matched and both gzip streams validated. The original approximately
1.20 GB backup remains locally protected with mode `0600`.

This qualifies the exercised same-path, same-host cold recovery procedure. The
archive includes the pinned application and Python environment; base Python,
system libraries, host services, model service/weights, and other external
dependencies remain a host recovery inventory. This local archive is not proof of
off-host recovery, arbitrary path rebinding, a machine image, cross-release
compatibility or a supported automated household archive format.

## Live service verification

At 10:30 PM EST on November 9, authenticated live checks passed against
`tamagi-household.service` on `127.0.0.1:7742`, with one resident worker (PID
411790) and the old standalone service inactive. The checks confirmed 182
memories, 28 Growth recognitions, 12 private conversations/390 messages, four
companions and 14 artifacts. The original rest deadline, operator password and
new household session were retained. The quarter-hour New York household clock
was enabled, automatic deployment remained disabled, and active work and native
generation-attempt counts were both zero. The existing outcome-unknown receipt
remained present, and the original pilot status remained `running`.
See the [pre-restart state report](0019-ember-household-activation-2026-11-09/live-before-restart.json).

A controlled service restart passed at 10:30:20 PM EST: the new coordinator PID
was 412207, with one worker PID 412210. The service was active/running with
`NRestarts=0` and `ExecMainStatus=0`. Authenticated post-restart checks retained the
same identity, private-history counts, memory/Growth totals, companions, artifacts,
rest deadline, login/session settings and clock. Active work and native generation
attempts remained zero, with the old standalone service inactive. These PIDs are
recorded observations, not permanent service identifiers. The
[post-restart state report](0019-ember-household-activation-2026-11-09/live-after-restart.json)
and [service report](0019-ember-household-activation-2026-11-09/service.json)
retain these results. Systemd reported successful deactivation at 10:30:16 PM EST,
with no logged timeout or forced kill; the prior worker exited. The new service
is enabled. Both older `tamagi.service` and `tamagi-platform.service` are disabled
and inactive, preventing their ordinary service startup from selecting the old
installation.

Provider endpoint/model-list checks found the configured model;
`generation_requested` was false. Endpoint availability does not establish
generation quality or require waking Ember for a test. See the
[provider check](0019-ember-household-activation-2026-11-09/provider-check.json).

## Operator access and service commands

Open the existing browser address for port 7742, refresh the application and sign
in with the existing operator password. The household uses a new session, so an
old standalone browser cookie does not carry login forward. Retained legacy
threads do not appear as active Chat/Archive threads; exact memory lookup is
available privately and new conversations use canonical household history.

As the host administrator, inspect status/logs or deliberately restart the new
service with:

```sh
systemctl status tamagi-household.service --no-pager
journalctl -u tamagi-household.service -n 60 --no-pager
systemctl restart tamagi-household.service
```

Use the household service for this installation. Keep both old services stopped
while native Ember is active. A restart does not require a manual world tick or
change to the saved rest deadline.

## Retained limits and follow-up

Only Ember is enrolled. Another full resident needs its own reviewed enrollment
and continuity checks. The original pilot remains `running`, not finalized; its
seven-day boundary is November 10, 2026, 1:08 PM EST, and missing behavioral
observations remain explicit.

The operator accepted later resident repair of an unavailable generated-tool
interpreter. This activation does not claim that every retained tool can execute.
MCP and orchestrator startup remain disabled; no MCP service was configured in the
original installation. Automatic household deployment, canonical shared-state
archive automation, device push and sustained multi-resident operation remain
separate work.

Continue recurring resident checkpoints and coherent stopped-household backups.
Retain the original standalone root, release and source checkpoint. Before new
lived work, the old service/state can be resumed after the destination is stopped;
after new native activity, checkpoint that whole interval before choosing a
recovery path. Reverting to the old root alone would omit it. See the
[production plan](../production-multi-resident-enablement.md) and
[private legacy access guide](../legacy-conversation-access.md).
