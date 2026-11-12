# Individual resident recovery

Individual recovery preserves private resident continuity. Canonical shared
records and publications need coordinated household recovery as a separate layer.
The native application provides offline operator commands and
**Settings → Home & recovery** controls for resident archives.

Native compatibility checks compare recorded enrollment release labels. The
development launcher can run changed checkout code against a retained label;
these rehearsals do not prove compatibility across actual application releases.
See the [launcher limitation](household-native.md#run-an-isolated-candidate)
before planning production adoption.

The three acceptance scenarios are:

1. Checkpoint a resident, introduce a private regression in a disposable copy,
   then restore that resident without rewinding housemates or shared state.
2. Lose the resident's private directory/database and replace it from the
   checkpoint, preserving their ID, memories, history and actual file bytes.
3. Restore the archive alone into a new one-resident household with rebound
   paths, fresh installation/session authority and no fabricated peer workers.

Data-loss recovery currently means checkpoint replacement. It does not merge
selected missing memories into a newer private database. The preview reports
that private changes after the checkpoint are absent from the replacement.
The displaced private directory is retained; recovery never overwrites it.

## Browser controls

Select the resident, open Settings → Home & recovery, and check the resident name
above the forms. Create an encrypted download with a repeated password of at
least 12 characters. Download links stay bound to that resident; encrypted server
copies expire after 24 hours or coordinator restart. Keep the downloaded archive
and password elsewhere for independent recovery.

Upload a resident archive and enter its password to validate a preview without
stopping anyone. The preview identifies the resident, checkpoint time, checked
file count and private continuity gap. A checkpoint for another resident is
rejected. Changing residents clears passwords, file selection and preview state;
delayed responses cannot populate the next resident's forms. Cancelling a preview
discards its decrypted staging directory and makes no resident changes.

Confirm replacement and that no other installation of this resident is active,
then choose the restored authority. The coordinator refuses backup/restore while
chat, scheduled work, private evaluations or unsettled generation attempts remain.
Once idle, it blocks new supported work, drains requests, stops the workers and
holds their lifetime/instance locks during the checkpoint or replacement. Workers
resume afterward. Other residents retain their current private continuity; normal
restart hydration bookkeeping still occurs. The shared clock is retained and
does not advance during the maintenance window.

Restoration retains the current shared messages/world/publications. Only the
selected private root is replaced. Old reply/invitation opportunities and execution
authority for that resident are closed/interrupted instead of replayed; other
residents' opportunities remain intact. Pending invitations to the restored
resident expire without being presented as a decline. Invite them again explicitly.
The UI invalidates private view responses and refreshes the restored resident
while preserving the current draft.

Browser disconnects do not cancel confirmed operations. A failed worker startup
exposes **Retry resuming residents**. If a registry selection cannot be reconciled,
workers stay stopped and new work stays blocked until that control can finish the
durable transition. Inspect the selected resident before retrying a restore whose
acknowledgment failed. A known, ownership-checked staging directory lets the next
coordinator remove decrypted previews after a crash. API downloads and previews
require operator authentication; resident API paths bypass service-worker caches.
The private browser staging directory sits beside the household directory, so
the runtime user needs permission to create that sibling directory.

Browser upload is limited to 1 GiB. Previews expire after 30 minutes or restart.
Migration into a new household still uses the command below. Shared-state recovery,
cross-release compatibility and automatic deployment are not enabled by these forms.

## Offline commands

Use the selected application release's Python environment. Stop the source
household and all its workers before exporting or replacing a resident. The
command refuses a held coordinator lock, worker lifetime lock or existing
private instance lock. It never stops or starts an installation itself.

```sh
python scripts/resident_recovery.py backup \
  --root /path/to/stopped-household --resident resident-ID \
  --output /separate/location/resident.tamagi.gpg

python scripts/resident_recovery.py preview \
  --archive /separate/location/resident.tamagi.gpg

python scripts/resident_recovery.py restore \
  --root /path/to/stopped-household --resident resident-ID \
  --archive /separate/location/resident.tamagi.gpg \
  --confirm --source-stopped --authority guarded

python scripts/resident_recovery.py migrate \
  --archive /separate/location/resident.tamagi.gpg \
  --destination /new/empty-household-path --release COMPATIBLE-RELEASE \
  --confirm --source-stopped --authority guarded
```

Passwords are prompted without echo. Automation may use the global
`--password-fd FD` option to read one line from an inherited descriptor. Passwords
are never command arguments, environment values, or printed in results.
The archive uses the existing GnuPG encryption and validated safe extraction.
Preview decrypts into private temporary storage, validates every manifest hash,
checks SQLite integrity/identity/schema and native resident metadata, and cleans
up on failure. CLI previews expire when their process exits; commit independently
validates the archive again. Do not treat preview IDs as persistent credentials.

## What recovery does

The archive contains the resident database, private workspace, private data
including legacy history/world graph/artwork, sanitized configuration and
application source/lockfiles. Dedicated credential stores, session secrets,
derived indexes, model weights and other standard portable exclusions remain
excluded. Conversation text and arbitrary artifacts can themselves contain
secrets; encryption does not sanitize their contents.

Native archives also include `data/recovery-evidence/context-<id>.json`, containing
the household roster and canonical conversations/messages accessible to this
resident at the checkpoint. Other residents' private conversations, memories
and files are not exported. Older historical context files survive subsequent
backups/migrations; new household context does not replace them. Imported artifact bytes already in the private
workspace travel with it; the shared public library is a later recovery layer.

Replacing a resident keeps the household's installation identity and operator
session, rotates the target's private installation identity, and selects a new
private root. The old root remains available for inspection. The shared store
updates that resident's registry binding, records recovery and settles their old
offers, pending invitations and execution authority. Shared messages, membership,
clock, read markers and public versions are retained. An fsynced transition
journal lets startup or resume finish both the selection and settlement after
interruption between the registry database and manifest updates. This recovery
does not execute uploaded code or repeat resident effects.

Migration creates a new household and private installation identity, requires
operator password setup and provider/credential reconnection, and starts with
autonomy and automatic deployment disabled. No peer is enrolled from memories
or historical author references. Archived shared transcripts remain private
historical evidence, **not active conversations in the new household's Chat or
Archive UI**. For an existing conversation-memory pointer, the dedicated private
memory lookup tries canonical history first, then an exact primary-owned legacy
SQL conversation only when the canonical conversation is missing. Denial and
transport errors do not authorize fallback. Missing transcripts leave only the
memory's stored window/snippet; recovery-context exports are not searched by this
lookup. See the [existing-user action plan](legacy-conversation-access.md).
A historical transcript viewer and optional canonical shared-state recovery
remain follow-up work; ordinary conversation lookup does not fall back to
private caches or import old threads into Chat.
MCP connections are described in the encrypted manifest but are not activated
on restored worker startup; reconnect them explicitly after review.

Native checkpoint restoration currently requires the archive's recorded release
to match the target release. The source archive is retained as reconstructible
evidence, not automatically selected as shared application code. An application
code regression needs separate compatible code recovery. Legacy portable v1
archives are accepted for explicit one-resident migration; qualify the selected
release's compatibility before committing them.

Older standalone exporters can omit private history and other `data/` files
outside sprites, even though they retain the runtime SQL database. Inspect the
archive's file manifest and omissions. Preserve a complete stopped-source
checkpoint before migration, and qualify any additional private-file capture on
a disposable copy. Never replace a newer home's database with a raw copy of the
old database; use a reviewed restore transition and report its continuity gap.

Queued private interactions/evaluations become interrupted and pending outbox
effects become outcome-unknown. Normal shared coordinator restart recovery still
closes old reply offers. No missed-tick catch-up, historical notification burst,
or uncertain effect replay is requested by resident restoration.

## Qualification boundaries

[Resident recovery tests](../tests/test_resident_recovery.py) cover encrypted
archives, compatibility, failure handling and real disposable workers.
[Browser recovery tests](../tests/test_household_browser_recovery.py) cover
maintenance, preview/commit, resume and retries. See [household testing](household-native-testing.md)
for repeatable browser and worker scenarios.

These checks do not establish hot resident-only maintenance, arbitrary
cross-release restoration, manually reconstructed historical identity, individual
moves into established households or coordinated whole-household recovery.
