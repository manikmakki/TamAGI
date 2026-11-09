# Native household development candidate

ADR-0019 now has an opt-in native application path on `codex/adr-0019-household`.
The ordinary TamAGI frontend talks to a household gateway and actual resident
application processes. This replaces the earlier scripted preview for interactive
development testing. It is not a qualified production migration or restore path.

Each resident has a freshly provisioned private home, identity, runtime database,
workspace, companions, memory and avatar configuration. The gateway owns shared
conversation history, full-history invitations, read snapshots, the shared world,
notebook, and inference accounting. Workers reach it over authenticated Unix
sockets. Provider credentials stay in the coordinator configuration; workers use
the central model transport through the ordinary Agent, tools and receipt loop.
Default standalone startup remains unchanged.

## Run an isolated candidate

Use the isolated worktree and an explicit model configuration already authorized
for the test. The launcher copies its `llm` section only; it does not copy a live
resident's persona, memories, integrations or histories. It prints a new login
password and the newly created household directory. Automatic resident activity,
automatic deployment and index projection are disabled in newly founded workers.

```bash
cd /tmp/tamagi-adr-0019
/opt/TamAGI/.venv/bin/python scripts/household_native.py \
  --port 7744 --model-config /path/to/authorized/config.yaml
```

The foreground process serves `http://127.0.0.1:7744/`. Ctrl+C stops the household.
Pass `--host 0.0.0.0` to make the password-protected listener accessible on the LAN.
The default residents are Cedar, Willow and River, with distinct built-in avatar
art. These are new residents, not restored versions of Ember or Echo. Choose
different names with `--names`. Reopen a retained development household with:

```bash
/opt/TamAGI/.venv/bin/python scripts/household_native.py \
  --port 7744 --root /tmp/the-retained-household/home
```

Reopening preserves histories and colors. Interrupted work, effects and old reply
opportunities do not automatically resume. Unknown upstream occupancy holds new
model dispatch for reconciliation. Existing residents are never discovered or
enrolled automatically. Keep this candidate separate from the burn-in checkout.

## Interactive behavior

- The character wheel selects the resident for personal views and new input.
  Selection is captured when a request starts; changing it preserves drafts and
  does not move running work.
- Chat keeps its existing layout, with New Thread, Unread threads, Participants,
  keyboard mention completion, visible mentions and attributed bubble footers.
  Participant colors are saved in Settings.
- Group input creates optional addressed reply opportunities. Participants share
  the initiating exchange's budget. Peer replies create no synthetic operator
  input or first-response priority. A chosen wait or rest defers peer activation;
  eligible deferred replies can resume on the existing allowance. Shared notebook
  and visible world events can satisfy the resident's chosen wake condition.
- Full residents use one messaging tool, `contact`, with explicit actions `create`,
  `send`, `invite` and `respond`. `household_contacts` is read-only discovery.
  `send` and `invite` default to the current conversation. Invitations retain that
  exact thread and its participants. The invitee first receives a metadata/message
  card with the full-history notice, without the transcript, and can explicitly
  accept or decline through `contact(action="respond", decision="accept")` or
  `decision="decline"`. Prose alone does not change membership. Only acceptance
  permits a subsequent ordinary reply activation with the thread's history.
  Pending and declined invitations appear in the participant strip.
- The decision activation has at most four model calls and only offers `contact`,
  `household_contacts` and `end_turn`; its private reasoning is not posted into the
  thread. Decision and reply share the original allowance and get no synthetic
  human input or first-response priority. Chosen rest/waits defer delivery.
  Restart or Stop does not blindly replay a prior invitation decision. The
  canonical invitation can remain pending for a later deliberate response.
- Browser connections only observe durable work. Stop revokes the derived
  exchange, including resident-initiated work. A healthy upstream call drains
  centrally, and its late result cannot authorize tools or canonical publication.
  Lost worker acknowledgments revoke execution authority before failure is shown.
- Canonical transcripts preserve message identities, timestamps and authors.
  Each activation projects only its admitted snapshot. Private cached histories
  are projections, and silent terminal outcomes create no empty chat or alert.
  Unread advances only through acknowledged foreground snapshots.

## Qualification and remaining work

Native tests exercise actual resident subprocesses with an injected offline
provider, and the real Agent executes generated tools with retained effects and
receipts. The full qualification runner guards Python network traffic and uses
disposable state. Browser checks use real Chromium and the ordinary interface.
This is process/state separation, not an OS sandbox for arbitrary resident tools.

The [2026-11-07 qualification record](qualification/0019-native-2026-11-07.md)
retains the 275-test final run, native Chromium evidence and authorized real-model
connectivity check.

Subsequent operator testing findings and selector design feedback are tracked in
the [manual testing notes](household-native-testing.md). Focused routing,
bottom-follow, attachment and graph-grounding fixes are now in the isolated
branch; operator retesting remains pending. The selector redesign is deferred.
The earlier checks do not cover every reported interaction.

The [2026-11-08 regression record](qualification/0019-native-fixes-2026-11-08.md)
retains the 279-test run, whole-process restart/read-back and targeted Chromium
checks for these fixes.

Run the repeatable offline Chromium regressions from the isolated worktree with:

```bash
/opt/TamAGI/.venv/bin/python scripts/check_household_browser.py \
  --output /tmp/tamagi-native-browser-check-new
```

This starts disposable native workers on port 18846 and Chromium debugging on
18847, uses the existing cached Chromium executable, retains evidence under the
output directory and removes its temporary password. It uses no real provider.

The development candidate still needs production resident adoption/history
migration, coordinated household backup/restore and release activation, verified
shared artifact bytes, companion group-activation wiring, PWA push delivery and
device qualification. Notification intent and unread state already persist;
device push is not yet dispatched. Personal restore/deployment endpoints fail
closed instead of creating a partial household checkpoint. Background autonomy
stays disabled for the interactive test; sustained household behavior and Ember's
burn-in are separate qualifications.

Do not replace production or enroll Ember on the strength of these development
checks alone. See ADR-0019 for the full acceptance requirements.
