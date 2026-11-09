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
The household coordinator can now own the scheduled World Thread; add
`--world-thread` to explicitly enable its single clock while worker timers stay
off. The flag persists for retained households. See the
[World Thread guide](household-world-thread.md) for rotation, sponsorship, operator
unread indicators and public artifact publication/import.

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

The development launcher runs workers from its current checkout. Reopening a
retained root preserves its enrollment release label; it does not pin executable
bytes or automatically activate/rebind a new release. The archive compatibility
check compares those stored labels, so repeated development restarts do not
qualify cross-release recovery. Production adoption requires a controlled,
qualified release and explicit resident migration; do not treat this launcher as
a production upgrade controller.

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

For an interactive invitation check, ask Willow in the desired private thread:
"Please invite Cedar into this conversation, keeping this same thread."
The participant strip first shows **Cedar · invited**. Cedar's model then receives
an invitation card without the transcript. An explicit
`contact(action="respond", decision="accept")` joins and permits a subsequent
ordinary reply with prior history; `decision="decline"` records refusal without
sharing it. Prose alone is not acceptance. The
[retained invitation trace](qualification/0019-native-invitations-2026-11-08/invitation-flow-proof.json)
shows the card, dispatched response and unchanged thread identity in an automated
actual-worker test with an injected provider. The next interactive request tests
the real resident's choice.
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

The [2026-11-09 World Thread record](qualification/0019-native-world-2026-11-09.md)
retains the full 303-test run, final focused world checks, native Chromium world,
operator-mention and public-library evidence, and retained-household activation.

Run the repeatable offline Chromium regressions from the isolated worktree with:

```bash
/opt/TamAGI/.venv/bin/python scripts/check_household_browser.py \
  --output /tmp/tamagi-native-browser-check-new
```

This starts disposable native workers on port 18846 and Chromium debugging on
18847, uses the existing cached Chromium executable, retains evidence under the
output directory and removes its temporary password. It uses no real provider.

The development candidate still needs production resident adoption/history
migration, coordinated household backup/restore and release activation,
companion group-activation wiring, PWA push delivery and
device qualification. Notification intent and unread state already persist;
device push is not yet dispatched. Browser deployment and shared-state recovery
remain disabled. [Individual resident recovery](resident-recovery.md) provides
Settings backup/preview/restore controls and offline operator commands; it does
not replace coordinated household recovery or qualify
production adoption. Background autonomy
is opt-in through the household clock; sustained household behavior and Ember's
burn-in are separate qualifications. Explicit public artifact versions and exact
private imports now retain real bytes; they do not replace coordinated recovery.

Do not replace production or enroll Ember on the strength of these development
checks alone. See ADR-0019 for the full acceptance requirements.

## Menu scope in a native household

Every menu page identifies its resident context and offers the existing resident
chooser. Choosing another resident while browsing keeps the current menu page.
The drawer also names the selected resident. Chat retains its existing avatar
selector, participant strip and attributed messages.

| Menu item | Scope |
| --- | --- |
| Map | The selected resident's world graph. The separate places list shows household places visible to that resident; pursuits belong to that resident. |
| Journal | The selected resident's entries and activity. The household clock and notebook are shared. |
| Inventory | The selected resident's private workshop files and saved artifacts. Shared library opens explicitly published household artifacts. |
| Companions | Companions introduced to the selected resident, including their memories and conversations. Other household residents are selected through the resident chooser. |
| Archive | Operator conversations that include the selected resident, plus that resident's private memories. Group threads retain all participants and attribution. |
| Settings — Home & recovery | Conversation colors are a household preference. Backup, restore, permissions and runtime evidence belong to the selected resident. Backup/restore briefly pause idle workers; other residents and current shared state are retained. Shared-state recovery and release changes remain unavailable. |
| Settings — Connections | The selected resident's secrets and connected tool services. The central model endpoint is coordinator configuration. |
| Settings — Appearance | The selected resident's avatar configuration and uploaded art. |

Embedded map, connection and appearance panels repeat the resident name. Inventory
previews name their resident, and preview/download links retain their captured
resident scope. Changing residents closes the old Journal and file details,
clears private lists, and ignores late memory results/counts from the old context.
Unread threads, operator mentions, the notebook and the published artifact
library retain their household scope.
