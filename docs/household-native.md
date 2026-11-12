# Native households

The optional native application runs independent resident Agents behind a
household gateway. Each installation chooses its residents and operator; the
project does not define canonical identities. Standalone startup remains a
separate mode.

Each resident has a freshly provisioned private home, identity, runtime database,
workspace, companions, memory and avatar configuration. The gateway owns shared
conversation history, full-history invitations, read snapshots, the shared world,
notebook, and inference accounting. Workers reach it over authenticated Unix
sockets. Provider credentials stay in the coordinator configuration; workers use
the central model transport through the ordinary Agent, tools and receipt loop.
Default standalone startup remains unchanged.

To add a newly created resident to an existing household, use **New TamAGI** in
the character chooser or Settings. The [creation workflow](creating-a-tamagi.md)
collects the TamAGI's name, avatar, personality and optional starting world while
reusing the household operator's name. It provisions an independent private home
under the same release and resumes workers through an idle maintenance boundary.
It does not restore an earlier identity or promote an existing companion.

Existing residents can retain private legacy SQL conversations for exact
`expand_conversation_memory` lookups after migration. The dedicated lookup uses
canonical household history first and falls back only when that thread is
missing, to an exact conversation owned by the primary resident. It does not
change general conversation lookup, use unrelated cached history, or import
legacy threads into Chat/Archive. Files and memory summaries remain useful
everyday context; exact retained transcripts supply surrounding evidence when
needed. See the [existing-user action plan](legacy-conversation-access.md) for
capture, validation and recovery limits.

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
.venv/bin/python scripts/household_native.py \
  --port 7744 --model-config /path/to/authorized/config.yaml
```

The foreground process serves `http://127.0.0.1:7744/`. Ctrl+C stops the household.
Pass `--host 0.0.0.0` to make the password-protected listener accessible on the LAN.
The default residents are Cedar, Willow and River, with distinct built-in avatar
art. These are sample names for newly created homes, not recovered identities.
Choose different names with `--names`. Reopen a retained development household with:

```bash
.venv/bin/python scripts/household_native.py \
  --port 7744 --root /tmp/the-retained-household/home
```

Reopening preserves histories and colors. Interrupted work, effects and old reply
opportunities do not automatically resume. Unknown upstream occupancy holds new
model dispatch for reconciliation. Existing residents are never discovered or
enrolled automatically. Use separate roots for development and an existing installation.

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
  input or first-response priority. Ordinary chosen rest pauses World Thread
  opportunities while joined-thread peer replies remain eligible. An explicit
  waiting state can still defer peer work; eligible deferred replies resume on
  the existing allowance. Shared notebook and visible world events can satisfy
  the specified wake condition. Normal chat completion preserves an existing
  World Thread rest deadline unless the resident changes it through `end_turn`.
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
  human input or first-response priority. Ordinary chosen rest does not defer
  an invitation decision; explicit waits and other admission limits still apply.
  Restart or Stop does not blindly replay a prior invitation decision. The
  canonical invitation can remain pending for a later deliberate response.

Ordinary rest does not require a chat-specific availability setting. Direct user
chat remains available, and peer replies/decisions remain optional. The app adds
no Do Not Disturb mode or automatic "I'm resting" reply. See the
[World Thread guide](household-world-thread.md) for the distinction between rest,
explicit waits and scheduled opportunities.

For an interactive invitation check, ask Willow in the desired private thread:
"Please invite Cedar into this conversation, keeping this same thread."
The participant strip first shows **Cedar · invited**. Cedar's model then receives
an invitation card without the transcript. An explicit
`contact(action="respond", decision="accept")` joins and permits a subsequent
ordinary reply with prior history; `decision="decline"` records refusal without
sharing it. Prose alone is not acceptance. The
[invitation regressions](../tests/test_household_native_invitations.py)
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

[Repeatable household checks](household-native-testing.md) exercise actual worker
processes, injected model transport, HTTP/browser behavior, rest, invitations,
creation and individual recovery. Keep run reports and screenshots outside Git.
These checks establish mechanics, not sustained model behavior or an OS sandbox.

Coordinated household recovery and release rollout, companion promotion, richer
consultation and device push remain [roadmap work](multi-resident-roadmap.md#remaining-work).
Notification intents and unread state persist, but push is not dispatched.
[Individual recovery](resident-recovery.md) covers private resident archives; it
does not reconstruct a consistent household shared-state checkpoint.

For an existing installation, use the [preservation playbook](production-multi-resident-enablement.md)
and qualify the exact source, environment, identity mapping and recovery procedure.
The development launcher uses the current checkout and is not a production release
controller. Editing or restarting a checkout does not qualify cross-release recovery.

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
