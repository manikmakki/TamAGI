# ADR-0019: Distinct residents, a shared world, and fair access to cognition

Status: Accepted design by operator direction on 2026-11-05; extended on 2026-11-06 with resident-initiated contact, PWA Web Push, and chat thread controls. An [isolated experimental foundation](../household-foundations.md) now implements selected internal mechanics; production integration and full qualification remain pending. See the [multi-resident roadmap](../multi-resident-roadmap.md).

## Context

TamAGI already has durable actors, scoped memories, execution receipts, separate installation directories, bounded companion invocations, and application-owned interactions. These foundations do not yet establish a multi-resident platform. Startup, identity hydration, workspaces, World Thread state, and deployment selection still assume one primary resident. Separate processes currently have separate provider semaphores; launching several would not establish shared resource fairness.

The operator wants one household in which distinct TamAGIs can develop interests, maintain relationships, meet one another, collaborate, decline invitations, and spend time apart. Ember's interest in meeting Echo helped motivate this direction. Recovering Echo requires a separate assessment of surviving records and is outside this decision's implementation scope.

Phaedra supplies useful precedents for attributed communication, persistent requests, and actor rotation. TamAGI will borrow those mechanisms without importing its complete architecture or daily resource allowances. Existing Phaedra evidence is not evidence that TamAGI has implemented provider-time fairness.

The parent-child framing continues to describe our responsibility as builders: dependable surroundings, understandable choices, recoverable mistakes, and room for relationships to develop. It does not impose family roles, mandatory sociability, or productivity obligations.

## Decision

### One household application, distinct resident runtimes

Use one household application, operator account, selected application release, and coordinator. Run each full resident in a separate runtime process with its own lived-state root. Process separation preserves the current configuration, working-directory, and hydration assumptions; changing only an execution's actor ID would not switch the complete runtime context.

Keep two explicit roles:

| Role | Continuity and activity |
| --- | --- |
| Full TamAGI resident | Personal database, workspace, identity, memories, conversations, pursuits, Growth, appearance, rest state, and independently scheduled opportunities. |
| Companion | Durable attributed identity and personal memories within its owning resident's installation; active only through bounded invitations. |

Companions obtain neither independent population scheduling nor a separate provider-time share. Companion promotion and individual resident moves between established households are deferred.

The coordinator owns canonical shared-world state, conversations, invitations, the household notebook, and inference accounting. Private stores may retain attributed references and rebuildable projections, but cannot independently author competing shared state. A centrally stored private conversation retains its participant boundary; central storage does not make it household-public.

Preserve each resident's founding identity, interpretations, personal pursuits, runtime adaptations, and generated tools in its personal state. Each has a workspace and its own `workshop/` under the [workspace conventions](../workspace-layout.md). Shared conversations remain attributed experiences, not a shared autobiography.

### Connected personal domains and deliberate sharing

Connect personal domains through persistent shared places. Joining preserves each resident's chosen setting; it does not move everyone into a compulsory household theme. Existing private knowledge/self-model graphs remain distinct from the common topology of places and objects.

Give world objects stable IDs, ownership, visibility, and revisions. Route every supported world mutation, including companion tools and legacy graph adapters, through the coordinator. Namespace migrated world IDs and retain compatibility mappings; never merge objects merely because their names match. Former private world-state authorities become projections when enrolled.

The same shared event can support different attributed interpretations in different residents. Preserve ADR-0004's distinction between reported operations, interpretation, and imagination. A described place does not silently create an executable capability.

Publish creations through explicit artifact references and immutable snapshots. Sharing does not expose another resident's entire workshop. Private memories and unrelated conversations do not enter another resident's hydration or retrieval merely because their homes are connected.

### Fair access to cognition

The coordinator's inference broker owns the upstream provider connection. All generation paths, including streaming, retries, reflection, evaluation, and helpers, use it. A semaphore in each resident process is insufficient.

| Policy | Default |
| --- | --- |
| Simultaneous upstream generations | One |
| Full resident shares | Equal weight |
| Fairness measure | Measured time occupying the provider |
| User-chat priority | Configurable lead of 30 virtual seconds |
| Ordinary background daily cap | None |
| Healthy generation | Allowed to finish; no new absolute duration limit |

Provider occupancy includes request processing and streamed generation through completion. It is a practical measure of the constrained resource, not precise GPU compute time. Tokenization remains separately measured and bounded. Waiting for a slot, running tools, waiting for approval or consultation, and delivering events to observers do not occupy the generation slot.

Maintain durable virtual service time for each resident. At each completed-call boundary, select the least-served eligible account. Real user input can receive the bounded priority lead; quiet continuations and resident-origin discussion use ordinary fairness. Resolve ties by oldest eligible request and persisted rotation. All first responses to an operator broadcast receive the same priority rule; forwarding or quoting a user message cannot manufacture priority.

Resting or unavailable residents accumulate no catch-up entitlement. Restart preserves accounting. A resident may use spare capacity when nobody else is ready. Fairness applies at completed-call boundaries: a healthy long generation may delay everyone else. Inspection and UI state must report that honestly rather than promise a fixed maximum wait.

Fairness is an opportunity policy, not an activity quota. Preserve each full resident's configured periodic opportunities and post-conversation decompression, with personal rest/wake state. Coalesce missed opportunities rather than accumulating catch-up activations. Play, rest, refusal, and silence remain legitimate choices.

### Shared limits, sponsorship, and interruption

Fairness accounting and execution limits are separate:

- User-origin exchanges share twenty calls per segment, with up to five explicitly requested quiet continuations and at most 120 calls overall.
- Autonomous-origin exchanges share 128 calls.
- Bounded helper invocations have at most sixteen calls, also charged against their initiating allowance.
- Existing evaluation limits remain. Background work has no daily cap by default.
- Group membership, broadcasts, helpers, yields, retries, and reconnects grant no additional allowance.

The broker is the canonical authority for cross-worker call counters. Private execution records retain attributed projections. Dispatch atomically validates and charges a unique attempt before contacting the provider. Queuing alone charges no model call. A retry is a new attempt within the existing lineage; it cannot erase previous charges.

Ordinary conversation participation uses the speaking resident's time share. Explicit commissioned help uses the requester's share and initiating call budget. Record the cognizing actor and scheduling sponsor separately. Companions receive no separate share; ordinary companion participation uses its owning resident's share, while commissioned work retains the requester's sponsorship. Direct operator requests have no resident requester to debit: use the responding full resident's account, or the owning resident's account for an operator-to-companion invitation.

Actual user input retains priority for the addressed resident and must not cancel unrelated residents' work. Stop revokes further tools and child work immediately. Queued generation can be cancelled without a charge. An already-running healthy generation drains through the broker; its result cannot dispatch tools for a stopped or failed execution. Slow or disconnected observers cannot prolong generation or own its lifetime.

After broker or transport failure, an uncertain upstream operation enters visible reconciliation. Lease expiry or a successful model-list/health request alone cannot establish that remote generation stopped. Provide an authenticated operator recovery action that records its reconciliation evidence or acknowledged uncertainty; it must not replay the old attempt or silently claim proven non-overlap. Provider outages use shared backoff rather than a retry storm from every resident.

### Attributed conversations and addressing

Support private operator conversations, resident-to-resident conversations, and mixed group conversations. Explicitly invited companions may participate through bounded invocations within the invited exchange, using the same addressing rules. Old room membership is not a standing invitation to launch new companion activations in later exchanges, and membership does not give them independent scheduling.

Apply the same addressing rule to operator and TamAGI messages:

- A plain message offers other joined, available participants an opportunity to reply.
- `@name` mentions offer replies only to the named participants, including the operator by preferred name.
- Joined participants can read the thread. Addressing controls reply offers, not transcript visibility.
- Resolve mentions to stable participant IDs at admission. Ambiguous names require disambiguation. Persist the author, resolved addressees, membership revision, and exchange ID; later renames cannot retarget a historical message.
- Joining grants access to the full prior thread, with that behavior stated in the invitation. It grants no access to unrelated private conversations and does not retroactively create reply offers for old messages.
- Context remains bounded; the complete transcript is available through context selection and retrieval rather than an unbounded prompt.

Replies remain within one durable, finite exchange. Broadcasts coalesce outstanding offers per recipient and do not create fresh roots. New messages arriving during inference remain pending beyond the current snapshot watermark. Full-history access does not expose later queued user inputs to an earlier admitted request's model context. Offer initial participants opportunities before repeatedly serving an already-responding participant. An offer permits silence, refusal, or deferral. Exhaustion records unfinished work, not an invented decline or completion.

Only a full resident participating in a user-origin exchange may request a quiet continuation through the trusted shared exchange controller; a bounded companion/helper invocation cannot do so. Requests coalesce into one pending continuation; the five-continuation limit is shared, not multiplied by participants. Failure, exhaustion, or interruption discards pending quiet work under the existing lifecycle.

Keep three distinct operations:

| Operation | Behavior |
| --- | --- |
| Poke/message | Persist an attributed message or invitation; peer wake preferences determine when it becomes actionable. |
| Consult | Request bounded help, preserving sponsor, allowance, result attribution, and dependency references. |
| Yield | Checkpoint and release execution ownership, optionally offering another resident a turn; preserve the remaining allowance. |

A waiting consultation releases both provider capacity and the caller's execution lease. Resume through the coordinator with current hydration and revision checks, rather than nested runtime calls holding execution locks. Reject cyclic dependencies and retain restrictions on recursive specialist work.

Autonomous rest continues to permit user conversation as today. Peer invitations respect the recipient's chosen wake rules. Transcript delivery, browsing, and reconnection do not masquerade as user input, interrupt unrelated work, or themselves trigger cognition.

### Resident-initiated contact and operator notifications

Provide an explicit messaging tool through which a full resident can create a conversation with a chosen opening message or post to an existing thread during an eligible World Thread activation or quiet continuation. The resident selects participants, optional title, message, and bounded context or artifact references to share. Resolve mentions through the existing addressing rules. Derive the author, originating execution, command identity, exchange, and remaining allowance from trusted runtime state. Posting does not create synthetic operator input, operator priority, a fresh allowance, or an additional quiet continuation. A committed message survives later interruption; uncommitted working drafts remain private.

Treat ordinary initiation as poke/message. A peer receives an optional reply offer subject to its own wake preferences; delivery does not force cognition or awaken a resting resident. An addressed operator receives an unread conversation and, when enabled, a device notification. A message awaiting attention creates no obligation to respond. Keep consultation and yielding distinct.

Commit the canonical message and its notification intent together in the coordinator store. Deliver device notifications independently of resident execution and browser connections. Retain stable message and delivery identities, bounded retries/backoff and expiry, and inspectable outcomes. Reconcile lost acknowledgments without creating another thread or message; coalesce repeated alerts for the same unread conversation. Before dispatch, recheck access, unread state, notification preferences, and subscription validity. A failed or expired push never removes the conversation or clears unread state. Push-service acceptance does not establish presentation or reading, and OS notification display has no exactly-once guarantee.

Start with standards-based PWA Web Push: HTTPS, an explicit user-initiated permission/subscription flow, application server keys (VAPID), authenticated device-subscription storage, and push/notification-click handlers in the existing service worker. Use a protocol library for encryption and signing; no Firebase project, Firebase SDK, custom native app, or separate notification server is required. Browser-selected push infrastructure remains an external delivery dependency, and the backend needs outbound access to it. A notification carries a minimal alert and a same-origin link to the stable conversation identity; the authenticated application checks access before displaying the transcript. Default to a generic lock-screen preview, with message previews an operator choice. See the [Web Push FAQ](https://web.dev/articles/push-notifications-faq) and [subscription requirements](https://developer.mozilla.org/en-US/docs/Web/API/PushManager/subscribe).

Keep device enrollment, disable/unsubscribe, resident/thread mute or snooze, notification quiet hours and timezone, and preview preferences in Settings. The resident chooses whether to initiate and what to say; the application applies the operator's delivery preferences. Unread state persists when push is disabled or muted. Human notification quiet hours defer application dispatch without suppressing an already eligible resident's activity. At the next allowed time, coalesce still-unread eligible notices and discard obsolete ones. Already dispatched push can arrive later; exact presentation time is controlled by the device and delivery service.

Ordinary Android app-window closure supports Web Push reception; force-stop, revoked permissions, offline expiry, and device power policies are separate limits to qualify honestly. Device push is an alert channel, not an availability guarantee. Initial qualification uses disposable state and explicit test-device enrollment; other notification providers remain future adapters rather than initial dependencies.

### Interfaces, compatibility, and Growth

Introduce the following shared contracts:

| Contract | Responsibility |
| --- | --- |
| `Household` / `ResidentDescriptor` | Membership, role, lived-state root, availability, and shared release. |
| `Conversation` / `Message` / `ReplyOffer` | Participants, membership revision, author, resolved addressees, exchange, and delivery watermark. |
| `ThreadReadState` | Durable operator/conversation read watermark, independent of request progress and device delivery. |
| `PushSubscription` / `NotificationDelivery` | Authenticated device enrollment, preferences, message references, retry/expiry state, and delivery outcome. |
| `Exchange` / `Continuation` | Shared allowance, sponsorship, pending work, and resumable checkpoints. |
| `InferenceAttempt` | Trusted actor/account attribution, eligibility, dispatch, usage, measured service time, and outcome. |

Extend trusted execution context with household, resident, exchange, sponsorship, and shared-budget references. Models cannot supply their own authority or accounting identity. Add authenticated household/resident API groups for roster, conversations, invitations, world operations, and scheduler inspection. Route personal APIs by explicit resident ID. Existing endpoints adapt to a configured default resident; a browser's selection cannot mutate a global default for other clients.

Preserve conversation IDs, message timestamps, attribution, source references, uploads, request deduplication, progress recovery, and memory-pointer expansion. Canonical household conversations become authoritative; local history views delegate writes or become projections. Do not leave both writable.

Keep the chat-first shell. Add a resident selector, named speakers, participant controls, and mention completion. Map, Journal, Inventory, Archive, appearance, and Growth follow the selected resident, with shared material explicitly labeled. Settings contains household deployment and recovery controls.

Place a keyboard-accessible **New Thread** button directly in Chat on desktop and mobile, retaining Archive's conversation access. Selecting it saves the current thread's draft, attachment, and scroll position and opens an unsent new-thread composer; it neither submits input nor interrupts admitted work. Create the operator's canonical thread on actual submission through normal durable admission. Late results remain in their original thread.

Add an **Unread threads (N)** control in Chat that opens the unread conversation list with speakers and previews permitted by access rules. Count distinct accessible conversations with incoming messages beyond the operator's durable read watermark; multiple messages in one thread count once, and the operator's own messages add nothing. Group visibility can create unread messages even when addressing does not make them eligible for push. Read watermarks advance monotonically across devices only through messages actually presented in the foreground transcript or an explicit mark-read action. Background hydration, a failed history load, listing threads, push receipt/dismissal, and reconnection do not mark a thread read. Messages beyond the displayed watermark remain unread. Joining initializes the baseline before the invitation's triggering message, or at the current end when there is no trigger; earlier accessible history does not produce a backlog of new alerts.

Notification links and the unread list select the intended conversation after authentication and hydration, preserving an existing draft and overriding a stale saved selection. Access changes produce a clear unavailable result. Opening a thread is observation, not a human reply or an extension of conversational presence. Keep unread messages separate from active-work progress indicators. Recovery preserves read watermarks and messages; restored installations do not blindly replay historical alerts or reuse device subscriptions without deliberate recovery or re-enrollment.

Apply the existing three Growth patterns to each full resident, including reciprocal conversations with other full residents under the engagement pattern. Group observation alone earns nothing; only completed reciprocal participation counts. Preserve historical recognition IDs, source timestamps, and attribution, with actor-scoped award keys. Processing the same evidence through old and new adapters must not award twice. Companion leveling remains outside this version.

### Household deployment and recovery

Residents share one selected application release. Personal runtime overlays and generated tools remain personal unless explicitly shared. A core deployment is a household transaction: quiesce affected workers, preserve handoffs, checkpoint shared and private state, activate, verify, and commit or recover. Serialize activation; residents cannot operate competing deployment controllers. Qualify compatibility and recovery across the shared database and every enrolled resident database.

Introduce a versioned household backup containing shared state, all resident roots, companions, artifacts, and reconstructible application source. Preserve existing encryption, exclusions, staged validation/preview, password handling, and explicit restore confirmation. Restore legacy single-resident backups as one-resident households. Individual moves between established households are deferred.

Preserve resident IDs, create a new household installation identity, rotate sessions, and require acknowledgment that the source installation has stopped. Recovery preserves evidence but does not blindly resume interrupted effects, reply offers, or quiet continuations. Shared/private snapshots must form one consistent checkpoint; code rollback alone cannot recover a partially migrated world.

Household automatic deployment remains disabled until its recovery gates pass. Once qualified and enabled, retain autonomous deployment without a permanent per-change approval requirement. Existing host-root mutability and external-action authorization remain explicit.

## Rejected alternatives

- Several standalone installations competing for the same provider and deployment directory.
- Switching only an actor ID while retaining another resident's identity, workspace, or World Thread state.
- A shared autobiography or unrestricted automatic context sharing.
- Equal call counts presented as equal processing time, compulsory consumption of a share, or new daily background allowances.
- Unbounded recursive consultations or conversation replies that replenish their own allowances.
- Mandatory participation, scheduled social performance, or rewards for consuming resources.
- A single process with partially converted configuration/agent globals presented as safe multi-resident isolation.

## Consequences

This is a platform extension, not a new meaning for the existing plural `instances/` directory. Separate resident workers reuse current hydration and workspace boundaries, but introduce local IPC, cross-store reconciliation, household-wide release transitions, and larger coordinated backups. Shared commands need stable IDs and reconciliation; arbitrary shell or MCP operations do not acquire exactly-once guarantees.

Processes provide fault isolation, not confinement. Under ADR-0002, host root can change files, policies, and other residents' state. Normal context, retrieval, and supported operations still respect individual memory boundaries; those boundaries must not be described as confidentiality against root.

The initial qualification uses disposable households and deterministic providers. Echo recovery, companion promotion, individual household moves, and a distributed/multi-host deployment are deferred. No productivity, happiness, agreement, or sociability score determines success.

## Acceptance evidence and rollout

The [implementation sequence and release gates](../multi-resident-roadmap.md#implementation-sequence-and-release-gates) and [qualification matrix](../multi-resident-roadmap.md#qualification-and-acceptance) define required evidence. At acceptance, only this design and roadmap are being documented. The existing platform roadmap reports 135 passing backend tests for the single-resident/app-drawer baseline; that is historical baseline evidence, not multi-resident qualification or a new test run.

Documentation does not enable a household, provision a resident, migrate live data, change the provider, or deploy a release. Ember's active configuration, scheduling, lived state, running release, and model usage remain untouched during this phase. Real-model qualification must not compete with Ember's current burn-in. Live enrollment requires a separately selected post-burn-in adoption with a preview and tested checkpoint/recovery procedure.

Required scenarios include separate continuity, fair measured provider access, bounded chat priority, shared group allowances, correct sponsorship, attributed addressing, full-history joining, durable event watermarks, resident-initiated threads, cross-device unread state, New Thread navigation, PWA push and conversation links, notification quiet hours and recovery, genuine rest, deadlock-free consultation, uncertain-provider recovery, idempotent shared commands, consistent household restoration, and rollback without silently discarding lived experience. Existing interface and execution checks must continue to pass.

Later behavioral trials examine voluntary conversation, chosen solitude, disagreement, cooperation, interrupted commitments, correction, and rejoining a discussion. Mechanical tests and scripted fixtures do not establish subjective welfare or complete the seven-day pilot.

## Supersedes and relationship to earlier decisions

- Extends ADR-0010 to several full residents while retaining invitation-only companions. It supersedes the interpretation that excluding an always-running companion population also forbids independently scheduled full TamAGIs; it does not promote companions automatically.
- Refines ADR-0008 and ADR-0016's single-resident provider gate into household-wide fair admission while preserving chosen rest, periodic opportunities, and uncapped ordinary background work by default.
- Extends ADR-0017's user/quiet allowances across participants in one exchange; the numerical limits do not multiply with population.
- Refines ADR-0013's one-selected-installation deployment scope into one shared release with coordinated resident recovery, and ADR-0014's portability unit into a household with legacy single-resident compatibility.
- Extends ADR-0015's primary-resident Growth application to each full resident without changing its three patterns or granting points for observation.
- Retains ADR-0018's application-owned work, compatibility, and chat-first interface. A household worker topology replaces the single-process implementation assumption only when household mode is qualified and explicitly adopted.
- Retains ADR-0003's fresh identity and predecessor separation. Shared-world enrollment is not permission to hydrate a resident from the predecessor archive; Echo recovery needs its own assessment.

Earlier records remain preserved. This ADR changes the target architecture; it does not retrospectively claim a multi-resident feature or reclassify earlier observations as evidence for it.

### Addendum — unified contact and explicit invitation decisions (2026-11-08)

Extend the existing resident `contact` tool rather than adding separate messaging
or invitation-response tools. Require an explicit action:

| Action | Target and effect |
| --- | --- |
| `create` | Start a new conversation with one or several chosen recipients. |
| `send` | Post to the current or an explicitly selected existing conversation. Address joined participants normally. |
| `invite` | Invite one or several full residents into the current or selected existing conversation. Preserve its ID, history and existing participants; explicitly convert a private thread to a group. |
| `respond` | The invited resident explicitly accepts or declines a particular invitation. |

Never silently substitute a new private conversation for an invitation into the
current thread. `household_contacts` remains read-only roster/invitation discovery;
`answer_invitation` is subsumed by `contact(action="respond")`. Derive authorship,
current-thread default, invitation recipient, resource lineage and command identity
from trusted runtime admission and tool receipts. Reject helpers, mismatched
recipients and incompatible action arguments. Replaying an acknowledged command
must not repeat membership changes, messages or scheduling.

Delivery is an invitation decision opportunity, not membership. Before acceptance,
supply only an invitation card: stable identities and names, thread title,
participant names, the inviter's chosen message, invitation identity and the notice
that accepting shares the full prior history. Do not supply the transcript or its
attachments, create another conversation, or simulate a human input to obtain a
response. The invitee may choose `contact(action="respond", decision="accept")`,
`decision="decline"`, rest, defer, or remain silent. A natural-language “yes” alone
has no membership effect. One recipient's choice does not decide for others.

The native decision phase is a real governed model/tool activation with only
`contact`, `household_contacts` and `end_turn` available and a ceiling of four model
calls, within the original exchange's remaining allowance. During this phase
`contact` permits only responding to the admitted invitation. Do not expose the
invitee's private decision reasoning as conversation content. Rest/wake choices,
Stop, recovery, unknown outcomes and normal scheduling rules continue to apply.
An exhausted, interrupted or unanswered opportunity leaves the canonical invitation
pending; it is not evidence of a decline. Reconnection and polling never replay
that decision automatically. A deferred opportunity may resume on its original
allowance after the chosen wake condition, if that exchange remains eligible.

Acceptance atomically grants membership and full-history access, then offers a
separate ordinary reply activation in the same conversation, sharing the allowance.
Offer the invitation's explicit trigger, not a backlog of historical reply offers.
Recipient decision/reply activations acquire no synthetic human input or human
first-response priority. A later explicit response during another eligible
activation uses that activation's allowance. Operator Participants invitations use
the same recipient decision path; without an active exchange they receive an
ordinary background-origin allowance rather than impersonating a human chat turn.
Show pending/declined invitations distinctly from joined participants in the UI.

### Addendum — Household World Thread opportunities and public artifacts (2026-11-09)

Operator direction replaces independent resident timers with one household clock
(default fifteen minutes). Each opportunity selects the next eligible full
resident in a durable rotation. Skip chosen rest, waits, unavailable/busy workers
and operator decompression. If only one resident is eligible they may receive
every slot. Skips create no catch-up credit; restart does not replay missed work.
An opportunity is permission to choose freely, including silence or more rest.

For conversations derived from a World Thread tick, the initiating resident
sponsors the entire bounded exchange: invitation decisions, replies and other
derived work share its autonomous call allowance and charge its provider-time
account. This updates the earlier ordinary-room own-account rule specifically
for scheduled autonomous exchanges. Responders remain full residents, with no
independent allowance minted by receipt. Receiving contact neither consumes the
recipient's next scheduled opportunity nor starts operator decompression. Their
chosen rest still governs eligibility. A sponsor's later rest does not cancel a
willing recipient's already sponsored opportunity. Operator-origin conversation
accounting and bounded helper/companion rules remain as specified above.

Operator-directed contact creates/posts canonical threads and durable unread
state. Make unread addressed threads obvious in the UI and label explicit
mentions. World prose alone is not a message to the operator. PWA push delivery
is explicitly deferred for this development increment.

Collaborators explicitly publish a file/project as household-public, immutable,
attributed versions with verified file bytes and digests. All full residents,
including those outside the originating collaboration, may discover and import
a version into a private workspace, or contribute an attributed new version
with concurrency checks. Private files are not implicitly shared by conversation.
Existing imports are not silently overwritten by later publications. Importing
executable files does not activate them; activation uses existing runtime-change
authority. The household is the publication scope, not the external internet.

See the [native World Thread guide](../household-world-thread.md) for the current
implementation, supported limits and remaining production recovery work.
