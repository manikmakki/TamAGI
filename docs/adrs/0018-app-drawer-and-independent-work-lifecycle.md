# ADR-0018: An app drawer, chat-first navigation, and independent work lifetimes

Status: Accepted design by operator direction on 2026-11-05; implemented with mechanical qualification recorded in the roadmap. Sustained behavioral observation remains separate.

## Context

Chat is TamAGI's primary relationship surface. The animated avatar, conversation, uploads, and existing visual language should remain familiar. Exploration outside chat should feel like opening a game's Map, Journal, or Inventory, while accommodating a cabin, wizard tower, city, or another setting without imposing one of them.

Navigation currently spreads related features across the chat drawer, Mind and memory flyouts, the graph, `/home`, and separate settings and appearance pages. `/home` introduces a competing dashboard and visual language. The operator chose a central app drawer and consistent destinations instead of a broad redesign of chat.

The operator also identified a lifecycle requirement: tools and activity must continue while another view is selected. Inspection before implementation on 2026-11-05 found more than a presentation issue. The following bullets describe that earlier snapshot; the linked source files now contain the implemented lifecycle:

- [The WebSocket handler](../../backend/api/chat.py) awaits `agent.chat()` and sends progress directly through that socket. Its cleanup calls `on_conversation_ended` for the last completed conversation.
- [The chat runner](../../backend/core/runtime/chat_turns.py) awaits progress callbacks during execution. A broken socket can fail a turn at an emission boundary. Completed receipts survive, but later work can stop.
- [The agent](../../backend/core/agent.py) awaits its chat child task. That separation supports quiet-turn preemption; it does not transfer ownership to a transport-independent service. Normal input is appended in memory before processing, with full conversation persistence generally occurring at the end.
- [The client](../../frontend/index.html) uses full-page navigation for several destinations and global selected-conversation/request state. A late result can interfere with a different selected conversation. Its request ID correlates events but does not deduplicate submission or provide reconnection replay.
- [Runtime mutation middleware](../../backend/main.py) wraps most API POST requests in a user execution. Memory search currently uses POST, so observation can also enter an interruption-capable lifecycle without a new conversational prompt.

Providing a dependable home means a choice to browse a journal should not accidentally cancel an accepted commitment or tell TamAGI that its visitor has left. Browser focus is not a reliable owner for work.

## Decision

### One app shell with Chat as its home

Retain the existing PWA and introduce a shared shell around its views. The top-left hamburger opens a compact drawer with icons and labels. Chat is the default destination and remains directly accessible from every view. Settings sits at the bottom of the drawer.

| Destination | Responsibility |
| --- | --- |
| Chat | The primary avatar and conversation experience, including existing uploads and Growth entry points. |
| Map | The existing world graph: places, connections, world objects, and contextual pursuits. Knowledge and self-model connections remain available; the label does not promise a geographic simulation. |
| Journal | World Thread and Mind entries, current activity, rest/wake information, and a clearly labeled household notebook with attributed shared notes. |
| Inventory | Creations in the active workspace's `workshop/`, with available descriptions, current-file versus saved-version distinctions, supported previews, and downloads. |
| Companions | The durable companion roster and existing conversation/task invitations, with attributed histories and explicit sharing. |
| Archive | Separate Memories and Conversations views. Remembered interpretations and source transcripts retain their distinct meanings. |
| Settings | Appearance and rig editing, connections and model configuration, and a distinct operator area for deployments, backup/restore, predecessor archives, and detailed inspection. |

Archive is the initial label; renaming it does not change its responsibility. Existing Growth explanations remain reachable from the chat status display within the shell. Pursuits remain contextual interests or commitments rather than a required new task dashboard.

Use chat's typography, colors, spacing, and control language throughout. The menu concepts and restrained symbols provide the game character. Do not introduce a fixed illustrated environment, chores, or progression requirements to justify navigation. Preserve the existing avatar choices and customization.

Move useful `/home` sections into these destinations and retire its separate dashboard as the consolidation completes. Preserve existing bookmarks, direct routes, and links through compatible view routing or redirects. Views support direct loading and browser Back/Forward.

The shell owns connection management and client state independently of the selected view. Returning to Chat restores its conversation, draft, pending upload selection, scroll position, and active progress. A full reload restores server-persisted conversation/work state; an unsent draft or browser-held attachment is not promised to survive a reload or browser closure. In-app view switching does not discard either.

Do not require the Chat DOM, canvas animation, browser tab, or browser process to stay active for execution. Hidden views may suspend rendering while the shell retains their state. Do not use a hidden iframe, pop-up, foreground lock, or service worker as the owner of agent work.

### Inventory has a deliberate boundary

Use the [active workspace conventions](../workspace-layout.md), not shell `~/workspace` or the repository's legacy workspace. Discover regular creations within `workshop/` without requiring the resident to remember `register_artifact` merely to make a creation visible. Do not scan the rest of the host or turn Inventory into a general filesystem browser.

An editable current file and a content-hashed saved artifact are distinguishable versions. Existing registered snapshots outside `workshop/` remain accessible in a labeled earlier-saved-artifacts section during migration; files and references are not silently moved or deleted. Filesystem discovery does not establish successful task completion or create memories from inferred descriptions.

Use existing authored descriptions where available and plain filenames otherwise. Do not add an inference call to manufacture a description for every file. Preview support is bounded by supported formats; active content cannot inherit the application session or operate privileged APIs. Unsupported formats remain downloadable. Detailed preview behavior can evolve separately.

This decision does not add editing, an IDE, arbitrary workspace browsing, a “Discuss this” action, or new chat-reference attachments. Existing chat upload behavior is preserved.

### The application owns admitted work

Add a small application-owned turn manager above the existing execution lifecycle. Initially it uses the current FastAPI process, an owned task registry, and canonical SQLite records; a separate worker service or new queue infrastructure is not required.

Transport handlers submit, observe, and control work. They do not own its lifetime. This applies to interactive chat and its quiet continuations, and to existing companion invitations exposed by the consolidated interface. Autonomous work continues through its existing execution service. Avoid a second independent tool runner or budget system.

On admission, authenticate and validate the input, resolve or create its conversation, and durably record the accepted input and a stable request identity before acknowledging acceptance. The record links the conversation, actor, input, execution when started, current status, and retained result. The message is persisted once; the runner does not append a second copy on startup. Persist attachment bytes or a durable reference before acknowledging an attached request.

Accept a client submission identifier scoped to the installation/resident and validate its payload fingerprint. Retrying the same identifier with the same payload returns the existing admission/result; reuse with different content is rejected. Deduplication happens before creating input events, interrupting other work, allocating an execution, or charging a model call. Losing an acknowledgment is not permission to repeat a shell operation.

Maintain the existing serialized resident execution and provider concurrency. Use a bounded admission queue with visible queued status; reject overload before acknowledgment rather than silently dropping accepted input. Reconnecting clients and additional observers do not create work. Create the trusted `ExecutionContext` inside the application-owned runner, not in an HTTP/WebSocket request context that can finish before the work does.

Classify observation, conversational admission, work control, and actual state mutation by their semantics rather than HTTP verb alone. Memory search, view loading, event subscription, and other observations must not start an interrupting user execution even if a compatibility endpoint uses POST. Retain authentication and appropriate mutation/checkpoint barriers for operations that really change state.

REST and WebSocket paths use the same admission service. Preserve existing response fields through adapters; an asynchronous client can receive an acceptance acknowledgment and then inspect or subscribe to the result. Compatibility handlers may wait for a result, but losing that waiter does not cancel the admitted execution.

### Connections observe retained state

Persist final responses and terminal statuses before delivery. Record bounded operational progress with stable event IDs/cursors, conversation and request attribution, and receipt references. Full tool results continue to live in canonical receipts rather than being copied into every UI event. Delivery or rendering failure cannot convert a successful receipt into a failed operation.

The execution publishes to this service, not directly to a socket. Fanout uses bounded subscriber buffers. Slow or disconnected observers detach or resynchronize without blocking inference or dispatch. This is durability for work state and selected progress, not a promise to retain every transient animation or stream token.

On refresh, reconnect, or another authenticated client, retrieve the current request snapshot and events after the client's cursor. Establish a consistent snapshot/cursor boundary so events arriving during reconnection are not lost. Deduplicate delivery by event identity. If a cursor has expired, return a fresh snapshot and retained result; do not rerun work to reconstruct a UI.

Route every event to its request and conversation, independently of the currently visible thread. A result in another conversation updates that conversation and its unread/activity indicator. It does not change the selected thread or insert text into it. Multiple observers see the same work and result; they do not duplicate exchanges, Growth recognition, or tool calls.

Provide concise shell-wide activity, completion/unread, and actionable-wait indicators. A guarded-profile approval is addressed to its existing request, has a server-owned deadline, and can be handled from another view. Navigation or reconnection does not grant approval, reset its timeout, or repeat the operation.

### Interruption, presence, and recovery are explicit

Navigating, selecting a thread, hiding a tab, losing the network, closing the browser, or logging out detaches observation. Already admitted bounded work can finish and retain its result while the backend remains available. These events do not submit input, request a quiet turn, reset allowances, or cancel work.

Actual new input retains ADR-0017's preemption of quiet work and ADR-0008's priority over autonomous work. Preserve the existing handling of an initial user segment; changing views does not introduce a new interruption policy. Add an authenticated, request-specific Stop control usable from the shared shell. It stops new child work and preserves completed receipts; an effect whose result cannot be established remains `outcome_unknown`.

Separate conversational visit state from socket state. Use one inactivity/decompression deadline: the later of the visit's last accepted conversational input or completion of its admitted user work, plus the configured resume interval (initially fifteen minutes). Do not expire a visit while its work is queued or running. Expiration closes the visit and permits the next ordinary World Thread opportunity without starting another fifteen-minute delay. An explicit end-of-visit action may request closure once admitted work settles; it does not cancel work or add a second delay. Route changes, observation polling, and connection heartbeats do not extend the deadline. Opening Archive or changing its selected historical thread does not itself end a visit. This operational visit boundary does not replace ADR-0015's two-hour Growth occasion boundary.

Departure/consolidation processing is deduplicated against its conversation/message watermark. Multiple tabs, reconnects, or an unchanged transcript cannot generate repeated departure summaries. The connection's existence does not indefinitely suppress independent World Thread opportunities. Chosen sleep and event waits retain their own wake conditions.

Browser disconnection and backend process failure have different recovery procedures. On deployment, shutdown, or crash, use the existing quiesce and receipt recovery rules. Unfinished executions become interrupted and dispatched unfinished operations become `outcome_unknown`. In this first implementation, admitted but unstarted requests are also surfaced as interrupted after restart rather than silently replayed. Queued quiet continuations are discarded as under ADR-0017. Reconnection retrieves that status; further execution requires a deliberate new action with retained evidence, without automatic retries of uncertain effects.

Keep authority and observation/control authentication server-side, including subscriber reconnection and approval/stop commands. These cooperative controls remain root-mutable under ADR-0002. Accepted work keeps its original budgets and authority; remaining connected is not a resource allowance.

## Rejected alternatives

- Replacing chat with another dashboard or redesigning it around an operator console: weakens the primary relationship surface.
- Keeping independent page menus and a separate `/home` visual system: preserves the fragmentation this change is intended to resolve.
- Prescribing cabin/tower/city illustrations: constrains the resident's chosen setting.
- Keeping a hidden chat page or relying on browser background execution: makes reliability depend on a client remaining alive.
- Preserving one WebSocket while leaving work owned by its handler: improves in-app browsing but leaves reloads, connection failures, and late-result routing unresolved.
- Ignoring socket-send exceptions alone: does not provide durable admission, result retrieval, deduplication, or correct presence semantics.
- Unbounded detached tasks or a fresh resource pool per connection: undermines the existing execution and budget contracts.
- Exactly-once claims for arbitrary effects or automatic replay after process failure: receipts can still have uncertain outcomes.
- Turning Inventory into an explorer/IDE or expanding chat attachments as part of navigation: exceeds the agreed scope.

## Consequences

The navigation change includes a bounded backend lifecycle refactor. A frontend-only drawer does not satisfy the decision. Submission, execution, delivery, conversation selection, and visit state acquire explicit responsibilities without replacing Python/FastAPI or requiring a frontend framework migration.

Work may finish after the user closes the browser. The interface must communicate accepted/running/completed/interrupted state and offer an explicit Stop action; it cannot imply that closing a view cancels a commitment. Global activity is concise, while detailed evidence remains optional inspection.

Operational records and retained results belong in canonical storage and backups. Event retention may be bounded if current snapshots and final outcomes remain recoverable. Preserve rollback-compatible migrations and existing deployment recovery. This does not promise uninterrupted inference across application restarts or unrestricted histories of every streaming update.

At acceptance, Inventory's folder discovery was a change from the registered-snapshot-only list. The implemented Inventory now distinguishes current workshop files from saved versions. Existing artifacts, notebook sharing, memory attribution, visitor continuity, Growth, authentication, uploads, rig settings, and recovery controls retain destinations through the consolidation.

## Implementation order

1. Establish application-owned admission, bounded execution ownership, durable result/event inspection, subscription recovery, and explicit interruption/presence behavior behind compatible chat APIs. Qualify these independently of the visual redesign.
2. Introduce the shared shell, drawer, addressable views, and persistent client conversation state. Preserve chat's appearance and current capabilities.
3. Consolidate Map, Journal/notebook, Inventory, Companions, Archive, and Settings; keep legacy-route adapters until their destinations work. Add workshop discovery with clear current/saved-version presentation.
4. Exercise real browser navigation and disconnection against disposable instances, then release through the existing validation and supervisor lifecycle. Retain the findings as mechanical evidence separate from the ongoing behavioral pilot.

This ADR was recorded before implementation. Acceptance of the decision did not itself establish implementation evidence.

## Acceptance evidence

At acceptance, evidence consisted of the source inspection above and the operator's navigation decisions. Earlier receipts/budget/restart tests did not by themselves establish transport-independent continuation. The implementation adds `test_interactions.py`, `test_inventory.py`, the retained application smoke, and disposable Chromium scenarios in `scripts/navigation-browser-smoke.mjs`. The [implementation guide](../navigation-and-work.md) and [roadmap](../platform-roadmap.md#interface-consolidation-and-work-ownership) distinguish those mechanical checks from the behavioral pilot. Acceptance scenarios:

- Chat remains the default; every drawer destination, direct link, legacy route, and Back/Forward transition is usable. Keyboard navigation, drawer focus return, accessible labels, and narrow/mobile layouts work without imposing a world setting.
- Switching views during a tool call and during a quiet continuation preserves the conversation, draft, upload selection, scroll position, limits, and ongoing progress. Returning requires no extra prompt. Browsing creates no departure event or inference call.
- Closing or refreshing the page, hiding the tab, losing a connection, and a slow/failing subscriber leave the same accepted execution running. It finishes once and the result is retrievable after reconnect.
- Lost acknowledgments, duplicate submissions from multiple clients, and replayed events do not duplicate input, tool effects, Growth, or final messages. Reused keys with changed content are rejected. Queue overflow is explicit before admission.
- A result arriving while another thread is selected remains attached to its original thread. Multiple observers do not steal results or reset each other's selected conversation.
- Reconnection during active progress has no snapshot/subscription gap; expired cursors recover through a snapshot. A final-result delivery failure preserves both the completed result and its receipts.
- New actual input and explicit Stop retain their intended interruption behavior. Browsing, reconnecting, and approval responses do not masquerade as new input. Approval deadlines and authority checks survive view changes without implicit authorization.
- Visit inactivity and decompression use the single defined deadline without a socket-close dependency or a doubled delay. Memory search and other observations do not create interrupting user executions. Unchanged conversations and multiple tabs do not duplicate consolidation/departure, while World Thread opportunities still resume when appropriate.
- Restart/deployment during execution preserves completed effects, exposes interrupted/unstarted work and unknown outcomes, and does not repeat effects or resume queued quiet work automatically. Existing budget, backup/restore, and recovery scenarios still pass.
- A creation in `workshop/` appears without manual registration. Current files and saved versions are labeled, older registered artifacts remain reachable, unsupported previews download, and preview content cannot operate with the application's session.
- Chat uploads, avatar/rig choices, memory-pointer expansion, Growth explanations, notebook attribution, companion memory boundaries, and operator recovery controls remain reachable and functional.

Record test names, disposable application/browser results, and any untested cases in the roadmap at implementation time. No live success, emotional state, or subjective welfare is inferred from this design record.

## Supersedes

Refines ADR-0014's interface organization: its home/workshop and operator functions become shared-shell destinations instead of an independent dashboard. It replaces the current implementation assumptions that a chat socket owns execution and socket closure denotes conversational departure.

ADR-0005's execution/receipt semantics, ADR-0008's rest and scheduling commitments, ADR-0016's context/autonomy policy, and ADR-0017's bounded turns, quiet continuations, and actual-input interruption remain in force. This record changes transport ownership and presentation, not those allowances or the resident's identity. Historical design documents and earlier records remain preserved.
