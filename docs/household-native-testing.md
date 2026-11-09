# Native household manual testing findings

Recorded from operator testing on 2026-11-07, America/New_York.
Candidate: the native household on port 7744, developed on
`codex/adr-0019-household`. Runtime implementation baseline:
`cb1d309144c8b53255c83f048cbeb157f6aa5124`.

These are operator-reported findings and design feedback. They were initially
recorded without changes. On 2026-11-08 the operator authorized investigation and
fixes in the isolated branch, with the character selector explicitly deferred.
Automated regression results below do not replace operator retesting or qualify
a production cutover. Ember's burn-in instance remains outside this work.
The earlier [qualification record](qualification/0019-native-2026-11-07.md)
describes the scenarios tested at that checkpoint, rather than establishing that
these newly reported cases work.

The [2026-11-08 regression record](qualification/0019-native-fixes-2026-11-08.md)
retains the clean-revision 279-test run and actual Chromium checks for these fixes.

| ID | Finding | Classification | Status |
| --- | --- | --- | --- |
| NATIVE-001 | Open in chat can display another resident's thread while retaining the current resident selection | Resident/thread routing | Fixed in branch; operator retest pending |
| NATIVE-002 | Replace the current character wheel with left/right cycling and background inactive avatars | Requested design direction | Deferred by operator on 2026-11-08 |
| NATIVE-003 | Group conversations do not follow new messages when already scrolled to the bottom | Chat scrolling | Fixed in branch; operator retest pending |
| NATIVE-004 | Images do not appear to attach or reach vision correctly in a group thread | Group attachments/vision | Transport/storage fixes in branch; real-model vision retest pending |
| NATIVE-005 | Residents describe apparently successful creations/world changes as imagined or nonpersistent | Grounding and persistence investigation | Retained creations verified; query/context fixes in branch; behavioral retest pending |
| NATIVE-006 | Saved reasoning loses bubble/Markdown formatting and live reasoning accumulates below multiple residents' messages | Reasoning presentation and progress lifecycle | Fixed in branch; operator retest pending |
| NATIVE-007 | A resident asked to invite someone into the current conversation creates a separate private contact thread | Resident invitation tooling | Operator confirmed successful Cedar-to-Willow invitation and visible pending state on 2026-11-09, after NATIVE-008 fix |
| NATIVE-008 | Redundant invitee mentions reject an otherwise valid invitation and hide the repair guidance | Invitation argument handling | Fixed; operator confirmed the real-model retry worked on 2026-11-09 |

## NATIVE-001 — Open in chat and selected resident disagree

Observed: clicking **Open in chat** in the main interface while viewing a
different TamAGI can open Willow's conversation while River remains selected.

Expected: opening a conversation should present a coherent resident/thread
context, with the active identity and intended recipients clear. Navigation must
preserve the appropriate draft and must not accidentally route subsequent input
to the previously selected resident. For a group, the selection behavior should
respect its actual participants and the agreed selector design.

Follow-up: reproduce the main-interface Open in chat path across residents and
inspect selection, displayed history, draft ownership and the resident captured
for the next send. Record private-thread and group-thread behavior separately.

2026-11-08: the general thread-opening handler did not reconcile its participants
with the selected resident. It now selects an available participant when needed,
and preserves selection when already a group participant. Navigation guards
discard stale opening requests. Drafts retain their thread/resident ownership.

## NATIVE-002 — Left/right resident selector

Requested direction:

- Use left/right controls to cycle through the TamAGI the user wants active.
- Keep the active avatar prominent in the foreground.
- Show inactive avatars to its left or right in the background, smaller and
  shaded, with static poses.
- Consider bringing participating residents forward in a multi-resident thread
  to show that they are active.

The left/right arrangement is the requested replacement for the current wheel.
The group behavior needs more thought: distinguish the resident selected by the
user from residents participating in the thread or currently responding. Decide
how multiple foreground avatars, animation and transitions communicate those
states without silently changing the user's selection or send context. No final
group layout has been agreed.

## NATIVE-003 — New group messages do not keep the transcript at the bottom

Observed: conversation threads in multi-resident mode do not appear to scroll
with incoming messages. The operator explicitly checked that the transcript was
already scrolled as far down as possible.

Expected: a transcript pinned to the bottom should continue following new
messages, including another resident's response and canonical history refreshes.
Reading older messages should preserve the reader's position.

Follow-up: check initial send, successive resident replies, interim/tool content,
partial group refreshes and final completion on desktop and mobile. The earlier
browser check established preservation of a scrolled-up position; it did not
establish correct bottom-follow behavior in this reported case.

2026-11-08: canonical child refreshes always restored the previous absolute
offset. They now capture whether the reader is pinned to the bottom before
rebuilding content and follow only in that case. Final completion and live
progress preserve a scrolled-up reader as well. Attachment previews reserve
their height so decoding does not move the transcript later.

## NATIVE-004 — Vision/image attachments in group threads

Observed: vision appears broken in a group thread; the image does not seem to
attach correctly. The report does not yet distinguish attachment rendering,
storage, routing or delivery to the vision model.

Expected: the original attachment remains associated with its human message and
reaches each intended responder with the correct message and resident context.
Its availability should be clear in the interface, and an attachment failure
should not silently appear to be a successful image submission.

Follow-up: compare the same image in private and group conversations. Trace the
attachment through upload/input handling, durable request admission, group
addressing, each admitted worker payload and the provider's image content. Check
that switching residents during upload cannot change attachment ownership.

2026-11-08: the retained image submission contained 8,456,108 base64 characters.
The first worker stored its image, but its tokenizer request was rejected by the
broker's 8 MiB JSON ceiling before any vision generation. That ceiling was smaller
than a permitted 8 MiB image after base64 expansion. Broker requests now allow
16 MiB including context; decoded attachments remain limited to 8 MiB and are
validated before input admission. Multiple large attachments in one retained
context can still exceed the total request ceiling.

There was also a group handoff gap: binary content belonged to the transient
work payload and only accompanied an operator-triggered activation. A peer reply
could become the next resident's trigger, losing the original image. New images
are now committed atomically with their canonical message, with media type, size
and SHA-256 metadata. Workers load verified bytes for images in their admitted
transcript, including historical images during peer-triggered and later turns.
Retrieval enforces membership and the current thread's admission watermark.
The browser renders the same retained attachment through authenticated retrieval.

Regressions cover a valid PNG whose encoded size exceeds the former ceiling,
two actual worker responders, a peer trigger, later image-free input, invalid
submission rejection, access boundaries and reopening the household. These
verify transport with an injected provider, not real-model image understanding.
The original failed submission predates canonical attachment storage; resend it
for the next interactive vision check.

## NATIVE-005 — Creations and world changes described as imagined

Observed: residents appear to think that artifacts they create, additions to the
world graph and similar changes are not persisting, or are only imagined. Tool
calls appear successful to the operator. The operator wondered whether the
temporary workspace contributes to this impression.

Open question: is this a mismatch between prompts, tool evidence and the
resident's explanation, or is some state actually missing, stored in the wrong
place, or absent from later context? Successful tool status alone does not settle
that question. This is not yet a confirmed data-loss report.

Expected distinction: a simulated world change can still be durable application
state. A saved file or artifact should be described according to its retained
bytes and receipt. A disposable `/tmp` household has a limited retention lifetime;
that is distinct from a creation being imaginary or disappearing after a turn.

Follow-up: choose concrete reported examples and compare their tool arguments,
receipts, read-back results, resident ownership, canonical world revisions and
subsequent context. Verify persistence across later turns and, in an isolated
recovery test, restart. Inspect the framing of simulation, imagination, artifacts
and workspace lifetime in prompts and tool responses alongside those storage
checks. Preserve the distinction between shared topology, private interpretations
and verified artifact bytes when drawing conclusions.

2026-11-08: the operator identified the end of the three-resident thread about
building a dice skill for Snake Eyes and adding locations. Read-only inspection
of the retained household found:

- Cedar and Willow each have a saved `skills/diceroller/main.py`, with successful
  write receipts. River's separate private workspace has no such file. These are
  saved source files; their existence alone does not establish tool activation.
- The canonical world retains The New Home, The Workshop, The Garden, The Commons
  and other created places. Their later descriptions as purely imagined additions
  are contradicted by the saved records.
- The default graph query returned a small selection of quests, skills, mysteries
  and lore, omitting locations altogether. Residents treated that partial result
  as a complete inventory. Reads also lacked a fresh canonical-world projection,
  so a peer's recently saved place could be missing from the local projection.
- Residents treated separate private filesystem observations as conflicting
  evidence about one shared directory. River's local absence does not establish
  absence in Willow's workspace. Some later prose repeated older receipts.

Graph reads now refresh visible canonical places, include a location sample in
the default view, and label query results as limited rather than inventories.
Native runtime context states private workspace ownership, explicit sharing and
the distinction between a durable simulation change and an unexecuted story.
No resident identity or conversation has been rewritten to enforce an answer.

An automated real-worker test creates a shared place and private file, stops the
whole disposable household, reopens it, checks the original revision and bytes,
and queries the place through another resident's actual graph tool. This verifies
those persistence paths. It does not guarantee future model explanations, repair
old contradictory memories, or qualify shared artifact publication/migration.

## NATIVE-006 — Reasoning formatting and a live bubble spanning resident turns

Reported on 2026-11-08 with screenshots: saved steps display as a plain browser
`Activity · 2 steps` disclosure, while an expanded live Reasoning bubble stays
below several messages and collects steps from different residents/activations.

Confirmed causes:

- The durable shell rendered saved reasoning with a separate `<details>` element
  and `textContent`, bypassing the existing styled bubble and Markdown renderer.
- Native child completion becomes `thread_changed` so the shared exchange stays
  running. The shell retained all prior child progress in the root work buffer
  and replayed it after every canonical refresh. The legacy renderer's single
  `currentInterimGroup` merged those replayed steps and pinned them below history.

Saved and live reasoning now share the existing collapsible bubble structure,
Thinking/Reasoning labels, step counts and Markdown rendering. Native Markdown is
sanitized. Each progress bubble displays its actual resident's name. Progress
and tool tags are scoped to the worker activation grant; completion retires only
that activation. A canonical publication also carries its trusted grant ID, so
history refreshes can discard progress already represented by a saved message.
Existing saved messages get the corrected presentation without rewriting history.

Regressions use actual resident processes and generated read tools. They cover
separate live/saved bubbles, Markdown and HTML sanitization, refreshing while the
next resident is still working, and repeated activations of the same resident in
one exchange. They also retain draft, scroll, attachment and navigation checks.

The [reasoning regression record](qualification/0019-native-reasoning-2026-11-08.md)
retains the 280-test clean-revision run, Chromium checks and expanded screenshot.

## NATIVE-007 — Resident contact creates a new thread instead of inviting into the current one

Operator report: in the latest private conversation with Willow, the operator
asked Willow to ping Cedar and invite them to that conversation. Willow said the
message went through to a new thread. The operator expected Cedar to join the
existing operator/Willow conversation.

Read-only inspection confirmed the distinction:

- The original thread, `conversation-91c54f8765d94688832b43e954b6b4eb`, still has
  only the operator and Willow as participants.
- Willow called `contact` with `participants=[Cedar]` and an opening message,
  omitting `conversation_id`. The successful tool receipt returned a different
  thread, `conversation-02a8a6bfa7d64014ba3e0e2c3c3f2f72`, containing only Willow
  and Cedar. The operator was not included.
- No canonical `conversation_invitation` was created. Cedar responded in the
  new private contact thread; that did not join the operator's original thread.

Current implementation: the resident `contact` tool can create a thread or send
to an existing thread's joined participants. It has no operation to invite a new
participant into an existing thread. The operator Participants route separately
supports converting a private thread to a group and recording full-history
invitations, but that operation is not exposed through resident invitation tools.

Expected follow-up: distinguish new contact from an explicit invitation into the
current conversation. Preserve the original thread and participants, disclose
full-history access, and let the invited resident choose membership before they
can read or respond there. Include invitation delivery/activation, refusal and
replay behavior in the regression coverage. Never silently substitute a new
private thread for a request to join the current discussion.

The initial investigation changed documentation only. Follow-up implementation
extends the existing `contact` tool with explicit create/send/invite/respond
actions. An invitation converts the original private thread to a group while
preserving its identity and existing participants. The invitee receives only a
card before choosing, through a bounded actual model/tool activation. Acceptance
adds membership and offers a normal reply in that same thread; decline or plain
text alone does not share history. Operator Participants uses the same flow.
Pending and declined invitations are visible in the participant strip.

Regression coverage includes actual worker acceptance, refusal, multiple invitees,
pre-join transcript denial, no synthetic operator input, replay, Stop, bounded
decision calls, and deferral through a chosen wake event. Existing test household
threads are not merged or retroactively modified; request a fresh invitation in
the desired thread for the interactive retest.

The [unified-contact regression record](qualification/0019-native-invitations-2026-11-08.md)
retains the 289-test clean-revision qualification, actual invitation flow proof
and Chromium pending-to-joined checks.

## NATIVE-008 — Valid invitation rejected when the model repeats recipients in mentions

Reported on 2026-11-09 in the operator's latest private Cedar conversation,
`conversation-af7f5b63f733452fad4eb67496b7e98f`. Cedar used `contact` with
`action="invite"`, the correct current conversation ID and Willow's stable ID in
`participants`. The same ID was also included in `mentions`. The coordinator
rejected all invitation mentions, and the worker hid the argument error behind
the generic message "Household coordinator rejected this resident operation."
Cedar consequently speculated that resident invitations were unsupported or
forbidden. No invitation record was created and Willow received no decision card.

Invitations now accept mentions that only repeat IDs in `participants`. They do
not address the ordinary thread notice to unjoined residents or bypass the
invitation decision. Other mentions are rejected with explicit repair guidance,
before any invitation or delivery is created. The tool descriptions clarify
which arguments belong to each action, including that `respond` answers a
received invitation rather than an ordinary greeting. Workers preserve bounded
coordinator argument-error text for HTTP 400 responses; authentication and other
failures retain the generic rejection.

Actual-worker regressions reproduce Cedar inviting Willow with both recipient
fields and check acceptance in the same thread. A second regression rejects an
unrelated mention without effects, delivers specific guidance to the model and
successfully retries the corrected invitation. Existing saved messages and
failed requests are not rewritten or replayed automatically.

Focused offline validation passed all 25 invitation and worker tests in 61.411
seconds, with 35 guarded Python processes, zero denied network operations and
unchanged source during the run. Providers were injected and homes disposable;
no real-model request or retained resident interaction was submitted. The report
and raw logs are retained at `/tmp/tamagi-invite-mentions-focused-20261109-final`.

## NATIVE-009 — Silent World Thread turns disappear or look empty in Journal

Reported on 2026-11-09 after the retained household's first three autonomous
opportunities. Read-only inspection found durable world_turn records for Cedar
(28 receipts: 27 succeeded, 1 failed), Willow (38: 36 succeeded, 2 failed), and
River (32: 24 succeeded, 8 failed). All three ended silently using
end_turn(status="waiting", wake_event="evaluation_completed"). Cedar's existing
Journal card had no body; Willow and River had no card at all.

Two reporting defects caused this. Terminal end_turn intentionally returns no
resident narrative, leaving the card blank. MonologueLog implements __len__, so
`if world.monologue_log` skipped the first append to an empty Journal. The open
Journal also did not refresh when a background tick finished.

The Journal now shows a distinctly labeled Recorded activity summary derived
from actual receipts, including successful/failed tool counts, tool names and
chosen wake condition. Related evaluation jobs show their current recorded
status separately from the historical tick result. Existing resident prose and
entry IDs are preserved. Missing legacy cards are read-only projections of
world_turn records; browsing neither rewrites history nor calls the model.
Future ticks append even to a fresh empty logger, and the household's five-second
refresh updates an open Journal.

Investigation also found a real evaluation handoff bug: tick completion could
mark a resident unavailable after an independent evaluation had already started,
leaving its generation queued indefinitely. Worker readiness now includes its
independent execution lease, and coordinator availability reconciliation protects
live broker executions even if a readiness snapshot is stale. This does not
turn a chosen wait into a new autonomous opportunity.

Regression coverage includes a fresh worker's first silent wait, retained-entry
identity, missing legacy projection without writes or model calls, resident
isolation, an evaluation completing after its initiating tick waits, stale
readiness, and Chromium Journal refresh/details at mobile width. See the
[Journal qualification record](qualification/0019-native-journal-2026-11-09.md).

### NATIVE-009 follow-up — Explicit Journal attribution

The operator noted that cards did not visibly identify the resident who took a
turn. Household Journals remain scoped to the selected resident; the shared
clock can report another resident's latest opportunity. The page now names its
owner in the heading, every entry names its resident, and the detail panel
repeats the attribution. Names come from the coordinator roster for the captured
resident request, after the existing selection-epoch guard. Switching Journal
owners clears the prior detail view and cached entries. Resident narrative and
stored records are unchanged. This is a frontend-only change served on browser
refresh, with no worker restart or interruption of scheduled turns.

## NATIVE-010 — Old coordinator did not finish shutdown with stranded evaluation requests

Observed during the authorized 2026-11-09 reload for NATIVE-009. The public 7744
listener closed and all three native worker processes exited, but old coordinator
PID 355694 remained alive after two Ctrl+C requests. Read-only preflight had
confirmed no active chat/world work and no running or unknown upstream attempt;
there were stranded queued evaluation attempts. Targeted termination of only
that old coordinator was needed before reopening the retained household.

The new process recovered all Journal cards and retained resident files/history.
Prior evaluation jobs settled as failed or interrupted, not successful, and were
not replayed. This observation does not establish the exact shutdown cause.
Follow-up before production adoption: reproduce shutdown with queued independent
evaluation requests, inspect the coordinator/private-server drain, and add a
bounded shutdown regression without abandoning uncertain upstream occupancy.
The Journal/availability fix does not claim to resolve this separate shutdown
case.
