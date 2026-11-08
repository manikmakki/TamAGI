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
