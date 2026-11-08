# Native household manual testing findings

Recorded from operator testing on 2026-11-07, America/New_York.
Candidate: the native household on port 7744, developed on
`codex/adr-0019-household`. Runtime implementation baseline:
`cb1d309144c8b53255c83f048cbeb157f6aa5124`.

These are operator-reported findings and design feedback. All five items remain
open. The request is to document them now and defer fixes; recording these notes
does not change or restart either the candidate or Ember's burn-in instance.
The earlier [qualification record](qualification/0019-native-2026-11-07.md)
describes the scenarios tested at that checkpoint, rather than establishing that
these newly reported cases work.

| ID | Finding | Classification | Status |
| --- | --- | --- | --- |
| NATIVE-001 | Open in chat can display another resident's thread while retaining the current resident selection | Resident/thread routing | Reported; awaiting reproduction |
| NATIVE-002 | Replace the current character wheel with left/right cycling and background inactive avatars | Requested design direction | Recorded; group presentation remains open |
| NATIVE-003 | Group conversations do not follow new messages when already scrolled to the bottom | Chat scrolling | Reported; awaiting reproduction |
| NATIVE-004 | Images do not appear to attach or reach vision correctly in a group thread | Group attachments/vision | Reported; failure stage unknown |
| NATIVE-005 | Residents describe apparently successful creations/world changes as imagined or nonpersistent | Grounding and persistence investigation | Reported; actual data loss unconfirmed |

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
