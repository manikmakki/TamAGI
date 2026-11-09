# Unified contact and invitation decisions — 2026-11-08

Clean runtime revision `46647a7cb60b83a73fea6f99fe303f6e005c9adb` implements the
ADR-0019 unified-contact addendum and addresses NATIVE-007. The full guarded suite
passed **289 tests in 148.623 seconds**, with **137 guarded Python processes**,
zero denied network operations and unchanged clean source. All test residents
were disposable and providers injected; no real model or live resident was used.
See the [machine report](0019-native-invitations-2026-11-08/qualification.json)
and [summary](0019-native-invitations-2026-11-08/unittest-summary.txt). Raw logs
remain at `/tmp/tamagi-native-invitations-suite-20261108-final`.

The nine new actual-worker tests exercise:

- Inviting into an existing private operator/Willow thread, retaining its ID and
  existing participants, with no new conversation or synthetic human message.
- A real governed invitation-decision activation: no transcript or image before
  acceptance, only the explicit invitation card, and only contact/discovery/end
  tools. The accepted reply then receives the earlier transcript and image.
- Explicit acceptance and decline through `contact(action="respond")`; natural
  language alone leaves the invitation pending. Browsing does not retry it.
- Independent acceptance/refusal for multiple invitees, including an unchanged
  private boundary for the resident who declines.
- Operator Participants using the same decision delivery, with no fabricated
  human chat input or first-response priority.
- Chosen waits deferring the decision until a real wake event, on the same
  exchange; a four-call decision ceiling; forbidden file/new-thread operations;
  Stop preventing a late acceptance; and replay without repeated membership,
  events, work or messages.

The [retained flow proof](0019-native-invitations-2026-11-08/invitation-flow-proof.json)
comes from the acceptance regression, after its assertions. It records the exact
card delivered to Cedar, the actual `respond`/`accept` action dispatched by the
worker, unchanged original/reply conversation IDs, denied pre-join history access,
history visible after joining, one shared exchange, zero invitee human-input
epochs and no extra event on replay. Four injected provider calls covered Willow's
invite/final, Cedar's decision and Cedar's joined reply. This verifies the actual
application protocol, not whether a real model will choose acceptance.

The [Chromium report](0019-native-invitations-2026-11-08/browser.json) passed
against actual native workers. It observed **Cedar · invited** before acceptance,
then joined participation and Cedar's ordinary reply in the same thread. It
checked that conversation count did not grow and no fake user message appeared.
All existing reasoning, attribution, draft, unread, desktop/mobile scrolling,
image, selection and navigation regressions also passed. Zero uncaught JavaScript
errors and zero denied network operations were recorded; all three Python fixture
processes activated the guard. See the
[guard report](0019-native-invitations-2026-11-08/browser-guard.json) and visually
inspected [pending invitation screenshot](0019-native-invitations-2026-11-08/native-invitation.png).

The retained candidate was idle before restart and reopened on `0.0.0.0:7744`
with its original data, password and background-autonomy setting. Ember's PID
152015 and port 7742 were untouched. Existing Willow/Cedar private conversations
were neither merged nor retroactively converted. Send a fresh invite request in
the desired original thread for the real-model interactive check.

These are development integration checks. Real-model choices, sustained autonomy,
PWA push, production migration, coordinated restore/release and other outstanding
ADR-0019 work remain separate qualifications. The guard is Python socket/DNS
protection, not OS network confinement.
