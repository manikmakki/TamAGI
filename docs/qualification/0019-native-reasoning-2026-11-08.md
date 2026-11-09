# Native reasoning presentation regression — 2026-11-08

Clean runtime revision `3cca39f685393cd31ee666ad75ba1af0e324eaea` addresses
[NATIVE-006](../household-native-testing.md#native-006--reasoning-formatting-and-a-live-bubble-spanning-resident-turns).
The full guarded suite passed **280 tests in 99.633 seconds**, with **110 guarded
Python processes**, zero denied network operations and unchanged clean source.
All test providers were injected and all resident roots disposable. See the
[machine report](0019-native-reasoning-2026-11-08/qualification.json) and
[summary](0019-native-reasoning-2026-11-08/unittest-summary.txt). Raw logs remain
at `/tmp/tamagi-native-reasoning-suite-20261108-final`.

The actual-worker regression covers three activations in one exchange: Cedar,
Willow, then Cedar again. Progress and completion boundaries retain distinct
grant/worker-work identities; each saved message has its trusted grant identity
and only its own interim steps.

The [Chromium checks](0019-native-reasoning-2026-11-08/browser.json) passed with
actual native worker processes and generated read tools. They verify styled
collapsible bubbles, sanitized Markdown, resident labels, completed steps directly
above their own messages, independent live steps while the next resident works,
refreshing during that work without replaying completed reasoning, and final
completion without a leftover live bubble. Existing navigation, draft, attachment,
desktop/mobile scrolling and color checks also passed. Zero uncaught JavaScript
errors and zero denied network operations were recorded; all three Python fixture
processes activated the guard. See the
[guard report](0019-native-reasoning-2026-11-08/browser-guard.json) and visually
inspected [expanded reasoning screenshot](0019-native-reasoning-2026-11-08/native-reasoning.png).

The retained candidate was idle before restart and was reopened on
`0.0.0.0:7744` with its existing household, password and history. Ember's PID
152015 and loopback port 7742 were untouched. No real model call was made for
these checks. Existing messages get the corrected renderer without rewriting
their stored content. Operator retesting remains pending. This is a focused
development regression check, not ADR-0019 production-cutover qualification.
