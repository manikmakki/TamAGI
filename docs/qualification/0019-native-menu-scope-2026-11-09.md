# Native household menu scope audit — 2026-11-09

[NATIVE-011](../household-native-testing.md#native-011--resident-context-was-implicit-across-menu-pages)
and the [menu scope guide](../household-native.md#menu-scope-in-a-native-household)
record the audit of every menu item, all Settings tabs, embedded panels and shared
sections. Resident pages now name their owner and expose the existing resident
chooser. Choosing another resident preserves the current menu page. Shared
sections and unavailable household recovery operations are explicit.

The [full Chromium regression](0019-native-menu-scope-2026-11-09/browser.json)
passed against disposable actual workers and injected providers, including all
prior chat, scrolling, vision, reasoning, invitation and World Thread assertions.
New checks traverse Inventory, Companions, Archive, Map, Journal and every
Settings tab as both Cedar and Willow. They verify scope descriptions, headings,
bound iframe resident IDs, private inventory preview/download bytes, shared
notebook content and author attribution, household color preferences, recovery
control availability, menu selection without leaving the current page and late
Archive memory results/counts being ignored after a switch.

Further review added guards for late private file previews and companion detail
responses, and cleared prior permission/evidence panels during selection. The
[final focused menu pass](0019-native-menu-scope-2026-11-09/focused-browser.json)
ran the menu assertions extracted from the canonical browser suite against a
fresh disposable household. It passed the existing menu checks plus held-preview
and stale-permission/evidence regressions. Its generated source and raw logs are
retained at `/tmp/tamagi-menu-details-browser-20261109`; full-run logs are at
`/tmp/tamagi-menu-scope-browser-20261109-final`.

Both runs had three guarded Python processes, zero denied network operations,
zero uncaught JavaScript errors and no external model provider. See the
[full guard report](0019-native-menu-scope-2026-11-09/browser-guard.json) and
[focused guard report](0019-native-menu-scope-2026-11-09/focused-guard.json).
Reviewed screenshots show [mobile Inventory](0019-native-menu-scope-2026-11-09/inventory-mobile.png),
[mobile Settings](0019-native-menu-scope-2026-11-09/settings-mobile.png) and
[desktop Inventory](0019-native-menu-scope-2026-11-09/inventory-desktop.png).

The full run predates the last detail-response guards; the focused pass covers
those changes. Subsequent cache bookkeeping bumps the service-worker cache and
aligns its preloaded script versions with the shell; a local consistency check
passed. The earlier 308-test Python qualification was not repeated for this
frontend-only change. The network guard is Python-level, not OS confinement.

The retained LAN household serves these static changes on refresh. No resident
worker, World Thread scheduler or production process was restarted. Listener
checks retained test PID 359591 on 7744 and production Ember PID 152015 on 7742.
