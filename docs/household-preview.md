# Interactive household development preview

For interactive work in the normal TamAGI interface, use the later
[native household candidate](household-native.md). This scripted panel remains a
foundation-debugging tool.

This is a separate browser test panel for [ADR-0019's experimental foundations](household-foundations.md). It uses three newly seeded fixtures, real private runtime subprocesses, canonical household services, and a scripted provider. It does not start the production TamAGI app or connect a language model. The existing chat shell, onboarding and PWA push are not part of this panel.

Run from a development checkout:

```bash
.venv/bin/python scripts/household_preview.py --port 7743
```

The launcher prints `http://localhost:7743/`, a generated preview password, and the newly created fixture root. It binds only to `127.0.0.1`. Open the URL and use that password; it is separate from the normal TamAGI password. The preview's signed session cookie includes its household identity, so it does not replace the running app's login cookie on the same hostname.

For a remote host, forward the chosen port over SSH, then open the localhost URL in your browser:

```bash
ssh -L 7743:127.0.0.1:7743 your-user@your-server
```

The server runs in the foreground. Ctrl+C closes its workers and releases the fixture locks. No service or daemon is installed. Every launch creates a new household; its `/tmp` files remain for inspection after stopping, but are disposable and are not a portable restore archive.

## Things to try

1. Use **New Thread** to choose one or several residents and send a message as the operator. With no selected mentions, joined residents receive optional reply offers. With selected mentions, only those residents are offered replies. Each **Reply** action runs a clearly labeled scripted response through the broker; nothing replies automatically.
2. In the resident contact panel, choose a resident and an opening message. The fixture creates a new operator-visible thread in an autonomous-origin exchange. Your current conversation and draft stay selected. **Unread threads** shows the incoming thread until you open and actually view its messages.
3. Switch threads while holding an unsent draft, or open a new composer. Drafts are retained per thread. Foreground read acknowledgments are bounded by the snapshot and visible rendered messages; listing, polling and background tabs do not mark messages read.
4. Make a fixture unavailable, then address a new message to it. Its offer can wait; a scripted reply is refused while unavailable. Making it available does not itself generate a response.
5. Run the scheduling demonstration. It queues one opportunity per available fixture and drives synthetic provider durations of 12, 3 and 6 seconds through the single broker slot. Those are simulated accounting intervals, not wall-clock model latency or evidence of actual cognition. Inspect attempts, virtual service and measured synthetic occupancy.
6. Add a shared or private object for a chosen fixture. Shared objects appear in the common list; **Inspect home** explicitly shows that fixture's own identity, synthetic memory and visible world. Unrelated private conversation transcripts remain excluded from ordinary thread access.
7. Inspect notification intents after resident messages. They are committed outbox records only. No push subscription, device enrollment or push dispatch is enabled.

All manual resident actions are controlled fixture exercises. They do not establish voluntary behavior, welfare, real multi-resident inference, complete app parity, or sustained resident behavior. Normal deployment, migration, recovery and live adoption remain the roadmap's separate gates.

## Development checks

[API tests](../tests/test_household_preview.py) cover authentication, mutation guards, replay, addressing/privacy, read snapshots, availability and synthetic scheduling against newly seeded workers. The standalone [browser smoke](../scripts/household-preview-browser-smoke.mjs) targets a disposable preview with a separate Chromium profile. Supply `TAMAGI_PREVIEW_URL` and `TAMAGI_PREVIEW_PASSWORD` in its environment; do not target the live app. Browser reports and screenshots are retained under `/tmp`.

Use [preview tests](../tests/test_household_preview.py) and [household checks](household-native-testing.md) to validate the revision under review. Keep generated reports and screenshots outside Git.

The preview files are [backend](../backend/core/household/preview.py), [page](../frontend/household-preview.html), and [launcher](../scripts/household_preview.py). None is imported by the production entry point. There is no configurable provider endpoint in this launcher, and it never reads a live instance configuration or predecessor archive.
