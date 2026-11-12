# Testing native households

Use disposable roots and deterministic providers to validate application mechanics.
The [native guide](household-native.md) describes interactive operation with an
explicit model configuration; the checks below do not need a production resident
or model endpoint. Keep results outside the tracked source.

## Backend and worker checks

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m unittest discover -s tests -p 'test_household_*.py' -q
```

The suite exercises actual private worker processes as well as unit-level stores,
transport and services. See [foundations](household-foundations.md) for the core
component map and [development](development.md) for lint and repository checks.

For a guarded full run, use an isolated checkout outside `/opt/TamAGI` and a new
output directory outside that checkout:

```bash
.venv/bin/python scripts/qualify_household.py   --source "$PWD" --output /tmp/tamagi-household-suite-new
```

The runner strips live configuration/credentials, uses disposable working state,
records the source fingerprint before/after, and propagates a Python socket/DNS
guard to child processes. Only fixture-bound loopback ports are allowed; the
default model service port is denied. This is Python instrumentation rather than
OS confinement of arbitrary native commands. Results include stdout/stderr,
guard activation/denial ledgers, timeout status and a JSON report.

## Browser checks

Install Node 22 or later and a Playwright Chromium executable in the environment.
Set `TAMAGI_CHROME` to that executable rather than relying on the local cache fallback. From
an isolated checkout outside `/opt/TamAGI`, run:

```bash
.venv/bin/python scripts/check_household_browser.py   --output /tmp/tamagi-household-browser-new
```

This starts the disposable native fixture on port 18846 and Chromium debugging on
18847. Use unoccupied ports. The runner removes its temporary credential and
retains screenshots, the fixture log, reports and guard ledgers in the output
folder. It uses an injected provider, not a real model. Browser fixtures are in
[tests/browser](../tests/browser).

The scenarios cover resident selection/drafts, new and unread threads, images,
reasoning sanitization, invitation consent, notebook attribution, World Thread,
shared artifact bytes, Inventory scope, Settings recovery and creation retries.
For standalone navigation checks, see [navigation](navigation-and-work.md#mechanical-qualification).

## Regression checklist

| Area | Check |
| --- | --- |
| Identity and routing | Every private view, chat request, attachment, receipt and tool uses the selected/admitted resident, including after another selection. |
| Conversation membership | Invitations retain the intended thread, hide its transcript before consent, and grant access only after an explicit accepted decision. |
| Peer progress | Replies share one exchange; ordinary rest permits peer work without changing its World Thread deadline. Explicit waits/backoff and budgets still apply. |
| History and unread | Read watermarks acknowledge visible snapshots only. Reconnection does not invent input, repeat effects or expose private predecessor history. |
| World and artifacts | One clock rotates eligible residents. Notebook authorship and public artifact publication/import retain trusted attribution and real bytes. |
| Creation | New residents start independently; busy maintenance is rejected and retries recover the same creation result. |
| Recovery | Wrong credentials/tampering fail before commit; restore preserves resident identity, rotates installation authority and reports private/shared scope. |
| Shutdown | Queued evaluations settle, pending work becomes inspectable and no fixture process or worker lock is left running. |

Useful focused modules include
[chat rest](../tests/test_household_chat_rest.py),
[invitations](../tests/test_household_native_invitations.py),
[creation](../tests/test_household_creation_app.py),
[browser recovery](../tests/test_household_browser_recovery.py), and
[individual recovery](../tests/test_resident_recovery.py).

## Limits

Scripted scenarios establish process, protocol and state behavior. Real provider
transport, device push, portable coordinated household recovery, automatic
household deployment and sustained behavior require their own evidence. A manual
UI exercise or one successful startup does not replace those checks.
