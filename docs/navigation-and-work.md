# Navigation and accepted work

Chat remains the default view. The app drawer opens Map, Journal, Inventory, Companions, Archive, and Settings without unloading chat or its draft, attachment selection, and scroll position. Addressable views support browser Back/Forward and direct links. `/home`, `/self-model`, `/rig-editor`, and earlier HTML links remain compatible entry points. Growth explanations open from the chat status display.

Journal contains current rest/wake information, Mind/World Thread entries, and the attributed household notebook. Archive keeps Memories and Conversations distinct. Map retains the existing graph and contextual pursuits. Settings combines operator recovery/deployment controls with connection settings and avatar editing. The established graph, connection, and rig editors use isolated visible panel adapters inside the shell; none of these panels owns chat execution. Existing avatar assets and rig data are preserved.

Inventory discovers regular files in the active resident's `workspace/workshop/`, without requiring artifact registration. Current files and saved versions are labeled separately; earlier registered artifacts outside the workshop retain download links. Preview is read-only. HTML and SVG have static, sandboxed previews with scripts and network access disabled; other unsupported or oversized formats are downloads. Inventory is not a general file explorer or editor.

## Application-owned requests

The platform owns accepted chat and operator-to-companion requests. Input, conversation identity, and submission identity are saved before acknowledgment. Immutable input is stored separately from mutable progress; accepted images use hashed files under `workspace/.attachments/` so progress revisions do not duplicate image bytes. One worker dispatches requests through the existing execution service, with eight queued requests at most. Chat still has twenty calls per segment and up to five explicitly chosen quiet continuations; the drawer grants no additional resources.

Closing a view, socket, or browser detaches observation. Work can finish while the backend remains running, and its result is retained. Request-specific Stop interrupts work and preserves completed receipts and uncertain outcomes. Actual new input still preempts quiet and autonomous work. Read-only memory searches and reconnection do not enter an interrupting user execution.

Submission identifiers deduplicate retries with the same input, including a lost acknowledgment. A reused identifier with different content is rejected. Each conversation receives its own results even when another thread is selected. User messages are persisted on admission; queued future messages are excluded from an earlier request's model context, and responses are placed after the input they answer.

The browser subscribes through the authenticated WebSocket and recovers through HTTP snapshots when disconnected. Progress uses stable event cursors, bounded observer notifications, and up to 256 retained progress events per request. Expired or out-of-range cursors resynchronize from retained state. Canonical tool receipts and final responses remain separate from this bounded UI progress history. A slow observer cannot block tool execution.

Supported API contracts:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/chat/requests` | Admit input with a stable `request_id`; returns 202 and `work_id`/conversation identity. |
| `GET /api/chat/requests` | Recent request states for recovery and activity indicators. |
| `GET /api/chat/requests/{work_id}?after=<cursor>` | Consistent request snapshot, retained events, cursor, and resynchronization indication. |
| `POST /api/chat/requests/{work_id}/stop` | Stop a specific request. |
| `POST /api/chat/requests/{work_id}/approval` | Resolve that request's unexpired guarded-tool approval. |
| `POST /api/chat` | Compatible wait-for-result adapter over the same admission service. |
| `POST /api/companions/{actor_id}/consult` | Compatible invocation; `background: true` returns admission immediately. |
| `GET /api/inventory` | Current workshop creations and separately retained snapshots. |

WebSocket clients may send a normal chat request, `subscribe` with `work_id`/`after`, a request-specific `stop`, or an approval response. Delivery uses the same retained events as HTTP recovery. Approval deadlines are server-owned and do not restart with a view or connection.

Non-platform installations retain the legacy chat adapter; durable browser-independent execution requires an enabled home platform.

## Presence and process recovery

A conversational visit uses one deadline: the later of accepted input or completion of admitted chat work, plus the configured resume interval (default fifteen minutes). Active and queued work prevent premature visit closure. Browsing and connection heartbeats do not extend that deadline. Departure/relational processing is deduplicated by the conversation's last-message watermark and does not add a second decompression interval. This operational boundary is independent of Growth's two-hour conversation occasions.

Backend restart or deployment is an interruption boundary. Pending requests are surfaced as interrupted, queued quiet work is discarded, and dispatched unfinished receipts remain `outcome_unknown`. Reconnection never replays the operation. Requests, inputs, receipts, results, and visit records are included in canonical backups; restoration does not resume old work automatically.

## Mechanical qualification

Run the normal suite with `.venv/bin/python -m unittest discover -s tests -q`. `test_interactions.py` covers admission/deduplication, observer loss/backpressure, bounded queues, replay/resynchronization, Stop and unknown effects, restart recovery, encrypted restoration, presence deadlines, and approval expiry. `test_inventory.py` covers unregistered discovery, saved versions, traversal/symlink rejection, and retaining the selected descriptor when a path is replaced. The existing application smoke still exercises authenticated REST/WebSocket chat, quiet preemption, Growth, and encrypted transfer.

For a real browser check against a fresh, disposable instance:

```bash
PYTHONPATH=. .venv/bin/python scripts/navigation-fixture.py --port 18742
# In another terminal, with Chromium installed:
TAMAGI_FIXTURE_URL=http://127.0.0.1:18742 \
TAMAGI_CHROME=/path/to/chromium \
node scripts/navigation-browser-smoke.mjs
```

The fixture uses a deterministic provider and writes only temporary test state. It never calls the live model or resident. Node 22 or later supplies the browser test's WebSocket implementation. Screenshots and the JSON report default to the ignored `.platform/navigation-browser/` directory. Start a fresh fixture for each browser run.

The browser scenarios exercise all drawer destinations and panel adapters, keyboard dismissal, draft/attachment preservation, browsing without inference or input events, isolated previews, late-result attribution, quiet-work interruption, page closure/reload, lost acknowledgments, queued-message ordering, mobile layout, and Back/Forward. Mechanical qualification does not establish subjective welfare or complete the seven-day behavioral pilot.

## Bundled browser dependencies

Marked 15.0.7 and D3 7.9.0 are served locally for chat rendering and the existing graph. Their license notices are retained in `frontend/assets/licenses/`. Optional web fonts fall back to local monospace fonts when unavailable. No external JavaScript delivery is required for the shell or graph.
