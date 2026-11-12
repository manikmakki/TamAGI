# Home platform: implementation and evidence

This page maps the source in this checkout to its components, tests and remaining
limitations. Start with [installation](running.md#install-a-new-home) for a new
standalone home or [native households](household-native.md) for shared operation.

[Accepted ADRs](adrs/README.md) record design intent. Their parent-child framing
concerns the builder's obligations: honest conditions, meaningful choices,
recoverable mistakes and room for independence. It does not prescribe a resident's
identity or prove subjective welfare.

Standalone homes have one primary resident and invitation-only companions. The
optional native path adds independent resident processes, shared conversations,
central inference accounting, unread threads, resident contact, creation and
individual recovery. The [multi-resident roadmap](multi-resident-roadmap.md)
distinguishes these features from coordinated household recovery/deployment,
companion promotion and PWA push that remain separate work. The service worker
currently caches assets; notification intents are not delivered device alerts.

## Interface consolidation and work ownership

The shared app shell implements Chat, Map, Journal, Inventory, Companions, Archive, and Settings. Growth explanations open in a dialog from the chat status display. Earlier entry routes remain compatible; `/home` selects the shell rather than a separate dashboard.

Accepted chat and operator-to-companion work belongs to the application. Durable admission identities, retained inputs/results, bounded progress history, deduplicated submission, Stop, and reconnection snapshots decouple work from its original browser/socket. The backend must stay running for work to finish. Restart and deployment interrupt execution; retained receipts and results support inspection, not automatic effect replay.

Inventory discovers regular files under the active workspace’s `workshop/`. Saved artifact versions are distinct from editable current files, and older registered artifacts retain links. HTML/SVG previews are static and isolated. Journal’s household notebook displays the newest of its displayed entries first.

See [navigation and accepted work](navigation-and-work.md) for API contracts, browser-fixture commands, queue/progress limits, and recovery behavior. The graph, connection settings, and rig editor use compatibility panels inside the shell.

## Delivered implementation

| Area | Current source | Evidence and limits |
| --- | --- | --- |
| Canonical records and execution | [store](../backend/core/runtime/store.py), [service](../backend/core/runtime/service.py) | [Runtime tests](../tests/test_platform_runtime.py) cover receipts, actor scope, interruption, rest, and persistence. [Graph persistence tests](../tests/test_world_graph_persistence.py) cover compatibility save failures, failed receipts and atomic JSON replacement. |
| Conversation ownership | [interactions](../backend/core/runtime/interactions.py), [chat turns](../backend/core/runtime/chat_turns.py) | [Interaction tests](../tests/test_interactions.py), [chat-turn tests](../tests/test_chat_turns.py), and application smoke cover admission, observer loss, quiet work, Stop, and restart boundaries. |
| Workshop and Inventory | [inventory](../backend/core/runtime/inventory.py) | [Inventory tests](../tests/test_inventory.py) cover discovery, saved versions, path/symlink rejection, and retained file descriptors. |
| Durable companions | [companions](../backend/core/runtime/companions.py) | Runtime/release tests cover attribution, bounded invitations, workflow dependencies, and dispatch restrictions. Companions have no independent population scheduler. |
| Runtime tools and evaluation | [adaptation](../backend/core/runtime/adaptation.py), [evaluation](../backend/core/runtime/evaluation.py), [live evaluator](../backend/core/runtime/live_evaluation.py) | [Runtime-change tests](../tests/test_runtime_change.py) and [release tests](../tests/test_platform_release.py) exercise worker trials, activation/revert, and comparisons. Default probes do not establish arbitrary functional usefulness. |
| Exact releases and recovery | [deployment](../backend/core/runtime/deployment.py), [supervisor](../scripts/tamagi-supervisor.py), [backup](../backend/core/runtime/backup.py) | Release/runtime tests and [application tests](../tests/test_platform_api.py) exercise encrypted transfer, safe restore, copied-state boot checks, and real supervisor recovery. External effects cannot be undone by code rollback. |
| Growth | [growth](../backend/core/runtime/growth.py), [ADR-0015](adrs/0015-growth-through-sustained-behavior.md) | [Growth tests](../tests/test_growth.py) cover deterministic recognition, source attribution, replay, rest, and portable restoration. Points do not measure welfare. |
| Context and wake recovery | [context views](../backend/core/runtime/context_view.py), [world adapter](../backend/core/runtime/world.py) | [Context tests](../tests/test_runtime_context.py), [long-horizon tests](../tests/test_long_horizon.py), and [evaluation lifecycle tests](../tests/test_evaluation_lifecycle.py) cover bounds, budgets, errors, and settled evaluation waits. |
| Fresh provisioning and pilot | [runtime CLI](../backend/runtime_cli.py) | Provisioning is separate from onboarding and the seven-real-day pilot. Missing observations remain untested capabilities. |

Tests are mechanical evidence with bounded coverage, not proof that every error path is correct or that sustained autonomous improvement has occurred.

## Architecture and normal operation

SQLite/WAL stores canonical actors, conversations, memories, world objects, pursuits, revisions, receipts, experiments, and deployment records. Workspace artifacts remain referenced files. Markdown identity/task files and JSON graph/history are service-managed views or compatibility exports in platform mode. Legacy non-platform storage follows different contracts.

Optional ChromaDB/Elasticsearch indexes supplement source memory; they are not required for the default lexical recall. See [integrations and memory](integrations.md) for enabling, rebuilding, and reconnecting them.

Every companion has attributed history and memories. Normal context/retrieval respects actor boundaries and explicit sharing. Operator inspection can access actors; host root can bypass application-level privacy. These are protocol boundaries, not confidentiality against the OS administrator.

Inference and tool dispatch use trusted execution context and durable accounting. [Current scheduling and allowances](running.md#scheduling-and-model-allowances) come from RuntimeConfig and ADRs 0016–0017. Background work is uncapped per day by default, individual activations remain bounded, and evaluations have a separate rolling allowance. User input takes priority. Choosing rest is valid.

In `unrestricted_host`, `exec` runs non-login Bash with `pipefail`; guarded execution uses direct argv and command policies. Timeout/interruption stops process groups, but earlier effects can remain. An unfinished dispatched effect is `outcome_unknown`, not safe to repeat automatically. See [shell tests](../tests/test_exec_shell.py).

## Autonomous context and recovery

Prompts use bounded previews while receipts retain original results. `inspect_receipt` pages output, errors, data, or arguments without rerunning the operation. Complete messages/artifacts remain source records even when omitted from a model view.

`runtime.model_context_tokens` defaults to 131072. A compatible server tokenizer counts the template and tools; the effective window is the smaller configured/server-reported capacity minus requested reply tokens and a 1024-token margin. A secondary 1 MiB input ceiling applies. Unsupported/unavailable tokenizers use a conservative byte view of at most 65536 bytes with five-minute backoff. Tokenization generates no completion and does not consume a model-call allowance. These limits do not guarantee that every provider accepts a request.

Tool previews allow 32 KiB by default; receipt pages allow up to 65536 characters. A fresh home disables early character-based history compression and allows up to 1000 recent messages before measured request bounds. The default reply allowance remains `llm.max_tokens: 2048`; an installation may choose a different value. Installations may choose a different reply allowance.

Activity records distinguish chosen rest/wait from provider/context failure and retry backoff. All terminal evaluation outcomes release `evaluation_completed` waits. Recovery marks orphaned queued/running jobs interrupted; due checks reconcile missed settled-job wake events. Failed evaluation is still failed evidence, including `BudgetExhausted`; waking does not bypass allowances or approve a tool.

## Growth and stage progression

The primary resident receives Growth for sustained reciprocal conversation, a meaningful pursuit return on a later occasion, and a successful correction explicitly linked to a previous failed tool operation. Three completed exchanges award one engagement point; eight award three cumulatively. Continuity and adaptation add one point per qualifying occasion. Patterns can add, while replay does not repeat awards.

Level is `1 + floor(sqrt(Growth / 5))`, with thresholds 0, 5, 20, and 45 for levels 1–4. Age, tool-call volume, and elapsed inactivity do not grant or deduct Growth. Stage naming adds no inference or permissions. Operator-to-companion conversations do not award the absent primary resident engagement.

The avatar shows Growth, progress, and age separately. **Why this level?** opens its recognition dialog. `GET /api/runtime/growth` supplies authenticated inspection. Recognition records and stage history are included in portable recovery.

[ADR-0015’s example](adrs/0015-growth-through-sustained-behavior.md#example-research-carried-into-conversation) explains recognition for research carried into conversation. It illustrates the rules rather than reporting a live resident’s level or completing behavioral observation.

## Installation, preservation, and recovery

Use the [running guide](running.md) for current instructions. Important distinctions:

- A fresh instance is a separate identity. Preserve a predecessor first; use restoration to preserve an existing home’s continuity.
- The canonical checkout, selected application release, and lived-state root are separate locations.
- The supplied root supervisor template assumes `/opt/TamAGI` and port 7742; the running guide provisions a guarded home on port 7741.
- The supplied Compose file uses legacy configuration and does not provision the home platform.
- Encrypted application backups do not include the OS, model weights, credentials, or unrelated host services. Keep a downloaded copy elsewhere.

See [workspace conventions](workspace-layout.md) for active paths and saved artifacts, and [integrations](integrations.md) for reconnecting credentials/indexes after recovery.

## Development contracts

`ExecutionContext` supplies trusted actor/authority/budget provenance. Dispatch checks eligibility and records attempts/results before narrative interpretation. A malformed narrative cannot erase a committed operation.

Generated tools use an argv subprocess contract: one JSON arguments object on stdin, one JSON result with boolean `success`, optional text `output`/`error`, and object `data` on stdout. Timeout is at most 120 seconds and each output stream is bounded to 1 MiB. Source files explicitly named in the command are snapshotted; sibling modules, environments, and external dependencies are not automatically bundled.

A tool with `world_effects: true` can return up to ten validated `data.world_operations` applied in one SQLite transaction. This supported simulated-operation contract does not make arbitrary shell, MCP, or compatibility graph writes transactional. Trials isolate application state/workspaces; root-authorized host effects can escape that temporary state.

`runtime_change` supports inspection, proposal, worker test, evaluation, activation, and revert. The three built-in behavioral probes cover rest, a saved next step, and an invalid location; a tool proposal also receives a worker-protocol probe. Promotion requires retained qualifying comparison evidence. Task-specific claims need task-specific runner-owned scenarios. `core_change` separately prepares, builds, and activates exact source releases.

## Known limitations

The [issue #5 fix](development.md#graph-persistence) logs graph save and edge errors, reports failed saves through the tool result and platform receipt, and replaces JSON from a synced temporary file. The in-memory mutation and a platform database commit can precede an export failure, so failure is not a rollback guarantee. Inspect current state before retrying. This does not make the database and JSON export a single transaction.

Model follow-through, unsupported provider behavior, and sustained improvement remain empirical questions. Passing contract tests cannot establish arbitrary capability, emotional state, or welfare.

The current MCP SSE connector does not forward configured environment values as authentication headers. Authenticated remote integrations need an appropriate stdio bridge; see [MCP setup](integrations.md#mcp-tools). Native Streamable HTTP is also outside the current adapter's transport choices.

## Operational qualification

Run from an installed checkout:

```bash
.venv/bin/python -m unittest discover -s tests -q
```

The suite includes disposable application processes, authenticated REST/WebSocket behavior, portable transfer, execution cancellation/persistence, generated tools, criteria comparisons, and supervisor transitions. [Navigation’s browser fixture](navigation-and-work.md#mechanical-qualification) supplies repeatable Chromium checks using a deterministic provider.

See [development and testing](development.md) for repository checks and
[household checks](household-native-testing.md) for guarded worker/browser runs.
Keep revision-specific reports in ignored local output or CI artifacts, and rerun
checks for the source being evaluated.

The pilot recorder is separate:

```bash
.venv/bin/python -m backend.runtime_cli pilot status --config /absolute/instance/config.yaml
.venv/bin/python -m backend.runtime_cli pilot finalize --config /absolute/instance/config.yaml
```

Finalization requires seven actual days after the pilot begins, plus retained observations of choice, rest, companion conversation, continued accepted work after restart, tool failure, deployment, and controlled rollback. Missing observations produce `incomplete_observations`. Rest or nonparticipation is not resident failure. Run these commands only when intentionally inspecting/finalizing an installation; routine development checks do not finalize a live installation's pilot.
