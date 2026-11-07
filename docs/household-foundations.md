# Household foundations: isolated implementation

Historical foundation checkpoint: experimental ADR-0019 implementation started on 2026-11-06 on `codex/adr-0019-household`, based on `codex/dependable-home` at `244dd695f7efbebee163da5df09f98412f4d96b4`. The development worktree is `/tmp/tamagi-adr-0019`. At that checkpoint household mode was not connected to application startup. The later [native integration candidate](household-native.md) supplies the current opt-in application path; no existing live installation has been enrolled.

This delivers a runnable foundation across parts of roadmap Phases B–D. It does **not** complete any of those phases. The [roadmap](multi-resident-roadmap.md) retains their full parity, integration, recovery, and adoption gates. Ember and Echo are not fixtures; all residents are newly seeded with distinguishable synthetic records.

The [qualification record](qualification/0019-foundations-2026-11-06.md) retains the clean-revision result: **232 passing tests, including 83 new household tests**, plus the successful three-process scenario. Real model calls and push dispatches were zero.

A later [interactive development preview](household-preview.md) exposes these fixtures in a separate local browser panel with scripted replies, new/unread threads, world commands and scheduling inspection. It remains separate from production app integration and starts no real model or notification traffic.

## Implemented boundary

| Component | Implemented behavior | Evidence |
| --- | --- | --- |
| [Canonical store](../backend/core/household/store.py) | Separate household database format, nested transactions, optimistic revisions, events, atomic command outcomes, replay and conflict detection. Foreign databases are inspected read-only before rejection. | [Store tests](../tests/test_household_store.py) |
| [Fixture registry](../backend/core/household/registry.py) | Explicit household/release/resident/installation bindings; new marked `/tmp` roots only; stable configured default; no import or enrollment of existing homes. Read-only identity checks and symlink rejection precede mutable hydration. | [Worker tests](../tests/test_household_workers.py) |
| [Worker and IPC](../backend/core/household/worker.py) | Real subprocesses compose existing RuntimeStore/RuntimeService independently, with one root lock, private identity/memory, process configuration and working directory. Versioned bounded stdio frames bind requests and responses to the admitted worker. A poisoned or cancelled channel cannot admit another request. | Worker tests; [coordinator tests](../tests/test_household_coordinator.py) |
| [Inference broker](../backend/core/household/broker.py) | Durable attempts and aggregate lineage; one provider slot; measured-time shares and a separate virtual eligibility floor; bounded first-response priority; rest/join rules; 20-call segments, five coalesced continuations, 128-call autonomous roots, 16-call helpers, sponsorship, evaluation limits, shared backoff. Stop drains running work; unknown occupancy holds admission for recorded reconciliation. | [Broker tests](../tests/test_household_broker.py) |
| [Provider driver](../backend/core/household/driver.py) | Injected deterministic-provider seam; application-owned generation independent of waiters; retained success/failure outcomes; stopped or lost workers cannot use late results. Ambiguous transport or failed settlement produces an inspectable hold. | [Driver tests](../tests/test_household_driver.py) |
| [Conversations](../backend/core/household/conversations.py) | Canonical private/group messages, resolved stable mentions, chosen membership with full-history notice, reciprocal optional reply offers and response links, existing-lineage resident contact, atomic notification intents, durable snapshot-bounded operator read watermarks. | [Conversation tests](../tests/test_household_conversations.py) |
| [World](../backend/core/household/world.py) | Canonical topology commands, owner/visibility/revision checks, location-cycle rejection, immutable selected artifact references, explicit namespaced import mappings. | [World tests](../tests/test_household_world.py) |
| [Sandbox coordinator](../backend/core/household/coordinator.py) | Explicit resident routing, one fixture execution lease, manual autonomous and operator admission, worker-loss revocation, actor-scoped outcome recovery, and composition of the cores above. | Coordinator tests; [three-process scenario](../tests/test_household_sandbox.py) |

Model arguments never select author, sponsor, root allowance, release, or private state root. The Python service methods accept trusted coordinator state; they are not authenticated public APIs. Production authentication, worker-to-coordinator generation IPC and tool admission are still required. Separate worker processes and application checks do not provide confidentiality against the host administrator.

## Run without the live instance

From the isolated worktree, use the existing interpreter to run the complete suite with a sanitized environment and inherited Python socket/DNS guard:

```bash
cd /tmp/tamagi-adr-0019
/opt/TamAGI/.venv/bin/python scripts/qualify_household.py \
  --output /tmp/tamagi-0019-qualification-new-run
```

The output directory must be new and outside the source and live checkout. The runner starts discovery from a fresh `/tmp` directory, excludes live configuration and credentials, and permits only loopback TCP ports actually bound by test processes. The model service's default port is always denied. It retains the source revision/diff fingerprint, process guard activation ledger, denials, stdout/stderr, timeout status, and before/after source comparison. Intentional guard-denial self-tests use separate asserted ledgers. This is Python network instrumentation, not OS confinement for arbitrary native commands; these reviewed fixtures do not launch external network clients.

Run a retained synthetic household scenario separately:

```bash
/opt/TamAGI/.venv/bin/python scripts/household_sandbox.py \
  --output /tmp/tamagi-0019-scenario-new-run.json
```

It creates its own marked fixture root under `/tmp`, starts three private runtime subprocesses, admits a shared operator exchange, drives unequal synthetic generation durations through the broker, preserves attributed replies, connects namespaced homes, replays a world command, and creates a resident-origin thread. A message arriving after a transcript snapshot remains unread after acknowledgment of that earlier snapshot. Contact retains the initiating autonomous exchange and produces notification intents without dispatching push. The JSON report names the retained fixture root and source fingerprint. No existing root/configuration is accepted, no model client is instantiated, and no resident schedule or deployment controller is started.

To inspect the code, use this worktree. Checking out the experimental branch in the live canonical checkout is unnecessary. The active `codex/dependable-home` checkout and its selected release remain unchanged.

## Remaining qualification

The fixture worker currently supports identity/state inspection only. It does not start Agent, Interactions, World Thread, MCP, a completion client, or deployment. Its private RuntimeService still seeds legacy local world records; those are fixture records, **not** enrolled shared-world projections. A successful fixture startup is not one-resident behavioral parity.

Next, wrap the existing one-resident accepted-work lifecycle with an injected offline model transport and compare stable IDs and behavior for chat/quiet work, companions, rest/wake, memory, identity, artifacts, Growth, Stop and reconnect. Then route all supported completion paths through authenticated broker IPC before enabling multiple cognizing workers. This includes streaming, retries, reflection, evaluations, specialist endpoint clients, and the provider `ping()` fallback. Unsupported paths must fail closed in household mode. The current standalone entry point retains its original behavior.

The production exchange controller still needs offer rotation, snapshot admission, continuation/consultation parking and cycle detection, wake policy, and current-context resume. Existing admitted broker executions retain their scheduling position between generations; the runtime adapter must explicitly report park/rest/unavailability rather than treating a pending offer as runnable cognition. Stop/late-result decisions must also govern actual tool leases, not only fixture results.

Legacy world/history writers must delegate to canonical services; artifact references need verified snapshot-byte publication; source-reference access must be validated at the sharing boundary; Growth must recognize each resident's own reciprocal participation without replay awards. Canonical shared data is currently exercised directly through internal services, without a production HTTP/browser adapter.

PWA enrollment/VAPID, authenticated subscription storage, a delivery sender with retry/coalescing/quiet-hour preferences, service-worker notification handlers, resident/thread selection, **Unread threads (N)**, and **New Thread** UI remain pending. A `notification_intent` is an atomic outbox prerequisite, not a sent notification. No push dependencies or devices were enrolled in this milestone.

Coordinated household backup/restore, migration, deployment, rollback and real-device push checks remain Phase E work. No live enrollment, adoption, Echo recovery, or burn-in changes are part of this branch. See the [retained qualification record](qualification/0019-foundations-2026-11-06.md) for actual evidence and limits.
