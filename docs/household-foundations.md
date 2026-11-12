# Household foundations

The household core separates private resident state from canonical shared state.
The [native application](household-native.md) integrates these services with real
resident Agents; the [scripted preview](household-preview.md) exposes lower-level
fixtures for debugging. Neither a successful fixture nor an accepted
[ADR](adrs/0019-distinct-residents-and-a-shared-world.md) establishes sustained
behavior or a portable production migration.

## Components

| Component | Responsibility | Tests |
| --- | --- | --- |
| [Store](../backend/core/household/store.py) | Separate shared SQLite database, transactions, optimistic revisions, command outcomes and replay. | [Store](../tests/test_household_store.py) |
| [Fixture registry](../backend/core/household/registry.py) | Explicit household/resident/installation/release bindings in newly seeded fixture roots. | [Workers](../tests/test_household_workers.py) |
| [Native registry](../backend/core/household/native_registry.py) | Private application homes and validated enrollment bindings. | [Native workers](../tests/test_household_native_worker.py) |
| [Fixture worker](../backend/core/household/worker.py) | Bounded versioned stdio frames, private runtime state and one root lock per worker. | [Coordinator](../tests/test_household_coordinator.py) |
| [Broker](../backend/core/household/broker.py) | One provider slot, durable attempts, shared lineage and limits, measured service fairness, backoff and uncertain-occupancy holds. | [Broker](../tests/test_household_broker.py) |
| [Driver](../backend/core/household/driver.py) | Provider ownership independent of waiters, retained outcomes and Stop/drain behavior. | [Driver](../tests/test_household_driver.py) |
| [Conversations](../backend/core/household/conversations.py) | Stable messages/mentions, consented membership, reply offers, unread snapshots and notification intents. | [Conversations](../tests/test_household_conversations.py) |
| [World](../backend/core/household/world.py) | Topology, ownership/visibility, revisions and namespaced artifact imports. | [World](../tests/test_household_world.py) |
| [Sandbox coordinator](../backend/core/household/coordinator.py) | Synthetic execution routing, worker-loss revocation and actor-scoped outcome recovery. | [Sandbox](../tests/test_household_sandbox.py) |

The fixture worker supports inspection and synthetic scenarios. Native workers
run the Agent, tools and receipt loop through authenticated coordinator IPC.
Model arguments never confer actor, sponsor, root allowance, release or private
state authority. Worker separation is a state boundary, not an OS sandbox against
the host administrator.

## Repeatable scenarios

```bash
.venv/bin/python scripts/household_sandbox.py   --output /tmp/tamagi-household-scenario.json
```

The scenario creates its own fixture roots, starts private subprocesses, exercises
attributed replies and synthetic provider durations, connects namespaced homes,
replays a world command and creates a resident-origin thread. Its provider is
injected; notification intents do not dispatch push.

For the native worker and browser checks, see [household testing](household-native-testing.md).
The [roadmap](multi-resident-roadmap.md) describes recovery, deployment and device
work that is not supplied by these core fixtures.
