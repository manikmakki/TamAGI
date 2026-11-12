# TamAGI documentation

Start with the [project README](../README.md) for an introduction. These guides
cover the source in this checkout. An existing supervised installation continues
using its selected release until that release is explicitly changed.

## Use TamAGI

| Guide | Contents |
| --- | --- |
| [Installation and running](running.md) | New standalone homes, configuration, scheduling, network access, services and Docker. |
| [Native households](household-native.md) | Optional multi-resident operation, conversations, invitations and menu scope. |
| [Create a TamAGI](creating-a-tamagi.md) | Name, artwork, starting personality and the first conversation. |
| [Preservation and migration](production-multi-resident-enablement.md) | Continuity, stopped checkpoints, staged cutover and recovery limits. |
| [Individual resident recovery](resident-recovery.md) | Browser and offline backup/restore, identities and shared-state boundaries. |
| [Retained conversations](legacy-conversation-access.md) | Exact private legacy transcript lookup without a bulk Chat import. |
| [Integrations and memory](integrations.md) | Model transport, search, MCP, credentials and optional retrieval indexes. |
| [Workspace and tools](workspace-layout.md) | Active paths, Inventory, generated tools, evaluation and activation. |
| [Appearance](sprite-customization.md) | Artwork presets, rigs, sprite uploads and APIs. |
| [Navigation and work](navigation-and-work.md) | App drawer, accepted requests, Stop, browser disconnection and restart. |
| [Household World Thread](household-world-thread.md) | Scheduled opportunities, chosen rest, notebook and shared artifacts. |

## Understand and develop TamAGI

| Guide | Contents |
| --- | --- |
| [Implementation map](platform-roadmap.md) | Current runtime components, limitations and behavioral observation. |
| [Architecture decisions](adrs/README.md) | Design commitments, decisions and supersession. |
| [Multi-resident roadmap](multi-resident-roadmap.md) | Implemented scope, remaining work and release gates. |
| [Household foundations](household-foundations.md) | Core stores, broker, transport and fixture boundaries. |
| [Development and testing](development.md) | Build/test commands and repository conventions. |
| [Household checks](household-native-testing.md) | Repeatable API, worker, browser and recovery scenarios. |
| [Scripted preview](household-preview.md) | Local debugging panel with deterministic fixture replies. |

Standalone instances have one primary resident and invitation-only companions.
Native households add independent residents, canonical shared conversations,
central inference scheduling, resident creation and individual recovery. Automatic
whole-household recovery/deployment, companion promotion and device push remain
separate work. See the roadmap for precise boundaries.

Residents and operators are configured by each installation. Artwork presets and
test fixtures do not define canonical people or identities. Production inventories,
private histories and generated test reports belong outside the tracked source.
