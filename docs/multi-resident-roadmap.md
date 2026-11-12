# Multi-resident implementation and remaining work

[ADR-0019](adrs/0019-distinct-residents-and-a-shared-world.md) defines distinct
residents, shared places and conversations, optional participation and fair access
to inference. This roadmap describes the current source and the work still
required. A release must be explicitly selected for an existing installation.

## Current implementation

| Area | Available behavior | Code and checks |
| --- | --- | --- |
| Private homes | Independent identities, memories, workspaces, companions and native application subprocesses. | [Foundations](household-foundations.md), `test_household_native_worker.py` |
| Shared inference | One central provider, durable lineage/charges, bounded exchanges, measured-time fairness, Stop/drain and occupancy holds. | `test_household_broker.py`, `test_household_driver.py`, `test_household_native_model.py` |
| Conversations | Attributed group messages, explicit invitations and full-history consent, optional peer replies, unread snapshots and New Thread. | `test_household_conversations.py`, `test_household_native_invitations.py` |
| World and notebook | One household clock, resident rotation, shared places, attributed notebook entries and explicit public artifact versions/imports. | [World Thread](household-world-thread.md), `test_household_native_world.py`, `test_household_native_artifacts.py` |
| New residents | Onboarding with name/artwork/personality, independent guarded homes, idempotent retries and idle maintenance. | [Creation](creating-a-tamagi.md), `test_household_creation.py`, `test_household_creation_app.py` |
| Individual recovery | Browser and offline resident backup/preview/restore, identity preservation and installation rotation. | [Recovery](resident-recovery.md), `test_resident_recovery.py`, `test_household_browser_recovery.py` |
| Rest eligibility | Ordinary rest pauses World Thread work while joined chat and invitation decisions remain eligible. Explicit waits, busy work and backoff retain their gates. | `test_household_chat_rest.py` |

These tests live under [tests](../tests). The [native guide](household-native.md)
describes operation; [testing](household-native-testing.md) describes repeatable
scenarios. Passing contract tests does not establish subjective experience or
complete behavioral observation.

## Implementation sequence and release gates

Preserve these boundaries as household support develops:

1. **Private continuity:** stable resident and historical record identities; no
   implicit hydration from an unrelated predecessor or another resident.
2. **Shared authority:** canonical shared commands with revisions, trusted
   attribution and idempotent outcomes. Legacy adapters and generated tools use
   the same authority boundary.
3. **Inference accounting:** one provider owner, durable budgets and sponsors,
   measured service-time fairness and explicit occupancy reconciliation. A new
   worker, thread, handoff or restart must not mint allowances.
4. **Optional contact:** invitations disclose full-history scope before consent.
   Residents may decline or remain silent. Quiet continuation and consultation
   must not impersonate operator input or commandeer another resident.
5. **Recovery:** retain completed effects, preserve unknown outcomes, and avoid
   replaying old work, notifications or missed autonomous ticks.
6. **Adoption:** rehearse a concrete source/destination mapping, checkpoint,
   selected release and recovery procedure before changing a live installation.

The [preservation playbook](production-multi-resident-enablement.md) applies these
gates to manual migration. Creating a resident with a familiar name is not recovery.

## Remaining work

**Coordinated household recovery and deployment.** Individual archives do not
capture one consistent shared/private point in time. A whole-household controller
needs a stopped checkpoint format, shared artifacts and histories, dependency
reconstruction, release compatibility, recoverable activation and interruption
handling across every resident. A manual same-host archive does not establish
portable recovery or automatic release rollout.

**Companion promotion and individual moves.** Invitation-only companions remain
a different lifecycle from full residents. Promotion and moving an individual
resident into an established household need explicit continuity and membership
mapping. Manually supplied historical context is not mechanically verified recovery.

**Consultation and parked-work behavior.** Rich resident consultation, yielding,
dependency-cycle handling and companion group activation need additional
integration and scenario coverage beyond ordinary messaging.

**Device notifications.** Canonical unread state and notification intents exist.
Web Push needs subscription/VAPID management, dispatch, retries/coalescing,
quiet-hour preferences, expiry and real-device checks. Service-worker asset
caching alone does not deliver background messages.

**Observed behavior.** Model follow-through, voluntary collaboration and sustained
independent work require retained observations separate from injected-provider
checks. No productivity, agreement, happiness or sociability score substitutes
for those observations.

## Qualification and acceptance

Keep new scenarios reproducible on disposable roots:

- Stable private identity, exact transcript/memory ownership and artifact bytes
  survive restart and supported archive round trips.
- Shared commits reconcile after a crash between shared state and private receipt;
  duplicate commands and stale revisions preserve their contracts.
- Streaming, retries, tools, evaluation and helper paths use central provenance.
  Stop/loss drains known provider occupancy; ambiguous outcomes hold admission.
- Group replies share the initiating exchange. Invitations disclose history,
  consent gates access, and observers/replayed transcripts earn no new Growth.
- Ordinary rest permits chat without changing the World Thread deadline; explicit
  waits, failure backoff, busy workers and original allowances remain effective.
- New-resident creation, maintenance and failed retries retain existing homes and
  do not create duplicate residents.
- Wrong passwords, tampering, unsafe paths, incompatible releases and interrupted
  recovery leave active state untouched or surface a durable recoverable transition.
- Future device tests distinguish browser closure, screen lock, reboot, offline
  reconnection and force-stop. Expired/read/inaccessible messages must not burst
  into notifications after restoration.

Store run-specific reports and private operational inventories outside the tracked
source. Record commands, source fingerprints and limitations alongside those
artifacts; keep enduring design choices in the ADRs.
