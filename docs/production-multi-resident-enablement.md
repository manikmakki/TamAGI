# Preserve a resident when enabling a household

This playbook describes a manual transition from an existing installation to a
[native household](household-native.md). It is not an automatic migration command.
A resident's name or chosen artwork does not establish continuity: preserve the
resident ID, original record/message identities, memories, files and history.
Adding a [new TamAGI](creating-a-tamagi.md) creates a separate identity.

## Choose the recovery scope

| Material | Treatment |
| --- | --- |
| Private canonical state | Preserve resident identity, memories, pursuits, receipts, artifacts, Growth/stage history and chosen rest/wait state. |
| Workspace and personal files | Inventory and retain exact bytes, including tool sources and uploaded appearance assets. |
| Legacy conversations | Preserve exact primary-owned SQL transcripts and pointers; validate [private expansion](legacy-conversation-access.md). Bulk Chat import is separate work. |
| Shared household state | Conversations, membership/read state, notebook, topology and public artifacts need a coordinated household checkpoint. |
| Installation authority | Map installation/release bindings explicitly. Supported resident restoration rotates installation/session identity. |
| Credentials and services | Reconnect deliberately; portable application archives omit dedicated credentials, model weights and unrelated host services. |

A full stopped same-host archive is useful for local recovery. It is not the
[portable individual archive](resident-recovery.md), nor proof of recovery on
another machine. Independently captured resident archives do not establish one
consistent shared-state checkpoint.

## Capture a coherent checkpoint

1. Record the source release, dependency environment, configuration paths, all
   resident IDs/installations, provider ownership and service supervision. Keep
   that inventory private rather than adding it to project documentation.
2. Wait for accepted conversations, world work, evaluations and provider attempts
   to finish. Resolve uncertain occupancy and outstanding transitions first.
3. Stop the supervisor/service so it cannot respawn workers. Confirm the listener
   is closed, every worker exited and all coordinator/private locks are available.
4. Recheck the stopped state for unfinished work and broker holds. Capture the
   household manifest, shared/private databases, eligible files, selected source,
   dependency environment and service metadata together, including SQLite state.
5. Check archive integrity and compare contents, membership, permissions and
   ownership with the stopped source. Retain a digest and a protected copy away
   from the machine. Preserve the original installation.

Do not run two active continuations of the same resident. Do not replace a newer
home's database with an older raw copy; that erases intervening activity and
bypasses installation and recovery checks.

## Rehearse the transition

Use a disposable copy with a deterministic provider and automatic clocks disabled.
Review the exact identity/root/release mapping before hydrating workers. Existing
supported restore tools enforce their archive compatibility; changing a release
label is not a substitute for a qualified transition.

Verify private records, exact conversation lookup, personal files, artifacts,
Growth, appearance, companion ownership and deliberate rest/wait state. Verify
shared histories, membership, notebook and public artifact bytes separately.
Account for documented startup projections and timestamps without relaxing
preservation of lived content. Exercise a clean restart and a complete recovery
of the rehearsal, including failures and interrupted transitions.

Automatic household release activation and coordinated portable household recovery
remain [separate work](multi-resident-roadmap.md#remaining-work). Bespoke maintenance
requires reviewed, recoverable transitions and a concrete local recovery plan.

## Cut over and observe

Take a fresh stopped checkpoint before the real cutover so later conversations
and work are retained. Select the tested source/environment and verify worker
bindings, operator access, private state and shared content before normal activity
resumes. Editing the checkout alone does not change a supervised release.

Check an ordinary conversation, chosen rest, invitation consent, an allowed
World Thread opportunity and another clean restart. Ordinary rest pauses World
Thread activity; chat remains eligible under the usual wait/busy/budget rules.
Validate any generated-tool interpreter or absolute-path changes before using
them. Add further residents only after the existing home's continuity is clear.

If recovery becomes necessary after new activity has resumed, preserve that newer
state first and report the continuity gap of any checkpoint replacement. Keep the
source installation and checkpoints until recovery has been demonstrated for the
selected release. Mechanical checks do not complete sustained behavioral observation.
