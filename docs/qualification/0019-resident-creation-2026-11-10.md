# Household resident creation — 2026-11-10

The native household now has a creation workflow for a genuinely new TamAGI in
an existing household. **New TamAGI** in the character chooser and **Create a new
TamAGI** in Settings open four steps: name/look/nature, personality, optional
starting world, and review. Vex, Echo and Ember artwork is previewed directly.
The operator's name comes from the household; no operator-name question is asked.
See the [operator guide](../creating-a-tamagi.md).

## Creation and continuity

The authenticated, same-origin `POST /api/household/residents` accepts a stable
request ID and bounded creation fields. `GET /api/household/creation` supplies
choices and defaults without provisioning anything. Households support up to
16 residents, with distinct participant names.

Creation uses the existing idle maintenance boundary. Active conversations,
private work or uncertain broker work prevent enrollment until resolved. Once
idle, workers stop, a fresh private home is provisioned, and a durable journal
selects the matching household manifest and database record. Workers then resume
with the updated roster. Shared conversations, existing private roots, default
resident, installation identities and schedule remain retained.

The new resident has fresh identity and installation IDs, selected personality
and avatar, private world objects and a mechanically initialized home. No prior
memory, relationship, private file, credential or integration is copied. Host
authority starts guarded; worker clocks, index projection and automatic
deployment remain disabled. The existing household clock controls later activity.
Creation itself requests no model generation.

Repeated requests recover the same resident. A changed payload under a used
request ID conflicts. The browser freezes an uncertain submission through lost
acknowledgments and temporary busy responses. Retrying a committed creation whose
worker failed to start retries resumption rather than founding another resident.
Startup reconciles a journal interrupted between database and manifest selection.

## Retained verification

The [source manifest](0019-resident-creation-2026-11-10/source.json) records the
working-tree implementation over `8945161a73417ccb4348c12a5878fe71cf927444`,
qualification provenance and implementation/test hashes. Tests ran from an
isolated source copy outside `/opt/TamAGI`; its before/after manifests match.
No clean committed revision is claimed. These source checks are separate from
the subsequent [production activation](0019-resident-creation-activation-2026-11-10.md).

- [Full offline report](0019-resident-creation-2026-11-10/suite.json) and
  [output](0019-resident-creation-2026-11-10/suite.log): **358 tests passed in
  369.419 seconds**, 239 guarded Python processes, zero denied network operations.
- Thirteen new regression cases cover fresh/private state, existing-state
  preservation, validation and capacity, owner/origin boundaries, idempotency,
  pre-selection cleanup, interrupted database/manifest selection, recovery
  collision, failed-worker retry, actual worker startup, coordinator restart and
  a new resident's first canonical conversation.
- [Browser run](0019-resident-creation-2026-11-10/browser-run.json) and
  [40 checks](0019-resident-creation-2026-11-10/browser.json): desktop and mobile
  Chromium, ten guarded Python processes, zero denied network operations and no
  uncaught browser exceptions. Creation checks cover actual avatar pixels,
  cancellation and draft preservation, Back/Next choices, duplicate clicks,
  lost acknowledgment followed by busy refusal and identical retry, one new
  identity, roster/colors/scoped chat updates and reload persistence.
- [Mobile creation](0019-resident-creation-2026-11-10/native-create-mobile.png)
  and [review](0019-resident-creation-2026-11-10/native-create-review.png)
  screenshots use only disposable residents.
- Python unused-code lint, syntax, JavaScript syntax, whitespace, and
  documentation links/examples pass. Native script/style versions are advanced
  so cached older assets do not hide the workflow after release activation.

The provider was injected and deterministic. The Python socket/DNS guard and
restricted browser DNS are not OS confinement of arbitrary commands. No live
resident, production model, production enrollment or service operation was used.

## Release scope

The operator subsequently [activated this workflow](0019-resident-creation-activation-2026-11-10.md)
in Ember's separately pinned production household, preserving its current state
and chosen rest. Deployment created no new resident; the activation record
retains its own checkpoint, transition and live verification evidence. The
standalone first-installation wizard is outside this change.
Recovering a historical resident, promoting a companion, cross-release household
recovery and sustained behavior with another full resident retain separate scope.
