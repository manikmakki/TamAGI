# ADR-0019 browser recovery controls — 2026-11-09

Settings → Home & recovery now provides selected-resident encrypted backup,
download, archive upload/preview, cancellation, explicit replacement confirmation
and worker resume controls. Browser recovery extends the qualified
[individual recovery implementation](0019-resident-recovery-2026-11-09.md).
Migration into a fresh household remains an offline operator command.

## Evidence

- [Broad offline suite](0019-browser-recovery-2026-11-09/suite.json): **326 tests
  passed in 316.551 seconds**, 215 guarded Python processes, matching source before/after hashes, no real
  provider and zero denied network operations.
  [Test output](0019-browser-recovery-2026-11-09/suite.log).
- [Final focused API/lifecycle checks](0019-browser-recovery-2026-11-09/focused-api.json):
  **8 tests passed in 71.623 seconds**, 45 guarded Python processes, zero denied
  operations, unchanged source and no real model. ResourceWarnings were promoted
  to errors. [Test output](0019-browser-recovery-2026-11-09/focused-api.log).
- [Chromium browser checks](0019-browser-recovery-2026-11-09/browser.json) and
  [guard summary](0019-browser-recovery-2026-11-09/browser-guard.json): passed
  against actual isolated native workers, seven guarded Python processes, zero
  denied operations and zero uncaught JavaScript errors. Existing conversation,
  invitation, world, public-library, menu and private-view race checks also passed.
- Reviewed mobile screenshots of the
  [preview/confirmation](0019-browser-recovery-2026-11-09/preview-mobile.png) and
  [completed operation](0019-browser-recovery-2026-11-09/complete-mobile.png).

The broad and browser runs preceded the last backend-only expiry-loop and
disconnected-export bookkeeping additions. The final focused run covers those
additions along with all browser recovery lifecycle cases. The broad suite was
not repeated for that final housekeeping change. These are scripted checks with
injected providers, not real-model recovery/behavioral evidence.

## Behavior exercised

- Backups and restore previews are bound to a resident; wrong-resident uploads,
  downloads and commits are rejected. Operator authentication and the normal
  same-origin mutation rules apply.
- Preview validates the encrypted archive without stopping workers. Cancelling
  or expiring it removes decrypted staging without changing resident data.
- Confirmation is required before private replacement. The old private root is
  retained; restored state gets a fresh private installation identity. Other
  residents' later memories/files, shared messages and the household clock remain.
- Busy conversations, running generations and retained queued private evaluation
  jobs are refused without stopping workers or discarding their work. Recovery
  shares the scheduler lock and does not advance its clock while paused.
- An idle maintenance boundary drains supported requests, blocks new work, stops
  workers, holds lifetime/instance locks and completes the copy/selection before
  restarting workers. New chat and private mutations return temporary unavailable
  responses during that boundary.
- A lost browser does not cancel the independent operation or abandon stopped
  workers. Disconnected exports retain download-expiry bookkeeping. Interrupted
  preview decryption completes its cleanup and closes multipart spools.
- Failed worker startup exposes retry. Persistent failure between registry
  database update and manifest selection leaves workers stopped and supported
  work blocked; retry finishes the durable journal before starting workers.
- The restored resident's old offers/grants/execution authority are settled to
  prevent replay. Pending invitations expire without inventing a decline. Peer
  opportunities and canonical conversation membership/history remain intact.
- The UI clears passwords, archive inputs and previews when residents change;
  delayed responses cannot bind to the next resident. A successful restore
  invalidates old private response epochs, refreshes private views and retains
  the current chat draft.
- An owned staging directory is cleared on coordinator restart. A periodic task
  removes expired previews/downloads even without HTTP requests. Expiry skips
  active operations. Scoped resident routes bypass the service-worker cache;
  encrypted downloads have `Cache-Control: no-store`.

## Test-instance activation

Read-only preflight found no active canonical work, unsettled generation attempt,
running private execution, queued/running private interaction or evaluation, or
imminent scheduled tick. Test coordinator PID 367928 and its workers exited cleanly
after SIGINT; no force termination was required.

The retained household resumed on **0.0.0.0:7744**, PID **377032**, interactive
session **71008**, with all three original resident roots and IDs, its existing
password and the `*/15 * * * *` World Thread enabled. Authenticated read-only HTTP
checks confirmed the browser capability, idle recovery status and availability
of Cedar, Willow and River. [Activation evidence](0019-browser-recovery-2026-11-09/activation.json).
No actual resident was restored or migrated during this activation.

Production Ember PID 152015 and its `127.0.0.1:7742` listener were unchanged.
The test restart performs normal hydration/reconciliation; retaining original
roots does not mean startup bookkeeping is byte-identical.

## Remaining limits

This is a short **household-wide idle pause**, not hot maintenance of one worker
while others continue generating. Browser upload is capped at 1 GiB; previews
expire after 30 minutes or restart, encrypted downloads after 24 hours or restart.
Native release compatibility checks remain in force. API phases, private root
selection and operator confirmation do not add an automatic deployment path.

Fresh-installation migration remains CLI-only. Canonical shared-state/public-library
recovery, historical transcript viewing after migration, production adoption and
NATIVE-010's queued-evaluation shutdown case retain separate qualification work.
The Python network guard is not OS confinement of arbitrary native commands.
