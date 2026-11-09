# ADR-0019 final branch review — 2026-11-09

Scope: `codex/adr-0019-household`, reviewed against `codex/dependable-home`.
Implementation revision: **`d8cb8af90027887b4f6988ab50c283e4c3b7ad7b`**.
The following evidence commit adds documentation and retained reports only.

The review covered resident identity/isolation, invitation consent and canonical
conversation history, world scheduling/sponsorship, backup/restore lifecycle,
browser scope and cache behavior, and qualification isolation. It is a code and
scripted regression review, not a production adoption or sustained behavior claim.

## Findings fixed

- **Queued-evaluation shutdown:** an independent evaluation could leave a queued
  generation HTTP handler waiting after model-server close. Shutdown now revokes
  execution authority, cancels undispatched attempts and wakes waiters. Healthy
  upstream calls drain normally; their late results cannot authorize effects.
  Unknown occupancy remains held for reconciliation. New admissions are refused.
- **Interrupted restoration:** target invitation/offer/grant settlement previously
  followed successful browser commit, outside the durable transition. It now
  commits with the resident binding and replays idempotently before workers resume,
  including CLI/startup and browser retry. Peer opportunities remain unchanged.
- **Settings response races:** evidence, predecessor archive and permission lists
  now guard the captured resident and epoch. Their actions retain that scope and
  reject stale selection, including detached controls.
- **Menu-to-Chat switching:** resident selection updates the hidden conversation
  and draft while preserving the current menu page. Reload reconciles older saved
  resident/thread mismatches.

The README and documentation index now describe the implemented opt-in native
candidate. The native/recovery guides explicitly explain that the development
launcher runs the current checkout while retaining enrollment release labels.
Comparing those labels alone does not qualify cross-release compatibility.

## Verification

- [Full suite report](0019-final-branch-review-2026-11-09/suite.json) and
  [test log](0019-final-branch-review-2026-11-09/suite.log): **331 tests passed in
  314.802 seconds**, 215 guarded Python processes, zero denied network operations,
  no live instance or real model used. Before/after manifests show the exact clean
  implementation revision above and unchanged source.
- [Focused shutdown report](0019-final-branch-review-2026-11-09/shutdown.json) and
  [test log](0019-final-branch-review-2026-11-09/shutdown.log): all 17 native model
  tests pass, including queued HTTP completion, healthy in-flight drain and unknown
  occupancy retention; one guarded process and zero network denials.
- [Focused recovery report](0019-final-branch-review-2026-11-09/settlement.json) and
  [test log](0019-final-branch-review-2026-11-09/settlement.log): two fault regressions
  pass, ten guarded processes, zero network denials. Failed manifest selection,
  browser Resume, startup replay, idempotency and peer preservation are checked.
- [Chromium report](0019-final-branch-review-2026-11-09/browser.json) and
  [guard summary](0019-final-branch-review-2026-11-09/browser-guard.json): passed
  against actual disposable worker processes, seven guarded Python processes,
  zero denied network operations and zero uncaught JavaScript errors. Includes
  the new delayed Settings/detached action, menu/draft and reload cases, plus the
  existing chat, reasoning, scroll, vision transport, invitation, world, public
  artifact and browser recovery regressions. The model provider is injected.
- Syntax checks pass for all 71 changed Python files and seven JavaScript/module
  files. Relative documentation file links and Git whitespace checks pass.

Reproduce from an isolated checkout, using new output directories:

```bash
/opt/TamAGI/.venv/bin/python scripts/qualify_household.py \
  --source /tmp/tamagi-adr-0019 --output /tmp/tamagi-review-suite-new
/opt/TamAGI/.venv/bin/python scripts/check_household_browser.py \
  --output /tmp/tamagi-review-browser-new
```

These guards cover Python socket/DNS traffic and isolated fixtures. They are not
OS confinement of arbitrary native commands. Mechanical continuity and transport
checks do not establish real-model behavior or vision understanding.

## PR #8 CI preflight follow-up

The first [PR workflow run](https://github.com/manikmakki/TamAGI/actions/runs/42704461269/job/129457478277)
stopped at **Check Python**, before unit tests and frontend checks ran. The log
confirmed successful dependency validation followed by 12 Ruff findings: eleven
unused imports and one unused local assignment. The local suite and syntax checks
above had not included CI's Ruff preflight; their passing results did not establish
that the complete GitHub workflow would pass.

The follow-up removes unused imports and the unused assignment while retaining
message validation. CI's exact Python preflight now passes locally: `pip check`,
Ruff 0.17.3 with `check backend tests scripts --select F`, and `compileall`.
The service worker and both inline frontend script blocks also pass the workflow's
Node syntax checks. All 17 native model tests pass in 1.456 seconds under the
offline guard, with one guarded process, zero denied operations and no real model.
The full GitHub workflow remains the merge check for the follow-up commit.

## Remaining rollout work

The branch is ready for PR review. Household mode remains opt-in; the review did
not enroll Ember, change production configuration or replace live data. Production
Ember PID 152015 and retained test coordinator PID 377032 were left running;
the test backend was not restarted to activate this review's final fixes.

Controlled production resident adoption/history migration, compatible release
activation and coordinated shared-state/public-library recovery remain separate
work. Individual migration is CLI-only; historical shared transcript evidence is
preserved privately but does not yet have a dedicated migration viewer. Automatic
household deployment remains disabled. Companion group activation, deferred PWA
push/device qualification, selector design and sustained real-model household
burn-in retain their documented scope. See the [native guide](../household-native.md)
and [roadmap](../multi-resident-roadmap.md).
