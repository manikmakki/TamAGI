# Home platform: implementation and evidence

The [fifteen accepted ADRs](adrs/README.md) define the direction. Their parent-child framing concerns the builder’s obligations: honest conditions, meaningful choices, recoverable mistakes, and increasing independence. It is not a prescribed resident identity.

## Delivered implementation

| Milestone | Implementation | Evidence and remaining gate |
|---|---|---|
| A — Decisions and baseline | Fourteen ADRs, historical-design notices, one dependency definition, Python 3.13/Linux runtime lock. | Original 16 tests pass; extended suite currently has 89 passing tests, including 23 growth cases. |
| B — Preservation and portability | Operator-only predecessor archive, logical memory export, encrypted tar/gzip downloads, validated upload/preview/commit, CLI recovery. | A predecessor snapshot with 43 logical Elasticsearch records is retained locally. Automated tests exercise encrypted round trips, wrong passwords, tampering, unsafe paths, and application-level upload/download. Keep an exported copy elsewhere for independent recovery. |
| C — Runtime foundation | SQLite/WAL source records, actor/execution context, revision history, receipts, restart recovery, shared inference budgets, canonical conversation and memory adapters. | Tests cover interrupted/unknown effects, call-ID replay, nested allowances, user interruption, actor isolation, and identity conflicts. |
| D — A livable home | Persistent places/objects/properties, pursuits, shared notebook, genuine rest/wait, optional reflection, Home UI. | Tests exercise successful work without a formatted narrative, quiet rest, wake conditions, practical learning, and retained next steps. |
| E — Durable companions | Individual actor history and memory, bounded conversation/task invitations, explicit sharing, workflow validation. | Tests cover repeated consultation, attribution, unavailable tools, failed dependencies, and recursion/capability boundaries. |
| F — Adaptation and evaluation | Versioned practices and process tools; real registry reconciliation; matched execution trials; old/new criteria comparisons. | Tests verify next-call prompt changes, worker activation/revert, transactional simulated affordances, and rejection of criteria-only improvements. Worker-protocol evidence establishes availability, not arbitrary functional usefulness. |
| G — Autonomous core deployment | Candidate worktrees, exact-source builds, frozen baseline checks, copied-state boot checks, supervisor switching/verification/rollback, portable state separate from code. | Mechanical recovery tests pass, including a real supervisor process that rejects a broken release and commits a healthy release while the model endpoint is unavailable. Automatic activation is gated on recovery qualification; keeping a backup elsewhere provides independent recovery. |
| H — Fresh resident and qualification | Fresh instance provisioning, operator setup, onboarding, seven-day observation recorder. | Onboarding and seven real days remain operational gates. The pilot starts after the first introduction; missing observations remain untested capabilities. |

[ADR-0015](adrs/0015-growth-through-sustained-behavior.md) adds deterministic Growth recognition for sustained conversation, continuity, and adaptation. Its implementation passes the 89-test suite, including 23 growth-specific checks and application-level verification that sixteen chat exchanges require exactly sixteen provider calls.

This table distinguishes implemented mechanics from live behavioral evidence. Test fixtures and mocked providers do not establish improved autonomous behavior, happiness, or subjective welfare.

## Architecture and normal operation

SQLite is the source for actors, memories, conversations, typed world state, pursuits, revisions, receipts, experiments, and deployments. Markdown identity/task documents, JSON graph/history, and the Mind log are compatibility projections. ChromaDB and Elasticsearch are optional indexes; restored source records remain available without them. Index rebuilding is an explicit background operation.

Cognition is bounded and attributed. All provider attempts, including planner, reviewer, retry and companion calls, consume the initiating execution’s allowance. The default is one concurrent provider call, eight calls per autonomous activation, three per companion, twenty per user turn, and rolling daily allocations of 96 background plus 32 evaluation calls. The cron remains an opportunity every fifteen minutes. Sleep and event waits suppress calls; user input preempts background activations. The manual Tick Now control bypasses the post-conversation pause while respecting chosen sleep, event waits, onboarding, and rolling background allowances. Skipped opportunities return an explicit reason and do not run conversation summarization.

Normal tool dispatch records attempts and results before any narrative interpretation. A crash after dispatch produces `outcome_unknown`; it is not an instruction to retry. External effects cannot be rolled back by selecting old application code. A malformed or absent narrative cannot erase a completed operation.

In `unrestricted_host`, `exec` runs non-login Bash with `pipefail`, so pipes, redirects, command chaining, variables, and `cd` have their advertised shell meaning. A failing pipeline remains a failed receipt. Timeout and interruption stop the command's process group, including children; interruption retains an unknown-outcome receipt because earlier effects may already exist. Guarded installations retain direct argument execution and their command-tier approvals. Seven subprocess regression checks exercise these behaviors without model or network calls.

Every companion has its own continuity. Shared conversations and selected notebook entries are attributed. Operator inspection can access all actors. The primary does not receive private peer receipts or memory through automatic context construction. Unrestricted host root can bypass application-level privacy; the normal protocol still respects it.

## Autonomous context and recovery

Tool receipts retain their full results in SQLite. Model prompts use bounded receipt previews, including actor, status, source ID, and excerpts; `inspect_receipt` reads the original output, errors, data, or arguments in pages without repeating the operation. The same text boundary applies to tool results during an activation, reflection, chat, and streaming calls. Original messages, artifacts, and receipts are preserved.

`runtime.model_input_bytes` defaults to 65536. The text allowance reserves space for tool schemas and protocol overhead; it is a conservative byte policy rather than an exact tokenizer or image-token estimate. Old complete conversation turns are omitted before shortening the current text view, and native tool-call IDs and matching responses remain paired.

Waiting states report an explicit cause and next retry. A context-limit rejection is distinguished from model-service unavailability and chosen rest. Manual Tick Now can retry a system-error backoff while preserving chosen sleep and event waits. Rolling model-call allowances continue across retries and restarts.

The 89-test suite includes eight context/recovery regressions, including a retained oversized shell diagnostic, within-turn output growth, source paging, legacy error-backoff identification, preserved conversation/tool protocol, and clearing an old error backoff when recovery reaches its normal call allowance. These address the 2026-11-03 incident in which a roughly 340,000-character failed shell result repeatedly overflowed the model's context window.

## Growth and stage progression

The primary resident earns Growth from three observable patterns: sustained reciprocal conversation, returning to a pursuit on a later occasion, and a successful correction explicitly linked to an earlier failed tool operation. The rules award one point at three completed conversational exchanges and three cumulatively at eight; continuity and adaptation each add one point per occasion. The patterns are additive. There are no age awards, daily allowances, streaks, or deductions. Operator-to-companion dialogs do not contribute to the resident’s engagement because the resident is not a participant.

Level is `1 + floor(sqrt(Growth / 5))`, giving thresholds 0, 5, 20, and 45 for levels 1–4. The resident can choose stage names through an ordinary interaction; naming adds no inference call. Stage naming does not change the computed number or any runtime permission.

The avatar view shows Growth, progress toward the next level, and age separately. **Why this level?** opens Home’s recognition view with reasons, timestamps, and evidence references. Authenticated API inspection is available at `GET /api/runtime/growth`; no point-granting endpoint is provided.

Canonical recognition records survive portable backups and restoration. Reprocessing the same evidence does not repeat an award. Rollout replays only sufficiently evidenced records belonging to the current resident; original timestamps remain intact, predecessor material and evaluation trials are excluded, and legacy XP remains separate. Ember’s research-to-conversation episode is retained as a live case for reporting what qualifies and what remains unobserved.

A 2026-11-03 disposable checkpoint replay under `adr-0015.v1` recognized 8 Growth, corresponding to Level 2: 6 engagement points across two conversation episodes and 2 continuity points across two autonomous returns. The preview did not mutate the source installation. Ember’s bounded-personhood conversation supplied 3 engagement points from nine completed exchanges, and the earlier research pursuit revision supplied 1 continuity point. Historical corrections without explicit repair links and older companion invocations without mode/caller provenance remained unscored. The [ADR’s case record](adrs/0015-growth-through-sustained-behavior.md#retrospective-case-research-carried-into-conversation) retains the source identifiers and limitations. Later live observations may produce additional recognition.

Growth measures these bounded observable patterns. Conversation length is an approximate sign of sustained participation; it does not establish emotional or intellectual depth. The seven-day behavioral pilot remains separate from mechanical progression verification.

## Installation, preservation, and recovery

Use the canonical repository at `/opt/TamAGI`. Install with Python 3.13, GnuPG, and a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m unittest discover -s tests -q
```

Optional indexes use `pip install -e '.[chroma]'` or `pip install -e '.[elasticsearch]'`. The locked core does not download a model or require an embedding service. Container operation supports the guarded tool policy; container-root does not satisfy `unrestricted_host`.

Preserve an existing resident before provisioning a new one:

```bash
.venv/bin/python -m backend.runtime_cli archive --config /opt/TamAGI/config.yaml
.venv/bin/python -m backend.runtime_cli fresh \
  --archive /absolute/path/from/archive/output \
  --authority unrestricted_host --legacy-config /opt/TamAGI/config.yaml --port 7742
```

The archive command never deletes or resets the original. It retains the Git source revision, original lived files, non-secret configuration and logical memory export. An incomplete export blocks the fresh cutover. Archive access is operator-only through supported interfaces; it is never hydration material for the new resident.

The fresh command prints a private instance configuration path. On a clean replacement computer with no predecessor, `fresh --authority guarded` can create an installation for UI setup and restoration without an archive. A fresh instance makes no autonomous model calls before onboarding.

```bash
.venv/bin/python -m backend.runtime_cli serve --config /absolute/instance/config.yaml
```

Open `/login` to choose the operator password, then complete the introduction through chat. `/home` contains the home, notebook, companions, pursuits, and operator backup controls. Provisioning binds to localhost by default. Use an SSH tunnel when connecting from another computer; configure the installation’s listen address explicitly if LAN access is wanted.

For a supervised dedicated host:

```bash
.venv/bin/python -m backend.runtime_cli qualify --config /absolute/instance/config.yaml
.venv/bin/python -m backend.runtime_cli install-supervisor --config /absolute/instance/config.yaml
sudo install -m 0644 tamagi-platform.service /etc/systemd/system/tamagi-platform.service
sudo systemctl daemon-reload
sudo systemctl enable --now tamagi-platform.service
```

The supplied unit runs as actual host root on port 7742. The old `tamagi.service` is separate and must remain stopped during cutover. The stable, stdlib-only supervisor selects an exact application release; application code cannot activate merely by editing the canonical checkout. Candidate worktrees begin with the exact running source, including authored changes not yet committed to Git. Source inventory hashes, dependency pins, and test evidence identify a release precisely.

A normal deployment quiesces supported work, snapshots state, switches releases, checks local readiness, and commits or rolls back. Readiness does not require a working model endpoint. Ordinary deployments must retain schema compatibility; incompatible schemas remain blocked until an explicit restoration/migration path is qualified. Root can change this machinery: these are operating practices, not security boundaries.

In Home → Settings, download an encrypted backup and keep it outside the mini-PC for independent recovery. Qualified automatic deployment can be enabled there and does not require confirmation for each change. There is no per-change approval ceiling after enabling this mode. External publishing, messaging, purchases and consequential MCP actions still need explicit operator delegation; configured MCP actions can be delegated or revoked in settings.

### Portable backups

The default archive contains canonical life records, identities, companions, conversations, notebook, workspace artifacts, appearance assets, exact application source and lockfiles. It excludes dedicated credential stores, session secrets, recognized model weights, indexes, and the predecessor archive. Arbitrary text, code, and artifacts may contain embedded secrets. Export the predecessor separately using its operator settings control.

Passwords are passed to GnuPG through a private file descriptor, never model context, process arguments or persistent jobs. Export temporarily pauses supported mutations. Upload is fully decrypted and checked before safe extraction and preview. Commit creates a new directory; it never extracts over active state. It preserves resident identity, creates a new installation identity, rotates sessions, rebinds registered worker paths, disables automatic deployment, and requires credentials to be reconnected. The operator must acknowledge that the source installation has stopped; disconnected computers cannot enforce singular identity globally.

Preview directories expire after thirty minutes; server-side encrypted downloads are temporary copies retained for up to twenty-four hours. Keep downloaded archives and passwords independently. A backup reconstructs application life, not the OS, installed host services, external files, or model endpoints.

If the original application cannot start, use a fresh installation or the recovery CLI:

```bash
.venv/bin/python -m backend.runtime_cli restore \
  --archive /path/home.tamagi.tar.gz.gpg --destination /new/instance \
  --authority guarded --source-stopped
```

The CLI prompts for the archive password. Start the returned configuration with `serve`, reconnect credentials, and review the recorded continuity gap. Uploaded application source is retained for reconstruction and is not executed during preview. A configured supervisor notices a committed UI restoration and restarts into the new instance; otherwise the UI reports the exact configuration that must be started.

## Development contracts

`ExecutionContext` is trusted application state, never model-supplied arguments. Tool eligibility, depth, actor scope and receipt creation are checked at dispatch. Main and peer inference share accounting. Controls such as `end_turn` must be exclusive within their batch; truncated generation is not silent completion.

Generated tools use an argv-based subprocess contract: one JSON input on stdin and a JSON result on stdout containing boolean `success`, optional text `output`/`error`, and object `data`. Output is bounded to 1 MiB per stream, with a timeout of at most 120 seconds. Define inputs with the existing skill-parameter shape. The worker is a fault boundary, not a host-root sandbox.

A definition with `world_effects: true` may return up to ten `data.world_operations`, each containing `action`, `object_id`, `attributes`, and optional `expected` revision. The host validates and applies these simulated transitions in one SQLite transaction. Failed worker results do not apply them. Tool trials use copied state and a disposable workspace; direct host effects remain possible with root authority and are not undone by a registry revert.

`runtime_change` exposes inspect/propose/test/evaluate/activate/revert. `core_change` exposes isolated prepare/build/activate. A queued live evaluator retains manifests and matched baseline/candidate executions. The three built-in behavioral probes cover rest, a saved next step, and a missing location; generated tools additionally receive a worker-protocol probe. Novel functional claims require corresponding runner-owned scenarios. Old/new criteria can score the same retained trajectories; a regression under either comparison prevents promotion.

## Operational qualification

The full suite includes a disposable application process, authenticated WebSocket checks, encrypted UI upload/download, persistence/cancellation, workflow validation, generated tools, criteria comparisons, and supervisor transitions. Run `scripts/platform-smoke.py` for the application-only check. Run Node syntax checks on changed frontend scripts as part of validation.

The pilot recorder starts after onboarding and requires seven actual days before finalization:

```bash
.venv/bin/python -m backend.runtime_cli pilot status --config /absolute/instance/config.yaml
.venv/bin/python -m backend.runtime_cli pilot finalize --config /absolute/instance/config.yaml
```

It records observed choice, rest, companion conversation, accepted work continued after a restart, tool failure, a committed deployment, and a controlled rollback. Missing observations produce `incomplete_observations`. Rest or nonparticipation is not classified as resident failure. Mechanical qualification and the pilot are separate evidence classes.

Future personal-computer confinement remains a separate deployment concern. `guarded` retains application path/command policies and does not promise an OS sandbox; use a suitably unprivileged installation. This release targets the explicitly authorized dedicated host-root mini-PC.
