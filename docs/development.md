# Development and testing

Use the environment in [installation](running.md#install-a-new-home): Linux,
Python 3.13, the pinned dependency lock and GnuPG for encrypted recovery checks.
Run checks from the source revision being reviewed.

## Routine checks

```bash
.venv/bin/python -m pip check
.venv/bin/python scripts/check_repository.py
.venv/bin/ruff check backend tests scripts --select F
.venv/bin/python -m unittest discover -s tests -q
```

The unit suite uses disposable roots and injected providers. It covers standalone
and household execution, identity, attribution, persistence, generated tools,
recovery and scheduling. [Household checks](household-native-testing.md) describe
network-guarded runs and real browser fixtures. CI also checks Python and frontend
syntax; see [the workflow](../.github/workflows/tests.yml).

Test results describe a particular revision and environment. Keep run reports,
logs, screenshots and source fingerprints in an ignored `.platform/` directory,
an external temporary directory or CI artifacts. Rerun the relevant checks for a
new revision; dated test totals are not a permanent capability guarantee.

## Graph persistence

The [world graph skill](../backend/skills/write_world_graph_skill.py) returns
failure and logs the exception when a save fails. Optional relationship failures
are logged and do not increase successful edge counts.
[SelfModel.save](../backend/core/self_model/store.py) serializes to a unique
adjacent temporary file, flushes and syncs it, then replaces the destination.
Failed serialization, writing, syncing or replacement preserves the last complete
JSON file. The [regressions](../tests/test_world_graph_persistence.py) cover every
write operation, partial writes, serialization failure, permissions, export paths
and failed platform receipts.

An in-memory mutation or a canonical SQLite commit can precede an export failure.
Database and JSON export are separate operations; inspect current state before
retrying. The fix for [issue #5](https://github.com/manikmakki/TamAGI/issues/5) does
not repair previously corrupted exports or roll back earlier commits.

## Repository conventions

Keep application code, tests and fixtures, build/dependency files, CI, licenses,
required frontend assets, maintained guides and architecture decisions in Git.
Keep lived state, operator configuration, credentials, workspaces, local tooling
state and generated reports outside the tracked tree. `.gitignore` covers common
outputs; `scripts/check_repository.py` checks tracked-file policy and document links.

Documentation examples use generic paths and placeholders. Residents have no
project-wide canonical names or identities. The appearance catalog may name
actual artwork presets; selecting artwork does not create or recover an identity.
Use stable IDs to explain identity semantics without publishing a live resident's
IDs, files, model history, process inventory or recovery archive location.

Update the [documentation index](README.md) when adding a maintained guide. Record
architecture decisions in [ADRs](adrs/README.md). Keep the root README a short
introduction with links to installation, operation and development details.
