# World graph persistence — issue #5

This change addresses [issue #5](https://github.com/manikmakki/TamAGI/issues/5)
on `codex/dependable-home`, based on
`8b4a26e8d673b4592da1ba53440b2bb890a35746`.

The [graph skill](../../backend/skills/write_world_graph_skill.py) reports a
failed save as `success=False` for new nodes, merges, updates, edges and deletion.
It logs the exception and omits successful mutation metadata. Optional edge
failures retain their existing partial-success behavior and accurate counts,
with exception logs identifying the rejected relationship.

[SelfModel.save](../../backend/core/self_model/store.py) writes to a unique
temporary file beside the destination, flushes and syncs it, then calls
`os.replace`. Failed serialization, writing, syncing or replacement leaves the
last complete JSON file intact and cleans up the temporary file. Existing file
permissions and explicit export paths are preserved.

This is not a rollback of the in-memory graph or an earlier canonical SQLite
commit. A platform export can fail after the canonical graph has been saved;
the result tells the caller to inspect current state before retrying. Database
and JSON persistence remain separate operations. Previously corrupted files
are not repaired by this change.

## Verification

The [ten focused tests](../../tests/test_world_graph_persistence.py) pass. They
exercise every write path with injected save failures, successful reloads,
optional edge failures, a real circular-reference serialization failure,
mid-write disk failure, sync/replacement failures, failed first saves, preserved
file permissions, export paths and a failed platform receipt after a canonical
commit. The original code reproduced false success and damaged JSON in these
scenarios.

The [complete offline suite](issue-5-world-graph-persistence-2026-11-12/report.json)
passed **373 tests in 372.003 seconds**, with
243 guarded Python processes and zero denied network operations.
The [test output](issue-5-world-graph-persistence-2026-11-12/stderr.log) and
[source manifest](issue-5-world-graph-persistence-2026-11-12/source.json) retain
the result and implementation/test hashes. The isolated source was unchanged
throughout the run. Tests used disposable state and injected providers; no live
resident or real model was used. The guard covers Python sockets/DNS, not
arbitrary native commands.

Ruff's undefined/unused-symbol checks, `git diff --check`, and documentation link
and example checks pass. These are source changes for review and merge; no
production release was selected and the GitHub issue remains open until closure.
Use `Fixes #5` in the merge pull request to close it when merged into the default
branch.
