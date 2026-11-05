# Workspace paths and tool development

The active resident's workspace is the base directory for relative `read`, `write`, and `exec` working-directory paths. Use the installation paths reported in runtime context or by `runtime_change(action="inspect")`. The platform maintains `WORKSPACE.md` at that root and a tool guide at `skills/README.md`. Do not infer paths from the shell user's home directory, the repository's example workspace, or an earlier installation. Activated subprocess tools also receive the active workspace as their working directory and as `TAMAGI_WORKSPACE`.

## Three separate locations

| Location | Purpose |
| --- | --- |
| Canonical repository, `/opt/TamAGI` on this installation | Application source and development work. Core changes go through isolated worktrees, validation, builds, and deployment. |
| Selected release | The exact built application selected by the supervisor. Editing a repository file does not update the running release. |
| Active instance and its workspace | Durable lived state and the resident's files. These survive application releases. The instance includes canonical SQLite state, histories, and configuration; its workspace contains the files below. |

An instance workspace may look like `/opt/TamAGI/data/instances/<instance>/workspace`. The runtime's actual configured path takes precedence over this example. Neither the repository's `workspace/` nor shell `~/workspace` automatically refers to it. Portable restoration may place it elsewhere.

## Workspace conventions

| Relative path | Purpose |
| --- | --- |
| `workshop/` | Creations intended for the resident's Inventory: documents, drawings, dashboards, and other finished or developing artifacts. Use this convention during chat, quiet work, and World Thread activity alike. |
| `skills/<tool_name>/` | Editable source and development notes for a generated subprocess tool. Creating files here does not activate them. |
| `scratch/` | Temporary experiments and disposable working files. These are not an independent source of durable state. |
| `research/` | Existing working notes and research. Keep existing files and references intact; a path convention does not silently relocate earlier work. |
| `.tools/` | Platform-managed, versioned copies of tool source. Do not edit activated copies; revise editable source and propose a new version. |
| `.artifacts/` | Platform-managed, content-hashed snapshots created by `register_artifact`. These are saved versions, distinct from editable working files. |
| `IDENTITY.md`, `SOUL.md`, `USER.md` | Service-managed identity views. Use `revise_identity` with a reason and expected revision. |
| `TASKS.md` | A compatibility view of canonical pursuits, maintained through the pursuit/task services. |
| `WORKSPACE.md`, `skills/README.md` | Platform path and tool-development guidance. Runtime context and inspection report the current installation's actual paths. |

`workshop/` is the intended boundary for a future Inventory view. The current home/workshop UI still lists **registered artifact snapshots from anywhere in the workspace**; it does not scan the folder. To retain and display a current creation with that UI, call `register_artifact` with its workspace-relative path. Register a revised file again when a new saved version is wanted. Registration is evidence preservation, not tool activation.

## Building a callable tool

Generated code runs in a separate process instead of being imported into FastAPI. Under `unrestricted_host`, that process may have host-root authority: process separation provides fault isolation, not confinement. A tool trial also retains the installation's authority profile, so a temporary trial workspace does not make host effects harmless.

Keep the first version self-contained, for example `skills/system_health/worker.py`:

```python
import json
import os
import sys

arguments = json.load(sys.stdin)
load = list(os.getloadavg())
print(json.dumps({
    "success": True,
    "output": "System load averages collected.",
    "data": {"load_averages": load},
}))
```

The platform sends one JSON arguments object on stdin and expects one JSON result object on stdout. The result requires a boolean `success`; `output` is text, `data` is an object, and optional `error` is text. Keep diagnostics on stderr so stdout remains valid JSON. The worker must exit successfully and stay within its configured timeout of 1–120 seconds and the output limits. A contract test establishes that the worker runs; it does not by itself establish that the capability improves behavior.

A tool proposal uses this definition as the `value` in `runtime_change(action="propose", surface="tool", ...)`:

```json
{
  "name": "system_health",
  "description": "Read the host's current system load averages.",
  "command": ["/usr/bin/python3", "skills/system_health/worker.py"],
  "parameters": {},
  "timeout": 10
}
```

`command` is an argv array, not a shell command. Choose an explicit, installed interpreter. `parameters` follows TamAGI's `Skill.parameters` shorthand: argument names map directly to definitions, and each required argument has `"required": true`. Do not wrap that mapping in top-level JSON Schema `type`, `properties`, or `required` fields. For example:

```json
{
  "interval": {
    "type": "integer",
    "description": "Sampling interval in seconds",
    "required": true
  }
}
```

The normal lifecycle is:

1. Write and locally inspect the editable worker. Record the intended outcome and relevant regression checks.
2. Call `runtime_change` with `action="propose"`, `surface="tool"`, its definition, and an intention. Keep the returned proposal ID.
3. Call `runtime_change(action="test_tool", proposal_id=..., arguments={...})`. Inspect the actual result and failure evidence; correct the editable source and make a new proposal if necessary.
4. Call `runtime_change(action="evaluate", proposal_id=...)`. This automatically queues a baseline/candidate comparison; it does not require human review, a separate comparative marker, or an `experiment_id`. The returned `job_id` tracks execution. Wait using `end_turn(status="waiting", wake_event="evaluation_completed")` or inspect later. Use `runtime_change(action="inspect", proposal_id=...)` for concise status, evaluation jobs, and a ready-to-use `next_call`. Do not submit another evaluation just to check status. Set `include_evidence=true` on inspection to read the full records. An interrupted job can leave an unfinished experiment marked `running`; that is incomplete evidence.
5. Only an experiment's **overall** decision `promote` qualifies; one successful scenario does not. Call `runtime_change(action="activate", proposal_id=..., experiment_id=...)` with **both** IDs from the same proposal and its qualifying experiment. A missing experiment ID requires supplying the ID, not rerunning evaluation. Confirm the resulting active revision and invoke the tool. The default evaluation checks general behavior and worker execution, not every output's accuracy or task-specific usefulness.
6. Revise through another proposal or use `runtime_change(action="revert", target="system_health", ...)` to restore or retire a version. Do not edit `.tools/` in place.

Proposal creation snapshots workspace files explicitly named in the command into `.tools/`. It does **not** automatically bundle sibling Python modules, imported packages, resource directories, or an interpreter environment. A copied worker that imports an adjacent helper can therefore fail after proposal even if the editable version worked. Prefer a self-contained worker or declare and validate its external dependencies and explicit interpreter. Process tools receive a minimal environment; do not rely on shell activation or inherited environment variables.

Installing a package into the running release's or repository's virtual environment is not durable dependency management: the next exact release can replace that environment. Application dependency changes belong in source/lockfiles and the release lifecycle. For a separately managed tool environment, retain its reproducible dependency requirements and reconstruction procedure rather than assuming a current absolute path is portable.

## Legacy compatibility

Non-platform mode retains directory discovery of Python files that subclass `backend.skills.base.Skill`. In platform mode, `discover_custom_skills` deliberately skips those imports. Copying a legacy `Skill` class into `skills/`, registering its source as an artifact, or restarting the service cannot substitute for the subprocess-tool lifecycle.
