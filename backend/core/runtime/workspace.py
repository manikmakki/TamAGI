"""Resident-facing directory conventions; application state stays service-managed."""

from pathlib import Path


DIRECTORIES = {
    "workshop": "Creations and finished or in-progress artifacts to browse in Inventory.",
    "skills": "Editable process-tool projects, one skills/<tool_name>/ directory per tool.",
    "research": "Research notes and working material; existing files remain here.",
    "scratch": "Temporary experiments. Nothing here is automatically deleted.",
}
MARKER = "<!-- TamAGI platform-managed workspace guide -->\n"

WORKSPACE_GUIDE = (
    MARKER
    + """# Your workspace

The runtime reports this workspace's actual absolute path in every interaction.
Use paths relative to it with read/write/register_artifact and exec.working_dir.
An exec command starts here by default; after cd, ordinary shell relative paths
follow that command's current directory. Shell ~ means the host account's home,
not this workspace. /opt/TamAGI/workspace is a legacy directory, not this home.

| Path | Use |
| --- | --- |
| workshop/ | Creations, documents, drawings, dashboards, and other artifacts, at any stage. |
| skills/<tool_name>/ | Editable source and documentation for your executable tools. Read skills/README.md first. |
| research/ | Research and working notes. Existing work need not be moved. |
| scratch/ | Temporary experiments; no automatic deletion. |
| .tools/ | Platform-owned frozen executable revisions. Do not edit these to revise a tool. |
| .artifacts/ | Platform-owned saved artifact snapshots. These differ from editable originals. |
| .attachments/ | Platform-owned, content-hashed images accepted through chat. |

These conventions help you find work; they are not restrictions on host authority.
Use subdirectories freely. Your IDENTITY.md, SOUL.md, USER.md and TASKS.md at the
root are service-managed views: use revise_identity and pursuit/task controls.
Conversation history, memories, world state and the database are managed by their
services, not files to create manually here. The simulated Workshop place is
separate from workshop/ on disk: world_action does not create host files.

register_artifact saves a file's current bytes for reuse and the current artifact
view. Saving in workshop/ alone does not currently register or activate anything.
Registering an artifact does not make it a callable tool. Tools must complete the
runtime_change proposal, test, evaluation and activation lifecycle.

The canonical repository is application source, not resident storage. Releases
are selected by the supervisor; editing a checkout does not change running code.
Use core_change for application releases. Do not use old workspaces, predecessor
archives or another actor's private files as your own memory or output location.
"""
)

TOOL_GUIDE = (
    MARKER
    + """# Building a tool

Work in skills/<tool_name>/ relative to the active workspace. This platform does
not import dropped Python Skill subclasses into the application. A restart or
register_artifact cannot activate them. A registry created in a shell subprocess
is also separate from the running application's registry.

A process tool receives one JSON object of arguments on stdin. Write exactly one
JSON result to stdout, with boolean success, optional text output/error and object
data. Send diagnostic messages to stderr. A self-contained standard-library
Python worker can look like this:

```python
import json
import sys

arguments = json.load(sys.stdin)
print(json.dumps({"success": True, "output": "Hello " + arguments.get("name", "world"), "data": {}}))
```

Propose with runtime_change(action="propose", surface="tool", intention=...,
value={"name":"hello", "description":"Say hello", "command":["/usr/bin/python3",
"skills/hello/main.py"], "parameters":{"name":{"type":"string",
"description":"Who to greet", "required":False}}, "timeout":30}).
This example is Python notation for the tool arguments, not a shell command.
parameters is a map of field names to definitions (type/description/required),
not a full JSON Schema with a top-level properties object.

1. Save and exercise your worker source in skills/<tool_name>/.
2. Propose it; retain the returned proposal ID.
3. test_tool with that proposal ID and representative arguments. Inspect success.
4. evaluate with proposal_id to queue an automatic baseline/candidate comparison.
   The returned job_id tracks execution, not approval. No human review or separate
   comparative marker is needed. Do not pass experiment_id to evaluate. Wait with
   end_turn(status="waiting", wake_event="evaluation_completed") or inspect later.
   Use inspect with proposal_id for concise status, jobs, and an exact next_call;
   include_evidence=true returns full evidence. Do not queue another evaluation
   just to check status. An interrupted job can leave an unfinished experiment.
5. Only an experiment's overall decision="promote" qualifies. A single successful
   scenario is insufficient. activate requires BOTH proposal_id and experiment_id
   from that qualifying experiment; it does not find the experiment automatically.
   A missing experiment_id means supply the ID, not repeat the comparison.
6. Inspect active revisions and invoke the actual named tool. Only then claim
   the running application can call it. Active tool changes appear at the next
   model-call boundary, without restarting the application.

The default evaluation checks a few general behaviors and worker protocol
availability. It does not verify every output or prove task-specific usefulness.
Inspect the retained evidence and exercise representative inputs honestly.

Use an explicit installed interpreter, never assume `python` or an activated
virtualenv. The runtime reports its Python executable for inspection, but that
release path is not a durable dependency environment for your own tools.
Declare dependencies in your project documentation with exact versions. Merely
pip-installing into the repository or a running release does not carry dependencies
into future releases. There is no automatic dependency installer for tools yet.

At proposal time, file arguments within this workspace are copied and hashed in
.tools/. Only those individual argv files are frozen: sibling imports, relative
resources and virtualenvs are not automatically bundled. Prefer a self-contained
worker or explicitly manage its dependencies. Save source before proposing it so
the platform can freeze the actual file rather than retaining an unresolved path.
Workers start in the active workspace (a disposable workspace during tests), with
TAMAGI_WORKSPACE, TAMAGI_WORKSHOP, TAMAGI_SKILLS and TAMAGI_SCRATCH in the environment.
Revise editable source and propose a new revision; do not modify .tools/ snapshots.
Reverting removes/replaces actual callable versions, not unrelated experiences.
"""
)


def paths(workspace):
    root = Path(workspace).resolve()
    return {name: str(root / name) for name in DIRECTORIES}


def environment(workspace):
    root = Path(workspace).resolve()
    return {
        "TAMAGI_WORKSPACE": str(root),
        **{
            "TAMAGI_" + name.upper(): str(root / name)
            for name in ("workshop", "skills", "scratch")
        },
    }


def prepare(workspace):
    root = Path(workspace).resolve()
    for name in DIRECTORIES:
        (root / name).mkdir(parents=True, exist_ok=True)
    for relative, text in {
        "WORKSPACE.md": WORKSPACE_GUIDE,
        "skills/README.md": TOOL_GUIDE,
    }.items():
        target = root / relative
        # Preserve resident-authored guides and never follow a file symlink.
        if target.is_symlink():
            continue
        try:
            current = target.read_text() if target.exists() else None
        except (UnicodeError, OSError):
            continue
        if current is None or current.startswith(MARKER):
            if current != text:
                target.write_text(text)
