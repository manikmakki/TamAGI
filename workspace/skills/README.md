# Tool source and legacy custom skills

This repository directory is not necessarily the active resident's workspace. Use the workspace path reported by the runtime, rather than assuming `/opt/TamAGI/workspace`, `/workspace`, or `~/workspace` is the current home.

For a home-platform instance, keep editable tool source in `skills/<tool_name>/` beneath its active workspace. Tools are JSON stdin/stdout subprocesses activated through `runtime_change`: **propose → test_tool → evaluate → activate**. Dropping a Python file here, restarting, or registering it as an artifact does not make it callable. Platform startup deliberately skips in-process custom imports.

See [workspace paths and tool development](../../docs/workspace-layout.md) for the supported contract, versioning, dependencies, and the distinction between creations and tools.

Legacy non-platform configurations retain startup discovery of `.py` files that subclass `backend.skills.base.Skill` in their configured custom-skill directory. Their `parameters` use a mapping of argument names to definitions, with `required: True` on individual arguments. That compatibility mechanism is separate from the home-platform tool lifecycle.
