"""Versioned soft configuration and executable tool definitions, with real reconciliation."""

from __future__ import annotations
import asyncio
import json
import hashlib
import shutil
from pathlib import Path
from backend.skills.base import Skill, SkillResult
from .store import uid
from .workspace import environment

SURFACES = {
    "conversation_prompt",
    "world_prompt",
    "working_procedure",
    "retrieval_settings",
    "tool",
}


class ProcessTool(Skill):
    def __init__(self, definition, workspace):
        self.definition = definition
        self.name = definition["name"]
        self.description = definition["description"]
        self.parameters = definition.get("parameters", {})
        self.workspace = Path(workspace)

    async def execute(self, **kwargs):
        args = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        for index, digest in self.definition.get("command_files", {}).items():
            source = Path(self.definition["command"][int(index)])
            if (
                not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest() != digest
            ):
                return SkillResult(
                    False,
                    error="Versioned tool source changed; test and activate a new revision",
                )
        process = await asyncio.create_subprocess_exec(
            *self.definition["command"],
            cwd=self.workspace,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C.UTF-8",
                **environment(self.workspace),
            },
        )
        try:

            async def bounded(stream):
                chunks = []
                size = 0
                while chunk := await stream.read(65536):
                    size += len(chunk)
                    if size > 1024 * 1024:
                        raise OverflowError("Worker output exceeded the contract limit")
                    chunks.append(chunk)
                return b"".join(chunks)

            async def exchange():
                process.stdin.write(json.dumps(args).encode() + b"\n")
                await process.stdin.drain()
                process.stdin.close()
                output, error = await asyncio.gather(
                    bounded(process.stdout), bounded(process.stderr)
                )
                await process.wait()
                return output, error

            output, error = await asyncio.wait_for(
                exchange(), self.definition.get("timeout", 30)
            )
            if process.returncode:
                return SkillResult(
                    False, error=f"Worker exited with status {process.returncode}"
                )
            value = json.loads(output)
            if not isinstance(value, dict) or not isinstance(
                value.get("success"), bool
            ):
                return SkillResult(
                    False,
                    error="Worker must return a JSON SkillResult with boolean success",
                )
            if (
                not isinstance(value.get("data", {}), dict)
                or value.get("error") is not None
                and not isinstance(value["error"], str)
            ):
                return SkillResult(
                    False, error="Worker data must be an object and error must be text"
                )
            operations = value.get("data", {}).get("world_operations", [])
            if operations and value["success"]:
                from . import get_runtime

                runtime = get_runtime()
                if (
                    not runtime
                    or not self.definition.get("world_effects")
                    or not isinstance(operations, list)
                    or len(operations) > 10
                ):
                    return SkillResult(
                        False,
                        error="World effects need a declared, bounded world_operations contract",
                    )
                applied = []
                with runtime.store.transaction():
                    for operation in operations:
                        if not isinstance(operation, dict):
                            raise ValueError("Invalid world operation")
                        record = runtime.world_action(
                            operation["action"],
                            operation.get("object_id", ""),
                            operation.get("attributes"),
                            operation.get("expected"),
                        )
                        applied.append(record)
                value["data"]["applied_world_operations"] = applied
            return SkillResult(
                value["success"],
                str(value.get("output", "")),
                value.get("error"),
                value.get("data") or {},
            )
        except (asyncio.CancelledError, asyncio.TimeoutError, OverflowError) as exc:
            import os, signal

            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            if isinstance(exc, OverflowError):
                return SkillResult(
                    False, error="Worker output exceeded the contract limit"
                )
            raise
        except (ValueError, TypeError, KeyError):
            return SkillResult(False, error="Worker returned invalid JSON")


class Adaptation:
    def __init__(self, runtime, registry):
        self.runtime = runtime
        self.registry = registry
        self.managed = set()
        self.core_names = set(registry._skills)

    def propose(self, surface, value, intention, checks=None):
        if surface not in SURFACES or not intention.strip():
            raise ValueError("Choose a named mutation surface and an intention")
        if surface == "tool":
            if (
                not isinstance(value, dict)
                or not value.get("name")
                or not value.get("description")
            ):
                raise ValueError("A worker needs a name and description")
            if value["name"] in self.core_names or value["name"] in (
                set(self.registry._skills) - self.managed
            ):
                raise ValueError("A generated tool cannot replace a built-in tool")
            command = value.get("command")
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(a, str) for a in command)
            ):
                raise ValueError("Worker command must be an argv array")
            if not 1 <= value.get("timeout", 30) <= 120:
                raise ValueError("Worker timeout must be 1–120 seconds")
            parameters = value.get("parameters", {})
            if not isinstance(parameters, dict) or any(
                not isinstance(field, dict) for field in parameters.values()
            ):
                raise ValueError(
                    "Worker parameters must map field names to definitions, not a full JSON Schema; read skills/README.md"
                )
        elif surface == "retrieval_settings":
            if not isinstance(value, dict) or any(
                k not in {"limit", "minimum_score"} for k in value
            ):
                raise ValueError(
                    "Only retrieval limit and minimum_score are mutable here"
                )
            if (
                not 1 <= value.get("limit", 5) <= 20
                or not 0 <= value.get("minimum_score", 0) <= 1
            ):
                raise ValueError("Retrieval setting is outside its supported bounds")
        elif not isinstance(value, str) or len(value) > 4000:
            raise ValueError("Prompt fragments must be text of at most 4000 characters")
        proposal_id = uid("proposal")
        if surface == "tool":
            value = dict(value)
            value["command"] = list(value["command"])
            value["command_files"] = {}
            directory = self.runtime.workspace / ".tools" / proposal_id
            for index, argument in enumerate(value["command"]):
                if argument.startswith("-") or len(argument) > 1024:
                    continue
                source = Path(argument)
                if not source.is_absolute():
                    source = self.runtime.workspace / source
                try:
                    source = source.resolve()
                    if (
                        not source.is_relative_to(self.runtime.workspace)
                        or not source.is_file()
                        or ".venv" in source.parts
                    ):
                        continue
                except OSError:
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                destination = directory / (str(index) + "-" + source.name)
                shutil.copy2(source, destination)
                value["command"][index] = str(destination)
                value["command_files"][str(index)] = hashlib.sha256(
                    destination.read_bytes()
                ).hexdigest()
        return self.runtime.store.put(
            "proposal",
            proposal_id,
            self.runtime.scope(),
            {
                "surface": surface,
                "value": value,
                "intention": intention,
                "checks": checks or [],
                "status": "proposed",
            },
        )

    async def test_tool(self, proposal_id, arguments):
        proposal = self.runtime.store.get("proposal", proposal_id)
        if not proposal:
            raise ValueError(
                "test_tool requires an existing proposal_id. Use runtime_change(action='inspect') to find it."
            )
        if proposal["data"]["surface"] != "tool":
            raise ValueError(
                "test_tool applies only to surface='tool'. For a prompt or retrieval change, use evaluate with proposal_id."
            )
        # Tests run in a separate workspace. The authority profile is still reported honestly.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="tool-trial-") as directory:
            from .store import RuntimeStore
            from .service import RuntimeService

            root = Path(directory)
            self.runtime.store.checkpoint(root / "runtime.sqlite3")
            trial_store = RuntimeStore(root / "runtime.sqlite3")
            trial = RuntimeService(
                trial_store,
                root / "workspace",
                self.runtime.config,
                self.runtime.source_root,
            )
            try:
                async with trial.execution("evaluation"):
                    worker = ProcessTool(proposal["data"]["value"], trial.workspace)
                    result = await worker.execute(**arguments)
            finally:
                trial_store.close()
        self.runtime.store.put(
            "tool_trial",
            uid("tooltrial"),
            self.runtime.scope(),
            {
                "proposal_id": proposal_id,
                "success": result.success,
                "result": result.to_dict(),
            },
            category="platform",
        )
        return result.to_dict()

    def activate(self, proposal_id, experiment_id):
        from .change_guidance import proposal_status

        guidance = proposal_status(self.runtime, proposal_id)
        proposal = self.runtime.store.get("proposal", proposal_id)
        if not experiment_id:
            raise ValueError(
                "activate requires experiment_id as well as proposal_id. Next call: "
                + json.dumps(guidance["next_call"])
                + ". "
                + guidance["message"]
            )
        experiment = self.runtime.store.get("experiment", experiment_id)
        if not experiment:
            raise ValueError(
                f"Unknown experiment_id {experiment_id!r}. Supply an experiment-... ID, not a proposal_id or job_id. "
                + "Next call: "
                + json.dumps(guidance["next_call"])
            )
        if experiment["data"].get("proposal_id") != proposal_id:
            raise ValueError(
                f"Experiment {experiment_id} belongs to a different proposal. "
                + "Next call: "
                + json.dumps(guidance["next_call"])
            )
        if experiment["data"].get("decision") != "promote":
            raise ValueError(
                f"Experiment {experiment_id} has overall decision={experiment['data'].get('decision')!r}; activation requires decision='promote'. "
                "Individual scenario scores do not qualify. Next call: "
                + json.dumps(guidance["next_call"])
            )
        data = proposal["data"]
        import hashlib
        from .store import encoded

        candidate_hash = hashlib.sha256(encoded(data).encode()).hexdigest()
        if (
            experiment["data"].get("manifest", {}).get("candidate_hash")
            != candidate_hash
        ):
            raise ValueError(
                "Proposal changed since comparative evaluation. This experiment approves a different version; evaluate the current proposal before activation."
            )
        if data["surface"] == "tool":
            trials = [
                r
                for r in self.runtime.store.list("tool_trial")
                if r["data"]["proposal_id"] == proposal_id and r["data"]["success"]
            ]
            if not trials:
                raise ValueError(
                    "Tool activation requires a successful worker contract test. Run runtime_change(action='test_tool', proposal_id='"
                    + proposal_id
                    + "', arguments={...}) with representative inputs. An existing matching decision=promote does not need to be rerun after a successful test."
                )
        target = data["value"]["name"] if data["surface"] == "tool" else data["surface"]
        old = self.runtime.store.get("runtime_revision", target)
        revision = self.runtime.store.put(
            "runtime_revision",
            target,
            self.runtime.actor_id,
            {
                "surface": data["surface"],
                "value": data["value"],
                "proposal_id": proposal_id,
                "experiment_id": experiment_id,
                "active": True,
            },
            expected=old["revision"] if old else 0,
            category="platform",
        )
        self.reconcile()
        return revision

    def revert(self, target, revision=0):
        history = self.runtime.store.history("runtime_revision", target)
        if revision:
            selected = next(
                (r["data"] for r in history if r["revision"] == revision), None
            )
            if selected is None:
                raise ValueError("Unknown revision")
        else:
            selected = {
                "surface": "tool" if target in self.managed else target,
                "value": None,
                "active": False,
            }
        result = self.runtime.store.put(
            "runtime_revision",
            target,
            self.runtime.actor_id,
            selected,
            category="platform",
        )
        self.reconcile()
        return result

    def reconcile(self):
        active = {
            r["id"]: r["data"]
            for r in self.runtime.store.list("runtime_revision")
            if r["data"].get("active") and r["data"]["surface"] == "tool"
        }
        for name in self.managed - set(active):
            self.registry.unregister(name)
        for name, data in active.items():
            if name in self.core_names:
                raise ValueError("Generated tool collides with a core capability")
            self.registry.register(ProcessTool(data["value"], self.runtime.workspace))
        self.managed = set(active)

    def prompt(self, lane):
        names = {f"{lane}_prompt", "working_procedure"}
        parts = []
        for r in self.runtime.store.list("runtime_revision"):
            if r["id"] in names and r["data"].get("active"):
                parts.append(
                    f"[Practice revision {r['id']}:{r['revision']}]\n{r['data']['value']}"
                )
        return "\n\n".join(parts)
