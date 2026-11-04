"""Small explicit affordances; trusted actor scope comes from execution context."""

from __future__ import annotations
import json
from datetime import datetime
from backend.skills.base import Skill, SkillResult
from backend.core.runtime.contracts import current_execution


class RuntimeSkill(Skill):
    def __init__(self, runtime, name, description, parameters, handler):
        self.runtime = runtime
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    async def execute(self, **kwargs):
        args = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        result = self.handler(**args)
        if hasattr(result, "__await__"):
            result = await result
        return SkillResult(
            True,
            json.dumps(result, ensure_ascii=False, default=str),
            data={"value": result},
        )


def param(kind, description="", required=False, enum=None):
    value = {"type": kind, "description": description, "required": required}
    if enum:
        value["enum"] = enum
    return value


def install_runtime_skills(runtime, registry, companions):
    def install(name, description, parameters, handler):
        registry.register(RuntimeSkill(runtime, name, description, parameters, handler))

    install(
        "inspect_receipt",
        "Inspect a retained tool receipt a page at a time. Context previews omit large output; this reads its original output, error, data or arguments without retrying the operation.",
        {
            "receipt_id": param("string", required=True),
            "section": param("string", enum=["output", "error", "data", "arguments"]),
            "offset": param("integer"),
            "limit": param("integer", "1–65536 characters; default 8192. Read larger sources in pages."),
        },
        runtime.inspect_receipt,
    )
    install(
        "name_stage",
        "Optionally name your current growth stage in one to three words. The platform computes your numeric level; naming does not grant points or change identity documents.",
        {"name": param("string", required=True)},
        runtime.growth.name_stage,
    )
    install(
        "world_action",
        "Inspect or change persistent simulated places and objects. These are simulation events, not physical host actions.",
        {
            "action": param(
                "string",
                required=True,
                enum=["inspect", "move", "create", "revise", "place"],
            ),
            "object_id": param("string"),
            "attributes": param("object"),
            "expected": param("integer"),
        },
        runtime.world_action,
    )
    install(
        "pursuit",
        "Keep a chosen interest or accepted commitment across sessions. Pause, abandon, or renegotiate honestly; creative play needs no completion test.",
        {
            "action": param("string", required=True, enum=["list", "create", "revise"]),
            "pursuit_id": param("string"),
            "attributes": param("object"),
            "expected": param("integer"),
        },
        runtime.pursuit,
    )

    def end_turn(status="completed", wake_at=None, wake_event=None):
        if isinstance(wake_at, str):
            wake_at = datetime.fromisoformat(wake_at).timestamp()
        return runtime.end_turn(status, wake_at, wake_event)

    install(
        "end_turn",
        "Finish silently without a report. Resting defaults to one hour. Wait for a named event or UTC ISO time. Must be the only tool in its batch.",
        {
            "status": param(
                "string",
                enum=["completed", "resting", "waiting", "declined", "blocked"],
            ),
            "wake_at": param("string", "ISO8601 with timezone"),
            "wake_event": param("string"),
        },
        end_turn,
    )
    install(
        "remember",
        "Keep your own practical lesson, fallible belief, preference, question, or imaginative story. Evidence references do not make an interpretation verified.",
        {
            "text": param("string", required=True),
            "kind": param(
                "string",
                enum=["lesson", "belief", "preference", "question", "imagination"],
            ),
            "evidence": param("array"),
        },
        runtime.remember,
    )
    install(
        "share_note",
        "Explicitly publish a note or selected personal memory into the household notebook, with your authorship.",
        {"text": param("string", required=True), "source_ids": param("array")},
        runtime.share,
    )
    install(
        "register_artifact",
        "Register a regular workspace file with its current content hash for future reuse and backup.",
        {"path": param("string", required=True)},
        runtime.register_artifact,
    )
    install(
        "revise_identity",
        "Read or revise one of your core identity documents with a reason and expected revision. A lesson does not require an identity change.",
        {
            "name": param(
                "string", required=True, enum=["IDENTITY.md", "SOUL.md", "USER.md"]
            ),
            "text": param("string"),
            "expected": param("integer"),
            "reason": param("string"),
            "sources": param("array"),
        },
        runtime.identity,
    )

    def companion(action, actor_id="", name="", seed="", changes=None, expected=None):
        context = current_execution.get()
        if action == "self":
            return runtime.store.get("actor", runtime.scope())
        if action == "revise_self":
            return companions.revise(runtime.scope(), changes or {}, expected)
        if context and context.depth:
            raise ValueError("Only the primary resident manages the companion roster")
        if action == "list":
            return companions.list()
        if action == "create":
            return companions.create(name, seed)
        if action == "archive":
            return companions.revise(actor_id, {"status": "archived"}, expected)
        raise ValueError("Unknown companion operation")

    install(
        "companion",
        "Create a lasting companion, inspect your roster or self, revise your own self-description, or archive a companion without deleting history.",
        {
            "action": param(
                "string",
                required=True,
                enum=["list", "create", "self", "revise_self", "archive"],
            ),
            "actor_id": param("string"),
            "name": param("string"),
            "seed": param("string"),
            "changes": param("object"),
            "expected": param("integer"),
        },
        companion,
    )
    install(
        "consult_companion",
        "Invite a named companion into a bounded conversation or specialist task. Personal memories remain individual; results are attributed.",
        {
            "actor_id": param("string", required=True),
            "content": param("string", required=True),
            "mode": param("string", enum=["conversation", "task"]),
            "thread_id": param("string"),
            "allowed_tools": param("array"),
        },
        companions.consult,
    )


def install_development_skills(runtime, registry):
    import asyncio

    def install(name, description, parameters, handler):
        registry.register(RuntimeSkill(runtime, name, description, parameters, handler))

    async def runtime_change(
        action,
        proposal_id="",
        surface="",
        value="",
        intention="",
        experiment_id="",
        target="",
        revision=0,
        arguments=None,
    ):
        context = current_execution.get()
        if context and context.depth:
            raise ValueError("Only the primary resident manages runtime experiments")
        if action == "inspect":
            return {
                "proposals": runtime.store.list("proposal"),
                "active": runtime.store.list("runtime_revision"),
                "experiments": runtime.store.list("experiment"),
            }
        if action == "propose":
            return runtime.adaptation.propose(surface, value, intention)
        if action == "test_tool":
            return await runtime.adaptation.test_tool(proposal_id, arguments or {})
        if action == "evaluate":
            return runtime.submit_evaluation(proposal_id)
        if action == "activate":
            return runtime.adaptation.activate(proposal_id, experiment_id)
        if action == "revert":
            return runtime.adaptation.revert(target, revision)
        raise ValueError("Unknown runtime experiment operation")

    install(
        "runtime_change",
        "Inspect, propose, test, evaluate, activate or revert a versioned runtime experiment. Evaluation is queued separately and retains comparison evidence.",
        {
            "action": param(
                "string",
                required=True,
                enum=[
                    "inspect",
                    "propose",
                    "test_tool",
                    "evaluate",
                    "activate",
                    "revert",
                ],
            ),
            "proposal_id": param("string"),
            "surface": param(
                "string",
                enum=[
                    "conversation_prompt",
                    "world_prompt",
                    "working_procedure",
                    "retrieval_settings",
                    "tool",
                ],
            ),
            "value": {
                "type": ["string", "object"],
                "description": "Prompt text, retrieval configuration, or process tool definition",
            },
            "intention": param("string"),
            "experiment_id": param("string"),
            "target": param("string"),
            "revision": param("integer"),
            "arguments": param("object"),
        },
        runtime_change,
    )

    async def core_change(action, candidate_id="", intention=""):
        context = current_execution.get()
        if context and context.depth:
            raise ValueError("Only the primary resident manages application releases")
        if action == "inspect":
            return {
                "deployments": runtime.store.list("deployment"),
                "automatic": runtime.config.automatic_deployment,
            }
        if action == "prepare":
            return await asyncio.to_thread(runtime.deployments.prepare, intention)
        if action == "build":
            return await asyncio.to_thread(runtime.deployments.build, candidate_id)
        if action == "activate":
            result = runtime.deployments.stage_activation(candidate_id)
            from backend.core.runtime.contracts import current_terminal

            current_terminal.set({"status": "completed", "handoff": result})
            return result
        raise ValueError("Unknown core-change operation")

    install(
        "core_change",
        "Prepare an isolated application worktree, build and validate its exact release, and activate through the supervisor after recovery qualification. Code activation restarts the process; save a handoff first.",
        {
            "action": param(
                "string",
                required=True,
                enum=["inspect", "prepare", "build", "activate"],
            ),
            "candidate_id": param("string"),
            "intention": param("string"),
        },
        core_change,
    )
