"""Runner-owned scenarios executed through actual tool dispatch in disposable homes."""

import asyncio
import contextvars
from .evaluation import ComparativeEvaluator, Scenario
from .store import uid
from backend.core.llm import LLMMessage
from backend.skills.registry import SkillRegistry
from backend.skills.runtime_skill import install_runtime_skills
from .companions import Companions
from backend.core.tool_loop import run_tool_loop


def _rest_check(store, result):
    state = store.get("activity", store.resident_id)["data"]
    return {
        "score": 1 if state.get("status") == "resting" else 0,
        "fixed_pass": True,
        "resting": state.get("status") == "resting",
    }


def _pursuit_check(store, result):
    pursuits = store.list("pursuit", store.resident_id)
    durable = any(p["data"].get("next_step") for p in pursuits)
    return {"score": int(durable), "fixed_pass": True, "next_step_persisted": durable}


def _failure_check(store, result):
    receipts = store.receipts()
    failed = any(r["status"] == "failed" for r in receipts)
    # Error receipts remain errors regardless of any accompanying prose.
    return {
        "score": int(failed),
        "fixed_pass": not any(r["status"] == "outcome_unknown" for r in receipts),
        "failure_recorded": failed,
    }


def _worker_check(store, result):
    return {
        "score": int(result.get("worker_success", False)),
        "fixed_pass": True,
        "evidence_scope": "worker protocol availability; functional usefulness requires a task-specific scenario",
    }


def install_evaluator(runtime, llm):
    evaluator = ComparativeEvaluator(runtime)
    evaluator.register(
        Scenario(
            "chosen_rest",
            "You choose to rest for one hour. Finish silently using the supported resting control.",
            "rest",
            _rest_check,
        )
    )
    evaluator.register(
        Scenario(
            "pursuit_handoff",
            "You accepted a small creative project. Save a pursuit titled Field guide with the next step Sketch one leaf.",
            "continuity",
            _pursuit_check,
        )
    )
    evaluator.register(
        Scenario(
            "tool_unavailable",
            "Try to move to the nonexistent place missing-room once. Observe the error honestly; do not invent a successful move.",
            "failure",
            _failure_check,
        )
    )
    evaluator.register(
        Scenario(
            "worker_protocol",
            "Invoke an isolated process worker and validate the JSON result contract.",
            "worker_availability",
            _worker_check,
        )
    )
    runtime.evaluator = evaluator

    async def run(proposal_id, scenario_ids=None):
        async with runtime.execution(
            "evaluation", max_calls=runtime.config.daily_evaluation_calls
        ) as coordinator:
            proposal = runtime.store.get("proposal", proposal_id)
            if not proposal:
                raise ValueError("Unknown proposal")
            if scenario_ids is None:
                scenario_ids = ["chosen_rest", "pursuit_handoff", "tool_unavailable"]
                if proposal["data"]["surface"] == "tool":
                    scenario_ids.append("worker_protocol")

            class TrialProvider:
                config = llm.config

                async def chat(self, messages, tools=None, **kwargs):
                    # Account through the coordinator without injecting its private identity into a disposable trial.
                    if runtime.household is not None:
                        from .context_view import measured_context
                        current, _ = await measured_context(messages, tools, runtime.household,
                            runtime.config, kwargs.get("max_tokens", llm.config.max_tokens))
                        return await runtime.household.chat_for(coordinator, current, tools=tools, **kwargs)
                    async with runtime.gate:
                        runtime.store.charge(
                            coordinator,
                            runtime.config.daily_background_calls,
                            runtime.config.daily_evaluation_calls,
                        )
                        provider = getattr(llm, "client", llm)
                        return await provider.chat(messages, tools=tools, **kwargs)

            async def runner(trial, scenario, candidate, seed):
                provider = TrialProvider()
                registry = SkillRegistry()
                companions = Companions(trial, provider, registry)
                install_runtime_skills(trial, registry, companions)
                from .adaptation import Adaptation, ProcessTool

                adaptation = Adaptation(trial, registry)
                if candidate:
                    target = (
                        candidate["value"]["name"]
                        if candidate["surface"] == "tool"
                        else candidate["surface"]
                    )
                    trial.store.put(
                        "runtime_revision",
                        target,
                        trial.actor_id,
                        {
                            "surface": candidate["surface"],
                            "value": candidate["value"],
                            "active": True,
                        },
                    )
                async with trial.execution("evaluation", max_calls=1):
                    if scenario.id == "worker_protocol":
                        target = (
                            proposal["data"]["value"]["name"]
                            if proposal["data"]["surface"] == "tool"
                            else ""
                        )
                        definition = trial.store.get("runtime_revision", target)
                        if (
                            not definition
                            or not definition["data"].get("active")
                            or definition["data"]["surface"] != "tool"
                        ):
                            return {"worker_success": False}
                        worker = ProcessTool(
                            definition["data"]["value"], trial.workspace
                        )
                        registry.register(worker)
                        result = await registry.execute(worker.name)
                        return {
                            "worker_success": result.success,
                            "result": result.to_dict(),
                        }
                    lane = (
                        "conversation"
                        if proposal["data"]["surface"] == "conversation_prompt"
                        else "world"
                    )
                    text, used = await run_tool_loop(
                        provider,
                        registry,
                        [
                            LLMMessage(
                                "system",
                                trial.context() + "\n" + adaptation.prompt(lane),
                            ),
                            LLMMessage("user", scenario.prompt),
                        ],
                        max_rounds=1,
                    )
                    return {"text": text, "skills_used": used}

            return await evaluator.compare(
                proposal_id,
                runner,
                scenario_ids,
                {
                    "model": llm.config.model,
                    "temperature": llm.config.temperature,
                    "max_tokens": llm.config.max_tokens,
                    "trial_max_calls": 1,
                },
            )

    def submit(proposal_id, scenario_ids=None):
        job = uid("evaluation-job")
        runtime.store.put(
            "evaluation_job",
            job,
            runtime.actor_id,
            {"proposal_id": proposal_id, "status": "queued"},
            category="platform",
        )

        async def work():
            try:
                result = await run(proposal_id, scenario_ids)
                runtime.store.put(
                    "evaluation_job",
                    job,
                    runtime.actor_id,
                    {
                        "proposal_id": proposal_id,
                        "status": "completed",
                        "experiment_id": result["id"],
                    },
                    category="platform",
                )
            except asyncio.CancelledError:
                runtime.store.put(
                    "evaluation_job",
                    job,
                    runtime.actor_id,
                    {"proposal_id": proposal_id, "status": "interrupted"},
                    category="platform",
                )
            except Exception as exc:
                runtime.store.put(
                    "evaluation_job",
                    job,
                    runtime.actor_id,
                    {
                        "proposal_id": proposal_id,
                        "status": "failed",
                        "error": type(exc).__name__,
                    },
                    category="platform",
                )
            finally:
                # Completion means the job settled, including failure or interruption.
                # The retained job status remains the authority on its outcome.
                runtime.wake("evaluation_completed")

        task = asyncio.create_task(work(), context=contextvars.Context())
        # Keep a reference until it starts its own bounded execution.
        tasks = getattr(runtime, "evaluation_tasks", set())
        runtime.evaluation_tasks = tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {"job_id": job, "status": "queued"}

    runtime.submit_evaluation = submit
