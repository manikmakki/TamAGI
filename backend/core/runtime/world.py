"""World Thread adapter: cognition is optional; narrative is never a commit prerequisite."""

from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from .contracts import BudgetExhausted, current_terminal
from .store import uid
from backend.core.llm import LLMMessage
from backend.core.world_state import WorldState, parse_new_state
from backend.core.tool_loop import run_tool_loop


async def tick(world, *, manual=False):
    runtime = world.agent.runtime
    if hasattr(runtime, "deployments"):
        runtime.deployments.synchronize()
    if getattr(world.agent.identity, "needs_onboarding", False):
        return {
            "status": "waiting",
            "detail": "The first introduction is pending",
            "reason": "onboarding",
            "activation_started": False,
        }
    if not runtime.due(ignore_decompression=manual, ignore_error_backoff=manual):
        state = runtime.activity_state()
        now = runtime.store.clock()
        if getattr(runtime, "interactions", None) and runtime.interactions.has_pending():
            reason = "user_work"
            detail = "Accepted conversational work is still running or queued."
            next_wake = None
        elif state.get("wait_reason") in {
            "context_limit",
            "request_rejected",
            "provider_unavailable",
            "runtime_unavailable",
            "generation_truncated",
        }:
            reason = state["wait_reason"]
            detail = (
                state.get("wait_detail")
                or "Waiting to retry a failed autonomous activation."
            )
            next_wake = state.get("wake_at")
        elif state.get("wake_at") and state["wake_at"] > now:
            reason = "sleep" if state["status"] == "resting" else "timed_wait"
            detail = "The chosen rest or timed wait has not ended."
            next_wake = state["wake_at"]
        elif state.get("wake_event"):
            reason = "event_wait"
            detail = "Waiting for " + state["wake_event"].replace("_", " ") + "."
            next_wake = None
        else:
            reason = "decompression"
            detail = "The post-conversation pause is still in effect."
            next_wake = runtime.store.meta("decompression_until")
        return {
            "status": state["status"] if reason not in {"decompression", "user_work"} else "waiting",
            "location": state.get("location", "home"),
            "next_wake_at": next_wake,
            "reason": reason,
            "detail": detail,
            "activation_started": False,
        }
    if world._tick_running:
        return {
            "status": "waiting",
            "detail": "An activation is already running",
            "reason": "busy",
            "activation_started": False,
        }
    world._tick_running = True
    pending = []
    try:
        async with runtime.execution("autonomy") as execution:
            await world.agent.flush_unsummarized_conversations()
            pending = world._pending_world_events[:]
            runtime.reconcile_identity()
            world.agent.personality.state.decay()
            prompt = (
                runtime.context()
                + "\n"
                + world.agent.identity.get_system_prompt_context()
            )

            history = [
                r["data"]
                for r in runtime.store.list("world_turn", runtime.actor_id, limit=10000)
            ][-5:]
            from .context_view import receipt_view

            receipts = [
                receipt_view(r)
                for r in runtime.store.receipts(limit=100)
                if r["actor"] == runtime.actor_id
            ][:8]
            messages = [LLMMessage("system", prompt)]
            import json

            messages.append(
                LLMMessage(
                    "user",
                    "Autonomous opportunity; no new user task has been assigned. "
                    "You may choose an activity, continue a pursuit, play, or end silently.\n"
                    + json.dumps(
                        {
                            "now": datetime.now(timezone.utc).isoformat(),
                            "recent_turns": history,
                            "recent_receipts": receipts,
                            "visitor_events": pending,
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            text, used = await run_tool_loop(
                world.agent.llm,
                world.agent.skills,
                messages,
                max_rounds=runtime.config.background_calls_per_activation,
                is_autonomous=True,
            )
            terminal = current_terminal.get()
            run = runtime.store.run(execution.execution_id)
            status = (
                terminal["status"]
                if terminal
                else (
                    "budget_exhausted"
                    if run["calls"] >= run["max_calls"] and not text
                    else "completed"
                )
            )
            state = runtime.store.get("activity", runtime.actor_id)
            data = state["data"]
            # Legacy formatted prose may supply a subjective mood/focus, never an unexecuted move.
            parsed = parse_new_state(text) if text else None
            if parsed:
                data = {**data, "mood": parsed.mood, "focus": parsed.focus}
            if not terminal or terminal.get("status") == "budget_exhausted":
                data = {
                    **data,
                    "status": status,
                    "wake_at": None,
                    "wake_event": None,
                    "wait_reason": None,
                    "wait_detail": None,
                }
                runtime.store.put("activity", runtime.actor_id, runtime.actor_id, data)
            now = datetime.now(timezone.utc).isoformat()
            location = runtime.store.get("world_object", data.get("location", "home"))
            location_name = location["data"]["name"] if location else "Home"
            projection = WorldState(
                timestamp=now,
                last_tick=now,
                location=location_name,
                mood=data.get("mood", ""),
                focus=data.get("focus", ""),
                available_actions=[
                    "Choose an activity",
                    "Continue a pursuit",
                    "Converse with a companion",
                    "Rest",
                ],
                raw_state_block=f"[New State]\nLocation/Setting: {location_name}\nInternal State/Mood: {data.get('mood', '')}\nCurrent Focus: {data.get('focus', '')}\nAvailable Actions: Choose freely, continue a pursuit, or rest.",
            )
            result = {
                "status": status,
                "execution_id": execution.execution_id,
                "activation_started": True,
                "location": location_name,
                "mood": data.get("mood", ""),
                "text": text,
                "receipts": [
                    r["id"] for r in runtime.store.receipts(execution.execution_id)
                ],
                "next_wake_at": data.get("wake_at"),
            }
            remaining_events = world._pending_world_events[len(pending) :]
            with runtime.store.transaction():
                runtime.store.put(
                    "world_turn",
                    uid("turn"),
                    runtime.actor_id,
                    result,
                    category="imagination",
                )
                runtime.store.put(
                    "world_projection",
                    "current",
                    runtime.actor_id,
                    projection.to_dict(),
                    category="interpretation",
                )
                runtime.store.event(
                    runtime.actor_id,
                    "world_turn_completed",
                    {
                        "execution_id": execution.execution_id,
                        "status": status,
                        "receipts": result["receipts"],
                    },
                )
                runtime.store.set_meta("provider_failures", 0)
                if pending:
                    runtime.store.put(
                        "pending_events",
                        "world",
                        runtime.actor_id,
                        {"events": remaining_events},
                        category="platform",
                    )
            if pending:
                world._pending_world_events = remaining_events
            try:
                world._state_store.save(projection)
            except OSError as exc:
                runtime.store.event(
                    runtime.actor_id,
                    "projection_unavailable",
                    {"projection": "world_state", "error": type(exc).__name__},
                )
            if text:
                world._append_to_thread(
                    "An autonomous opportunity, not a user assignment.", text
                )
            if pending:
                try:
                    world._save_pending_world_events()
                except OSError as exc:
                    runtime.store.event(
                        runtime.actor_id,
                        "projection_unavailable",
                        {"projection": "pending_events", "error": type(exc).__name__},
                    )
            if world.monologue_log:
                world.monologue_log.append(
                    type="world_activity",
                    source="autonomous",
                    title=f"{location_name} — {status}",
                    content=text,
                    metadata={
                        "status": status,
                        "execution_id": execution.execution_id,
                        "receipts": result["receipts"],
                        "next_wake_at": data.get("wake_at"),
                        "wake_event": data.get("wake_event"),
                        "wait_reason": data.get("wait_reason"),
                        "timezone": runtime.config.timezone,
                    },
                )
            if world.agent.consolidation:
                # Reflection shares the same budget and may legitimately produce no learning.
                if run["calls"] < run["max_calls"]:
                    try:
                        await world.agent.consolidation.maybe_consolidate()
                    except Exception as exc:
                        runtime.store.event(
                            runtime.actor_id,
                            "reflection_deferred",
                            {"error": type(exc).__name__},
                        )
            world.agent.personality.save_state()
            return result
    except BudgetExhausted as exc:
        return {
            "status": "budget_exhausted",
            "detail": str(exc) + ". Completed effects remain recorded; work can be reconsidered at a later opportunity.",
        }
    except asyncio.CancelledError:
        return {
            "status": "interrupted",
            "detail": "User interaction takes priority; completed effects remain recorded",
        }
    except Exception as exc:
        from .service import failure_reason

        reason, detail = failure_reason(exc)
        failures = runtime.store.meta("provider_failures", 0) + 1
        runtime.store.set_meta("provider_failures", failures)
        old = runtime.store.get("activity", runtime.actor_id)
        runtime.store.put(
            "activity",
            runtime.actor_id,
            runtime.actor_id,
            {
                **old["data"],
                "status": "waiting",
                "wake_at": runtime.store.clock()
                + min(3600, 30 * 2 ** min(failures - 1, 7)),
                "wake_event": None,
                "wait_reason": reason,
                "wait_detail": detail,
            },
            category="platform",
        )
        runtime.store.event(
            runtime.actor_id,
            "autonomy_unavailable",
            {"error_class": type(exc).__name__, "attempt": failures, "reason": reason},
        )
        return {
            "status": "waiting",
            "detail": detail + " A bounded retry is scheduled.",
            "reason": reason,
        }
    finally:
        world._tick_running = False
