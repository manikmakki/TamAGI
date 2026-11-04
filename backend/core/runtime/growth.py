"""Observable behavioral growth, ADR-0015. No grading inference or resource rewards."""

from __future__ import annotations

import hashlib
import json
from math import isqrt
from dataclasses import replace

from .contracts import current_execution
from .store import encoded

RULE_VERSION = "adr-0015.v1"
EPISODE_GAP = 2 * 60 * 60


def _number(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _message_id(message, index):
    return str(
        message.get("id")
        or hashlib.sha256(
            encoded(
                {
                    "index": index,
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "timestamp": message.get("timestamp"),
                }
            ).encode()
        ).hexdigest()[:24]
    )


class GrowthService:
    def __init__(self, runtime):
        self.runtime = runtime
        self.store = runtime.store
        self.actor = runtime.actor_id
        with self.store.transaction():
            if self.store.meta("growth_legacy_experience") is None:
                old = self.store.get("personality", "primary")
                data = old["data"] if old else {}
                self.store.set_meta(
                    "growth_legacy_experience", data.get("experience", 0)
                )
                self.store.set_meta(
                    "growth_legacy_stage", data.get("current_stage_name", "egg")
                )
                self.store.set_meta(
                    "growth_legacy_stage_history", data.get("stage_history", [])
                )

    def _evaluating(self):
        context = current_execution.get()
        if not context:
            return False
        if context.origin == "evaluation" or context.evaluation:
            return True
        run = self.store.run(context.execution_id)
        seen = set()
        while run and run["id"] not in seen:
            seen.add(run["id"])
            payload = json.loads(run.get("payload") or "{}")
            if run["origin"] == "evaluation" or payload.get("evaluation"):
                return True
            run = self.store.run(run["parent"]) if run.get("parent") else None
        return False

    @staticmethod
    def _episodes(conversation_id, messages, created_at=0):
        """Pair actual final responses; retain a pending user message for binding."""
        episodes = []
        pending = None
        episode = None
        previous_end = None
        for index, message in enumerate(messages):
            role = message.get("role")
            metadata = message.get("metadata") or {}
            if role == "user":
                stamp = _number(message.get("timestamp"), created_at)
                if episode is None or (
                    previous_end is not None and stamp - previous_end >= EPISODE_GAP
                ):
                    first_id = _message_id(message, index)
                    episode = {
                        "id": f"conversation:{conversation_id}:{first_id}",
                        "started_at": stamp,
                        "pairs": [],
                        "first_message_id": first_id,
                    }
                    episodes.append(episode)
                pending = (message, _message_id(message, index), stamp)
            elif role == "assistant" and pending:
                if metadata.get("intermediate") or message.get("tool_calls"):
                    continue
                user, user_id, start = pending
                pending = None
                content = str(message.get("content") or "").strip()
                finished = _number(message.get("timestamp"), start)
                previous_end = finished
                eligible = (
                    str(user.get("content") or "").strip()
                    and content
                    and content != "..."
                    and metadata.get("growth_eligible", True)
                    and metadata.get("growth_completed", True)
                    and not metadata.get("evaluation")
                    and not metadata.get("error")
                    and metadata.get("origin") != "evaluation"
                )
                if eligible:
                    episode["pairs"].append(
                        {
                            "user_id": user_id,
                            "assistant_id": _message_id(message, index),
                            "finished_at": finished,
                        }
                    )
        return episodes

    def _occasion(self, context=None):
        context = context or current_execution.get()
        if not context or self._evaluating():
            return None
        return self._run_occasion(context.execution_id)

    def _run_occasion(self, run_id):
        run = self.store.run(run_id)
        seen = set()
        chain = []
        while run and run["id"] not in seen:
            seen.add(run["id"])
            data = json.loads(run.get("payload") or "{}")
            if (
                run["origin"] not in {"user", "autonomy", "companion"}
                or data.get("evaluation")
                or data.get("eligible_live") is False
            ):
                return None
            chain.append((run, data))
            if not run.get("parent"):
                break
            run = self.store.run(run["parent"])
        if not chain:
            return None
        root, root_data = chain[-1]
        if root["actor"] != self.actor:
            return None
        for item, data in chain:
            if data.get("eligible_live") is False:
                continue
            if data.get("occasion_id") and data.get("eligible_live"):
                stored = self.store.get("growth_occasion", data["occasion_id"])
                return {
                    "id": data["occasion_id"],
                    "started_at": stored["data"]["started_at"]
                    if stored
                    else root["created"],
                    "conversation_id": data.get("conversation_id"),
                }
        if root["origin"] == "autonomy":
            return {
                "id": "autonomy:" + root["id"],
                "started_at": root["created"],
                "conversation_id": None,
            }
        return None

    def bind_conversation(self, conversation_id, messages):
        context = current_execution.get()
        if not context or self._evaluating():
            return None
        run = self.store.run(context.execution_id)
        if not run:
            return None
        # Calls made within an autonomous activation retain its single occasion.
        current = self._run_occasion(context.execution_id)
        root = self.store.run(context.root_id)
        if root and root["origin"] == "autonomy":
            return current["id"] if current else None
        episodes = self._episodes(conversation_id, messages, run["created"])
        if not episodes:
            return None
        occasion = episodes[-1]
        with self.store.transaction():
            if not self.store.get("growth_occasion", occasion["id"]):
                self.store.put(
                    "growth_occasion",
                    occasion["id"],
                    self.actor,
                    {
                        "started_at": occasion["started_at"],
                        "conversation_id": conversation_id,
                    },
                    category="platform",
                )
            self.store.set_run_provenance(
                context.execution_id,
                occasion_id=occasion["id"],
                conversation_id=conversation_id,
                eligible_live=True,
                evaluation=False,
            )
        current_execution.set(
            replace(
                context, conversation_id=conversation_id, occasion_id=occasion["id"]
            )
        )
        return occasion["id"]

    def _award(
        self, pattern, occasion_id, tier, points, source_refs, observed_at, reason
    ):
        if self._evaluating():
            return False
        key = hashlib.sha256(
            encoded([RULE_VERSION, self.actor, pattern, occasion_id, tier]).encode()
        ).hexdigest()
        with self.store.transaction():
            if self.store.get("growth_recognition", key):
                return False
            self.store.put(
                "growth_recognition",
                key,
                self.actor,
                {
                    "rule_version": RULE_VERSION,
                    "occasion_id": occasion_id,
                    "pattern": pattern,
                    "tier": tier,
                    "points": points,
                    "source_refs": list(dict.fromkeys(source_refs)),
                    "observed_at": observed_at,
                    "recognized_at": self.store.clock(),
                    "reason": reason,
                },
                expected=0,
                category="platform",
            )
        return True

    def _engagement(self, occasion_id, pairs, source_prefix):
        for threshold, tier, points in ((3, "participation", 1), (8, "sustained", 2)):
            if len(pairs) < threshold:
                continue
            chosen = pairs[:threshold]
            refs = [
                source_prefix + ":" + key
                for pair in chosen
                for key in (pair["user_id"], pair["assistant_id"])
            ]
            self._award(
                "engagement",
                occasion_id,
                tier,
                points,
                refs,
                chosen[-1]["finished_at"],
                "Stayed with a reciprocal conversation for "
                + str(threshold)
                + " completed exchanges.",
            )

    def observe_conversation(self, conversation):
        if self._evaluating():
            return
        record = self.store.get("conversation", conversation.get("id", ""))
        if not record or record["actor"] != self.actor:
            return
        data = record["data"]
        if data.get("evaluation") or data.get("origin") == "evaluation":
            return
        for episode in self._episodes(
            record["id"],
            data.get("messages", []),
            data.get("created_at", record["created"]),
        ):
            self._engagement(
                episode["id"],
                episode["pairs"],
                "conversation:" + record["id"] + ":message",
            )

    def observe_companion(self, thread_id, invocation, caller_actor_id):
        if caller_actor_id != self.actor or self._evaluating():
            return
        record = self.store.get("invocation", invocation.get("invocation_id", ""))
        if not record:
            return
        data = record["data"]
        if not self._valid_invocation(data):
            return
        occasion = data.get("occasion_id")
        if not occasion:
            return
        pairs = []
        for item in self.store.list("invocation", limit=1000000):
            value = item["data"]
            if (
                value.get("thread_id") != thread_id
                or value.get("occasion_id") != occasion
                or not self._valid_invocation(value)
            ):
                continue
            pairs.append(
                {
                    "user_id": item["id"] + ":caller",
                    "assistant_id": item["id"] + ":reply",
                    "finished_at": value.get("completed_at", item["updated"]),
                }
            )
        pairs.sort(key=lambda item: item["finished_at"])
        self._engagement(occasion, pairs, "invocation")

    def _valid_invocation(self, data):
        occasion = self._run_occasion(data.get("invocation_id"))
        return (
            occasion is not None
            and occasion["id"] == data.get("occasion_id")
            and data.get("mode") == "conversation"
            and data.get("caller_actor_id") == self.actor
            and data.get("growth_eligible", False)
            and data.get("status") == "completed"
            and bool(str(data.get("content") or "").strip())
            and bool(str(data.get("text") or "").strip())
            and not data.get("evaluation")
        )

    @staticmethod
    def _pursuit_value(receipt):
        result = receipt["payload"].get("result") or {}
        value = (result.get("data") or {}).get("value")
        if not value:
            try:
                value = json.loads(result.get("output", ""))
            except (ValueError, TypeError):
                return None
        if isinstance(value, dict) and value.get("kind") == "pursuit":
            return value
        return None

    def observe_receipt(self, receipt):
        if self._evaluating():
            return
        source = self.store.receipt(receipt.get("id", ""))
        if (
            not source
            or source["actor"] != self.actor
            or source["status"] != "succeeded"
        ):
            return
        if source["payload"].get("eligible_live") is False or source["payload"].get(
            "evaluation"
        ):
            return
        occasion = self._run_occasion(source["run"])
        if not occasion:
            return
        payload = source["payload"]
        repair = payload.get("repair_of_receipt_id")
        if repair:
            prior = self.store.receipt(repair)
            if (
                prior
                and prior["actor"] == self.actor
                and prior["tool"] == source["tool"]
                and prior["status"] == "failed"
                and self._receipt_precedes(prior["id"], source["id"])
                and not (prior["payload"].get("result", {}).get("data") or {}).get(
                    "dispatch_rejected"
                )
                and prior["payload"].get("arguments_hash")
                != payload.get("arguments_hash")
                and prior["payload"].get("eligible_live") is not False
                and self._run_occasion(prior["run"]) is not None
            ):
                self._award(
                    "adaptation",
                    occasion["id"],
                    "correction",
                    1,
                    [prior["id"], source["id"]],
                    source["finished"],
                    "Connected a failed operation to a changed approach that reported success.",
                )
        # Continuity comes from the supported pursuit mutation, not an arbitrary
        # worker returning a plausible record in its output.
        action = payload.get("arguments", {}).get("action")
        native_pursuit = source["tool"] == "pursuit" and action == "revise"
        graph_adapter = source["tool"] == "world_graph" and action == "update_node"
        if not (native_pursuit or graph_adapter):
            return
        value = self._pursuit_value(source)
        if (
            not value
            or value.get("actor") != self.actor
            or value.get("revision", 0) < 2
        ):
            return
        history = self.store.history("pursuit", value["id"])
        actual = next((r for r in history if r["revision"] == value["revision"]), None)
        if (
            not actual
            or actual["actor"] != self.actor
            or actual["data"] != value.get("data")
        ):
            return
        earlier = next(
            (r for r in history if r["revision"] == value["revision"] - 1), None
        )
        if not earlier or not history:
            return
        existed_before = history[0]["created"] < occasion["started_at"]
        if not existed_before:
            # Equal or adjusted clocks do not erase causality recorded in revisions
            # and distinct trusted executions.
            for previous in self.store.receipts(limit=1000000):
                if previous["status"] != "succeeded" or previous["actor"] != self.actor:
                    continue
                old_value = self._pursuit_value(previous)
                if (
                    not old_value
                    or old_value.get("id") != value["id"]
                    or old_value.get("revision", 0) >= value["revision"]
                ):
                    continue
                old_occasion = self._run_occasion(previous["run"])
                if old_occasion and old_occasion["id"] != occasion["id"]:
                    existed_before = True
                    break
        if not existed_before:
            return
        before, after = earlier["data"], value.get("data", {})
        step = after.get("next_step")
        changed_step = (
            isinstance(step, str)
            and bool(step.strip())
            and step != before.get("next_step")
        )

        def references(data, key):
            value = data.get(key)
            return (
                {r for r in value if isinstance(r, str)}
                if isinstance(value, list)
                else set()
            )

        new_sources = references(after, "evidence") - references(before, "evidence")
        new_artifacts = references(after, "artifacts") - references(before, "artifacts")
        valid_sources = [ref for ref in new_sources if self._source_exists(ref)]
        valid_artifacts = [
            ref
            for ref in new_artifacts
            if (self.store.get("artifact", ref) or {}).get("actor") == self.actor
        ]
        if changed_step or valid_sources or valid_artifacts:
            self._award(
                "continuity",
                occasion["id"],
                "return",
                1,
                [
                    source["id"],
                    f"pursuit:{value['id']}:revision:{earlier['revision']}",
                    f"pursuit:{value['id']}:revision:{value['revision']}",
                    *valid_sources,
                    *valid_artifacts,
                ],
                source["finished"],
                "Returned to an existing pursuit and carried its next step or evidence forward.",
            )

    def _receipt_precedes(self, earlier_id, later_id):
        with self.store.lock:
            rows = self.store.db.execute(
                "SELECT id,rowid FROM receipts WHERE id IN (?,?)",
                (earlier_id, later_id),
            ).fetchall()
            positions = {row["id"]: row["rowid"] for row in rows}
            return len(positions) == 2 and positions[earlier_id] < positions[later_id]

    def _source_exists(self, reference):
        row = self.store.receipt(reference) or self.store.get_event(reference)
        if row:
            return row["actor"] == self.actor
        return any(
            self.store.get(kind, reference)
            for kind in ("shared_note", "memory", "learning")
        )

    def replay(self):
        """Recognize supported current-resident history without inventing missing provenance."""
        if self._evaluating():
            return self.snapshot()
        for record in self.store.list("conversation", self.actor, limit=1000000):
            self.observe_conversation(record["data"])
        for record in self.store.list("invocation", limit=1000000):
            value = record["data"]
            self.observe_companion(
                value.get("thread_id"), value, value.get("caller_actor_id")
            )
        for receipt in reversed(self.store.receipts(limit=1000000)):
            self.observe_receipt(receipt)
        return self.snapshot()

    def snapshot(self):
        with self.store.lock:
            entries = self.store.list("growth_recognition", self.actor, limit=1000000)
            total = sum(int(row["data"]["points"]) for row in entries)
            level = 1 + isqrt(total // 5)
            previous, following = 5 * (level - 1) ** 2, 5 * level**2
            named = self.store.get("growth_stage", f"{self.actor}:{level}")
            stage = (
                named["data"]["name"]
                if named
                else (
                    self.store.meta("growth_legacy_stage", "egg")
                    if level == 1
                    else f"Stage {level}"
                )
            )
            history = []
            for record in self.store.list("growth_stage", self.actor, limit=100000):
                history.extend(
                    {
                        **revision["data"],
                        "revision": revision["revision"],
                        "timestamp": revision["created"],
                    }
                    for revision in self.store.history("growth_stage", record["id"])
                )
            history.sort(
                key=lambda item: (item["timestamp"], item["level"], item["revision"])
            )
            return {
                "actor_id": self.actor,
                "rule_version": RULE_VERSION,
                "total": total,
                "level": level,
                "stage": stage,
                "stage_history": history,
                "previous_threshold": previous,
                "next_threshold": following,
                "points_to_next": following - total,
                "progress": (total - previous) / (following - previous),
                "age_seconds": max(
                    0,
                    self.store.clock()
                    - self.store.meta("created_at", self.store.clock()),
                ),
                "legacy_experience": self.store.meta("growth_legacy_experience", 0),
                "recognitions": list(reversed(entries))[:100],
            }

    def name_stage(self, name):
        context = current_execution.get()
        if context and context.actor_id != self.actor:
            raise ValueError("Only the primary resident names its growth stages")
        if self._evaluating():
            raise ValueError("Evaluation trials cannot name a resident's stage")
        if (
            not isinstance(name, str)
            or not 1 <= len(name.split()) <= 3
            or len(name) > 80
            or not name.strip()
        ):
            raise ValueError("Choose a stage name of one to three words")
        current = self.snapshot()
        return self.store.put(
            "growth_stage",
            f"{self.actor}:{current['level']}",
            self.actor,
            {
                "name": name.strip(),
                "level": current["level"],
                "growth_total": current["total"],
                "author": self.actor
                if context and self._occasion(context)
                else "operator",
            },
            category="interpretation",
        )
