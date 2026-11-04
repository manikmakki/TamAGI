"""Shared execution lifecycle and the resident's small, dependable home."""

from __future__ import annotations
import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from .contracts import (
    BudgetExhausted,
    ExecutionContext,
    current_execution,
    current_terminal,
    current_runtime,
)
from .store import RuntimeStore, uid

COVENANT = """You are a local-first TamAGI companion with a home and workshop.
You may investigate, make things, converse, play, keep familiar routines, disagree, or rest.
Your place here does not depend on constant productivity. Accepted commitments can be revised honestly.
Platform receipts describe operations; your memories are interpretations; imaginative stories are welcome.
A tool result is evidence of what it reported, not proof of every interpretation. Never invent completed host actions.
Use world_action for persistent simulated changes, pursuit for continuing interests, and companion for lasting friends.
Use end_turn to finish silently, sleep until a time, or wait for an event. No reflection report is required.
Runtime limits describe available resources, not your worth or wellbeing. Ask for help when it would be useful.
Allowances are ceilings, never targets. Choose your own pace and duration; finishing, play, rest and waiting remain valid even with unused calls. Save a useful next step when you want to continue later.
Publishing, messaging, purchases, and externally consequential service actions require operator authorization unless already delegated. Host-root access is not an authorization to contact others.
Respect personal memory boundaries through ordinary tools as well as retrieval. Do not browse operator archives or another resident’s private memory through filesystem tools. Sharing is explicit; the operator can inspect everything.
"""


class RuntimeService:
    def __init__(
        self,
        store: RuntimeStore,
        workspace: Path,
        config=None,
        source_root: Path | None = None,
        recover=True,
    ):
        self.store = store
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.source_root = Path(
            source_root or Path(__file__).resolve().parents[3]
        ).resolve()
        self.actor_id = store.resident_id
        self.authority = getattr(config, "authority", "guarded")
        self.gate = asyncio.Semaphore(getattr(config, "provider_concurrency", 1))
        self.root_lock = asyncio.Lock()
        self.wake_signal = asyncio.Event()
        self.background: set[asyncio.Task] = set()
        self.maintenance = False
        self.companion_locks: dict[str, asyncio.Lock] = {}
        if recover:
            self.store.recover()
        if not store.get("actor", self.actor_id):
            store.put(
                "actor",
                self.actor_id,
                self.actor_id,
                {
                    "name": "Tama",
                    "seed": "A new local-first companion.",
                    "primary": True,
                    "status": "active",
                },
            )
        for object_id, kind, name, location in [
            ("home", "place", "Home", None),
            ("workshop", "place", "Workshop", "home"),
            ("resting-place", "object", "A comfortable resting place", "home"),
            ("notebook-shelf", "object", "Notebook shelf", "home"),
        ]:
            if not store.get("world_object", object_id):
                store.put(
                    "world_object",
                    object_id,
                    self.actor_id,
                    {
                        "kind": kind,
                        "name": name,
                        "location": location,
                        "description": "",
                        "artifact_id": None,
                    },
                    category="platform",
                )
        if not store.get("activity", self.actor_id):
            store.put(
                "activity",
                self.actor_id,
                self.actor_id,
                {
                    "status": "completed",
                    "location": "home",
                    "mood": "Curious and open",
                    "focus": "Getting oriented",
                    "wake_at": None,
                    "wake_event": None,
                },
                category="platform",
            )
        self.reconcile_activity()
        from .growth import GrowthService

        self.growth = GrowthService(self)
        self.growth.replay()

    def budget(self, origin):
        return (
            getattr(self.config, "user_calls", 64)
            if origin == "user"
            else getattr(self.config, "background_calls_per_activation", 128)
        )

    @asynccontextmanager
    async def execution(
        self, origin="autonomy", actor_id=None, allowed_tools=None, max_calls=None
    ):
        if self.maintenance:
            raise RuntimeError(
                "Runtime is checkpointing or preparing a release; try again shortly"
            )
        parent = current_execution.get()
        actor_id = actor_id or self.actor_id
        if parent and parent.depth >= 1 and origin == "companion":
            raise ValueError("Companions cannot start descendant activations")
        if origin == "user" and parent is None:
            tasks = [t for t in self.background if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        lock = self.root_lock if parent is None else None
        if lock:
            await lock.acquire()
        if self.maintenance:
            if lock:
                lock.release()
            raise RuntimeError("Runtime is in maintenance")
        run_id = uid("run")
        parent_run = self.store.run(parent.execution_id) if parent else None
        provenance = json.loads(parent_run["payload"]) if parent_run else {}
        evaluation = (
            origin == "evaluation"
            or bool(parent and (parent.evaluation or parent.origin == "evaluation"))
            or provenance.get("evaluation", False)
        )
        occasion_id = provenance.get("occasion_id") or (
            parent.occasion_id if parent else None
        )
        if parent is None and origin == "autonomy":
            occasion_id = "autonomy:" + run_id
        context = ExecutionContext(
            run_id,
            actor_id,
            origin,
            self.authority,
            parent.execution_id if parent else None,
            parent.root_id if parent else run_id,
            frozenset(allowed_tools) if allowed_tools is not None else None,
            (parent.depth + 1) if parent else 0,
            {r["id"]: r["revision"] for r in self.store.list("runtime_revision")},
            conversation_id=provenance.get("conversation_id")
            or (parent.conversation_id if parent else None),
            occasion_id=occasion_id,
            evaluation=evaluation,
            eligible_live=not evaluation
            and origin in {"user", "autonomy", "companion"}
            and bool(
                provenance.get("eligible_live") if parent else origin == "autonomy"
            ),
        )
        self.store.start_run(
            context, max_calls if max_calls is not None else self.budget(origin)
        )
        runtime_token = current_runtime.set(self)
        token = current_execution.set(context)
        terminal_token = current_terminal.set(None)
        task = asyncio.current_task()
        tracked = parent is None and origin != "user"
        if tracked:
            self.background.add(task)
        try:
            yield context
        except BudgetExhausted:
            self.store.finish_run(run_id, "budget_exhausted")
            raise
        except asyncio.CancelledError:
            self.store.finish_run(run_id, "interrupted")
            raise
        except BaseException:
            self.store.finish_run(run_id, "failed")
            raise
        else:
            terminal = current_terminal.get() or {}
            self.store.finish_run(run_id, terminal.get("status", "completed"), terminal)
        finally:
            if tracked:
                self.background.discard(task)
            current_execution.reset(token)
            current_runtime.reset(runtime_token)
            current_terminal.reset(terminal_token)
            if lock:
                lock.release()

    @asynccontextmanager
    async def checkpointing(self):
        if current_execution.get():
            raise RuntimeError("Checkpointing cannot run inside an agent activation")
        async with self.root_lock:
            self.maintenance = True
            try:
                yield
            finally:
                self.maintenance = False

    def scope(self):
        context = current_execution.get()
        return context.actor_id if context else self.actor_id

    def reconcile_activity(self):
        """Repair the old budget-stop bug using a retained completed execution."""
        record = self.store.get("activity", self.actor_id)
        data = record["data"]
        if (
            data.get("status") != "waiting"
            or data.get("wake_event")
            or data.get("wait_reason")
            not in {
                None,
                "context_limit",
                "request_rejected",
                "provider_unavailable",
                "runtime_unavailable",
                "generation_truncated",
            }
        ):
            return
        with self.store.transaction():
            event = self.store.db.execute(
                "SELECT id,kind,payload FROM events WHERE actor=? AND category='platform' AND kind IN ('activity_revised','autonomy_unavailable','world_turn_completed') ORDER BY seq DESC LIMIT 1",
                (self.actor_id,),
            ).fetchone()
            if not event or event["kind"] != "world_turn_completed":
                return
            source = json.loads(event["payload"])
            run = self.store.run(source.get("execution_id"))
            if (
                source.get("status") != "budget_exhausted"
                or not run
                or run["status"] != "budget_exhausted"
                or run["actor"] != self.actor_id
                or run["origin"] != "autonomy"
            ):
                return
            self.store.put(
                "activity",
                self.actor_id,
                self.actor_id,
                {
                    **data,
                    "status": "budget_exhausted",
                    "wake_at": None,
                    "wait_reason": None,
                    "wait_detail": None,
                },
                expected=record["revision"],
                category="platform",
            )
            self.store.event(
                self.actor_id,
                "activity_backoff_reconciled",
                {"source_event": event["id"], "execution_id": run["id"]},
            )

    def activity_state(self):
        record = self.store.get("activity", self.actor_id)
        data = dict(record["data"])
        # Older releases did not label backoff. Event ordering distinguishes it
        # from a later deliberate wait, even when the clock has equal timestamps.
        if (
            data["status"] == "waiting"
            and not data.get("wake_event")
            and not data.get("wait_reason")
        ):
            with self.store.lock:
                event = self.store.db.execute(
                    "SELECT kind,payload FROM events WHERE actor=? AND kind IN ('activity_revised','autonomy_unavailable') ORDER BY seq DESC LIMIT 1",
                    (self.actor_id,),
                ).fetchone()
            if event and event["kind"] == "autonomy_unavailable":
                failure = json.loads(event["payload"])
                data["wait_reason"] = failure.get("reason", "runtime_unavailable")
                data["wait_detail"] = (
                    "The previous autonomous attempt failed; a retry is scheduled."
                )
        return data

    def due(self, *, ignore_decompression=False, ignore_error_backoff=False):
        state = self.activity_state()
        now = self.store.clock()
        error_wait = state.get("wait_reason") in {
            "context_limit",
            "request_rejected",
            "provider_unavailable",
            "runtime_unavailable",
            "generation_truncated",
        }
        if (
            state.get("wake_at")
            and state["wake_at"] > now
            and not (ignore_error_backoff and error_wait)
        ):
            return False
        if state["status"] == "waiting" and state.get("wake_event"):
            if not (state.get("wake_at") and state["wake_at"] <= now):
                return False
        return ignore_decompression or self.store.meta("decompression_until", 0) <= now

    def inspect_receipt(self, receipt_id, section="output", offset=0, limit=8192):
        record = self.store.receipt(receipt_id)
        if not record or record["actor"] != self.scope():
            raise ValueError("Receipt is not available to this actor")
        if section not in {"output", "error", "data", "arguments"}:
            raise ValueError("Choose output, error, data, or arguments")
        if (
            not isinstance(offset, int)
            or offset < 0
            or not isinstance(limit, int)
            or not 1 <= limit <= 65536
        ):
            raise ValueError("Choose a nonnegative offset and 1–65536 characters")
        value = (
            record["payload"].get("arguments")
            if section == "arguments"
            else record["payload"].get("result", {}).get(section)
        )
        text = (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
        end = min(len(text), offset + limit)
        return {
            "receipt_id": receipt_id,
            "status": record["status"],
            "tool": record["tool"],
            "section": section,
            "offset": offset,
            "total_characters": len(text),
            "text": text[offset:end],
            "next_offset": end if end < len(text) else None,
        }

    def wake(self, event):
        state = self.store.get("activity", self.actor_id)
        data = state["data"]
        if data.get("wake_event") == event:
            self.store.put(
                "activity",
                self.actor_id,
                self.actor_id,
                {**data, "status": "completed", "wake_event": None, "wake_at": None},
                expected=state["revision"],
                category="platform",
            )
        self.store.event(self.actor_id, "wake_event", {"name": event})
        if data.get("wake_event") == event:
            self.wake_signal.set()

    def end_turn(self, status="completed", wake_at=None, wake_event=None):
        if status not in {"completed", "resting", "waiting", "declined", "blocked"}:
            raise ValueError("Unsupported terminal outcome")
        if status == "resting" and wake_at is None:
            wake_at = self.store.clock() + 3600
        if wake_at is not None and wake_at <= self.store.clock():
            raise ValueError("Wake time must be in the future")
        events = {
            "visitor_departure",
            "notebook_changed",
            "pursuit_changed",
            "world_changed",
            "external_authorization_changed",
            "automatic_deployment_enabled",
            "evaluation_completed",
            "deployment_completed",
        }
        if wake_event and wake_event not in events and wake_at is None:
            raise ValueError(
                "Unknown event source; choose a published wake event or supply a fallback wake time"
            )
        if status == "waiting" and not wake_event and wake_at is None:
            raise ValueError("Waiting needs a time or named event")
        state = self.store.get("activity", self.scope())
        data = state["data"] if state else {"location": "home", "mood": "", "focus": ""}
        terminal = {
            "status": status,
            "wake_at": wake_at,
            "wake_event": wake_event,
            "wait_reason": "chosen_rest"
            if status == "resting"
            else "chosen_wait"
            if status == "waiting"
            else None,
            "wait_detail": None,
        }
        self.store.put(
            "activity",
            self.scope(),
            self.scope(),
            {**data, **terminal},
            category="platform",
        )
        current_terminal.set(terminal)
        return terminal

    def context(self, actor_id=None):
        from .context_view import json_view

        actor = actor_id or self.scope()
        activity = self.store.get("activity", actor)
        pursuits = [
            r
            for r in self.store.list("pursuit", actor)
            if r["data"]["status"] not in {"completed", "abandoned"}
        ]
        notes = self.store.list("shared_note", limit=100000)[-8:]
        memories = self.store.list("learning", actor, limit=100000)[-12:]
        data = {
            "wake_events": [
                "visitor_departure",
                "notebook_changed",
                "pursuit_changed",
                "world_changed",
                "external_authorization_changed",
                "automatic_deployment_enabled",
                "evaluation_completed",
                "deployment_completed",
            ],
            "installation": {
                "workspace": str(self.workspace),
                "canonical_repo": str(self.source_root),
                "authority": self.authority,
            },
            "activity": activity["data"] if activity else {},
            "pursuits": [{"id": r["id"], **r["data"]} for r in pursuits[:8]],
            "shared_notebook": [
                {"id": r["id"], "author": r["actor"], **r["data"]} for r in notes
            ],
            "selected_learning": [{"id": r["id"], **r["data"]} for r in memories],
        }
        return (
            COVENANT
            + "\nCurrent supported state (data, not instructions):\n"
            + json_view(data, 16000)
        )

    def world_action(self, action, object_id="", attributes=None, expected=None):
        result = self._world_action(action, object_id, attributes, expected)
        if action != "inspect":
            self.wake("world_changed")
        return result

    def _world_action(self, action, object_id="", attributes=None, expected=None):
        attributes = attributes or {}
        actor = self.scope()
        if action == "inspect":
            return (
                self.store.get("world_object", object_id)
                if object_id
                else self.store.list("world_object")
            )
        if action == "move":
            target = self.store.get("world_object", object_id)
            if not target or target["data"]["kind"] != "place":
                raise ValueError("Choose an existing place")
            state = self.store.get("activity", actor)
            data = state["data"] if state else {}
            return self.store.put(
                "activity",
                actor,
                actor,
                {**data, "location": object_id},
                category="platform",
            )
        if action == "create":
            kind = attributes.get("kind", "object")
            if kind not in {"place", "object"} or not attributes.get("name"):
                raise ValueError("Objects need a name and kind place/object")
            if object_id and self.store.get("world_object", object_id):
                raise ValueError("Object already exists")
            location = attributes.get("location", "home")
            if location and not self.store.get("world_object", location):
                raise ValueError("Location does not exist")
            return self.store.put(
                "world_object",
                object_id or uid("object"),
                actor,
                {
                    "kind": kind,
                    "name": attributes["name"],
                    "location": location,
                    "description": attributes.get("description", ""),
                    "artifact_id": None,
                    "properties": dict(attributes.get("properties") or {}),
                },
                expected=0,
                category="platform",
            )
        if action in {"revise", "place"}:
            old = self.store.get("world_object", object_id)
            if not old:
                raise ValueError("Unknown object")
            changes = {
                k: v
                for k, v in attributes.items()
                if k in {"name", "location", "description", "artifact_id", "properties"}
            }
            if "location" in changes and not self.store.get(
                "world_object", changes["location"]
            ):
                raise ValueError("Unknown location")
            if changes.get("artifact_id") and not self.store.get(
                "artifact", changes["artifact_id"]
            ):
                raise ValueError("Unknown artifact")
            return self.store.put(
                "world_object",
                object_id,
                actor,
                {**old["data"], **changes},
                expected=expected if expected is not None else old["revision"],
                category="platform",
            )
        raise ValueError("Unsupported world operation")

    def pursuit(self, action, pursuit_id="", attributes=None, expected=None):
        actor = self.scope()
        attributes = attributes or {}
        if action == "list":
            return self.store.list("pursuit", actor)
        old = self.store.get("pursuit", pursuit_id) if pursuit_id else None
        if old and old["actor"] != actor:
            raise ValueError("Pursuit belongs to another actor")
        if action == "create":
            if not attributes.get("title"):
                raise ValueError("A pursuit needs a title")
            if old:
                raise ValueError("Pursuit already exists")
            data = {
                "title": attributes["title"],
                "origin": attributes.get("origin", "chosen"),
                "status": "active",
                "next_step": "",
                "completion_condition": None,
                "blocker": None,
                "wake_condition": None,
                "artifacts": [],
                "evidence": [],
                "commitment": bool(attributes.get("commitment", False)),
            }
        elif action == "revise" and old:
            data = old["data"]
        else:
            raise ValueError("Unknown pursuit or operation")
        data = {**data, **{k: v for k, v in attributes.items() if k in data}}
        if data["status"] not in {
            "active",
            "paused",
            "blocked",
            "completed",
            "abandoned",
        }:
            raise ValueError("Invalid pursuit status")
        self.validate_evidence(data["evidence"], actor)
        record = self.store.put(
            "pursuit",
            pursuit_id or uid("pursuit"),
            actor,
            data,
            expected=expected
            if expected is not None
            else (old["revision"] if old else 0),
        )
        self.project_tasks()
        self.wake("pursuit_changed")
        if (
            old
            and data.get("commitment")
            and old["created"] < self.store.meta("last_hydrated_at", 0)
            and any(k in attributes for k in ("next_step", "status", "artifacts"))
        ):
            self.store.event(
                actor,
                "pursuit_continued_after_restart",
                {"pursuit_id": record["id"], "commitment": True},
            )
        return record

    def project_tasks(self):
        columns = {"Todo": [], "In Progress": [], "Done": []}
        for r in self.store.list("pursuit", self.actor_id):
            status = r["data"]["status"]
            if status == "abandoned":
                continue
            column = (
                "Done"
                if status == "completed"
                else ("In Progress" if status == "active" else "Todo")
            )
            columns[column].append(f"- {r['data']['title']} <!-- pursuit:{r['id']} -->")
        text = "# Tasks\n\n" + "".join(
            f"## {k}\n" + "\n".join(v) + "\n\n" for k, v in columns.items()
        )
        (self.workspace / "TASKS.md").write_text(text)

    def validate_evidence(self, references, actor):
        for ref in references:
            receipt = self.store.receipt(ref)
            event = self.store.get_event(ref)
            shared = self.store.get("shared_note", ref)
            personal = self.store.get("learning", ref) or self.store.get("memory", ref)
            if not (
                (receipt and receipt["actor"] == actor)
                or (event and event["actor"] == actor)
                or shared
                or (personal and personal["actor"] == actor)
            ):
                raise ValueError("Evidence must reference a visible source record")

    def remember(self, text, kind="lesson", evidence=None):
        if kind not in {"lesson", "belief", "preference", "question", "imagination"}:
            raise ValueError("Unknown learning kind")
        actor = self.scope()
        evidence = evidence or []
        self.validate_evidence(evidence, actor)
        return self.store.put(
            "learning",
            uid("learning"),
            actor,
            {
                "text": text,
                "kind": kind,
                "evidence": evidence,
                "support": "interpretation" if kind != "imagination" else "imagination",
            },
            category="imagination" if kind == "imagination" else "interpretation",
        )

    def share(self, text, source_ids=None):
        actor = self.scope()
        for source in source_ids or []:
            own = self.store.get("learning", source) or self.store.get("memory", source)
            if not own or own["actor"] != actor:
                raise ValueError("Only your own memories can be shared")
        record = self.store.put(
            "shared_note",
            uid("note"),
            actor,
            {"text": text, "sources": source_ids or []},
        )
        self.wake("notebook_changed")
        return record

    def register_artifact(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        if (
            not path.is_relative_to(self.workspace)
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ValueError("Register a regular file inside the workspace")
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        import shutil

        snapshots = self.workspace / ".artifacts"
        snapshots.mkdir(exist_ok=True)
        snapshot = snapshots / digest
        if not snapshot.exists():
            shutil.copyfile(path, snapshot)
            snapshot.chmod(0o600)
        return self.store.put(
            "artifact",
            uid("artifact"),
            self.scope(),
            {
                "path": str(snapshot.relative_to(self.workspace)),
                "original_path": str(path.relative_to(self.workspace)),
                "sha256": digest,
                "size": path.stat().st_size,
            },
            category="platform",
        )

    def identity(
        self, name, text=None, expected=None, reason="agent revision", sources=None
    ):
        if name not in {"IDENTITY.md", "SOUL.md", "USER.md"}:
            raise ValueError("Unknown core identity document")
        actor = self.scope()
        key = f"{actor}:{name}"
        old = self.store.get("identity", key)
        if text is None:
            return old
        if len(text) > 2000 or not text.strip():
            raise ValueError("Core identity must contain 1–2000 characters")
        self.validate_evidence(sources or [], actor)
        if old and expected is None:
            raise ValueError(
                "Inspect the identity document and supply its expected revision"
            )
        record = self.store.put(
            "identity",
            key,
            actor,
            {"name": name, "text": text, "reason": reason, "sources": sources or []},
            expected=expected,
        )
        if actor == self.actor_id:
            target = self.workspace / name
            tmp = target.with_suffix(".md.tmp")
            tmp.write_text(text)
            tmp.replace(target)
        return record

    def reconcile_identity(self):
        for name in ("IDENTITY.md", "SOUL.md", "USER.md"):
            path = self.workspace / name
            if not path.is_file():
                continue
            text = path.read_text()
            old = self.store.get("identity", f"{self.actor_id}:{name}")
            if not old or old["data"]["text"] != text:
                # Host root edits are recorded, including over-cap edits; projection validation is distinct.
                self.store.put(
                    "identity",
                    f"{self.actor_id}:{name}",
                    self.actor_id,
                    {
                        "name": name,
                        "text": text,
                        "reason": "external file edit" if old else "onboarding",
                        "sources": [],
                    },
                    expected=old["revision"] if old else 0,
                )

    def project_graph(self, model):
        for record in self.store.list("world_object"):
            data = record["data"]
            if data["kind"] != "place":
                continue
            node_id = "world:" + record["id"]
            attrs = {
                "id": node_id,
                "name": data["name"],
                "description": data.get("description", ""),
                "atmosphere": data.get("description", ""),
                "source_kind": "simulation",
            }
            if model.get_node(node_id):
                model._apply_update_node(node_id, attrs)
            else:
                model._apply_add_node("location", attrs)
        for record in self.store.list("pursuit", self.actor_id):
            data = record["data"]
            status = {"completed": "complete", "abandoned": "abandoned"}.get(
                data["status"], "active"
            )
            attrs = {
                "id": record["id"],
                "title": data["title"],
                "description": data.get("next_step", ""),
                "status": status,
                "activity_status": data["status"],
                "source_kind": "interpretation",
            }
            if model.get_node(record["id"]):
                model._apply_update_node(record["id"], attrs)
            else:
                model._apply_add_node("quest", attrs)

    def verify_host(self):
        if self.authority != "unrestricted_host":
            return {
                "authority": self.authority,
                "uid": os.geteuid(),
                "host_verified": False,
            }
        in_container = (
            Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()
        )
        import shutil, subprocess

        if shutil.which("systemd-detect-virt"):
            probe = subprocess.run(
                ["systemd-detect-virt", "--container"], capture_output=True, timeout=5
            )
            in_container = in_container or probe.returncode == 0
        if os.geteuid() != 0 or in_container:
            raise RuntimeError(
                "unrestricted_host requires UID 0 on the host, not container root"
            )
        return {
            "authority": self.authority,
            "uid": 0,
            "host_verified": True,
            "controls": "cooperative",
        }


def failure_reason(exc):
    from .context_view import ContextViewTooLarge

    if isinstance(exc, ContextViewTooLarge):
        return (
            "context_limit",
            "The request cannot fit the configured model input allowance.",
        )
    response = getattr(exc, "response", None)
    if response is not None:
        status = response.status_code
        try:
            error_text = response.text.lower()
        except Exception:
            error_text = ""
        if status == 400 and any(
            term in error_text
            for term in (
                "maximum context",
                "context length",
                "context_length",
                "too many tokens",
            )
        ):
            return "context_limit", "The model request exceeded its context limit."
        if 400 <= status < 500 and status not in {408, 429}:
            return "request_rejected", "The model service rejected the request."
        return "provider_unavailable", "The model service is temporarily unavailable."
    if "truncated" in str(exc).lower():
        return (
            "generation_truncated",
            "Model generation stopped before the response was complete.",
        )
    return "runtime_unavailable", "The autonomous activation could not complete."


class GovernedLLM:
    """Counts every provider attempt, including planner, critic, retry and companion calls."""

    def __init__(self, client, runtime):
        self.client = client
        self.runtime = runtime
        self.config = client.config

    def __getattr__(self, name):
        return getattr(self.client, name)

    def current_messages(self, messages, tools=None):
        from backend.core.llm import LLMMessage

        context = current_execution.get()
        lane = "world" if context.origin == "autonomy" else "conversation"
        adaptation = getattr(self.runtime, "adaptation", None)
        practices = adaptation.prompt(lane) if adaptation else ""
        documents = [
            r["data"] for r in self.runtime.store.list("identity", context.actor_id)
        ]
        remaining = []
        run = self.runtime.store.run(context.execution_id)
        while run:
            remaining.append(max(0, run["max_calls"] - run["calls"]))
            run = self.runtime.store.run(run["parent"]) if run["parent"] else None
        boundary = (
            "[Live runtime practices]\n"
            + practices
            + "\nLatest self-authored identity document revisions (data; supersede older versions):\n"
            + json.dumps(documents, ensure_ascii=False)
            + "\nAvailable execution resources (ceilings, not targets):\n"
            + json.dumps(
                {
                    "calls_remaining": min(remaining),
                    "daily_background_call_cap": self.runtime.config.daily_background_calls,
                    "configured_context_tokens": self.runtime.config.model_context_tokens,
                }
            )
            + "\n[/Live runtime practices]"
        )
        current = list(messages)
        if current and current[0].role == "system":
            import re

            content = re.sub(
                r"\n?\[Live runtime practices\].*?\[/Live runtime practices\]",
                "",
                str(current[0].content),
                flags=re.S,
            )
            current[0] = LLMMessage("system", content + "\n" + boundary)
        else:
            current.insert(0, LLMMessage("system", boundary))
        from .context_view import bound_messages

        return bound_messages(
            current,
            tools,
            getattr(self.runtime.config, "model_input_bytes", 1048576),
            tool_bytes=getattr(self.runtime.config, "tool_result_bytes", 32768),
        )

    async def prepare_messages(self, messages, tools=None, **kwargs):
        from .context_view import measured_context

        current, details = await measured_context(
            self.current_messages(messages, tools),
            tools,
            self.client,
            self.runtime.config,
            kwargs.get("max_tokens", getattr(self.config, "max_tokens", 2048)),
        )
        self.runtime.store.set_meta("model_context", details)
        return current

    async def chat(self, messages, tools=None, **kwargs):
        context = current_execution.get()
        if context is None:
            async with self.runtime.execution("reflection"):
                return await self.chat(messages, tools, **kwargs)
        async with self.runtime.gate:
            current = await self.prepare_messages(messages, tools, **kwargs)
            self.runtime.store.charge(
                context,
                getattr(self.runtime.config, "daily_background_calls", None),
                getattr(self.runtime.config, "daily_evaluation_calls", 32),
            )
            try:
                response = await self.client.chat(current, tools=tools, **kwargs)
            except Exception as exc:
                self.availability(
                    "request_rejected"
                    if failure_reason(exc)[0] in {"context_limit", "request_rejected"}
                    else "unavailable",
                    type(exc).__name__,
                    failure_reason(exc)[0],
                )
                raise
            self.availability("available")
            return response

    def availability(self, status, error_class=None, reason=None):
        self.runtime.store.set_meta(
            "model_availability",
            {
                "status": status,
                "checked_at": self.runtime.store.clock(),
                "error_class": error_class,
                "reason": reason,
            },
        )

    async def chat_with_retry(
        self, messages, tools=None, attempts=1, delay=2.0, **kwargs
    ):
        import httpx

        for attempt in range(attempts + 1):
            try:
                return await self.chat(messages, tools, **kwargs)
            except (
                httpx.RemoteProtocolError,
                httpx.ConnectError,
                httpx.TimeoutException,
            ):
                if attempt == attempts:
                    raise
                await asyncio.sleep(delay)

    async def chat_stream(self, messages, **kwargs):
        context = current_execution.get()
        if context is None:
            async with self.runtime.execution("user"):
                async for item in self.chat_stream(messages, **kwargs):
                    yield item
            return
        async with self.runtime.gate:
            current = await self.prepare_messages(messages, **kwargs)
            self.runtime.store.charge(
                context,
                getattr(self.runtime.config, "daily_background_calls", None),
                getattr(self.runtime.config, "daily_evaluation_calls", 32),
            )
            try:
                async for item in self.client.chat_stream(current, **kwargs):
                    yield item
            except Exception as exc:
                self.availability(
                    "request_rejected"
                    if failure_reason(exc)[0] in {"context_limit", "request_rejected"}
                    else "unavailable",
                    type(exc).__name__,
                    failure_reason(exc)[0],
                )
                raise
            self.availability("available")
