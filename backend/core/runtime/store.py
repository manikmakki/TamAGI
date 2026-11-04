"""Transactional local source records; filesystem and external effects are not transactions."""

from __future__ import annotations
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager, closing
from pathlib import Path
from typing import Any
from .contracts import RuntimeConflict, BudgetExhausted, ActionReceipt, Record


def uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class RuntimeStore:
    SCHEMA = 1

    def __init__(self, path: str | Path, clock=time.time):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.lock = threading.RLock()
        self.db = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,
          actor TEXT NOT NULL,kind TEXT NOT NULL,category TEXT NOT NULL,created REAL NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS records(kind TEXT NOT NULL,id TEXT NOT NULL,actor TEXT NOT NULL,
          revision INTEGER NOT NULL,created REAL NOT NULL,updated REAL NOT NULL,payload TEXT NOT NULL,
          PRIMARY KEY(kind,id));
        CREATE INDEX IF NOT EXISTS records_actor ON records(kind,actor,updated);
        CREATE TABLE IF NOT EXISTS revisions(kind TEXT NOT NULL,id TEXT NOT NULL,revision INTEGER NOT NULL,
          actor TEXT NOT NULL,created REAL NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(kind,id,revision));
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,actor TEXT NOT NULL,origin TEXT NOT NULL,
          parent TEXT,root TEXT NOT NULL,status TEXT NOT NULL,created REAL NOT NULL,finished REAL,
          max_calls INTEGER NOT NULL,calls INTEGER NOT NULL DEFAULT 0,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS charges(id INTEGER PRIMARY KEY,run TEXT NOT NULL,bucket TEXT NOT NULL,created REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS charges_window ON charges(bucket,created);
        CREATE TABLE IF NOT EXISTS receipts(id TEXT PRIMARY KEY,run TEXT NOT NULL,actor TEXT NOT NULL,
          tool TEXT NOT NULL,call_key TEXT,started REAL NOT NULL,finished REAL,status TEXT NOT NULL,payload TEXT NOT NULL,
          UNIQUE(run,call_key));
        """)
        with self.transaction():
            version = self.meta("schema")
            if version is not None and version != self.SCHEMA:
                raise ValueError(f"Unsupported runtime schema {version}")
            self.set_meta("schema", self.SCHEMA)
            if self.meta("resident_id") is None:
                self.set_meta("resident_id", uid("resident"))
                self.set_meta("installation_id", uid("installation"))
                self.set_meta("created_at", self.clock())
        self.resident_id = self.meta("resident_id")

    @contextmanager
    def transaction(self):
        with self.lock:
            nested = self.db.in_transaction
            if not nested:
                self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                if not nested:
                    self.db.execute("COMMIT")
            except BaseException:
                if not nested:
                    self.db.execute("ROLLBACK")
                raise

    def meta(self, key, default=None):
        with self.lock:
            row = self.db.execute(
                "SELECT value FROM meta WHERE key=?", (key,)
            ).fetchone()
            return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.transaction():
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES(?,?)", (key, encoded(value))
            )

    def event(self, actor, kind, payload, category="platform"):
        if category not in {"platform", "interpretation", "imagination", "legacy"}:
            raise ValueError("Unknown evidence category")
        with self.transaction():
            event_id = uid("event")
            self.db.execute(
                "INSERT INTO events(id,actor,kind,category,created,payload) VALUES(?,?,?,?,?,?)",
                (event_id, actor, kind, category, self.clock(), encoded(payload)),
            )
            return event_id

    def get_event(self, event_id):
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM events WHERE id=?", (event_id,)
            ).fetchone()
            return {**dict(row), "payload": json.loads(row["payload"])} if row else None

    def events(self, limit=50, after=0, actor=None):
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM events WHERE seq>? AND (? IS NULL OR actor=?) ORDER BY seq DESC LIMIT ?",
                (after, actor, actor, min(max(limit, 1), 500)),
            ).fetchall()
            return [
                {**dict(r), "payload": json.loads(r["payload"])} for r in reversed(rows)
            ]

    def get(self, kind, record_id) -> Record[dict[str, Any]] | None:
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM records WHERE kind=? AND id=?", (kind, record_id)
            ).fetchone()
            return self._record(row) if row else None

    def list(self, kind, actor=None, limit=200):
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM records WHERE kind=? AND (? IS NULL OR actor=?) ORDER BY created,id LIMIT ?",
                (kind, actor, actor, limit),
            ).fetchall()
            return [self._record(row) for row in rows]

    @staticmethod
    def _record(row):
        return {**dict(row), "data": json.loads(row["payload"])}

    def put(
        self, kind, record_id, actor, data, expected=None, category="interpretation"
    ):
        with self.transaction():
            old = self.get(kind, record_id)
            revision = old["revision"] if old else 0
            if expected is not None and expected != revision:
                raise RuntimeConflict(
                    "Record changed; inspect its current revision before revising"
                )
            now = self.clock()
            self.db.execute(
                "INSERT OR REPLACE INTO records VALUES(?,?,?,?,?,?,?)",
                (
                    kind,
                    record_id,
                    actor,
                    revision + 1,
                    old["created"] if old else now,
                    now,
                    encoded(data),
                ),
            )
            self.db.execute(
                "INSERT INTO revisions VALUES(?,?,?,?,?,?)",
                (kind, record_id, revision + 1, actor, now, encoded(data)),
            )
            self.event(
                actor,
                f"{kind}_revised",
                {"id": record_id, "revision": revision + 1},
                category,
            )
            return self.get(kind, record_id)

    def history(self, kind, record_id):
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM revisions WHERE kind=? AND id=? ORDER BY revision",
                (kind, record_id),
            ).fetchall()
            return [{**dict(r), "data": json.loads(r["payload"])} for r in rows]

    def delete(self, kind, record_id, actor):
        with self.transaction():
            self.db.execute(
                "DELETE FROM records WHERE kind=? AND id=? AND actor=?",
                (kind, record_id, actor),
            )
            self.event(actor, f"{kind}_deleted", {"id": record_id})

    def start_run(self, context, max_calls):
        with self.transaction():
            self.db.execute(
                "INSERT INTO runs(id,actor,origin,parent,root,status,created,max_calls,payload) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    context.execution_id,
                    context.actor_id,
                    context.origin,
                    context.parent_id,
                    context.root_id,
                    "running",
                    self.clock(),
                    max_calls,
                    encoded(
                        {
                            "conversation_id": context.conversation_id,
                            "occasion_id": context.occasion_id
                            or (
                                "autonomy:" + context.root_id
                                if context.origin == "autonomy"
                                and not context.parent_id
                                else None
                            ),
                            "evaluation": context.evaluation
                            or context.origin == "evaluation",
                            "eligible_live": bool(
                                (
                                    context.eligible_live
                                    or context.origin == "autonomy"
                                    and not context.parent_id
                                )
                                and not context.evaluation
                                and context.origin in {"user", "autonomy", "companion"}
                            ),
                        }
                    ),
                ),
            )
            self.event(
                context.actor_id,
                "execution_started",
                {
                    "execution_id": context.execution_id,
                    "origin": context.origin,
                    "parent": context.parent_id,
                },
            )

    def set_run_provenance(self, run_id, **fields):
        """Merge application-owned context; never accept these fields from tool arguments."""
        with self.transaction():
            run = self.run(run_id)
            if not run:
                raise ValueError("Execution is unavailable")
            payload = {**json.loads(run["payload"]), **fields}
            self.db.execute(
                "UPDATE runs SET payload=? WHERE id=?", (encoded(payload), run_id)
            )
            return payload

    def finish_run(self, run_id, status, payload=None):
        with self.transaction():
            run = self.run(run_id)
            existing = json.loads(run["payload"]) if run else {}
            self.db.execute(
                "UPDATE runs SET status=?,finished=?,payload=? WHERE id=?",
                (
                    status,
                    self.clock(),
                    encoded({**existing, **(payload or {})}),
                    run_id,
                ),
            )

    def run(self, run_id):
        with self.lock:
            row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def charge(self, context, background_limit=None, evaluation_limit=32):
        with self.transaction():
            run = self.run(context.execution_id)
            if not run or run["status"] != "running":
                raise BudgetExhausted("Execution is not running")
            root = self.run(context.root_id)
            # Charge the whole ancestor chain: grandchildren cannot reset parent caps.
            ancestors = []
            cursor = run
            while cursor:
                ancestors.append(cursor)
                if cursor["calls"] >= cursor["max_calls"]:
                    raise BudgetExhausted("Activation call allowance exhausted")
                cursor = self.run(cursor["parent"]) if cursor["parent"] else None
            bucket = (
                "evaluation"
                if any(r["origin"] == "evaluation" for r in ancestors)
                else ("user" if root["origin"] == "user" else "background")
            )
            limit = evaluation_limit if bucket == "evaluation" else background_limit
            if bucket != "user" and limit is not None:
                spent = self.db.execute(
                    "SELECT count(*) FROM charges WHERE bucket=? AND created>?",
                    (bucket, self.clock() - 86400),
                ).fetchone()[0]
                if spent >= limit:
                    raise BudgetExhausted(f"Rolling {bucket} allowance exhausted")
            for ancestor in ancestors:
                self.db.execute(
                    "UPDATE runs SET calls=calls+1 WHERE id=?", (ancestor["id"],)
                )
            self.db.execute(
                "INSERT INTO charges(run,bucket,created) VALUES(?,?,?)",
                (context.execution_id, bucket, self.clock()),
            )

    def begin_receipt(
        self, context, tool, call_key=None, arguments=None, repair_of_receipt_id=None
    ):
        with self.transaction():
            if call_key:
                old = self.db.execute(
                    "SELECT id FROM receipts WHERE run=? AND call_key=?",
                    (context.execution_id, call_key),
                ).fetchone()
                if old:
                    return self.receipt(old[0]), False
            receipt_id = uid("receipt")
            run = self.run(context.execution_id)
            provenance = json.loads(run["payload"]) if run else {}
            self.db.execute(
                "INSERT INTO receipts(id,run,actor,tool,call_key,started,status,payload) VALUES(?,?,?,?,?,?,?,?)",
                (
                    receipt_id,
                    context.execution_id,
                    context.actor_id,
                    tool,
                    call_key,
                    self.clock(),
                    "started",
                    encoded(
                        {
                            "arguments_hash": hashlib.sha256(
                                encoded(arguments or {}).encode()
                            ).hexdigest(),
                            "arguments": redacted(arguments or {}),
                            "execution_start_revisions": context.revisions,
                            "conversation_id": provenance.get(
                                "conversation_id", context.conversation_id
                            ),
                            "occasion_id": provenance.get(
                                "occasion_id", context.occasion_id
                            ),
                            "evaluation": provenance.get(
                                "evaluation", context.evaluation
                            ),
                            "eligible_live": provenance.get(
                                "eligible_live", context.eligible_live
                            ),
                            "repair_of_receipt_id": repair_of_receipt_id,
                            "runtime_revisions": {
                                r["id"]: r["revision"]
                                for r in self.list("runtime_revision")
                            },
                        }
                    ),
                ),
            )
            return self.receipt(receipt_id), True

    def finish_receipt(self, receipt_id, status, result):
        with self.transaction():
            old = self.receipt(receipt_id)
            payload = {**old["payload"], "result": result}
            self.db.execute(
                "UPDATE receipts SET status=?,finished=?,payload=? WHERE id=?",
                (status, self.clock(), encoded(payload), receipt_id),
            )
            self.event(
                old["actor"],
                "action_result",
                {"receipt_id": receipt_id, "tool": old["tool"], "status": status},
            )

    def receipt(self, receipt_id) -> ActionReceipt | None:
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM receipts WHERE id=?", (receipt_id,)
            ).fetchone()
            return {**dict(row), "payload": json.loads(row["payload"])} if row else None

    def receipts(self, run_id=None, limit=50) -> list[ActionReceipt]:
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM receipts WHERE (? IS NULL OR run=?) ORDER BY started DESC LIMIT ?",
                (run_id, run_id, limit),
            ).fetchall()
            return [self.receipt(row[0]) for row in rows]

    def recover(self):
        with self.transaction():
            self.db.execute(
                "UPDATE receipts SET status='outcome_unknown',finished=? WHERE status='started'",
                (self.clock(),),
            )
            self.db.execute(
                "UPDATE runs SET status='interrupted',finished=? WHERE status='running'",
                (self.clock(),),
            )

    def checkpoint(self, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, closing(sqlite3.connect(destination)) as target:
            self.db.backup(target)

    def close(self):
        with self.lock:
            self.db.close()


def redacted(value):
    if isinstance(value, dict):
        return {
            k: (
                "[redacted]"
                if any(
                    marker in k.lower()
                    for marker in (
                        "password",
                        "token",
                        "api_key",
                        "secret",
                        "private_key",
                    )
                )
                else redacted(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redacted(v) for v in value]
    return value
