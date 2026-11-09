"""Independent canonical household store, never a resident RuntimeStore adapter."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from .contracts import Conflict, NotFound


def uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class HouseholdStore:
    SCHEMA = 1
    APPLICATION_ID = 0x54414848

    def __init__(self, path: str | Path, clock=time.time):
        supplied = Path(path).absolute()
        if supplied != supplied.resolve():
            raise ValueError("Household database paths must be canonical")
        if any(Path(str(supplied) + suffix).is_symlink() for suffix in ("", "-wal", "-shm", "-journal")):
            raise ValueError("A household database and its sidecars cannot be symlinks")
        self.path = supplied.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.root = self.path.parent
        self.clock = clock
        self.lock = threading.RLock()
        self._savepoint = 0
        existed = self.path.exists()
        if existed:
            # Opening a foreign database read/write could recover a hot journal.
            # Probe its format with SQLite's read-only mode before mutable access.
            probe = sqlite3.connect(f"file:{quote(str(self.path), safe='/')}?mode=ro", uri=True)
            try:
                probe.execute("PRAGMA query_only=ON")
                if probe.execute("PRAGMA application_id").fetchone()[0] != self.APPLICATION_ID:
                    raise ValueError("This is not a household database; resident stores cannot be enrolled here")
                row = probe.execute("SELECT value FROM hh_meta WHERE key='schema'").fetchone()
                if not row or json.loads(row[0]) != self.SCHEMA:
                    raise ValueError("Unsupported household schema")
            finally:
                probe.close()
        self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        try:
            # Inspect before changing pragmas or creating tables in an existing file.
            if existed:
                application = self.db.execute("PRAGMA application_id").fetchone()[0]
                if application != self.APPLICATION_ID:
                    raise ValueError("This is not a household database; resident stores cannot be enrolled here")
                row = self.db.execute("SELECT value FROM hh_meta WHERE key='schema'").fetchone()
                if not row or json.loads(row[0]) != self.SCHEMA:
                    raise ValueError("Unsupported household schema")
            self.db.execute("PRAGMA busy_timeout=5000")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA journal_mode=WAL")
            with self.transaction():
                self.db.execute(f"PRAGMA application_id={self.APPLICATION_ID}")
                for sql in (
                    "CREATE TABLE IF NOT EXISTS hh_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS hh_records(kind TEXT NOT NULL,id TEXT NOT NULL,revision INTEGER NOT NULL,created REAL NOT NULL,updated REAL NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(kind,id))",
                    "CREATE TABLE IF NOT EXISTS hh_events(seq INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,kind TEXT NOT NULL,actor TEXT NOT NULL,created REAL NOT NULL,payload TEXT NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS hh_commands(actor TEXT NOT NULL,id TEXT NOT NULL,fingerprint TEXT NOT NULL,operation TEXT NOT NULL,created REAL NOT NULL,result TEXT NOT NULL,PRIMARY KEY(actor,id))",
                ):
                    self.db.execute(sql)
                self.set_meta("schema", self.SCHEMA)
        except BaseException:
            self.db.close()
            raise

    @contextmanager
    def transaction(self):
        with self.lock:
            nested = self.db.in_transaction
            self._savepoint += 1
            name = f"hh_{self._savepoint}"
            self.db.execute(f"SAVEPOINT {name}" if nested else "BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                if nested:
                    self.db.execute(f"ROLLBACK TO {name}")
                    self.db.execute(f"RELEASE {name}")
                else:
                    self.db.execute("ROLLBACK")
                raise
            else:
                self.db.execute(f"RELEASE {name}" if nested else "COMMIT")

    def meta(self, key: str, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM hh_meta WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def set_meta(self, key: str, value):
        with self.transaction():
            self.db.execute("INSERT OR REPLACE INTO hh_meta VALUES(?,?)", (key, encoded(value)))

    @staticmethod
    def _record(row):
        result = dict(row)
        result["data"] = json.loads(result.pop("payload"))
        return result

    def get(self, kind: str, record_id: str):
        with self.lock:
            row = self.db.execute("SELECT * FROM hh_records WHERE kind=? AND id=?", (kind, record_id)).fetchone()
            return self._record(row) if row else None

    def list(self, kind: str):
        with self.lock:
            return [self._record(row) for row in self.db.execute(
                "SELECT * FROM hh_records WHERE kind=? ORDER BY created,id", (kind,)
            ).fetchall()]

    def put(self, kind: str, record_id: str, data, expected: int | None = None):
        payload = encoded(data)
        with self.transaction():
            old = self.get(kind, record_id)
            revision = old["revision"] if old else 0
            if expected is not None and expected != revision:
                raise Conflict("Record revision changed")
            now = self.clock()
            self.db.execute("INSERT OR REPLACE INTO hh_records VALUES(?,?,?,?,?,?)", (
                kind, record_id, revision + 1, old["created"] if old else now, now, payload
            ))
            return self.get(kind, record_id)

    def delete(self, kind: str, record_id: str, expected: int | None = None):
        with self.transaction():
            old = self.get(kind, record_id)
            if old is None:
                raise NotFound(record_id)
            if expected is not None and expected != old["revision"]:
                raise Conflict("Record revision changed")
            self.db.execute("DELETE FROM hh_records WHERE kind=? AND id=?", (kind, record_id))

    def event(self, kind: str, actor: str, data):
        with self.transaction():
            event_id = uid("event")
            cursor = self.db.execute("INSERT INTO hh_events(id,kind,actor,created,payload) VALUES(?,?,?,?,?)", (
                event_id, kind, actor, self.clock(), encoded(data)
            ))
            return {"seq": cursor.lastrowid, "id": event_id, "kind": kind, "actor": actor,
                    "created": self.clock(), "data": json.loads(encoded(data))}

    def events(self, after: int = 0):
        with self.lock:
            return [self._record(row) for row in self.db.execute(
                "SELECT * FROM hh_events WHERE seq>? ORDER BY seq", (after,)
            ).fetchall()]

    def command(self, actor: str, command_id: str, operation: str, payload, apply: Callable):
        if not all(isinstance(v, str) and v for v in (actor, command_id, operation)):
            raise ValueError("A shared command requires stable actor, command, and operation identities")
        fingerprint = hashlib.sha256(encoded({"operation": operation, "payload": payload}).encode()).hexdigest()
        with self.transaction():
            old = self.db.execute("SELECT fingerprint,result FROM hh_commands WHERE actor=? AND id=?", (actor, command_id)).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise Conflict("Command identity was already used with different input")
                return json.loads(old["result"])
            result = encoded(apply())
            self.db.execute("INSERT INTO hh_commands VALUES(?,?,?,?,?,?)", (
                actor, command_id, fingerprint, operation, self.clock(), result
            ))
            self.event("command_committed", actor, {"command_id": command_id, "operation": operation})
            return json.loads(result)

    def command_result(self, actor: str, command_id: str):
        """Reconcile an acknowledged or lost shared effect without executing it."""
        with self.lock:
            row = self.db.execute("SELECT operation,result FROM hh_commands WHERE actor=? AND id=?", (actor, command_id)).fetchone()
            if row is None:
                raise NotFound("No committed outcome for this actor and command")
            return {"operation": row["operation"], "result": json.loads(row["result"])}

    def close(self):
        with self.lock:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
