"""One durable household clock; scheduled opportunities are separate from service time."""
from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from croniter import croniter
import httpx

from .store import uid
from .native_service import TERMINAL


class NativeWorldScheduler:
    def __init__(self, manager, config):
        self.manager, self.store, self.config = manager, manager.store, config
        self.lock = asyncio.Lock()
        self.task = None
        self.zone = ZoneInfo(config.runtime.timezone)
        if not croniter.is_valid(config.autonomy.schedule):
            raise ValueError("Choose a valid household World Thread cron schedule")
        if self.store.meta("native_world_schedule") is None:
            self.store.set_meta("native_world_schedule", {"next_at": self.next_after(self.store.clock()), "last_resident_id": None})

    @property
    def enabled(self):
        return self.config.autonomy.enabled and self.config.world_thread.enabled

    def next_after(self, now):
        return croniter(self.config.autonomy.schedule, datetime.fromtimestamp(now, self.zone)).get_next(datetime).timestamp()

    def status(self):
        return {**self.store.meta("native_world_schedule"), "enabled": self.enabled,
                "schedule": self.config.autonomy.schedule, "timezone": str(self.zone),
                "active": [self.manager.public(r) for r in self.manager.records()
                           if r["data"]["kind"] == "world" and r["data"]["status"] not in TERMINAL]}

    def start(self):
        self.task = asyncio.create_task(self._loop())

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def _loop(self):
        while True:
            await asyncio.sleep(1)
            try:
                await self.opportunity()
            except Exception as exc:
                self.store.event("world_schedule_unavailable", "operator", {"error_class": type(exc).__name__})

    async def opportunity(self, *, manual=False):
        async with self.lock:
            if getattr(self.manager, "maintenance", False):
                return {"status": "maintenance"}
            now = self.store.clock()
            state = self.store.meta("native_world_schedule")
            if not manual and (not self.enabled or now < state["next_at"]):
                return {"status": "waiting", "next_at": state["next_at"]}
            # Advance once from NOW. Restart/overrun never creates catch-up turns.
            self.store.set_meta("native_world_schedule", {**state, "next_at": self.next_after(now)})
            start, end = self.config.autonomy.active_hours_start, self.config.autonomy.active_hours_end
            hour = datetime.fromtimestamp(now, self.zone).hour
            in_hours = start <= hour < end if start < end else hour >= start or hour < end
            if not in_hours:
                return self.skipped("outside_active_hours", now)
            if self.status()["active"]:
                return self.skipped("previous_tick_running", now)
            residents = list(self.manager.hosts)
            last = state.get("last_resident_id")
            offset = (residents.index(last) + 1) % len(residents) if last in residents else 0
            for index in range(len(residents)):
                actor = residents[(offset + index) % len(residents)]
                try:
                    ready = await self.manager.hosts[actor].json("GET", "/_native/ready")
                except (RuntimeError, httpx.TransportError):
                    continue
                if not ready.get("world_due", ready.get("due")) or ready.get("pending"):
                    continue
                if getattr(self.manager, "maintenance", False):
                    return {"status": "maintenance"}
                work = self.manager.submit_world(actor, now)
                latest = self.store.meta("native_world_schedule")
                self.store.set_meta("native_world_schedule", {**latest, "last_resident_id": actor, "last_tick_at": now, "last_work_id": work["work_id"]})
                return {"status": "queued", **work}
            return self.skipped("no_eligible_resident", now)

    def skipped(self, reason, now):
        self.store.event("world_tick_skipped", "operator", {"reason": reason, "scheduled_at": now})
        return {"status": "skipped", "reason": reason}
