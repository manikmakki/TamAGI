"""Canonical native world dispatch and explicit founding-topology mappings."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

from .contracts import AccessDenied, ProtocolError
from .store import uid
from .world import World


class NativeWorld:
    def __init__(self, registry, model):
        self.registry, self.store, self.model = registry, registry.store, model
        self.world = World(self.store)
        self.namespaces = {actor: actor for actor in registry.descriptors}
        self._seed_domains()

    def _seed_domains(self):
        common = "household:commons"
        self.world.create("operator", "native-world:commons", common, "place", "Household commons", visibility="shared")
        for actor, descriptor in self.registry.descriptors.items():
            namespace = self.namespaces[actor]
            prior = self.store.get("world_import", actor + ":" + namespace)
            if prior is None:
                source = Path(descriptor["root"]) / "runtime.sqlite3"
                database = sqlite3.connect(f"file:{quote(str(source), safe='/')}?mode=ro", uri=True)
                try:
                    database.execute("PRAGMA query_only=ON")
                    rows = database.execute("SELECT id,payload FROM records WHERE kind='world_object' ORDER BY created,id").fetchall()
                    records = []
                    for identity, payload in rows:
                        data = json.loads(payload)
                        records.append({"id": identity, "kind": data["kind"], "name": data["name"],
                            "location": data.get("location"), "visibility": data.get("visibility", "shared"),
                            "attributes": {"description": data.get("description", ""), "properties": data.get("properties", {})}})
                finally:
                    database.close()
                if records:
                    self.world.import_records(actor, "native-world:founding:" + actor, namespace, records)
            mapping = self.store.get("world_import", actor + ":" + namespace)
            home = mapping["data"]["mapping"].get("home") if mapping else None
            if home:
                def connect(actor=actor, home=home):
                    record = self.world.get(actor, home)
                    if record["data"].get("location") is None:
                        return self.world.update(actor, "native-world:connect-update:" + actor,
                            home, {"location": common}, record["revision"])
                    return record
                self.store.command(actor, "native-world:connect:" + actor, "native.world.connect",
                                   {"home": home, "common": common}, connect)

    def _resolve(self, resident, identity):
        if not identity:
            return None
        if not isinstance(identity, str):
            raise ProtocolError("World object identity must be text")
        mapping = self.store.get("world_import", resident + ":" + self.namespaces[resident])
        return mapping["data"]["mapping"].get(identity, identity) if mapping else identity

    @staticmethod
    def _attributes(supplied, previous=None):
        attributes = dict(previous or {})
        if "attributes" in supplied:
            if not isinstance(supplied["attributes"], dict):
                raise ProtocolError("World attributes must be an object")
            attributes.update(supplied["attributes"])
        for key in ("description", "properties"):
            if key in supplied:
                attributes[key] = supplied[key]
        return attributes

    def dispatch(self, resident, body):
        self.registry.resolve(resident)
        if not isinstance(body, dict) or set(body) != {"grant_id", "local_execution_id", "action", "arguments", "command_id"}:
            raise ProtocolError("Invalid native world command")
        action, arguments = body["action"], body["arguments"]
        if action not in {"inspect", "move", "create", "revise", "place"} or not isinstance(arguments, dict) or set(arguments) - {"object_id", "attributes", "expected"}:
            raise ProtocolError("Unsupported native world operation")
        actor = resident
        if body["local_execution_id"]:
            _, execution, _ = self.model._active(resident, body["local_execution_id"])
            actor = execution["actor_id"]
        elif action != "inspect":
            raise AccessDenied("A shared world mutation needs a current execution")
        identity = self._resolve(resident, arguments.get("object_id", ""))
        if action == "inspect":
            return self.world.get(actor, identity) if identity else self.world.list(actor)
        command = body["command_id"]
        if not isinstance(command, str) or not 1 <= len(command) <= 256:
            raise ProtocolError("A world mutation needs a stable command identity")
        return self.store.command(actor, command, "native.world.dispatch",
            {"action": action, "arguments": arguments},
            lambda: self._mutate(resident, actor, action, identity, arguments, command + ":effect"))

    def _mutate(self, resident, actor, action, identity, arguments, command):
        supplied = arguments.get("attributes") or {}
        if not isinstance(supplied, dict) or set(supplied) - {"kind", "name", "location", "description", "properties", "attributes", "visibility", "visible_to", "artifact_id"}:
            raise ProtocolError("Unsupported world attributes")
        if supplied.get("artifact_id"):
            raise ProtocolError("Publish a verified immutable artifact snapshot before attaching it to the shared world")
        if action == "move":
            def move():
                target = self.world.get(actor, identity)
                if target["data"]["kind"] != "place":
                    raise ProtocolError("Choose a visible place")
                self.store.put("world_position", actor, {"actor_id": actor, "location": identity})
                self.store.event("resident_moved", actor, {"location": identity})
                return {"target": target}
            return self.store.command(actor, command, "native.world.move", {"location": identity}, move)
        if action == "create":
            namespace = self.namespaces[resident] + ":"
            raw_id = arguments.get("object_id") or uid("object")
            self.world._identifier(raw_id)
            if ":" in raw_id and not raw_id.startswith(namespace):
                raise AccessDenied("Create world IDs in the current resident's namespace")
            identity = raw_id if raw_id.startswith(namespace) else namespace + raw_id
            return self.world.create(actor, command, identity, supplied.get("kind", "object"), supplied.get("name", ""),
                visibility=supplied.get("visibility", "shared"), visible_to=supplied.get("visible_to"),
                attributes=self._attributes(supplied), location=self._resolve(resident, supplied.get("location", "home")))
        old = self.world.get(actor, identity)
        if "kind" in supplied and supplied["kind"] != old["data"]["kind"]:
            raise ProtocolError("World object kind cannot be rewritten")
        changes = {key: supplied[key] for key in ("name", "visibility", "visible_to") if key in supplied}
        if "location" in supplied:
            changes["location"] = self._resolve(resident, supplied["location"])
        if set(supplied) & {"attributes", "description", "properties"}:
            changes["attributes"] = self._attributes(supplied, old["data"]["attributes"])
        expected = arguments.get("expected")
        if expected is None:
            # Legacy graph adapters submit field-level changes without carrying a
            # revision. Read and apply that narrow patch in one coordinator command.
            expected = old["revision"]
        return self.world.update(actor, command, identity, changes, expected)
