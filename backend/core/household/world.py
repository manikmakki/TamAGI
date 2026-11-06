"""Revisioned shared topology and explicitly published immutable artifact references."""

from __future__ import annotations

import re
import uuid

from .contracts import AccessDenied, Conflict, NotFound, ProtocolError


class World:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _visible(actor, data):
        return data["owner"] == actor or data["visibility"] == "shared" or actor in data.get("visible_to", [])

    def get(self, actor, object_id):
        record = self.store.get("world_object", object_id)
        if not record:
            raise NotFound("Unknown world object")
        if not self._visible(actor, record["data"]):
            raise AccessDenied("This world object is private")
        return record

    def list(self, actor):
        return [record for record in self.store.list("world_object") if self._visible(actor, record["data"])]

    def _changed(self, actor, record, operation):
        self.store.event("world_changed", actor,
                         {"object_id": record["id"], "revision": record["revision"], "operation": operation})
        return record

    @staticmethod
    def _identifier(value):
        if not isinstance(value, str) or not value or len(value) > 256 or "\0" in value:
            raise ProtocolError("Choose a nonempty stable object ID of at most 256 characters")

    def _validate(self, actor, data, object_id):
        if data.get("kind") not in {"place", "object"}:
            raise ProtocolError("World entries must be a place or object")
        if not isinstance(data.get("name"), str) or not data["name"].strip() or len(data["name"]) > 200:
            raise ProtocolError("World entries need a name of at most 200 characters")
        if data.get("visibility") not in {"private", "shared"}:
            raise ProtocolError("Choose private or shared visibility")
        if not isinstance(data.get("visible_to", []), list) or any(not isinstance(p, str) or not p for p in data.get("visible_to", [])):
            raise ProtocolError("Visibility grants must be explicit participant IDs")
        if not isinstance(data.get("attributes", {}), dict):
            raise ProtocolError("World attributes must be an object")
        location = data.get("location")
        if location is not None:
            self._identifier(location)
        visited = {object_id}
        while location:
            if location in visited:
                raise Conflict("World locations cannot form a cycle")
            visited.add(location)
            parent = self.get(actor, location)
            if parent["data"]["kind"] != "place":
                raise Conflict("Only a place can contain a world object")
            location = parent["data"].get("location")

    def create(self, actor, command_id, object_id, kind, name, visibility="private",
               visible_to=None, attributes=None, location=None):
        payload = {"object_id": object_id, "kind": kind, "name": name,
                   "visibility": visibility, "visible_to": visible_to,
                   "attributes": attributes, "location": location}

        def apply():
            self._identifier(object_id)
            if self.store.get("world_object", object_id):
                raise Conflict("World object ID already exists")
            data = {"kind": kind, "name": name, "owner": actor, "visibility": visibility,
                    "visible_to": visible_to if visible_to is not None else [],
                    "attributes": attributes if attributes is not None else {},
                    "location": location, "artifact_refs": []}
            self._validate(actor, data, object_id)
            return self._changed(actor, self.store.put("world_object", object_id, data), "create")

        return self.store.command(actor, command_id, "world.create", payload, apply)

    def update(self, actor, command_id, object_id, changes, expected):
        payload = {"object_id": object_id, "changes": changes, "expected": expected}

        def apply():
            record = self.get(actor, object_id)
            if record["data"]["owner"] != actor:
                raise AccessDenied("Only the owner may revise this world object")
            if type(expected) is not int or expected < 1:
                raise ProtocolError("World revisions require an expected revision")
            if not isinstance(changes, dict) or set(changes) - {"name", "visibility", "visible_to", "attributes", "location"}:
                raise ProtocolError("Object identity, owner, kind, and artifact references cannot be rewritten")
            data = {**record["data"], **changes}
            self._validate(actor, data, object_id)
            return self._changed(actor, self.store.put("world_object", object_id, data, expected=expected), "update")

        return self.store.command(actor, command_id, "world.update", payload, apply)

    def publish_artifact(self, actor, command_id, object_id, artifact_id, sha256,
                         expected, filename="", size=None):
        payload = {"object_id": object_id, "artifact_id": artifact_id, "sha256": sha256,
                   "expected": expected, "filename": filename, "size": size}

        def apply():
            record = self.get(actor, object_id)
            if record["data"]["owner"] != actor:
                raise AccessDenied("Only the owner may publish an artifact through this object")
            if type(expected) is not int or expected < 1:
                raise ProtocolError("Publishing requires an expected object revision")
            self._identifier(artifact_id)
            if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise ProtocolError("Artifact references require an immutable SHA-256 digest")
            if not isinstance(filename, str) or "/" in filename or "\\" in filename or filename in {".", ".."}:
                raise ProtocolError("Publish a filename, not a private workspace path")
            if size is not None and (type(size) is not int or size < 0):
                raise ProtocolError("Artifact size must be a nonnegative integer")
            reference_id = actor + ":" + artifact_id
            reference = {"owner": actor, "artifact_id": artifact_id, "sha256": sha256,
                         "filename": filename, "size": size}
            prior = self.store.get("world_artifact", reference_id)
            if prior and prior["data"] != reference:
                raise Conflict("A published artifact reference is immutable; publish a new snapshot ID")
            if not prior:
                self.store.put("world_artifact", reference_id, reference)
            if reference_id in record["data"]["artifact_refs"]:
                if expected != record["revision"]:
                    raise Conflict("World object revision changed")
                return {"object": record, "artifact": self.store.get("world_artifact", reference_id)}
            updated = self.store.put("world_object", object_id,
                                     {**record["data"], "artifact_refs": [*record["data"]["artifact_refs"], reference_id]},
                                     expected=expected)
            self._changed(actor, updated, "publish_artifact")
            return {"object": updated, "artifact": self.store.get("world_artifact", reference_id)}

        return self.store.command(actor, command_id, "world.publish_artifact", payload, apply)

    def artifact(self, actor, object_id, reference_id):
        record = self.get(actor, object_id)
        if reference_id not in record["data"]["artifact_refs"]:
            raise NotFound("Artifact is not published through this object")
        artifact = self.store.get("world_artifact", reference_id)
        if not artifact:
            raise NotFound("Published artifact reference is unavailable")
        return artifact

    def import_records(self, actor, command_id, namespace, records):
        payload = {"namespace": namespace, "records": records}

        def apply():
            if not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", namespace):
                raise ProtocolError("Imports need an explicit stable namespace")
            if not isinstance(records, list) or not records:
                raise ProtocolError("Import at least one explicit source record")
            mapping_id = actor + ":" + namespace
            prior = self.store.get("world_import", mapping_id)
            if prior:
                if prior["data"]["source_records"] != records:
                    raise Conflict("This namespace has already imported a different source manifest")
                return prior
            mapping = {}
            for source in records:
                if not isinstance(source, dict):
                    raise ProtocolError("Import source records must be objects")
                source_id = source.get("id")
                self._identifier(source_id)
                if source_id in mapping:
                    raise Conflict("Duplicate source ID in import")
                target_id = namespace + ":" + source_id
                self._identifier(target_id)
                if self.store.get("world_object", target_id):
                    raise Conflict("Namespaced destination already exists; names are never merged")
                mapping[source_id] = target_id
            # All imported locations are remapped through this exact manifest.
            # Persist in dependency order and reject missing parents or cycles.
            pending = list(records)
            while pending:
                progressed = False
                for source in pending[:]:
                    location = source.get("location")
                    if location and location not in mapping:
                        raise NotFound("Imported location is absent from the source manifest")
                    if location and not self.store.get("world_object", mapping[location]):
                        continue
                    target_id = mapping[source["id"]]
                    allowed = {"id", "kind", "name", "visibility", "visible_to", "attributes", "location"}
                    if set(source) - allowed:
                        raise ProtocolError("Import explicit topology only; publish artifacts and identity separately")
                    data = {"kind": source.get("kind"), "name": source.get("name"), "owner": actor,
                            "visibility": source.get("visibility", "private"),
                            "visible_to": source.get("visible_to", []), "attributes": source.get("attributes", {}),
                            "location": mapping.get(location), "artifact_refs": [],
                            "provenance": {"namespace": namespace, "source_id": source["id"]}}
                    self._validate(actor, data, target_id)
                    self._changed(actor, self.store.put("world_object", target_id, data), "import")
                    pending.remove(source)
                    progressed = True
                if not progressed:
                    raise Conflict("Imported topology contains a location cycle")
            return self.store.put("world_import", mapping_id,
                                  {"owner": actor, "namespace": namespace, "mapping": mapping,
                                   "source_records": records})

        return self.store.command(actor, command_id, "world.import", payload, apply)
