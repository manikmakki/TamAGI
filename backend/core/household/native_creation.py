"""Explicit owner-created residents, enrolled without copying another private home."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import unicodedata

import yaml

from backend.config import TamAGIConfig
from backend.core.identity import DEFAULT_IDENTITY, DEFAULT_SOUL, DEFAULT_USER, ONBOARDING_STEPS, VIBE_DESCRIPTIONS
from backend.core.personality import TamAGIState
from backend.core.runtime import RuntimeService, RuntimeStore
from .contracts import Conflict, NotFound, PROTOCOL_VERSION, ProtocolError
from .native_host import NativeHost
from .resident_recovery import JOURNAL as RECOVERY_JOURNAL, atomic_json, sync_directory, sync_tree
from .store import uid

SOURCE = Path(__file__).resolve().parents[3]
JOURNAL = "resident-creation.json"
OPERATION = "household.resident.create"
LIMIT = 16
AVATARS = [{"id": value, "label": value.title()} for value in ("vex", "echo", "ember")]
DEFAULTS = {"name": "", "avatar": "vex", "creature": "digital companion", "vibe": "casual_warm",
            "emoji": "🥚", "values": ["curiosity", "honesty", "efficiency"], "world_setting": ""}
FIELDS = {"tamagi_name": "name", "tamagi_creature": "creature", "tamagi_vibe": "vibe",
          "tamagi_emoji": "emoji", "tamagi_values": "values", "world_setting": "world_setting"}


def creation_schema():
    steps = []
    for original in ONBOARDING_STEPS:
        if original.get("field") not in FIELDS:
            continue
        step = deepcopy(original)
        step["id"] = step["field"] = FIELDS[original["field"]]
        step["default"] = deepcopy(DEFAULTS[step["field"]])
        if step["field"] == "vibe":
            step["options"] = [option for option in step["options"] if option["value"] != "custom"]
        steps.append(step)
    return {"avatars": deepcopy(AVATARS), "steps": steps, "defaults": deepcopy(DEFAULTS), "limit": LIMIT}


def validate_request(body):
    if not isinstance(body, dict) or set(body) - {*DEFAULTS, "request_id"}:
        raise ProtocolError("Choose the new resident's identity and appearance")
    request = {**deepcopy(DEFAULTS), **body}
    if not isinstance(request.get("request_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", request["request_id"]):
        raise ProtocolError("Resident creation requires a stable request ID")
    for key, maximum in (("name", 80), ("creature", 160), ("vibe", 200), ("emoji", 32), ("world_setting", 2000)):
        value = request[key]
        if not isinstance(value, str):
            raise ProtocolError("Resident identity fields must be text")
        value = unicodedata.normalize("NFC", value.strip())
        if (not value and key != "world_setting") or len(value) > maximum:
            raise ProtocolError(f"Choose {key.replace('_', ' ')} within {maximum} characters")
        if any(unicodedata.category(character).startswith("C") and not (key == "world_setting" and character in "\n\t") for character in value):
            raise ProtocolError("Resident identity fields contain unsupported control characters")
        request[key] = value
    if not isinstance(request["avatar"], str) or request["avatar"] not in {avatar["id"] for avatar in AVATARS}:
        raise ProtocolError("Choose one of the available resident avatars")
    if request["vibe"] not in VIBE_DESCRIPTIONS:
        raise ProtocolError("Choose one of the available personality styles")
    values = request["values"]
    if not isinstance(values, list) or len(values) > 8 or any(not isinstance(value, str) or value not in {
        option["value"] for step in ONBOARDING_STEPS if step.get("field") == "tamagi_values" for option in step["options"]
    } for value in values):
        raise ProtocolError("Choose up to eight of the available values")
    request["values"] = list(dict.fromkeys(values))
    return request


def command_id(request):
    return "resident-create:" + request["request_id"]


def public_resident(descriptor):
    return {"id": descriptor["resident_id"], "name": descriptor["name"], "avatar": descriptor["avatar"]}


def existing_result(registry, request):
    try:
        previous = registry.store.command_result("operator", command_id(request))
    except NotFound:
        return None
    if previous["operation"] != OPERATION:
        raise Conflict("This request ID already belongs to another operation")
    # The shared command verifies the original payload fingerprint on a retry.
    return registry.store.command("operator", command_id(request), OPERATION, request, lambda: None)


def prepare_resident(registry, request, root):
    """Provision fresh canonical defaults without the standalone predecessor guard."""
    root.mkdir(mode=0o700)
    try:
        (root / "data").mkdir(mode=0o700)
        coordinator = TamAGIConfig.model_validate(yaml.safe_load((registry.root / "config.yaml").read_text()))
        config = TamAGIConfig()
        config.runtime.enabled = True
        config.runtime.instance_dir = str(root)
        config.runtime.canonical_repo = str(SOURCE)
        config.runtime.authority = "guarded"
        config.runtime.timezone = coordinator.runtime.timezone
        config.runtime.automatic_deployment = config.runtime.index_projection = False
        config.workspace.path = str(root / "workspace")
        config.history.persist_path = str(root / "data/history")
        config.history.context_compress_threshold = 0
        config.history.max_messages_per_conversation = 1000
        config.self_model.data_path = str(root / "data/self_model.json")
        config.world_thread.state_path = str(root / "data/world_state.json")
        config.world_thread.thread_path = str(root / "data/world_thread.json")
        config.world_thread.consolidation.state_path = str(root / "data/consolidation_state.json")
        config.autonomy.enabled = config.world_thread.enabled = config.orchestrator.enabled = False
        config.server.host, config.server.port = "127.0.0.1", 0
        config.auth.enabled = True
        config.tamagi.name = request["name"]
        config.llm.base_url, config.llm.api_key = "http://127.0.0.1:1/native-broker-only", ""
        for field in ("model", "temperature", "max_tokens", "num_ctx"):
            setattr(config.llm, field, getattr(coordinator.llm, field))
        store = RuntimeStore(root / "runtime.sqlite3")
        try:
            runtime = RuntimeService(store, root / "workspace", config.runtime, SOURCE)
            actor = runtime.actor_id
            store.set_meta("household_id", registry.household_id)
            old_actor = store.get("actor", actor)
            store.put("actor", actor, actor, {**old_actor["data"], "name": request["name"],
                "seed": "A newly created resident in this household; no previous memories were imported."})
            vibe = VIBE_DESCRIPTIONS.get(request["vibe"], request["vibe"])
            values = "\n".join(f"- **{value.title()}**" for value in request["values"])
            operator = registry.manifest["operator"]["name"]
            documents = {
                "IDENTITY.md": DEFAULT_IDENTITY.format(name=request["name"], creature=request["creature"],
                    vibe=vibe, emoji=request["emoji"], stage="egg (just hatched!)"),
                "SOUL.md": DEFAULT_SOUL.format(values=values, communication_style=vibe, extra_principles="",
                    what_matters="Beginning a new life in this household. Values, interests and relationships may grow through lived experience. No previous memories have been imported."),
                "USER.md": DEFAULT_USER.format(name=operator, address_as=operator, timezone=config.runtime.timezone,
                    interests="(not shared yet)", comm_pref="(not shared yet)",
                    notes="Newly met. No prior personal relationship is assumed."),
            }
            for filename, content in documents.items():
                store.put("identity", actor + ":" + filename, actor,
                    {"name": filename, "text": content, "reason": "owner-created resident"}, category="platform")
                (root / "workspace" / filename).write_text(content)
            personality = TamAGIState(name=request["name"], personality_traits=vibe + "; values: " + ", ".join(request["values"]))
            store.put("personality", "primary", actor, asdict(personality), category="platform")
            for record in store.list("world_object"):
                data = {**record["data"], "visibility": "private"}
                if record["id"] == "home":
                    data["description"] = request["world_setting"]
                store.put("world_object", record["id"], actor, data, category="platform")
            descriptor = {"id": actor, "resident_id": actor, "installation_id": store.meta("installation_id"),
                "household_id": registry.household_id, "release": registry.release, "protocol": PROTOCOL_VERSION,
                "root": str(root), "config_path": str(root / "config.yaml"), "name": request["name"], "avatar": request["avatar"]}
        finally:
            store.close()
        (root / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
        (root / "config.yaml").chmod(0o600)
        atomic_json(root / "instance.json", descriptor)
        atomic_json(root / "data/sprite_config.json", {"avatar": request["avatar"]})
        sync_tree(root)
        return descriptor
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def reconcile_creation(registry):
    """Complete a durable selected enrollment before workers can resume."""
    path = registry.root / JOURNAL
    if not path.exists():
        return
    if path.is_symlink() or (registry.root / RECOVERY_JOURNAL).exists():
        raise Conflict("Resident creation conflicts with another durable transition")
    transition = json.loads(path.read_text())
    try:
        before, after, descriptor = transition["before"], transition["after"], transition["resident"]
        request = validate_request(transition["request"])
        expected = {**before, "residents": [*before["residents"], descriptor]}
        actor = descriptor["resident_id"]
        if (transition.get("format") != 1 or transition.get("household_id") != registry.household_id
                or before.get("household_id") != registry.household_id or after != expected
                or registry.manifest not in (before, after) or any(item["resident_id"] == actor for item in before["residents"])
                or descriptor["name"] != request["name"] or descriptor["avatar"] != request["avatar"]):
            raise Conflict("Resident creation journal does not match this household")
    except (KeyError, TypeError) as error:
        raise Conflict("Resident creation journal is incomplete") from error
    from .native_registry import NativeRegistry
    NativeRegistry.validate(registry, descriptor)
    result = {"status": "created", "resident": public_resident(descriptor)}

    def enroll():
        existing = registry.store.get("resident", actor)
        if existing is not None and existing["data"] != descriptor:
            raise Conflict("New resident identity conflicts with the household registry")
        if existing is None:
            registry.store.put("resident", actor, descriptor, expected=0)
            registry.store.event("resident_created", "operator", {"resident_id": actor, "name": descriptor["name"]})
        return result

    with registry.store.transaction():
        saved = registry.store.command("operator", command_id(request), OPERATION, request, enroll)
        if saved != result or registry.store.get("resident", actor)["data"] != descriptor:
            raise Conflict("Resident creation request conflicts with its durable result")
    atomic_json(registry.root / "household.json", after)
    registry.manifest = after
    registry.descriptors = {item["resident_id"]: item for item in after["residents"]}
    path.unlink()
    sync_directory(registry.root)


class NativeCreation:
    def __init__(self, registry, recovery, model, hosts, world, coordinator_socket, *, lost=None):
        self.registry, self.recovery, self.model, self.hosts = registry, recovery, model, hosts
        self.world, self.coordinator_socket, self.lost = world, coordinator_socket, lost
        recovery.creation = self

    @staticmethod
    def schema():
        return creation_schema()

    def register(self):
        for actor, descriptor in self.registry.descriptors.items():
            if actor not in self.hosts:
                self.hosts[actor] = NativeHost(self.registry, descriptor, self.coordinator_socket, lost=self.lost)
                self.model.broker.register_resident(actor, available=False)
                self.world.namespaces[actor] = actor
        self.world._seed_domains()

    def reconcile(self):
        reconcile_creation(self.registry)
        self.register()

    def check_new(self, request):
        if len(self.registry.descriptors) >= LIMIT:
            raise Conflict("This household already has sixteen residents")
        names = [item["name"] for item in self.registry.descriptors.values()]
        names.extend([self.registry.manifest["operator"]["name"], "operator", *self.registry.descriptors])
        if request["name"].casefold() in {name.casefold() for name in names}:
            raise Conflict("Choose a name distinct from the household's current participants")

    def availability(self, result):
        host = self.hosts.get(result["resident"]["id"])
        available = host is not None and host.process is not None and host.process.returncode is None
        unavailable = [actor for actor, item in self.hosts.items() if item.process is None or item.process.returncode is not None]
        return {**result, "resident": {**result["resident"], "available": available},
                "workers_available": not unavailable, "unavailable_residents": unavailable}

    async def create(self, body):
        request = validate_request(body)
        if self.recovery.busy or self.recovery.closing:
            raise Conflict("Another household maintenance operation is in progress")
        existing = existing_result(self.registry, request)
        if existing is not None and not (self.registry.root / JOURNAL).exists():
            if not self.availability(existing)["workers_available"]:
                await self.recovery.run(self.registry.manifest["default_resident_id"], "resuming", lambda: existing)
            return self.availability(existing)
        if (self.registry.root / RECOVERY_JOURNAL).exists():
            raise Conflict("Finish the pending resident recovery before creating someone new")
        if not (self.registry.root / JOURNAL).exists():
            self.check_new(request)

        def work():
            self.reconcile()
            prior = existing_result(self.registry, request)
            if prior is not None:
                return prior
            self.check_new(request)
            root = self.registry.root / "residents" / uid("created")
            descriptor = prepare_resident(self.registry, request, root)
            before = deepcopy(self.registry.manifest)
            after = {**before, "residents": [*before["residents"], descriptor]}
            try:
                atomic_json(self.registry.root / JOURNAL, {"format": 1, "household_id": self.registry.household_id,
                    "request": request, "resident": descriptor, "before": before, "after": after})
            except BaseException:
                if not (self.registry.root / JOURNAL).exists():
                    shutil.rmtree(root, ignore_errors=True)
                raise
            self.reconcile()
            return existing_result(self.registry, request)

        result = await self.recovery.run(self.registry.manifest["default_resident_id"], "creating", work)
        return self.availability(result)
