"""Resident-side canonical conversation adapter for the native application.

Only the coordinator's authenticated local transport supplies admissions. Private
records below are observed projections and an outbox, never a second chat authority.
"""

from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from urllib.parse import quote

import httpx

from .contracts import AccessDenied, Conflict, ProtocolError, WorkerBinding
from .store import uid


def admission_context():
    from .model_transport import current_household_admission

    return current_household_admission


def record_data(record):
    if not isinstance(record, dict):
        raise ProtocolError("A canonical record must be an object")
    if "data" not in record:
        return dict(record)
    return {**{key: record[key] for key in ("created", "updated") if key in record},
            **record.get("data", {}), "id": record["id"]}


def validate_native_binding(binding: WorkerBinding, config_path: Path | None = None):
    root = Path(binding.root)
    if not root.is_absolute() or root != root.resolve() or not root.is_dir():
        raise AccessDenied("Native resident root must be explicit and canonical")
    for path in (root / "runtime.sqlite3", root / "instance.json", root / "workspace"):
        if not path.exists() or path.is_symlink():
            raise AccessDenied("Native resident state is missing or uses a symlink")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise AccessDenied("Native resident projections cannot redirect through symlinks")
    manifest = json.loads((root / "instance.json").read_text())
    for key in ("resident_id", "installation_id", "household_id", "release", "protocol"):
        if manifest.get(key) != getattr(binding, key):
            raise Conflict("Native resident manifest does not match its worker binding")
    household = root.parent.parent
    marker_path = household / "household.json"
    if root.parent != household / "residents" or marker_path.is_symlink():
        raise AccessDenied("Native worker must belong to an explicit household")
    marker = json.loads(marker_path.read_text())
    if marker.get("household_id") != binding.household_id or marker.get("release") != binding.release or marker.get("protocol") != binding.protocol:
        raise Conflict("Native household identity/release does not match the worker")
    descriptors = marker.get("residents", [])
    if not any(descriptor.get("resident_id", descriptor.get("id")) == binding.resident_id
               and descriptor.get("root") == binding.root for descriptor in descriptors):
        raise AccessDenied("This resident is not enrolled in the selected household")
    if config_path is not None and (config_path != config_path.resolve() or config_path.parent != root or not config_path.is_file()):
        raise AccessDenied("Worker configuration must belong to its resident root")
    database = sqlite3.connect(f"file:{quote(str(root / 'runtime.sqlite3'), safe='/')}?mode=ro", uri=True)
    try:
        database.execute("PRAGMA query_only=ON")
        metadata = {key: json.loads(value) for key, value in database.execute(
            "SELECT key,value FROM meta WHERE key IN ('resident_id','installation_id','household_id')"
        )}
    finally:
        database.close()
    if metadata != {"resident_id": binding.resident_id, "installation_id": binding.installation_id, "household_id": binding.household_id}:
        raise Conflict("Native database identity does not match its worker binding")
    return marker


class NativeWorkerBridge:
    def __init__(self, runtime, binding, uds_path, token, model_client, *, transport=None):
        if runtime.actor_id != binding.resident_id or runtime.workspace.parent != Path(binding.root):
            raise AccessDenied("Native bridge cannot bind another resident's runtime")
        self.runtime, self.binding, self.model_client = runtime, binding, model_client
        marker = json.loads((Path(binding.root).parent.parent / "household.json").read_text())
        self.names = {descriptor.get("resident_id", descriptor.get("id")): descriptor.get("name", "TamAGI")
                      for descriptor in marker["residents"]}
        self.names["operator"] = marker.get("operator", {}).get("name", "Operator")
        self.agent = None
        self.client = httpx.Client(base_url="http://household", timeout=20,
            transport=transport or httpx.HTTPTransport(uds=str(uds_path)),
            headers={"X-Tamagi-Worker": token})

    def __getattr__(self, name):
        # Runtime lifecycle, shared resources and continuation hooks use the same
        # canonical model transport; there is no independent worker allowance.
        if name not in {"start_execution", "end_execution", "resources", "lineage", "chat_for", "request_quiet", "advance_quiet"}:
            raise AttributeError(name)
        return getattr(self.model_client, name)

    def assert_active(self, context):
        return self.model_client.assert_active(context)

    def _request(self, method, path, **kwargs):
        response = self.client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise AccessDenied("Household coordinator rejected this resident operation")
        return response.json()

    def validate_admission(self, grant, snapshot):
        grant = copy.deepcopy(grant)
        identity = grant.get("id") or grant.get("grant_id")
        if not isinstance(identity, str) or not identity or len(identity) > 256:
            raise ProtocolError("A household admission needs a stable grant ID")
        grant["id"] = identity
        if grant.get("resident_id", self.runtime.actor_id) != self.runtime.actor_id:
            raise AccessDenied("Admission belongs to another resident")
        if grant.get("origin") not in {"user", "autonomy"}:
            raise ProtocolError("Unknown household exchange origin")
        if not isinstance(grant.get("exchange_id"), str) or not grant["exchange_id"]:
            raise ProtocolError("Admission needs its canonical exchange")
        conversation = record_data(snapshot.get("conversation", {}))
        conversation_id = grant.get("conversation_id")
        if conversation.get("id") != conversation_id or self.runtime.actor_id not in conversation.get("participants", []):
            raise AccessDenied("Admission does not grant access to this conversation")
        through = grant.get("through")
        if type(through) is not int or through < 1:
            raise ProtocolError("Admission needs an explicit transcript watermark")
        if snapshot.get("watermark", through) != through:
            raise ProtocolError("Admission and transcript watermarks differ")
        messages = [record_data(message) for message in snapshot.get("messages", [])]
        previous, identities = 0, set()
        for message in messages:
            sequence = message.get("sequence")
            if type(sequence) is not int or not previous < sequence <= through or message.get("id") in identities:
                raise ProtocolError("Canonical transcript is unordered or exceeds its admission")
            if not isinstance(message.get("id"), str) or not isinstance(message.get("content"), str) or not isinstance(message.get("author"), str):
                raise ProtocolError("Canonical messages require stable attribution")
            previous = sequence
            identities.add(message["id"])
        trigger = record_data(grant.get("message", {}))
        matching = next((message for message in messages if message["id"] == trigger.get("id")), None)
        if matching is None or any(trigger.get(key) != matching.get(key) for key in ("id", "author", "content", "sequence")):
            raise ProtocolError("Admitted input is not the selected canonical message")
        grant["message"] = matching
        grant["input_author"] = matching["author"]
        return grant, {"conversation": conversation, "messages": messages, "watermark": through}

    def _message(self, message):
        from backend.core.agent import Message

        message = record_data(message)
        author = message["author"]
        metadata = {**message.get("metadata", {}), "author_id": author,
            "author_name": message.get("author_name") or message.get("speaker") or self.names.get(author, author),
            "canonical": True, "sequence": message["sequence"],
            "participants": message.get("participants", []), "exchange_id": message.get("lineage", {}).get("exchange_id"),
            "conversation_id": message["conversation_id"]}
        return Message("assistant" if author == self.runtime.actor_id else "user", message["content"],
                       id=message["id"], timestamp=message.get("timestamp", 0), metadata=metadata)

    def project(self, agent, snapshot, grant):
        from backend.core.agent import Conversation

        grant, snapshot = self.validate_admission(grant, snapshot)
        incoming = snapshot["conversation"]
        old = agent.conversations.get(incoming["id"])
        conversation = Conversation(id=incoming["id"], title=incoming.get("title", "Conversation"),
            created_at=incoming.get("created_at", incoming.get("created", 0)),
            updated_at=max((message.get("timestamp", 0) for message in snapshot["messages"]), default=0),
            messages=[self._message(message) for message in snapshot["messages"]])
        if old is not None:
            for field in ("world_context_initialized", "world_context_summary", "world_context_turn_count", "world_context_latest_turn_id", "memory_archived_through"):
                setattr(conversation, field, getattr(old, field))
        agent.conversations[conversation.id] = conversation
        return conversation

    def input_record(self, grant):
        return self._message(grant["message"]).to_dict()

    @staticmethod
    def speaker(message):
        metadata = message.metadata
        return metadata.get("author_name") or metadata.get("author_id") or ("User" if message.role == "user" else "TamAGI")

    def provider_content(self, message):
        if message.metadata.get("author_id") == self.runtime.actor_id:
            return message.content
        text = f"[Attributed message from {self.speaker(message)}; participant_id={message.metadata.get('author_id', 'operator')}]\n{message.content}"
        if image := message.metadata.get("image"):
            grant = admission_context().get() or {}
            source = self._request("GET", "/_native/conversations/" + quote(message.metadata["conversation_id"], safe="")
                + "/attachments/" + quote(message.id, safe=""), params={"grant_id": grant.get("id", "")})
            content = base64.b64decode(source["image_data"], validate=True)
            if len(content) != image["size"] or hashlib.sha256(content).hexdigest() != image["sha256"]:
                raise ProtocolError("Attached image content changed")
            return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {
                "url": f"data:{image['media_type']};base64,{source['image_data']}"}}]
        return text

    def save_conversation(self, conversation):
        current = admission_context().get()
        # Silent terminal outcomes belong to the retained execution record.
        # They create neither an empty shared message nor an unread alert.
        conversation.messages[:] = [message for message in conversation.messages
            if message.metadata.get("canonical") or message.role != "assistant" or message.content.strip()]
        for index, message in enumerate(list(conversation.messages)):
            if message.metadata.get("canonical"):
                continue
            if message.role != "assistant":
                raise AccessDenied("Workers cannot author canonical incoming messages")
            if not current or current.get("conversation_id") != conversation.id:
                raise AccessDenied("A shared response needs its admitted conversation")
            command_id = "message:" + message.id
            body = {"grant_id": current.get("id") or current.get("grant_id"),
                    "local_message": message.to_dict(), "command_id": command_id}
            self.runtime.store.put("household_outbox", command_id, self.runtime.actor_id,
                {"status": "pending", "payload": body}, category="platform")
            try:
                result = self._request("POST", "/_native/conversations/commit", json=body)
                canonical = record_data(result.get("message", result))
                if canonical.get("author") != self.runtime.actor_id or canonical.get("conversation_id") != conversation.id:
                    raise ProtocolError("Coordinator returned another author's message")
                conversation.messages[index] = self._message(canonical)
                self.runtime.store.put("household_outbox", command_id, self.runtime.actor_id,
                    {"status": "committed", "payload": body, "message_id": canonical["id"]}, category="platform")
            except Exception:
                self.runtime.store.put("household_outbox", command_id, self.runtime.actor_id,
                    {"status": "outcome_unknown", "payload": body}, category="platform")
                raise
        self.runtime.store.put("household_projection", conversation.id, self.runtime.actor_id,
            {"canonical_projection": True, "conversation": conversation.to_dict()}, category="platform")
        self.runtime.growth.observe_conversation(conversation.to_dict())

    def load_conversations(self, agent):
        from backend.core.agent import Conversation, Message

        for record in self.runtime.store.list("household_projection", self.runtime.actor_id, limit=100000):
            payload = record["data"]
            if payload.get("canonical_projection") is not True:
                continue
            value = dict(payload["conversation"])
            value.pop("message_count", None)
            value["messages"] = [Message(**message) for message in value.get("messages", [])]
            allowed = Conversation.__dataclass_fields__
            conversation = Conversation(**{key: item for key, item in value.items() if key in allowed})
            agent.conversations[conversation.id] = conversation

    def fetch_conversation(self, conversation_id):
        from backend.core.agent import Conversation

        grant = admission_context().get() or {}
        parameters = {"grant_id": grant.get("id", grant.get("grant_id", ""))}
        if grant.get("conversation_id") == conversation_id:
            parameters["through"] = grant["through"]
        result = self._request("GET", "/_native/conversations/" + quote(conversation_id, safe=""), params=parameters)
        metadata = record_data(result["conversation"])
        return Conversation(id=metadata["id"], title=metadata.get("title", "Conversation"),
                            messages=[self._message(message) for message in result["messages"]])

    def world_action(self, action, object_id="", attributes=None, expected=None):
        from backend.core.runtime import current_execution

        context = current_execution.get()
        if action != "inspect" and context is None:
            raise AccessDenied("A shared world mutation needs an admitted execution")
        if context is not None and action != "inspect":
            self.assert_active(context)
        receipt = next((receipt for receipt in self.runtime.store.receipts(context.execution_id, limit=100)
                        if receipt["status"] == "started" and receipt["tool"] in {"world_action", "write_world_graph"}), None) if context else None
        command_id = "world:" + (receipt["id"] if receipt else uid("operation"))
        grant = admission_context().get() or {}
        payload = {"grant_id": grant.get("id", grant.get("grant_id")),
                   "local_execution_id": context.execution_id if context else None,
                   "action": action, "arguments": {"object_id": object_id, "attributes": attributes or {}, "expected": expected},
                   "command_id": command_id}
        if action != "inspect":
            self.runtime.store.put("household_outbox", command_id, self.runtime.actor_id,
                {"status": "pending", "payload": payload}, category="platform")
        try:
            result = self._request("POST", "/_native/world", json=payload)
            if action != "inspect":
                self.runtime.store.put("household_outbox", command_id, self.runtime.actor_id,
                    {"status": "committed", "payload": payload}, category="platform")
        except Exception:
            if action != "inspect":
                self.runtime.store.put("household_outbox", command_id, self.runtime.actor_id,
                    {"status": "outcome_unknown", "payload": payload}, category="platform")
            raise
        # A resident's current location is personal state; the place itself is
        # canonical. The coordinator returns its validated stable place identity.
        if action == "move":
            target = result.get("target", result)
            if not isinstance(target, dict) or not isinstance(target.get("id"), str):
                raise ProtocolError("Coordinator did not return the selected place")
            actor = context.actor_id
            state = self.runtime.store.get("activity", actor)
            result = self.runtime.store.put("activity", actor, actor,
                {**(state["data"] if state else {}), "location": target["id"]}, category="platform")
        return result

    def notebook(self, action="inspect", text="", source_ids=None):
        from backend.core.runtime import current_execution

        if action not in {"inspect", "publish"}:
            raise ProtocolError("Unsupported shared notebook operation")
        context = current_execution.get()
        if action == "publish":
            if context is None:
                raise AccessDenied("Publishing a shared note needs an admitted execution")
            self.assert_active(context)
        receipt = next((receipt for receipt in self.runtime.store.receipts(context.execution_id, limit=100)
                        if receipt["status"] == "started" and receipt["tool"] == "share_note"), None) if context else None
        command = "notebook:" + (receipt["id"] if receipt else uid("operation"))
        payload = {"local_execution_id": context.execution_id if context else None,
                   "action": action, "text": text, "source_ids": source_ids or [], "command_id": command}
        if action == "publish":
            self.runtime.store.put("household_outbox", command, self.runtime.actor_id,
                {"status": "pending", "payload": payload}, category="platform")
        try:
            result = self._request("POST", "/_native/notebook", json=payload)
        except Exception:
            if action == "publish":
                self.runtime.store.put("household_outbox", command, self.runtime.actor_id,
                    {"status": "outcome_unknown", "payload": payload}, category="platform")
            raise
        if action == "publish":
            self.runtime.store.put("household_outbox", command, self.runtime.actor_id,
                {"status": "committed", "payload": payload}, category="platform")
        return result

    def close(self):
        self.client.close()


def build_worker_client(config, runtime):
    from .model_transport import NativeModelClient

    binding = WorkerBinding.parse(json.loads(os.environ["TAMAGI_NATIVE_BINDING"]))
    validate_native_binding(binding, Path(os.environ["TAMAGI_CONFIG"]))
    token = os.environ["TAMAGI_NATIVE_TOKEN"]
    uds_path = os.environ["TAMAGI_NATIVE_COORDINATOR_SOCKET"]
    client = NativeModelClient(uds_path, token, binding.resident_id, config.llm)
    runtime.household = NativeWorkerBridge(runtime, binding, uds_path, token, client)
    return client


def install_worker(runtime, agent, config):
    bridge = getattr(runtime, "household", None)
    if bridge is None:
        raise AccessDenied("Native worker started without its canonical bridge")
    bridge.agent = agent
    # NativeAuthentication and model/bridge transports have captured their private
    # credentials. Ordinary exec/generated-tool children must not inherit them.
    for variable in ("TAMAGI_NATIVE_TOKEN", "TAMAGI_NATIVE_BINDING", "TAMAGI_NATIVE_COORDINATOR_SOCKET"):
        os.environ.pop(variable, None)


class NativeDeploymentDisabled:
    """Personal runtime inspection remains usable; activation belongs centrally."""

    def __init__(self, runtime):
        self.root = runtime.workspace.parent / "household-deployment-disabled"

    def synchronize(self):
        return None

    def selected(self):
        return None

    def __getattr__(self, name):
        def disabled(*args, **kwargs):
            raise ValueError("Household deployment requires a qualified coordinated controller")
        return disabled
