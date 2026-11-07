"""Canonical attributed conversations; delivery never implies cognition or reading.

The coordinator supplies every author and execution lineage. This service does not
schedule models, mint allowances, or turn a peer message into operator input.
"""

from __future__ import annotations

import hashlib
import re
import uuid

from .contracts import AccessDenied, Conflict, NotFound, ProtocolError


def _id(prefix):
    return prefix + "-" + uuid.uuid4().hex


def _stable(prefix, *parts):
    return prefix + "-" + hashlib.sha256("\0".join(parts).encode()).hexdigest()


class Conversations:
    def __init__(self, store, participants=None, validate_lineage=None):
        self.store = store
        self._participants = participants
        self.validate_lineage = validate_lineage

    def _roster(self):
        if callable(self._participants):
            participants = self._participants()
        elif self._participants is not None:
            participants = self._participants
        else:
            participants = [
                {"id": record["id"], **record["data"]}
                for record in self.store.list("resident")
            ] + [{"id": "operator", "name": "Operator", "kind": "human"}]
        return {p["id"]: dict(p) for p in participants}

    def _participant(self, actor):
        participant = self._roster().get(actor)
        if not participant:
            raise NotFound("Unknown conversation participant")
        return participant

    def _conversation(self, conversation_id, viewer=None):
        record = self.store.get("conversation", conversation_id)
        if not record:
            raise NotFound("Unknown conversation")
        if viewer is not None and viewer not in record["data"]["participants"]:
            raise AccessDenied("Conversation is not visible to this participant")
        return record

    def _human(self, actor):
        if self._participant(actor).get("kind") != "human":
            raise AccessDenied("Read acknowledgments belong to human participants")

    def _lineage(self, author, lineage):
        participant = self._participant(author)
        if lineage is not None and not isinstance(lineage, dict):
            raise ProtocolError("Execution lineage must be a trusted object")
        supplied = dict(lineage or {})
        if participant.get("kind") != "human":
            if self.validate_lineage is None:
                raise AccessDenied("Resident messages require a trusted lineage validator")
            checked = self.validate_lineage(supplied, author)
            trusted = dict(checked if checked is not None else supplied)
            if not trusted.get("execution_id") or not trusted.get("exchange_id"):
                raise ProtocolError("Validated resident lineage needs an execution and exchange")
            for key in ("execution_id", "exchange_id", "root_id"):
                if supplied.get(key) and supplied[key] != trusted.get(key):
                    raise AccessDenied("Message lineage does not match its execution")
            return trusted
        # Human-origin exchange admission is the coordinator's responsibility too.
        # Plain operator messages need no inference to be persisted.
        return supplied

    @staticmethod
    def _content(content):
        if not isinstance(content, str) or not content.strip() or len(content) > 32000:
            raise ProtocolError("A message needs 1–32000 characters")
        return content

    @staticmethod
    def _context_refs(references):
        if references is None:
            return []
        if not isinstance(references, list) or len(references) > 16:
            raise ProtocolError("Share at most sixteen explicit source references")
        for reference in references:
            if not isinstance(reference, dict) or set(reference) - {"kind", "id", "summary"}:
                raise ProtocolError("Shared context must use attributed source references")
            if reference.get("kind") not in {"message", "note", "artifact", "world_object"}:
                raise ProtocolError("Unsupported shared-context source kind")
            if not isinstance(reference.get("id"), str) or not 1 <= len(reference["id"]) <= 256:
                raise ProtocolError("Shared context needs a stable source ID")
            if not isinstance(reference.get("summary", ""), str) or len(reference.get("summary", "")) > 2000:
                raise ProtocolError("Keep each shared source summary bounded")
        return [dict(reference) for reference in references]

    def _addressees(self, members, content, mentions):
        if mentions is None:
            mentions = re.findall(r"(?<![\w@])@([^\s@.,!?;:()\[\]{}<>]+)", content)
        if isinstance(mentions, str) or not isinstance(mentions, (list, tuple)):
            raise ProtocolError("Mentions must be a list of participant names or IDs")
        if not mentions:
            return None
        roster = self._roster()
        resolved = []
        for mention in mentions:
            if not isinstance(mention, str) or not mention.strip():
                raise ProtocolError("Invalid mention")
            name = mention.removeprefix("@").casefold()
            exact = [mention] if mention in roster else []
            candidates = exact or [
                actor for actor, participant in roster.items()
                if any(str(alias).casefold() == name for alias in
                       [participant.get("name", ""), *participant.get("aliases", [])])
            ]
            if len(candidates) != 1:
                raise ProtocolError("Mention is unknown or ambiguous; choose a stable participant ID")
            if candidates[0] not in members:
                raise AccessDenied("Mentioned participant has not joined this conversation")
            if candidates[0] not in resolved:
                resolved.append(candidates[0])
        return resolved

    def create(self, author, participants, command_id, title="", group=False,
               opening=None, mentions=None, lineage=None, context_refs=None):
        payload = {"participants": participants, "title": title, "group": group,
                   "opening": opening, "mentions": mentions, "lineage": lineage,
                   "context_refs": context_refs}

        def apply():
            self._participant(author)
            if not isinstance(participants, (list, tuple)):
                raise ProtocolError("Choose conversation participants explicitly")
            if any(not isinstance(member, str) or not member for member in participants):
                raise ProtocolError("Participants must be stable participant IDs")
            if type(group) is not bool:
                raise ProtocolError("Group selection must be a boolean")
            members = list(dict.fromkeys([author, *participants]))
            if len(members) < 2 or (not group and len(members) != 2):
                raise ProtocolError("Private conversations need two participants; use a group for more")
            for member in members:
                self._participant(member)
            if not isinstance(title, str) or len(title) > 200:
                raise ProtocolError("Conversation title must be at most 200 characters")
            # Validate inside command.apply: a committed replay remains recoverable
            # after its initiating execution was stopped or exhausted.
            trusted = self._lineage(author, lineage)
            if opening is None and self._participant(author).get("kind") != "human":
                raise ProtocolError("Resident initiation needs a visible opening message")
            conversation_id = _id("conversation")
            now = self.store.clock()
            data = {"title": title, "group": bool(group), "created_by": author,
                    "participants": members, "membership_version": 1,
                    "membership_history": [{"participant_id": member, "joined_at": now,
                                            "left_at": None, "version": 1} for member in members],
                    "last_sequence": 0}
            self.store.put("conversation", conversation_id, data)
            for member in members:
                if self._participant(member).get("kind") == "companion" and trusted.get("exchange_id"):
                    self._companion_invitation(conversation_id, member, trusted, author)
            message = self._append(author, conversation_id, opening, mentions, trusted, context_refs) if opening is not None else None
            # Initial participants see the opening as unread; there is no backlog.
            for member in members:
                if self._participant(member).get("kind") == "human":
                    self._initialize_read(member, conversation_id, 0)
            return {"conversation": self.store.get("conversation", conversation_id), "message": message}

        return self.store.command(author, command_id, "conversation.create", payload, apply)

    def send(self, author, conversation_id, content, command_id, mentions=None, lineage=None, context_refs=None,
             reply_offer_id=None, admitted_id=None, admitted_timestamp=None, metadata=None):
        payload = {"conversation_id": conversation_id, "content": content,
                   "mentions": mentions, "lineage": lineage, "context_refs": context_refs,
                   "reply_offer_id": reply_offer_id, "admitted_id": admitted_id,
                   "admitted_timestamp": admitted_timestamp, "metadata": metadata}

        def apply():
            self._conversation(conversation_id, author)
            return self._append(author, conversation_id, content, mentions, self._lineage(author, lineage),
                                context_refs, reply_offer_id, admitted_id, admitted_timestamp, metadata)

        return self.store.command(author, command_id, "conversation.send", payload, apply)

    def _append(self, author, conversation_id, content, mentions, lineage, context_refs=None, reply_offer_id=None,
                admitted_id=None, admitted_timestamp=None, metadata=None):
        content = self._content(content)
        conversation = self._conversation(conversation_id, author)
        data = conversation["data"]
        addressees = self._addressees(data["participants"], content, mentions)
        recipients = addressees if addressees is not None else data["participants"]
        recipients = [member for member in recipients if member != author]
        sequence = data["last_sequence"] + 1
        if admitted_id is not None and (not isinstance(admitted_id, str) or not 1 <= len(admitted_id) <= 256):
            raise ProtocolError("Invalid admitted message identity")
        message_id = admitted_id or _id("message")
        if self.store.get("message", message_id):
            raise Conflict("Message identity already exists")
        answered_offer = self.store.get("reply_offer", reply_offer_id) if reply_offer_id else None
        if reply_offer_id and (not answered_offer or answered_offer["data"]["recipient"] != author
                               or answered_offer["data"]["conversation_id"] != conversation_id
                               or answered_offer["data"]["exchange_id"] != lineage.get("exchange_id")
                               or answered_offer["data"]["status"] not in {"pending", "deferred"}):
            raise AccessDenied("Response must answer this participant's open offer in the same exchange")
        message = {"conversation_id": conversation_id, "author": author,
                   "content": content, "timestamp": self.store.clock() if admitted_timestamp is None else admitted_timestamp, "sequence": sequence,
                   "membership_version": data["membership_version"],
                   "participants": list(data["participants"]), "addressees": list(recipients),
                   "explicit_mentions": addressees is not None, "lineage": dict(lineage),
                   "context_refs": self._context_refs(context_refs), "reply_offer_id": reply_offer_id,
                   "reply_to": list(answered_offer["data"]["message_ids"]) if answered_offer else [],
                   "metadata": dict(metadata or {})}
        self.store.put("message", message_id, message)
        self.store.put("conversation", conversation_id, {**data, "last_sequence": sequence},
                       expected=conversation["revision"])
        exchange_id = lineage.get("exchange_id")
        for recipient in recipients:
            kind = self._participant(recipient).get("kind")
            if kind == "human":
                intent_id = _stable("notification", message_id, recipient)
                self.store.put("notification_intent", intent_id,
                               {"conversation_id": conversation_id, "message_id": message_id,
                                "recipient": recipient, "sequence": sequence, "status": "pending",
                                "created_at": self.store.clock()})
            elif exchange_id:
                if kind == "companion" and not self.store.get(
                        "companion_exchange", _stable("companion", conversation_id, exchange_id, recipient)):
                    continue
                offer_id = _stable("offer", conversation_id, exchange_id, recipient)
                prior = self.store.get("reply_offer", offer_id)
                # A deliberate decline/closure remains final. An answered offer
                # can receive another turn in this still-validated exchange.
                if prior and prior["data"]["status"] in {"silent", "declined", "exhausted", "closed"}:
                    continue
                offer = prior["data"] if prior else {
                    "conversation_id": conversation_id, "exchange_id": exchange_id,
                    "recipient": recipient, "status": "pending", "message_ids": [],
                    "lineage": dict(lineage), "created_at": self.store.clock(), "generation": 1, "history": []}
                if prior and offer["status"] == "answered":
                    outcome = {key: offer[key] for key in
                               ("generation", "status", "message_ids", "decided_at", "response_message_id") if key in offer}
                    offer = {**offer, "status": "pending", "message_ids": [],
                             "generation": offer.get("generation", 1) + 1,
                             "history": [*offer.get("history", []), outcome]}
                    offer.pop("decided_at", None)
                    offer.pop("response_message_id", None)
                self.store.put("reply_offer", offer_id,
                               {**offer, "message_ids": [*offer["message_ids"], message_id]},
                               expected=prior["revision"] if prior else None)
        if answered_offer:
            self.store.put("reply_offer", reply_offer_id,
                           {**answered_offer["data"], "status": "answered", "response_message_id": message_id,
                            "decided_at": self.store.clock()}, expected=answered_offer["revision"])
        self.store.event("conversation_message", author, {"conversation_id": conversation_id,
                                                         "message_id": message_id, "sequence": sequence})
        return {"id": message_id, **message}

    def _companion_invitation(self, conversation_id, companion_id, lineage, author):
        record_id = _stable("companion", conversation_id, lineage["exchange_id"], companion_id)
        return self.store.put("companion_exchange", record_id,
                              {"conversation_id": conversation_id, "companion_id": companion_id,
                               "exchange_id": lineage["exchange_id"], "invited_by": author,
                               "lineage": dict(lineage)})

    def invite_companion(self, author, conversation_id, companion_id, command_id, lineage):
        def apply():
            self._conversation(conversation_id, author)
            self._conversation(conversation_id, companion_id)
            if self._participant(companion_id).get("kind") != "companion":
                raise ProtocolError("Only companions need bounded exchange invitations")
            trusted = self._lineage(author, lineage)
            if not trusted.get("exchange_id"):
                raise ProtocolError("A companion invitation needs an admitted exchange")
            return self._companion_invitation(conversation_id, companion_id, trusted, author)

        return self.store.command(author, command_id, "conversation.invite_companion",
                                  {"conversation_id": conversation_id, "companion_id": companion_id,
                                   "lineage": lineage}, apply)

    def invite(self, author, conversation_id, participant_id, command_id, trigger_message_id=None):
        payload = {"conversation_id": conversation_id, "participant_id": participant_id,
                   "trigger_message_id": trigger_message_id}

        def apply():
            conversation = self._conversation(conversation_id, author)
            self._participant(participant_id)
            if not conversation["data"]["group"]:
                raise Conflict("A private conversation cannot gain another participant")
            if participant_id in conversation["data"]["participants"]:
                raise Conflict("Participant has already joined")
            if trigger_message_id:
                trigger = self.store.get("message", trigger_message_id)
                if not trigger or trigger["data"]["conversation_id"] != conversation_id:
                    raise NotFound("Invitation trigger is not a message in this conversation")
            invitation_id = _stable("invitation", conversation_id, participant_id)
            return self.store.put("conversation_invitation", invitation_id,
                                  {**payload, "invited_by": author, "status": "pending",
                                   "history_notice": "Joining grants access to the full prior thread."})

        return self.store.command(author, command_id, "conversation.invite", payload, apply)

    def join(self, author, conversation_id, participant_id, command_id, trigger_message_id=None):
        payload = {"conversation_id": conversation_id, "participant_id": participant_id,
                   "trigger_message_id": trigger_message_id}

        def apply():
            if author != participant_id:
                raise AccessDenied("The invited participant must choose to join")
            self._participant(author)
            conversation = self._conversation(conversation_id)
            data = conversation["data"]
            invitation_id = _stable("invitation", conversation_id, participant_id)
            invitation = self.store.get("conversation_invitation", invitation_id)
            if not invitation or invitation["data"]["status"] != "pending":
                raise AccessDenied("Joining needs a pending explicit invitation")
            if participant_id in data["participants"]:
                raise Conflict("Participant has already joined")
            trigger_id = invitation["data"].get("trigger_message_id")
            if trigger_message_id and trigger_message_id != trigger_id:
                raise Conflict("Join cannot change the admitted invitation trigger")
            trigger = self.store.get("message", trigger_id) if trigger_id else None
            baseline = trigger["data"]["sequence"] - 1 if trigger else data["last_sequence"]
            version = data["membership_version"] + 1
            joined = {**data, "participants": [*data["participants"], participant_id],
                      "membership_version": version,
                      "membership_history": [*data["membership_history"],
                                             {"participant_id": participant_id,
                                              "joined_at": self.store.clock(), "left_at": None,
                                              "version": version}]}
            self.store.put("conversation", conversation_id, joined, expected=conversation["revision"])
            self.store.put("conversation_invitation", invitation_id,
                           {**invitation["data"], "status": "joined"}, expected=invitation["revision"])
            if self._participant(author).get("kind") == "human":
                self._initialize_read(author, conversation_id, baseline)
            return self.store.get("conversation", conversation_id)

        return self.store.command(author, command_id, "conversation.join", payload, apply)

    def leave(self, author, conversation_id, command_id):
        def apply():
            conversation = self._conversation(conversation_id, author)
            data = conversation["data"]
            version = data["membership_version"] + 1
            history = [dict(entry) for entry in data["membership_history"]]
            for entry in history:
                if entry["participant_id"] == author and entry["left_at"] is None:
                    entry["left_at"] = self.store.clock()
            return self.store.put("conversation", conversation_id,
                                  {**data, "participants": [p for p in data["participants"] if p != author],
                                   "membership_version": version, "membership_history": history},
                                  expected=conversation["revision"])

        return self.store.command(author, command_id, "conversation.leave",
                                  {"conversation_id": conversation_id}, apply)

    def history(self, viewer, conversation_id, through=None):
        self._conversation(conversation_id, viewer)
        return sorted([{"id": record["id"], **record["data"]}
                       for record in self.store.list("message")
                       if record["data"]["conversation_id"] == conversation_id
                       and (through is None or record["data"]["sequence"] <= through)],
                      key=lambda message: message["sequence"])

    def snapshot(self, viewer, conversation_id):
        with self.store.transaction():
            conversation = self._conversation(conversation_id, viewer)
            watermark = conversation["data"]["last_sequence"]
            snapshot_id = _id("snapshot")
            self.store.put("conversation_snapshot", snapshot_id,
                           {"viewer": viewer, "conversation_id": conversation_id, "watermark": watermark,
                            "membership_version": conversation["data"]["membership_version"]})
            return {"snapshot_id": snapshot_id, "conversation": conversation,
                    "messages": self.history(viewer, conversation_id, through=watermark), "watermark": watermark}

    @staticmethod
    def _read_id(human, conversation_id):
        return _stable("read", human, conversation_id)

    def _initialize_read(self, human, conversation_id, baseline):
        read_id = self._read_id(human, conversation_id)
        prior = self.store.get("thread_read", read_id)
        self.store.put("thread_read", read_id,
                       {"human": human, "conversation_id": conversation_id,
                        "watermark": max(baseline, prior["data"]["watermark"] if prior else 0)},
                       expected=prior["revision"] if prior else None)

    def mark_read(self, human, conversation_id, watermark, command_id, snapshot_id=None, explicit=False):
        payload = {"conversation_id": conversation_id, "watermark": watermark,
                   "snapshot_id": snapshot_id, "explicit": explicit}

        def apply():
            self._human(human)
            conversation = self._conversation(conversation_id, human)
            if type(watermark) is not int or watermark < 0 or watermark > conversation["data"]["last_sequence"]:
                raise ProtocolError("Invalid transcript watermark")
            if not explicit:
                snapshot = self.store.get("conversation_snapshot", snapshot_id) if snapshot_id else None
                if not snapshot or snapshot["data"]["viewer"] != human or snapshot["data"]["conversation_id"] != conversation_id:
                    raise AccessDenied("Read acknowledgment requires this participant's transcript snapshot")
                if watermark > snapshot["data"]["watermark"]:
                    raise Conflict("Newer messages were not in the displayed transcript")
            self._initialize_read(human, conversation_id, watermark)
            return self.store.get("thread_read", self._read_id(human, conversation_id))

        return self.store.command(human, command_id, "conversation.read", payload, apply)

    def unread(self, human):
        self._human(human)
        threads = []
        for conversation in self.store.list("conversation"):
            if human not in conversation["data"]["participants"]:
                continue
            state = self.store.get("thread_read", self._read_id(human, conversation["id"]))
            watermark = state["data"]["watermark"] if state else 0
            incoming = [message for message in self.history(human, conversation["id"])
                        if message["sequence"] > watermark and message["author"] != human]
            if incoming:
                threads.append({"conversation_id": conversation["id"], "title": conversation["data"]["title"],
                                "count": len(incoming), "latest": incoming[-1], "watermark": watermark})
        return {"count": len(threads), "threads": threads}

    def offers(self, participant):
        self._participant(participant)
        return [record for record in self.store.list("reply_offer")
                if record["data"]["recipient"] == participant
                and record["data"]["status"] == "pending"
                and participant in self._conversation(record["data"]["conversation_id"])["data"]["participants"]]

    def decide_offer(self, participant, offer_id, status, command_id):
        def apply():
            if status not in {"silent", "declined", "deferred", "answered", "exhausted", "closed"}:
                raise ProtocolError("Unsupported reply-offer outcome")
            record = self.store.get("reply_offer", offer_id)
            if not record:
                raise NotFound("Unknown reply offer")
            if record["data"]["recipient"] != participant:
                raise AccessDenied("A participant controls only its own reply offers")
            self._conversation(record["data"]["conversation_id"], participant)
            if record["data"]["status"] != "pending":
                raise Conflict("Reply offer has already ended")
            return self.store.put("reply_offer", offer_id,
                                  {**record["data"], "status": status, "decided_at": self.store.clock()},
                                  expected=record["revision"])

        return self.store.command(participant, command_id, "conversation.offer",
                                  {"offer_id": offer_id, "status": status}, apply)
