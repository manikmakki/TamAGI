"""One resident messaging tool; membership decisions retain trusted authority."""
from __future__ import annotations

from backend.core.runtime import current_execution, current_terminal
from backend.skills.runtime_skill import RuntimeSkill, param
from .contracts import AccessDenied, Conflict, ProtocolError
from .store import uid
from .worker_bridge import admission_context


def install_messaging(runtime, skills):
    def contact(action, message="", participants=None, conversation_id=None, title="", mentions=None,
                context_refs=None, invitation_id=None, decision=None):
        context = current_execution.get()
        if not context or context.actor_id != runtime.actor_id or context.depth:
            raise AccessDenied("Only a full resident may use household contact")
        runtime.household.assert_active(context)
        grant = admission_context().get() or {}
        if grant.get("kind") == "invitation" and action != "respond":
            raise AccessDenied("Choose accept or decline before using the invited conversation")
        receipt = next((r for r in runtime.store.receipts(context.execution_id, limit=100)
                        if r["status"] == "started" and r["tool"] == "contact"), None)
        command = "contact:" + (receipt["id"] if receipt else uid("operation"))
        body = {"action": action, "grant_id": grant.get("id"), "local_execution_id": context.execution_id,
                "command_id": command, "message": message, "participants": participants,
                "conversation_id": conversation_id or (grant.get("conversation_id") if action in {"send", "invite"} else None),
                "title": title, "mentions": mentions, "context_refs": context_refs,
                "invitation_id": invitation_id or (grant.get("invitation_id") if action == "respond" else None),
                "decision": decision}
        runtime.store.put("household_outbox", command, runtime.actor_id,
                          {"status": "pending", "payload": body}, category="platform")
        status = "outcome_unknown"
        try:
            result = runtime.household._request("POST", "/_native/contact", json=body)
            status = "committed"
            if grant.get("kind") == "invitation" and action == "respond":
                current_terminal.set({"status": "completed"})
            return result
        finally:
            runtime.store.put("household_outbox", command, runtime.actor_id,
                              {"status": status, "payload": body}, category="platform")

    skills.register(RuntimeSkill(runtime, "contact",
        "Household messaging: choose action=create for a NEW thread, send for an existing thread, invite to bring one or more residents INTO THE CURRENT THREAD, or respond ONLY to accept/decline an invitation you received. send/invite default to your current thread. To invite, supply participants and message; mentions are unnecessary. To respond, supply decision and optional invitation_id only, with no message. Never substitute create for inviting into a conversation. Joining reveals the full prior thread only after the invitee accepts. Uses the current shared allowance; creates no human input. household_contacts lists stable IDs and invitations.",
        {"action": param("string", required=True, enum=["create", "send", "invite", "respond"]),
         "message": param("string", "Opening/post/invitation message; required for create, send and invite"),
         "participants": param("array", "One or more stable recipient IDs for create or invite"),
         "conversation_id": param("string", "Existing thread ID; defaults to current for send/invite"),
         "title": param("string", "New thread title for create only"),
         "mentions": param("array", "Joined addressee IDs for create/send; for invite, may only repeat IDs in participants"),
         "context_refs": param("array", "Explicit attributed references for create/send"),
         "invitation_id": param("string", "Invitation to answer; defaults to the invitation currently delivered"),
         "decision": param("string", "Required for respond to a received invitation; omit for other actions", enum=["accept", "decline"])}, contact))

    def contacts():
        result = runtime.household._request("GET", "/_native/contacts")
        grant = admission_context().get() or {}
        result["current_conversation_id"] = None if grant.get("kind") == "invitation" else grant.get("conversation_id")
        return result
    skills.register(RuntimeSkill(runtime, "household_contacts",
        "Inspect household participant IDs, your current thread ID and pending full-history invitation cards. No transcript is disclosed before acceptance.", {}, contacts))


class NativeMessaging:
    def __init__(self, registry, model, manager):
        self.registry, self.model, self.manager, self.store = registry, model, manager, registry.store

    def card(self, invitation):
        data = invitation["data"]
        names = {p["id"]: p["name"] for p in self.registry.participants()}
        return {"id": invitation["id"], "invited_by_name": names.get(data["invited_by"], data["invited_by"]),
                "participant_names": [names.get(p, p) for p in data.get("participants", [])], **{k: data.get(k) for k in
            ("conversation_id", "participant_id", "invited_by", "status", "history_notice", "message", "title", "participants")}}

    def contacts(self, actor):
        return {"participants": self.registry.participants(), "invitations": [self.card(r)
                for r in self.store.list("conversation_invitation")
                if r["data"]["participant_id"] == actor and r["data"]["status"] == "pending"]}

    def invite_members(self, actor, conversation_id, members, message, command, lineage, trigger=None):
        conversation = self.manager.conversations._conversation(conversation_id, actor)
        if not isinstance(members, list) or not members or any(not isinstance(m, str) or m not in self.registry.descriptors for m in members):
            raise ProtocolError("Choose one or more full resident IDs to invite")
        self.manager.conversations._content(message)
        if len(message) > 4096:
            raise ProtocolError("Keep an invitation message within 4096 characters")
        targets = list(dict.fromkeys(m for m in members if m not in conversation["data"]["participants"]))
        if not targets:
            return {"conversation_id": conversation_id, "invitations": [], "notice": "Those residents have already joined."}
        pending_targets = {r["data"]["participant_id"]: r for r in self.store.list("conversation_invitation")
                           if r["data"]["conversation_id"] == conversation_id and r["data"]["status"] == "pending"}
        if all(target in pending_targets for target in targets):
            return {"conversation_id": conversation_id, "invitations": [self.card(pending_targets[t]) for t in targets],
                    "notice": "Those invitations are already pending; no duplicate delivery was created."}
        if not conversation["data"]["group"]:
            conversation = self.store.put("conversation", conversation_id,
                {**conversation["data"], "group": True}, expected=conversation["revision"])
        if trigger is None:
            trigger = self.manager.conversations.send(actor, conversation_id, message, command + ":notice",
                                                      mentions=[actor], lineage=lineage)
        invitations = []
        for target in targets:
            pending = next((r for r in self.store.list("conversation_invitation")
                if r["data"]["conversation_id"] == conversation_id and r["data"]["participant_id"] == target
                and r["data"]["status"] == "pending"), None)
            if pending:
                invitations.append(self.card(pending))
                continue
            invitation = self.manager.conversations.invite(actor, conversation_id, target,
                command + ":invite:" + target, trigger_message_id=trigger["id"])
            invitation = self.store.put("conversation_invitation", invitation["id"], {**invitation["data"],
                "message": message, "title": conversation["data"]["title"],
                "participants": list(conversation["data"]["participants"]), "lineage": lineage})
            self.store.put("native_invitation_offer", invitation["id"], {
                "invitation_id": invitation["id"], "version": invitation["revision"], "recipient": target,
                "conversation_id": conversation_id, "exchange_id": lineage["exchange_id"], "status": "pending",
                "trigger_message_id": trigger["id"], "lineage": lineage})
            invitations.append(self.card(invitation))
            self.store.event("invitation_created", actor, {"invitation_id": invitation["id"], "conversation_id": conversation_id})
        self.manager.enqueue_contact(actor, conversation_id, trigger, lineage)
        return {"conversation_id": conversation_id, "invitations": invitations,
                "notice": "Invitations sent into this existing thread. Residents choose whether to join; accepting shares its full prior history."}

    def respond(self, actor, body, lineage, grant):
        invitation = self.store.get("conversation_invitation", body.get("invitation_id"))
        if not invitation or invitation["data"]["participant_id"] != actor:
            raise AccessDenied("Invitation belongs to another participant")
        if grant.get("kind") == "invitation" and (grant.get("invitation_id") != invitation["id"]
                or grant.get("invitation_version") != invitation["revision"]):
            raise AccessDenied("This decision activation is limited to its admitted invitation")
        decision = body.get("decision")
        if decision not in {"accept", "decline"}:
            raise ProtocolError("Choose decision=accept or decline")
        if invitation["data"]["status"] != "pending":
            raise Conflict("Invitation has already been answered")
        conversation_id = invitation["data"]["conversation_id"]
        if decision == "accept":
            self.manager.conversations.join(actor, conversation_id, actor,
                f"join:{invitation['id']}:{invitation['revision']}")
            self.manager.offer_invitation_reply(actor, invitation, lineage)
        else:
            self.store.put("conversation_invitation", invitation["id"], {**invitation["data"], "status": "declined"})
        offer = self.store.get("native_invitation_offer", invitation["id"])
        if offer:
            self.store.put("native_invitation_offer", offer["id"], {**offer["data"], "status": "answered", "decision": decision})
        self.store.event("invitation_answered", actor, {"invitation_id": invitation["id"], "conversation_id": conversation_id, "decision": decision})
        return {"invitation_id": invitation["id"], "conversation_id": conversation_id,
                "decision": decision, "joined": decision == "accept", "history_notice": invitation["data"]["history_notice"]}

    def contact(self, actor, body):
        def apply():
            _, execution, _ = self.model._active(actor, body["local_execution_id"])
            if execution["actor_id"] != actor or execution["helper"]:
                raise AccessDenied("Only a full resident uses household contact")
            binding = self.model._binding(actor, body["local_execution_id"])
            lineage = self.model.broker.validate_lineage(binding["broker_execution_id"], actor)
            grant_id = binding["request"].get("grant_id")
            grant = self.manager.grant(grant_id, actor) if grant_id else {}
            action = body.get("action")
            if grant.get("kind") == "invitation" and action != "respond":
                raise AccessDenied("Invitation decisions cannot create, send, or invite")
            if action == "respond":
                if any(body.get(key) for key in ("message", "participants", "conversation_id", "title", "mentions", "context_refs")):
                    raise ProtocolError("respond takes an invitation_id and decision, not message recipients")
                return self.respond(actor, body, lineage, grant)
            if body.get("decision") or body.get("invitation_id"):
                raise ProtocolError("Use action=respond for an invitation decision")
            conversation_id = body.get("conversation_id")
            if action == "invite":
                if not conversation_id:
                    raise ProtocolError("An invitation needs an existing thread")
                if any(body.get(key) for key in ("title", "context_refs")):
                    raise ProtocolError("invite shares its message in the existing thread; use create/send for other message options")
                mentions = body.get("mentions")
                members = body.get("participants")
                if mentions and (not isinstance(mentions, list) or not isinstance(members, list)
                        or any(not isinstance(m, str) or m not in members for m in mentions)):
                    raise ProtocolError("For invite, mentions may only repeat IDs in participants; use participants to choose invitees")
                # Redundant recipient mentions do not address the canonical notice
                # to unjoined residents. They still receive only invitation cards.
                return self.invite_members(actor, conversation_id, body.get("participants"),
                    body.get("message"), body["command_id"], lineage)
            if action == "send":
                if not conversation_id or body.get("participants"):
                    raise ProtocolError("send needs an existing thread; use invite to add recipients")
                message = self.manager.conversations.send(actor, conversation_id, body["message"], body["command_id"] + ":send",
                    mentions=body.get("mentions"), lineage=lineage, context_refs=body.get("context_refs"))
            elif action == "create":
                members = body.get("participants")
                if conversation_id or not isinstance(members, list) or not members:
                    raise ProtocolError("create needs recipients and no existing thread ID")
                result = self.manager.conversations.create(actor, members, body["command_id"] + ":create", title=body.get("title", ""),
                    group=len(set([actor, *members])) > 2, opening=body["message"], mentions=body.get("mentions"),
                    lineage=lineage, context_refs=body.get("context_refs"))
                conversation_id, message = result["conversation"]["id"], result["message"]
            else:
                raise ProtocolError("Choose contact action=create, send, invite, or respond")
            self.manager.enqueue_contact(actor, conversation_id, message, lineage)
            return {"conversation_id": conversation_id, "message": message}
        return self.store.command(actor, body["command_id"], "native.contact", body, apply)
