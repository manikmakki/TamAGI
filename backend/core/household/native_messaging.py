"""Resident-authored contact through trusted native execution authority."""
from __future__ import annotations

from backend.core.runtime import current_execution
from backend.skills.runtime_skill import RuntimeSkill, param
from .contracts import AccessDenied, ProtocolError
from .store import uid
from .worker_bridge import admission_context


def install_messaging(runtime, skills):
    def contact(message, participants=None, conversation_id=None, title="", mentions=None, context_refs=None):
        context = current_execution.get()
        if not context or context.actor_id != runtime.actor_id or context.depth:
            raise AccessDenied("Only a full resident may initiate household contact")
        runtime.household.assert_active(context)
        receipt = next((r for r in runtime.store.receipts(context.execution_id, limit=100)
                        if r["status"] == "started" and r["tool"] == "contact"), None)
        command = "contact:" + (receipt["id"] if receipt else uid("operation"))
        grant = admission_context().get() or {}
        body = {"grant_id": grant.get("id") or grant.get("grant_id"), "local_execution_id": context.execution_id,
                "command_id": command, "message": message, "participants": participants,
                "conversation_id": conversation_id, "title": title, "mentions": mentions, "context_refs": context_refs}
        runtime.store.put("household_outbox", command, runtime.actor_id,
                          {"status": "pending", "payload": body}, category="platform")
        try:
            result = runtime.household._request("POST", "/_native/contact", json=body)
            status = "committed"
            return result
        except Exception:
            status = "outcome_unknown"
            raise
        finally:
            runtime.store.put("household_outbox", command, runtime.actor_id,
                              {"status": status, "payload": body}, category="platform")

    skills.register(RuntimeSkill(runtime, "contact",
        "Choose to contact your human or another household resident. Create a thread with an opening message or post to an existing thread. This shares only the message and explicit source references you choose. It uses your current shared allowance and creates no human input. Inspect household_contacts for stable participant IDs first.",
        {"message": param("string", required=True), "participants": param("array", "Stable participant IDs for a new thread"),
         "conversation_id": param("string"), "title": param("string"), "mentions": param("array"),
         "context_refs": param("array", "Explicit attributed references: kind, id, optional summary")}, contact))
    def contacts():
        return runtime.household._request("GET", "/_native/contacts")
    skills.register(RuntimeSkill(runtime, "household_contacts", "Inspect household participant IDs and your pending full-history invitations.", {}, contacts))
    def invitation(invitation_id, accept=True):
        context = current_execution.get()
        if not context or context.actor_id != runtime.actor_id or context.depth:
            raise AccessDenied("Only the invited full resident chooses membership")
        return runtime.household._request("POST", "/_native/invitations/decide", json={
            "local_execution_id": context.execution_id, "invitation_id": invitation_id, "accept": accept})
    skills.register(RuntimeSkill(runtime, "answer_invitation", "Choose whether to join an invited thread. Joining shares its full prior history; unrelated private threads remain private.",
                                {"invitation_id": param("string", required=True), "accept": param("boolean")}, invitation))


class NativeMessaging:
    def __init__(self, registry, model, manager):
        self.registry, self.model, self.manager, self.store = registry, model, manager, registry.store

    def contacts(self, actor):
        return {"participants": self.registry.participants(), "invitations": [r for r in self.store.list("conversation_invitation")
                if r["data"]["participant_id"] == actor and r["data"]["status"] == "pending"]}

    def decide(self, actor, body):
        _, execution, _ = self.model._active(actor, body["local_execution_id"])
        if execution["actor_id"] != actor or execution["helper"]:
            raise AccessDenied("Only the invited resident may join")
        invitation = self.store.get("conversation_invitation", body["invitation_id"])
        if not invitation or invitation["data"]["participant_id"] != actor:
            raise AccessDenied("Invitation belongs to another participant")
        if body.get("accept") is True:
            return self.manager.conversations.join(actor, invitation["data"]["conversation_id"], actor,
                "join:" + invitation["id"])
        self.store.put("conversation_invitation", invitation["id"], {**invitation["data"], "status": "declined"})
        return {"declined": True}

    def contact(self, actor, body):
        def apply():
            _, execution, _ = self.model._active(actor, body["local_execution_id"])
            if execution["actor_id"] != actor or execution["helper"]:
                raise AccessDenied("Only a full resident initiates household contact")
            binding = self.model._binding(actor, body["local_execution_id"])
            lineage = self.model.broker.validate_lineage(binding["broker_execution_id"], actor)
            conversation_id = body.get("conversation_id")
            if conversation_id:
                message = self.manager.conversations.send(actor, conversation_id, body["message"], body["command_id"] + ":send",
                    mentions=body.get("mentions"), lineage=lineage, context_refs=body.get("context_refs"))
            else:
                members = body.get("participants")
                if not isinstance(members, list) or not members:
                    raise ProtocolError("Choose recipients for the opening message")
                result = self.manager.conversations.create(actor, members, body["command_id"] + ":create", title=body.get("title", ""),
                    group=len(set([actor, *members])) > 2, opening=body["message"], mentions=body.get("mentions"),
                    lineage=lineage, context_refs=body.get("context_refs"))
                conversation_id, message = result["conversation"]["id"], result["message"]
            self.manager.enqueue_contact(actor, conversation_id, message, lineage)
            return {"conversation_id": conversation_id, "message": message}
        return self.store.command(actor, body["command_id"], "native.contact", body, apply)
