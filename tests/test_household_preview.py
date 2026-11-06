"""Preview API acceptance using disposable roots and real offline fixture workers."""

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.core.household.preview import create_preview_app
from backend.core.household.registry import SandboxRegistry


class HouseholdPreviewTests(unittest.TestCase):
    password = "disposable household preview test password"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tamagi-preview-test-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.registry = SandboxRegistry.create("preview-test-release", parent=self.temporary.name)
        self.app = create_preview_app(self.registry, self.password)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.sequence = 0
        self.actors = [record["id"] for record in self.registry.store.list("resident")]

    def post(self, path, body=None, *, key=None, expected=200, **kwargs):
        self.sequence += 1
        headers = {"X-Household-Preview": "1", "Idempotency-Key": key or f"test-command-{self.sequence}"}
        headers.update(kwargs.pop("headers", {}))
        response = self.client.post("/api/preview" + path, json={} if body is None else body,
                                    headers=headers, **kwargs)
        self.assertEqual(response.status_code, expected, response.text)
        return response.json()

    def login(self):
        return self.post("/login", {"password": self.password})

    def state(self):
        response = self.client.get("/api/preview/state")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def human_thread(self, actor=None, *, key=None, content="A manual fixture question", **kwargs):
        return self.post("/threads", {"participants": [actor or self.actors[0]],
            "content": content, "title": "Fixture thread", **kwargs}, key=key)

    def resident_thread(self, actor=None, participants=None, *, key=None, content="A chosen scripted observation"):
        actor = actor or self.actors[0]
        return self.post(f"/residents/{actor}/initiate", {
            "participants": ["operator"] if participants is None else participants,
            "content": content, "title": "Resident initiated fixture"}, key=key)

    def snapshot(self, thread):
        response = self.client.get("/api/preview/threads/" + thread)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def offer(self, actor, thread):
        return next(record for record in self.state()["offers"]
                    if record["data"]["recipient"] == actor
                    and record["data"]["conversation_id"] == thread)

    def reply(self, actor, thread, *, key=None):
        return self.post(f"/residents/{actor}/reply", {"offer_id": self.offer(actor, thread)["id"]}, key=key)

    def test_authentication_protects_state_and_uses_signed_preview_cookie(self):
        health = self.client.get("/api/preview/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"ready": True, "fixture_only": True, "real_model": False})
        self.assertNotIn(self.registry.household_id, health.text)
        self.assertEqual(self.client.get("/api/preview/state").status_code, 401)
        self.assertEqual(self.client.get("/api/preview/residents/" + self.actors[0]).status_code, 401)
        redirect = self.client.get("/", follow_redirects=False)
        self.assertEqual(redirect.status_code, 303)
        self.assertEqual(redirect.headers["location"], "/login")
        page = self.client.get("/login")
        self.assertEqual(page.status_code, 200)
        self.assertIn("scripted", page.text.lower())
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.post("/login", {"password": "incorrect"}, expected=401)
        response = self.client.post("/api/preview/login", json={"password": self.password},
                                    headers={"X-Household-Preview": "1"})
        self.assertEqual(response.status_code, 200)
        cookie = response.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=strict", cookie)
        name = next(iter(self.client.cookies))
        self.assertTrue(name.startswith("tamagi_household_preview_"))
        self.assertNotEqual(name, "session")
        self.assertNotIn(self.password, response.text)
        self.assertEqual(len(self.state()["residents"]), 3)
        self.client.cookies.clear()
        self.client.cookies.set(name, "tampered-session")
        self.assertEqual(self.client.get("/api/preview/state").status_code, 401)
        self.client.cookies.clear()
        self.login()
        self.post("/logout")
        self.assertEqual(self.client.get("/api/preview/state").status_code, 401)

    def test_cookie_identity_does_not_collide_with_another_preview_on_same_host(self):
        self.login()
        cookie_name = next(iter(self.client.cookies))
        cookie_value = self.client.cookies.get(cookie_name)
        other = SandboxRegistry.create("preview-test-release", parent=self.temporary.name)
        with TestClient(create_preview_app(other, self.password)) as client:
            client.cookies.set(cookie_name, cookie_value)
            self.assertEqual(client.get("/api/preview/state").status_code, 401)
            response = client.post("/api/preview/login", json={"password": self.password},
                                   headers={"X-Household-Preview": "1"})
            self.assertEqual(response.status_code, 200)
            other_cookie_name = response.headers["set-cookie"].split("=", 1)[0]
            self.assertNotEqual(cookie_name, other_cookie_name)
        self.assertEqual(self.client.get("/api/preview/state").status_code, 200)

    def test_mutations_require_json_preview_header_and_same_origin(self):
        self.login()
        body = {"participants": [self.actors[0]], "content": "Guarded fixture input"}
        response = self.client.post("/api/preview/threads", json=body)
        self.assertEqual(response.status_code, 403)
        response = self.client.post("/api/preview/threads", content="{}", headers={"X-Household-Preview": "1"})
        self.assertEqual(response.status_code, 403)
        for origin in ("http://unrelated.test", "http://testserver.evil.test", "https://testserver", "null", "http://["):
            with self.subTest(origin=origin):
                response = self.client.post("/api/preview/threads", json=body,
                    headers={"X-Household-Preview": "1", "Origin": origin})
                self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(len(self.registry.store.list("conversation")), 0)
        self.assertEqual(len(self.registry.store.list("broker_exchange")), 0)
        accepted = self.post("/threads", body, headers={"Origin": "http://testserver"})
        self.assertEqual(accepted["message"]["author"], "operator")
        response = self.client.get("/api/preview/state", headers={"Host": "unexpected.test"})
        self.assertEqual(response.status_code, 400)

    def test_new_thread_is_actual_human_admission_and_replays_without_fresh_resources(self):
        self.login()
        body = {"participants": [self.actors[0]], "content": "An attributed human input", "title": "A new fixture"}
        first = self.post("/threads", body, key="stable-thread")
        self.assertEqual(self.post("/threads", body, key="stable-thread"), first)
        self.post("/threads", {**body, "content": "Different input"}, key="stable-thread", expected=409)
        self.post("/threads", {**body, "author": self.actors[0], "origin": "autonomy"}, expected=422)
        self.assertEqual(first["message"]["author"], "operator")
        self.assertEqual(first["message"]["lineage"]["origin"], "user")
        self.assertEqual(len(self.registry.store.list("conversation")), 1)
        self.assertEqual(len(self.registry.store.list("message")), 1)
        self.assertEqual(len(self.registry.store.list("broker_exchange")), 1)
        self.assertEqual(len(self.registry.store.list("broker_charge")), 0)
        self.assertEqual(self.state()["unread_count"], 0)

    def test_manual_reply_uses_fake_provider_retains_attribution_and_deduplicates(self):
        self.login()
        actor = self.actors[0]
        opening = self.human_thread(actor)
        thread = opening["conversation"]["id"]
        offer = self.offer(actor, thread)
        body = {"offer_id": offer["id"]}
        first = self.post(f"/residents/{actor}/reply", body, key="one-scripted-reply")
        self.assertEqual(self.post(f"/residents/{actor}/reply", body, key="one-scripted-reply"), first)
        self.post(f"/residents/{actor}/reply", {"offer_id": "different-offer"}, key="one-scripted-reply", expected=404)
        self.assertEqual(first["message"]["author"], actor)
        self.assertIn("Scripted fixture reply", first["message"]["content"])
        self.assertIn("no language model was called", first["message"]["content"])
        self.assertEqual(first["message"]["lineage"]["exchange_id"], opening["message"]["lineage"]["exchange_id"])
        self.assertEqual(len(self.registry.store.list("broker_charge")), 1)
        self.assertEqual(len(self.registry.store.list("message")), 2)
        state = self.state()
        self.assertEqual(len(state["provider_timeline"]), 1)
        self.assertEqual(state["unread_count"], 1)
        self.assertTrue(all(a["data"]["status"] == "completed" for a in state["attempts"]))
        another = self.human_thread(actor)["conversation"]["id"]
        different_offer = self.offer(actor, another)
        self.post(f"/residents/{actor}/reply", {"offer_id": different_offer["id"]},
                  key="one-scripted-reply", expected=409)
        self.assertEqual(len(self.registry.store.list("broker_charge")), 1)

    def test_read_snapshot_race_and_older_device_acknowledgment_preserve_unread(self):
        self.login()
        actor = self.actors[0]
        thread = self.human_thread(actor)["conversation"]["id"]
        before = self.snapshot(thread)
        self.reply(actor, thread)
        self.assertEqual(self.state()["unread_count"], 1)
        self.post(f"/threads/{thread}/read", {"snapshot_id": before["snapshot_id"],
            "watermark": before["watermark"] + 1}, expected=409)
        self.post(f"/threads/{thread}/read", {"snapshot_id": before["snapshot_id"],
            "watermark": before["watermark"]})
        self.assertEqual(self.state()["unread_count"], 1)
        current = self.snapshot(thread)
        self.assertEqual(self.state()["unread_count"], 1, "Fetching alone must not mark messages read")
        body = {"snapshot_id": current["snapshot_id"], "watermark": current["watermark"]}
        self.post(f"/threads/{thread}/read", body, key="displayed-current")
        self.post(f"/threads/{thread}/read", body, key="displayed-current")
        self.assertEqual(self.state()["unread_count"], 0)
        self.post(f"/threads/{thread}/read", {"snapshot_id": before["snapshot_id"], "watermark": before["watermark"]})
        self.assertEqual(self.state()["unread_count"], 0)

    def test_resident_contact_keeps_autonomous_lineage_and_thread_counts_distinct(self):
        self.login()
        first = self.resident_thread(key="chosen-opening")
        second = self.resident_thread(self.actors[1])
        self.assertEqual(self.resident_thread(key="chosen-opening"), first)
        self.assertEqual(first["message"]["author"], self.actors[0])
        self.assertEqual(first["message"]["lineage"]["origin"], "autonomy")
        self.assertNotEqual(first["message"]["lineage"]["exchange_id"], second["message"]["lineage"]["exchange_id"])
        self.assertEqual(len(self.registry.store.list("broker_exchange")), 2)
        self.assertEqual(len(self.registry.store.list("broker_charge")), 0)
        self.assertEqual(self.state()["unread_count"], 2)
        thread = first["conversation"]["id"]
        self.post(f"/threads/{thread}/messages", {"content": "A real operator follow-up"})
        self.reply(self.actors[0], thread)
        state = self.state()
        self.assertEqual(state["unread_count"], 2)
        self.assertEqual(next(t for t in state["conversations"] if t["id"] == thread)["unread"], 2)

    def test_peer_private_thread_is_not_exposed_by_operator_thread_routes_or_state(self):
        self.login()
        private = self.resident_thread(self.actors[0], [self.actors[1]], content="Private peer fixture content")
        thread = private["conversation"]["id"]
        state = self.state()
        self.assertNotIn(thread, [t["id"] for t in state["conversations"]])
        self.assertNotIn("Private peer fixture content", str(state))
        self.assertEqual(state["unread_count"], 0)
        self.assertFalse(any(o["data"]["conversation_id"] == thread for o in state["offers"]))
        self.assertEqual(self.client.get("/api/preview/threads/" + thread).status_code, 403)
        self.post(f"/threads/{thread}/messages", {"content": "An unrelated operator message"}, expected=403)
        self.post(f"/threads/{thread}/read", {"watermark": 1, "snapshot_id": "unrelated"}, expected=403)
        offer = next(r for r in self.registry.store.list("reply_offer") if r["data"]["conversation_id"] == thread)
        self.post(f"/residents/{self.actors[1]}/reply", {"offer_id": offer["id"]}, expected=403)
        self.assertEqual(len(self.registry.store.list("broker_charge")), 0)

    def test_worker_inspection_is_explicit_and_world_visibility_remains_scoped(self):
        self.login()
        actor = self.actors[0]
        private = self.post("/world", {"owner": actor, "name": "Private fixture object", "visibility": "private"})
        shared = self.post("/world", {"owner": actor, "name": "Shared fixture object", "visibility": "shared"})
        state = self.state()
        self.assertNotIn(private["id"], [r["id"] for r in state["world"]])
        self.assertIn(shared["id"], [r["id"] for r in state["world"]])
        identities, roots = set(), set()
        for selected in self.actors:
            response = self.client.get("/api/preview/residents/" + selected)
            self.assertEqual(response.status_code, 200, response.text)
            inspected = response.json()
            self.assertEqual(inspected["resident_id"], selected)
            self.assertTrue(inspected["fixture_only"])
            self.assertFalse(inspected["generation_enabled"])
            self.assertTrue(Path(inspected["cwd"]).is_relative_to(self.registry.root))
            identities.add(inspected["identities"][0]["text"])
            roots.add(inspected["cwd"])
            self.assertEqual(private["id"] in [r["id"] for r in inspected["world"]], selected == actor)
        self.assertEqual(len(identities), 3)
        self.assertEqual(len(roots), 3)
        self.assertEqual(len(self.registry.store.list("broker_charge")), 0)
        self.assertEqual(self.client.get("/api/preview/residents/unknown").status_code, 404)

    def test_unavailable_fixture_keeps_offer_and_scheduling_only_runs_manual_eligible_work(self):
        self.login()
        actor = self.actors[0]
        thread = self.human_thread(actor)["conversation"]["id"]
        offer = self.offer(actor, thread)
        self.post(f"/residents/{actor}/availability", {"available": False})
        self.post(f"/residents/{actor}/availability", {"available": "false"}, expected=422)
        self.post(f"/residents/{actor}/reply", {"offer_id": offer["id"]}, key="deferred-reply", expected=409)
        self.post(f"/residents/{actor}/initiate", {"participants": ["operator"], "content": "Deferred opening"}, expected=409)
        self.assertEqual(len(self.registry.store.list("broker_charge")), 0)
        demo = self.post("/scheduling-demo", key="manual-demo")
        repeated = self.post("/scheduling-demo", key="manual-demo")
        self.assertEqual(set(demo["attempt_ids"]), set(repeated["attempt_ids"]))
        self.assertEqual(len(demo["attempt_ids"]), 2)
        self.assertEqual(len(self.registry.store.list("broker_charge")), 2)
        self.assertEqual(len(self.registry.store.list("message")), 1)
        self.assertNotIn(actor, [r["actor_id"] for r in self.state()["provider_timeline"]])
        for _ in range(3):
            self.state()
            self.snapshot(thread)
        self.assertEqual(len(self.registry.store.list("broker_charge")), 2, "Browsing must never schedule cognition")
        self.assertEqual(self.offer(actor, thread)["id"], offer["id"])
        self.post(f"/residents/{actor}/availability", {"available": True})
        self.post(f"/residents/{actor}/reply", {"offer_id": offer["id"]}, key="deferred-reply")
        self.assertEqual(len(self.registry.store.list("broker_charge")), 3)

    def test_invalid_demo_bodies_cannot_start_generations_and_all_unavailable_means_silence(self):
        self.login()
        self.post("/scheduling-demo", {"unexpected": "input"}, expected=422)
        response = self.client.post("/api/preview/scheduling-demo", content="not JSON",
            headers={"X-Household-Preview": "1", "Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(self.registry.store.list("broker_charge")), 0)
        for actor in self.actors:
            self.post(f"/residents/{actor}/availability", {"available": False})
        result = self.post("/scheduling-demo")
        self.assertEqual(result["attempt_ids"], [])
        self.assertEqual(len(self.registry.store.list("broker_charge")), 0)
        self.assertEqual(self.state()["provider_timeline"], [])


if __name__ == "__main__":
    unittest.main()
