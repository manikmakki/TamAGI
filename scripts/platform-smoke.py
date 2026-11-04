import tempfile, os, json
from pathlib import Path
from backend.config import TamAGIConfig, set_config
from backend.core.runtime import RuntimeStore
from fastapi.testclient import TestClient

original = Path.cwd()
with tempfile.TemporaryDirectory(prefix="tamagi-app-smoke-") as d:
    root = Path(d)
    store = RuntimeStore(root / "runtime.sqlite3")
    store.close()
    config = TamAGIConfig()
    config.runtime.enabled = True
    config.runtime.instance_dir = str(root)
    config.runtime.authority = "guarded"
    config.autonomy.enabled = False
    import yaml

    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump()))
    os.environ["TAMAGI_CONFIG"] = str(path)
    set_config(config)
    from backend.main import app

    try:
        with TestClient(app) as client:
            response = client.get("/api/health/ready")
            assert response.status_code == 200, response.text
            assert client.get("/api/runtime").status_code == 401
            assert client.get("/api/runtime/growth").status_code == 401
            from starlette.websockets import WebSocketDisconnect

            try:
                with client.websocket_connect("/api/ws"):
                    raise AssertionError("Unauthenticated WebSocket was accepted")
            except WebSocketDisconnect as exc:
                assert exc.code == 1008
            response = client.post(
                "/api/auth/setup", json={"password": "a long operator test password"}
            )
            assert response.status_code == 200, response.text
            response = client.get("/api/runtime")
            assert response.status_code == 200, response.text
            assert response.json()["home"]
            response = client.post(
                "/api/pursuits",
                json={
                    "action": "create",
                    "attributes": {
                        "title": "A leaf study",
                        "next_step": "Draw one outline",
                    },
                },
            )
            assert response.status_code == 200, response.text
            assert client.get("/home").status_code == 200
            with client.websocket_connect(
                "/api/ws", headers={"origin": "http://testserver"}
            ):
                pass
            assert (
                client.post(
                    "/api/runtime/notebook", json={"text": "A note from the visitor"}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/api/runtime/notebook",
                    json={"text": "cross origin"},
                    headers={"Origin": "https://unrelated.example"},
                ).status_code
                == 403
            )
            # The manual endpoint owns an autonomous execution, not a nested user allowance.
            from unittest.mock import patch, PropertyMock, AsyncMock
            from backend.api.chat import get_agent
            from backend.core.runtime import GovernedLLM, current_execution
            from backend.core.llm import LLMResponse

            agent = get_agent()
            runtime = agent.runtime

            class Provider:
                config = agent.llm.config
                calls = 0

                async def chat(self, messages, **kwargs):
                    self.calls += 1
                    assert current_execution.get().origin == "autonomy"
                    assert current_execution.get().parent_id is None
                    return LLMResponse(content="Looking around the workshop.")

            provider = Provider()
            world = agent._world_thread
            world._running = True
            runtime.store.set_meta("decompression_until", runtime.store.clock() + 900)
            try:
                with (
                    patch.object(
                        type(agent.identity),
                        "needs_onboarding",
                        new_callable=PropertyMock,
                        return_value=False,
                    ),
                    patch.object(agent, "llm", GovernedLLM(provider, runtime)),
                    patch.object(
                        agent,
                        "flush_unsummarized_conversations",
                        new_callable=AsyncMock,
                    ) as flush,
                ):
                    response = client.post("/api/world/tick")
                    assert response.status_code == 200, response.text
                    assert response.json()["activation_started"], response.text
                    assert provider.calls == 1
                    assert flush.await_count == 1
                    runtime.end_turn("resting", wake_at=runtime.store.clock() + 3600)
                    response = client.post("/api/world/tick")
                    assert response.json()["reason"] == "sleep", response.text
                    assert response.json()["activation_started"] is False
                    assert provider.calls == 1 and flush.await_count == 1
            finally:
                world._running = False

            # Growth follows completed conversations without extra scoring/naming calls.
            class ConversationProvider:
                config = agent.llm.config
                calls = 0

                async def chat(self, messages, **kwargs):
                    self.calls += 1
                    return LLMResponse(content="Here is a thought to explore together.")

            conversation_provider = ConversationProvider()
            with (
                patch.object(
                    type(agent.identity),
                    "needs_onboarding",
                    new_callable=PropertyMock,
                    return_value=False,
                ),
                patch.object(agent, "llm", GovernedLLM(conversation_provider, runtime)),
                patch.object(agent, "qa_pipeline", None),
            ):
                for thread in range(2):
                    conversation_id = None
                    for turn in range(8):
                        body = {"message": "Let us continue discussing this idea."}
                        if conversation_id:
                            body["conversation_id"] = conversation_id
                        response = client.post("/api/chat", json=body)
                        assert response.status_code == 200, response.text
                        conversation_id = response.json()["conversation_id"]
                        growth = client.get("/api/runtime/growth")
                        assert growth.status_code == 200, growth.text
                        if thread == 0 and turn == 0:
                            assert growth.json()["total"] == 0, growth.text
                        if thread == 0 and turn == 2:
                            assert growth.json()["total"] == 1, growth.text
                assert conversation_provider.calls == 16, conversation_provider.calls
            growth = client.get("/api/runtime/growth").json()
            assert growth["total"] == 6, growth
            assert growth["level"] == 2, growth
            assert growth["stage"] == "Stage 2", growth
            state = client.get("/api/state").json()["state"]
            assert state["experience"] == growth["total"], state
            assert state["level"] == growth["level"], state
            assert state["growth"]["total"] == growth["total"], state
            assert client.post(
                "/api/runtime/growth", json={"total": 9000}
            ).status_code in (404, 405)
            # The operator's direct peer conversation must not become Ember's engagement.
            response = client.post(
                "/api/companions",
                json={"name": "Juniper", "seed": "A patient conversational partner"},
            )
            assert response.status_code == 200, response.text
            peer = response.json()
            thread_id = None
            with patch.object(
                runtime.companions, "llm", GovernedLLM(conversation_provider, runtime)
            ):
                for _ in range(3):
                    response = client.post(
                        "/api/companions/" + peer["id"] + "/consult",
                        json={
                            "content": "A question from the operator",
                            "thread_id": thread_id,
                        },
                    )
                    assert response.status_code == 200, response.text
                    thread_id = response.json()["thread_id"]
            assert client.get("/api/runtime/growth").json()["total"] == 6
            response = client.post(
                "/api/backups", json={"password": "a long backup test password"}
            )
            assert response.status_code == 200, response.text
            download = client.get(response.json()["download"])
            assert download.status_code == 200
            response = client.post(
                "/api/restores/preview",
                data={"password": "a long backup test password"},
                files={
                    "archive": (
                        "home.gpg",
                        download.content,
                        "application/octet-stream",
                    )
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["resident_id"]
            assert client.get("/api/operator-archives").status_code == 200
            print(
                json.dumps(
                    {
                        "startup": "passed",
                        "authentication": "passed",
                        "home": "passed",
                        "pursuits": "passed",
                        "csrf": "passed",
                        "growth_conversation_and_inspection": "passed",
                        "encrypted_download_upload_preview": "passed",
                    }
                )
            )
    finally:
        os.chdir(original)
