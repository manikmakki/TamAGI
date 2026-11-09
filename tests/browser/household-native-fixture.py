import asyncio
import json
import os
from pathlib import Path
import secrets
import tempfile

import uvicorn
import yaml

from backend.auth import hash_password
from backend.config import TamAGIConfig
from backend.core.household.native_app import create_household_app
from backend.core.household.native_registry import NativeRegistry
from backend.core.llm import LLMResponse, ToolCall


class Provider:
    def __init__(self, registry):
        self.calls = 0
        self.cedar = next(actor for actor, d in registry.descriptors.items() if d['name'] == 'Cedar')

    async def count_context_tokens(self, messages, tools=None):
        return None

    async def chat(self, messages, tools=None, **options):
        self.calls += 1
        await asyncio.sleep(1 if self.calls <= 2 else 3)
        if 'Household invitation decision.' in str(messages[0].content):
            card = json.loads(messages[-1].content.split('\n', 1)[1])
            return LLMResponse(tool_calls=[ToolCall(f'browser-accept-{self.calls}', 'contact', {
                'action': 'respond', 'invitation_id': card['id'], 'decision': 'accept'})])
        current = next((str(m.content) for m in reversed(messages) if m.role == 'user'), '')
        if 'Browser invite Cedar' in current:
            if not any(m.role == 'tool' for m in messages):
                return LLMResponse(tool_calls=[ToolCall(f'browser-invite-{self.calls}', 'contact', {
                    'action': 'invite', 'participants': [self.cedar], 'message': 'Please join this existing discussion, Cedar.'})])
            return LLMResponse(content='@Tester Cedar has an invitation into this thread and can choose whether to join.')
        if any("Browser image" in str(message.content) for message in messages):
            assert any(isinstance(message.content, list) and any(block.get("type") == "image_url" for block in message.content)
                       for message in messages), "The native worker did not receive its image"
        if not any(message.role == "tool" for message in messages):
            return LLMResponse(content="**Checking the workspace** with `read`. <img src=x onerror=\"window.reasoningUnsafe=true\">",
                tool_calls=[ToolCall(f"browser-read-{self.calls}", "read", {"path": "WORKSPACE.md"})])
        return LLMResponse(content=f"@Tester Actual worker browser answer {self.calls}.")

    async def chat_stream(self, messages, tools=None, **options):
        yield await self.chat(messages, tools, **options)

    async def close(self):
        pass


temporary = tempfile.TemporaryDirectory(prefix="hh-browser-", dir="/tmp")
root = Path(temporary.name) / "h"
registry = NativeRegistry.create(root, names=["Cedar", "Willow"], operator_name="Tester",
                                 release="deterministic-browser-fixture", offline=True)
config = TamAGIConfig.model_validate(yaml.safe_load((root / "config.yaml").read_text()))
password = secrets.token_urlsafe(24)
config.auth.password_hash = hash_password(password)
credential = Path(os.environ.get("TAMAGI_BROWSER_CREDENTIAL", "/tmp/tamagi-native-browser-credential"))
credential.write_text(password)
credential.chmod(0o600)
app = create_household_app(config, provider=Provider(registry), registry=registry)
print("Fresh native browser household: " + str(root), flush=True)
try:
    uvicorn.run(app, host="127.0.0.1", port=18846, access_log=False, log_level="warning")
finally:
    credential.unlink(missing_ok=True)
    temporary.cleanup()
