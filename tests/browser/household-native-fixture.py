import asyncio
import json
import os
from pathlib import Path
from contextlib import closing
import secrets
import sqlite3
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
        if any(m.role == 'user' and str(m.content).startswith('Summarize this conversation') for m in messages):
            return LLMResponse(content='I reflected on the attributed conversation before choosing my next activity.')
        if len(messages) > 1 and 'Autonomous opportunity;' in str(messages[1].content):
            names = [call['function']['name'] for message in messages for call in (message.tool_calls or [])]
            if 'household_artifact' not in names:
                return LLMResponse(tool_calls=[ToolCall(f'browser-publish-{self.calls}', 'household_artifact', {
                    'action': 'publish', 'path': 'workshop/browser-shared.txt', 'title': 'A public browser artifact'})])
            if 'contact' not in names:
                return LLMResponse(tool_calls=[ToolCall(f'browser-world-contact-{self.calls}', 'contact', {
                    'action': 'create', 'participants': ['operator'], 'mentions': ['operator'],
                    'title': 'World Thread browser ping', 'message': '@Tester I have something to share from my autonomous turn.'})])
            return LLMResponse(tool_calls=[ToolCall(f'browser-world-end-{self.calls}', 'end_turn', {'status': 'completed'})])
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
for descriptor in registry.descriptors.values():
    project = Path(descriptor['root']) / 'workspace/workshop'
    project.mkdir(exist_ok=True)
    (project / 'browser-shared.txt').write_text('Published by a real native worker.\n')
    (project / 'owner.txt').write_text(descriptor['name']+' owns this private workshop file.\n')
registry.store.put('shared_note', 'browser-shared-note', {'author': list(registry.descriptors)[1],
    'text': 'A notebook note visible across the household.', 'source_ids': []})
password = secrets.token_urlsafe(24)
config.auth.password_hash = hash_password(password)
credential = Path(os.environ.get("TAMAGI_BROWSER_CREDENTIAL", "/tmp/tamagi-native-browser-credential"))
credential.write_text(password)
credential.chmod(0o600)
app = create_household_app(config, provider=Provider(registry), registry=registry)

# Disposable browser-fixture control, never installed in the application. Prior
# chat checks intentionally leave real operator decompression in place. Clear it
# only for this explicit test, without waiting fifteen wall-clock minutes.
@app.post('/api/fixture/prepare-world')
async def prepare_world():
    for descriptor in registry.descriptors.values():
        with closing(sqlite3.connect(Path(descriptor['root']) / 'runtime.sqlite3')) as db:
            db.execute("UPDATE meta SET value='0' WHERE key='decompression_until'")
            db.commit()
    config.autonomy.enabled = True
    state = registry.store.meta('native_world_schedule')
    registry.store.set_meta('native_world_schedule', {**state, 'next_at': registry.store.clock() + 900})
    return {'ready': True}
app.router.routes.insert(0, app.router.routes.pop())

@app.get('/api/fixture/world-evidence')
async def world_evidence():
    private = []
    for descriptor in registry.descriptors.values():
        with closing(sqlite3.connect(Path(descriptor['root']) / 'runtime.sqlite3')) as db:
            private.append({'name': descriptor['name'], 'receipts': [json.loads(row[0]) for row in db.execute('SELECT payload FROM receipts ORDER BY started DESC LIMIT 4')]})
    return {'world': app.state.world_scheduler.status(),
            'works': [app.state.interactions.public(r) for r in app.state.interactions.records() if r['data']['kind'] == 'world'],
            'messages': [r['data'] for r in registry.store.list('message')][-5:], 'private': private}
app.router.routes.insert(0, app.router.routes.pop())
print("Fresh native browser household: " + str(root), flush=True)
try:
    uvicorn.run(app, host="127.0.0.1", port=18846, access_log=False, log_level="warning")
finally:
    credential.unlink(missing_ok=True)
    temporary.cleanup()
