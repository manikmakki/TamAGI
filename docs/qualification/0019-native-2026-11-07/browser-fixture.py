import asyncio
from pathlib import Path
import secrets
import tempfile

import uvicorn
import yaml

from backend.auth import hash_password
from backend.config import TamAGIConfig
from backend.core.household.native_app import create_household_app
from backend.core.household.native_registry import NativeRegistry
from backend.core.llm import LLMResponse


class Provider:
    def __init__(self):
        self.calls = 0

    async def count_context_tokens(self, messages, tools=None):
        return None

    async def chat(self, messages, tools=None, **options):
        self.calls += 1
        await asyncio.sleep(1 if self.calls == 1 else 4)
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
credential = Path("/tmp/tamagi-native-browser-credential")
credential.write_text(password)
credential.chmod(0o600)
app = create_household_app(config, provider=Provider(), registry=registry)
print("Fresh native browser household: " + str(root), flush=True)
try:
    uvicorn.run(app, host="127.0.0.1", port=18846, access_log=False, log_level="warning")
finally:
    credential.unlink(missing_ok=True)
    temporary.cleanup()
