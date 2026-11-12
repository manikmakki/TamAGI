"""Launch a localhost household preview in this foreground process; no live models."""

import argparse
from pathlib import Path
import secrets
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from backend.core.household.preview import create_preview_app
from backend.core.household.registry import SandboxRegistry
from scripts.qualify_household import _manifest


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7743)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose an unprivileged local TCP port")
    source = _manifest(SOURCE)
    release = source["revision"] + ("+dirty." + source["diff_sha256"][:12] if source["dirty"] else "")
    registry = SandboxRegistry.create(release, operator_name="Preview operator")
    password = secrets.token_urlsafe(12)
    app = create_preview_app(registry, password)
    print(f"\nHousehold preview: http://localhost:{args.port}/\n"
          f"Password: {password}\nFixtures: {registry.root}\n"
          "Scripted responses only; no real model, Ember state, or push delivery.\n"
          "This is a foreground process. Ctrl+C stops it; each launch creates fresh fixtures.\n", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, proxy_headers=False, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
