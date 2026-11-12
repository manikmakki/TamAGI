"""Run the native TamAGI household in an isolated foreground process."""
from pathlib import Path
import argparse
import secrets
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))


def main():
    import uvicorn
    import yaml
    from backend.auth import hash_password
    from backend.config import load_config
    from backend.core.household.native_registry import NativeRegistry
    from backend.core.household.native_app import create_household_app
    from scripts.qualify_household import _manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Listener address; use 0.0.0.0 for LAN access")
    parser.add_argument("--port", type=int, default=7744)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--world-thread", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable the coordinator's household World Thread; worker timers stay off")
    parser.add_argument("--model-config", type=Path, help="Copy only the authorized model transport into the private coordinator config")
    parser.add_argument("--names", nargs="+", default=["Cedar", "Willow", "River"])
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose an unprivileged TCP port")
    source = _manifest(SOURCE)
    release = source["revision"] + ("+dirty." + source["diff_sha256"][:12] if source["dirty"] else "")
    if args.root and args.root.exists():
        registry = NativeRegistry(args.root)
        config = load_config(args.root / "config.yaml")
        password = None
    else:
        if not args.model_config:
            parser.error("A new household requires an explicit authorized --model-config")
        llm = yaml.safe_load(args.model_config.read_text())["llm"]
        root = args.root or Path(tempfile.mkdtemp(prefix="tamagi-native-", dir="/tmp")) / "home"
        registry = NativeRegistry.create(root, names=args.names, release=release, llm=llm, port=args.port)
        config = load_config(root / "config.yaml")
        password = secrets.token_urlsafe(12)
        config.auth.password_hash = hash_password(password)
        (root / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
        (root / "config.yaml").chmod(0o600)
    if args.world_thread is not None:
        config.autonomy.enabled = args.world_thread
        (registry.root / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
        (registry.root / "config.yaml").chmod(0o600)
    app = create_household_app(config, registry=registry)
    print(f"\nNative TamAGI: http://{args.host}:{args.port}/\nHousehold: {registry.root}\n"
          + (f"Password: {password}\n" if password else "Use this household's existing password.\n")
          + f"Real resident workers; household World Thread {'enabled (' + config.autonomy.schedule + ')' if config.autonomy.enabled else 'disabled'}. Ctrl+C stops this process.\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=False, access_log=False)


if __name__ == "__main__":
    main()
