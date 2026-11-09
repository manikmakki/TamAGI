"""Offline encrypted resident backup, preview, replacement and migration.

Passwords are read interactively (or from an inherited FD), never argv/config.
This command does not stop, restart, or select any running installation.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))


def main():
    from backend.core.household.resident_recovery import (
        ResidentArchive, backup_resident, migrate_resident, restore_resident, stopped_household)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--password-fd", type=int, help="Read one password line from an inherited descriptor")
    commands = parser.add_subparsers(dest="action", required=True)
    export = commands.add_parser("backup", help="Checkpoint one resident in a stopped household")
    export.add_argument("--root", required=True, type=Path)
    export.add_argument("--resident", required=True)
    export.add_argument("--output", required=True, type=Path)
    for name in ("preview", "restore", "migrate"):
        command = commands.add_parser(name)
        command.add_argument("--archive", required=True, type=Path)
        if name != "preview":
            command.add_argument("--confirm", action="store_true", help="Accept checkpoint replacement and its continuity gap")
            command.add_argument("--source-stopped", action="store_true", help="Acknowledge the source resident is no longer active")
            command.add_argument("--authority", required=True, choices=("guarded", "unrestricted_host"))
        if name == "restore":
            command.add_argument("--root", required=True, type=Path)
            command.add_argument("--resident", required=True)
        elif name == "migrate":
            command.add_argument("--destination", required=True, type=Path)
            command.add_argument("--release", required=True, help="Compatible selected application release; native archives require an exact match")
            command.add_argument("--port", type=int, default=7744)
    args = parser.parse_args()
    if args.password_fd is None:
        password = getpass.getpass("Archive password: ")
        if args.action == "backup" and getpass.getpass("Repeat archive password: ") != password:
            parser.error("Passwords do not match")
    else:
        with os.fdopen(os.dup(args.password_fd)) as stream:
            password = stream.readline(1026).rstrip("\n")
    try:
        if args.action == "backup":
            with stopped_household(args.root) as registry:
                result = backup_resident(registry, args.resident, args.output, password)
        else:
            with tempfile.TemporaryDirectory(prefix="tamagi-resident-preview-") as temporary:
                archive = ResidentArchive(temporary)
                try:
                    preview = archive.stage(args.archive, password)
                    if args.action == "preview":
                        result = preview
                    elif args.action == "restore":
                        with stopped_household(args.root, damaged_resident=args.resident) as registry:
                            result = restore_resident(registry, archive, preview, args.resident,
                                confirmed=args.confirm, source_stopped=args.source_stopped, authority=args.authority)
                    else:
                        if not 1024 <= args.port <= 65535:
                            raise ValueError("Choose an unprivileged listener port")
                        result = migrate_resident(archive, preview, args.destination, release=args.release,
                            confirmed=args.confirm, source_stopped=args.source_stopped,
                            authority=args.authority, port=args.port)
                finally:
                    archive.cleanup()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, PermissionError, LookupError, OSError) as error:
        parser.exit(1, f"Recovery failed: {error}\n")


if __name__ == "__main__":
    main()
