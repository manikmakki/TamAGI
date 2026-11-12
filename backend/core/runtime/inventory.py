"""Read-only discovery of workshop creations, separate from saved snapshots."""

import mimetypes
import os
from pathlib import Path
import stat


class Inventory:
    def __init__(self, runtime):
        self.runtime = runtime
        self.root = runtime.workspace / "workshop"

    def list(self):
        if self.root.is_symlink():
            raise ValueError("Workshop directory is unavailable")
        artifacts = self.runtime.store.list("artifact", limit=100000)
        descriptions = {
            r["data"]["artifact_id"]: r["data"].get("description", "")
            for r in self.runtime.store.list("world_object", limit=100000)
            if r["data"].get("artifact_id")
        }
        entries = []
        for path in sorted(self.root.rglob("*")):
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(self.root.resolve())
            ):
                continue
            relative = str(path.relative_to(self.root))
            # openat validation also excludes symlinks in parent components.
            try:
                with self.open(relative) as file:
                    info = os.fstat(file.fileno())
            except (OSError, ValueError):
                continue
            versions = [
                r
                for r in artifacts
                if r["data"].get("original_path") == "workshop/" + relative
            ]
            description = next(
                (
                    descriptions[r["id"]]
                    for r in reversed(versions)
                    if descriptions.get(r["id"])
                ),
                "",
            )
            entries.append(
                {
                    "path": relative,
                    "name": path.name,
                    "size": info.st_size,
                    "modified_at": info.st_mtime,
                    "media_type": mimetypes.guess_type(path.name)[0]
                    or "application/octet-stream",
                    "description": description,
                    "versions": versions,
                }
            )
            if len(entries) >= 500:
                break
        return {
            "items": entries,
            "limit": 500,
            "earlier_saved_artifacts": [
                r
                for r in artifacts
                if not r["data"].get("original_path", "").startswith("workshop/")
            ],
        }

    def open(self, relative):
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or any(part in {".", ".."} for part in relative.split("/"))
        ):
            raise ValueError("Choose a regular workshop file")
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parts[:-1]:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                os.close(directory)
                directory = child
            descriptor = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ValueError("Choose a regular workshop file")
            return os.fdopen(descriptor, "rb")
        finally:
            os.close(directory)
