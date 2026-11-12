#!/usr/bin/env python3
"""Stable stdlib-only supervisor; copy into .platform/bootstrap.py at installation.

Same-host root may edit this controller. It is a recovery mechanism, not an authority boundary.
"""

import argparse, json, os, signal, sqlite3, subprocess, time, urllib.request
from pathlib import Path


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def select(root, target):
    temporary = root / "current.next"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target, target_is_directory=True)
    temporary.replace(root / "current")


def reconcile(root, state, healthy=None):
    if state.get("phase") == "switching":
        select(root, Path(state["candidate"]))
        state["phase"] = "verifying"
    elif state.get("phase") == "verifying" and healthy is not None:
        if healthy:
            state["phase"] = "committed"
        elif state.get("previous") and Path(state["previous"]).is_dir():
            select(root, Path(state["previous"]))
            state["phase"] = "rolled_back"
        else:
            state["phase"] = "recovery_required"
    atomic(root / "deployment.json", state)
    return state


def serve(root, port):
    stop = False
    child = None

    def shutdown(*_):
        nonlocal stop
        stop = True
        if child and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    while not stop:
        active = json.loads((root / "active.json").read_text())
        state = (
            json.loads((root / "deployment.json").read_text())
            if (root / "deployment.json").exists()
            else {}
        )
        if state.get("phase") == "quiescing":
            state["phase"] = "switching"
            atomic(root / "deployment.json", state)
        if state.get("phase") == "switching":
            state = reconcile(root, state)
        if state.get("phase") == "recovery_required":
            raise SystemExit(
                "No healthy previous release; restore through the recovery CLI"
            )
        release = (root / "current").resolve()
        environment = {
            **os.environ,
            "TAMAGI_CONFIG": active["config"],
            "PYTHONPATH": str(release / "source"),
        }
        child = subprocess.Popen(
            [str(release / "venv/bin/python"), "-m", "backend.main"],
            cwd=active["instance"],
            env=environment,
        )
        verifying = state.get("phase") == "verifying"
        healthy = False
        deadline = time.monotonic() + 90
        while child.poll() is None and not stop:
            # New activation requests are durable even if the application process is broken.
            pending = (
                json.loads((root / "deployment.json").read_text())
                if (root / "deployment.json").exists()
                else {}
            )
            new_active = json.loads((root / "active.json").read_text())
            if pending.get("phase") == "quiescing":
                connection = sqlite3.connect(
                    Path(pending["instance"]) / "runtime.sqlite3"
                )
                try:
                    running = connection.execute(
                        "SELECT count(*) FROM runs WHERE status='running'"
                    ).fetchone()[0]
                finally:
                    connection.close()
                if not running or time.time() - pending.get("created_at", 0) > 30:
                    pending["phase"] = "switching"
                    atomic(root / "deployment.json", pending)
            if pending.get("phase") == "switching" or new_active != active:
                child.terminate()
                break
            if verifying:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/health/ready", timeout=2
                    ) as response:
                        health = json.load(response)
                    healthy = (
                        bool(health.get("data_ready"))
                        and health.get("release") == release.name
                    )
                except Exception:
                    healthy = False
                if healthy:
                    state = reconcile(root, state, True)
                    verifying = False
                elif time.monotonic() > deadline:
                    child.terminate()
                    break
            time.sleep(1)
        if child.poll() is None:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if verifying and not healthy:
            reconcile(root, state, False)
        if not stop:
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=7741)
    args = parser.parse_args()
    serve(args.root.resolve(), args.port)
