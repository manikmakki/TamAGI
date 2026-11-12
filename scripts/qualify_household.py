"""Run the complete suite from disposable state with Python network access guarded.

This is a development qualification runner, not a live instance command. It does
not load an installation, call a model, or select an application release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


# Loaded by every Python test child. Popen propagation matters because existing
# application smoke tests deliberately replace PYTHONPATH in their children.
SOCKET_GUARD = r'''
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import time

_directory = Path(os.environ["TAMAGI_QUALIFICATION_GUARD"]).parent
_ports = _directory / "ports.json"
_denials = _directory / "denied.jsonl"
_activations = _directory / "activations.jsonl"

def _append(path, value):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()

def _deny(operation, address):
    # Never record URLs, headers, request bodies, credentials, or environment.
    _append(_denials, {"pid": os.getpid(), "operation": operation,
                      "address": repr(address)[:256], "time": time.time()})
    raise PermissionError("Offline qualification blocked network access")

def _loopback(host):
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return str(host).lower() == "localhost"

def _allowed_ports(add=None):
    with _ports.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        values = json.load(stream)
        if add is not None and add not in values:
            values.append(add)
            stream.seek(0)
            json.dump(values, stream)
            stream.truncate()
            stream.flush()
        return values

def _destination(operation, address):
    if (not isinstance(address, tuple) or len(address) < 2
            or not _loopback(address[0]) or address[1] == 11434
            or address[1] not in _allowed_ports()):
        _deny(operation, address)

_bind = socket.socket.bind
_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_sendto = socket.socket.sendto
_sendmsg = getattr(socket.socket, "sendmsg", None)
_getaddrinfo = socket.getaddrinfo
_gethostbyname = socket.gethostbyname
_gethostbyname_ex = socket.gethostbyname_ex
_gethostbyaddr = socket.gethostbyaddr

def _guarded_bind(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        if not _loopback(address[0]) or address[1] == 11434:
            _deny("bind", address)
        result = _bind(self, address)
        if self.type & socket.SOCK_STREAM:
            _allowed_ports(self.getsockname()[1])
        return result
    return _bind(self, address)

def _guarded_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _destination("connect", address)
    return _connect(self, address)

def _guarded_connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _destination("connect_ex", address)
    return _connect_ex(self, address)

def _guarded_sendto(self, data, *args):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _deny("sendto", args[-1] if args else None)
    return _sendto(self, data, *args)

def _guarded_sendmsg(self, buffers, *args):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _deny("sendmsg", args[-1] if args else None)
    return _sendmsg(self, buffers, *args)

def _guarded_getaddrinfo(host, port, *args, **kwargs):
    if host is not None and not _loopback(host):
        _deny("getaddrinfo", (host, port))
    return _getaddrinfo(host, port, *args, **kwargs)

def _guarded_gethostbyname(host):
    if not _loopback(host):
        _deny("gethostbyname", host)
    return _gethostbyname(host)

def _guarded_gethostbyname_ex(host):
    if not _loopback(host):
        _deny("gethostbyname_ex", host)
    return _gethostbyname_ex(host)

def _guarded_gethostbyaddr(host):
    if not _loopback(host):
        _deny("gethostbyaddr", host)
    return _gethostbyaddr(host)

_popen_init = subprocess.Popen.__init__
def _guarded_popen(self, *args, **kwargs):
    # Preserve all inherited guard settings even when a test supplies a new env.
    environment = dict(kwargs.get("env") or os.environ)
    directory = _directory
    requested = environment.get("TAMAGI_QUALIFICATION_GUARD")
    if requested and Path(requested).parent != _directory:
        # The runner's own tests nest a qualification run. A byte-identical guard
        # gets its own ledger so expected guard-denial fixtures remain explicit.
        candidate = Path(requested)
        if candidate.is_file() and candidate.read_bytes() == (_directory / "sitecustomize.py").read_bytes():
            directory = candidate.parent
    environment["TAMAGI_QUALIFICATION_GUARD"] = str(directory / "sitecustomize.py")
    paths = environment.get("PYTHONPATH", "").split(os.pathsep)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(directory)] + [p for p in paths if p and p not in (str(_directory), str(directory))]
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    kwargs["env"] = environment
    return _popen_init(self, *args, **kwargs)

try:
    socket.socket.bind = _guarded_bind
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.socket.sendto = _guarded_sendto
    if _sendmsg is not None:
        socket.socket.sendmsg = _guarded_sendmsg
    socket.getaddrinfo = _guarded_getaddrinfo
    socket.gethostbyname = _guarded_gethostbyname
    socket.gethostbyname_ex = _guarded_gethostbyname_ex
    socket.gethostbyaddr = _guarded_gethostbyaddr
    subprocess.Popen.__init__ = _guarded_popen
    _append(_activations, {"pid": os.getpid()})
except BaseException:
    # Python normally ignores sitecustomize errors, which would fail open.
    os._exit(97)
'''


def _outside_live_tree(path: Path) -> Path:
    resolved = path.resolve()
    live = Path("/opt/TamAGI").resolve()
    if resolved == live or live in resolved.parents:
        raise ValueError("Qualification paths must be outside /opt/TamAGI")
    return resolved


def _manifest(source: Path) -> dict:
    def git(*arguments):
        return subprocess.check_output(["git", "--no-optional-locks", "-C", str(source), *arguments])

    revision = git("rev-parse", "HEAD").decode().strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    digest = hashlib.sha256(git("diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD"))
    for name in sorted(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
        if not name:
            continue
        path = source / os.fsdecode(name)
        digest.update(name + b"\0")
        if path.is_symlink():
            digest.update(os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"revision": revision, "dirty": dirty, "diff_sha256": digest.hexdigest()}


def _guard_environment(guard: Path, source: Path, cwd: Path) -> dict[str, str]:
    # Do not inherit provider credentials, proxy settings, Python startup hooks,
    # user-specific temp roots, or installation/index configuration.
    environment = {name: os.environ[name] for name in
                   ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "SYSTEMROOT")
                   if name in os.environ}
    environment.update({
        "TAMAGI_QUALIFICATION_GUARD": str(guard / "sitecustomize.py"),
        "TAMAGI_CONFIG": str(cwd / "no-live-config.yaml"),
        "TAMAGI_AUTONOMY_ENABLED": "false",
        "PYTHONPATH": os.pathsep.join((str(guard), str(source))),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return environment


def prepare_guard(guard: Path) -> None:
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(SOCKET_GUARD)
    (guard / "ports.json").write_text("[]")
    (guard / "denied.jsonl").touch()
    (guard / "activations.jsonl").touch()


def run_qualification(source: Path, output: Path, *, timeout: float = 600) -> dict:
    source = _outside_live_tree(source)
    output = _outside_live_tree(output)
    if output == source or source in output.parents:
        raise ValueError("Evidence must be outside the source worktree")
    if not (source / "tests").is_dir():
        raise ValueError("Source must contain the tests directory")
    output.mkdir(parents=True, exist_ok=False)
    (output / "disposable-qualification.json").write_text(json.dumps({
        "purpose": "offline household development tests", "live_instance": False,
    }) + "\n")
    before = _manifest(source)
    started = time.time()
    command = [sys.executable, "-m", "unittest", "discover", "-s", str(source / "tests"), "-q"]
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="tamagi-offline-suite-", dir="/tmp") as temporary:
        disposable = Path(temporary)
        guard = disposable / "guard"
        cwd = disposable / "cwd"
        cwd.mkdir()
        prepare_guard(guard)
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(
                command, cwd=cwd, env=_guard_environment(guard, source, cwd),
                stdout=stdout, stderr=stderr, start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal

                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = process.returncode
        denied_text = (guard / "denied.jsonl").read_text()
        activation_text = (guard / "activations.jsonl").read_text()
        (output / "denied.jsonl").write_text(denied_text)
        (output / "guard-activations.jsonl").write_text(activation_text)
    after = _manifest(source)
    unchanged = before == after
    report = {
        "schema": 1, "evidence_class": "scripted",
        "source": str(source), "source_before": before, "source_after": after,
        "command": command, "started_at": started, "finished_at": time.time(),
        "returncode": returncode, "timed_out": timed_out,
        "denied_network_operations": len(denied_text.splitlines()),
        "guarded_python_processes": len(activation_text.splitlines()),
        "source_unchanged": unchanged,
        "passed": returncode == 0 and not timed_out and not denied_text and bool(activation_text) and unchanged,
        "limits": "Python socket/DNS guard; this is not OS confinement of arbitrary native commands.",
        "live_instance_used": False, "real_model_used": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory outside the source/live tree")
    parser.add_argument("--timeout", type=float, default=600)
    arguments = parser.parse_args()
    try:
        report = run_qualification(arguments.source, arguments.output, timeout=arguments.timeout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Qualification could not run: {error}\n")
    print(json.dumps({"passed": report["passed"], "report": str(arguments.output.resolve() / "report.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
