"""Run native Chromium regressions against disposable, offline resident workers."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from qualify_household import _guard_environment, _outside_live_tree, prepare_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = _outside_live_tree(Path(__file__).resolve().parents[1])
    output = _outside_live_tree(args.output)
    if output == source or source in output.parents:
        raise ValueError("Browser evidence must be outside the source worktree")
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="tamagi-native-browser-run-") as temporary:
        root = Path(temporary)
        cwd, guard = root / "cwd", root / "guard"
        cwd.mkdir()
        prepare_guard(guard)
        environment = _guard_environment(guard, source, cwd)
        credential = root / "credential"
        environment["TAMAGI_BROWSER_CREDENTIAL"] = str(credential)
        with (output / "fixture.log").open("w") as log:
            fixture = subprocess.Popen([sys.executable, str(source / "tests/browser/household-native-fixture.py")],
                cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 30
                while not credential.exists():
                    if fixture.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Disposable browser fixture failed to start; inspect fixture.log")
                    time.sleep(.1)
                # Credential is written before lifespan startup; give the browser
                # its own readiness retry rather than contacting an unrelated app.
                time.sleep(3)
                result = subprocess.run(["node", str(source / "tests/browser/household-native.mjs")],
                    cwd=cwd, env=environment, capture_output=True, text=True, timeout=180)
                (output / "browser.stdout").write_text(result.stdout)
                (output / "browser.stderr").write_text(result.stderr)
                if result.returncode == 0:
                    browser_report = json.loads(result.stdout.strip().splitlines()[-1])
                    (output / "browser.json").write_text(json.dumps(browser_report, indent=2) + "\n")
                    for name in ("native-desktop.png", "native-mobile-wheel.png", "native-reasoning.png", "native-invitation.png"):
                        shutil.copyfile(Path(browser_report["artifacts"]) / name, output / name)
            finally:
                fixture.terminate()
                try:
                    fixture.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    fixture.kill()
                    fixture.wait()
        denials = (guard / "denied.jsonl").read_text()
        activations = (guard / "activations.jsonl").read_text()
        (output / "denied.jsonl").write_text(denials)
        (output / "guard-activations.jsonl").write_text(activations)
    report = {"passed": result.returncode == 0 and not denials, "returncode": result.returncode,
              "guarded_python_processes": len(activations.splitlines()),
              "denied_network_operations": len(denials.splitlines()), "external_provider": False,
              "limits": "Injected provider; Python network guard and restricted Chromium DNS, not OS network confinement."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "evidence": str(output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
