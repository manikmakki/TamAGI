"""Run ADR-0019 fixtures without opening any active TamAGI installation."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from backend.core.household.sandbox import run_scenario
from scripts.qualify_household import _manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New report path under /tmp")
    arguments = parser.parse_args()
    output = arguments.output.absolute()
    if output != output.resolve() or not output.is_relative_to(Path("/tmp").resolve()):
        parser.error("Report must use a canonical disposable /tmp path")
    if output.exists():
        parser.error("Choose a new report path")
    source = _manifest(SOURCE)
    release = source["revision"] + ("+dirty." + source["diff_sha256"][:12] if source["dirty"] else "")
    report = asyncio.run(run_scenario(release))
    report["source"] = source
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(output), "fixtures": report["fixture_root"], "checks": report["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
