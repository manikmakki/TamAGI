#!/usr/bin/env python3
"""Run the Phase 0 Evolution Lab static evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.evolution import EvolutionRunner


def main() -> None:
    run, run_path = EvolutionRunner().run()
    summary = run.summary
    print(f"Evolution Phase 0 gut check: {summary['total']} test(s)")
    print(
        "candidate_wins={candidate_wins} baseline_wins={baseline_wins} "
        "ties={ties} insufficient_data={insufficient_data}".format(**summary)
    )
    for result in run.results:
        print(f"- {result.test_id}: {result.decision} ({result.rationale})")
    print(f"Audit run: {run_path}")


if __name__ == "__main__":
    main()
