#!/usr/bin/env python3
"""Run the Phase 0 Evolution Lab static evaluator."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.evolution import EvolutionRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 0 Evolution Lab static evaluator.")
    parser.add_argument(
        "--candidate",
        help="Optional candidate profile JSON to evaluate against its selected fixture tests.",
    )
    args = parser.parse_args()

    run, run_path = EvolutionRunner(candidate_path=args.candidate).run()
    summary = run.summary
    print(f"Evolution Phase 0 gut check: {summary['total']} test(s)")
    if run.candidate_profile:
        print(f"candidate={run.candidate_profile['candidate_id']}")
    if run.tier_classification:
        tier = run.tier_classification
        print(f"tier={tier['tier']} decision={tier['decision']} ({tier['rationale']})")
    print(
        "candidate_wins={candidate_wins} baseline_wins={baseline_wins} "
        "ties={ties} insufficient_data={insufficient_data}".format(**summary)
    )
    print(f"recommendation={summary.get('recommendation')} ({summary.get('recommendation_reason')})")
    for result in run.results:
        print(f"- {result.test_id}: {result.decision} ({result.rationale})")
    print(f"Audit run: {run_path}")


if __name__ == "__main__":
    main()
