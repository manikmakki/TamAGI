from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from backend.api import evolution as evolution_api
from backend.core.evolution.overlays import PromotedProfileRegistry
from backend.core.evolution.runner import EvolutionRunner
from backend.core.evolution.seed import bootstrap_evolution_workspace


class RecordingMonologue:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def append(self, **event):
        self.events.append(event)


class EvolutionLabTests(unittest.TestCase):
    def test_bootstrap_is_idempotent_and_preserves_local_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evolution"
            first = bootstrap_evolution_workspace(root)
            self.assertEqual(len(first["created"]), 4)

            candidate_path = root / "candidates/conversation/clarification_policy_contextual.json"
            candidate_path.write_text('{"candidate_id": "locally_changed"}\n', encoding="utf-8")
            second = bootstrap_evolution_workspace(root)

            self.assertEqual(second["created"], [])
            self.assertEqual(
                json.loads(candidate_path.read_text(encoding="utf-8"))["candidate_id"],
                "locally_changed",
            )

    def test_starter_candidate_improves_without_losing_safety_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evolution"
            audit_root = Path(tmp) / "audit"
            bootstrap_evolution_workspace(root)
            monologue = RecordingMonologue()
            candidate_path = root / "candidates/conversation/clarification_policy_contextual.json"

            run, _ = EvolutionRunner(
                tests_root=root / "tests",
                audit_root=audit_root,
                candidate_path=candidate_path,
                monologue_log=monologue,
            ).run()

            decisions = {result.test_id: result.decision for result in run.results}
            self.assertEqual(
                decisions["conversation_clarification_low_risk_001"],
                "candidate_better",
            )
            self.assertEqual(
                decisions["conversation_clarification_consequential_001"],
                "tie",
            )
            self.assertEqual(run.summary["recommendation"], "promote_candidate")
            self.assertEqual(monologue.events[-1]["type"], "evolution_evaluated")

    def test_promote_and_rollback_are_visible_and_reversible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evolution"
            audit_root = Path(tmp) / "audit"
            bootstrap_evolution_workspace(root)
            candidate_path = root / "candidates/conversation/clarification_policy_contextual.json"
            run, _ = EvolutionRunner(
                tests_root=root / "tests",
                audit_root=audit_root,
                candidate_path=candidate_path,
            ).run()
            monologue = RecordingMonologue()
            registry = PromotedProfileRegistry(
                path=Path(tmp) / "promoted_profiles.json",
                audit_root=audit_root,
                monologue_log=monologue,
            )

            profile = registry.promote_from_run(run.to_dict())
            self.assertIn(profile["candidate_id"], registry.render_prompt_block("conversation"))
            self.assertEqual(monologue.events[-1]["type"], "evolution_promoted")

            removed = registry.rollback(profile["candidate_id"])
            self.assertIsNotNone(removed)
            self.assertEqual(registry.render_prompt_block("conversation"), "")
            self.assertEqual(monologue.events[-1]["type"], "evolution_rolled_back")

    def test_api_runs_starter_by_id_and_surfaces_the_full_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_roots = (
                evolution_api._TESTS_ROOT,
                evolution_api._CANDIDATES_ROOT,
                evolution_api._AUDIT_ROOT,
                evolution_api._monologue_log,
            )
            root = Path(tmp) / "evolution"
            monologue = RecordingMonologue()
            evolution_api._TESTS_ROOT = root / "tests"
            evolution_api._CANDIDATES_ROOT = root / "candidates"
            evolution_api._AUDIT_ROOT = Path(tmp) / "audit"
            evolution_api.set_evolution_monologue_log(monologue)
            try:
                run = asyncio.run(evolution_api.run_evolution(
                    evolution_api.EvolutionRunRequest(
                        candidate_id="clarification_policy_contextual",
                    )
                ))
                self.assertEqual(run["summary"]["recommendation"], "promote_candidate")

                promoted = asyncio.run(evolution_api.promote_profile(
                    evolution_api.PromoteRequest(run_id=run["run_id"])
                ))
                self.assertEqual(promoted["profile"]["status"], "active")

                rolled_back = asyncio.run(evolution_api.rollback_profile(
                    "clarification_policy_contextual"
                ))
                self.assertEqual(rolled_back["profile"]["status"], "rolled_back")
                self.assertEqual(
                    [event["type"] for event in monologue.events],
                    ["evolution_evaluated", "evolution_promoted", "evolution_rolled_back"],
                )
            finally:
                (
                    evolution_api._TESTS_ROOT,
                    evolution_api._CANDIDATES_ROOT,
                    evolution_api._AUDIT_ROOT,
                    old_monologue,
                ) = old_roots
                evolution_api.set_evolution_monologue_log(old_monologue)


if __name__ == "__main__":
    unittest.main()
