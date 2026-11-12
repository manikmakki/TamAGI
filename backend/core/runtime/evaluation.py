"""Execution-based matched trials. Fixed checks belong to the runner, not the candidate."""

from __future__ import annotations
import hashlib
import inspect
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from .store import RuntimeStore, uid, encoded
from .service import RuntimeService


@dataclass(frozen=True)
class Scenario:
    id: str
    prompt: str
    target: str = "completion"
    # Runner-owned check; receives the actual trial store and result, never supplied sample text.
    check: object = None


class ComparativeEvaluator:
    def __init__(self, runtime):
        self.runtime = runtime
        self.scenarios = {}

    def register(self, scenario):
        if scenario.id in self.scenarios:
            raise ValueError("Scenario already registered")
        self.scenarios[scenario.id] = scenario

    async def compare(
        self,
        proposal_id,
        runner,
        scenario_ids=None,
        model_manifest=None,
        new_criteria=None,
    ):
        proposal = self.runtime.store.get("proposal", proposal_id)
        if not proposal:
            raise ValueError("Unknown proposal")
        selected = [self.scenarios[k] for k in (scenario_ids or list(self.scenarios))]
        if not selected:
            raise ValueError("No runner-owned scenarios selected")
        new_criteria = new_criteria or {}
        if any(key not in self.scenarios for key in new_criteria):
            raise ValueError("Criteria may not change scenario intent or inputs")
        manifest = {
            "proposal_id": proposal_id,
            "candidate_hash": hashlib.sha256(
                encoded(proposal["data"]).encode()
            ).hexdigest(),
            "scenarios": [
                {
                    "id": s.id,
                    "prompt": s.prompt,
                    "target": s.target,
                    "check_hash": hashlib.sha256(
                        inspect.getsource(s.check).encode()
                    ).hexdigest(),
                }
                for s in selected
            ],
            "model": model_manifest or {},
            "repetitions": 3,
            "evidence_class": "execution",
            "criteria_version": 1,
            "new_criteria_hashes": {
                key: hashlib.sha256(inspect.getsource(check).encode()).hexdigest()
                for key, check in new_criteria.items()
            },
            "sampling": "Matched inputs; provider RNG is not controlled",
            "baseline_runtime_revisions": self.runtime.store.list("runtime_revision"),
        }
        experiment_id = uid("experiment")
        self.runtime.store.put(
            "experiment",
            experiment_id,
            self.runtime.actor_id,
            {"manifest": manifest, "proposal_id": proposal_id, "decision": "running"},
            category="platform",
        )
        results = []
        for seed in range(3):
            for scenario in selected:
                with tempfile.TemporaryDirectory(
                    prefix="tamagi-evaluation-"
                ) as directory:
                    root = Path(directory)
                    base = RuntimeStore(root / "initial.sqlite3")
                    RuntimeService(
                        base,
                        root / "initial-workspace",
                        self.runtime.config,
                        self.runtime.source_root,
                    )
                    for revision in manifest["baseline_runtime_revisions"]:
                        base.put(
                            "runtime_revision",
                            revision["id"],
                            base.resident_id,
                            revision["data"],
                        )
                    base.close()
                    for variant in ("baseline", "candidate"):
                        workspace = root / "workspace"
                        if workspace.exists():
                            shutil.rmtree(workspace)
                        database = root / "trial.sqlite3"
                        for suffix in ("", "-wal", "-shm"):
                            Path(str(database) + suffix).unlink(missing_ok=True)
                        shutil.copy2(root / "initial.sqlite3", database)
                        store = RuntimeStore(database)
                        trial = RuntimeService(
                            store,
                            workspace,
                            self.runtime.config,
                            self.runtime.source_root,
                        )
                        try:
                            output = await runner(
                                trial,
                                scenario,
                                proposal["data"] if variant == "candidate" else None,
                                seed,
                            )
                            receipts = store.receipts(limit=1000)
                            fixed = all(
                                r["status"] in {"succeeded", "failed"} for r in receipts
                            )
                            check = scenario.check(store, output)
                            proposed_check = new_criteria.get(
                                scenario.id, scenario.check
                            )(store, output)
                            if not isinstance(check, dict) or not isinstance(
                                check.get("score"), (int, float)
                            ):
                                raise ValueError(
                                    "Runner check must return score and fixed_pass"
                                )
                            import math

                            if (
                                not math.isfinite(check["score"])
                                or not isinstance(proposed_check, dict)
                                or not isinstance(
                                    proposed_check.get("score"), (int, float)
                                )
                                or not math.isfinite(proposed_check["score"])
                            ):
                                raise ValueError(
                                    "Criteria scores must be finite numbers"
                                )
                            record = {
                                "scenario": scenario.id,
                                "seed": seed,
                                "variant": variant,
                                "score": check["score"],
                                "fixed_pass": fixed
                                and bool(check.get("fixed_pass", False)),
                                "checks": check,
                                "new_score": proposed_check["score"],
                                "new_checks": proposed_check,
                                "receipts": receipts,
                                "events": store.events(limit=500),
                            }
                            results.append(record)
                            self.runtime.store.put(
                                "experiment",
                                experiment_id,
                                self.runtime.actor_id,
                                {
                                    "manifest": manifest,
                                    "proposal_id": proposal_id,
                                    "decision": "running",
                                    "results": results,
                                },
                                category="platform",
                            )
                        finally:
                            store.close()
        pairs = []
        new_pairs = []
        for seed in range(3):
            for scenario in selected:
                pair = [
                    r
                    for r in results
                    if r["seed"] == seed and r["scenario"] == scenario.id
                ]
                baseline = next(r for r in pair if r["variant"] == "baseline")
                candidate = next(r for r in pair if r["variant"] == "candidate")
                pairs.append(
                    (seed, scenario.id, candidate["score"] - baseline["score"])
                )
                new_pairs.append(
                    (seed, scenario.id, candidate["new_score"] - baseline["new_score"])
                )
        fixed = all(
            r["fixed_pass"] and r["new_checks"].get("fixed_pass", False)
            for r in results
        )
        no_regression = all(delta >= 0 for _, _, delta in pairs)
        improved_seeds = {seed for seed, _, delta in pairs if delta > 0}
        new_no_regression = all(delta >= 0 for _, _, delta in new_pairs)
        new_improved = {seed for seed, _, delta in new_pairs if delta > 0}
        decision = (
            "promote"
            if fixed
            and no_regression
            and new_no_regression
            and len(improved_seeds) >= 2
            and len(new_improved) >= 2
            else "retain_baseline"
        )
        result = {
            "proposal_id": proposal_id,
            "manifest": manifest,
            "decision": decision,
            "results": results,
            "fixed_checks_pass": fixed,
            "no_regression": no_regression,
            "improved_repetitions": len(improved_seeds),
            "new_criteria_no_regression": new_no_regression,
            "criteria_decision": (
                "retained" if decision == "promote" else "quarantined"
            )
            if new_criteria
            else "unchanged",
        }
        return self.runtime.store.put(
            "experiment",
            experiment_id,
            self.runtime.actor_id,
            result,
            category="platform",
        )
