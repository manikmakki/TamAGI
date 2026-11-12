"""Concise, read-only lifecycle guidance for resident runtime experiments."""

import hashlib

from .store import encoded


def change_call(action, **arguments):
    return {"tool": "runtime_change", "arguments": {"action": action, **arguments}}


def proposal_status(runtime, proposal_id):
    store = runtime.store
    proposal = store.get("proposal", proposal_id)
    if not proposal:
        raise ValueError(
            f"Unknown proposal_id {proposal_id!r}. Use runtime_change(action='inspect') "
            "to find a proposal ID, or action='propose' to create one."
        )
    data = proposal["data"]
    experiments = sorted(
        (
            r
            for r in store.list("experiment")
            if r["data"].get("proposal_id") == proposal_id
        ),
        key=lambda r: (r["created"], r["id"]),
        reverse=True,
    )
    digest = hashlib.sha256(encoded(data).encode()).hexdigest()
    qualifying = next(
        (
            r
            for r in experiments
            if r["data"].get("decision") == "promote"
            and r["data"].get("manifest", {}).get("candidate_hash") == digest
        ),
        None,
    )
    jobs = sorted(
        (
            r
            for r in store.list("evaluation_job")
            if r["data"].get("proposal_id") == proposal_id
        ),
        key=lambda r: (r["created"], r["id"]),
        reverse=True,
    )
    tested = any(
        r["data"].get("proposal_id") == proposal_id and r["data"].get("success")
        for r in store.list("tool_trial")
    )
    target = data["value"]["name"] if data["surface"] == "tool" else data["surface"]
    revision = store.get("runtime_revision", target)
    result = {
        "proposal_id": proposal_id,
        "target": target,
        "surface": data["surface"],
        "worker_test_passed": tested if data["surface"] == "tool" else None,
        "qualifying_experiment_id": qualifying["id"] if qualifying else None,
    }
    if (
        revision
        and revision["data"].get("active")
        and revision["data"].get("proposal_id") == proposal_id
    ):
        result.update(
            status="active",
            message="This proposal is active. Invoke the named tool with representative arguments to confirm normal use."
            if data["surface"] == "tool"
            else "This runtime revision is active.",
            next_call=None,
        )
    elif data["surface"] == "tool" and not tested:
        result.update(
            status="needs_worker_test",
            message="Run test_tool with this proposal_id and representative arguments. A successful worker test is required separately from comparative evaluation.",
            next_call=change_call("test_tool", proposal_id=proposal_id, arguments={}),
        )
    elif qualifying:
        result.update(
            status="ready_to_activate",
            message="Comparative evaluation passed (decision=promote). Activate using BOTH IDs below; another evaluation is unnecessary.",
            next_call=change_call(
                "activate", proposal_id=proposal_id, experiment_id=qualifying["id"]
            ),
        )
    elif jobs and jobs[0]["data"].get("status") in {"queued", "running"}:
        result.update(
            status="evaluation_pending",
            job_id=jobs[0]["id"],
            message="An evaluation is already queued or running. Inspect again after evaluation_completed; do not submit duplicate evaluations. To wait silently, end_turn(status='waiting', wake_event='evaluation_completed').",
            next_call=change_call("inspect", proposal_id=proposal_id),
        )
    elif jobs and jobs[0]["data"].get("status") in {"interrupted", "failed"}:
        job = jobs[0]["data"]
        result.update(
            status="evaluation_" + job["status"],
            message="The latest evaluation "
            + job["status"]
            + ". Any unfinished experiment is incomplete evidence, even if its stored decision says running. "
            + (
                "Inspect the job error and correct its cause before evaluating again."
                if job["status"] == "failed"
                else "Submit evaluate again to start a fresh comparison."
            ),
            next_call=change_call(
                "inspect", proposal_id=proposal_id, include_evidence=True
            )
            if job["status"] == "failed"
            else change_call("evaluate", proposal_id=proposal_id),
        )
    elif experiments:
        result.update(
            status="no_qualifying_evaluation",
            message="No experiment approves this exact proposal version. Only the overall decision=promote qualifies; a successful scenario alone does not. Inspect the evidence before revising or retrying.",
            next_call=change_call(
                "inspect", proposal_id=proposal_id, include_evidence=True
            ),
        )
    else:
        result.update(
            status="needs_evaluation",
            message="Submit evaluate with proposal_id to start the automated baseline/candidate comparison. It returns a job_id; the completed experiment supplies the experiment_id used for activation.",
            next_call=change_call("evaluate", proposal_id=proposal_id),
        )
    return result


def inspect_changes(runtime, proposal_id="", include_evidence=False):
    store = runtime.store
    if proposal_id:
        proposal_status(runtime, proposal_id)  # Validate before building the view.
    proposals = (
        [store.get("proposal", proposal_id)] if proposal_id else store.list("proposal")
    )
    selected = {r["id"] for r in proposals}
    experiments = [
        r for r in store.list("experiment") if r["data"].get("proposal_id") in selected
    ]
    jobs = [
        r
        for r in store.list("evaluation_job")
        if r["data"].get("proposal_id") in selected
    ]
    trials = [
        r for r in store.list("tool_trial") if r["data"].get("proposal_id") in selected
    ]
    active = [
        r
        for r in store.list("runtime_revision")
        if not proposal_id or r["data"].get("proposal_id") == proposal_id
    ]
    return {
        "changes": [proposal_status(runtime, r["id"]) for r in proposals],
        "experiments": experiments
        if include_evidence
        else [
            {
                "id": r["id"],
                "proposal_id": r["data"].get("proposal_id"),
                "decision": r["data"].get("decision"),
                "fixed_checks_pass": r["data"].get("fixed_checks_pass"),
                "no_regression": r["data"].get("no_regression"),
                "improved_repetitions": r["data"].get("improved_repetitions"),
                "trial_count": len(r["data"].get("results", [])),
            }
            for r in experiments
        ],
        "jobs": jobs
        if include_evidence
        else [{"id": r["id"], **r["data"]} for r in jobs],
        "tool_trials": trials
        if include_evidence
        else [
            {
                "id": r["id"],
                "proposal_id": r["data"].get("proposal_id"),
                "success": r["data"].get("success"),
                "error": r["data"].get("result", {}).get("error"),
            }
            for r in trials
        ],
        "proposals": proposals
        if include_evidence
        else [
            {
                "id": r["id"],
                "surface": r["data"]["surface"],
                "intention": r["data"]["intention"],
            }
            for r in proposals
        ],
        "active": active
        if include_evidence
        else [
            {
                "target": r["id"],
                "revision": r["revision"],
                **{
                    key: r["data"].get(key)
                    for key in ("active", "proposal_id", "experiment_id")
                },
            }
            for r in active
        ],
        "evidence": "Use inspect with proposal_id and include_evidence=true for full retained records. An experiment's overall decision, not its last scenario score, controls activation.",
    }
