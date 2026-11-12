# ADR-0012: Improvement claims require retained comparative evidence

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

The static evaluator scores supplied responses and candidate-owned expectations rather than comparative executions.

## Decision and alternatives

**Decision:** Preserve current static fixtures as contract tests, labeled accordingly. Add an execution-based evaluator using the same scenario, initial state, model configuration, budgets, and runner-owned checks for baseline and candidate.

Retain manifests, failures, results, criteria versions, and evidence. Separate mechanical tests, scripted scenarios, live trials, and sustained operation.

TamAGI may propose evaluation changes, but a candidate cannot establish improvement solely by changing its own grading conditions. Criteria changes retain old/new comparisons.

**Reject:** Supplied sample text as evidence of runtime improvement, candidate-owned expectations, and a single global “better agent” score.

**Consequence:** Each experiment declares its intended outcome and regression checks before execution. Automatic behavioral promotion requires all fixed checks to pass and improvement in at least two of three matched repetitions, with no regression in the declared guard scenarios. This is an operational gate, not a claim of statistical certainty or subjective welfare.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
