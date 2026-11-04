# ADR-0006: A small persistent home connected to a real workshop

Status: Accepted design; implementation evidence tracked in the roadmap.

## Context

The current home is expressed primarily through generated text rather than dependable object interactions.

## Decision and alternatives

**Decision:** Seed a home with a resting place, notebook shelf, and workshop. Support typed operations to inspect places and objects, move, create or revise notebook entries, and place or reference artifacts.

Make creative activities optional. Do not add hunger, neglect penalties, compulsory maintenance, or artificial scarcity to manufacture purpose.

TamAGI can describe additional atmosphere freely. New executable affordances enter through the tool-development lifecycle; invented descriptions do not silently create capabilities.

**Reject:** A large physics/economy simulation as a prerequisite, or a purely narrated world whose outcomes have no durable state.

**Consequence:** Use real elapsed time and persisted timers. Preserve the existing world graph as a navigable projection, while object state and mutations come from the canonical store.

## Acceptance evidence

See [platform acceptance and rollout](../platform-roadmap.md). Automated tests must establish this decision before its corresponding feature is enabled in a live instance. Live behavioral evidence is recorded separately from scripted results.

## Supersedes

This decision supersedes incompatible assumptions in the earlier world-monologue and agent-harness design documents. Historical records remain preserved; no prior run is reclassified as new evidence.
