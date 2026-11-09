# ADR-0015: Growth through sustained behavior

Status: Accepted and implemented; mechanical acceptance verified on 2026-11-03.

## Context

TamAGI should retain the pleasure of watching a virtual companion grow. Removing action-based XP left its level and stage progression frozen.

Growth recognizes sustained participation, continuity, and adaptation. A lively conversation counts more than a brief greeting. The rules remain understandable without a large achievement catalog such as [Hermes’s tiered collection](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/built-in-plugins.md).

## Decision

Use **Growth points** awarded by a small, deterministic set of behavioral rules. Display age separately. There are no daily quotas, cooldowns, streaks, deductions, or rewards for mere passage of time. Levels grant no additional authority, resources, or capabilities.

| Pattern | Observable evidence | Growth |
|---|---|---|
| Engagement | Sustained reciprocal conversation with the user or a companion: three completed exchanges receive initial recognition; eight receive deeper recognition. | 1 point at three exchanges; 3 points total at eight. |
| Continuity | In a later occasion, the resident returns to an existing pursuit and changes its next step or adds new artifact/evidence references. Completion is unnecessary. Native pursuit and world-graph adapter updates use the same canonical revisions. | 1 point per occasion, regardless of how many pursuits qualify. |
| Adaptation | The resident explicitly connects an earlier failed operation to a later successful attempt using the same tool with changed arguments. | 1 point per occasion, regardless of how many corrections qualify. |

The patterns are additive. Conversation length is an approximate sign of sustained participation; it does not establish emotional or intellectual depth. No additional model judge is introduced.

Only exchanges in which the primary resident participates contribute to its engagement. An operator’s direct conversation with a companion does not count toward the resident’s Growth.

An **occasion** is a conversation episode or one autonomous activation. A conversation episode starts with a new thread or after two hours without an exchange. This boundary groups activity; it imposes no waiting requirement for growth. Companion invocations within an autonomous activation belong to that activation. Only completed participant exchanges count toward engagement; tool rounds, internal summaries, and intermediate assistant messages do not count.

For cumulative Growth `G`, **level = 1 + floor(sqrt(G / 5))**. The first thresholds are 0, 5, 20, and 45 Growth for levels 1–4. The UI shows progress toward the next threshold.

The resident may name each new stage during an ordinary interaction. Numeric levels remain platform-computed. Naming is optional and requires no extra inference call; an unnamed stage displays “Stage N.” The initial egg stage remains available.

## Implementation and consequences

- Keep a canonical SQLite recognition ledger with actor, rule version, occasion, pattern, points, source references, observed timestamps, and recognition timestamps. Award each criterion once; the engagement upgrade adds only the remaining two points.
- Record trusted conversation/occasion identifiers with executions. Optional `repair_of_receipt_id` tool-call metadata cites a failed operation. Dispatch validates the reference and removes this metadata before calling the tool; unknown outcomes cannot qualify as corrections.
- Observe normal conversation persistence and completed receipts. Rest, provider outages, and idle periods trigger no progression inference.
- Expose authenticated growth inspection with “Why this level?” explanations. Replace the XP label with Growth; retain existing numeric API fields as compatibility projections and provide explicit growth details.
- Replace the legacy XP-derived level/stage calculations and automatic stage-naming call. Remove residual action awards, including knowledge-feeding XP. Preserve stage names with attribution and history.
- Apply progression to the primary resident in this version. Companion conversations can contribute to its engagement; companions receive no separate leveling system yet.
- Replay sufficiently evidenced records from the current resident at rollout. Recognize them at rollout while preserving their original event timestamps. Exclude predecessor material and evaluation trials; preserve historical legacy XP separately.
- Include recognition history in portable backups. Replays, restarts, and restores preserve totals without duplicate awards. Supported interfaces offer no arbitrary point-granting operation; host-root mutability remains as documented in ADR-0002.

`GET /api/runtime/growth` provides the authenticated inspection view. `/api/state` retains its level, experience, and stage compatibility fields and adds explicit growth state. Ordinary model interaction can use `name_stage` to supply a name; it cannot supply a numeric level.

## Rejected alternatives

- Raw action XP rewards volume without establishing a behavioral pattern.
- Age-only levels or daily allowances let time determine too much of progression.
- Model-assigned scores introduce grading variability and extra inference.
- Large achievement trees or mandatory categories add complexity and can turn optional interests into obligations.

## Acceptance evidence

At the 2026-11-03 introduction, the suite had 82 passing tests, including 23 growth-specific cases. This is historical acceptance evidence, not the current suite total. A disposable application test completed 16 chat turns with exactly 16 provider calls, confirming that recognition and naming added no inference in that scenario. The checks cover:

- A greeting earns zero engagement Growth; three exchanges earn one point; eight earn three cumulatively.
- Continuity and adaptation earn their defined points, including during autonomous work.
- Repeated writes, unchanged retries, narration, and arbitrary level claims grant no points.
- A failed operation alone earns nothing; a valid linked correction can qualify.
- Multiple patterns add correctly; duplicate processing never repeats an award.
- Growth and stage history survive interruption, restart, and portable restoration.
- Rest or absence never deducts progress or creates catch-up obligations.

### Retrospective case: research carried into conversation

A disposable checkpoint replay on 2026-11-03 used rule version `adr-0015.v1` against the current resident’s canonical records. The source installation was not changed by this preview. It recognized **8 Growth, corresponding to Level 2**: six engagement points across two conversation episodes and two continuity points across two autonomous returns. Adaptation contributed zero.

For the bounded-personhood sequence specifically:

| Criterion | Retained evidence | Recognition |
|---|---|---|
| Engagement | Conversation `f868e037-2571-4d98-9b34-3924c1d9e374` contained nine completed exchanges. | 1 point at three exchanges, then 2 more at eight: **3 total**. |
| Continuity | During autonomous execution `run-68e9f6df7347459ab87e9f887bef2d46`, pursuit `pursuit-001` advanced from revision 1 to revision 2. The successful operation is retained as `receipt-d4c314e7c8d947b6b47816a31f9dd4a9`. | **1 point**. |
| Adaptation | Historical failures and later successes lacked explicit repair links. | **Unscored**. |

The observed research-to-conversation case therefore supports four of the eight previewed points. The later pursuit return from revision 2 to revision 3 accounts for the second continuity point; it is a separate occasion. Earlier companion invocations omit conversation-mode/caller provenance and remain unscored. Cross-context discussion is useful behavioral evidence, but is not an additional fourth Growth pattern.

The recognition ledger keeps the original source timestamps and a separate retrospective recognition timestamp. The 8-point figure is the checkpoint preview, not a promise that later live records remain unchanged. This case demonstrates application of the rules; mechanical verification and the seven-day operational pilot remain separate evidence classes. See [implementation and evidence](../platform-roadmap.md).

## Supersedes

This record supersedes the progression portion of ADR-0009 and the legacy action-based XP rules. ADR-0009’s identity and learning decisions remain in force. Historical legacy XP remains historical; it is not silently reclassified as Growth.
