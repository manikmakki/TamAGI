# ADR-0017: Bounded user turns and quiet continuations

Status: Accepted by operator choice on 2026-11-04.

## Context

The interactive chat path retained a separate five-round loop after its runtime allowance was raised. Eight turns in the retained research conversation stopped at five calls, and intermediate promises could become final answers marked completed. Eleven replies made no tool call. Increasing a numerical allowance alone does not make a promise become a running job.

AgentOxide provides a useful mechanism: short bounded turns and a small number of explicitly requested quiet continuations. Its step is one model call, not one individual tool operation. TamAGI adopts that meaning with the operator's requested limit of twenty.

## Decision

An actual user prompt permits twenty model calls in its first turn. The primary resident may call `request_quiet_turn` to request one continuation, finishing its current turn successfully before the continuation starts. Requests within a turn coalesce. At most five quiet continuations may start for that prompt, each with twenty calls: at most 120 across the full chain. Retries and companion calls consume these same allowances. Companions cannot request additional quiet turns.

Requested quiet work runs before the final conversational reply. Intermediate drafts remain progress, not delivered final answers. Ordinary replies need no continuation; quiet completion, rest, waiting, refusal, and blockage remain valid. No phrase detector or model judge forces action from promising language. Exhaustion and errors clear queued work and produce explicit procedural outcomes rather than recycled commentary or `...`.

Preserve the tool protocol and working context across continuations without inventing new user messages. Store each logical turn's ordinal, calls, status, and receipt references. Later actual turns can retrieve recent receipt context from that conversation. Queued work does not automatically resume after interruption or restart; canonical effects and receipt references remain recoverable.

Input received while quiet work is running interrupts the quiet child task and clears further continuation. Input already waiting prevents another quiet turn from starting. The WebSocket reader keeps receiving during inference. Request identifiers prevent stale status events from replacing a newer interaction. A quiet indicator permits the user to send a new message; silent endings appear as procedural status, not invented speech.

The World Thread retains its independent 128-call activation ceiling, no daily background cap, voluntary rest, and normal opportunity schedule. The measured 131072-token context policy remains in force. User-turn controls never replenish background allocations or erase charges.

## Rejected alternatives

- Keeping independent legacy and runtime limits: leaves invisible contradictory cutoffs.
- Automatically granting five continuations to every prompt: forces extra activity.
- Treating intermediate narration as completion: obscures unfinished work.
- Counting every tool operation as an AgentOxide step: changes the referenced mechanism; batches and delegated provider calls need explicit accounting.
- Cancelling the whole WebSocket handler to interrupt quiet inference: can close the conversation connection.
- Treating a textual promise as a scheduled job: confuses narration with executable state.

## Consequences

An opted-in chain may use up to 120 provider calls before replying. There is no guaranteed improvement in model follow-through: the model must choose tools and continuations. Normal conversation can end after one call. Root execution receipts preserve effects even if a later stage fails; interruption can leave an unknown outcome that requires inspection before retry.

Controls and counters are trusted dispatch state. Host root can still alter them under ADR-0002's cooperative authority model. A completed conversation response does not verify completion of a broader task; operational claims continue to require evidence.

## Acceptance evidence

Automated cases exercise more than five calls, a complete 120-call chain, coalesced requests, sixth-turn rejection, exhaustion with misleading commentary, queue discard after failure, early rest, pending input, interruption preserving an uncertain effect, companion eligibility, and receipt recovery. A disposable full application check crosses the five-round boundary, withholds the final reply until quiet work finishes, and tests new input during an active quiet tool through the WebSocket. Sustained behavioral findings remain part of the pilot.

## Supersedes

Supersedes ADR-0016's 64-call user-turn default and the platform chat path's legacy five-round ceiling and narrative fallback. ADR-0016's autonomous allowance and context-window decisions remain in force. Earlier records are retained unchanged.
