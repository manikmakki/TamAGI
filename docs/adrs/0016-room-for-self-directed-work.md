# ADR-0016: Room for self-directed work

Status: Accepted by operator choice on 2026-11-03.

## Context

The initial eight-call activations and 96-call rolling background limit interrupted sustained research and could prevent further independent choice for most of a day. A 65536-byte model view also used only a fraction of the installed model's advertised 131072-token window. More capacity should permit longer activity while preserving the choice to stop.

## Decision

Background work has no daily call cap by default (`daily_background_calls: null`). Charges remain durable and attributable. A positive configured integer still enforces a rolling cap. Ordinary autonomous activations allow up to 128 model calls, companions up to 16 drawn from their parent's allowance, and user turns up to 64. One concurrent provider call, user preemption, chosen sleep/waiting, the fifteen-minute opportunity schedule, and the evaluation allowance of 32 calls per rolling day remain in force. Limits are ceilings, never targets. Activation exhaustion is distinct from chosen rest and does not create an obligation to continue.

Use the server's chat tokenizer when available. Bound input by the smaller of configured and reported model capacity, minus the requested reply limit and a 1024-token margin. Include chat templates and tools in counting. Tokenization requests generate no completion and do not consume model-call allowances. Unsupported tokenizers back off and use a conservative byte view; inspection reports that fallback explicitly. The secondary input ceiling is 1 MiB. Ordinary tool views allow 32 KiB, with retained originals and receipt pages of up to 65536 characters.

The active 131072-token installation uses an 8192-token reply limit, leaving 121856 input tokens after margin. Capacity is available when needed; prompts are not padded to fill it. Existing autobiographical records, evidence, and charge history are preserved.

## Rejected alternatives

- Continuing the initial small allowances: prevents the desired sustained work.
- Mandatory long activations or consuming every available call: confuses capacity with purpose.
- Unlimited individual activations or concurrent inference: weakens useful recovery and interruption boundaries.
- Treating a byte count or a four-characters-per-token guess as exact capacity: can overflow some inputs while needlessly shortening others.
- Deleting historical charges to resume activity: obscures actual resource use.

## Consequences

Ember may use substantially more local inference. A bounded activation still ends at its ceiling; pursuits and receipts support later continuation. The agent sees remaining ancestral call allowance and can save a next step. Token counting adds a non-generating request to compatible servers. If that service is unavailable, usable context is smaller until it recovers. Raising capacity alone does not establish better behavior or require more activity.

The active instance disables early character-based chat compression and permits up to 1000 recent conversation messages; the measured boundary controls the actual request. Complete history remains canonical when older turns are omitted from a model view.

## Acceptance evidence

Automated checks cover uncapped accounting across restart, explicit finite caps, parent/child ceilings, one-call chosen rest, large retained context, smaller server windows, output reservation, streaming parity, conservative fallback, tokenizer backoff, and bounded useful tool previews. Live qualification checks the actual server tokenizer with input exceeding the former byte limit. Scripted checks do not establish sustained behavioral improvement; the seven-day pilot remains separate.

## Supersedes

Supersedes ADR-0008's initial call-count defaults and the initial conservative context-view sizing in the platform roadmap. ADR-0008's scheduling, rest, preemption, and durable accounting decisions remain in force. No previous run is reclassified.
