"""
Q&A Belief Pipeline — Structured clarification with self-model updates.

Design philosophy: TamAGI is autonomous by default, agentic by choice.
Agenticity is earned. The entropy score on each UncertaintyNode is the
behavioral governor: high entropy → ask questions, build knowledge.
As the user answers, entropy falls. Once it drops below the threshold,
TamAGI stops asking and acts — that is earned agenticity.

Flow per conversation turn (interactive only, never autonomous):
  1. check_gate(conv_id, user_message)
       → keyword pre-filter (no LLM)
       → lazy subtype classification (1 LLM call, cached on node)
       → question generation (1 LLM call, subtype-aware)
       → saves PendingClarification to disk
       → returns PendingClarification (or None if no gate fires)
  2. If gate fired last turn: process_answer(pending, user_answer)
       → LLM-assisted belief extraction
       → creates BeliefNodes, updates entropy
       → logs to monologue
       → returns context hint for system prompt injection
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.core.llm import LLMClient
    from backend.core.monologue import MonologueLog
    from backend.core.self_model import SelfModel

from backend.core.llm import LLMMessage

logger = logging.getLogger("tamagi.qa_pipeline")

# Goal-pattern regex — mirrors agent.py's _detect_conversation_signals
_GOAL_RE = re.compile(
    r"\bi (?:want|need|would like|am trying|am working on|plan to|am going to)\b"
    r"|\bhelp me\b|\bcan you please\b|\bplease help\b",
    re.IGNORECASE,
)

_VALID_SUBTYPES = {"domain", "preference", "capability", "consequence"}

# Words too short or too common to be meaningful overlap tokens
_MIN_TOKEN_LEN = 4


@dataclass
class PendingClarification:
    """A clarification question asked in a given conversation, awaiting an answer."""
    id: str
    conversation_id: str
    uncertainty_id: str
    uncertainty_subtype: str
    question: str
    original_task: str   # user message that triggered the gate
    created_at: str      # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PendingClarification:
        return cls(**d)


class QAPipeline:
    """
    Manages the clarification gate: decides when to ask, generates questions,
    processes answers, and updates the self-model with extracted beliefs.
    """

    def __init__(
        self,
        llm: "LLMClient",
        self_model: "SelfModel",
        monologue_log: "MonologueLog",
        data_path: str = "data/qa_pending.json",
        entropy_threshold: float = 0.7,
        enabled: bool = True,
    ) -> None:
        self._llm = llm
        self._self_model = self_model
        self._monologue_log = monologue_log
        self._data_path = Path(data_path)
        self._entropy_threshold = entropy_threshold
        self._enabled = enabled
        # {conversation_id: PendingClarification}
        self._pending: dict[str, PendingClarification] = {}

    # ── Persistence ───────────────────────────────────────────

    def load(self) -> None:
        """Load pending clarifications from disk."""
        if not self._data_path.exists():
            return
        try:
            raw = json.loads(self._data_path.read_text())
            self._pending = {k: PendingClarification.from_dict(v) for k, v in raw.items()}
            if self._pending:
                logger.info("Q&A pipeline: loaded %d pending clarification(s)", len(self._pending))
        except Exception as exc:
            logger.warning("Q&A pipeline: failed to load pending state: %s", exc)

    def save(self) -> None:
        """Persist pending clarifications to disk."""
        try:
            self._data_path.parent.mkdir(parents=True, exist_ok=True)
            self._data_path.write_text(
                json.dumps({k: v.to_dict() for k, v in self._pending.items()}, indent=2)
            )
        except Exception as exc:
            logger.warning("Q&A pipeline: failed to save pending state: %s", exc)

    # ── Public API ────────────────────────────────────────────

    def get_pending(self, conv_id: str) -> PendingClarification | None:
        return self._pending.get(conv_id)

    def close_pending(self, conv_id: str) -> None:
        self._pending.pop(conv_id, None)
        self.save()

    async def check_gate(
        self,
        conv_id: str,
        user_message: str,
    ) -> PendingClarification | None:
        """
        Decide whether a clarification question should be asked before TamAGI
        proceeds with this user message. Returns a PendingClarification (already
        saved to disk) if the gate fires, or None to proceed normally.

        Earned agenticity: the gate fires when entropy > threshold. As Q&A reduces
        entropy, the gate stops firing and TamAGI acts without asking.
        """
        if not self._enabled:
            return None

        # Only check for task-like messages
        if len(user_message) < 20 or not _GOAL_RE.search(user_message):
            return None

        # Find high-entropy uncertainty nodes
        try:
            all_nodes = self._self_model.get_uncertainty_map()
        except Exception as exc:
            logger.debug("Q&A gate: failed to read uncertainty map: %s", exc)
            return None

        candidates = [n for n in all_nodes if n.entropy_score >= self._entropy_threshold]
        if not candidates:
            return None

        # Keyword pre-filter: find overlap between message tokens and node domain tokens
        msg_tokens = _tokens(user_message)
        if not msg_tokens:
            return None

        matched_node = None
        for node in sorted(candidates, key=lambda n: n.entropy_score, reverse=True):
            domain_tokens = _tokens(node.domain)
            if msg_tokens & domain_tokens:
                matched_node = node
                break

        if matched_node is None:
            logger.debug("Q&A gate: no relevant uncertainty for this message")
            return None

        # Classify subtype lazily (cached forever on node after first classification)
        if not matched_node.subtype:
            subtype = await self._classify_subtype(matched_node)
            matched_node.subtype = subtype
            try:
                self._self_model._apply_update_node(matched_node.id, {"subtype": subtype})
            except Exception as exc:
                logger.debug("Q&A gate: could not cache subtype on node: %s", exc)

        # Consequence nodes are handled by the approval system at execution time
        if matched_node.subtype == "consequence":
            logger.debug(
                "Q&A gate: consequence node %s — deferring to approval system",
                matched_node.id,
            )
            return None

        # Generate the clarifying question
        try:
            question = await self._generate_question(matched_node, user_message)
        except Exception as exc:
            logger.warning("Q&A gate: question generation failed: %s", exc)
            return None

        # Build and persist the pending state
        pending = PendingClarification(
            id=f"qa-{uuid.uuid4().hex[:8]}",
            conversation_id=conv_id,
            uncertainty_id=matched_node.id,
            uncertainty_subtype=matched_node.subtype,
            question=question,
            original_task=user_message,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._pending[conv_id] = pending
        self.save()

        logger.info(
            "Q&A gate: clarifying on node %s (subtype=%s entropy=%.2f) for conv=%s",
            matched_node.id, matched_node.subtype, matched_node.entropy_score, conv_id,
        )

        # Log to inner monologue
        try:
            self._monologue_log.append(
                type="clarification_asked",
                source="self",
                title=f"Clarification needed: {matched_node.domain[:60]}",
                content=question,
                metadata={
                    "qa_id": pending.id,
                    "uncertainty_id": matched_node.id,
                    "uncertainty_subtype": matched_node.subtype,
                    "entropy": matched_node.entropy_score,
                    "original_task": user_message[:200],
                },
            )
        except Exception as exc:
            logger.debug("Q&A gate: monologue append failed: %s", exc)

        return pending

    async def process_answer(
        self,
        pending: PendingClarification,
        answer: str,
    ) -> str:
        """
        Extract beliefs from the user's answer, update the self-model, and
        reduce entropy on the uncertainty node that prompted the question.

        Returns a context hint string for injection into the system prompt so the
        LLM knows to resume the original task.
        """
        # Extract structured beliefs from the answer
        beliefs: list[dict] = []
        try:
            beliefs = await self._extract_beliefs(pending, answer)
        except Exception as exc:
            logger.warning("Q&A pipeline: belief extraction failed: %s", exc)

        # Snapshot entropy before update
        entropy_before: float = 1.0
        try:
            node_dict = self._self_model.get_node(pending.uncertainty_id)
            if node_dict:
                entropy_before = float(node_dict.get("entropy_score", 1.0))
        except Exception:
            pass

        # Create BeliefNodes and wire them into the self-model
        created_belief_ids: list[str] = []
        for b in beliefs:
            try:
                bid = f"b-qa-{uuid.uuid4().hex[:8]}"
                self._self_model._apply_add_node("belief", {
                    "id": bid,
                    "description": b["description"],
                    "confidence": b["confidence"],
                    "evidence_count": 1,
                })
                self._self_model.auto_wire_node(bid)
                created_belief_ids.append(bid)
            except Exception as exc:
                logger.debug("Q&A pipeline: failed to add belief node: %s", exc)

        # Reduce entropy — each answered Q&A earns trust in this domain.
        # Formula: delta = avg(confidences) * 0.25, so even a 1.0-confidence
        # answer reduces entropy by at most 0.25 per exchange (avoids instant unlocking).
        entropy_after = entropy_before
        if beliefs:
            avg_conf = sum(b["confidence"] for b in beliefs) / len(beliefs)
            delta = round(avg_conf * 0.25, 3)
            entropy_after = max(0.05, round(entropy_before - delta, 3))
            try:
                self._self_model._apply_update_node(pending.uncertainty_id, {
                    "entropy_score": entropy_after,
                    "last_explored": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as exc:
                logger.debug("Q&A pipeline: entropy update failed: %s", exc)

        logger.info(
            "Q&A gate: processed answer for %s → %d belief(s) extracted, entropy %.3f→%.3f",
            pending.id, len(created_belief_ids), entropy_before, entropy_after,
        )

        # Audit trail in inner monologue
        try:
            self._monologue_log.append(
                type="belief_updated",
                source="user",
                title=f"Beliefs updated from Q&A: {pending.uncertainty_subtype}",
                content=(
                    f"User answered clarification about '{pending.uncertainty_id}'. "
                    f"Extracted {len(beliefs)} belief(s). "
                    f"Entropy: {entropy_before:.3f} → {entropy_after:.3f}."
                ),
                metadata={
                    "qa_id": pending.id,
                    "uncertainty_id": pending.uncertainty_id,
                    "uncertainty_subtype": pending.uncertainty_subtype,
                    "entropy_before": entropy_before,
                    "entropy_after": entropy_after,
                    "beliefs": beliefs,
                    "belief_node_ids": created_belief_ids,
                },
            )
        except Exception as exc:
            logger.debug("Q&A pipeline: monologue append failed: %s", exc)

        return f"Original task: {pending.original_task}"

    # ── Private LLM helpers ───────────────────────────────────

    async def _classify_subtype(self, node: "Any") -> str:
        """One-shot LLM call to classify an uncertainty node's subtype. Cached forever."""
        messages = [
            LLMMessage("system", (
                "You are a knowledge-gap classifier. Reply with exactly one word from the list below.\n\n"
                "Options:\n"
                "- domain: a topic or area where knowledge is lacking\n"
                "- preference: how the user prefers something to be done or styled\n"
                "- capability: a task or skill TamAGI may not be able to perform well\n"
                "- consequence: unknown downstream effects or risks of an action\n\n"
                "Reply with exactly one word, lowercase, no punctuation."
            )),
            LLMMessage("user", f'Gap: "{node.domain}"'),
        ]
        try:
            resp = await self._llm.chat(messages)
            word = resp.content.strip().lower().split()[0] if resp.content.strip() else "domain"
            return word if word in _VALID_SUBTYPES else "domain"
        except Exception as exc:
            logger.debug("Q&A pipeline: subtype classification failed: %s", exc)
            return "domain"

    async def _generate_question(self, node: "Any", user_message: str) -> str:
        """Generate a single, focused clarifying question shaped by the uncertainty subtype."""
        _instructions = {
            "domain": (
                "You have limited knowledge about a domain and need to ask one focused question "
                "to build understanding before proceeding. Ask about the topic itself — what it is, "
                "how it works, or the specific aspect you're unsure about."
            ),
            "preference": (
                "You don't know the user's preferences or intent for this task. Ask one question "
                "to clarify what they want — their style, constraints, priorities, or expectations."
            ),
            "capability": (
                "You're unsure whether you can do this well or what approach the user expects. "
                "Ask one question to clarify the scope, success criteria, or acceptable approach "
                "before attempting the task."
            ),
        }
        instruction = _instructions.get(node.subtype, _instructions["domain"])

        messages = [
            LLMMessage("system", (
                f"{instruction}\n\n"
                "Rules:\n"
                "- Ask only ONE question.\n"
                "- Be brief and natural — as if speaking directly to the user.\n"
                "- Do not explain why you are asking.\n"
                "- End with a question mark."
            )),
            LLMMessage("user", (
                f"User's request: \"{user_message[:200]}\"\n"
                f"Knowledge gap domain: \"{node.domain}\""
            )),
        ]
        try:
            resp = await self._llm.chat(messages)
            question = resp.content.strip()
            if question and not question.endswith("?"):
                question = question.rstrip(".!") + "?"
            return question or f"Could you tell me more about {node.domain}?"
        except Exception as exc:
            logger.warning("Q&A pipeline: question generation failed: %s", exc)
            return f"Before I proceed, could you help me understand more about {node.domain}?"

    async def _extract_beliefs(
        self,
        pending: PendingClarification,
        answer: str,
    ) -> list[dict]:
        """
        Extract up to 3 structured beliefs from the user's answer.
        Returns list of {"description": str, "confidence": float}.
        """
        messages = [
            LLMMessage("system", (
                "You are a belief extraction engine. Extract concrete beliefs from a user's answer.\n\n"
                "Rules:\n"
                "- Only extract things the user explicitly stated or clearly implied.\n"
                "- Confidence 0.8–1.0 for explicit statements; 0.5–0.7 for implied.\n"
                "- Maximum 3 beliefs. Fewer is fine if the answer is simple.\n"
                "- Beliefs should be short, factual statements (not questions or instructions).\n\n"
                "Reply using this exact format, one per line:\n"
                "BELIEF: [statement] | CONFIDENCE: [0.0-1.0]"
            )),
            LLMMessage("user", (
                f"Question asked: \"{pending.question}\"\n"
                f"User's answer: \"{answer[:400]}\"\n"
                f"Domain context: {pending.uncertainty_subtype} — {pending.uncertainty_id}"
            )),
        ]
        try:
            resp = await self._llm.chat(messages)
            return _parse_beliefs(resp.content)
        except Exception as exc:
            logger.warning("Q&A pipeline: belief extraction LLM call failed: %s", exc)
            return []


# ── Module-level helpers ──────────────────────────────────────

def _tokens(text: str) -> set[str]:
    """Lowercase word tokens longer than _MIN_TOKEN_LEN (cheap stop-word filter)."""
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= _MIN_TOKEN_LEN}


def _parse_beliefs(text: str) -> list[dict]:
    """Parse LLM output in 'BELIEF: X | CONFIDENCE: Y' format."""
    results = []
    for m in re.finditer(r"BELIEF:\s*(.+?)\s*\|\s*CONFIDENCE:\s*([\d.]+)", text, re.IGNORECASE):
        desc = m.group(1).strip()
        try:
            conf = min(1.0, max(0.0, float(m.group(2))))
        except ValueError:
            conf = 0.5
        if desc:
            results.append({"description": desc, "confidence": conf})
        if len(results) >= 3:
            break
    return results
