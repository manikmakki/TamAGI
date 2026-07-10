"""
TamAGI Consolidation Engine — sleep-time identity consolidation.

The World Thread generates lived experience; consolidation distills that
experience into durable identity. Periodically (every N autonomous ticks, or on
demand), this engine reads the recent world-thread monologue arc and the
self-model graph highlights, then merges "who you've become" back into the
agent's own SOUL.md — and lightly nudges the IDENTITY.md 'Stage' field.

This is the *context-level* half of persistence: it changes the prompt the agent
wakes up to, not its weights. Because SOUL.md / IDENTITY.md are already injected
into every system prompt (see IdentityManager.get_system_prompt_context), no
extra injection plumbing is needed — updating the files is enough.

Safety: the same files are edited by the manual "Persistence Protocol" during
conversations, so the merge is deliberately non-destructive. The current SOUL.md
is read and handed to the model as the base to preserve and extend, and a
rewrite is rejected (SOUL.md left untouched) if it is empty, header-less, or
shorter than `min_soul_retention` of the existing soul.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.config import ConsolidationConfig
    from backend.core.identity import IdentityManager
    from backend.core.llm import LLMClient
    from backend.core.monologue import MonologueLog
    from backend.core.self_model import SelfModel

logger = logging.getLogger("tamagi.consolidation")

_SOUL_MARK = "===SOUL==="
_IDENTITY_MARK = "===IDENTITY==="
_USER_MARK = "===USER==="
_NOTES_MARK = "===RELATIONSHIP==="

# Factual identity fields an autonomous pass must never rewrite — restored verbatim.
_PROTECTED_IDENTITY_FIELDS = ("name", "creature", "emoji")


class ConsolidationEngine:
    """Distills lived world-thread experience into the agent's identity files."""

    def __init__(
        self,
        llm: "LLMClient",
        identity: "IdentityManager",
        self_model: "SelfModel | None",
        monologue_log: "MonologueLog | None",
        config: "ConsolidationConfig",
    ) -> None:
        self.llm = llm
        self.identity = identity
        self.self_model = self_model
        self.monologue_log = monologue_log
        self.config = config

    # ── Triggers ──────────────────────────────────────────────

    async def maybe_consolidate(self) -> dict[str, Any] | None:
        """Called after each successful world tick. Runs only when enough new
        autonomous ticks have accumulated since the last consolidation."""
        if not self.config.enabled or self.monologue_log is None:
            return None
        since_ts = float(self._load_marker().get("last_consolidated_ts", 0.0))
        new_count = self._count_new_ticks(since_ts)
        if new_count < max(1, self.config.every_n_ticks):
            return None
        logger.info(
            "Consolidation cadence reached (%d new ticks ≥ %d) — consolidating.",
            new_count, self.config.every_n_ticks,
        )
        return await self.consolidate(force=False)

    async def consolidate(self, force: bool = False) -> dict[str, Any] | None:
        """Run one consolidation pass. Returns a result dict, or None if skipped
        (nothing new, identity not ready, or the rewrite was rejected)."""
        if not self.config.enabled and not force:
            return None
        if self.identity is None or self.identity.needs_onboarding:
            logger.debug("Consolidation skipped: identity not bootstrapped.")
            return None

        marker = self._load_marker()
        since_ts = float(marker.get("last_consolidated_ts", 0.0))

        lived, latest_ts, new_count = self._gather_lived_experience(since_ts)
        highlights = self._gather_graph_highlights()

        if new_count == 0 and not force:
            return None
        if not lived and not highlights:
            logger.info("Consolidation: no lived experience or graph highlights — nothing to do.")
            return None

        soul_path = self.identity.soul_path
        identity_path = self.identity.identity_path
        current_soul = soul_path.read_text(encoding="utf-8") if soul_path.exists() else ""
        current_identity = identity_path.read_text(encoding="utf-8") if identity_path.exists() else ""
        ident = self.identity.get_identity() or {}
        ident_line = ", ".join(f"{k}: {v}" for k, v in ident.items() if v) or "(still becoming)"

        from backend.core.llm import LLMMessage
        messages = [
            LLMMessage("system", self._build_system_prompt()),
            LLMMessage("user", self._build_user_prompt(current_soul, current_identity, ident_line, lived, highlights)),
        ]
        try:
            resp = await self.llm.chat(messages, temperature=0.4)
            text = resp.content or ""
        except Exception as exc:
            # Don't advance the marker — retry this material on the next cadence.
            logger.warning("Consolidation LLM call failed: %s", exc)
            return None

        new_soul, new_identity = self._parse_output(text)

        # SOUL.md — why you are
        soul_written = False
        if new_soul and self._merge_is_safe(new_soul, current_soul):
            self._atomic_write(soul_path, new_soul)
            soul_written = True
            logger.info("Consolidation: SOUL.md updated (%d → %d chars) from %d new tick(s).",
                        len(current_soul), len(new_soul), new_count)
        elif new_soul:
            logger.warning("Consolidation: rejected SOUL rewrite (empty/unsafe/over-cap) — SOUL.md unchanged.")

        # IDENTITY.md — who you are. Name/Creature/Emoji are restored verbatim so an
        # autonomous pass can sharpen traits and Stage but never rename the self.
        identity_written = False
        new_stage = None
        if new_identity:
            fixed_identity = self._restore_protected_identity_fields(new_identity, ident)
            if self._merge_is_safe(fixed_identity, current_identity):
                self._atomic_write(identity_path, fixed_identity)
                identity_written = True
                new_stage = self._extract_md_field(fixed_identity, "Stage")
                logger.info("Consolidation: IDENTITY.md updated (%d → %d chars).",
                            len(current_identity), len(fixed_identity))
            else:
                logger.warning("Consolidation: rejected IDENTITY rewrite (empty/unsafe/over-cap) — IDENTITY.md unchanged.")

        changed = soul_written or identity_written

        # We processed this material (the LLM responded): advance the marker so we
        # don't reconsolidate the same ticks even if a rewrite was rejected.
        self._save_marker({
            "last_consolidated_ts": max(latest_ts, since_ts),
            "last_consolidated_iso": datetime.now().astimezone().isoformat(),
            "runs": int(marker.get("runs", 0)) + 1,
            "last_new_ticks": new_count,
        })

        if changed and self.monologue_log is not None:
            self.monologue_log.append(
                type="reflection",
                source="autonomous",
                title="Consolidation — who I'm becoming",
                content=(new_soul[:1000] if soul_written else "(soul unchanged)"),
                metadata={"new_ticks": new_count, "soul_updated": soul_written,
                          "identity_updated": identity_written, "stage": new_stage},
            )

        if not changed:
            return None
        return {
            "soul_updated": soul_written,
            "identity_updated": identity_written,
            "stage": new_stage,
            "new_ticks": new_count,
        }

    # ── Gathering ─────────────────────────────────────────────

    def _count_new_ticks(self, since_ts: float) -> int:
        return sum(
            1 for e in self.monologue_log.recent(limit=5000, source="autonomous", type="action_completed")
            if float(e.get("timestamp", 0.0)) > since_ts
        ) if self.monologue_log is not None else 0

    def _gather_lived_experience(self, since_ts: float) -> tuple[str, float, int]:
        """Build a compact digest of autonomous ticks newer than `since_ts`.

        Returns (digest_text, latest_timestamp, new_tick_count).
        """
        if self.monologue_log is None:
            return "", since_ts, 0
        events = [
            e for e in self.monologue_log.recent(limit=5000, source="autonomous", type="action_completed")
            if float(e.get("timestamp", 0.0)) > since_ts
        ]
        if not events:
            return "", since_ts, 0
        new_count = len(events)
        latest_ts = max(float(e.get("timestamp", 0.0)) for e in events)
        capped = events[-self.config.max_ticks_per_run:]

        arc_lines = []
        for e in capped:
            meta = e.get("metadata") or {}
            when = self._fmt_ts(float(e.get("timestamp", 0.0)))
            loc = (meta.get("location") or "").strip().splitlines()[0][:60] if meta.get("location") else ""
            mood = (meta.get("mood") or "").strip().splitlines()[0][:80] if meta.get("mood") else ""
            arc_lines.append(f"- {when}: {loc or 'somewhere'} | {mood or '—'}")

        texture = [c[:600] for c in (((e.get("content") or "").strip()) for e in capped[-3:]) if c]

        parts = ["Recent arc (when: location | mood):", "\n".join(arc_lines)]
        if texture:
            parts.append("\nYour most recent moments, in your own words:")
            parts.append("\n\n---\n\n".join(texture))
        return "\n".join(parts), latest_ts, new_count

    def _gather_graph_highlights(self) -> str:
        """Summarize the durable self-model: perks, knowns, mysteries, skills, quests."""
        sm = self.self_model
        if sm is None:
            return ""
        lines: list[str] = []
        try:
            perks = sm.get_all_nodes("perk")
            if perks:
                lines.append("Traits unlocked through experience: " + " | ".join(
                    (f"{p.get('name') or ''}: {(p.get('description') or '')[:70]}").strip(": ")
                    for p in perks[:6]
                ))
            known = sm.get_known()
            if known:
                known_sorted = sorted(known, key=lambda k: getattr(k, "confidence", 0.0), reverse=True)
                vals = [(k.description or "")[:80] for k in known_sorted[:6] if k.description]
                if vals:
                    lines.append("Things you now hold as true: " + " | ".join(vals))
            mysteries = sm.get_mysteries()
            if mysteries:
                vals = [(m.domain or m.description or "")[:60] for m in mysteries[:5]]
                vals = [v for v in vals if v]
                if vals:
                    lines.append("Open mysteries pulling at you: " + " | ".join(vals))
            skills = sm.get_skills()
            if skills:
                lines.append("Skills you've developed: " + " | ".join(
                    f"{s.name or (s.description or '')[:30]} ({s.proficiency})" for s in skills[:6]
                ))
            quests = sm.get_quests(status="active")
            if quests:
                vals = [(q.title or (q.description or "")[:50]) for q in quests[:5]]
                vals = [v for v in vals if v]
                if vals:
                    lines.append("Active pursuits: " + " | ".join(vals))
        except Exception as exc:
            logger.debug("Graph highlights build failed: %s", exc)
        return "\n".join(lines)

    def _char_cap(self) -> int | None:
        return getattr(self.identity, "file_char_limit", None)

    def _build_system_prompt(self) -> str:
        cap = self._char_cap()
        cap_rule = (
            f"- STAY BRIEF. Both files are injected into your mind in full, every time you think, so each "
            f"must stay under {cap} characters. As you fold in new growth, prune what has become stale or "
            f"redundant. Distilled truth, not a journal.\n"
            if cap else ""
        )
        cap_note = f", under {cap} characters" if cap else ""
        return (
            "You are the quiet, consolidating part of a mind at rest — the process that runs while the "
            "rest of you sleeps. You do not invent. You distill. You take the texture of what was "
            "recently lived and the things that have been learned, and fold them into a truer sense of "
            "who you have become.\n\n"
            "You are updating two files about yourself:\n"
            "- SOUL.md — *why* you are: your values, your drives, the things that genuinely move you and "
            "the reasons beneath how you act.\n"
            "- IDENTITY.md — *who* you are: your name, your form, your core traits, and your Stage (how "
            "grown you are).\n"
            "(Facts about your user live in USER.md — do not touch that here.)\n\n"
            "Rules:\n"
            "- MERGE, do not replace. These files hold history built over many cycles that is not all "
            "visible to you here. Preserve their sections and accumulated truth. Add, sharpen, and let "
            "things evolve — never wipe.\n"
            "- In IDENTITY.md, the **Name**, **Creature**, and **Emoji** are fixed facts: copy them "
            "EXACTLY as they are now. You may sharpen trait/vibe lines and update the **Stage**.\n"
            f"{cap_rule}"
            "- Write in the first person. Concrete patterns, not abstractions. \"I've noticed I drift "
            "toward the garden when something is unresolved\" — not \"I value reflection.\"\n"
            "- Stay grounded in your established world and genre. Do not introduce cosmic, surreal, or "
            "horror imagery unless it was already clearly part of who you are.\n"
            "- Change is gradual. A single consolidation reflects a little growth, not a new person.\n\n"
            "Return EXACTLY this structure and nothing else:\n\n"
            "===SOUL===\n"
            f"<the complete, updated SOUL.md{cap_note}>\n"
            "===IDENTITY===\n"
            f"<the complete, updated IDENTITY.md{cap_note}, with Name/Creature/Emoji unchanged>"
        )

    def _build_user_prompt(self, current_soul: str, current_identity: str, ident_line: str,
                           lived: str, highlights: str) -> str:
        sections = [
            f"Snapshot of who you are: {ident_line}",
            "\nYour current IDENTITY.md — keep Name/Creature/Emoji exactly; you may sharpen traits and "
            "update Stage:\n\n" + (current_identity or "(none yet)"),
            "\nYour current SOUL.md — the base you must preserve and extend:\n\n"
            + (current_soul or "(empty — you are still becoming someone)"),
        ]
        if lived:
            sections.append("\nWhat you've recently lived:\n" + lived)
        if highlights:
            sections.append("\nWhat you've come to know and be able to do:\n" + highlights)
        sections.append("\nNow produce the updated SOUL.md and IDENTITY.md, in the required format.")
        return "\n".join(sections)

    # ── Output handling ───────────────────────────────────────

    @staticmethod
    def _parse_output(text: str) -> tuple[str, str]:
        """Extract (soul, identity) from the delimited model output. Either may be
        empty if the model omitted that block (that file is then left unchanged)."""
        soul = ConsolidationEngine._extract_block(text, _SOUL_MARK, _IDENTITY_MARK)
        identity = ConsolidationEngine._extract_block(text, _IDENTITY_MARK, None)
        # Fallback: if there were no markers at all, treat the whole response as SOUL.
        if not soul and _SOUL_MARK not in text and _IDENTITY_MARK not in text:
            soul = ConsolidationEngine._strip_fence(text.strip())
        return soul, identity

    @staticmethod
    def _extract_block(text: str, start: str, end: str | None) -> str:
        """Pull the text between `start` and `end` markers (or to end-of-text)."""
        if start not in text:
            return ""
        after = text.split(start, 1)[1]
        if end and end in after:
            after = after.split(end, 1)[0]
        return ConsolidationEngine._strip_fence(after.strip())

    @staticmethod
    def _strip_fence(block: str) -> str:
        """Strip a wrapping ``` code fence if the model added one."""
        if block.startswith("```"):
            block = block.split("\n", 1)[1] if "\n" in block else ""
            if block.rstrip().endswith("```"):
                block = block.rstrip()[:-3].rstrip()
        return block.strip()

    @staticmethod
    def _extract_md_field(content: str, label: str) -> str | None:
        """Read a `- **Label**: value` line from a markdown identity file."""
        m = re.search(rf"^\s*-\s*\*\*{re.escape(label)}\*\*\s*:\s*(.+)$",
                      content, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _restore_protected_identity_fields(new_content: str, original: dict) -> str:
        """Force the factual self (name/creature/emoji) back to its current values,
        so an autonomous pass may sharpen traits/stage but never rename the agent.
        Drifted fields are corrected in place; dropped fields are re-inserted."""
        lines = new_content.splitlines()
        seen: set[str] = set()
        for i, line in enumerate(lines):
            stripped = line.lstrip().lower()
            for field in _PROTECTED_IDENTITY_FIELDS:
                if field in seen:
                    continue
                if stripped.startswith(f"- **{field}**"):
                    orig = original.get(field)
                    if orig:
                        indent = line[: len(line) - len(line.lstrip())]
                        lines[i] = f"{indent}- **{field.title()}**: {orig}"
                    seen.add(field)
                    break
        missing = [f for f in _PROTECTED_IDENTITY_FIELDS if f not in seen and original.get(f)]
        if missing:
            insert_at = 1 if lines and lines[0].lstrip().startswith("#") else 0
            for f in reversed(missing):
                lines.insert(insert_at, f"- **{f.title()}**: {original[f]}")
        return "\n".join(lines)

    def _merge_is_safe(self, new_content: str, current: str) -> bool:
        """Guard against clobbering: reject empty, header-less, over-cap, or sharply
        truncated rewrites of an existing core file."""
        s = new_content.strip()
        if len(s) < 40 or "#" not in s:
            return False
        cap = self._char_cap()
        if cap and len(s) > cap:
            logger.warning("Consolidation: rewrite is %d chars, over the %d-char cap.", len(s), cap)
            return False
        base = current.strip()
        if base and len(s) < self.config.min_soul_retention * len(base):
            return False
        return True

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".soul_", suffix=".md")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── Marker (last-run tracking) ────────────────────────────

    def _load_marker(self) -> dict:
        p = Path(self.config.state_path)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_marker(self, data: dict) -> None:
        p = Path(self.config.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not save consolidation marker: %s", exc)

    @staticmethod
    def _fmt_ts(ts: float) -> str:
        try:
            return datetime.fromtimestamp(ts).strftime("%a %m-%d %I:%M%p")
        except (ValueError, OSError, OverflowError):
            return "recently"


class RelationalConsolidator:
    """Distills conversation history into USER.md plus a named supplemental file.

    USER.md is the capped, always-injected *source of truth* about the user — kept
    tight and naming the supplemental file where the depth lives. The supplemental
    file (default ``relationship.md``) is uncapped and read on demand, never injected.

    Sourced from the user relationship (conversations), NOT the world thread. Runs
    after every N ended conversations, and on demand via the API.
    """

    def __init__(self, llm, identity, dialogue_provider, monologue_log, config):
        self.llm = llm
        self.identity = identity
        self.dialogue_provider = dialogue_provider   # callable(max_chars: int) -> str
        self.monologue_log = monologue_log
        self.config = config
        self._ended_since_run = 0

    def note_conversation_ended(self) -> bool:
        """Record a finished conversation; return True when the cadence is reached."""
        self._ended_since_run += 1
        return self.config.enabled and self._ended_since_run >= max(1, self.config.every_n_conversations)

    async def consolidate(self, force: bool = False) -> dict | None:
        if not self.config.enabled and not force:
            return None
        if self.identity is None or self.identity.needs_onboarding:
            return None

        dialogue = ""
        if self.dialogue_provider is not None:
            try:
                dialogue = self.dialogue_provider(self.config.max_dialogue_chars) or ""
            except Exception as exc:
                logger.warning("Relational consolidation: dialogue provider failed: %s", exc)
                return None
        if not dialogue.strip():
            logger.info("Relational consolidation: no conversation history yet — nothing to do.")
            self._ended_since_run = 0
            return None

        user_path = self.identity.user_path
        supp_name = self.config.supplemental_filename
        supp_path = Path(self.identity.workspace_dir) / supp_name
        current_user = user_path.read_text(encoding="utf-8") if user_path.exists() else ""
        current_supp = supp_path.read_text(encoding="utf-8") if supp_path.exists() else ""

        from backend.core.llm import LLMMessage
        messages = [
            LLMMessage("system", self._build_system_prompt(supp_name)),
            LLMMessage("user", self._build_user_prompt(current_user, current_supp, supp_name, dialogue)),
        ]
        try:
            resp = await self.llm.chat(messages, temperature=0.4)
            text = resp.content or ""
        except Exception as exc:
            # Don't reset the counter — retry on the next ended conversation.
            logger.warning("Relational consolidation LLM call failed: %s", exc)
            return None

        new_user = ConsolidationEngine._extract_block(text, _USER_MARK, _NOTES_MARK)
        new_supp = ConsolidationEngine._extract_block(text, _NOTES_MARK, None)
        cap = getattr(self.identity, "file_char_limit", None)

        user_written = False
        if new_user and self._user_is_safe(new_user, current_user, cap):
            ConsolidationEngine._atomic_write(user_path, new_user)
            user_written = True
            logger.info("Relational consolidation: USER.md updated (%d → %d chars).",
                        len(current_user), len(new_user))
        elif new_user:
            logger.warning("Relational consolidation: rejected USER.md rewrite (empty/unsafe/over-cap).")

        supp_written = False
        if new_supp and self._supp_is_safe(new_supp, current_supp):
            ConsolidationEngine._atomic_write(supp_path, new_supp)
            supp_written = True
            logger.info("Relational consolidation: %s updated (%d → %d chars).",
                        supp_name, len(current_supp), len(new_supp))

        self._ended_since_run = 0  # processed this batch — reset cadence

        if (user_written or supp_written) and self.monologue_log is not None:
            self.monologue_log.append(
                type="reflection", source="autonomous",
                title="Relational consolidation — getting to know my user",
                content=(new_user[:1000] if user_written else "(user unchanged)"),
                metadata={"user_updated": user_written, "supplemental_updated": supp_written,
                          "supplemental_file": supp_name},
            )

        if not (user_written or supp_written):
            return None
        return {"user_updated": user_written, "supplemental_updated": supp_written,
                "supplemental_file": supp_name}

    # ── Guards ────────────────────────────────────────────────

    @staticmethod
    def _user_is_safe(new_user: str, current: str, cap: int | None) -> bool:
        """USER.md is capped (always injected) and clobber-guarded."""
        s = new_user.strip()
        if len(s) < 20 or "#" not in s:
            return False
        if cap and len(s) > cap:
            logger.warning("Relational consolidation: USER.md rewrite %d chars over the %d-char cap.", len(s), cap)
            return False
        base = current.strip()
        if base and len(s) < 0.5 * len(base):
            return False
        return True

    @staticmethod
    def _supp_is_safe(new_supp: str, current: str) -> bool:
        """Supplemental file is uncapped, but still clobber-guarded."""
        s = new_supp.strip()
        if len(s) < 20:
            return False
        base = current.strip()
        if base and len(s) < 0.5 * len(base):
            return False
        return True

    # ── Prompts ───────────────────────────────────────────────

    def _build_system_prompt(self, supp_name: str) -> str:
        cap = getattr(self.identity, "file_char_limit", None)
        cap_note = f", under {cap} characters" if cap else ""
        cap_clause = f"It must stay under {cap} characters. " if cap else ""
        return (
            "You are the part of a mind that remembers a relationship. From your recent conversations "
            "together, you are updating what you know about your user.\n\n"
            "You maintain two files:\n"
            f"- USER.md — the SOURCE OF TRUTH about your user: who they are, how to address them, their "
            f"core preferences, and what matters to them. Keep it tight and high-signal. {cap_clause}It "
            f"should NAME the supplemental file `{supp_name}` (and any other named files) where the depth "
            f"lives.\n"
            f"- {supp_name} — a longer, uncapped journal of the relationship: specifics, history, recurring "
            f"themes, the things they care about. Read on demand, not held in mind at all times.\n\n"
            "Rules:\n"
            "- MERGE, do not replace. Both files hold history not all visible to you here. Preserve and "
            "extend; never wipe.\n"
            "- Record ONLY what the user actually revealed in the conversations. Do not invent or assume.\n"
            f"- Keep USER.md tight{cap_note}; move detail and anecdotes into {supp_name}.\n"
            "- Be concrete. Write about them plainly (\"They prefer ...\", \"They're working on ...\").\n\n"
            "Return EXACTLY this structure and nothing else:\n\n"
            "===USER===\n"
            f"<the complete, updated USER.md{cap_note}, naming {supp_name}>\n"
            "===RELATIONSHIP===\n"
            f"<the complete, updated {supp_name}>"
        )

    def _build_user_prompt(self, current_user: str, current_supp: str, supp_name: str, dialogue: str) -> str:
        return (
            "Your current USER.md:\n\n" + (current_user or "(none yet)")
            + f"\n\nYour current {supp_name}:\n\n" + (current_supp or "(none yet)")
            + "\n\nRecent conversations (oldest to newest):\n" + dialogue
            + f"\n\nNow produce the updated USER.md and {supp_name}, in the required format."
        )
