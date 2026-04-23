"""
Three-layer conversation memory for the leasing chatbot.

Layer 1 — raw_history  : last N user/assistant exchanges (deque)
Layer 2 — summary      : LLM-generated summary, refreshed every few turns
Layer 3 — state        : structured dict (topic, intent, entities, preferences)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

_MAX_RAW_EXCHANGES = 5  # each exchange = 1 user + 1 assistant message
_SUMMARY_REFRESH_EVERY = 3  # exchanges between summary updates


@dataclass
class ConversationMemory:
    raw_history: deque = field(default_factory=lambda: deque(maxlen=_MAX_RAW_EXCHANGES * 2))
    summary: str = ""
    state: dict = field(
        default_factory=lambda: {
            "current_topic": None,
            "current_intent": None,
            "referenced_entities": {},
            "user_preference": None,
        }
    )
    _exchange_count: int = field(default=0, repr=False)

    def add_exchange(self, user_msg: str, assistant_msg: str) -> None:
        self.raw_history.append({"role": "user", "content": user_msg})
        self.raw_history.append({"role": "assistant", "content": assistant_msg})
        self._exchange_count += 1

    def get_recent_history(self) -> list[dict]:
        return list(self.raw_history)

    def format_history_for_prompt(self) -> str:
        if not self.raw_history:
            return "No prior conversation."
        lines = []
        for msg in self.raw_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def needs_summary_update(self) -> bool:
        return self._exchange_count > 0 and self._exchange_count % _SUMMARY_REFRESH_EVERY == 0

    def update_summary(self, new_summary: str) -> None:
        self.summary = new_summary

    def update_state(self, intent: str | None = None, topic: str | None = None) -> None:
        if intent:
            self.state["current_intent"] = intent
        if topic:
            self.state["current_topic"] = topic
