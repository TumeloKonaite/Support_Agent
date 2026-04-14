from __future__ import annotations

from abc import ABC, abstractmethod

from src.app.domain.support.models import ConversationTurn


class ConversationStore(ABC):
    """Contract for loading and saving conversation history."""

    @abstractmethod
    def load(self, session_id: str) -> list[ConversationTurn]:
        """Load the conversation history for a session."""

    @abstractmethod
    def save(self, session_id: str, messages: list[ConversationTurn]) -> None:
        """Persist the conversation history for a session."""
