from __future__ import annotations

import json
from pathlib import Path

from src.app.domain.support.models import ConversationTurn
from src.app.infrastructure.storage.conversation_store import ConversationStore


class FileConversationStore(ConversationStore):
    """Persist conversation history as JSON files on disk."""

    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str) -> list[ConversationTurn]:
        """Load a conversation from disk, returning an empty history if missing."""
        file_path = self._resolve_path(session_id)
        if not file_path.exists():
            return []

        try:
            with file_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError):
            return []

        return [
            ConversationTurn(role=message["role"], content=message["content"])
            for message in payload
            if isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and isinstance(message.get("content"), str)
        ]

    def save(self, session_id: str, messages: list[ConversationTurn]) -> None:
        """Persist a conversation to disk."""
        file_path = self._resolve_path(session_id)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                file,
                indent=2,
                ensure_ascii=False,
            )

    def _resolve_path(self, session_id: str) -> Path:
        """Return the JSON file path for a session."""
        return self._storage_dir / f"{session_id}.json"
