import json
import tempfile
import unittest
from pathlib import Path

from src.app.domain.support.models import ConversationTurn
from src.app.infrastructure.storage.file_conversation_store import (
    FileConversationStore,
)


class FileConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "conversations"
        self.store = FileConversationStore(self.storage_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_returns_empty_list_when_file_missing(self) -> None:
        self.assertEqual(self.store.load("missing-session"), [])

    def test_load_returns_existing_conversation(self) -> None:
        conversation_path = self.storage_dir / "session-1.json"
        conversation_path.write_text(
            json.dumps(
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ]
            ),
            encoding="utf-8",
        )

        loaded = self.store.load("session-1")

        self.assertEqual(
            loaded,
            [
                ConversationTurn(role="user", content="hello"),
                ConversationTurn(role="assistant", content="hi there"),
            ],
        )

    def test_save_persists_conversation_as_json(self) -> None:
        self.store.save(
            "session-2",
            [
                ConversationTurn(role="user", content="question"),
                ConversationTurn(role="assistant", content="answer"),
            ],
        )

        conversation_path = self.storage_dir / "session-2.json"
        self.assertTrue(conversation_path.exists())
        self.assertEqual(
            json.loads(conversation_path.read_text(encoding="utf-8")),
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )
